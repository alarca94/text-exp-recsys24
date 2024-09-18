import copy
import gc
import logging
import math
from types import SimpleNamespace
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import Tensor

from ..utils.constants import *


class TaskAggregator:
    def __init__(self, agg_cfg, tasks):
        agg_type = getattr(agg_cfg, 'name', 'avg')

        if agg_type == 'avg':
            self.forward = self.avg
        elif agg_type == 'sum':
            self.forward = self.sum

        if hasattr(agg_cfg, 'weights'):
            self.task_w = {task: getattr(agg_cfg.weights, str(task)) for task in tasks}
        else:
            self.task_w = {task: 1 for task in tasks}

        self.total_w = sum(self.task_w.values())

    def __call__(self, losses):
        return self.forward(losses)

    def sum(self, losses):
        return sum(losses[task] * w for task, w in self.task_w.items())

    def avg(self, losses):
        return self.sum(losses) / self.total_w


class BaseModel(nn.Module):
    TASKS = []
    INPUT_TYPE = None
    MODEL_TYPE = None
    SEQ_MODE = SeqMode.NO_SEQ

    def __init__(self, data_info, cfg, **kwargs):
        super().__init__()

        self.cfg = copy.deepcopy(cfg)
        self.logger = kwargs.get('logger', logging.getLogger(PROJECT_NAME))
        if not self.cfg.greedy:
            self.cfg.topk = getattr(self.cfg, 'topk', 3)
        self.data_info = data_info
        self.tok = self.data_info.__dict__.pop('tok')
        self.mappers = self.data_info.__dict__.pop('mappers')

        self.curr_stage = getattr(self.cfg, 'stage', 0)
        self.loss_args = vars(getattr(self.cfg, 'loss', SimpleNamespace()))

        if self.requires_common_blocks():
            self._init_common_blocks()

        self.task_agg = self._get_taskagg()

    @property
    def n_stages(self):
        return 1

    def _get_taskagg(self):
        if hasattr(self.cfg.mtl_aggregator, 'weights'):
            tasks = [t for t in self.TASKS if str(t) in vars(self.cfg.mtl_aggregator.weights).keys()]
        else:
            tasks = self.TASKS[:]

        return TaskAggregator(self.cfg.mtl_aggregator, tasks)

    def requires_common_blocks(self):
        return True

    def _init_common_blocks(self):
        self.def_emsize = getattr(self.cfg, 'emsize', 256)
        self.hsize = getattr(self.cfg, 'hsize', self.def_emsize)

        self.user_emb = self.get_embeddings(U_COL, self.data_info.n_users)
        self.item_emb = self.get_embeddings(I_COL, self.data_info.n_items, self.data_info.special_items[PAD_TOK])

        self.loss_fns = self.get_loss_fns()

    def get_embeddings(self, desc, n_emb, padding_idx=None):
        """
        Returns an embedding matrix and the necessary projection matrix to the desired hidden size. If the model
        requires pretrained embeddings, this method extracts them from the self.data_info dataset property.
        Also, it freezes the embedding layer if not finetuning is required.
        """
        if getattr(self.cfg, f'load_{desc}_embeddings', False):
            emb = nn.Embedding.from_pretrained(getattr(self.data_info, f'{desc}_emb'), padding_idx=padding_idx)
            delattr(self.data_info, f'{desc}_emb')
            gc.collect()
        else:
            # NOTE (emsize lookup order): model config --> data_info --> default value
            emsize = getattr(self.cfg, f'{desc}_emsize', getattr(self.data_info, f'{desc}_emsize', self.def_emsize))
            emb = nn.Embedding(n_emb, emsize, padding_idx=padding_idx)

        if not getattr(self.cfg, f'{desc}_finetune_flag', True):
            emb.weight.requires_grad = False

        use_proj = (emb.weight.shape[-1] != self.hsize) and not getattr(self.cfg, 'avoid_proj', False)
        if use_proj or getattr(self.cfg, 'use_proj', False):
            bias = getattr(self.cfg, 'proj_bias', False)
            return nn.Sequential(emb, nn.Linear(emb.weight.shape[-1], self.hsize, bias=bias))

        return emb

    def get_loss_fns(self):
        """
        Returns a dict of {task: loss[Callable]} pairs
        """
        raise NotImplementedError

    def get_from_batch(self, batch, col):
        return batch[col]

    def prepare_xy(self, batch, generation=False):
        """
        Returns the inputs and labels for the current stage (training_stage, evaluation and generation)
        """
        raise NotImplementedError

    def set_stage(self, stage):
        """
        If the model requires a multi-stage training, this method is used to update variables and callables as necessary
        for the current stage
        """
        self.curr_stage = stage

    def set_batchwise_loss_args(self, inputs, labels):
        """
        Add arguments required by the loss function that will be passed along
        """
        pass

    def compute_loss(self, batch, return_preds=False):
        """
        [Batch preparation into inputs, labels] + [Model forward and Loss computation for the current stage] +
        [Loss aggregation based on the selected TaskAggregator]
        """
        inputs, labels = self.prepare_xy(batch)

        preds = self(inputs)

        self.set_batchwise_loss_args(inputs, labels)

        losses = {}
        for task, loss_fn in self.loss_fns.items():
            if task in preds:
                # NOTE: Not all tasks require labels
                losses[task] = loss_fn(preds[task], labels.get(task, None), **self.loss_args)

        if len(losses.values()) > 1:
            losses['loss'] = self.task_agg(losses)
        else:
            losses['loss'] = list(losses.values())[0]

        if return_preds:
            return preds, losses

        # Empty loss_args to free some memory
        self.loss_args = {}

        return losses

    def generate_step(self, batch):
        pass

    def check_gen_end(self, word_var, words_lens, idx):
        is_eos = word_var == self.tok.eos_token_id  # self.data_info.special_toks[EOS_TOK]
        not_end = words_lens == 0

        if idx != self.cfg.txt_len - 1:
            words_lens[not_end * is_eos] = idx + 1
            # words_lens[not_end * is_eos] = i  # exclude eos

            # break if whole batch end
            if (words_lens != 0).all():
                return True

            return False
        else:
            # reach max len
            words_lens[not_end] = self.cfg.txt_len
            return True

    def generate(self, batch):
        inputs, labels = self.prepare_xy(batch, generation=True)

        out = {t: None for t in self.TASKS}
        for idx in range(self.cfg.txt_len):
            self.generate_step(inputs)
        # if hasattr(self, 'forward_eval'):
        #     return self.forward_eval(inputs), labels
        # elif hasattr(self, 'postprocess'):
        #     preds = self(inputs)
        #     # The postprocess needs to adjust the predictions and return an updated preds dict
        #     return self.postprocess({t: preds[t].detach().clone() for t in preds}, inputs), labels

        # return self(inputs), labels


class EncDecHFBaseModel(BaseModel):
    INPUT_TYPE = InputType.TEMPLATE
    MODEL_TYPE = ModelType.GENERATIVE

    def requires_common_blocks(self):
        return False

    def compute_loss(self, batch, return_preds=False):
        inputs, labels = self.prepare_xy(batch)

        preds = {}
        for t in labels:
            preds[t] = self.forward(inputs[t])

        losses = {t: p.loss for t, p in preds.items()}

        if len(losses.values()) > 1:
            losses['loss'] = self.task_agg(losses)
        else:
            losses['loss'] = list(losses.values())[0]

        if return_preds:
            return preds, losses

        return losses


class MLPEncoder(nn.Module):
    def __init__(self, nuser, nitem, emsize, hidden_size, nlayers, pad_idx):
        super(MLPEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize, padding_idx=pad_idx)
        self.encoder = nn.Linear(emsize * 2, hidden_size * nlayers)
        self.tanh = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        initrange = 0.08
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()

    def forward(self, user, item):  # (batch_size,)
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)
        ui_concat = torch.cat([u_src, i_src], 1)  # (batch_size, emsize * 2)
        hidden = self.tanh(self.encoder(ui_concat))  # (batch_size, hidden_size * nlayers)
        state = hidden.reshape((-1, self.nlayers, self.hidden_size)).permute(1, 0, 2).contiguous()  # (num_layers, batch_size, hidden_size)
        return state


class LSTMDecoder(nn.Module):
    def __init__(self, ntoken, emsize, hidden_size, num_layers, dropout, pad_ix, batch_first=True):
        super(LSTMDecoder, self).__init__()
        self.word_embeddings = nn.Embedding(ntoken, emsize, padding_idx=pad_ix)
        self.lstm = nn.LSTM(emsize, hidden_size, num_layers, dropout=dropout, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.08
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, seq, ht, ct):  # seq: (seq_len, batch_size), ht & ct: (nlayers, batch_size, hidden_size)
        seq_emb = self.word_embeddings(seq)  # (batch_size, seq_len, emsize)
        output, (ht, ct) = self.lstm(seq_emb, (ht, ct))  # (seq_len, batch_size, hidden_size) vs. (nlayers, batch_size, hidden_size)
        decoded = self.linear(output)  # (batch_size, seq_len, ntoken)
        return func.log_softmax(decoded, dim=-1), ht, ct


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = func.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        attns = []

        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)
        attns = torch.stack(attns)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return func.relu
    elif activation == "gelu":
        return func.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # d_model: word embedding size
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,) -> (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
        '''
        probably to prevent from rounding error
        e^(idx * (-log 10000 / d_model)) -> (e^(log 10000))^(- idx / d_model) -> 10000^(- idx / d_model) -> 1/(10000^(idx / d_model))
        since idx is an even number, it is equal to that in the formula
        '''
        pe[:, 0::2] = torch.sin(position * div_term)  # even number index, (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # odd number index
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, d_model) -> (1, max_len, d_model) -> (max_len, 1, d_model)
        self.register_buffer('pe', pe)  # will not be updated by back-propagation, can be called via its name

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, hidden_sizes=(), out_units=1):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList(nn.Linear(hidden_sizes[h], hidden_sizes[h+1]) for h in range(len(hidden_sizes)-1))
        self.out = nn.Linear(hidden_sizes[-1], out_units)
        # self.linear1 = nn.Linear(emsize, emsize)
        # self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for l in self.linears:
            l.weight.data.uniform_(-initrange, initrange)
            l.bias.data.zero_()
        # self.linear1.weight.data.uniform_(-initrange, initrange)
        # self.linear2.weight.data.uniform_(-initrange, initrange)
        # self.linear1.bias.data.zero_()
        # self.linear2.bias.data.zero_()

    def forward(self, *hidden):  # (batch_size, emsize)
        mlp_vector = torch.cat(hidden, 1)
        for l in self.linears:
            mlp_vector = self.sigmoid(l(mlp_vector))  # (batch_size, seq_len, emsize)
        rating = torch.squeeze(self.out(mlp_vector))  # (batch_size, seq_len)
        return rating