""" Code adapted from https://github.com/zhanhuijing/ExBERT """

import numpy as np
import torch

from torch import nn
from torch.nn import functional as func

from ..base import PositionalEncoding
from ..generative.peter import PETER
from ...utils.constants import Task, InputType, ModelType, U_COL, I_COL, EXP_COL, SeqMode


def get_attn_pad_mask(device, ui_len, seq_q, seq_k, pad_idx):
    seq_q = seq_q.to(device)
    seq_k = seq_k.to(device)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    left = torch.zeros(batch_size, ui_len).bool().to(device)

    # eq(zero) is PAD token
    pad_attn_mask = torch.cat([left, seq_k.data.eq(pad_idx)], dim=1).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q + ui_len, len_k + ui_len)


def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1) + 2, seq.size(1) + 2]

    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask[:, 0, 1] = 0

    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ExBERT(PETER):
    TASKS = [Task.RATING, Task.EXPLANATION, Task.CONTEXT, Task.NSP]
    INPUT_TYPE = InputType.CUSTOM
    MODEL_TYPE = ModelType.HYBRID
    SEQ_MODE = SeqMode.HIST_UI_EXP

    def __init__(self, data_info, cfg, **kwargs):
        super(ExBERT, self).__init__(data_info, cfg, **kwargs)

        d_model = self.hsize
        n_heads = cfg.nhead
        n_layers = cfg.nlayers
        peter_mask = (cfg.mask_type.lower() == 'peter')
        d_k = d_model // n_heads
        d_v = d_k
        d_ff = cfg.nhid
        dropout = getattr(cfg, 'dropout', 0.5)
        pad_idx = self.tok.pad_token_id

        self.u_ix = 0
        self.i_ix = 1
        self.src_len += 1  # CLS Token

        del self.encoder
        self.encoder = Encoder(self.word_emb, d_model, n_layers, n_heads, d_k, d_v, d_ff, dropout, pad_idx)
        self.decoder = Decoder(self.word_emb, d_model, n_layers, n_heads, d_k, d_v, d_ff, dropout, pad_idx, peter_mask)

        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()

        self.classifier = nn.Linear(d_model, 2)

    def predict_nsp(self, hidden, **kwargs):
        h_pooled = self.activ(self.fc(hidden[3]))  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2]

        return logits_clsf

    def prepare_xy(self, batch, generation=False):
        inputs, labels = super(ExBERT, self).prepare_xy(batch, generation)

        # Only compute RATING, EXP AND CONTEXT losses for the positive samples
        inputs['nsp_mask'] = (batch['nsp'] == 1)
        for task in labels:
            if len(labels[task].shape) > 1:
                labels[task] = labels[task][:, inputs['nsp_mask']]
            else:
                labels[task] = labels[task][inputs['nsp_mask']]

        labels[Task.NSP] = batch['nsp']
        return inputs, labels

    def predict_tasks(self, hidden, **kwargs):
        preds = super(ExBERT, self).predict_tasks(hidden, **kwargs)
        preds[Task.NSP] = self.predict_nsp(hidden, **kwargs)
        return preds

    def unpack_batch(self, batch):
        bsz = batch[f'{U_COL}_profile'].shape[0]
        enc_inputs = torch.cat([batch[f'{U_COL}_profile'].view(bsz, -1), batch[f'{I_COL}_profile'].view(bsz, -1)], dim=1)
        dec_inputs = torch.cat([torch.full_like(batch[U_COL], self.tok.cls_token_id), batch[EXP_COL]], dim=1)
        return batch[U_COL], batch[I_COL], batch[EXP_COL], enc_inputs, dec_inputs, batch['nsp_mask']

    def get_loss_fns(self):
        loss_fns = super(ExBERT, self).get_loss_fns()
        loss_fns[Task.NSP] = nn.CrossEntropyLoss()
        return loss_fns

    def forward(self, batch, **kwargs):
        user, item, text, enc_inputs, dec_inputs, nsp_mask = self.unpack_batch(batch)

        u_src = self.user_emb(user)
        i_src = self.item_emb(item)
        enc_outputs, enc_self_attns = self.encoder(u_src, i_src, enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(u_src, i_src, dec_inputs, enc_inputs, enc_outputs)

        preds = self.predict_tasks(dec_outputs.transpose(1, 0).contiguous())
        preds[Task.EXPLANATION] = preds[Task.EXPLANATION][:, nsp_mask]
        for t in [Task.RATING, Task.CONTEXT]:
            preds[t] = preds[t][nsp_mask]
        return preds


class Encoder(nn.Module):
    def __init__(self, src_emb, d_model, n_layers, n_heads, d_k, d_v, d_ff, dropout, pad_idx):
        super(Encoder, self).__init__()
        self.src_emb = src_emb
        self.pos_emb = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])
        self.pad_idx = pad_idx

        self.ui_len = 2

    def forward(self, user_emb, item_emb, enc_inputs):
        device = enc_inputs.device
        enc_outputs = self.src_emb(enc_inputs)

        enc_outputs = torch.cat([user_emb, item_emb, enc_outputs], dim=1)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        enc_self_attn_mask = get_attn_pad_mask(device, self.ui_len, enc_inputs, enc_inputs, self.pad_idx).to(device)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn = torch.mean(enc_self_attn, dim=1)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask, self.d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)

        return self.layer_norm(output + residual), attn  # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, tgt_emb, d_model, n_layers, n_heads, d_k, d_v, d_ff, dropout, pad_idx, peter_mask):
        super(Decoder, self).__init__()
        self.tgt_emb = tgt_emb
        self.pos_emb = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])
        self.pad_idx = pad_idx
        self.peter_mask = peter_mask
        self.ui_len = 2

    def forward(self, user_emb, item_emb, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = torch.cat([user_emb, item_emb, dec_outputs], dim=1)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        device = dec_inputs.device

        dec_self_attn_pad_mask = get_attn_pad_mask(device, self.ui_len, dec_inputs, dec_inputs,
                                                   pad_idx=self.pad_idx).to(device)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).to(device)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(device, self.ui_len, dec_inputs, enc_inputs, self.pad_idx).to(device)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attn = torch.mean(dec_self_attn, dim=1)
            dec_enc_attn = torch.mean(dec_enc_attn, dim=1)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
