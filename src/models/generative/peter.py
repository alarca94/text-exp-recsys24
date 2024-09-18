""" Code adapted from https://github.com/lileipisces/PETER """

from collections import defaultdict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import *
from src.utils.funcs import predict
from ..base import MLP, PositionalEncoding, TransformerEncoderLayer, TransformerEncoder, BaseModel
from .base import generate_peter_mask, generate_square_subsequent_mask


class PETER(BaseModel):
    TASKS = [Task.RATING, Task.EXPLANATION, Task.CONTEXT]
    INPUT_TYPE = InputType.REGULAR
    MODEL_TYPE = ModelType.GENERATIVE
    SEQ_MODE = SeqMode.NO_SEQ

    def __init__(self, data_info, cfg, **kwargs):
        super().__init__(data_info, cfg, **kwargs)

        dropout = getattr(cfg, 'dropout', 0.5)
        nhead = cfg.nhead
        nhid = cfg.nhid
        nlayers = cfg.nlayers
        self.mask_type = getattr(cfg, 'mask_type', 'peter')

        self.word_emb = self.get_embeddings(EXP_COL, data_info.n_wtokens, self.tok.pad_token_id)  # data_info.special_toks[PAD_TOK])

        self.pos_encoder = PositionalEncoding(self.hsize, dropout)  # emsize: word embedding size
        encoder_layers = TransformerEncoderLayer(self.hsize, nhead, nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above

        self.hidden2token = nn.Linear(self.hsize, self.data_info.n_wtokens)
        self.recommender = MLP(hidden_sizes=(self.hsize, self.def_emsize))

        u_len, i_len, f_len, e_len = self.data_info.lengths.values()
        self.ui_len = u_len + i_len
        self.src_len = self.ui_len
        if getattr(cfg, 'use_feature', False):
            self.src_len += f_len
        self.tgt_len = e_len - 1
        self.context_len = e_len
        self.i_ix = 0
        self.u_ix = i_len  # i_len is always 1 as the user/item encoding is 1:1 relation

        if self.mask_type.lower() == 'peter':
            self.attn_mask = generate_peter_mask(self.src_len, self.tgt_len)
        else:
            self.attn_mask = generate_square_subsequent_mask(self.src_len + self.tgt_len)

        self.init_weights()

        self.text_criterion = nn.NLLLoss(ignore_index=self.tok.pad_token_id)  # self.data_info.special_toks[PAD_TOK])
        # NOTE: UNK_TOK is not ignored (Also, MASK_TOK or SEP_TOK are not used to encode sentences). If all special
        #  tokens need to be ignored, create special_tok_mask with the following two lines:
        # special_tok_mask = self.tok.get_special_tokens_mask(range(len(self.tok)), already_has_special_tokens=True)
        # special_tok_mask = 1.0 - torch.FloatTensor(special_tok_mask)
        special_tok_mask = torch.ones(len(self.tok), dtype=torch.float)
        ignore_idxs = [self.tok.bos_token_id, self.tok.pad_token_id, self.tok.eos_token_id]
        assert all([ix is not None for ix in ignore_idxs])
        special_tok_mask[ignore_idxs] = 0
        self.context_criterion = nn.NLLLoss(weight=special_tok_mask)
        self.rating_criterion = nn.MSELoss(reduction='none')

    def init_weights(self):
        initrange = 0.1
        self.user_emb.weight.data.uniform_(-initrange, initrange)
        self.item_emb.weight.data.uniform_(-initrange, initrange)
        self.word_emb.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

    def get_val_loss(self, results):
        return results[Task.EXPLANATION] + results[Task.RATING] * int(self.task_agg.task_w[Task.RATING] != 0)

    def prepare_xy(self, batch, generation=False):
        inputs = batch.copy()
        if generation:
            inputs[EXP_COL] = torch.full_like(batch[U_COL], self.tok.bos_token_id)  # self.data_info.special_toks[BOS_TOK])
        else:
            inputs[EXP_COL] = inputs[EXP_COL][..., :-1].contiguous()

        labels = {
            Task.RATING: batch[RAT_COL].squeeze(1),
            Task.EXPLANATION: batch[EXP_COL][..., 1:].contiguous(),
            Task.CONTEXT: batch[CONTEXT_COL].T.contiguous()
        }

        if getattr(self.cfg, 'use_feature', False):
            inputs[EXP_COL] = torch.cat([inputs[FEAT_COL], inputs[EXP_COL]], 1)

        labels[Task.EXPLANATION] = labels[Task.EXPLANATION].T.contiguous()
        return inputs, labels

    def unpack_batch(self, batch):
        return batch[U_COL].T, batch[I_COL].T, batch[EXP_COL].T

    def get_embedding_input(self, user, item, text):
        u_src = self.user_emb(user)  # (1, batch_size, emsize)
        i_src = self.item_emb(item)  # (seq_len, batch_size, emsize)
        w_src = self.word_emb(text)  # (total_len - ui_len - hist_len, batch_size, emsize)
        src = torch.cat([i_src, u_src, w_src], 0)  # (total_len, batch_size, emsize)
        src = src * math.sqrt(self.hsize)
        return src

    def get_masks(self, user, item, text, total_len):
        # total_len = self.ui_len + text.size(0)
        attn_mask = self.attn_mask[:total_len, :total_len].to(user.device)
        middle = torch.zeros_like(user, dtype=torch.bool)
        left = torch.zeros_like(item, dtype=torch.bool)
        right = text == self.tok.pad_token_id  # self.data_info.special_toks[PAD_TOK]
        key_padding_mask = torch.cat([left, middle, right], 0).t()
        return attn_mask, key_padding_mask

    def predict_context(self, hidden, **kwargs):
        context_prob = self.hidden2token(hidden[self.i_ix])  # (batch_size, ntoken)
        log_context_dis = F.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, hidden, **kwargs):
        rating = self.recommender(hidden[self.u_ix])  # (batch_size,)
        return rating

    def predict_exp(self, hidden, **kwargs):
        word_prob = self.hidden2token(hidden[self.src_len:])  # (tgt_len, batch_size, ntoken)
        log_word_prob = F.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def predict_tasks(self, hidden, **kwargs):
        return {
            Task.EXPLANATION: self.predict_exp(hidden, **kwargs),
            Task.RATING: self.predict_rating(hidden, **kwargs),
            Task.CONTEXT: self.predict_context(hidden, **kwargs)
        }

    def rating_loss(self, preds, labels, **kwargs):
        return self.rating_criterion(preds, labels).mean()

    def exp_loss(self, preds, labels, **kwargs):
        return self.text_criterion(preds.view(-1, self.data_info.n_wtokens), labels.view(-1))

    def context_loss(self, preds, labels, **kwargs):
        seq = labels.detach().clone()
        context_dis = preds.unsqueeze(0).repeat((seq.shape[0], 1, 1))

        return self.context_criterion(context_dis.view(-1, self.data_info.n_wtokens), labels.view(-1))

    def get_loss_fns(self):
        loss_fns = {
            Task.EXPLANATION: self.exp_loss,
            Task.CONTEXT: self.context_loss,
            Task.RATING: self.rating_loss
        }
        return loss_fns

    def forward(self, batch, **kwargs):
        """
        :param user: (batch_size, 1), torch.int64
        :param item: (batch_size, 1), torch.int64
        :param text: (total_len - ui_len, batch_size), torch.int64
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken)
        :return log_context_dis: (batch_size, ntoken)
        :return rating: (batch_size,)
        :return attns: (nlayers, batch_size, total_len, total_len)
        """
        user, item, text = self.unpack_batch(batch)

        src = self.get_embedding_input(user, item, text)
        src = self.pos_encoder(src)

        attn_mask, key_padding_mask = self.get_masks(user, item, text, total_len=src.size(0))

        # hidden: (L, BSZ, HSZ) vs. attns: (N_LAYERS, BSZ, L, L)
        hidden, attns = self.encoder(src, attn_mask, key_padding_mask)
        res = self.predict_tasks(hidden)
        res['attns'] = attns
        return res

    def process_first_genstep(self, pred, batch):
        pred[Task.RATING] = pred[Task.RATING].tolist()
        pred[Task.CONTEXT] = predict(pred[Task.CONTEXT], topk=self.context_len).tolist()
        return pred

    def generate(self, batch, **kwargs):
        inputs, labels = self.prepare_xy(batch, generation=True)

        preds = defaultdict(list)
        words, probs = [], []
        words_lens = torch.zeros(batch['size'], dtype=torch.long, device=inputs[U_COL].device)
        for idx in range(self.tgt_len):
            pred = self(inputs)
            if idx == 0:
                for t, v in self.process_first_genstep(pred, batch).items():
                    preds[t].extend(v)

            word_probs = pred[Task.EXPLANATION][-1].exp()
            _, word_var = word_probs.max(-1)

            words.append(word_var)
            # probs.append(prob_var)

            if self.check_gen_end(word_var, words_lens, idx):
                break

            inputs[EXP_COL] = torch.cat((inputs[EXP_COL], word_var.unsqueeze(1)), dim=1)

        words = torch.stack(words, dim=1)
        # probs = torch.stack(probs, dim=1)

        preds[Task.EXPLANATION] = words.cpu().numpy().tolist()
        return preds, labels

