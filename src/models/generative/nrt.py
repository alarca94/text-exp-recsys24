""" Code adapted from https://github.com/lileipisces/NRT """

import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as func

from .base import BaseModel
from src.utils.constants import *


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class NRTEncoder(nn.Module):
    def __init__(self, nuser, nitem, emsize, hidden_size, num_layers=4, max_r=5, min_r=1, pad_idx=None):
        super(NRTEncoder, self).__init__()
        self.max_r = int(max_r)
        self.min_r = int(min_r)
        self.num_rating = self.max_r - self.min_r + 1

        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize, padding_idx=pad_idx)
        self.first_layer = nn.Linear(emsize * 2, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()

        self.encoder = nn.Linear(emsize * 2 + self.num_rating, hidden_size)
        self.tanh = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()

    def forward(self, user, item):  # (batch_size,)
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)
        ui_concat = torch.cat([u_src, i_src], 1)  # (batch_size, emsize * 2)
        hidden = self.sigmoid(self.first_layer(ui_concat))  # (batch_size, hidden_size)
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  # (batch_size, hidden_size)
        rating = torch.squeeze(self.last_layer(hidden), dim=-1)  # (batch_size,)

        rating_int = torch.clamp(rating, min=self.min_r, max=self.max_r).type(torch.int64)  # (batch_size,)
        rating_one_hot = func.one_hot(rating_int - self.min_r, num_classes=self.num_rating)  # (batch_size, num_rating)

        encoder_input = torch.cat([u_src, i_src, rating_one_hot], 1)  # (batch_size, emsize * 2 + num_rating)
        encoder_state = self.tanh(self.encoder(encoder_input)).unsqueeze(0)  # (1, batch_size, hidden_size)

        return rating, encoder_state  # (batch_size,) vs. (1, batch_size, hidden_size)


class GRUDecoder(nn.Module):
    def __init__(self, ntoken, emsize, hidden_size, pad_ix):
        super(GRUDecoder, self).__init__()
        self.word_embeddings = nn.Embedding(ntoken, emsize, padding_idx=pad_ix)
        self.gru = nn.GRU(emsize, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, seq, hidden):  # seq: (batch_size, seq_len), hidden: (nlayers, batch_size, hidden_size)
        seq_emb = self.word_embeddings(seq)  # (batch_size, seq_len, emsize)
        output, hidden = self.gru(seq_emb, hidden)  # (batch_size, seq_len, hidden_size) vs. (nlayers, batch_size, hidden_size)
        decoded = self.linear(output)  # (batch_size, seq_len, ntoken)
        return func.log_softmax(decoded, dim=-1), hidden


class NRT(BaseModel):
    TASKS = [Task.RATING, Task.EXPLANATION]
    INPUT_TYPE = InputType.REGULAR
    MODEL_TYPE = ModelType.GENERATIVE
    SEQ_MODE = SeqMode.NO_SEQ

    def __init__(self, data_info, cfg, **kwargs):
        super().__init__(data_info, cfg, **kwargs)
        i_pad_idx = data_info.special_items[PAD_TOK]
        w_pad_idx = self.tok.pad_token_id  # data_info.special_toks[PAD_TOK]
        nuser = data_info.n_users
        nitem = data_info.n_items
        n_wtokens = data_info.n_wtokens

        self.encoder = NRTEncoder(nuser, nitem, cfg.emsize, cfg.hidden_size, cfg.nlayers_rr, data_info.max_rating,
                                  data_info.min_rating, pad_idx=i_pad_idx)
        self.decoder = GRUDecoder(n_wtokens, cfg.emsize, cfg.hidden_size, w_pad_idx)

    def unpack_batch(self, batch):
        return batch[U_COL].squeeze(1), batch[I_COL].squeeze(1), batch[EXP_COL]

    def prepare_xy(self, batch, generation=False):
        inputs = batch.copy()
        if generation:
            inputs[EXP_COL] = torch.full_like(batch[U_COL], self.tok.bos_token_id)  # self.data_info.special_toks[BOS_TOK])
        else:
            inputs[EXP_COL] = inputs[EXP_COL][..., :-1].contiguous()

        # inputs[f'{EXP_COL}_offset'] = 0

        labels = {
            Task.RATING: batch[RAT_COL].squeeze(1),
            Task.EXPLANATION: batch[EXP_COL][..., 1:].contiguous()
        }

        # NOTE: NRT never uses the feature
        # if self.cfg.use_feature:
        #     batch[EXP_COL] = torch.cat([batch[FEAT_COL], batch[EXP_COL]], 0)  # (src_len + tgt_len - 2, batch_size)
        #     inputs[f'{EXP_COL}_offset'] += batch[FEAT_COL].shape[-1]

        return inputs, labels

    def exp_loss(self, preds, labels, **kwargs):
        return func.nll_loss(preds.view(-1, self.data_info.n_wtokens), labels.view(-1),
                             ignore_index=self.tok.pad_token_id)  # self.data_info.special_toks[PAD_TOK])

    def get_loss_fns(self):
        loss_fns = {
            Task.EXPLANATION: self.exp_loss,
            Task.RATING: nn.MSELoss()
        }
        return loss_fns

    def forward(self, batch):  # (batch_size,) vs. (batch_size, seq_len)
        user, item, seq = self.unpack_batch(batch)
        rating, hidden = self.encoder(user, item)
        log_word_prob, _ = self.decoder(seq, hidden)
        return {Task.RATING: rating, Task.EXPLANATION: log_word_prob}

    def generate(self, batch, **kwargs):
        inputs, labels = self.prepare_xy(batch, generation=True)
        user, item, seq = self.unpack_batch(inputs)

        rating_p = []
        words, probs = [], []
        words_lens = torch.zeros(user.shape[0], dtype=torch.long, device=user.device)
        for idx in range(self.cfg.txt_len):
            if idx == 0:
                rating, hidden = self.encoder(user, item)
                log_word_prob, hidden = self.decoder(seq, hidden)
                rating_p.extend(rating.tolist())
            else:
                log_word_prob, hidden = self.decoder(seq, hidden)

            word_probs = log_word_prob.squeeze(1).exp()  # (batch_size, ntoken)
            _, word_var = word_probs.max(-1)

            words.append(word_var)
            # probs.append(prob_var)

            if self.check_gen_end(word_var, words_lens, idx):
                break

            seq = word_var.unsqueeze(1)

        words = torch.stack(words, dim=1)
        # probs = torch.stack(probs, dim=1)

        pred = {Task.RATING: rating_p, Task.EXPLANATION: words.cpu().numpy().tolist()}
        return pred, labels
