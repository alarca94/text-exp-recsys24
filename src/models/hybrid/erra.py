""" Code adapted from https://github.com/Complex-data/ERRA """

import math
import copy
import torch

from torch import nn

from src.utils.constants import Task, InputType, EXP_COL, U_COL, I_COL, FEAT_COL, ASP_COL, ModelType
from ..generative.base import generate_square_subsequent_mask
from ..generative.peter import PETER


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def generate_erra_mask(src_len, tgt_len):
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    mask[:src_len, :src_len] = False
    return mask


class ERRA(PETER):
    TASKS = [Task.RATING, Task.EXPLANATION, Task.CONTEXT, Task.ASPECT]
    INPUT_TYPE = InputType.CUSTOM
    MODEL_TYPE = ModelType.HYBRID

    def __init__(self, data_info, cfg, **kwargs):
        super().__init__(data_info, cfg, **kwargs)

        # NOTE: ERRA implementation requires that the Sentence Transformer has the same emsize as the user/item/word embeddings

        self.recommender = MLP1(self.def_emsize)

        self.sigmoid = nn.Sigmoid()

        if self.mask_type == 'peter':
            # Account for the u_sv and i_sv
            self.attn_mask = generate_erra_mask(self.src_len, self.tgt_len + 2)

        self.uif_len = self.src_len
        self.src_len += 2

        self.init_weights()

    def get_embedding_input(self, user, item, text, u_sv, i_sv):
        u_src = self.user_emb(user)  # (1, batch_size, emsize)
        i_src = self.item_emb(item)  # (seq_len, batch_size, emsize)
        w_src = self.word_emb(text)  # (total_len - ui_len - hist_len, batch_size, emsize)

        split_ix = self.uif_len - self.ui_len
        src = torch.cat([i_src, u_src, w_src[:split_ix], u_sv.unsqueeze(0), i_sv.unsqueeze(0), w_src[split_ix:]], 0)
        src = src * math.sqrt(self.hsize)

        return src

    def get_masks(self, user, item, text, total_len):
        attn_mask = self.attn_mask[:total_len, :total_len].to(user.device)
        left = torch.zeros(self.src_len, user.size(-1), device=user.device, dtype=torch.bool)
        right = text[2:] == self.tok.pad_token_id
        key_padding_mask = torch.cat([left, right], 0).t()
        return attn_mask, key_padding_mask

    def prepare_xy(self, batch, generation=False):
        batch[FEAT_COL] = batch.pop(ASP_COL)
        return super().prepare_xy(batch, generation)

    def predict_rating(self, hidden, **kwargs):
        rat_in = torch.cat((hidden[self.i_ix], kwargs['u_src'], kwargs['i_src']), dim=-1)
        rating = self.recommender(rat_in)  # (batch_size,)
        return rating

    def aspect_loss(self, preds, labels, **kwargs):
        ui_aspects = kwargs[f'{Task.ASPECT.value}_values'].unsqueeze(0).repeat(preds.shape[0], 1, 1)
        txt_mask = kwargs[f'{Task.ASPECT.value}_mask'].to(torch.float)
        preds = preds.gather(-1, ui_aspects)
        return - ((preds.sum(2) / preds.size(2) * txt_mask).sum(0) / txt_mask.sum(0)).mean()

    def get_loss_fns(self):
        loss_fns = super().get_loss_fns()
        if self.cfg.use_feature:
            loss_fns[Task.ASPECT] = self.aspect_loss
        return loss_fns

    def set_batchwise_loss_args(self, inputs, labels):
        if self.cfg.use_feature:
            self.loss_args[f'{Task.ASPECT.value}_values'] = inputs[FEAT_COL]
            self.loss_args[f'{Task.ASPECT.value}_mask'] = labels[Task.EXPLANATION] != self.tok.pad_token_id

    def unpack_batch(self, batch):
        return batch[U_COL].T, batch[I_COL].T, batch[EXP_COL].T, batch[f'{U_COL}_sv'], batch[f'{I_COL}_sv']

    def predict_tasks(self, hidden, **kwargs):
        preds = super().predict_tasks(hidden, **kwargs)
        if self.cfg.use_feature:
            preds[Task.ASPECT] = preds[Task.EXPLANATION]
        return preds

    def forward(self, batch, **kwargs):
        u, i, t, u_sv, i_sv = self.unpack_batch(batch)

        src = self.get_embedding_input(u, i, t, u_sv, i_sv)
        src = self.pos_encoder(src)

        attn_mask, key_padding_mask = self.get_masks(u, i, t, total_len=src.size(0))

        hidden, attns = self.encoder(src, attn_mask, key_padding_mask)
        # NOTE: ERRA authors assume that user/item_ids are not encoded into multiple tokens e.g. P5 Encoding
        res = self.predict_tasks(hidden, u_src=src[1], i_src=src[0])
        res['attns'] = attns
        return res


class MLP1(nn.Module):
    def __init__(self, emsize=384):
        super(MLP1, self).__init__()
        self.linear3 = nn.Linear(emsize*3, emsize)
        self.linear4 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear3.weight.data.uniform_(-initrange, initrange)
        self.linear4.weight.data.uniform_(-initrange, initrange)
        self.linear3.bias.data.zero_()
        self.linear4.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize)
        mlp_vector = self.sigmoid(self.linear3(hidden))  # (batch_size, emsize)
        rating = torch.squeeze(self.linear4(mlp_vector))  # (batch_size,)
        return rating