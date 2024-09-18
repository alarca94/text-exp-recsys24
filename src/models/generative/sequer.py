""" Code adapted from https://github.com/alarca94/sequer-recsys23 """

import torch
import torch.nn.functional as F

from src.utils.constants import *
from .base import generate_sequer_mask
from .peter import PETER
from ...utils.funcs import predict


class SEQUER(PETER):
    TASKS = [Task.RATING, Task.EXPLANATION, Task.CONTEXT, Task.NEXT_ITEM]
    INPUT_TYPE = InputType.CUSTOM
    MODEL_TYPE = ModelType.GENERATIVE
    SEQ_MODE = SeqMode.HIST_ITEM

    def __init__(self, data_info, cfg, **kwargs):
        super().__init__(data_info, cfg, **kwargs)

        nextitem_loss = getattr(cfg, 'nextitem_loss', 'CE')
        last_only = nextitem_loss.lower() in ['bpr', 'bce']

        self.attn_mask = generate_sequer_mask(self.src_len, self.tgt_len, self.mask_type,
                                              lengths=self.data_info.lengths)

        self.nextit_criterion = NextItemLoss(loss_type=nextitem_loss, ignore_index=self.mappers[I_COL].get_idx(PAD_TOK),
                                             last_only=last_only)
        # self.nexitem_criterion = nn.NLLLoss(ignore_index=self.mappers[I_COL].get_idx(UNK_TOK))
        self.init_weights()

    def get_masks(self, user, item, text, total_len):
        attn_mask = self.attn_mask[:total_len, :total_len].to(user.device)
        middle = torch.zeros_like(user, dtype=torch.bool)
        left = (item == self.mappers[I_COL].get_idx(PAD_TOK))  # torch.zeros_like(item, dtype=torch.bool)
        right = text == self.tok.pad_token_id  # self.data_info.special_toks[PAD_TOK]
        key_padding_mask = torch.cat([left, middle, right], 0).t()
        return attn_mask, key_padding_mask

    def prepare_xy(self, batch, generation=False):
        if generation:
            batch.pop(NEG_OPT_COL, None)
        else:
            batch.pop(NEG_EVAL_COL, None)

        inputs, labels = super().prepare_xy(batch, generation)
        labels[Task.RATING] = labels[Task.RATING].T.contiguous()
        labels[Task.NEXT_ITEM] = batch[I_COL][:, 1:].squeeze(0).T.contiguous()
        if generation:
            labels[Task.RATING] = torch.gather(labels[Task.RATING], 0, batch[SEQ_LEN_COL] - 1).detach().clone()
            labels[Task.NEXT_ITEM] = torch.gather(labels[Task.NEXT_ITEM], 0, batch[SEQ_LEN_COL] - 2).squeeze(0).detach().clone()
        else:
            labels[Task.NEXT_ITEM] = (labels[Task.NEXT_ITEM], batch.get(NEG_OPT_COL, torch.tensor([]))[:, 1:].squeeze(0).T.contiguous())

        return inputs, labels

    def predict_context(self, hidden, **kwargs):
        context_prob = self.hidden2token(hidden[self.u_ix])  # (batch_size, n_token)
        log_context_dis = F.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, hidden, **kwargs):
        if self.SEQ_MODE not in [SeqMode.NO_SEQ, SeqMode.HIST_ITEM_RATING_LAST]:
            rating = self.recommender(hidden[:self.u_ix])
        else:
            # rating = self.recommender(hidden[self.lens[I_COL] - 1:self.lens[I_COL]])
            batch_ixs = torch.arange(hidden.shape[1], device=hidden.device)
            rating = self.recommender(hidden[kwargs[SEQ_LEN_COL] - 1, batch_ixs])
        return rating

    def predict_next(self, hidden, **kwargs):
        return hidden[:self.u_ix - 1]

    def unpack_batch(self, batch):
        u, i, e = super().unpack_batch(batch)
        return u, i, e, batch[SEQ_LEN_COL].squeeze(1)

    def predict_tasks(self, hidden, **kwargs):
        preds = super(SEQUER, self).predict_tasks(hidden, **kwargs)
        preds[Task.NEXT_ITEM] = self.predict_next(hidden, **kwargs)
        return preds

    def nextitem_loss(self, preds, labels, **kwargs):
        pos_labels = labels[0]
        neg_labels = labels[1]
        return self.nextit_criterion(self.item_emb, preds, pos_labels, neg_labels, kwargs[SEQ_LEN_COL])

    def rating_loss(self, preds, labels, **kwargs):
        return self.rating_criterion(preds, labels).mean()

    def get_loss_fns(self):
        loss_fns = super().get_loss_fns()
        loss_fns[Task.NEXT_ITEM] = self.nextitem_loss
        return loss_fns

    def set_batchwise_loss_args(self, inputs, labels):
        self.loss_args[SEQ_LEN_COL] = inputs[SEQ_LEN_COL].squeeze(1)

    def forward(self, batch, **kwargs):
        """
        :param user: (batch_size,), torch.int64
        :param item: (batch_size, seq_len), torch.int64
        :param text: (total_len - ui_len - hist_len, batch_size), torch.int64
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken) if gen_mode=False; the last token (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        :return attns: (nlayers, batch_size, total_len, total_len)
        :return log_next_item: (batch_size, hist_len, nitem) if next_item_prediction=True; None otherwise
        :return
        """
        user, item, text, seq_len = self.unpack_batch(batch)

        src = self.get_embedding_input(user, item, text)
        src = self.pos_encoder(src)
        attn_mask, key_padding_mask = self.get_masks(user, item, text, total_len=src.size(0))

        hidden, attns = self.encoder(src, attn_mask, key_padding_mask)
        res = self.predict_tasks(hidden, **{SEQ_LEN_COL: seq_len})
        res['attns'] = attns
        return res

    def process_first_genstep(self, pred, batch):
        batch_ixs = torch.arange(batch[U_COL].shape[0])
        seq_len = batch[SEQ_LEN_COL].squeeze(1)
        if len(pred[Task.RATING].shape) > 1:
            pred[Task.RATING] = pred[Task.RATING][seq_len - 1, batch_ixs]
        pred[Task.NEXT_ITEM] = F.log_softmax(torch.matmul(pred[Task.NEXT_ITEM][seq_len - 2, batch_ixs],
                                                          self.item_emb.weight.T), dim=-1).squeeze(0)
        # [BSZ, HSZ] * [HSZ, NI] = BSZ, NI
        pred[Task.NEXT_ITEM_SAMPLE] = torch.gather(pred[Task.NEXT_ITEM], 1, batch[NEG_EVAL_COL]).tolist()
        pred[Task.NEXT_ITEM] = predict(pred[Task.NEXT_ITEM], topk=max(TOP_KS)).tolist()
        return super().process_first_genstep(pred, batch)


class NextItemLoss:
    def __init__(self, loss_type='CE', ignore_index=-100, last_only=False):
        self.ignore_index = ignore_index
        self.last_only = last_only
        self.loss_type = loss_type
        if loss_type == 'CE':
            self.loss_fn = self.ce_loss
        elif loss_type == 'BCE':
            self.loss_fn = self.bce_loss
        elif loss_type == 'BPR':
            self.loss_fn = self.bpr_loss

    def ce_loss(self, emb_layer, seq_output, pos_target, neg_target):
        logits = F.log_softmax(torch.matmul(seq_output, emb_layer.weight.T), dim=-1)
        logits = logits.view(-1, emb_layer.weight.shape[0])
        return F.nll_loss(logits, pos_target, reduction='mean', ignore_index=self.ignore_index)

    def bce_loss(self, emb_layer, seq_output, pos_target, neg_target):
        eps = 1e-24
        pos_emb = emb_layer(pos_target)
        neg_emb = emb_layer(neg_target)

        pos_logits = torch.sum(pos_emb.view(-1, seq_output.shape[-1]) * seq_output, -1)
        neg_logits = torch.sum(neg_emb.view(-1, seq_output.shape[-1]) * seq_output, -1)
        loss = torch.mean(- torch.log(torch.sigmoid(pos_logits) + eps) - torch.log(1 - torch.sigmoid(neg_logits) + eps))
        return loss

    def bpr_loss(self, emb_layer, seq_output, pos_target, neg_target):
        gamma = 1e-10
        pos_items_emb = emb_layer(pos_target)
        neg_items_emb = emb_layer(neg_target)
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
        loss = -torch.log(gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

    def __call__(self, embeddings, input, pos_target, neg_target=None, lengths=None):
        if self.last_only:
            batch_range = torch.arange(input.shape[1])
            # Filter cold-start cases NOTE: not necessary if we add UNK/MASK token for cold-start cases
            # mask = lengths > 1
            # batch_range = batch_range[mask]
            # lengths = lengths[mask]
            # return self.loss_fn(embeddings, input[-1], pos_target[-1], neg_target[-1])
            # As lengths include the whole item_seq length (with cand_item)
            return self.loss_fn(embeddings, input[lengths-2, batch_range], pos_target[lengths-2, batch_range],
                                neg_target[lengths-2, batch_range])
        return self.loss_fn(embeddings, input, pos_target.reshape((-1,)), neg_target.reshape((-1,)))
