import torch

from ..base import BaseModel
from src.utils.constants import InputType, I_COL, U_COL, Task, ModelType, SeqMode
from src.utils.in_out import plot_mask


def generate_square_subsequent_mask(total_len):
    mask = torch.tril(torch.ones(total_len, total_len))  # (total_len, total_len), lower triangle -> 1.; others 0.
    mask = mask == 0  # lower -> False; others True
    return mask


def generate_peter_mask(src_len, tgt_len):
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    mask[0, 1] = False  # allow to attend for user and item
    return mask


def peter_like_mask(mask, lengths):
    hist_len = lengths[I_COL] - 1
    mask[hist_len, hist_len + 1] = False  # allow last item to attend to the user
    return mask


def bert_like_mask(mask, lengths):
    ui_len = lengths[U_COL] + lengths[I_COL]
    mask[:ui_len, :ui_len] = False
    return mask


def sawer_like_mask(mask, lengths):
    mask[:lengths[I_COL], lengths[I_COL]] = False
    return mask


def generate_sequer_mask(src_len, tgt_len, mask_type='default', plot=False, lengths=None):
    total_len = src_len + tgt_len
    if 'mlm' in mask_type:
        # Candidate item ID will be masked and added after the User Token
        total_len += 1

    mask = generate_square_subsequent_mask(total_len)

    if mask_type == 'peter':
        mask = peter_like_mask(mask, lengths)
    elif mask_type.startswith('bert'):
        mask = bert_like_mask(mask, lengths)
    elif mask_type.startswith('sawer'):
        mask = sawer_like_mask(mask, lengths)

    if plot:
        plot_mask(mask)
    return mask


class GenBaseModel(BaseModel):
    TASKS = [Task.RATING, Task.EXPLANATION]
    INPUT_TYPE = InputType.REGULAR
    MODEL_TYPE = ModelType.GENERATIVE
    SEQ_MODE = SeqMode.NO_SEQ

    def __init__(self, data_info, cfg, **kwargs):
        super().__init__(data_info, cfg, **kwargs)

    def generate(self, batch, **kwargs):
        pass