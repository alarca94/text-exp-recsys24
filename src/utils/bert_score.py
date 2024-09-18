import time
from collections import defaultdict

import logging
import numpy as np
import pandas as pd
import torch
import gc

from bert_score.utils import (bert_cos_score_idf, get_hash, get_idf_dict, get_model, get_tokenizer,
                              lang2model, model2layers)
from tqdm import tqdm


def chunk_score(
        df,
        cand_c,
        ref_c,
        mapper,
        chunk_size=200000,
        model_type=None,
        num_layers=None,
        idf=False,
        device=None,
        batch_size=64,
        nthreads=4,
        all_layers=False,
        use_fast_tokenizer=False):

    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = get_tokenizer(model_type, use_fast_tokenizer)
    model = get_model(model_type, num_layers, all_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    else:
        idf_dict = get_idf_dict(mapper.get_vocab(), tokenizer, nthreads=nthreads)

    group_ixs = (df[cand_c].str.len() * df[ref_c].str.len()).cumsum() // chunk_size
    group_ixs = (np.argwhere(group_ixs.values[1:] - group_ixs.values[:-1])[:, 0] + 1).tolist()
    logging.info(f'{len(group_ixs) + 1} groups created')
    group_ixs = [0] + group_ixs + [df.shape[0]]
    scores = []
    for i in tqdm(range(len(group_ixs) - 1)):
        chunk = df.iloc[group_ixs[0]:group_ixs[1]][[cand_c, ref_c]]
        scores.append(compute_bert_score(chunk, mapper, cand_c, ref_c, model, tokenizer, idf_dict, device, batch_size))

    return pd.concat(scores)


def compute_bert_score(chunk, mapper, cand_c, ref_c, model, tokenizer, idf_dict, device, batch_size):
    # index_order = data.index.values
    chunk = chunk.explode(cand_c).explode(ref_c)
    chunk = chunk.apply(lambda c: mapper.get_items(c), axis=0)
    chunk['y'] = bert_score(refs=chunk[ref_c].tolist(), cands=chunk[cand_c].tolist(), device=device,
                            model=model, tokenizer=tokenizer, idf_dict=idf_dict, use_fast_tokenizer=True,
                            batch_size=batch_size)[-1].numpy()
    chunk.reset_index(inplace=True)
    chunk.drop(ref_c, axis=1, inplace=True)
    chunk = chunk.groupby(['index', cand_c], sort=False).agg('max')
    chunk.reset_index(level=0, inplace=True)
    torch.cuda.empty_cache()
    gc.collect()
    return chunk.groupby('index', sort=False).agg(list)


def bert_score(
    cands,
    refs,
    model,
    tokenizer,
    idf_dict,
    device='cpu',
    batch_size=64,
    all_layers=False,
    lang=None,
    use_fast_tokenizer=False,
):
    assert len(cands) == len(refs), "Different number of candidates and references"

    all_preds = bert_cos_score_idf(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=False,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
    ).cpu()

    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

    return out
