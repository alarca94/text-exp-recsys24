import os
import copy
import datetime
from types import SimpleNamespace

import torch
import random
import numpy as np

from collections.abc import Mapping

from typing import Optional, Union, Iterable

from .constants import Task, RNG_SEED, DATE_FORMAT, CFG_ENTRY_TYPES


def init_seed(seed=RNG_SEED, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def arglatest(batch, tok_id, batch_first):
    if not batch_first:
        return arglatest(batch.t(), tok_id, batch_first=True)

    start_ixs = torch.zeros((batch.shape[0], 1), dtype=torch.long)
    ixs = torch.nonzero(batch == tok_id)  # self.corpus.toks[EXP_COL].sep_token_id)
    # if len(ixs):
    last_sep_ixs = torch.nonzero(torch.cat((ixs[1:, 0], (ixs[-1, 0] + 1).unsqueeze(0))) - ixs[:, 0])
    start_ixs[ixs[last_sep_ixs, 0].squeeze()] = ixs[last_sep_ixs, 1]
    return start_ixs


def get_span_ixs(start_ixs, span_size):
    return start_ixs + torch.arange(span_size).repeat(start_ixs.shape[0], 1)


def gather_span(data, start_ixs, span_size, dim=1):
    select_ixs = get_span_ixs(start_ixs, span_size)
    return torch.gather(data, index=select_ixs, dim=dim)


def predict(log_context_dis, topk):
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = word_prob.topk(topk, dim=1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)


def now_time():
    return '[' + datetime.datetime.now().strftime(DATE_FORMAT) + ']: '


def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """
    # max_memory_reserved() returns an approximation but still differs from the nvidia-smi output
    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)


def results2ray(results):
    new_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            v = results2ray(v)
        new_results[str(k)] = v
    return new_results


def update_dict(d, new_d):
    for k, v in new_d.items():
        if isinstance(v, Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def dict2namespace(d):
    out = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, Mapping):
            v = dict2namespace(v)
        elif isinstance(v, list) and isinstance(v[0], Mapping):
            v = [dict2namespace(vi) for vi in v]
        setattr(out, k, v)
    return out


def namespace2dict(ns):
    d = vars(copy.deepcopy(ns))
    for k, v in d.items():
        if isinstance(v, SimpleNamespace):
            d[k] = namespace2dict(v)
        elif isinstance(v, list) and isinstance(v[0], SimpleNamespace):
            d[k] = [namespace2dict(vi) for vi in v]
        elif isinstance(v, torch.device):
            d[k] = f'{v.type}'
    return d


def flatten_dict(d, key_sep='-'):
    ks = list(d.keys())
    rm_ks = []
    for k in ks:
        if isinstance(d[k], dict):
            d[k] = flatten_dict(d[k], key_sep)
            for ki, vi in d[k].items():
                d[f'{k}{key_sep}{ki}'] = vi
            rm_ks.append(k)
        elif isinstance(d[k], (list, tuple)) and d[k] and isinstance(d[k][-1], dict):
            for i, vi in enumerate(d[k]):
                d[k][i] = flatten_dict(vi, key_sep)
                for kj, vij in d[k][i].items():
                    d[f'{k}{key_sep}{i}{key_sep}{kj}'] = vij
            rm_ks.append(k)

    for k in rm_ks:
        d.pop(k, None)

    return d


def dict2txt(d, entry_sep='|'):
    return entry_sep.join([v if entry_sep not in v else f'"{v}"' for v in [f'{k}:{v}' for k, v in d.items()]])


def multigpu_embed(embedder, texts, batch_size, chunk_size):
    pool = embedder.start_multi_process_pool()
    embs = embedder.encode_multi_process(texts, pool, batch_size=batch_size, chunk_size=chunk_size)
    embedder.stop_multi_process_pool(pool)
    return embs


def chunk_embed(embedder, texts, batch_size, chunk_size, show_progress_bar=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_chunks = int(np.ceil(len(texts) / chunk_size))
    embs = []
    for i in range(n_chunks):
        embs.append(embedder.encode(texts[i * chunk_size: (i + 1) * chunk_size], batch_size,
                                    show_progress_bar=show_progress_bar, device=device))
    return np.concatenate(embs, axis=0)


def is_sentence_transformer_model(
    model_name_or_path: str,
    token: Optional[Union[bool, str]] = None,
    cache_folder: Optional[str] = None,
    revision: Optional[str] = None,
) -> bool:
    return bool(load_file_path(model_name_or_path, "modules.json", token, cache_folder, revision=revision))


def load_file_path(
    model_name_or_path: str,
    filename: str,
    token: Optional[Union[bool, str]],
    cache_folder: Optional[str],
    revision: Optional[str] = None,
) -> Optional[str]:
    # If file is local
    file_path = os.path.join(model_name_or_path, filename)
    if os.path.exists(file_path):
        return file_path

    # If file is remote
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            model_name_or_path,
            filename=filename,
            revision=revision,
            library_name="sentence-transformers",
            token=token,
            cache_dir=cache_folder,
        )
    except Exception:
        return


def nested_cfg_update(d, change_d, inplace=False, is_leaf=False):
    if not inplace:
        new_d = copy.deepcopy(d)
    else:
        new_d = d
    for k, v in vars(change_d).items():
        if is_leaf or not isinstance(v, SimpleNamespace):
            setattr(new_d, k, v)
        elif k in CFG_ENTRY_TYPES:
            setattr(new_d, k, nested_cfg_update(getattr(new_d, k), v, is_leaf=True))
        else:
            setattr(new_d, k, nested_cfg_update(getattr(new_d, k), v))

    if not inplace:
        return new_d
