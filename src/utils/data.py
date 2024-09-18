import copy
import importlib
import logging
import os.path
import re
import gc
# import shutil
import sys
# import time
# import zipfile

import random
from collections import defaultdict

from functools import partial
from typing import Iterable, Callable

from types import SimpleNamespace

import nltk
import pandas as pd
import numpy as np
import spacy
import torch
import torch.nn.functional as F

from torch_geometric.data import HeteroData, Data, Batch
from torch_geometric.loader import DataLoader
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader, Subset
from tokenizers import Tokenizer, trainers
from tokenizers.models import WordLevel, BPE
from tokenizers.pre_tokenizers import WhitespaceSplit, ByteLevel
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate import bleu_score

from .constants import *
from .in_out import load_partition
from .templates import get_task_templates


def get_context(docs, pos_tags=('NOUN', 'ADJ')):
    nlp = spacy.load("en_core_web_sm")
    contexts = []
    for s in nlp.pipe(docs):
        contexts.append(' '.join([tok.text for tok in s if tok.pos_ in pos_tags]))
    return contexts


def get_data_class(input_type, cfg):
    if input_type == InputType.REGULAR:
        data_path = 'src.utils.data'
        data_class_name = 'GenDataset'
    elif input_type == InputType.TEMPLATE:
        data_path = 'src.utils.data'
        data_class_name = 'TemplateDataset'
    elif input_type == InputType.SEQUENTIAL:
        data_path = 'src.utils.data'
        data_class_name = 'SequentialDataset'
    elif input_type == InputType.CUSTOM:
        data_path = getattr(cfg.data, 'path', 'src.utils.data')
        data_class_name = f'{cfg.model.name}Dataset'
    else:
        raise NotImplementedError('The selected model does not have an implemented input type')

    try:
        data_module = importlib.import_module(data_path)
        data_class = getattr(data_module, data_class_name)
    except Exception:
        raise ValueError(f'Unable to find the data class "{data_path}.{data_class_name}" could not be found.')

    return data_class


class Mapper:
    def __init__(self, idx2item: pd.Series):
        idx2item.sort_index(inplace=True)
        # Assert the index is monotonically sorted with +1 increase each time
        assert idx2item.index.tolist() == list(range(len(idx2item)))
        self.item2idx = pd.Series(idx2item.index.values, index=idx2item.values)
        self.idx2item = idx2item.to_frame('raw')

    def add_items(self, items):
        items = list(set(items).difference(self.idx2item['raw']))
        offset = len(self.item2idx)
        new_item2idx = {item: i + offset for i, item in enumerate(items)}
        self.item2idx = pd.concat([self.item2idx, pd.Series(new_item2idx)])
        self.idx2item = pd.concat((self.idx2item, pd.Series(new_item2idx.keys(),
                                                            index=new_item2idx.values()).to_frame(name='raw')))

    def replace_items(self, item2new):
        self.item2idx.rename(index=item2new, inplace=True)
        self.idx2item.replace(item2new, inplace=True)

    def get_vocab(self):
        return self.item2idx.index.tolist()

    def get_item(self, idx):
        return self.idx2item['raw'][idx]

    def get_items(self, idxs):
        return self.idx2item['raw'][idxs].tolist()

    def get_idx(self, item):
        return self.item2idx[item]

    def get_idxs(self, items):
        return self.item2idx[items].tolist()

    def get_enc(self, idxs):
        return list(self.idx2item['enc'][idxs])

    def get_emb(self, idxs):
        return self.embs[list(idxs)].tolist()
        # return list(self.idx2item['emb'][idxs])

    def del_embs(self):
        del self.embs
        gc.collect()

    def enc_len(self):
        if 'enc' in self.idx2item.columns:
            return self.idx2item['enc'].str.len().max()
        return 1

    def encode(self, tok, **kwargs):
        self.idx2item['enc'] = tok(self.idx2item['raw'].tolist(), **kwargs)['input_ids']

    def embed(self, sent_emb, **kwargs):
        batch_size = kwargs.get('batch_size', 64)
        chunk_size = kwargs.get('chunk_size', 10000)
        if torch.cuda.device_count() > 1 or not torch.cuda.is_available():
            self.embs = self.multigpu_embed(sent_emb, batch_size, chunk_size)
        else:
            self.embs = self.chunk_embed(sent_emb, batch_size, chunk_size)

    def multigpu_embed(self, embedder, batch_size, chunk_size):
        pool = embedder.start_multi_process_pool()
        embs = embedder.encode_multi_process(self.idx2item['raw'], pool,
                                             batch_size=batch_size, chunk_size=chunk_size)
        embedder.stop_multi_process_pool(pool)
        return embs

    def chunk_embed(self, embedder, batch_size, chunk_size):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        texts = self.idx2item['raw'].tolist()
        n_chunks = int(np.ceil(len(texts) / chunk_size))
        embs = []
        for i in range(n_chunks):
            embs.append(embedder.encode(texts[i * chunk_size: (i + 1) * chunk_size], batch_size,
                                        device=device, show_progress_bar=False))
        return np.concatenate(embs, axis=0)

    def save_embed(self, path_to_file):
        np.save(path_to_file, self.embs)
        # self.idx2item['emb'].to_pickle(path_to_file)

    def load_embed(self, path_to_file):
        self.embs = np.load(path_to_file)
        # self.idx2item['emb'] = pd.read_pickle(path_to_file).loc[self.idx2item.index.values].values

    def __len__(self):
        return len(self.item2idx)


class EntityTokenizer:
    def __init__(self, special_tokens=None):
        self.idx2entity = []
        if special_tokens is not None:
            self.idx2entity += list(special_tokens.values())
        self.entity2idx = pd.Series({e: i for i, e in enumerate(self.idx2entity)}, dtype=int)

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def add_entities(self, es: list):
        es = list(set(es).difference(self.idx2entity))
        offset = len(self.entity2idx)
        self.entity2idx = pd.concat([self.entity2idx, pd.Series({es: i + offset for i, es in enumerate(es)})])
        self.idx2entity.extend(es)

    def encode(self, entity):
        return self.entity2idx[entity]

    def batch_encode(self, entities):
        return self.entity2idx[entities].values

    def batch_decode(self, idxs):
        return np.array(self.idx2entity)[idxs]

    def __len__(self):
        return len(self.idx2entity)


def get_tokenizer(path, data, tokenizer_name='default', special_tokens=None, vocab_size=VOCAB_SIZE, retrain=False,
                  padding_side='right', truncation_side='right', test=False, use_auth_token=False):
    if special_tokens is None:
        special_tokens = dict()
    if test:
        tok_path = os.path.join(path, f'{tokenizer_name}{TEST_SUFFIX}.json')
    else:
        tok_path = os.path.join(path, f'{tokenizer_name}.json')
    special_tok_vals = list(special_tokens.values())

    if os.path.exists(tok_path) and not retrain:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path, padding_side=padding_side,
                                            truncation_side=truncation_side)
        special_tokens = {k: v if getattr(tokenizer, k) is None else getattr(tokenizer, k) for k, v in
                          special_tokens.items()}
        tokenizer.add_special_tokens(special_tokens)
    elif tokenizer_name == I_COL:
        tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOK))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        trainer = trainers.WordLevelTrainer(special_tokens=special_tok_vals)
        tokenizer.train_from_iterator(iter(data), trainer=trainer, length=len(data))
        tokenizer.save(tok_path)
        tokenizer = get_tokenizer(path, data, tokenizer_name, special_tokens, padding_side=padding_side,
                                  truncation_side=truncation_side, retrain=False)
    elif tokenizer_name == U_COL:
        tokenizer = EntityTokenizer()
        tokenizer.add_entities(data)
    elif tokenizer_name == 'bpe':
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOK))
        # tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS_TOK} $0 {EOS_TOK}",
            special_tokens=[(BOS_TOK, special_tok_vals.index(BOS_TOK)), (EOS_TOK, special_tok_vals.index(EOS_TOK))],
        )
        trainer = trainers.BpeTrainer(special_tokens=special_tok_vals, show_progress=True,
                                      vocab_size=vocab_size + len(special_tokens))
        tokenizer.train_from_iterator(iter(data), trainer=trainer, length=len(data))
        tokenizer.save(tok_path)
        tokenizer = get_tokenizer(path, data, tokenizer_name, special_tokens, padding_side='right',
                                  truncation_side='right', retrain=False)
    elif tokenizer_name == 'default':
        tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOK))
        # tokenizer.normalizer = Lowercase()
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS_TOK} $0 {EOS_TOK}",
            special_tokens=[(BOS_TOK, special_tok_vals.index(BOS_TOK)), (EOS_TOK, special_tok_vals.index(EOS_TOK))],
        )
        trainer = trainers.WordLevelTrainer(special_tokens=special_tok_vals, show_progress=True,
                                            vocab_size=vocab_size + len(special_tokens))
        tokenizer.train_from_iterator(iter(data), trainer=trainer, length=len(data))
        tokenizer.save(tok_path)
        tokenizer = get_tokenizer(path, data, tokenizer_name, special_tokens, padding_side='right',
                                  truncation_side='right', retrain=False)
    else:
        # if tokenizer_name in ['gpt2', 'meta-llama/Llama-2-7b-hf']:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True,  # **special_tokens,
                                                      use_auth_token=use_auth_token)
            special_tokens = {k: v for k, v in special_tokens.items() if k not in tokenizer.special_tokens_map}
            tokenizer.add_special_tokens(special_tokens)
        except Exception:
            raise ValueError(f'Tokenizer not implemented: {tokenizer_name}')

    return tokenizer


def postprocessing(string):
    '''
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub('\(', ' ( ', string)
    string = re.sub('\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string


# class GloVe:
#     def __init__(self, f_name):
#         if not os.path.isfile(os.path.join(WE_PATH, f'{f_name}.txt')):
#             self._download_we(f_name)
#
#         w2ix = {}
#         vectors = []
#
#         # Read in the data.
#         with open(os.path.join(WE_PATH, f'{f_name}.txt'), 'r', encoding='utf-8') as savefile:
#             for i, line in enumerate(savefile):
#                 tokens = line.split(' ')
#
#                 word = tokens[0]
#                 entries = tokens[1:]
#
#                 w2ix[word] = i
#                 vectors.append(np.array(float(x) for x in entries))
#
#         self.w2ix = pd.Series(w2ix)
#         self.vectors = np.concatenate(vectors, axis=0)
#
#     def
#
#     @staticmethod
#     def _download_we(f_name):
#         with urllib.request.urlopen(f'http://nlp.stanford.edu/data/wordvecs/{f_name}.zip') \
#                 as response, open(os.path.join(WE_PATH, f'{f_name}.zip'), 'wb') as out_file:
#             shutil.copyfileobj(response, out_file)
#
#         with zipfile.ZipFile(os.path.join(WE_PATH, f'{f_name}.zip'), 'r') as zip_ref:
#             zip_ref.extractall(WE_PATH)
#
#         os.remove(os.path.join(WE_PATH, f'{f_name}.zip'))


def batch_list2dict(batch, pad_mask_cols=None):
    batch_size = len(batch)
    new_batch = {k: [] for k in batch[0]}
    # for k in self.cols['pad_mask']:
    #     new_batch[f'{k}_mask'] = []

    aux = defaultdict(list)
    for bix, sample in enumerate(batch):
        for k in new_batch.keys():
            if pad_mask_cols and not k.endswith('_mask') and k in pad_mask_cols:
                if isinstance(sample[k], (list, tuple)):
                    aux[f'{k}_mask'].extend([torch.BoolTensor([1] * len(l)) for l in sample[k]])
                else:
                    aux[f'{k}_mask'].append(torch.BoolTensor([1] * len(sample[k])))
            if isinstance(sample[k], (list, tuple)):
                # Assign sample index to each item in the list
                aux[f'{k}_bix'].append(torch.ones(len(sample[k]), dtype=torch.long) * bix)
                new_batch[k].extend(sample[k])
            else:
                new_batch[k].append(sample[k])

    new_batch.update(aux)
    new_batch['size'] = batch_size
    return new_batch


def pad2longest(tensors, dim=0, side='left', value=0):
    """
    Adds padding to the LEFT of each tensor up to the longest length in the batch
    """
    last_dim = len(tensors[0].shape) - 1
    if dim not in [-1, last_dim]:
        tensors = [tensor.transpose(dim, last_dim) for tensor in tensors]
        return pad2longest(tensors, dim=-1, side=side, value=value).transpose(dim + 1, last_dim + 1)

    max_l = max([tensor.shape[-1] for tensor in tensors])
    if side == 'left':
        return torch.stack([F.pad(tensor, pad=(max_l - tensor.shape[-1], 0), value=value) for tensor in tensors])
    else:
        return torch.stack([F.pad(tensor, pad=(0, max_l - tensor.shape[-1]), value=value) for tensor in tensors])


def pad2len(tensors, length, dim=0, side='left', value=0):
    """
    Adds padding to the LEFT of each tensor up to the given length
    """
    last_dim = len(tensors[0].shape) - 1
    if dim not in [-1, last_dim]:
        tensors = [tensor.transpose(dim, last_dim) for tensor in tensors]
        return pad2len(tensors, length, dim=-1, side=side, value=value).transpose(dim + 1, last_dim + 1)

    if side == 'left':
        return torch.stack([F.pad(tensor, pad=(length - tensor.shape[-1], 0), value=value) for tensor in tensors])
    else:
        return torch.stack([F.pad(tensor, pad=(0, length - tensor.shape[-1]), value=value) for tensor in tensors])
    # return torch.stack([F.pad(tensor, pad=(length - tensor.shape[-1], 0), value=0) for tensor in tensors])


def sliding_w_hist(g: pd.DataFrame, max_l: int = MAX_HIST_LEN):
    return [g.index.values[max(i - max_l, 0):i].tolist() for i in range(g.shape[0])]


class BaseDataset(Dataset):
    def __init__(self, data_cfg, model_cfg, device, data_file='reviews.pkl', **kwargs):
        self.cfg = data_cfg  # self.get_config(cfg)
        self.model_cfg = model_cfg
        self.logger = kwargs.get('logger', logging.getLogger(PROJECT_NAME))
        self.cfg.seed = kwargs.get('seed', None)
        self.cfg.hist_len = getattr(self.cfg, 'hist_len', 0)
        self.do_eval = getattr(self.cfg, 'do_eval', True)
        self.workers = {
            'train': getattr(data_cfg, 'train_workers', 0),
            'valid': getattr(data_cfg, 'eval_workers', 5),
            'test': getattr(data_cfg, 'eval_workers', 5)
        }
        self.batch_size = {
            'train': getattr(data_cfg, 'train_batch_size', data_cfg.batch_size),
            'valid': getattr(data_cfg, 'eval_batch_size', data_cfg.batch_size),
            'test': getattr(data_cfg, 'eval_batch_size', data_cfg.batch_size)
        }
        self.add_special_toks = False
        self.device = device

        self.phase = 'train'

        suffix = ''
        if self.cfg.test_flag:
            suffix = TEST_SUFFIX
            data_file = f'{suffix}.'.join(data_file.split('.'))
            self.cfg.fold = self.cfg.fold + suffix

        self.suffix = suffix
        self.aux_path = os.path.join(DATA_PATHS[self.cfg.dataset], DATA_MODE, self.cfg.fold, 'aux')

        # Keep track of which columns to add to the batch
        self._reset_stage()

        # READ AND ADJUST DATA (self.data, self.split_ixs, self.mappers)
        self.read_data(data_file, suffix)

        assert not (self.data[EXP_COL].apply(len) == 0).any()
        assert not (self.data[FEAT_COL].apply(len) == 0).any()
        # print(f'User max len: {self.data[U_HIST_EXP_COL].apply(len).max()}')
        # print(f'Item max len: {self.data[I_HIST_EXP_COL].apply(len).max()}')

        self.feature_set = self.mappers[FEAT_COL].get_vocab()
        self.mappers[FEAT_COL].replace_items({'': UNK_TOK})

        # GET GENERAL INFORMATION ABOUT THE DATASET
        self.data_info = self.get_data_info()

        self.item_set = set(self.mappers[I_COL].item2idx.tolist())

        self.handle_aux_cols()

        # ASSERT TEST USERS / ITEMS APPEAR IN TRAIN SET
        train_ixs = self.split_ixs['train']  # + self.split_ixs['valid']
        # QUESTION: Should we consider train or train+valid to train the tokenizer? CompExp uses the train file only
        train_exps = self.mappers[EXP_COL].get_items(self.data.loc[train_ixs, EXP_COL].explode().unique())
        if self.cfg.assert_test_in_train:
            self.check_test_in_train(train_ixs)

        self.logger.info(f'Getting {self.cfg.word_tok} tokenizer...')
        tok_path = os.path.join(DATA_PATHS[self.cfg.dataset], DATA_MODE, self.cfg.fold, 'tokenizers')
        self.tok = get_tokenizer(tok_path, train_exps, self.cfg.word_tok, SPECIAL_TOKENS, self.cfg.vocab_sz,
                                 self.cfg.retrain_tok, test=self.cfg.test_flag, use_auth_token=self.cfg.hf_auth_token)

        self.data_info.n_wtokens = len(self.tok)
        # self.data_info.special_toks = {v: self.tok.convert_tokens_to_ids(v) for v in SPECIAL_TOKENS.values()}
        self.data_info.tok = self.tok
        self.data_info.mappers = self.mappers

        # SELECT SUBSET OF TRAINING DATA FOR TESTING SPEED PURPOSES
        # if self.cfg.test_flag:
        #     self.split_ixs['train'] = self.split_ixs['train'][:1000]

        # PROCESS PAST INFORMATION
        if self.cfg.seq_mode.requires_past_info():
            self.handle_past_info()

        self._finish_init()
        self.train()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data.loc[idx].to_dict()
        sample['index'] = idx

        data = {
            # f'gt_{EXP_COL}': torch.LongTensor(sample[EXP_COL]),
            # f'gt_{FEAT_COL}': torch.LongTensor(list(set(sum(sample[FEAT_COL], []))))
        }

        for getter_fn in self.getter_fns:
            getter_fn(sample, data)

        for mod_fn in self.mod_fns:
            data = mod_fn(sample, data)

        return data

    def _reset_stage(self):
        self.cols = {
            'orig': [U_COL, I_COL],  # torch.LongTensor() that does not require encoding
            'float': [],  # torch.FloatTensor()
            'enc': {},
            # torch.LongTensor() that does require encoding (self.mappers) -- Contains dict {sample_col: mapper}
            'emb': {},  # torch.FloatTensor() that requires embedding -- Contains dict {sample_col: mapper}
            'processed': [],  # Already processed as torch.tensor()
            'pad_mask': []  # Get padding mask for these columns
        }
        self.keep_cols = [U_COL, I_COL, EXP_COL, FEAT_COL, RAT_COL, TIMESTEP_COL]
        self.getter_fns = [self.get_basic]  # Methods to decide input and target cols to be retrieved
        self.mod_fns = []  # Post process to retrieved info e.g. convert graph to PyG Data

    def eval(self, phase):
        assert phase in ['train', 'valid', 'test']
        self.phase = phase.lower()

    def train(self):
        self.phase = 'train'

    def get_hist_len(self):
        if not hasattr(self.cfg, 'hist_len'):
            self.cfg.hist_len = 0

    def get_hist_data(self, data_file, suffix=''):
        raise NotImplementedError()

    def set_stage(self, stage, model=None):
        # By default, data does not change from one training-eval stage to another
        pass

    def get_data_info(self):
        return SimpleNamespace(
            dataset=f'{self.cfg.dataset}_{self.cfg.fold}',
            n_users=len(self.mappers[U_COL]),
            n_items=len(self.mappers[I_COL]),
            n_feats=len(self.feature_set),
            n_sents=len(self.mappers[EXP_COL]),
            max_rating=self.data[RAT_COL].max(skipna=True),
            mean_rating=self.data[RAT_COL].mean(skipna=True),
            min_rating=self.data[RAT_COL].min(skipna=True),
            special_items={v: self.mappers[I_COL].get_idx(v) for v in SPECIAL_TOKENS.values()}
        )

    def _finish_init(self):
        # ENCODE BASIC FEATURES (FEAT, EXP, CONTEXT)
        self.handle_encodings()

        self.data_info.lengths = self.get_required_lengths()

        # Remove non-necessary columns from self.data
        rm_cols = set(self.data.columns).difference([c for cols in self.cols.values() for c in cols] + self.keep_cols)
        self.data.drop(rm_cols, axis=1, inplace=True)

    def get_basic(self, sample, data):
        for c, v in sample.items():
            if c in self.cols['processed']:
                data[c] = v
            elif c in self.cols['orig']:
                data[c] = torch.tensor(v, dtype=torch.long)
            elif c in self.cols['enc']:
                m = self.cols['enc'][c]
                if isinstance(v, (list, tuple)):
                    data[c] = [torch.LongTensor(self.mappers[m].get_enc(si)) for si in v]
                else:
                    data[c] = torch.LongTensor(self.mappers[m].get_enc(v))
            elif c in self.cols['emb']:
                m = self.cols['emb'][c]
                if isinstance(v, (list, tuple)):
                    data[f'{EMB_PREFIX}_{c}'] = [torch.LongTensor(self.mappers[m].get_emb(si)) for si in v]
                else:
                    data[f'{EMB_PREFIX}_{c}'] = torch.LongTensor(self.mappers[m].get_emb(v))
            elif c in self.cols['float']:
                data[c] = torch.FloatTensor(v)

    def read_data(self, data_file, suffix=''):
        self.logger.info('Reading data, mappers and split indexes...')
        ixs_dir = os.path.join(DATA_PATHS[self.cfg.dataset], DATA_MODE, str(self.cfg.fold))
        data_path = os.path.join(DATA_PATHS[self.cfg.dataset], DATA_MODE, data_file)
        assert os.path.exists(data_path)
        data = pd.read_pickle(data_path)
        split_ixs = load_partition(ixs_dir)

        # Assert uniqueness of indexes in dataset and partitions
        assert data.index.is_unique
        for _, v in split_ixs.items():
            assert len(np.unique(v)) == len(v)

        # Ensure monotonic index by steps of 1
        self.monotonic_ix = (data.sort_index().index.values == np.arange(stop=data.shape[0], dtype=int)).all()
        if not self.monotonic_ix:
            data.reset_index(inplace=True)
            mapping = pd.Series(data.index.values, index=data['index'])
            for c, v in split_ixs.items():
                split_ixs[c] = mapping.loc[split_ixs[c]].tolist()
            data.drop('index', inplace=True, axis=1)

        if not pd.api.types.is_float_dtype(data[RAT_COL].dtype):
            data[RAT_COL] = pd.to_numeric(data[RAT_COL], downcast="float")

        # NOTE: P5 does subword tokenization of user and item encoded IDs.
        mapper_path = os.path.join(DATA_PATHS[self.cfg.dataset], DATA_MODE, 'mappers')
        mappers = {}
        for c in (U_COL, I_COL, EXP_COL, FEAT_COL):
            if os.path.isfile(os.path.join(mapper_path, f'{c}{suffix}.pkl')):
                mapper = pd.read_pickle(os.path.join(mapper_path, f'{c}{suffix}.pkl'))
            else:
                raise FileNotFoundError(f'Encodings for column {c} were not found. Please, run process_data.py before '
                                        f'running the experiments.')

            mappers[c] = Mapper(mapper)

        mappers[I_COL].add_items(SPECIAL_TOKENS.values())  # mappers[I_COL].get_idx(UNK_TOK)
        mappers[EXP_COL].add_items([''])

        self.data, self.split_ixs, self.mappers = data, split_ixs, mappers

        unrolled = self.data[[I_COL, U_COL, FEAT_COL]].explode(FEAT_COL).explode(FEAT_COL)
        self.u2f = unrolled.groupby(U_COL)[FEAT_COL].agg(set)
        self.i2f = unrolled.groupby(I_COL)[FEAT_COL].agg(set)

    def get_required_lengths(self):
        return {c: self.mappers[c].enc_len() for c in [U_COL, I_COL, FEAT_COL, EXP_COL]}
        #     U_COL: self.data[U_COL].apply(len).max(),
        #     I_COL: self.data[I_COL].apply(len).max(),
        #     FEAT_COL: self.mappers[FEAT_COL].enc_len(),
        #     EXP_COL: self.mappers[EXP_COL].enc_len(),
        #     # U_HIST_EXP_COL: self.cfg.hist_len if self.cfg.seq_mode == SeqMode.HIST_ITEM_U_EXP_EMB else self.cfg.hist_len * (
        #     #         self.cfg.txt_len + 1)
        # }

    def get_transform_mapping(self, col_data):
        """
        Gets the encoding mapper and transforms the passed data (inplace operation)
        """

        def replace(r):
            if isinstance(r[0], list):
                return [replace(ri) for ri in r]
            else:
                return mapper.loc[r].tolist()

        unique_vals = col_data.copy()
        is_nested = False
        while isinstance(unique_vals.iloc[0], list):
            is_nested = True
            unique_vals = unique_vals.explode()
        unique_vals = sorted(unique_vals.unique().tolist())
        # unique_vals = col_data.unique().tolist()
        mapper = pd.Series(range(len(unique_vals)), index=unique_vals)
        if is_nested:
            col_data = col_data.apply(replace)
        else:
            col_data = mapper.loc[col_data.values].tolist()
        return mapper

    def handle_aux_cols(self):
        if self.cfg.requires_rating:
            self.data[RAT_COL] = self.data[RAT_COL].values[:, None].tolist()
            self.cols['float'].append(RAT_COL)
        else:
            self.data.drop(RAT_COL, axis=1, inplace=True)

        if self.cfg.requires_context and self.cfg.mod_context_flag:
            aux = self.data[EXP_COL].explode()
            aux = self.mappers[EXP_COL].get_items(aux.tolist())
            self.data[CONTEXT_COL] = get_context(aux.groupby(level=0).agg(' '.join).tolist(), pos_tags=['NOUN', 'ADJ'])
            self.mappers[CONTEXT_COL] = Mapper(self.get_transform_mapping(self.data[CONTEXT_COL]))

    def handle_past_info(self):
        pass

    def handle_encodings(self, ui_as_list=True):
        """
        This method tokenizes / embeds all required sentences (in self.mappers)
        """
        self.logger.info('Encoding necessary fields...')

        if ui_as_list:
            # Make User and Item values become lists
            # NOTE: USER and ITEM columns are supposed to be already encoded in process_data.py
            self.data[I_COL] = self.data[I_COL].values[:, None].tolist()
            self.data[U_COL] = self.data[U_COL].values[:, None].tolist()

        if self.cfg.requires_feat:
            self.cols['enc'][FEAT_COL] = FEAT_COL

        if self.cfg.requires_exp:
            self.cols['enc'][EXP_COL] = EXP_COL

        if self.cfg.requires_context:
            if self.cfg.mod_context_flag:
                assert CONTEXT_COL in self.mappers
                self.cols['enc'][CONTEXT_COL] = CONTEXT_COL
            else:
                # self.mappers[CONTEXT_COL] = copy.deepcopy(self.mappers[EXP_COL])
                self.cols['enc'][CONTEXT_COL] = EXP_COL

        for m in set(self.cols['enc'].values()):
            if m == EXP_COL:
                self.mappers[m].encode(self.tok, truncation=True, max_length=self.cfg.txt_len,
                                       add_special_tokens=self.add_special_toks)
            elif m == CONTEXT_COL:
                self.mappers[m].encode(self.tok, truncation=True, max_length=self.cfg.txt_len, add_special_tokens=False)
            elif m in [FEAT_COL, ASP_COL]:
                self.mappers[m].encode(self.tok, add_special_tokens=False)

        if self.cfg.emb_models is not None:
            self.logger.info('Obtaining necessary embeddings...')
            if not hasattr(self, 'precomputed_emb_dim'):
                self.precomputed_emb_dim = {}
            for emb_model in self.cfg.emb_models:
                # Load Sentence Embedder only when necessary
                c_req_m = [c for c in emb_model.apply_to
                           if not os.path.isfile(os.path.join(DATA_PATHS[self.cfg.dataset],
                                                              DATA_MODE, 'embeddings',
                                                              f"emb_{emb_model.name.replace('/', '%')}_{c}.npy"))]
                if c_req_m:
                    embedder = self.get_emb_model(emb_model)

                for c in emb_model.apply_to:
                    self.cols['emb'][c] = c
                    f_name = f"emb_{emb_model.name.replace('/', '%')}_{c}{self.suffix}.npy"
                    emb_path = os.path.join(DATA_PATHS[self.cfg.dataset], DATA_MODE, 'embeddings')
                    if not os.path.isdir(emb_path):
                        os.makedirs(emb_path)

                    path_to_file = os.path.join(emb_path, f_name)
                    if os.path.isfile(path_to_file):
                        self.logger.info(f"Loading {emb_model.name} embeddings for {c} field...")
                        self.mappers[c].load_embed(path_to_file)
                    else:
                        logging.info(f"Computing {emb_model.name} embeddings for {c} field...")
                        self.mappers[c].embed(embedder,
                                              batch_size=getattr(emb_model, 'batch_size', self.batch_size['train']))
                        logging.info(f"Saving {emb_model.name} embeddings for {c} field...")
                        self.mappers[c].save_embed(path_to_file)

                    self.precomputed_emb_dim[c] = self.mappers[c].embs.shape[1]
                    # embs = torch.FloatTensor(self.mappers[c].get_emb(range(len(self.mappers[c]))))
                    # setattr(self.data_info, f'{c}_emb', embs)

            # if self.cfg.requires_exp:
            #     self.cols['emb'].append(EXP_COL)
            #     self.mappers[EXP_COL].embed(sent_emb, batch_size=self.cfg.batch_size)
            # if self.cfg.requires_feat:
            #     self.cols['emb'].append(FEAT_COL)
            #     self.mappers[FEAT_COL].embed(sent_emb, batch_size=self.cfg.batch_size)

    def get_emb_model(self, emb_cfg):
        if emb_cfg.type == 'huggingface':
            return SentenceTransformer(emb_cfg.name, device=self.device, use_auth_token=self.cfg.hf_auth_token)
        # elif cfg.type == 'glove':
        #     f_name = getattr(cfg, 'f_name', 'glove.42B.300d')
        #     raise NotImplementedError('GloVe embeddings are not implemented into the framework yet')
        else:
            raise NotImplementedError(f'{emb_cfg.type} embeddings are not implemented into the framework yet')

    def neg_sample(self, item_seq: set, n_neg=99, replace=False):
        # neg_items = self.item_set.difference(self.user_item.loc[user])
        # assert item_seq.issubset(self.item_set)
        neg_items = list(self.item_set.difference(item_seq))
        if n_neg >= len(neg_items):
            random.shuffle(neg_items)
        else:
            neg_items = np.random.choice(neg_items, n_neg, replace=replace).tolist()
        return neg_items

    def get_gen_labels(self, gen_steps, n_batches, batch_size, pred_tasks, phase):
        labels = {}
        # Labels for EXP and FEAT assume the split was not shuffled
        select_ixs = self.split_ixs[phase]
        if 0 < gen_steps < n_batches:
            select_ixs = select_ixs[:gen_steps * batch_size]

        for meta_c in [U_COL, I_COL]:
            labels[f'metadata-{meta_c}'] = self.data.loc[select_ixs, meta_c].explode().tolist()
        labels[Task.EXPLANATION] = self.data.loc[select_ixs, EXP_COL].tolist()
        labels[Task.FEAT] = self.data.loc[select_ixs, FEAT_COL]
        if isinstance(labels[Task.FEAT].iloc[0][0], Iterable):
            labels[Task.FEAT] = labels[Task.FEAT].apply(partial(sum, start=[])).apply(lambda l: list(set(l))).tolist()
        else:
            labels[Task.FEAT] = labels[Task.FEAT].apply(lambda l: list(set(l))).tolist()

        if Task.RATING in pred_tasks:
            labels[Task.RATING] = self.data.loc[select_ixs, RAT_COL].tolist()
        for t in [Task.NEXT_ITEM, Task.TOPN]:
            if t in pred_tasks:
                labels[t] = labels[f'metadata-{I_COL}']

        return labels

    def get_subset(self, split):
        return Subset(dataset=self, indices=self.split_ixs[split])

    def get_dataloaders(self, drop_last=False):
        dls = []
        for split in ['train', 'valid', 'test']:
            dls.append(self.get_dataloader(split, drop_last=(drop_last if split == 'train' else False)))
        return dls

    def get_dataloader(self, split=None, drop_last=False):
        return DataLoader(self.get_subset(split), self.batch_size[split], num_workers=self.workers[split],
                          collate_fn=self.collate, shuffle=(split == 'train'), drop_last=drop_last)
        # val_dl = DataLoader(self.get_subset('valid'), self.cfg.batch_size, num_workers=self.eval_workers,
        #                     collate_fn=self.collate, shuffle=False)
        # tst_dl = DataLoader(self.get_subset('test'), self.cfg.batch_size, num_workers=self.eval_workers,
        #                     collate_fn=self.collate, shuffle=False)
        # return trn_dl, val_dl, tst_dl

    def check_test_in_train(self, train_ixs):
        def clean_histories(hist_series: pd.Series):
            return hist_series.explode() \
                .replace(unk_items, self.mappers[I_COL].get_idx(UNK_TOK)) \
                .dropna() \
                .astype(np.int64) \
                .groupby(level=0).agg(list)

        logging.info('Checking test items are present in the train+valid set...')
        train_items = set(self.data.loc[train_ixs, I_COL].tolist())
        train_users = set(self.data.loc[train_ixs, U_COL].tolist())

        val_tst_ixs = self.split_ixs['valid'] + self.split_ixs['test']
        mask = self.data.loc[val_tst_ixs, I_COL].isin(train_items)

        unk_items = self.data.loc[np.array(val_tst_ixs)[np.argwhere(~mask.values)[:, 0]], I_COL].unique()
        self.item_set.difference_update(unk_items)

        # Replace unseen items from the item history column with the UNK_ITEM
        if U_HIST_I_COL in self.data.columns:
            self.data.loc[val_tst_ixs, U_HIST_I_COL] = clean_histories(self.data.loc[val_tst_ixs, U_HIST_I_COL])

        mask = mask & self.data.loc[val_tst_ixs, U_COL].isin(train_users)
        # self.data['split'] = 0
        # self.data.loc[self.split_ixs['valid'], 'split'] = 1
        # self.data.loc[self.split_ixs['test'], 'split'] = 2
        # split_cnts = self.data['split'].value_counts()
        # diff_counts = self.data['split'].value_counts() - split_cnts
        # assert diff_counts.loc[0] == 0
        # for i, phase in ['valid', 'test']:
        #     self.split_ixs[phase] =

        for phase in ['valid', 'test']:
            n_samples = len(self.split_ixs[phase])
            keep_ixs = np.argwhere(mask.values[:n_samples])[:, 0]
            self.split_ixs[phase] = np.array(self.split_ixs[phase])[keep_ixs].tolist()
            logging.info(f'Out of {n_samples} {phase} samples, {n_samples - len(self.split_ixs[phase])} were removed '
                         f'as either user or item was not present in the training set')
            mask = mask[n_samples:]

        self.data = self.data.loc[sum(self.split_ixs.values(), start=[])].sort_values(TIMESTEP_COL)
        # self.split_ixs['test'] = np.array(self.split_ixs['test'])[np.argwhere(mask.values)[:, 0]].tolist()

    def flatten_cols(self, filter_unique=False, preserve_order=False):
        self.data[FEAT_COL] = self.data[FEAT_COL].apply(partial(sum, start=[]))

        if preserve_order:
            # NOTE 1: ~x3 slower but preserves temporal order of sentences (unique starts from most recent sentences)
            # NOTE 2: pandas .unique() states "Uniques are returned in order of appearance"
            unique_fn = lambda l: pd.Series(l[::-1]).unique().tolist()[::-1]
        else:
            unique_fn = lambda l: list(set(l))

        if filter_unique:
            for c in [FEAT_COL, EXP_COL]:
                self.data[c] = self.data[c].apply(unique_fn)

    def collate(self, batch):
        batch = batch_list2dict(batch, self.cols['pad_mask'])

        for c, _ in self.cols['enc'].items():
            batch[c] = self.tok.pad({'input_ids': batch[c]}, return_tensors='pt', padding='longest')['input_ids']

        for k in self.cols['pad_mask']:
            k = k.replace("_mask", "")
            batch[f'{k}_mask'] = pad2longest(batch[f'{k}_mask'], dim=0, side='right', value=0)

        aux = {}
        for k in batch:
            if isinstance(batch[k], list):
                try:
                    batch[k] = torch.stack(batch[k])
                except RuntimeError:
                    lens = list(map(len, batch[k]))
                    if len(set(lens)) > 1 and not k.endswith('_bix') and f'{k}_bix' not in batch:
                        aux[f'{k}_bix'] = torch.repeat_interleave(torch.arange(len(batch[k])), torch.tensor(lens))
                    batch[k] = torch.cat(batch[k])

        if aux:
            batch.update(aux)
        return batch

    # def collate_eval(self, batch):
    #     raise NotImplementedError('Collate_eval is not implemented for this type of Dataset')


class GenDataset(BaseDataset):
    def _finish_init(self):
        gen_sampling = getattr(self.cfg, 'gen_sampling', 'alternate')
        if gen_sampling == 'alternate':
            self.getter_fns = [self.sample_exps] + self.getter_fns
        elif gen_sampling == 'fixed':
            self.sample_fixed()
            self.getter_fns = [self.sample_exps_fixed] + self.getter_fns
            self.keep_cols += [f'sampled_{FEAT_COL}', f'sampled_{EXP_COL}']
        else:
            raise NotImplementedError(f'Generative sampling "{gen_sampling}" not implemented. '
                                      f'Use "fixed" or "alternate" instead.')

        self.add_special_toks = True
        super()._finish_init()

    def sample_fixed(self):
        fe = self.data[[FEAT_COL, EXP_COL]].explode([FEAT_COL, EXP_COL]).sample(frac=1)
        fe = fe[~fe.index.duplicated(keep='first')]
        fe[FEAT_COL] = fe[FEAT_COL].map(random.choice)
        fe.rename(lambda c: f'sampled_{c}', inplace=True, axis=1)
        self.data = self.data.join(fe)
        assert not any(self.data.index.duplicated())
        assert not self.data[[f'sampled_{FEAT_COL}', f'sampled_{EXP_COL}']].isna().values.any()

    def sample_exps_fixed(self, sample, data, map_to_str=False):
        sample[EXP_COL] = sample.pop(f'sampled_{EXP_COL}')
        sample[FEAT_COL] = sample.pop(f'sampled_{FEAT_COL}')
        if self.cfg.requires_context:
            # NOTE: In the next getter_fn, the CONTEXT mapper will be used to map the index to the actual context
            sample[CONTEXT_COL] = sample.get(CONTEXT_COL, sample[EXP_COL])

        if map_to_str:
            sample[FEAT_COL] = self.mappers[FEAT_COL].get_item(sample[FEAT_COL])
            sample[EXP_COL] = self.mappers[EXP_COL].get_item(sample[EXP_COL])

    def sample_exps(self, sample, data, map_to_str=False):
        # Sample exp and corresponding feats
        ix = random.randint(0, len(sample[EXP_COL]) - 1)

        if self.cfg.requires_context:
            # NOTE: In the next getter_fn, the CONTEXT mapper will be used to map the index to the actual context
            sample[CONTEXT_COL] = sample.get(CONTEXT_COL, sample[EXP_COL])[ix]

        sample[EXP_COL] = sample[EXP_COL][ix]
        sample[FEAT_COL] = sample[FEAT_COL][ix]
        # QUESTION: Do we really want to sample a single feature? What if we take all features for the selected exp.?
        # Sample a single feat from all possibilities
        ix = random.randint(0, len(sample[FEAT_COL]) - 1)
        sample[FEAT_COL] = sample[FEAT_COL][ix]
        if map_to_str:
            sample[FEAT_COL] = self.mappers[FEAT_COL].get_item(sample[FEAT_COL])
            sample[EXP_COL] = self.mappers[EXP_COL].get_item(sample[EXP_COL])


class SeqDataset(BaseDataset):
    def _finish_init(self):
        if self.cfg.requires_nextitem:
            self.getter_fns.append(self.get_neg_samples)

        super()._finish_init()

        # Set the length of the item history in the input sequence
        if U_HIST_I_COL in self.data.columns and hasattr(self.data_info, 'lengths'):
            self.data_info.lengths[I_COL] *= min(self.data[U_HIST_I_COL].apply(len).max(), self.cfg.hist_len) + 1

    def read_data(self, data_file, suffix='', load_hist=True):
        super().read_data(data_file, suffix)

        if load_hist:
            self.get_hist_data(data_file, suffix)
            if self.cfg.requires_nextitem:
                self.get_negative_samples(suffix)

    def get_hist_len(self):
        if not hasattr(self.cfg, 'hist_len'):
            self.cfg.hist_len = DEFAULT_HIST_LEN

    def get_hist_data(self, data_file, suffix=''):
        def getter(ixs, select):
            # NOTE: if we want to retrieve flattened lists, use .explode() right before .tolist()
            return self.data[select].loc[ixs].tolist()

        hist_data_path = os.path.join(DATA_PATHS[self.cfg.dataset], DATA_MODE, f'{HIST_PREFIX}_{data_file}')
        if os.path.isfile(hist_data_path) and self.monotonic_ix:
            self.data = pd.concat([self.data, pd.read_pickle(hist_data_path)], axis=1)
        else:
            tqdm.pandas()
            orig_index = self.data.index.values
            hist_cols = []
            for groupby in [U_COL, I_COL]:
                logging.info(f'Extracting history for {groupby} column')
                hist_cols.append(f'{groupby}_{HIST_PREFIX}')
                self.data.sort_values([groupby, TIMESTEP_COL], inplace=True)
                histories = self.data[[groupby, TIMESTEP_COL]].groupby(groupby).progress_apply(sliding_w_hist)
                self.data[hist_cols[-1]] = sum(histories.tolist(), [])
            self.data = self.data.loc[orig_index]
            # self.data[hist_cols].to_pickle(hist_data_path)

        cols = defaultdict(list)
        if SeqMode.HIST_ITEM.value <= self.cfg.seq_mode.value <= SeqMode.HIST_ITEM_U_EXP_EMB.value:
            cols[U_COL].append(I_COL)
            cols[U_COL].append(RAT_COL)
        if self.cfg.seq_mode.value >= SeqMode.HIST_ITEM_U_EXP.value:
            cols[U_COL].append(EXP_COL)
        if self.cfg.seq_mode.value == SeqMode.HIST_UI_EXP.value:
            # cols[U_COL].append(FEAT_COL)
            # cols[I_COL].append(FEAT_COL)
            cols[I_COL].append(EXP_COL)

        self.get_hist_len()

        for groupby, select_cols in cols.items():
            max_l = getattr(self.cfg, f"{groupby}_hist_len", self.cfg.hist_len)
            idxr_col = f'{groupby}_{HIST_PREFIX}'
            has_hist_mask = self.data[idxr_col].str.len() > 0
            idxr = self.data.loc[has_hist_mask, idxr_col]
            if max_l < MAX_HIST_LEN:
                idxr = idxr.map(lambda ixs: ixs[-max_l:])
            for c in select_cols:
                new_c = f'{idxr_col}_{c}'
                self.data[new_c] = np.empty((len(self.data), 0)).tolist()
                self.data.loc[has_hist_mask, new_c] = idxr.map(partial(getter, select=c))

            self.data.drop(idxr_col, axis=1, inplace=True)

    def get_negative_samples(self, suffix=''):
        neg_sample_path = os.path.join(DATA_PATHS[self.cfg.dataset], DATA_MODE, f'user_neg_samples{suffix}.pkl')
        self.u2i = self.data[[U_COL, I_COL]].groupby(U_COL)[I_COL].agg(set)  # .str[-1]
        if os.path.exists(neg_sample_path):
            self.user_neg_samples = pd.read_pickle(neg_sample_path)
            if not set(self.u2i.index.tolist()).issubset(set(self.user_neg_samples.keys())):
                os.remove(neg_sample_path)
                raise Exception(f'Missing users in "user_neg_samples.pkl". Run process_data.get_aux_data(model_name='
                                f'"{self.model_cfg.name}, "dataset="{self.cfg.dataset}") to get the  auxiliary data '
                                f'for this model')
        else:
            raise FileNotFoundError(f'Run process_data.get_aux_data(model_name="{self.model_cfg.name}, '
                                    f'"dataset="{self.cfg.dataset}") to get the  auxiliary data for this model')

    def flatten_cols(self, filter_unique=False, preserve_order=False):
        super().flatten_cols(filter_unique, preserve_order)

        for c in [col for col in self.data.columns if HIST_PREFIX in col]:
            mask = self.data[c].str.len() > 0
            if FEAT_COL in c:
                self.data.loc[mask, c] = self.data.loc[mask, c].parallel_apply(
                    lambda ls: sum(map(partial(sum, start=[]), ls), []))
            elif EXP_COL in c:
                self.data.loc[mask, c] = self.data.loc[mask, c].apply(partial(sum, start=[]))
            if filter_unique:
                if preserve_order:
                    self.data.loc[mask, c] = self.data.loc[mask, c].parallel_apply(
                        lambda l: pd.Series(l[::-1]).unique().tolist()[::-1])
                else:
                    self.data.loc[mask, c] = self.data.loc[mask, c].apply(lambda l: list(set(l)))

    def ensure_max_len(self):
        for c in self.data.columns:
            if f'{HIST_PREFIX}_' in c and self.cfg.hist_len != 'all':
                self.data[c] = self.data[c].apply(lambda h: h[-self.cfg.hist_len:])

    def handle_past_info(self):
        super().handle_past_info()

        # Limit context length
        self.ensure_max_len()

        if self.cfg.requires_nextitem:
            # NOTE: Add UNK_TOKEN for cold_start recommendation (to estimate one item as context)
            if self.cfg.coldstart_fill:
                fill_lambda = lambda h: h if h else [self.mappers[I_COL].get_idx(UNK_TOK)]
                self.data[U_HIST_I_COL] = self.data[U_HIST_I_COL].apply(fill_lambda)
            self.cols['orig'].append(U_HIST_I_COL)

            # self.data[I_COL] = (self.data[U_HIST_I_COL] + self.data[I_COL]).str.join(' ')
            if self.cfg.seq_mode != SeqMode.HIST_ITEM_RATING_LAST:
                # self.data[U_HIST_RAT_COL] = self.data[U_HIST_RAT_COL].apply(lambda h: h if h else [self.mean_rating])
                if self.cfg.coldstart_fill:
                    fill_lambda = lambda h: h if h else [self.data_info.mean_rating]
                    self.data[U_HIST_RAT_COL] = self.data[U_HIST_RAT_COL].apply(fill_lambda)
                self.cols['float'].append(U_HIST_RAT_COL)

        if self.cfg.seq_mode.requires_past_user_exp():
            if self.cfg.coldstart_fill:
                fill_lambda = lambda h: h if h else [[self.mappers[EXP_COL].get_idx('')]]
                self.data[U_HIST_EXP_COL] = self.data[U_HIST_EXP_COL].apply(fill_lambda)
            self.cols['enc'][U_HIST_EXP_COL] = EXP_COL
            self.cols['enc'][U_HIST_FEAT_COL] = FEAT_COL
            if self.cfg.emb_models is not None:
                self.cols['emb'][U_HIST_EXP_COL] = EXP_COL
                self.cols['emb'][U_HIST_FEAT_COL] = FEAT_COL

        if self.cfg.seq_mode.requires_past_ui_exp():
            if self.cfg.coldstart_fill:
                fill_lambda = lambda h: h if h else [[self.mappers[EXP_COL].get_idx('')]]
                self.data[I_HIST_EXP_COL] = self.data[I_HIST_EXP_COL].apply(fill_lambda)
            self.cols['enc'][I_HIST_EXP_COL] = EXP_COL
            self.cols['enc'][I_HIST_FEAT_COL] = FEAT_COL
            if self.cfg.emb_models is not None:
                self.cols['emb'][I_HIST_EXP_COL] = EXP_COL
                self.cols['emb'][I_HIST_FEAT_COL] = FEAT_COL

    # def handle_encodings(self):
    #     # Encode USER and ITEM columns
    #     super().handle_encodings()

    def get_neg_samples(self, sample, data, only_past_pos=True):
        if only_past_pos:
            seq = sample[U_HIST_I_COL] + sample[I_COL]
        else:
            seq = self.u2i[sample[U_COL]]
        # Sample negative items for optimization with BCE (1 per item in the sequence - except cand. item)
        data[NEG_OPT_COL] = torch.LongTensor(self.neg_sample(seq, len(seq), replace=True))
        # Sample negative items for evaluation metrics computation
        data[NEG_EVAL_COL] = torch.LongTensor([sample[I_COL][-1]] + self.user_neg_samples[sample[U_COL][0]])


class SEQUERDataset(SeqDataset, GenDataset):
    def _finish_init(self):
        super()._finish_init()
        self.getter_fns.append(self.merge_hist)
        # self.cols['pad_mask'].extend([I_COL, RAT_COL])
        # self.cols['orig'].remove(U_HIST_I_COL)
        # self.data[I_COL] = self.data[U_HIST_I_COL] + self.data[I_COL]

    #     feat_len = self.mappers[FEAT_COL].idx2item['enc'].str.len()
    #     exp_len = self.mappers[EXP_COL].idx2item['enc'].str.len()
    #     item_len = min(self.data[U_HIST_I_COL].apply(len).max(), self.cfg.hist_len) + 1
    #     self.data_info.lengths = [1, item_len, feat_len, exp_len]
    #
    #     # TODO: Set pad cols
    #     pass

    def merge_hist(self, sample, data):
        data[I_COL] = torch.cat((data.pop(U_HIST_I_COL), data[I_COL]), dim=0)
        data[RAT_COL] = torch.cat((data.pop(U_HIST_RAT_COL), data[RAT_COL]), dim=0)
        data[SEQ_LEN_COL] = torch.LongTensor([len(data[I_COL])])

    def collate(self, batch):
        batch = batch_list2dict(batch, self.cols['pad_mask'])

        for c, _ in self.cols['enc'].items():
            batch[c] = self.tok.pad({'input_ids': batch[c]}, return_tensors='pt', padding='longest')['input_ids']

        i_pad_ix = self.mappers[I_COL].get_idx(PAD_TOK)
        batch[I_COL] = pad2len(batch[I_COL], self.cfg.hist_len + 1, side='right', value=i_pad_ix)
        batch[NEG_OPT_COL] = pad2len(batch[NEG_OPT_COL], self.cfg.hist_len + 1, side='right', value=i_pad_ix)
        batch[RAT_COL] = pad2len(batch[RAT_COL], self.cfg.hist_len + 1, side='right')

        for k in self.cols['pad_mask']:
            k = k.replace("_mask", "")
            batch[f'{k}_mask'] = pad2longest(batch[f'{k}_mask'], dim=0, side='right', value=0)

        aux = {}
        for k in batch:
            if isinstance(batch[k], list):
                try:
                    batch[k] = torch.stack(batch[k])
                except RuntimeError:
                    lens = list(map(len, batch[k]))
                    if len(set(lens)) > 1 and not k.endswith('_bix') and f'{k}_bix' not in batch:
                        aux[f'{k}_bix'] = torch.repeat_interleave(torch.arange(len(batch[k])), torch.tensor(lens))
                    batch[k] = torch.cat(batch[k])

        if aux:
            batch.update(aux)
        return batch


class TemplateDataset(SeqDataset, GenDataset):
    def _finish_init(self):
        # NOTE: This implementation does not behave exactly the same as POD. In POD implementation, an epoch iterates
        #  through the training dataset n_tasks times (once per task, alternating tasks between batches). In our
        #  implementation, tasks are alternated between batches but a single pass through the dataset occurs at each
        #  epoch, resetting the sampling indexing each time. Thus, not all samples will be prompted for every task.
        super(TemplateDataset, self)._finish_init()

        # Make sure there is only one worker loading the data to avoid collisions on the task retrieval
        self.cfg.train_workers = 0

        self._init_templater_params()

        # By default, all templates are computed for a single batch
        self.templater = self.alternate_templater

        # Make sure that getter_fns are preppended to this list in subclasses to make sure each sample has the necessary
        # info to format the model templates. Otherwise, make sure to override the get_templates function to add this info
        self.getter_fns = [partial(self.sample_exps, map_to_str=True), self.get_candidate_pool, self.get_templates]
        pad_mask = [f'{t.value}_{in_type}' for t in self.tasks for in_type in ['encoder', 'decoder']]
        # pad_mask += [f'{t.value}_whole_word_ids' for t in self.templater.tasks]
        self.cols['pad_mask'].extend(pad_mask)

    def _init_templater_params(self):
        self.col_seps, self.task2temp = get_task_templates(self.model_cfg.name)
        self.tasks = list(self.task2temp.keys())
        self.task2evalix = {t: -1 for t in self.tasks}
        self.curr_task_id = 0
        self.n_tasks = len(self.task2temp)

        # The last template is used to evaluate the model
        self.eval_ix = -1

    def switch_task(self):
        self.curr_task_id = (self.curr_task_id + 1) % len(self.task2temp)

    def train(self):
        super().train()
        self.templater = self.alternate_templater

    def eval(self, phase):
        super().eval(phase)
        self.templater = self.joint_templater

    def _format_template(self, sample, template_info):
        """ The current implementation supports encoder-decoder architectures with (encoder, decoder) input """
        template, format_cols, label_col = template_info
        return template.format(*[self._to_string(sample, c) for c in format_cols]), str(sample[label_col])

    def _to_string(self, sample, c):
        if isinstance(sample[c], list):
            sample[c] = map(str, sample[c])
            if c in self.col_seps.keys():
                return self.col_seps[c].join(sample[c])
            return ' '.join(sample[c])

        return sample[c]

    def alternate_templater(self, sample):
        curr_task = self.tasks[self.curr_task_id]
        return {curr_task: self._format_template(sample, random.choice(self.task2temp[curr_task]))}

    def sample_mixed_templater(self, sample):
        curr_task = random.choice(self.tasks)
        return {curr_task: self._format_template(sample, random.choice(self.task2temp[curr_task]))}

    def joint_templater(self, sample):
        return {t: self._format_template(sample, self.task2temp[t][self.task2evalix[t]]) for t in self.tasks}

    def handle_encodings(self, ui_as_list=False):
        super(TemplateDataset, self).handle_encodings(ui_as_list)

    def get_candidate_pool(self, sample, data):
        pos_items = self.u2i[sample[U_COL]]
        cand_pool = self.neg_sample(pos_items, self.cfg.negative_num, replace=False) + [sample[I_COL]]
        random.shuffle(cand_pool)
        sample[f'{I_COL}_pool'] = cand_pool

    def get_templates(self, sample, data):
        # Example: {'nextitem_id': id, 'nextitem_encoder': enc_encoder_input, 'nextitem_decoder': enc_decoder_input}
        for k, v in self.templater(sample).items():
            data[f'{k.value}_id'] = torch.LongTensor([self.tasks.index(k)])
            # print(f'The prompt is: {v[0]}. The expected output is: {v[1]}')
            for i, in_type in enumerate(['encoder', 'decoder']):
                # Sub-word tokenization for both input and labels
                enc = self.tok(v[i], truncation=True, return_tensors='pt')
                data[f'{k.value}_{in_type}'] = enc['input_ids'][0]
                if in_type == 'encoder':
                    # print(f'The tokens are: {enc.tokens()}')
                    data[f'{k.value}_whole_word_ids'] = torch.LongTensor(self.get_whole_word_ids(enc.tokens(),
                                                                                                 self.tok.unk_token))
                else:
                    # Truncate decoding explanation
                    data[f'{k.value}_{in_type}'] = data[f'{k.value}_{in_type}'][:self.cfg.txt_len]

    @staticmethod
    def get_whole_word_ids(subw_tokens, unk_token=UNK_TOK):
        """ Based on POD's implementation where non-ID tokens share embedding and only USER/ITEM IDs take different
        whole-word IDs """
        whole_word_ids = [0] * len(subw_tokens)
        i, curr_id = 0, 0
        while i < len(subw_tokens) - 2:
            if subw_tokens[i] == '_':
                j = i + 1
                while j < len(subw_tokens) and (subw_tokens[j].isdigit() or subw_tokens[j] == unk_token):
                    j += 1
                if j > i + 1:
                    curr_id += 1
                    whole_word_ids[i - 1:j] = [curr_id] * (j + 1 - i)
                    i = j
                else:
                    i += 2
            i += 1
        return whole_word_ids

    def get_dataloader(self, split=None, drop_last=False):
        return DataLoader(self.get_subset(split), self.batch_size[split], num_workers=self.workers[split],
                          collate_fn=self.collate, shuffle=(split == 'train'), drop_last=drop_last)

    def collate(self, batch):
        self.switch_task()
        batch = batch_list2dict(batch, self.cols['pad_mask'])

        for k in batch.keys():
            if k in self.cols['pad_mask']:
                batch[k] = self.tok.pad({'input_ids': batch[k]}, return_tensors='pt', padding='longest')['input_ids']
                batch[f'{k}_mask'] = pad2longest(batch[f'{k}_mask'], dim=0, side='right', value=0)
            elif 'whole_word_ids' in k:
                batch[k] = pad2longest(batch[k], dim=0, side='right', value=0)

        for k in batch:
            if isinstance(batch[k], list):
                try:
                    batch[k] = torch.stack(batch[k])
                except RuntimeError:
                    batch[k] = torch.cat(batch[k])

        return batch


class PODDataset(TemplateDataset):
    class AlternateTaskLoader:
        def __init__(self, dataset: TemplateDataset, split: str, tasks: Iterable[Task], min_hist: int, batch_size: int,
                     collate_fn: Callable, shuffle: bool, drop_last: bool):
            self.min_hist = min_hist
            self.batch_size = batch_size
            self.collate_fn = collate_fn
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.idx = 0

            self.dls = [self.get_dataloader(dataset, dataset.split_ixs[split], t) for t in tasks]
            self.iters = [iter(dl) for dl in self.dls]

            self.n_batches = sum(map(len, self.dls))

        def __len__(self):
            # NOTE: POD's length is misleading as it means it will take mean(map(len, self.dls)) from each dataloader
            #  instead of selecting the min length dl * 3 (for no replacement) or max length dl * 3 (for replacement)
            return self.n_batches

        def __iter__(self):
            self.idx = 0
            return self

        def __next__(self):
            if self.idx < self.n_batches:
                i = self.idx % len(self.iters)
                try:
                    batch = next(self.iters[i])
                except StopIteration:
                    self.iters[i] = iter(self.dls[i])
                    batch = next(self.iters[i])
                finally:
                    self.idx += 1
                return batch
            raise StopIteration

        def get_dataloader(self, dataset, split_ixs, task):
            if task == Task.NEXT_ITEM:
                split_ixs = self.seq_rec_filter(dataset.data.loc[split_ixs], self.min_hist)
            subset = Subset(dataset=dataset, indices=split_ixs)
            return DataLoader(subset, self.batch_size, num_workers=0,
                              collate_fn=self.collate_fn, shuffle=self.shuffle, drop_last=self.drop_last)

        @staticmethod
        def seq_rec_filter(data: pd.DataFrame, min_hist: int = 4):
            valid_mask = data[U_HIST_I_COL].str.len() >= min_hist
            return data.loc[valid_mask].index.tolist()

    def get_dataloader(self, split=None, drop_last=True):
        if split == 'train':
            return PODDataset.AlternateTaskLoader(self, split, self.tasks, getattr(self.cfg, 'min_hist', 4),
                                                  self.batch_size[split], self.collate, True, drop_last)
        return super().get_dataloader(split, drop_last=False)


class GREENerDataset(SeqDataset):
    def _finish_init(self):
        # NOTE: For larger datasets, it would be better to avoid sentence embeddings (they are not finetuned by default)
        #  and modify graph[EXP_COL].x to contain the embedded sentences instead of the index.
        #  Memory-wise seems much better.

        # self.mod_fns.append(self.build_graph)
        self.cfg.requires_feat = False
        self.hard_labeling = False
        self.sep = "|"

        setattr(self.data_info, f'tfidf_{EXP_COL}_emb', self.get_tfidf_emb())

        # self.mod_fns.append(self.to_pyg_heterograph)
        self.getter_fns.append(self.get_graph)

        super()._finish_init()

        avail_ixs = set([int(f[:-3]) for f in os.listdir(self.graph_path) if f.endswith('.pt')])
        self.split_ixs['train'] = [i for i in self.split_ixs['train'] if i in avail_ixs]
        self.split_ixs['valid'] = [i for i in self.split_ixs['valid'] if i in avail_ixs]

        # Do not retrieve the embeddings for the batch (Simply compute them in the handle_encodings method)
        self.cols['emb'] = dict()
        self.cols['enc'] = dict()

    def handle_encodings(self, ui_as_list=True):
        super().handle_encodings(ui_as_list)

        for c in [FEAT_COL, EXP_COL]:
            setattr(self.data_info, f'{c}_emb', torch.FloatTensor(self.mappers[c].embs))
            self.mappers[c].del_embs()

    def get_graph(self, sample, data):
        data['graph'] = torch.load(os.path.join(self.graph_path, f'{sample["index"]}.pt'))
        # data['graph'][EXP_COL].x = torch.FloatTensor(self.mappers[EXP_COL].get_emb(data['graph'][EXP_COL].x.numpy()))

    def get_hist_len(self):
        def ceil2num(num_to_round, num):
            return int(np.ceil(num_to_round / num) * num)

        # Handcrafted values to resemble the provided GREENer processed average graph size
        target_i_sents = 400
        target_u_sents = 150

        self.data['exp_len'] = self.data[EXP_COL].str.len()
        req_i_len = target_i_sents // self.data.groupby(I_COL)['exp_len'].mean().mean()
        req_u_len = target_u_sents // self.data.groupby(U_COL)['exp_len'].mean().mean()
        setattr(self.cfg, f'{I_COL}_hist_len', ceil2num(req_i_len, 5))
        setattr(self.cfg, f'{U_COL}_hist_len', ceil2num(req_u_len, 5))
        self.data.drop('exp_len', axis=1, inplace=True)

    def read_data(self, data_file, suffix='', load_hist=True):
        dir_suffix = ''
        if getattr(self.cfg, 'filter_item_attr', True):
            dir_suffix = f'_item_attr'
        self.graph_path = os.path.join(self.aux_path, f'GREENer_graphs_{self.cfg.hist_len}{dir_suffix}')

        if not os.path.isdir(self.graph_path):
            raise FileNotFoundError(f'Run process_data.get_aux_data(model_name="greener", dataset="{self.cfg.dataset}")'
                                    f' to get the auxiliary data for this model')

        if os.path.isdir(self.graph_path):
            super().read_data(data_file, suffix, load_hist=False)
        else:
            super().read_data(data_file, suffix)

        # Compute Unique Feat/Exp, BLEU and Adj.
        self.f2e, self.e2f = self.get_feat_exp_links(suffix)

        # g_data = self.build_load_graphs(suffix)

        # TODO: Rename columns in pickle file
        # new_names = {c: c.replace(EXP_COL, HIST_EXP_COL).replace(FEAT_COL, HIST_FEAT_COL) for c in g_data.columns}
        # g_data.rename(new_names, axis=1, inplace=True)
        # g_data[f'{EXP_COL}|x'] = self.data[EXP_COL].values
        # g_data[f'{FEAT_COL}|x'] = self.data[FEAT_COL].values

        # self.data = pd.concat((self.data, g_data), axis=1)

        self.cols['orig'] = []  # [c for c in g_data.columns if c.split("|")[-1] != 'y']
        # self.cols['float'].extend([c for c in g_data.columns if c.split("|")[-1] == 'y'])

    def assert_min_context(self):
        rm_indexes = self.data.index.values[self.data[f'{HIST_EXP_COL}{self.sep}y'].isna().values.nonzero()]
        for phase in ['train', 'valid', 'test']:
            rm_phase = np.isin(self.split_ixs[phase], rm_indexes)
            self.split_ixs[phase] = np.array(self.split_ixs[phase])[~rm_phase].tolist()
            if rm_phase.sum() > 0:
                logging.info(f'{rm_phase.sum()} instances have been excluded from {phase}-set due to lack of context')

    def get_tfidf_emb(self, vocab_sz=20000, stopwords='english', lowercase=True, min_df=5, unique_corpus=True):
        """
        It returns a CSR matrix with the TF-IDF representation of each sentence in the corpus
        """
        p = r'\b\d+([:\.]\d+)?(\w+)?\b'
        corpus = self.mappers[EXP_COL].idx2item['raw'].apply(lambda s: re.sub(p, '', s))
        if not unique_corpus:
            exps = sorted(self.data.loc[self.split_ixs['train'] + self.split_ixs['valid'], EXP_COL].explode().tolist())
            corpus = corpus.loc[exps]
            exp_ixs = np.argwhere((np.array(exps[1:]) - np.array(exps[:-1]) > 0))[:, 0]
        corpus = corpus.tolist()
        tfidf_m = TfidfVectorizer(max_features=vocab_sz, stop_words=stopwords, lowercase=lowercase, min_df=min_df)
        tfidf_emb = tfidf_m.fit_transform(corpus)
        if not unique_corpus:
            tfidf_emb = tfidf_emb[exp_ixs, :]
        return tfidf_emb

    def get_feat_exp_links(self, suffix=''):
        f_path = os.path.join(DATA_PATHS[self.cfg.dataset], DATA_MODE, 'mappers', f'{EXP_COL}2{FEAT_COL}{suffix}.pkl')
        if not os.path.exists(f_path):
            logging.info('Extracting feat-exp links...')
            # ef_pairs = self.data[[FEAT_COL, EXP_COL]].explode([EXP_COL, FEAT_COL])
            # ef_pairs[FEAT_COL] = ef_pairs[FEAT_COL].apply(tuple)
            # dedup_pairs = set(map(tuple, ef_pairs.values.tolist()))
            # assert len(dedup_pairs) == ef_pairs[EXP_COL].nunique()
            # values, index = zip(*dedup_pairs)
            # e2f = pd.Series(values, index=index, name=FEAT_COL).apply(set).rename_axis(EXP_COL)
            e2f = self.data[[FEAT_COL, EXP_COL]].explode([EXP_COL, FEAT_COL]) \
                .explode(FEAT_COL) \
                .groupby(EXP_COL) \
                .agg(set)[FEAT_COL]
            if not self.cfg.test_flag:
                e2f.to_pickle(f_path)
        else:
            logging.info('Loading feat-exp links...')
            e2f = pd.read_pickle(f_path)
        f2e = e2f.explode().reset_index().groupby(FEAT_COL).agg(set)
        return f2e[EXP_COL], e2f

    def filter_hist_item_attr(self):
        """ Filter those user explanations that do not contain an item attribute """

        # NOTE: It takes ~3 mins on TripAdvisor
        def filter_user_exps(r):
            return [e for e in r[U_HIST_EXP_COL] if self.e2f[e].intersection(r.i_feats)]

        self.data['i_feats'] = self.data[I_COL].map(self.i2f)
        mask = self.data[U_HIST_EXP_COL].str.len() > 0
        self.data.loc[mask, U_HIST_EXP_COL] = self.data.loc[mask, ['i_feats', U_HIST_EXP_COL]].parallel_apply(
            filter_user_exps,
            axis=1)
        self.data.drop('i_feats', axis=1, inplace=True)

    def select_exp_nodes(self, max_exps=510, max_gt=40, min_u=40):
        max_hist_exps = pd.Series([max_exps] * len(self.data), index=self.data.index.values)
        max_hist_exps.loc[self.split_ixs['train']] -= self.data[EXP_COL].str.len().clip(0, max_gt)

        n_u_exps = self.data[U_HIST_EXP_COL].str.len()

        need_clip_mask = (self.data[I_HIST_EXP_COL].str.len() + self.data[U_HIST_EXP_COL].str.len()) > max_exps

        n_i_poss_exps = (max_hist_exps - np.minimum(min_u, n_u_exps)).loc[need_clip_mask]
        i_exps = self.data[I_HIST_EXP_COL].copy()
        i_exps.loc[need_clip_mask] = [es[-l:] for es, l in zip(i_exps.loc[need_clip_mask].tolist(), n_i_poss_exps)]

        n_u_poss_exps = (max_hist_exps - i_exps.str.len()).loc[need_clip_mask]
        u_exps = self.data[U_HIST_EXP_COL].copy()
        u_exps.loc[need_clip_mask] = [es[-l:] for es, l in zip(u_exps.loc[need_clip_mask].tolist(), n_u_poss_exps)]

        trn_exps = self.data.loc[self.split_ixs['train'], EXP_COL].apply(lambda l: l[:max_gt])
        u_exps.loc[self.split_ixs['train']] = (u_exps.loc[self.split_ixs['train']] + trn_exps).apply(
            lambda l: list(set(l)))
        i_exps.loc[self.split_ixs['train']] = (i_exps.loc[self.split_ixs['train']] + trn_exps).apply(
            lambda l: list(set(l)))
        exps = (u_exps + i_exps).apply(lambda l: list(set(l)))

        return u_exps, i_exps, exps

    def build_load_graphs(self, suffix=''):
        if getattr(self.cfg, 'filter_item_attr', True):
            suffix = f'_item_attr{suffix}'
        g_file = f'GREENer_graphs_{self.cfg.hist_len}{suffix}.pkl'

        if os.path.exists(os.path.join(self.aux_path, g_file)):
            logging.info('Loading HeteroGraph data...')
            g_data = pd.read_pickle(os.path.join(self.aux_path, g_file))
        else:
            feat_occurrence = self.f2e.apply(len)
            e2f_count = self.e2f.apply(len)

            logging.info(f'Feat. occurrence:\n\tMin: {feat_occurrence.min()} -- Mean: {feat_occurrence.mean():.4f}'
                         f' -- Max: {feat_occurrence.max()}')
            logging.info(f'Exp. Feat. Count:\n\tMin: {e2f_count.min()} -- Mean: {e2f_count.mean():.4f}'
                         f' -- Max: {e2f_count.max()}')

            # NOTE (P5 datasets): There are some explanations where its feature does not appear as a complete word in the text
            #  (stemmed). Some examples of these features are: "ar" (are), "od" (odd), "ab" (able/about),
            #  "pro" (protect/progression/probably/problem/provider), "ap" (app/application/approach/appears/apart).
            #  This stems OBVIOUSLY are wrong and lemmatization in SENTIRES toolkit may behave incorrectly. Steps to
            #  check it are commented out below:

            logging.info(f'Flattening columns...')
            self.flatten_cols(filter_unique=True, preserve_order=True)  # Global flattening of histories

            if getattr(self.cfg, 'filter_item_attr', False):
                logging.info(f'Filtering User Context to include only sentences containing current item attrs...')
                self.filter_hist_item_attr()

            logging.info('Selecting explanations...')
            self.data[U_HIST_EXP_COL], self.data[I_HIST_EXP_COL], self.data[HIST_EXP_COL] = self.select_exp_nodes()

            # logging.info('Computing BERT Labels...')

            # Merge user and item histories
            # self.data[HIST_FEAT_COL] = (self.data[U_HIST_FEAT_COL] + self.data[I_HIST_FEAT_COL]).apply(lambda l: sorted(list(set(l))))
            # self.data[HIST_EXP_COL] = (self.data[U_HIST_EXP_COL] + self.data[I_HIST_EXP_COL]).apply(lambda l: sorted(list(set(l))))

            # NOTE: Computing BERT labels takes ~15mins in SportsAndOutdoors (max_hist_len = 10)
            # NOTE: It takes too long for long histories (even days)
            # self.data.loc[:, 'BERT_y'] = [[]] * self.data.shape[0]
            # bert_labels = chunk_score(self.data, cand_c=HIST_EXP_COL, ref_c=EXP_COL, mapper=self.mappers[EXP_COL],
            #                           chunk_size=1e6, model_type='microsoft/deberta-base-mnli', batch_size=128,
            #                           idf=True, device=self.device, use_fast_tokenizer=True)
            # self.data.loc[bert_labels.index.values, 'BERT_y'] = bert_labels['y'].values

            # NOTE: Building Graphs takes ~20mins in SportsAndOutdoors (max_hist_len = 10)
            logging.info('Extracting and Saving GREENer Graphs...')
            tqdm.pandas()
            self.transforms = []  # ToUndirected() --> Performed directly in the model forward
            # self.data = self.data.head(115)
            g_data = self.data.parallel_apply(self.build_heterograph, axis=1)

            if not self.cfg.test_flag:
                pd.to_pickle(g_data, os.path.join(self.aux_path, g_file))

        return g_data

    def find_feat_exp(self, feat=None, exp=None):
        mask = np.ones(self.data.shape[0]).astype(np.bool)
        if feat is not None:
            if isinstance(feat, int):
                mask &= (self.data[FEAT_COL].str[0].values == feat)
            else:
                mask &= self.data[FEAT_COL].str[0].isin(feat).values

        if exp is not None:
            if isinstance(exp, int):
                mask &= (self.data[EXP_COL].str[0].values == exp)
            else:
                mask &= self.data[EXP_COL].str[0].isin(exp).values

        sub_data = self.data.loc[mask, [EXP_COL, FEAT_COL]]
        return list(zip(self.mappers[FEAT_COL].get_items(sub_data[FEAT_COL].str[0].values),
                        self.mappers[EXP_COL].get_items(sub_data[EXP_COL].str[0].values)))

    def build_heterograph(self, row):
        # logging.info('Building HeteroGraph (x, edge_index, y) fields...')
        # cols = [U_HIST_FEAT_COL, U_HIST_EXP_COL, I_HIST_FEAT_COL, I_HIST_EXP_COL, HIST_FEAT_COL, HIST_EXP_COL]
        # print(colored(f'Analizing row {row.name}: {row[cols].values}', Colors.BLUE))
        g = {}

        # NOTE: The order of x will be USER || ITEM || SENTS || FEATS
        # Get unique feats/exps and get enc_dict (All these columns have been transformed to set and back to list)
        u_exps = row[U_HIST_EXP_COL]
        i_exps = row[I_HIST_EXP_COL]
        exps = sorted(row[HIST_EXP_COL])

        if not exps:
            return None

        u_feats = list(set().union(*self.e2f.loc[u_exps].tolist()))
        i_feats = list(set().union(*self.e2f.loc[i_exps].tolist()))
        feats = sorted(list(set(u_feats + i_feats)))

        g[f'{U_COL}{self.sep}x'] = [row[U_COL]]
        g[f'{I_COL}{self.sep}x'] = [row[I_COL]]
        g[f'{FEAT_COL}{self.sep}x'] = feats
        g[f'{EXP_COL}{self.sep}x'] = exps

        feat_map = pd.Series(range(len(feats)), index=feats)
        # print(f'The feat. map is: {feat_map.to_dict()}')
        exp_map = pd.Series(range(len(exps)), index=exps)
        # print(f'The exp. map is: {exp_map.to_dict()}')

        # Build sf_adj (binarized edges)
        source_ixs, target_ixs = zip(*[[s, f] for f in feats for s in (set(exps) & self.f2e[f])])
        # print(f'From exp. {source_ixs} to feat. {target_ixs}')
        source_ixs = exp_map.loc[list(source_ixs)].tolist()
        target_ixs = feat_map.loc[list(target_ixs)].tolist()
        if isinstance(source_ixs, int):
            source_ixs, target_ixs = [source_ixs], [target_ixs]
        g[f'{EXP_COL}{self.sep}{FEAT_COL}{self.sep}edge_index'] = np.stack((source_ixs, target_ixs))

        # Build uf_adj
        source_ixs = [0] * len(u_feats)
        target_ixs = feat_map.loc[u_feats].tolist()
        # print(f'After map from user {source_ixs} to feat. {target_ixs}')
        g[f'{U_COL}{self.sep}{FEAT_COL}{self.sep}edge_index'] = np.stack((source_ixs, target_ixs))

        # Build if_adj
        source_ixs = [0] * len(i_feats)
        target_ixs = feat_map.loc[i_feats].tolist()
        # print(f'After map from item {source_ixs} to feat. {target_ixs}')
        g[f'{I_COL}{self.sep}{FEAT_COL}{self.sep}edge_index'] = np.stack((source_ixs, target_ixs))

        # Get sentence labels
        # NOTE: Hard labels make it difficult as exact sentences have rarely been written in the past
        hard_labels = np.isin(exps, row[EXP_COL]).tolist()
        gt = self.mappers[EXP_COL].get_items(row[EXP_COL])

        bleu_refs = [nltk.word_tokenize(s) for s in gt]
        bleu_cands = [nltk.word_tokenize(s) for s in self.mappers[EXP_COL].get_items(exps)]
        # bleu_labels = [max(compute_bleu([[ref]], [cand], 4)[0] for ref in bleu_refs) for cand in bleu_cands]
        bleu_labels = [sum(self.get_sentence_bleu(bleu_refs, cand, types=[2, 3])) for cand in bleu_cands]
        bleu_soft_labels = list(map(self.get_softlabel, bleu_labels))

        # bert_labels = row['BERT_y']

        # print(f'Labels: {list(zip(hard_labels, bleu_labels, bert_labels))}')
        g[f'{EXP_COL}{self.sep}y'] = list(zip(hard_labels, bleu_soft_labels, bleu_labels))  # , bert_labels))

        # Get feature labels
        g[f'{FEAT_COL}{self.sep}y'] = np.isin(feats, row[FEAT_COL])

        for transform in self.transforms:
            g = transform(g)

        return pd.Series(g, name=row.name)

    @staticmethod
    def get_sentence_bleu(references, hypotheses, types=[1, 2, 3, 4]):
        """ This is used to compute sentence-level bleu
        param: references: list of reference sentences, each reference sentence is a list of tokens
        param: hypoyheses: hypotheses sentences, this is a list of tokenized tokens
        return:
            bleu-1, bleu-2, bleu-3, bleu-4
        """
        type_weights = [[1.0, 0., 0., 0],
                        [0.5, 0.5, 0.0, 0.0],
                        [1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0],
                        [0.25, 0.25, 0.25, 0.25]]

        sf = bleu_score.SmoothingFunction()
        bleu_scores = []
        for type in types:
            bleu_scores.append(bleu_score.sentence_bleu(
                references, hypotheses, smoothing_function=sf.method1, weights=type_weights[type - 1]))
        return bleu_scores

    @staticmethod
    def get_softlabel(bleu_score):
        if bleu_score < 0.5:
            return 0
        elif bleu_score < 1.25:
            return 1
        elif bleu_score < 2:
            return 2

        return 3

    def to_pyg_heterograph(self, sample, data):
        g = HeteroData()
        fields = list(data.keys())
        for field in fields:
            if field.endswith(f"{self.sep}x"):
                node_type, _ = field.split(self.sep)
                g[node_type].x = data[field]
            elif field.endswith(f"{self.sep}edge_index"):
                source_node_type, target_node_type, _ = field.split(self.sep)
                g[source_node_type, target_node_type].edge_index = data[field]
            elif field.endswith(f"{self.sep}y"):
                node_type, _ = field.split(self.sep)
                g[node_type].y = data[field]
            else:
                continue
            data.pop(field, None)

        data['graph'] = g
        return data

    def collate(self, batch):
        batch = batch_list2dict(batch, self.cols['pad_mask'])

        batch['graph'] = Batch.from_data_list(batch['graph'])
        # batch['graph'], _, _ = collate(batch['graph'][0].__class__, data_list=batch['graph'])

        # for k in [f'gt_{EXP_COL}', f'gt_{FEAT_COL}']:
        #     lens = list(map(len, batch[k]))
        #     batch[f'{k}_bix'] = torch.repeat_interleave(torch.arange(len(batch[k])), torch.tensor(lens))
        #     batch[k] = torch.cat(batch[k])

        return batch


class ESCOFILTDataset(BaseDataset):
    """ Code adapted from original implementation at: https://github.com/reinaldncku/ESCOFILT/blob/main """

    def _finish_init(self):
        from src.models.extractive.escofilt import ModelProcessor

        self.ratios = {
            U_COL: getattr(self.cfg, f'{U_COL}_ratio', 0.4),
            I_COL: getattr(self.cfg, f'{I_COL}_ratio', 0.4)
        }
        model = getattr(self.cfg, 'summarizer_model', 'bert-large-uncased')
        hidden = getattr(self.cfg, 'summarizer_hidden', -2)
        reduce_opt = getattr(self.cfg, 'summarizer_reduce_opt', 'mean')

        summarizer = ModelProcessor(model, hidden, reduce_opt, random_state=self.cfg.seed)

        # The predicted sentences are only extracted from the item's previous explanations.
        self.data_info.user_emb = self.extract_sum_embs(summarizer, U_COL, self.data_info.n_users)
        self.i_top_sents, self.data_info.item_emb = self.extract_sum_embs(summarizer, I_COL, self.data_info.n_items,
                                                                          return_sents=True)

        self.getter_fns.append(self.get_ui_exp_pred)
        self.cols['float'].append(RAT_COL)
        super()._finish_init()

    def get_ui_exp_pred(self, sample, data):
        # NOTE: self.i_top_sents contains the closest sentences to each item centroids. Thus, selecting a random set of
        #  TOP-K explanations from each item's summary seems logical. (not present in original implementation)
        selected = self.i_top_sents[sample[I_COL][0]]
        if len(selected) > self.model_cfg.topk:
            selected = random.sample(selected, self.model_cfg.topk)
        # selected = self.i_top_sents.loc[sample[U_COL][0]][:self.model_cfg.topk]
        data[EXP_COL] = torch.LongTensor(selected)

    def filter_sentences(self, sents, min_len, max_len):
        return [s for s in sents if min_len < len(s) < max_len]

    def extract_sum_embs(self, summer, col, n, return_sents=False):
        ratio = self.ratios[col]
        topk = self.model_cfg.topk if return_sents else None
        f_prefix = f'ESCOFILT_{col}_{str(int(ratio * 100))}_{summer.name.replace("/", "%")}_{self.cfg.seed}'
        emb_path = os.path.join(self.aux_path, f'{f_prefix}_embeddings.npy')
        sent_path = os.path.join(self.aux_path, f'{f_prefix}_sentences.json')
        # ixs_path = os.path.join(self.aux_path, f'{f_prefix}_ixs.npy')
        selected = {}
        if os.path.isfile(emb_path):
            agg_embeddings = np.load(emb_path)
            # ixs = np.load(ixs_path)
            if return_sents:
                selected = pd.read_json(sent_path, typ='series').to_dict()
        else:
            # ixs = []
            unk_entries = np.zeros((n,), dtype=bool)
            agg_embeddings = None

            grouped_df = self.data.loc[self.split_ixs['train']].groupby(col)
            for cid, cdata in tqdm(grouped_df):
                # body = ' '.join(self.mappers[EXP_COL].get_items(cdata[EXP_COL].explode().tolist()))
                # body = ' '.join(cdata[REV_COL].str.lower().tolist())
                # sents = None
                # top_sents = []

                # NOTE: E088 error is raised as text length exceeds max_length (1.000.000) --> Passing sents directly
                # if len(body) > 1000000:
                # NOTE: ConvergenceWarning warns about duplicate points in X --> passing unique points
                body = self.mappers[EXP_COL].get_items(cdata[EXP_COL].explode().unique().tolist())
                sents = self.filter_sentences([s.strip() for s in body], 10, 800)

                try:
                    # NOTE: max_sents is necessary for aggregated users as otherwise clustering takes too long
                    top_sents, ci_emb = summer.run_embeddings(body, sents, ratio=ratio, aggregate='mean',
                                                              min_length=10, max_length=800, min_sents=topk,
                                                              max_sents=5000)

                    if ci_emb is None:
                        logging.info("Init. NaN ", cid, " <<<<< ")
                        if len(body) > 1000000:
                            sents = self.filter_sentences([s.strip() for s in body], 10, 1900)
                        top_sents, ci_emb = summer.run_embeddings(body, sents, ratio=ratio, aggregate='mean',
                                                                  min_length=10, max_length=1900, min_sents=topk,
                                                                  max_sents=5000)

                        if ci_emb is None:
                            unk_entries[cid] = True
                            logging.warning("Still, NaN-affected ID: ", cid, " <<<<< ")

                    top_sents = self.mappers[EXP_COL].get_idxs(top_sents)
                    # ixs.append(cid)
                except Exception as e:
                    logging.error(e)
                    logging.error("Offending ID (via Exception): ", cid)
                    logging.error("Bye-bye for now!")
                    sys.exit()

                if ci_emb is not None:
                    if agg_embeddings is None:
                        agg_embeddings = np.zeros((n, ci_emb.shape[-1]))
                    agg_embeddings[cid] += ci_emb

                # Fill missing sentences with empty explanation
                if return_sents:
                    n_sents = len(top_sents)
                    if len(sents) > topk:
                        assert len(top_sents) >= topk
                    top_sents = list(set(top_sents))
                    # If non-unique sentences were retrieved, fill with random remaining item/user sentences
                    if n_sents > len(top_sents):
                        # Select random sentences from the item explanations that weren't selected before
                        remain_sents = list(set(cdata[EXP_COL].explode().tolist()).difference(top_sents))
                        if remain_sents and len(top_sents) < topk:
                            top_sents += random.sample(remain_sents, topk - len(top_sents))
                    # Fill with empty explanation
                    top_sents += [self.mappers[EXP_COL].get_idx('')] * (topk - len(top_sents))
                    selected[cid] = top_sents

            # NOTE: Modify ESCOFILT default behavior for those users/items that do not get a valid embedding
            if sum(unk_entries) > 0:
                logging.warning("Setting None entries to the average embedding...")
                agg_embeddings[unk_entries] += agg_embeddings[~unk_entries].mean(0)

            np.save(emb_path, agg_embeddings)
            # np.save(ixs_path, np.array(ixs))
            if return_sents:
                pd.Series(selected).to_json(sent_path)

        if not hasattr(self, 'precomputed_emb_dim'):
            self.precomputed_emb_dim = {}
        self.precomputed_emb_dim[col] = agg_embeddings.shape[-1]
        if return_sents:
            # Filter top-k
            # for cid in selected:
            #     selected[cid] = selected[cid][:self.model_cfg.topk]
            return selected, agg_embeddings

        return agg_embeddings


class ERRADataset(GenDataset):
    def _finish_init(self):
        self.get_aspect_data()

        self.meta_path = os.path.join(self.aux_path, 'ERRA_supportVecs_and_aspects.pkl')

        self.n_aspects = getattr(self.cfg, 'retrieval_n_aspects', 2)

        valid_ixs = self.split_ixs['train'] + self.split_ixs['valid']
        retrieval_args = {'ui_aspect_revs': None, 'sent_embs': None}
        if not os.path.isfile(self.meta_path):
            retrieval_args['ui_aspect_revs'] = self.data.loc[valid_ixs, [U_COL, I_COL, ASP_COL, REV_COL]]
            self.u2a = self.data.loc[valid_ixs, [U_COL, ASP_COL]].explode(ASP_COL).explode(ASP_COL).drop_duplicates()
            self.u2a = self.u2a.groupby(U_COL)[ASP_COL].agg(list)

        self.cols['enc'][EXP_COL] = EXP_COL
        self.cols['enc'][ASP_COL] = ASP_COL
        super()._finish_init()

        self.getter_fns.append(self.get_support_and_aspects)  # self.getter_fns
        # self.cols['processed'].extend([ASP_COL, 'user_support', 'item_support'])

        if not os.path.isfile(self.meta_path):
            trn_val_exps = sorted(self.data.loc[valid_ixs, EXP_COL].explode().unique().tolist())
            retrieval_args['sent_embs'] = pd.Series(self.mappers[EXP_COL].get_emb(trn_val_exps), index=trn_val_exps)

        self.s_uv, self.s_vu, self.a_uv = self.retrieval(**retrieval_args)

        # self.data_info.lengths[ASP_COL] = self.mappers[ASP_COL].enc_len() * self.n_aspects
        self.data_info.lengths[FEAT_COL] = self.mappers[ASP_COL].enc_len() * self.n_aspects

    def get_aspect_data(self):
        # TODO: Extract aspects column. For now, the aspects will only be the features
        self.data[ASP_COL] = self.data[FEAT_COL]
        self.mappers[ASP_COL] = copy.deepcopy(self.mappers[FEAT_COL])

    def get_support_and_aspects(self, sample, data):
        ui_aspects = self.a_uv[f'{sample[U_COL][0]}_{sample[I_COL][0]}']
        data[ASP_COL] = torch.LongTensor(sum(self.mappers[ASP_COL].get_enc(ui_aspects), start=[]))
        data[f'{U_COL}_sv'] = torch.FloatTensor(self.s_uv.loc[sample[U_COL][0]])
        data[f'{I_COL}_sv'] = torch.FloatTensor(self.s_vu.loc[sample[I_COL][0]])

    def retrieval(self, ui_aspect_revs, sent_embs):
        if os.path.isfile(self.meta_path):
            u_support, i_support, ui2a = pd.read_pickle(self.meta_path)
        else:
            # Initialize Sentence Transformer
            # NOTE: In the paper, authors mention they use SentenceBERT. However, sentence-transformers/stsb-bert-base
            #  is deprecated due to low quality sentence embeddings and, also, they actually use a model with 384-dim.
            #  The actual Sentence Transformer checkpoint used for the experiments is not specified.
            def_emb_cfg = SimpleNamespace(name='all-MiniLM-L6-v2', type='huggingface', batch_size=128)
            emb_cfg = getattr(self.cfg, 'emb_models', [def_emb_cfg])[0]
            if not hasattr(emb_cfg, 'batch_size'):
                emb_cfg.batch_size = def_emb_cfg.batch_size
            emb_model = self.get_emb_model(emb_cfg)

            # Retrieve n support sentences per user/item
            u_exp_query = self.get_exp_query_embs(emb_model, emb_cfg, U_COL, ui_aspect_revs)
            i_exp_query = self.get_exp_query_embs(emb_model, emb_cfg, I_COL, ui_aspect_revs)

            n = getattr(self.cfg, 'retrieval_n_exps', 3)
            u_support, i_support = self.get_ui_support_exps(u_exp_query, i_exp_query, np.array(sent_embs.tolist()),
                                                            sent_embs.index.values, n)
            # Obtain Sentence-Transformer embeddings for support sentences
            u_support[:] = emb_model.encode(u_support.tolist(), emb_cfg.batch_size, device=self.device,
                                            show_progress_bar=False).tolist()
            i_support[:] = emb_model.encode(i_support.tolist(), emb_cfg.batch_size, device=self.device,
                                            show_progress_bar=False).tolist()

            # Extract Item aspect query vectors
            self.mappers[ASP_COL].embed(emb_model, batch_size=emb_cfg.batch_size)
            i_aspect_query = self.get_asp_query_embs(I_COL, ui_aspect_revs)

            # Retrieve user-interest aspects per product interaction
            ui_pairs = self.data[[U_COL, I_COL]].explode([U_COL, I_COL]).drop_duplicates()
            ui2a = ui_pairs.apply(partial(self.select_aspects, query_embs=i_aspect_query), axis=1)
            ui2a = pd.Series(ui2a.tolist(), index=(ui_pairs[U_COL].map(str) + '_' + ui_pairs[I_COL].map(str)).tolist())

            if not self.cfg.test_flag:
                pd.to_pickle([u_support, i_support, ui2a], self.meta_path)

        return u_support, i_support, ui2a

    def select_aspects(self, row: pd.Series, query_embs: pd.Series):
        cand_asps = self.u2a[row[U_COL]]
        n = getattr(self.cfg, 'retrieval_n_aspects', 2)
        if len(cand_asps) > n:
            cand_asps_embs = np.array(self.mappers[ASP_COL].get_emb(cand_asps))
            query = query_embs.loc[row[I_COL]]
            cos_sim = (query @ cand_asps_embs.T) / (np.linalg.norm(cand_asps_embs) * np.linalg.norm(query))
            # NOTE: Default value set to 2 (derived from pre-computed filename in original code: "cell_aspect_top2.pt")
            return np.array(cand_asps)[np.argpartition(-cos_sim, n)[:n]].tolist()
        return cand_asps

    def get_ui_support_exps(self, u_embs, i_embs, sent_embs, sent_ixs, n):
        import faiss

        # Build the faiss index for Aproximate Nearest Neighbor
        dim = u_embs.iloc[0].shape[-1]
        if getattr(self.cfg, 'use_ann', True):  # sent_embs.shape[0] > 1_000_000:
            cells = getattr(self.cfg, 'ann_cells', 500)
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, cells, faiss.METRIC_L2)
            index.train(sent_embs)
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(sent_embs)

        _, u_sent_ixs = index.search(np.array(u_embs.tolist()), n)
        _, i_sent_ixs = index.search(np.array(i_embs.tolist()), n)

        u_sent_ixs = sent_ixs[u_sent_ixs]
        i_sent_ixs = sent_ixs[i_sent_ixs]

        u_sents = list(map(lambda l: ' '.join(self.mappers[EXP_COL].get_items(l)), u_sent_ixs))
        i_sents = list(map(lambda l: ' '.join(self.mappers[EXP_COL].get_items(l)), i_sent_ixs))

        return pd.Series(u_sents, index=u_embs.index.values), pd.Series(i_sents, index=i_embs.index.values)

    def get_exp_query_embs(self, model: SentenceTransformer, model_cfg: SimpleNamespace, groubpy_c: str,
                           rev_df: pd.DataFrame) -> pd.Series:
        grouped_revs = rev_df[[groubpy_c, REV_COL]].groupby(groubpy_c)[REV_COL]
        return grouped_revs.agg(lambda rs: model.encode(rs.tolist(), model_cfg.batch_size, device=self.device,
                                                        show_progress_bar=False).mean(axis=0))

    def get_asp_query_embs(self, groubpy_c: str, asp_df: pd.DataFrame) -> pd.Series:
        asp_df = asp_df[[groubpy_c, ASP_COL]].explode(ASP_COL).explode(ASP_COL).drop_duplicates()
        grouped_asps = asp_df.groupby(groubpy_c)[ASP_COL]
        return grouped_asps.agg(lambda asp: np.array(self.mappers[ASP_COL].get_emb(asp.tolist())).mean(axis=0))


class ExBERTDataset(SeqDataset, GenDataset):
    def _finish_init(self):
        # NOTE: It would make more sense to do it in the data loading process (that is, with prob. p, get negative sample)
        # Augment the training dataset with negative samples
        self.add_special_toks = True

        self.data['nsp'] = 1
        self.augment_dataset()
        self.profile_words = min(getattr(self.cfg, 'profile_words', 15), self.cfg.txt_len - 2)
        self.cols['emb'] = {}

        profile_sampler = getattr(self.cfg, 'profile_sampler', 'temporal_rand')
        if profile_sampler == 'temporal_rand':
            self.profile_sampler = partial(self.temporal_rand, level=getattr(self.cfg, 'profile_sampler_level', 'interaction'))
        elif profile_sampler == 'sbert_gt':
            self.profile_sampler = self.sbert_gt
            self.cols['emb'][EXP_COL] = EXP_COL

        self.getter_fns += [partial(self.get_profile, col=U_HIST_EXP_COL),
                            partial(self.get_profile, col=I_HIST_EXP_COL)]
            # if hasattr(self.cfg, 'emb_models'):
        #     self.getter_fns += [partial(self.get_profile_sbert, col=U_HIST_EXP_COL),
        #                         partial(self.get_profile_sbert, col=I_HIST_EXP_COL)]
        #     self.cols['emb'][EXP_COL] = EXP_COL
        # else:
        #     self.profile_sampler = partial(self.temporal_rand, level='interaction')
        #     self.getter_fns += [partial(self.get_profile, col=U_HIST_EXP_COL),
        #                         partial(self.get_profile, col=I_HIST_EXP_COL)]

        self.cols['orig'].append('nsp')

        super()._finish_init()

        if profile_sampler != 'temporal_rand':
            self.cols['emb'].pop(EXP_COL)
        pop_keys = [c for c in self.cols['enc'].keys() if HIST_PREFIX in c]
        for c in pop_keys:
            self.cols['enc'].pop(c)
        self.empty_exp = self.mappers[EXP_COL].get_enc(self.mappers[EXP_COL].get_idx(''))
        self.empty_exp += [self.tok.pad_token_id] * (self.profile_words + 2 - len(self.empty_exp))
        self.empty_exp = torch.LongTensor(self.empty_exp)

    def truncate_torch_exp(self, e):
        # The explanation starts with BOS and ends with EOS. We need to keep the first self.profile_words tokens in between.
        n_pad = self.profile_words + 2 - len(e)
        if n_pad <= 0:
            # Truncate
            return torch.LongTensor(e[:self.profile_words + 1] + e[-1:])
        # Pad
        return torch.LongTensor(e + [self.tok.pad_token_id] * n_pad)

    def temporal_rand(self, sample, col, level='interaction'):
        hist_exps = sample[col]
        if level == 'interaction':
            # Option 1: Random at the interaction-level from recent past
            return list(map(random.choice, hist_exps))
        elif level == 'global':
            # Option 2: Random from the flattened recent past
            flat_hist = list(set(sum(hist_exps, start=[])))
            if len(flat_hist) > self.cfg.hist_len:
                return random.choice(flat_hist)
            return flat_hist
        else:
            raise ValueError(f'Invalid value for "level". It can be one of {{"global", "interaction"}}')

    def temporal_div(self, sample, col, level='interaction'):
        raise NotImplementedError
        # hist_exps = sample[col]
        # if level == 'interaction':
        #     if len(hist_exps) > 1:
        #         pass
        #     # import pyomo.environ as pe
        #     # import pyomo.opt as po
        #     # hist_embs = list(map(self.mappers[EXP_COL].get_emb, hist_exps))
        #     # ILP_m = pe.ConcreteModel()
        #     # solver = po.SolverFactory('gurobi', solver_io="python")
        #     return [random.choice(hist_exps[0])]
        # elif level == 'global':
        #     # Option 2: Random from the flattened recent past
        #     flat_hist = list(set(sum(hist_exps, start=[])))
        #     if len(flat_hist) > self.cfg.hist_len:
        #         return random.choice(flat_hist)
        #     return flat_hist
        # else:
        #     raise ValueError(f'Invalid value for "level". It can be one of {{"global", "interaction"}}')

    def sbert_gt(self, sample, col):
        hist_exps = list(set(sum(sample[col], start=[])))
        if len(hist_exps) > self.cfg.hist_len:
            hist_exps = np.array(hist_exps)
            hist_embs = np.array(self.mappers[EXP_COL].get_emb(hist_exps))
            gt_emb = np.array(self.mappers[EXP_COL].get_emb([sample[EXP_COL]])[0])
            # NOTE: The paper does not mention which similarity score is being used to compute the profile
            sims = (hist_embs @ gt_emb.T) / (np.linalg.norm(hist_embs, axis=1) * np.linalg.norm(gt_emb))
            return hist_exps[np.argsort(sims)[::-1][:self.cfg.hist_len]].tolist()
        return hist_exps

    def get_profile(self, sample, data, col):
        # NOTE: Profile Length is handled in read_data() using the hist_len argument of the config file
        # The user profile has shape: [BSZ, HIST_LEN, TXT_LEN] --> Batch size not considered in this step
        # Sample exps. from col history
        # hist_exps = sample[col]
        # hist_exps = list(map(random.choice, hist_exps))
        hist_exps = self.profile_sampler(sample, col)

        # Get Encodings (truncated to the max. length specified by profile_words)
        hist_exps = list(map(self.truncate_torch_exp, self.mappers[EXP_COL].get_enc(hist_exps)))

        # Pad the sequence with empty explanations
        hist_exps += [self.empty_exp] * (self.cfg.hist_len - len(hist_exps))

        # Pad all explanations to the maximum length
        data[f'{col.split("_")[0]}_profile'] = torch.stack(hist_exps)

        return data

    def get_profile_sbert(self, sample, data, col):
        # Unlike get_profile(), this method assumes we know the actual groundtruth in order to select proper ui profile
        hist_exps = list(set(sum(sample[col], start=[])))
        if len(hist_exps) > self.cfg.hist_len:
            hist_exps = np.array(hist_exps)
            hist_embs = np.array(self.mappers[EXP_COL].get_emb(hist_exps))
            gt_emb = np.array(self.mappers[EXP_COL].get_emb([sample[EXP_COL]])[0])
            # NOTE: The paper does not mention which similarity score is being used to compute the profile
            sims = (hist_embs @ gt_emb.T) / (np.linalg.norm(hist_embs, axis=1) * np.linalg.norm(gt_emb))
            hist_exps = hist_exps[np.argsort(sims)[::-1][:self.cfg.hist_len]].tolist()

        # Get Encodings (truncated to the max. length specified by profile_words)
        hist_exps = list(map(self.truncate_torch_exp, self.mappers[EXP_COL].get_enc(hist_exps)))

        # Pad the sequence with empty explanations
        hist_exps += [self.empty_exp] * (self.cfg.hist_len - len(hist_exps))

        # Pad all explanations to the maximum length
        data[f'{col.split("_")[0]}_profile'] = torch.stack(hist_exps)

        return data

    def augment_dataset(self):
        neg_rate = getattr(self.cfg, 'neg_rate', 0.5)
        aux_file = os.path.join(self.aux_path, f'ExBERT_NegativeReviews_{str(neg_rate).replace(".", ",")}.pkl')
        if os.path.isfile(aux_file):
            logging.info('Loading Negative Reviews...')
            neg_samples = pd.read_pickle(aux_file)
            assert len(set(self.split_ixs['train']).intersection(neg_samples.index.tolist())) == 0
        else:
            logging.info('Computing Negative Reviews...')
            trn_ixs = self.split_ixs['train']
            replace_cols = [FEAT_COL, EXP_COL]
            neg_samples = []
            for i, row in tqdm(self.data.loc[trn_ixs].iterrows(), total=len(trn_ixs)):
                if random.random() < neg_rate:
                    neg_row = row.copy(deep=True)
                    # NOTE: They select a negative review from the set of all reviews. However, it may be even better to
                    #  select it from the user and/or item profiles
                    neg_i = random.choice(list(set(trn_ixs).difference([i])))
                    neg_row[replace_cols] = self.data.loc[neg_i, replace_cols]
                    neg_samples.append(neg_row)
            neg_samples = pd.concat(neg_samples, axis=1).T
            neg_samples['nsp'] = 0
            neg_samples.index += self.data.index.values.max() + 1
            logging.info('Saving Negative Reviews...')
            if not self.cfg.test_flag:
                neg_samples.to_pickle(aux_file)

        self.data = pd.concat([self.data, neg_samples], axis=0)
        self.split_ixs['train'] += neg_samples.index.tolist()
