import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import random
import shutil
import sys
import re
from functools import partial

import logging
import nltk
# stanza.download('en')

from nltk.translate import bleu_score

import pandas as pd
import numpy as np

from tqdm import tqdm
tqdm.pandas()

from src.utils.constants import *
from src.utils.in_out import load_partition
from src.utils.data import Mapper, BaseDataset

# Download Amazon datasets: wget https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/{Category}.json.gz
# Download Amazon metadata: https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz
# from utils.data import get_tokenizer
from src.utils.funcs import init_seed


def iterative_kcore_filter(df, kcore, verbose=1):
    def kcore_filter(d, col):
        return d[d[col].map(d[col].value_counts()).ge(kcore)]
    copy_df = df.copy()
    prev_sz = -1
    if verbose == 1:
        logging.info(f'Starting Iterative K-Core item and user filtering with K = {kcore}...')
        logging.info(f'Initial number of interactions: {df.shape[0]}')
        logging.info(f'Initial number of users: {df[U_COL].nunique()}')
        logging.info(f'Initial number of items: {df[I_COL].nunique()}')
    while prev_sz != copy_df.shape[0]:
        # Filter by user profile size
        prev_sz = copy_df.shape[0]
        copy_df = kcore_filter(copy_df, U_COL)
        copy_df = kcore_filter(copy_df, I_COL)

    if verbose == 1:
        logging.info(f'Final number of interactions: {copy_df.shape[0]}')
        logging.info(f'Final number of users: {copy_df[U_COL].nunique()}')
        logging.info(f'Final number of items: {copy_df[I_COL].nunique()}')
    return copy_df


def split_data(datasets, test_ratio=0.1, mode='feat_only', f_name='reviews.pkl', split='0'):
    val_ratio = test_ratio

    for dataset in datasets:
        logging.info(f'Splitting dataset {dataset}')
        data = pd.read_pickle(os.path.join(DATA_PATHS[dataset], mode, f_name))
        if isinstance(data, (list, tuple)):
            data = pd.DataFrame.from_records(data)

        data = data.sort_values(by=[U_COL, TIMESTEP_COL])
        assert (data.index.values == list(range(data.shape[0]))).all()

        ucounts = data.groupby(U_COL).size().values
        logging.info(f'Min. user count: {ucounts.min()}. Max. user count: {ucounts.max()}')
        # Users with 2 interactions or less go to the training set
        mask = (ucounts > 2).astype(int)
        uoffsets = ucounts.cumsum()
        split_ixs = np.zeros((data.shape[0], ), dtype=int)
        if isinstance(test_ratio, float):
            assert isinstance(val_ratio, float)
            assert test_ratio < 1.0
            tst_start_ixs = uoffsets - np.maximum(ucounts * test_ratio, mask).astype(int)
            val_start_ixs = tst_start_ixs - np.maximum(ucounts * val_ratio, mask).astype(int)
        elif isinstance(test_ratio, int):
            assert isinstance(val_ratio, int)
            assert all(ucounts > (test_ratio + val_ratio))
            tst_start_ixs = uoffsets - test_ratio
            val_start_ixs = tst_start_ixs - val_ratio
        else:
            raise TypeError('test_ratio is neither int nor float')
        for vix, tix, offset in zip(val_start_ixs, tst_start_ixs, uoffsets):
            split_ixs[tix:offset] = 2
            split_ixs[vix:tix] = 1

        if not os.path.exists(os.path.join(DATA_PATHS[dataset], mode, split)):
            os.makedirs(os.path.join(DATA_PATHS[dataset], mode, split))
        else:
            shutil.rmtree(os.path.join(DATA_PATHS[dataset], mode, split))
            os.mkdir(os.path.join(DATA_PATHS[dataset], mode, split))

        trn_ixs = data.iloc[np.argwhere(split_ixs == 0).squeeze()].index.values
        val_ixs = data.iloc[np.argwhere(split_ixs == 1).squeeze()].index.values
        tst_ixs = data.iloc[np.argwhere(split_ixs == 2).squeeze()].index.values
        assert (trn_ixs == np.argwhere(split_ixs == 0).squeeze()).all()
        assert (val_ixs == np.argwhere(split_ixs == 1).squeeze()).all()
        assert (tst_ixs == np.argwhere(split_ixs == 2).squeeze()).all()

        trn = data.loc[trn_ixs]
        val = data.loc[val_ixs]
        tst = data.loc[tst_ixs]
        trn_u = set(trn.user.unique())
        val_u = set(val.user.unique())
        tst_u = set(tst.user.unique())
        assert len(val_u.difference(trn_u)) == 0
        assert len(tst_u.difference(val_u)) == 0
        assert len(tst_u.difference(trn_u)) == 0
        if mode != 'feat_only':
            assert len(trn_u.difference(val_u)) == 0
            assert len(trn_u.difference(tst_u)) == 0
        trn_i = trn.item.unique()
        val_i = val.item.unique()
        tst_i = tst.item.unique()
        val_i_diff = set(val_i).difference(trn_i)
        tst_i_diff = set(tst_i).difference(trn_i)
        logging.info(f"Number of items in valid but not in train: {len(val_i_diff)}")
        logging.info(f"Number of items in test but not in train: {len(tst_i_diff)}")
        logging.info(f"Percentage of missing item samples (Val): {sum(val.item.isin(val_i_diff)) / val.shape[0]:.4f}")
        logging.info(f"Percentage of missing item samples (Test): {sum(tst.item.isin(tst_i_diff)) / tst.shape[0]:.4f}")
        logging.info(f"Ratio of non-empty feature in training: {sum(trn[FEAT_COL] != '') / trn.shape[0]:.4f}")
        logging.info(f"Ratio of non-empty feature in valid: {sum(val[FEAT_COL] != '') / val.shape[0]:.4f}")
        logging.info(f"Ratio of non-empty feature in test: {sum(tst[FEAT_COL] != '') / tst.shape[0]:.4f}")
        logging.info(f'Final ratios for dataset {dataset} are: {np.unique(split_ixs, return_counts=True)[1] / len(split_ixs)}')

        # Assert that train, valid, and test sets were properly created based on timestep
        trn_latest_steps = trn[[U_COL, TIMESTEP_COL]].groupby(U_COL)[TIMESTEP_COL].agg('max')
        val_earliest_steps = val[[U_COL, TIMESTEP_COL]].groupby(U_COL)[TIMESTEP_COL].agg('min')
        val_latest_steps = val[[U_COL, TIMESTEP_COL]].groupby(U_COL)[TIMESTEP_COL].agg('max')
        tst_earliest_steps = tst[[U_COL, TIMESTEP_COL]].groupby(U_COL)[TIMESTEP_COL].agg('min')

        phase_steps = trn_latest_steps.to_frame('train_latest')
        phase_steps['valid_earliest'] = phase_steps['train_latest'].tolist()
        phase_steps.loc[val_earliest_steps.index.values, 'valid_earliest'] = val_earliest_steps.tolist()
        phase_steps['valid_latest'] = phase_steps['valid_earliest'].tolist()
        phase_steps.loc[val_latest_steps.index.values, 'valid_latest'] = val_latest_steps.tolist()
        phase_steps['test_earliest'] = phase_steps['valid_latest'].tolist()
        phase_steps.loc[tst_earliest_steps.index.values, 'test_earliest'] = tst_earliest_steps.tolist()

        assert ((phase_steps['valid_earliest'] - phase_steps['train_latest']) > 0).all()
        assert ((phase_steps['valid_latest'] - phase_steps['valid_earliest']) > 0).all()
        assert ((phase_steps['test_earliest'] - phase_steps['valid_latest']) > 0).all()

        np.save(os.path.join(DATA_PATHS[dataset], mode, split, 'train'), trn_ixs)
        np.save(os.path.join(DATA_PATHS[dataset], mode, split, 'validation'), val_ixs)
        np.save(os.path.join(DATA_PATHS[dataset], mode, split, 'test'), tst_ixs)

    logging.info('Finished!')


def encode(dataset, data, cols, suffix='', save=True, mode=DATA_MODE):
    mapping = {c: None for c in cols}
    new_data = data.copy()

    if not os.path.isdir(os.path.join(DATA_PATHS[dataset], mode, 'mappers')):
        os.makedirs(os.path.join(DATA_PATHS[dataset], mode, 'mappers'))

    for c in mapping.keys():
        logging.info(f'Encoding column: {c}')
        vals = new_data[c].explode()
        if isinstance(vals.iloc[0], list):
            logging.info(f'Column: {c} contains nested lists')
            vals.index = pd.MultiIndex.from_tuples(list(zip(vals.index.tolist(), range(len(vals)))))
            vals = vals.explode()
        if isinstance(vals.iloc[0], str):
            vals = vals.apply(lambda t: re.sub('\\x00', '', t))
        unique_vals = vals.unique().tolist()
        rev_mapping = pd.Series(range(len(unique_vals)), index=unique_vals)
        vals = vals.map(rev_mapping, na_action='ignore')
        assert not vals.isna().any()
        # if isinstance(new_data[c].iloc[0], list):
        if vals.index.get_level_values(0).value_counts().nunique() > 1:
            levels = list(range(vals.index.nlevels))
            logging.info(f'Column: {c} has {len(levels)}-level multi-index')
            while levels:
                vals = vals.groupby(level=levels).agg(list)
                levels = levels[:-1]
        new_data[c] = vals
        if save:
            mapping = pd.Series(rev_mapping.index.values, index=rev_mapping)
            mapping.to_pickle(os.path.join(DATA_PATHS[dataset], mode, 'mappers', f'{c}{suffix}.pkl'))

    return new_data


def iterative_kcore_filter_revs(df, min_revs=5, kcore=5, subset=[U_COL], verbose=1):
    """ Filter users with less than min_revs unique reviews """
    def kcore_rev_filter(d, col):
        dedup_data = d.drop_duplicates(subset=[c, REV_COL])
        dedup_counts = dedup_data[c].value_counts()
        return d[d[col].map(dedup_counts) >= min_revs]

    # logging.info(colored(f'Filtering out {subset} with less than {kcore} unique reviews', Colors.BLUE))
    prev_sz = -1
    copy_df = df.copy()

    while prev_sz != copy_df.shape[0]:
        prev_sz = copy_df.shape[0]
        for c in subset:
            copy_df = kcore_rev_filter(copy_df, c)
        copy_df = iterative_kcore_filter(copy_df, kcore, verbose=0)

    if verbose:
        logging.info(f'Started with {df.shape[0]} samples and ended with {copy_df.shape[0]} samples')
        logging.info(f'Started with {df[U_COL].nunique()} users and ended with {copy_df[U_COL].nunique()} users')
        logging.info(f'Started with {df[I_COL].nunique()} items and ended with {copy_df[I_COL].nunique()} items')
        logging.info(f'Started with {df[REV_COL].nunique()} unique revs and ended with {copy_df[REV_COL].nunique()} unique revs')

    return copy_df


def sliding_w_hist(g: pd.DataFrame, max_l: int = MAX_HIST_LEN):
    return [g.index.values[max(i - max_l, 0):i].tolist() for i in range(g.shape[0])]


def process_xrec_datasets(test=False):
    from src.utils.in_out import colored, Colors

    def sample(weights, max_samples=600):
        if len(weights) > max_samples:
            p = (weights / weights.sum()).values
            ixs = np.random.choice(weights.index.values, max_samples, p=p, replace=False)
            return ixs.tolist()
        return weights.index.tolist()

    datasets = [
        ('tripadvisor', 4, 5, False, None, None),
        ('yelp', 4, 5, False, None, None),
        ('ratebeer', 20, 20, True, 6, 700)
    ]
    suffix = TEST_SUFFIX if test else ''
    f_name = f'reviews{suffix}.pkl'
    split = '0' + (TEST_SUFFIX if test else '')
    for dataset, min_revs, kcore, dedup_uir, min_exp_toks, max_u_samples in datasets:
        if not os.path.exists(os.path.join(DATA_PATHS[dataset], DATA_MODE)):
            os.makedirs(os.path.join(DATA_PATHS[dataset], DATA_MODE))
            logging.info(colored(f'{dataset} dataset does not contain the reviews.pkl file needed under '
                                 f'{os.path.join(DATA_PATHS[dataset], DATA_MODE)}', Colors.RED))
            continue

        logging.info(f'Reading {dataset} dataset...')
        if os.path.isfile(os.path.join(DATA_PATHS[dataset], DATA_MODE, f'old_{f_name}')):
            data = pd.read_pickle(os.path.join(DATA_PATHS[dataset], DATA_MODE, f'old_{f_name}'))
            logging.info(colored(f'WARNING: Emptying {split} folder as the dataset is being recreated', Colors.YELLOW))
        else:
            data = pd.read_pickle(os.path.join(DATA_PATHS[dataset], DATA_MODE, f_name))
            data.to_pickle(os.path.join(DATA_PATHS[dataset], DATA_MODE, f'old_{f_name}'))
        # data = pd.DataFrame.from_records(pd.read_pickle(os.path.join(DATA_PATHS[dataset], mode, f_name)))

        # Fix column format
        rat_cols = [c for c in data.columns if RAT_COL in c]
        data[rat_cols] = data[rat_cols].astype(float)
        data[TIMESTAMP_COL] = pd.to_datetime(data[TIMESTAMP_COL])

        if dedup_uir:
            logging.info('Deduplicating user-item interactions and user reviews...')
            data = data.sort_values(TIMESTAMP_COL).reset_index(drop=True)
            data = data.drop_duplicates(subset=[U_COL, I_COL], keep='first')
            data = data.drop_duplicates(subset=[U_COL, REV_COL], keep='first')

        logging.info('Iterative k-core filtering...')
        data = iterative_kcore_filter_revs(data, min_revs=min_revs, kcore=kcore, subset=[U_COL, I_COL])

        if min_exp_toks is not None:
            logging.info(f'Filtering out explanation sentences with less than {min_exp_toks} tokens...')
            data = data.explode([FEAT_COL, EXP_COL])
            data = data[data[EXP_COL].str.split().str.len() >= min_exp_toks]
            groupby_cols = {c: list for c in [FEAT_COL, EXP_COL]}
            dedup_cols = {c: 'first' for c in data.columns if c not in groupby_cols}
            data = data.groupby(level=0).agg({**dedup_cols, **groupby_cols})
            data = iterative_kcore_filter_revs(data, min_revs=min_revs, kcore=kcore, subset=[U_COL, I_COL])

        if max_u_samples is not None:
            logging.info(f'Filtering out users exceeding {max_u_samples} interactions...')
            u_cnts = data.value_counts(U_COL)
            valid_us = u_cnts[u_cnts <= max_u_samples].index.values
            data = iterative_kcore_filter_revs(data[data[U_COL].isin(valid_us)], min_revs=min_revs, kcore=kcore,
                                               subset=[U_COL, I_COL])

        # Extract encoding mappers and save to separate file
        logging.info('Encoding...')
        data = encode(dataset, data, [U_COL, I_COL, FEAT_COL, EXP_COL], suffix, DATA_MODE)

        logging.info('Adding Time Step order')
        data.sort_values(TIMESTAMP_COL, inplace=True)
        data[TIMESTEP_COL] = list(range(data.shape[0]))

        # Extract sequential features grouped by U_COL and I_COL
        # NOTE 1: There is no ADJ_COL in XRec datasets
        # NOTE 2: max_seq_length is selected through EDA (Users surpassing 200-500 reviews are aggregated users in TripAdvisor)
        logging.info('Extracting sequential features...')
        tqdm.pandas()
        data = data.sort_values(by=[U_COL, TIMESTEP_COL]).reset_index(drop=True)

        orig_index = data.index.values
        assert (orig_index == list(range(data.shape[0]))).all()
        old_cols = list(data.columns)
        hist_cols = []
        for groupby in [U_COL, I_COL]:
            logging.info(f'Extracting history for {groupby} column')
            hist_cols.append(f'{groupby}_{HIST_PREFIX}')
            data.sort_values([groupby, TIMESTEP_COL], inplace=True)
            histories = data[[groupby, TIMESTEP_COL]].groupby(groupby).progress_apply(sliding_w_hist)
            data[hist_cols[-1]] = sum(histories.tolist(), [])
        data = data.loc[orig_index]

        pd.to_pickle(data[old_cols], os.path.join(DATA_PATHS[dataset], DATA_MODE, f_name))
        pd.to_pickle(data[hist_cols], os.path.join(DATA_PATHS[dataset], DATA_MODE, f'{HIST_PREFIX}_{f_name}'))

    split_data(datasets=[d[0] for d in datasets], mode=DATA_MODE, f_name=f_name, split=split)


def read_data(dataset, fold=0, data_file='reviews.pkl'):
    suffix = ''

    ixs_dir = os.path.join(DATA_PATHS[dataset], DATA_MODE, str(fold))
    data_path = os.path.join(DATA_PATHS[dataset], DATA_MODE, data_file)
    assert os.path.exists(data_path)
    data = pd.read_pickle(data_path)
    split_ixs = load_partition(ixs_dir)

    if not data[RAT_COL].dtype == float:
        data[RAT_COL] = data[RAT_COL].astype(float)

    # NOTE: P5 does subword tokenization of user and item encoded IDs.
    mapper_path = os.path.join(DATA_PATHS[dataset], DATA_MODE, 'mappers')
    mappers = {}
    for c in (U_COL, I_COL, EXP_COL, FEAT_COL):
        mapper = pd.read_pickle(os.path.join(mapper_path, f'{c}{suffix}.pkl'))
        mappers[c] = Mapper(mapper)

    mappers[I_COL].add_items(SPECIAL_TOKENS.values())  # mappers[I_COL].get_idx(UNK_TOK)
    mappers[EXP_COL].add_items([''])

    return data, split_ixs, mappers


def flatten_cols(data, filter_unique=False, preserve_order=False):
    data[FEAT_COL] = data[FEAT_COL].apply(partial(sum, start=[]))

    if preserve_order:
        # NOTE 1: ~x3 slower but preserves temporal order of sentences (unique starts from most recent sentences)
        # NOTE 2: pandas .unique() states "Uniques are returned in order of appearance"
        unique_fn = lambda l: pd.Series(l[::-1]).unique().tolist()[::-1]
    else:
        unique_fn = lambda l: list(set(l))

    if filter_unique:
        for c in [FEAT_COL, EXP_COL]:
            data[c] = data[c].apply(unique_fn)

    for c in [col for col in data.columns if HIST_PREFIX in col]:
        mask = data[c].str.len() > 0
        if FEAT_COL in c:
            data.loc[mask, c] = data.loc[mask, c].parallel_apply(lambda ls: sum(map(partial(sum, start=[]), ls), []))
        elif EXP_COL in c:
            data.loc[mask, c] = data.loc[mask, c].apply(partial(sum, start=[]))
        if filter_unique:
            if preserve_order:
                data.loc[mask, c] = data.loc[mask, c].parallel_apply(unique_fn)
            else:
                data.loc[mask, c] = data.loc[mask, c].apply(unique_fn)


def get_hist_data(data, hist_data_path, cols, hist_len):
    def sliding_w_hist(g: pd.DataFrame, max_l: int = MAX_HIST_LEN):
        return [g.index.values[max(i - max_l, 0):i].tolist() for i in range(g.shape[0])]

    def getter(ixs, select):
        return data[select].loc[ixs].tolist()

    if os.path.isfile(hist_data_path):
        data = pd.concat([data, pd.read_pickle(hist_data_path)], axis=1)
    else:
        tqdm.pandas()
        orig_index = data.index.values
        hist_cols = []
        for groupby in [U_COL, I_COL]:
            logging.info(f'Extracting history for {groupby} column')
            hist_cols.append(f'{groupby}_{HIST_PREFIX}')
            data.sort_values([groupby, TIMESTEP_COL], inplace=True)
            histories = data[[groupby, TIMESTEP_COL]].groupby(groupby).progress_apply(sliding_w_hist)
            data[hist_cols[-1]] = sum(histories.tolist(), [])
        data = data.loc[orig_index]
        data[hist_cols].to_pickle(hist_data_path)

    for groupby, select_cols in cols.items():
        max_l = hist_len[groupby]
        idxr_col = f'{groupby}_{HIST_PREFIX}'
        has_hist_mask = data[idxr_col].str.len() > 0
        idxr = data.loc[has_hist_mask, idxr_col]
        if max_l < MAX_HIST_LEN:
            idxr = idxr.map(lambda ixs: ixs[-max_l:])
        for c in select_cols:
            new_c = f'{idxr_col}_{c}'
            data[new_c] = np.empty((len(data), 0)).tolist()
            data.loc[has_hist_mask, new_c] = idxr.map(partial(getter, select=c))

        data.drop(idxr_col, axis=1, inplace=True)
    return data


def get_sentence_bleu(references, hypotheses, types=[1, 2, 3, 4]):
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


def get_aux_data_greener(dataset='tripadvisor', filter_item_attr=True, max_n_process=3):
    import torch
    from torch_geometric.data import HeteroData
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers=min(os.cpu_count(), max_n_process), progress_bar=False)

    def get_hist_len():
        def ceil2num(num_to_round, num):
            return int(np.ceil(num_to_round / num) * num)

        # Handcrafted values to resemble the provided GREENer processed average graph size
        target_i_sents = 400
        target_u_sents = 150

        data['exp_len'] = data[EXP_COL].str.len()
        req_i_len = target_i_sents // data.groupby(I_COL)['exp_len'].mean().mean()
        req_u_len = target_u_sents // data.groupby(U_COL)['exp_len'].mean().mean()
        hist_len = {
            I_COL: ceil2num(req_i_len, 5),
            U_COL: ceil2num(req_u_len, 5)
        }
        data.drop('exp_len', axis=1, inplace=True)
        return hist_len

    def filter_hist_item_attr():
        """ Filter those user explanations that do not contain an item attribute """

        # NOTE: It takes ~3 mins on TripAdvisor
        def filter_user_exps(r):
            return [e for e in r[U_HIST_EXP_COL] if e2f[e].intersection(r.i_feats)]

        data['i_feats'] = data[I_COL].map(i2f)
        mask = data[U_HIST_EXP_COL].str.len() > 0
        data.loc[mask, U_HIST_EXP_COL] = data.loc[mask, ['i_feats', U_HIST_EXP_COL]].parallel_apply(filter_user_exps,
                                                                                                    axis=1)
        data.drop('i_feats', axis=1, inplace=True)

    def select_exp_nodes(max_exps=510, max_gt=40, min_u=40):
        max_hist_exps = pd.Series([max_exps] * len(data), index=data.index.values)
        max_hist_exps.loc[split_ixs['train']] -= data[EXP_COL].str.len().clip(0, max_gt)

        n_u_exps = data[U_HIST_EXP_COL].str.len()

        need_clip_mask = (data[I_HIST_EXP_COL].str.len() + data[U_HIST_EXP_COL].str.len()) > max_exps

        n_i_poss_exps = (max_hist_exps - np.minimum(min_u, n_u_exps)).loc[need_clip_mask]
        i_exps = data[I_HIST_EXP_COL].copy()
        i_exps.loc[need_clip_mask] = [es[-l:] for es, l in zip(i_exps.loc[need_clip_mask].tolist(), n_i_poss_exps)]

        n_u_poss_exps = (max_hist_exps - i_exps.str.len()).loc[need_clip_mask]
        u_exps = data[U_HIST_EXP_COL].copy()
        u_exps.loc[need_clip_mask] = [es[-l:] for es, l in zip(u_exps.loc[need_clip_mask].tolist(), n_u_poss_exps)]

        trn_exps = data.loc[split_ixs['train'], EXP_COL].apply(lambda l: l[:max_gt])
        u_exps.loc[split_ixs['train']] = (u_exps.loc[split_ixs['train']] + trn_exps).apply(lambda l: list(set(l)))
        i_exps.loc[split_ixs['train']] = (i_exps.loc[split_ixs['train']] + trn_exps).apply(lambda l: list(set(l)))
        exps = (u_exps + i_exps).apply(lambda l: list(set(l)))

        return u_exps, i_exps, exps

    def get_softlabel(bleu_score):
        if bleu_score < 0.5:
            return 0
        elif bleu_score < 1.25:
            return 1
        elif bleu_score < 2:
            return 2

        return 3

    def build_heterograph(row, aux_path, save=True):
        g_file = f'{row.name}.pt'
        if os.path.isfile(os.path.join(aux_path, g_file)):
            return None

        g = HeteroData()

        u_exps = row[U_HIST_EXP_COL]
        i_exps = row[I_HIST_EXP_COL]
        exps = sorted(row[HIST_EXP_COL])

        if not exps:
            return None

        u_feats = list(set().union(*e2f.loc[u_exps].tolist()))
        i_feats = list(set().union(*e2f.loc[i_exps].tolist()))
        feats = sorted(list(set(u_feats + i_feats)))

        g[U_COL].x = torch.LongTensor([row[U_COL]])
        g[I_COL].x = torch.LongTensor([row[I_COL]])
        g[FEAT_COL].x = torch.LongTensor(feats)
        g[EXP_COL].x = torch.LongTensor(exps)

        feat_map = pd.Series(range(len(feats)), index=feats)
        # logging.info(f'The feat. map is: {feat_map.to_dict()}')
        exp_map = pd.Series(range(len(exps)), index=exps)
        # logging.info(f'The exp. map is: {exp_map.to_dict()}')

        # Build sf_adj (binarized edges)
        source_ixs, target_ixs = zip(*[[s, f] for f in feats for s in (set(exps) & f2e[f])])
        # logging.info(f'From exp. {source_ixs} to feat. {target_ixs}')
        source_ixs = exp_map.loc[list(source_ixs)].tolist()
        target_ixs = feat_map.loc[list(target_ixs)].tolist()
        if isinstance(source_ixs, int):
            source_ixs, target_ixs = [source_ixs], [target_ixs]
        g[EXP_COL, FEAT_COL].edge_index = torch.LongTensor(np.stack((source_ixs, target_ixs)))

        # Build uf_adj
        source_ixs = [0] * len(u_feats)
        target_ixs = feat_map.loc[u_feats].tolist()
        # logging.info(f'After map from user {source_ixs} to feat. {target_ixs}')
        g[U_COL, FEAT_COL].edge_index = torch.LongTensor(np.stack((source_ixs, target_ixs)))

        # Build if_adj
        source_ixs = [0] * len(i_feats)
        target_ixs = feat_map.loc[i_feats].tolist()
        # logging.info(f'After map from item {source_ixs} to feat. {target_ixs}')
        g[I_COL, FEAT_COL].edge_index = torch.LongTensor(np.stack((source_ixs, target_ixs)))

        # Get sentence labels
        # NOTE: Hard labels make it difficult as exact sentences have rarely been written in the past
        hard_labels = np.isin(exps, row[EXP_COL]).tolist()
        gt = mappers[EXP_COL].get_items(row[EXP_COL])

        bleu_refs = [nltk.word_tokenize(s) for s in gt]
        bleu_cands = [nltk.word_tokenize(s) for s in mappers[EXP_COL].get_items(exps)]
        # bleu_labels = [max(compute_bleu([[ref]], [cand], 4)[0] for ref in bleu_refs) for cand in bleu_cands]
        bleu_labels = [sum(get_sentence_bleu(bleu_refs, cand, types=[2, 3])) for cand in bleu_cands]
        bleu_soft_labels = list(map(get_softlabel, bleu_labels))

        g[EXP_COL].y = torch.FloatTensor(list(zip(hard_labels, bleu_soft_labels, bleu_labels)))
        g[FEAT_COL].y = torch.LongTensor(np.isin(feats, row[FEAT_COL]))

        if save:
            # g_file = f'{row.name}.pt'
            torch.save(g, os.path.join(aux_path, g_file))
            return None
        else:
            return pd.Series(g.to_dict(), name=row.name)

    def build_heterograph_dict(row):
        g = {}
        sep = '|'

        u_exps = row[U_HIST_EXP_COL]
        i_exps = row[I_HIST_EXP_COL]
        exps = sorted(row[HIST_EXP_COL])

        if not exps:
            return None

        u_feats = list(set().union(*e2f.loc[u_exps].tolist()))
        i_feats = list(set().union(*e2f.loc[i_exps].tolist()))
        feats = sorted(list(set(u_feats + i_feats)))

        g[f'{U_COL}{sep}x'] = [row[U_COL]]
        g[f'{I_COL}{sep}x'] = [row[I_COL]]
        g[f'{FEAT_COL}{sep}x'] = feats
        g[f'{EXP_COL}{sep}x'] = exps

        feat_map = pd.Series(range(len(feats)), index=feats)
        # logging.info(f'The feat. map is: {feat_map.to_dict()}')
        exp_map = pd.Series(range(len(exps)), index=exps)
        # logging.info(f'The exp. map is: {exp_map.to_dict()}')

        # Build sf_adj (binarized edges)
        source_ixs, target_ixs = zip(*[[s, f] for f in feats for s in (set(exps) & f2e[f])])
        # logging.info(f'From exp. {source_ixs} to feat. {target_ixs}')
        source_ixs = exp_map.loc[list(source_ixs)].tolist()
        target_ixs = feat_map.loc[list(target_ixs)].tolist()
        if isinstance(source_ixs, int):
            source_ixs, target_ixs = [source_ixs], [target_ixs]
        g[f'{EXP_COL}{sep}{FEAT_COL}{sep}edge_index'] = np.stack((source_ixs, target_ixs))

        # Build uf_adj
        source_ixs = [0] * len(u_feats)
        target_ixs = feat_map.loc[u_feats].tolist()
        # logging.info(f'After map from user {source_ixs} to feat. {target_ixs}')
        g[f'{U_COL}{sep}{FEAT_COL}{sep}edge_index'] = np.stack((source_ixs, target_ixs))

        # Build if_adj
        source_ixs = [0] * len(i_feats)
        target_ixs = feat_map.loc[i_feats].tolist()
        # logging.info(f'After map from item {source_ixs} to feat. {target_ixs}')
        g[f'{I_COL}{sep}{FEAT_COL}{sep}edge_index'] = np.stack((source_ixs, target_ixs))

        # Get sentence labels
        # NOTE: Hard labels make it difficult as exact sentences have rarely been written in the past
        hard_labels = np.isin(exps, row[EXP_COL]).tolist()
        gt = mappers[EXP_COL].get_items(row[EXP_COL])

        bleu_refs = [nltk.word_tokenize(s) for s in gt]
        bleu_cands = [nltk.word_tokenize(s) for s in mappers[EXP_COL].get_items(exps)]
        # bleu_labels = [max(compute_bleu([[ref]], [cand], 4)[0] for ref in bleu_refs) for cand in bleu_cands]
        bleu_labels = [sum(get_sentence_bleu(bleu_refs, cand, types=[2, 3])) for cand in bleu_cands]
        bleu_soft_labels = list(map(get_softlabel, bleu_labels))

        g[f'{EXP_COL}{sep}y'] = list(zip(hard_labels, bleu_soft_labels, bleu_labels))
        g[f'{FEAT_COL}{sep}y'] = np.isin(feats, row[FEAT_COL])

        return pd.Series(g, row.name)

    def to_pyg_heterograph(data):
        g = HeteroData()
        for field in [U_COL, I_COL, EXP_COL, FEAT_COL]:
            g[field].x = torch.LongTensor(data[f'{field}|x'])
        for field in [U_COL, I_COL, EXP_COL]:
            g[field, FEAT_COL].edge_index = torch.LongTensor(data[f'{field}|{FEAT_COL}|edge_index'])
        g[EXP_COL].y = torch.FloatTensor(data[f'{EXP_COL}|y'])
        g[FEAT_COL].y = torch.LongTensor(data[f'{FEAT_COL}|y'])
        return g

    logging.info('Reading the data, split and mappers...')
    data, split_ixs, mappers = read_data(dataset)

    keep_cols = [U_COL, I_COL, FEAT_COL, EXP_COL, TIMESTEP_COL]
    data = data[keep_cols]

    # Load Hist data
    logging.info('Loading historical data...')
    hist_data_path = os.path.join(DATA_PATHS[dataset], DATA_MODE, f'{HIST_PREFIX}_reviews.pkl')
    data = get_hist_data(data, hist_data_path, cols={U_COL: [EXP_COL], I_COL: [EXP_COL]}, hist_len=get_hist_len())

    logging.info('Getting user/item/exp to feature mappers...')
    e2f_path = os.path.join(DATA_PATHS[dataset], DATA_MODE, 'mappers', f'{EXP_COL}2{FEAT_COL}.pkl')
    if os.path.isfile(e2f_path):
        e2f = pd.read_pickle(e2f_path)
    else:
        e2f = data[[FEAT_COL, EXP_COL]].explode([EXP_COL, FEAT_COL]).explode(FEAT_COL).groupby(EXP_COL).agg(set)[FEAT_COL]
        e2f.to_pickle(e2f_path)
    f2e = e2f.explode().reset_index().groupby(FEAT_COL).agg(set)[EXP_COL]

    unrolled = data[[I_COL, U_COL, FEAT_COL]].explode(FEAT_COL).explode(FEAT_COL)
    # Get mappers
    i2f = unrolled.groupby(I_COL)[FEAT_COL].agg(set)
    # u2f = unrolled.groupby(U_COL)[FEAT_COL].agg(set)

    logging.info('Flattening columns...')
    flatten_cols(data, filter_unique=True, preserve_order=True)

    if filter_item_attr:
        logging.info('Filtering out user exps. without any item attribute...')
        # QUESTION: Should we only consider item attributes from the train+valid set?
        filter_hist_item_attr()

    logging.info('Selecting exp. nodes based on max. n_nodes...')
    max_exps, max_gt, min_u = 510, 40, 40
    data[U_HIST_EXP_COL], data[I_HIST_EXP_COL], data[HIST_EXP_COL] = select_exp_nodes(max_exps, max_gt, min_u)

    logging.info('Filling up test samples without explanation candidates...')

    def sample_exps(item, n=510):
        poss_exps = list(i2e.loc[item])
        if len(poss_exps) <= n:
            return poss_exps
        return random.sample(poss_exps, n)

    mask = np.zeros(data.shape[0], dtype=bool)
    mask[split_ixs['test']] = True
    mask = mask & (data.sort_index()[HIST_EXP_COL].str.len() == 0).values
    if sum(mask) > 0:
        trn_val_exps = data.loc[split_ixs['train'] + split_ixs['valid'], [I_COL, EXP_COL]].explode(EXP_COL)
        i2e = trn_val_exps.groupby(I_COL)[EXP_COL].agg(set)
        sampled_exps = data.loc[mask, I_COL].apply(partial(sample_exps, n=max_exps))
        data.loc[mask, HIST_EXP_COL] = sampled_exps
        data.loc[mask, I_HIST_EXP_COL] = sampled_exps

    logging.info('Building and Saving HeteroGraphs...')
    folder = f'GREENer_graphs_custom{"_item_attr" if filter_item_attr else ""}'
    aux_path = os.path.join(DATA_PATHS[dataset], DATA_MODE, '0', 'aux', folder)
    if not os.path.isdir(aux_path):
        os.makedirs(aux_path)
    data.parallel_apply(partial(build_heterograph, aux_path=aux_path), axis=1)

    logging.info('Finished...')


class DebugDataset(BaseDataset):
    def _finish_init(self):
        pass


def complete_cfg(cfg, dataset='tripadvisor', fold='0', seq_mode=SeqMode.NO_SEQ, test_flag=True, batch_size=128,
                 model_tasks=None, seed=RNG_SEED):
    if hasattr(cfg, 'stages'):
        mod_cfg = cfg.stages[0]
    else:
        mod_cfg = cfg
    mod_cfg.data.requires_exp = (Task.EXPLANATION in model_tasks or getattr(mod_cfg.data, 'requires_exp', False))
    mod_cfg.data.requires_feat = (Task.FEAT in model_tasks or getattr(mod_cfg.data, 'requires_feat', False))
    mod_cfg.data.requires_rating = (Task.RATING in model_tasks or getattr(mod_cfg.data, 'requires_rating', False))
    mod_cfg.data.requires_context = (Task.CONTEXT in model_tasks or getattr(mod_cfg.data, 'requires_context', False))
    mod_cfg.data.requires_nextitem = (
                Task.NEXT_ITEM in model_tasks or getattr(mod_cfg.data, 'requires_nextitem', False))
    mod_cfg.data.txt_len = TXT_LEN.get(dataset, 35)
    mod_cfg.data.seq_mode = seq_mode
    mod_cfg.data.fold = fold
    mod_cfg.data.dataset = dataset
    mod_cfg.data.test_flag = test_flag
    cfg.data.batch_size = batch_size
    mod_cfg.data.txt_len = TXT_LEN.get(dataset, 35)
    mod_cfg.model.txt_len = TXT_LEN.get(dataset, 35)
    mod_cfg.seed = seed
    mod_cfg.gen_flag = False
    mod_cfg.save_results = False
    mod_cfg.data.eval_workers = 0


def get_uneg_samples(dataset, n_neg=99):
    def neg_sample(item_seq: set, replace: bool = False):
        # neg_items = self.item_set.difference(self.user_item.loc[user])
        # assert item_seq.issubset(self.item_set)
        neg_items = list(item_set.difference(item_seq))
        if n_neg >= len(neg_items):
            random.shuffle(neg_items)
        else:
            neg_items = np.random.choice(neg_items, n_neg, replace=replace).tolist()
        return neg_items

    logging.info(f"Sampling User Negative items for the {dataset} dataset...")
    init_seed(RNG_SEED)

    logging.info('Reading the data, split and mappers...')
    data, split_ixs, mappers = read_data(dataset)

    keep_cols = [U_COL, I_COL, FEAT_COL, EXP_COL, TIMESTEP_COL]
    data = data[keep_cols]

    item_set = set(mappers[I_COL].item2idx.tolist())

    logging.info('Sampling user-negatives...')
    neg_sample_path = os.path.join(DATA_PATHS[dataset], DATA_MODE, 'user_neg_samples.pkl')
    u2i = data[[U_COL, I_COL]].groupby(U_COL)[I_COL].agg(set)
    user_neg_samples = {uid: neg_sample(iset) for uid, iset in u2i.items()}

    logging.info('Saving negative samples...')
    pd.to_pickle(user_neg_samples, neg_sample_path)


def get_aux_data(model_name, **kwargs):
    dataset = kwargs['dataset']
    if 'greener' in model_name.lower():
        get_aux_data_greener(dataset, kwargs.get('filter_item_attr', True), kwargs.get('max_n_process', 2))
    elif any([m in model_name.lower() for m in ['sequer']]):
        get_uneg_samples(dataset, n_neg=kwargs.get('n_neg', 99))

    logging.info('Finished!')


if __name__ == '__main__':
    # pandarallel.initialize(progress_bar=False)
    init_seed(RNG_SEED)

    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', None)
    pd.set_option('expand_frame_repr', False)

    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s | %(levelname)s - %(message)s',
                        handlers=handlers)

    process_xrec_datasets()

    # get_aux_data(model_name='sequer', dataset='tripadvisor')
    # get_aux_data(model_name='greener', dataset='ratebeer', filter_item_attr=True, max_n_process=2)
