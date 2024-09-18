import argparse
import json
import logging
import math
import os
import queue
import re
import sys
import time
from typing import List, Dict, Callable

import nltk
import numpy as np
import pandas as pd
import stanza
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from src.utils.constants import FEAT_COL, EXP_COL, REV_COL, I_COL, U_COL, RAT_COL, TIMESTAMP_COL, DATA_PATHS


def read_data(dataset: str):
    print(f'Loading {dataset} dataset...')
    st = time.time()
    dmap = {'tripadvisor': 'TripAdvisor', 'yelp': 'Yelp', 'ratebeer': 'RateBeer'}
    if dataset.lower() == 'tripadvisor':
        data = []
        for fi, f in enumerate(os.listdir(os.path.join(DATA_PATHS[dataset.lower()], 'orig', 'json'))):
            if f.endswith('.json'):
                with open(os.path.join(DATA_PATHS[dataset.lower()], 'orig', 'json', f), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                for review in json_data['Reviews']:
                    data.append({
                        I_COL: json_data['HotelInfo']['HotelID'],
                        U_COL: review['Author'],
                        RAT_COL: review['Ratings']['Overall'],
                        f'{RAT_COL}_service': review['Ratings'].get('Service', np.nan),
                        f'{RAT_COL}_cleanliness': review['Ratings'].get('Cleanliness', np.nan),
                        f'{RAT_COL}_value': review['Ratings'].get('Value', np.nan),
                        f'{RAT_COL}_sleepQuality': review['Ratings'].get('Sleep Quality', np.nan),
                        f'{RAT_COL}_rooms': review['Ratings'].get('Rooms', np.nan),
                        f'{RAT_COL}_location': review['Ratings'].get('Location', np.nan),
                        TIMESTAMP_COL: review['Date'],
                        REV_COL: review['Content']
                    })

        data = pd.DataFrame.from_records(data)

        with open(os.path.join(DATA_PATHS[dataset.lower()], 'orig', 'feature', 'features.json'), 'r') as f:
            feats = set(json.loads(f.read()).keys())

    elif dataset.lower() == 'yelp':
        data = pd.read_pickle(os.path.join(DATA_PATHS[dataset.lower()], 'orig', 'review_splits.pkl'))
        data = data['train'] + data['val'] + data['test']
        data = pd.DataFrame.from_records(data)

        col_map = {'reviewerID': U_COL, 'asin': I_COL, 'reviewText': REV_COL, 'unixReviewTime': TIMESTAMP_COL,
                   'overall': RAT_COL}
        data.rename(col_map, axis=1, inplace=True)
        data.dropna(subset=[REV_COL], inplace=True, axis=0)
        data.reset_index(drop=True, inplace=True)

        # feats = data['feature'].unique().tolist()
        with open(os.path.join(DATA_PATHS[dataset.lower()], 'orig', 'feature', 'features.json'), 'r') as f:
            feats = set(json.loads(f.read()))
        data = data[list(col_map.values())]

    elif dataset.lower() == 'ratebeer':
        data = pd.read_json(os.path.join(DATA_PATHS[dataset.lower()], 'orig', 'ratebeer.json'), orient='records', lines=True,
                            encoding='ISO-8859-1')

        c_map = {'beer/beerId': I_COL, 'review/overall': RAT_COL, 'review/time': TIMESTAMP_COL,
                 'review/profileName': U_COL, 'review/text': REV_COL, 'review/appearance': f'{RAT_COL}_appearance',
                 'review/aroma': f'{RAT_COL}_aroma', 'review/palate': f'{RAT_COL}_palate',
                 'review/taste': f'{RAT_COL}_taste'}

        data = data[c_map.keys()]
        data.rename(c_map, axis=1, inplace=True)

        for c in c_map.values():
            if c.startswith(RAT_COL):
                data[c] = data[c].str.split('/').str[0]
                data[c].fillna(-1, inplace=True)
                data[c] = data[c].astype(int)

        with open(os.path.join(DATA_PATHS[dataset.lower()], 'orig', 'feature', 'features.json'), 'r') as f:
            feats = set(json.loads(f.read()).keys())
    else:
        raise NotImplementedError(f'read_data() does not currently supports {dataset} dataset.')

    # NOTE: In RateBeer, there are NaN reviews and empty reviews
    data = data[data[REV_COL].str.len() > 0]
    # data.dropna(subset=[REV_COL], inplace=True)
    print(f'Ellapsed time {time.time() - st:.4f}s')
    return data, feats


def clean_sentence(s):
    s = re.sub(r'https?://.*', ' ', s)
    s = re.sub(r'\.\.+', '...', s)
    s = re.sub(r'`', '\'', s)
    s = re.sub(r'^\s*-', '', s) \
        .replace('*', ' ') \
        .replace('-', ' - ') \
        .replace('/', ' / ') \
        .replace('~', ' ')

    s = re.sub(r'\s\s+', ' ', s).strip().lower()
    return s


def preprocess_reviews(data, verbose=False):
    revs = data[REV_COL]
    sents = revs.apply(nltk.sent_tokenize).explode()
    sents2 = sents.str.split('\n').explode()
    if verbose:
        if len(sents) != len(sents2):
            diff = sents.value_counts(sort=False) != sents2.value_counts(sort=False)
            print(f'List of sentences where we needed to split_lines: {diff[diff].index.tolist()}')
        else:
            assert (sents.value_counts(sort=False) == sents2.value_counts(sort=False)).all()

    # Clean sentences
    sents = sents2.apply(clean_sentence)

    # Filter too long and short sentences
    lens = sents.str.len()
    mask = np.logical_and(lens > 3, lens < 400)
    sents2 = sents[mask]

    # Aggregate back review sentences
    return sents2.groupby(level=0).agg("\n\n".join)


def nlp_exp_rule(doc, feats: set):
    out = {FEAT_COL: [], EXP_COL: []}
    for sen in doc.sentences:
        feats_in_exp = set()
        is_exp = False
        for w in sen.words:
            if w.text in feats:
                feats_in_exp.add(w.text)
                while w.deprel == 'compound':
                    w = sen.words[w.head - 1]

                if w.deprel == 'nsubj' or w.deprel == 'nsubj:pass':
                    head = sen.words[w.head - 1]
                    if head.xpos in {'JJ', 'JJR', 'JJS', 'RB', 'VBN', 'NN'}:
                        is_exp = True
            elif w.xpos in {'JJ', 'JJR', 'JJS'}:
                head = sen.words[w.head - 1]
                if head.deprel == 'root' and head.text in feats:
                    is_exp = True
        if is_exp:
            out[EXP_COL].append(' '.join(w.text for w in sen.words))
            out[FEAT_COL].append(list(feats_in_exp))
    return out


def naive_exp_rule(doc, feats: set):
    out = {FEAT_COL: [], EXP_COL: []}

    for sen in doc.sentences:
        feats_in_exp = set()
        is_exp = False

        words = sen.words
        for word in words:
            text = word.text

            if text in feats:
                feats_in_exp.add(text)
                is_exp = True

        if is_exp:
            out[EXP_COL].append(' '.join(w.text for w in words))
            out[FEAT_COL].append(list(feats_in_exp))

    return out


@torch.no_grad()
def extract_exps_stanza(dataset: str, nlp_processors: str, rule_fn: Callable, batch_size: int = 64):
    handlers = [logging.StreamHandler(stream=sys.stdout)]

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s - %(message)s',
                        handlers=handlers)

    if not os.path.isdir(os.path.join(DATA_PATHS[dataset], 'feat_only')):
        os.makedirs(os.path.join(DATA_PATHS[dataset], 'feat_only'))

    logging.info('Reading data...')
    data, feats = read_data(dataset)

    logging.info('Preprocessing reviews...')
    sentences = preprocess_reviews(data)
    data['sents'] = ''
    data.loc[sentences.index.values, 'sents'] = sentences
    # NOTE - Yelp: ~1k samples are removed due to either too long or too short reviews
    data = data[data['sents'] != ''].reset_index(drop=True)
    sentences = data['sents'].tolist()
    data.drop('sents', axis=1, inplace=True)
    # data = data.iloc[:(batch_size * 10)]

    logging.info('Converting to tokenized documents...')
    nlp = stanza.Pipeline('en', processors=nlp_processors, tokenize_no_ssplit=True,
                          use_gpu=(nlp_processors != 'tokenize'), logging_level='ERROR')

    # n_batches = int(np.ceil(len(sentences)/batch_size))
    feat_exps = [rule_fn(d, feats) for d in tqdm(nlp.stream(sentences, batch_size=batch_size), total=len(sentences))]

    logging.info('Extracting Feats-Exps from documents...')
    # exps = pd.DataFrame.from_records(list(map(partial(rule_fn, feats=feats), docs)))
    feat_exps = pd.DataFrame.from_records(feat_exps)

    data = pd.concat((data, feat_exps), axis=1)

    # NOTE - Yelp: ~65k samples are removed due to empty explanations
    empty_mask = data[EXP_COL].apply(len) == 0
    if sum(empty_mask) > 0:
        logging.info(f'Removing {sum(empty_mask)} / {data.shape[0]} records due to empty explanation')
        data[empty_mask].reset_index(drop=True).to_pickle(
            os.path.join(DATA_PATHS[dataset], 'feat_only', 'empty_exps.pkl'))
        data = data[~empty_mask].reset_index(drop=True)

    logging.info('Saving processed data...')

    data.to_pickle(os.path.join(DATA_PATHS[dataset], 'feat_only', 'reviews.pkl'))


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    parser = argparse.ArgumentParser(description='Feature-Explanation Extraction Script')
    parser.add_argument('--dataset', type=str, default='tripadvisor',
                        help='dataset name (tripadvisor, ratebeer, yelp)')
    args = parser.parse_args()

    dataset = args.dataset

    dataset_map = {
        'yelp': ('tokenize,mwt,pos,lemma,depparse', nlp_exp_rule),
        'tripadvisor': ('tokenize,mwt,pos,lemma,depparse', nlp_exp_rule),
        'ratebeer': ('tokenize', naive_exp_rule)
    }
    extract_exps_stanza(dataset, *dataset_map[dataset])
