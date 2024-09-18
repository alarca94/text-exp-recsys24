import logging
import re
from typing import Iterable, List, Dict

import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score
from nltk import ngrams
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .bleu import compute_bleu
from .constants import *
from .data import postprocessing
from .in_out import save_results
from .rouge import rouge_multi


def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge_multi(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    # formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(references, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def unique_sentence_ratio(sentences):
    unique_sents = set(sentences)
    return len(unique_sents) / len(sentences), len(unique_sents)


def token_diversity(samples):
    rep_count, total_count = 0, 0
    seq_rep_2_list, uniq_tokens = [], set()
    for tokens in samples:
        # Compute rep/l
        uniq = set(tokens)
        rep_count += (len(tokens) - len(uniq))
        total_count += len(tokens)

        # Compute seq_rep_2
        grams = list(ngrams(tokens, 2))
        if grams:
            seq_rep_2 = 1 - len(set(grams)) / len(grams)
            seq_rep_2_list.append(seq_rep_2)

        uniq_tokens |= set(uniq)

    rep_l = np.nan if total_count == 0 else rep_count / total_count
    return rep_l, np.mean(seq_rep_2_list), len(uniq_tokens)


def feature_detect_old(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_detect(texts, feature_set):
    p = re.compile(f'\\b({"|".join(feature_set)})\\b')
    return [set(p.findall(t)) for t in texts]


def feature_matching_ratio(feature_batch, test_feature, ignore=None):
    count = 0
    norm = sum([f != ignore for f in test_feature])
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea != ignore and fea in fea_set:
            count += 1

    return count / norm  # len(feature_batch)


def feature_pr(pred_feats, gt_feats, ignore=None):
    p_sum, r_sum = 0, 0

    length = 0
    for pred, gt in zip(pred_feats, gt_feats):
        if not gt:
            continue

        if ignore is not None:
            pred.discard(ignore)
        matches = pred.intersection(gt)

        p_sum += len(matches) / len(pred) if pred else 0
        r_sum += len(matches) / len(gt)

        length += 1

    return p_sum / length, r_sum / length


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def hr(y_true, y_pred):
    # y_true :: [BSZ, ]  |||  y_pred :: [BSZ, K]
    return np.equal(y_true.reshape(-1, 1), y_pred).any(axis=1).mean()


def mrr(y_true, y_pred):
    # y_true :: [BSZ, ]  |||  y_pred :: [BSZ, K]
    rank_i = np.argwhere(np.equal(y_true.reshape(-1, 1), y_pred))[:, 1] + 1
    return 1 / y_pred.shape[0] * (1 / rank_i).sum()


def ndcg(y_true, y_pred, rels=None):
    # y_true :: [BSZ, ]  |||  y_pred :: [BSZ, K].
    # Implementation for next item prediction (Only one item is relevant at a given time)
    def dcg(rel, rank):
        return ((2 ** rel - 1) / np.log2(rank + 1)).sum()

    if rels is None:
        rels = np.ones(y_true.max() + 1)

    ranks = np.argwhere(np.equal(y_true.reshape(-1, 1), y_pred))
    rel_i = rels[y_true[ranks[:, 0]]]
    rank_i = ranks[:, 1] + 1
    return dcg(rel_i, rank_i) / dcg(np.ones_like(y_true), np.ones_like(y_true))


def minmax_norm(vals, minmax_vals):
    vals = np.clip(torch.tensor(vals), *minmax_vals)
    return (vals - minmax_vals[0]) / (minmax_vals[1] - minmax_vals[0])


def feat_miss_ratio(pred_feats: List[set], items: List[int], i2f: Dict[int, set], eps: float = 1e-8):
    num, den, fhr, missed_docs = 0, 0, 0, 0
    for i, f in zip(items, pred_feats):
        n_missed = len(f.difference(i2f[i]))
        den += len(f)
        num += n_missed
        fhr += n_missed / (len(f) + eps)
        missed_docs += int(n_missed > 0)

    cfhr = np.nan if den == 0 else num / den
    return cfhr, missed_docs / len(pred_feats), fhr / len(pred_feats)


class Evaluator:
    def __init__(self, tasks, data, prediction_path, exp_metadata, test_flag, gen_flag, save_flag, model_type,
                 device='cpu', logger=None, do_lower_case=False):
        self.tasks = [t if t != Task.MLM else Task.NEXT_ITEM for t in tasks]
        self.results = exp_metadata.copy()
        self.rating_range = (data.data_info.min_rating, data.data_info.max_rating)
        self.w_tok = data.tok  # [EXP_COL]
        self.mappers = data.mappers
        self.i_pad_tok = data.data_info.special_items[PAD_TOK]
        self.feature_set = data.feature_set
        self.i2f = data.i2f
        self.model_type = model_type
        # self.i_tok = toks[I_COL]
        # self.u_tok = toks[U_COL]
        self.prediction_path = prediction_path
        self.do_lower_case = do_lower_case
        self.device = device
        if self.do_lower_case:
            self.feature_set = list(set([f.lower() for f in self.feature_set]))
        self.test_flag = test_flag
        self.gen_flag = gen_flag
        self.save_flag = save_flag
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(PROJECT_NAME)

        # NOTE: ngram_idf is only computed for word unigrams (same as CompExp)
        self.ngram_idf = self.get_ngram_idf()

        task_metric_map = {
            Task.RATING: self.compute_rating,
            Task.NEXT_ITEM: self.compute_next_item,
            Task.TOPN: self.compute_topn,
            Task.EXPLANATION: self.compute_exp
        }

        self.metric_fns = {task: task_metric_map[task] for task in self.tasks if task in task_metric_map}

    def get_ngram_idf(self):
        unique_sent_words = self.mappers[EXP_COL].idx2item['raw'].apply(postprocessing).str.split().apply(set)
        return np.log(len(unique_sent_words) / unique_sent_words.explode().value_counts()) + 1

    def decode_skip(self, ids, skip_decode=False):
        if skip_decode:
            text = ids
        else:
            text = self.w_tok.decode(ids)

        # Make sure special tokens are separated by a space
        for tok in [getattr(self.w_tok, k) for k in list(SPECIAL_TOKENS.keys())]:
            text = re.sub(f'\s*{tok}\s*', f' {tok} ', text)

        # Make NLG evaluation case-insensitive
        if self.do_lower_case:
            text = text.lower()

        text = postprocessing(text)  # PETER processes punctuations: "good!" -> "good !"

        tokens = []
        for i, token in enumerate(text.split()):
            if token in [self.w_tok.eos_token, self.w_tok.pad_token]:  # or i == TXT_LEN:
                break
            tokens.append(token)
        return tokens

    def compute_rating(self, labels, preds):
        # Compute RMSE and MAE metrics
        pred = np.clip(preds[Task.RATING], *self.rating_range)
        self.results['RMSE'] = np.sqrt(mean_squared_error(labels[Task.RATING], pred))
        self.logger.info('RMSE {:7.4f}'.format(self.results['RMSE']))
        self.results['MAE'] = mean_absolute_error(labels[Task.RATING], pred)
        self.logger.info('MAE {:7.4f}'.format(self.results['MAE']))

    def full_rec_eval(self, labels, preds, ks, pad_tok, prefix=''):
        preds = np.array(preds)

        y_true = np.array(labels)
        mask = y_true != pad_tok
        y_true = y_true[mask]

        for k in ks:
            if k > preds.shape[-1]:
                continue

            y_pred = preds[mask, :k]
            self.results[f'{prefix}HR@{k}'] = hr(y_true, y_pred)
            self.logger.info('{}HR@{} {:7.4f} %'.format(prefix, k, self.results[f'{prefix}HR@{k}'] * 100))
            self.results[f'{prefix}MRR@{k}'] = mrr(y_true, y_pred)
            self.logger.info('{}MRR@{} {:7.4f} %'.format(prefix, k, self.results[f'{prefix}MRR@{k}'] * 100))
            self.results[f'{prefix}NDCG@{k}'] = ndcg(y_true, y_pred)
            self.logger.info('{}NDCG@{} {:7.4f} %'.format(prefix, k, self.results[f'{prefix}NDCG@{k}'] * 100))

    def compute_next_item(self, labels, pred):
        """ Logit version of the sequential recommendation evaluation.
         Task.NEXT_ITEM considers all items in the vocabulary
         Task.NEXT_ITEM_SAMPLE considers the positive item (first) and n_neg negative items (last)
        """
        def get_sample_metrics():
            topk_mask = pred[Task.NEXT_ITEM_SAMPLE] <= k
            ndcg = (1.0 / np.log2(pred[Task.NEXT_ITEM_SAMPLE][topk_mask] + 1.0)).sum() / len(topk_mask)
            hr = topk_mask.mean()
            return hr, ndcg

        # Compute HR@k, MRR@k and NDCG@k with k in (5, 10, 20). Ignore index will ignore cold_start users in the test
        self.full_rec_eval(labels[Task.NEXT_ITEM], pred[Task.NEXT_ITEM], TOP_KS, self.i_pad_tok)

        if Task.NEXT_ITEM_SAMPLE in pred:
            pred[Task.NEXT_ITEM_SAMPLE] = np.array(pred[Task.NEXT_ITEM_SAMPLE])
            pred[Task.NEXT_ITEM_SAMPLE] = (-pred[Task.NEXT_ITEM_SAMPLE]).argsort().argsort()[:, 0] + 1
            self.results[f'S-MRR'] = (1.0 / pred[Task.NEXT_ITEM_SAMPLE]).mean()
            self.logger.info('S-MRR {:7.4f} %'.format(self.results[f'S-MRR'] * 100))

            for k in TOP_KS:
                self.results[f'S-HR@{k}'], self.results[f'S-NDCG@{k}'] = get_sample_metrics()
                self.logger.info('S-HR@{} {:7.4f} %'.format(k, self.results[f'S-HR@{k}'] * 100))
                self.logger.info('S-NDCG@{} {:7.4f} %'.format(k, self.results[f'S-NDCG@{k}'] * 100))

    def compute_topn(self, labels, pred):
        self.full_rec_eval(labels[Task.TOPN], pred[Task.TOPN], TOP_KS, self.i_pad_tok, prefix='TN-')

    def decode_labels(self, labels, mapper=None, sep=None, as_tokens=True, skip_decode=True):
        if mapper is not None:
            labels = [mapper.get_items(idxs) for idxs in labels]

        if sep is not None:
            if as_tokens:
                return [self.decode_skip(sep.join(idxs), skip_decode=skip_decode) for idxs in labels]
            else:
                return [sep.join(self.decode_skip(sep.join(ids), skip_decode=skip_decode)) for ids in labels]

        return [self.decode_skip(ids) for ids in labels]

    @staticmethod
    def labels2toks(labels, mapper, split=True):
        gt = []
        for exps in labels:
            exps = mapper.get_items(exps)
            exps = [postprocessing(exp) for exp in exps]
            if split:
                exps = [exp.split() for exp in exps]
            gt.append(exps)
        return gt

    def decode_preds(self, preds, mapper=None):
        exps = []
        for exp in preds:
            if mapper is not None:
                text = mapper.get_item(exp)
            else:
                text = re.sub(f'^(\s?{self.w_tok.pad_token}\s?)+', '', self.w_tok.decode(exp))
                for token in [getattr(self.w_tok, k) for k in list(SPECIAL_TOKENS.keys())]:
                    text = re.sub(f'\s*{token}\s*', f' {token} ', text)

            if self.do_lower_case:
                text = text.lower()

            text = postprocessing(text)  # PETER processes punctuations: "good!" -> "good !"

            if mapper is not None:
                exps.append(text.split())
            else:
                tokens = []
                for i, token in enumerate(text.split()):
                    if token in [self.w_tok.eos_token, self.w_tok.pad_token]:  # or i == TXT_LEN:
                        break
                    tokens.append(token)
                exps.append(tokens)
        return exps

    def compute_exp(self, labels, pred):
        gt_tokens = self.labels2toks(labels[Task.EXPLANATION], self.mappers[EXP_COL])
        gt_feats = self.labels2toks(labels[Task.FEAT], self.mappers[FEAT_COL], split=False)
        gt_feats = list(map(set, gt_feats))
        if self.model_type == ModelType.EXTRACTIVE:
            # Select the first explanation
            if isinstance(pred[Task.EXPLANATION][0], Iterable):
                pred[Task.EXPLANATION] = [exps[0] for exps in pred[Task.EXPLANATION]]
            pred_tokens = self.decode_preds(pred[Task.EXPLANATION], self.mappers[EXP_COL])
        elif self.model_type in [ModelType.GENERATIVE, ModelType.HYBRID]:
            pred_tokens = self.decode_preds(pred[Task.EXPLANATION])

        gt_text = [list(map(' '.join, toks)) for toks in gt_tokens]
        pred_text = list(map(' '.join, pred_tokens))

        # TODO: Add IDF-BLEU for multiple references
        # ngram_idf = self.get_ngram_idf()
        for n in [1, 2, 4]:
            self.results[f'BLEU-{n}'] = bleu_score(gt_tokens, pred_tokens, n_gram=n, smooth=False)
            self.logger.info('BLEU-{} {:7.4f}'.format(n, self.results[f'BLEU-{n}']))
        # self.results['USR'], self.results['USN'] = unique_sentence_percent(pred_tokens)
        self.results['USR'], self.results['USN'] = unique_sentence_ratio(pred_text)
        self.logger.info('USR {:7.4f} | USN {:7}'.format(self.results['USR'], self.results['USN']))
        pred_feats = feature_detect(pred_text, self.feature_set)
        self.results['DIV'] = feature_diversity(pred_feats)  # time-consuming
        self.logger.info('DIV {:7.4f}'.format(self.results['DIV']))
        self.results['FCR'] = feature_coverage_ratio(pred_feats, self.feature_set)
        self.logger.info('FCR {:7.4f}'.format(self.results['FCR']))
        # self.results['FMR'] = feature_matching_ratio(pred_feats, gt_feats, ignore=UNK_TOK)
        # self.logger.info('FMR {:7.4f}'.format(self.results['FMR']))
        self.results['FP'], self.results['FR'] = feature_pr(pred_feats, gt_feats, ignore=self.w_tok.unk_token)
        self.logger.info('FP {:7.4f} | FR {:7.4f}'.format(self.results['FP'], self.results['FR']))
        # NOTE (2024-01-14): ROUGE changed to ROUGE_Multi as we now have multiple references per hypothesis
        #  (https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py)
        ROUGE = rouge_score(gt_text, pred_text)  # a dictionary
        for (k, v) in ROUGE.items():
            self.results[k] = v
            self.logger.info('{} {:7.4f}'.format(k, v))

        # CompExp new metrics: IDF/Word, Avg. Exp. Length, Exp. 2-gram Repetition and Word IDF
        self.results['rep/l'], self.results['seq_rep_2'], self.results['UTN'] = token_diversity(pred_tokens)
        self.logger.info(f"rep/l {self.results['rep/l']:7.4f} | seq_rep_2 {self.results['seq_rep_2']:7.4f} | "
                                  f"UTN {self.results['UTN']:7}")

        self.results['avg_len'] = np.mean(list(map(len, pred_tokens)))
        self.results['w_idf'] = np.mean([np.mean([self.ngram_idf.get(w, 1) for w in pred]) for pred in pred_tokens])
        self.logger.info(f'avg_len {self.results["avg_len"]:7.4f} | w_idf {self.results["w_idf"]:7.4f}')

        # Hallucination metric
        pred_feat_enc = [set(self.mappers[FEAT_COL].get_idxs(list(feats))) for feats in pred_feats]
        self.results['C-FHR'], self.results['D-FHR'], self.results['FHR'] = feat_miss_ratio(pred_feat_enc, labels[f'metadata-{I_COL}'], self.i2f)
        self.logger.info(f'C-FHR {self.results["C-FHR"]:.4f} | D-FHR {self.results["D-FHR"]:.4f} | FHR {self.results["FHR"]:.4f}')

        # NOTE: bert_score already admits multiple references per hypothesis
        bert_model = 'microsoft/deberta-base-mnli'
        BERTScore, hash_code = bert_score(refs=gt_text, cands=pred_text, model_type=bert_model, idf=True,
                                          use_fast_tokenizer=True, device=self.device, return_hash=True)

        for (v, k) in zip(BERTScore, ['p_score', 'r_score', 'f_score']):
            k = f'BERTScore/{k}'
            v = v.mean().item()
            self.results[k] = v
            self.logger.info('{} {:7.4f}'.format(k, v))

        if self.device == torch.device('cuda'):
            torch.cuda.empty_cache()

        if self.gen_flag:
            self.print_generated(labels, pred, gt_text, pred_text, self.prediction_path)
            self.logger.info('Generated text saved to ({})'.format(self.prediction_path))

    def print_generated(self, labels, pred, gt_text=None, pred_text=None, prediction_path=None):
        if gt_text is None:
            gt_tokens = self.labels2toks(labels[Task.EXPLANATION], self.mappers[EXP_COL])
            gt_text = [list(map(' '.join, toks)) for toks in gt_tokens]

        gt_text = [str(ts) for ts in gt_text]

        if pred_text is None:
            pred_tokens = self.decode_preds(pred[Task.EXPLANATION])
            pred_text = list(map(' '.join, pred_tokens))

        df = {'True': gt_text, 'Pred': pred_text}
        if Task.CONTEXT in pred:
            df['Cntx'] = self.w_tok.batch_decode(pred[Task.CONTEXT])

        if Task.RATING in pred:
            df['Rating'] = pred[Task.RATING]

        # QUESTION: Do we need to prefix these keys with "metadata-"?
        df['Items'] = self.mappers[I_COL].get_items(labels[f'metadata-{I_COL}'])
        df['User'] = self.mappers[U_COL].get_items(labels[f'metadata-{U_COL}'])
        df = pd.DataFrame.from_dict(df, orient='columns')
        if prediction_path is not None:
            df.to_json(prediction_path, orient="records", indent=2)
        else:
            logging.debug(df.to_json(orient="records", indent=2))

    def evaluate(self, labels, pred):
        self.logger.info(f'Computing performance metrics')
        # results.update({m: 0 for m in })
        for task, metric_fn in self.metric_fns.items():
            if task in labels:
                metric_fn(labels, pred)
            else:
                logging.warning(f'Task {task} appears as a model task but no labels are given for it!')

        if self.save_flag:
            save_results(self.results)
