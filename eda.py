import importlib
import os
import re
from ast import literal_eval

import torch

from process_data import read_data, complete_cfg, DebugDataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import nltk
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.constants import *
from src.utils.in_out import colored, Colors, load_config


def get_datasets():
    datasets = {}
    for d, path in DATA_PATHS.items():
        if 'amazon' not in d:
            data, split_ixs, mappers = read_data(d, fold=0, data_file='reviews.pkl')
            datasets[d] = (data, split_ixs, mappers)
    return datasets


def camel_case(s):
    return re.sub('[_-]+', ' ', s).title()


def get_words(s):
    ws = nltk.word_tokenize(s)
    ws = [w.lower() for w in ws if w.isalpha() and w not in set(stopwords.words('english'))]
    return ws


def basic_stats():
    datasets = get_datasets()
    stats = []
    for d, (data, _, mappers) in datasets.items():
        stats.append({'dataset': camel_case(d)})
        stats[-1]['#Users'] = data[U_COL].nunique()
        stats[-1]['#Items'] = data[I_COL].nunique()
        stats[-1]['#Reviews'] = data.shape[0]
        stats[-1]['#Features'] = len(mappers[FEAT_COL])
        stats[-1]['#Explanations'] = len(mappers[EXP_COL])
        stats[-1]['#Reviews / User'] = data[U_COL].value_counts().values.mean()
        sent_lens = mappers[EXP_COL].idx2item['raw'].apply(nltk.word_tokenize).str.len()
        stats[-1]['Avg. #Words / Explanation'] = sent_lens.mean()
        stats[-1]['Max. #Words / Explanation'] = sent_lens.max()
        stats[-1]['Min. #Words / Explanation'] = sent_lens.min()
        fe_data = data[[FEAT_COL, EXP_COL]].explode([FEAT_COL, EXP_COL]).drop_duplicates(subset=EXP_COL, keep='first')
        stats[-1]['Avg. #Features / Explanation'] = fe_data[FEAT_COL].apply(lambda s: list(set(s))).str.len().mean()
        stats[-1]['Avg. #Exps / Interaction'] = data[EXP_COL].str.len().mean()
        stats[-1]['95\% Exp. Length'] = (sent_lens.value_counts().sort_index().cumsum() / len(sent_lens)).searchsorted(0.95)
        stats[-1]['#Sparsity'] = stats[-1]['#Reviews'] / (stats[-1]['#Items'] * stats[-1]['#Users']) * 100

    stats = pd.DataFrame.from_records(stats).set_index('dataset', drop=True)
    stats = stats.applymap(lambda x: str.format("{:0_.2f}", x).replace('_', ',').replace('.00', ''))
    print(stats.T.to_latex())


def check_text_length():
    import nltk

    datasets = ['tripadvisor', 'yelp', 'ratebeer']
    threshold, ceil_base = 0.95, 5
    for dataset in datasets:
        data, split_ixs, mappers = read_data(dataset)

        exp_lens = mappers[EXP_COL].idx2item['raw'].apply(nltk.word_tokenize).str.len()
        txt_len = (exp_lens.value_counts().sort_index().cumsum() / len(exp_lens)).searchsorted(threshold)
        txt_len = int(np.ceil(txt_len / ceil_base) * ceil_base)
        print(f'Text length for {dataset} should be: {txt_len}')


def feature_hallucination(recompute=False):
    import seaborn as sns
    from src.utils.evaluation import feature_detect

    def count_groups(row):
        gs = []
        valid_fs = i2f.loc[row[I_COL]]
        for i in range(len(g_f)):
            gs.append(len(g_f[i].difference(valid_fs)))
        return gs

    # Group features based on occurrence
    # Compute the prediction ratio per group
    # Compute the hallucination ratio per group
    # Plot them
    n_groups = 4
    g_cols = list(range(n_groups))
    fold = 0
    module_path = 'src.models'
    model_module = importlib.import_module(module_path)
    seeds = (1111, 24, 53, 126)
    datasets = ('ratebeer', 'tripadvisor', 'yelp')
    models = ('nrt', 'peter', 'sequer-best', 'pod', 'escofilt', 'greener', 'erra', 'exbert')  # , 'escofilt', 'greener-noilp', 'exbert-naive')
    models_n = {}
    # fig = plt.figure(constrained_layout=True)
    # fig, axes = plt.subplots(nrows=len(datasets), ncols=2, sharex=True)
    # dmap = {'yelp': 'Yelp', 'ratebeer': 'RateBeer', 'tripadvisor': 'TripAdvisor'}
    if os.path.isfile(os.path.join(LOG_PATH, 'tmp', 'yelp_hallu_pop.csv')) and not recompute:
        res = pd.read_csv(os.path.join(RES_PATH, f'{DATA_MODE}_merged_results.csv'))
        res['model_name'] = res['model_name'].str.lower()
        res.date = pd.to_datetime(res.date, format='%Y-%m-%dT%H:%M:%SZ')
        res = res[res.split_ix == fold]
        res.sort_values('date', ascending=False, inplace=True)
        # unique_key = ['model_name', 'dataset', 'split_ix', 'seed']
        # res = res.drop_duplicates(subset=unique_key, keep='first', ignore_index=True)

        # norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=1000)
        for model in models:
            config = load_config(f'{model}.yaml')  # .stages[0]
            model_n = (config.model.name + config.model.suffix).replace('-best', '')
            if model not in models_n:
                models_n[model] = model_n

        fig, axes = plt.subplots(nrows=len(datasets), ncols=2, sharex='col', sharey='row', figsize=(10, 8))  # , gridspec_kw={'width_ratios': [5, 4]}
        dmap = {'yelp': 'Yelp', 'ratebeer': 'RateBeer', 'tripadvisor': 'TripAdvisor'}
        for di, dataset in enumerate(datasets):
            config = load_config(f'peter.yaml')  # .stages[0]
            model_c = getattr(model_module, config.model.name)
            complete_cfg(config, dataset=dataset, model_tasks=model_c.TASKS, seq_mode=model_c.SEQ_MODE, test_flag=False)

            g_pop = pd.read_csv(os.path.join(LOG_PATH, 'tmp', f'{dataset}_hallu_pop.csv'))
            # axes[di, 0].set_ylabel(dmap[dataset])
            # mapper = {f'{m}_hallu': m for m in ['Baseline'] + list(models_n.values())}
            mapper = {f'{m}_hallu': m for m in list(models_n.values())}
            mapper_base = {f'{m}_unbiased': m for m in list(models_n.values())}
            cbar = False
            cbar_kws_1, cbar_kws_2 = {}, {}
            if di == 0:
                cbar_kws_1 = {'location': 'top', 'use_gridspec': False, 'label': r'$\bf{Hallucination\ bias}$',
                              }  # "ticks": [-100, -75, -50, -25, 0, 250, 500, 750, 1000]}
                cbar_kws_2 = {'location': 'top', 'use_gridspec': False, 'label': r'$\bf{Popularity\ Bias}$'}
                cbar = True

            # ESCOFILT shouldn't be able to hallucinate as explanations are taken from past item reviews
            g_pop[f'ESCOFILT_hallu'] = float('nan')
            sa_pop = g_pop[list(mapper.keys())].rename(mapper, axis=1).T
            sa_pop = (sa_pop - g_pop[list(mapper_base.keys())].values.T) / g_pop[list(mapper_base.keys())].values.T

            sa = sns.heatmap(sa_pop * 100, annot=True, cmap="vlag_r", fmt='.3g', vmin=-100, vmax=100,
                             ax=axes[di, 0], cbar_kws=cbar_kws_1, cbar=cbar, linewidth=.5, linecolor='0.75')

            mapper = {f'{m}_bias': m for m in list(models_n.values())}
            sb = sns.heatmap(g_pop[list(mapper.keys())].rename(mapper, axis=1).T * 100, annot=True, cmap="vlag_r", fmt='.3g',
                             vmin=-100, vmax=100, ax=axes[di, -1], cbar_kws=cbar_kws_2, cbar=cbar, linewidth=.5, linecolor='0.75')

            sa.set(ylabel=r'$\bf{' + dmap[dataset] + '}$')
            if di == len(datasets) - 1:
                sa.set_xlabel('Popularity Groups')
                sb.set_xlabel('Popularity Groups')

        plt.savefig('images/fhallu_norm_train_base.pdf', format="pdf", bbox_inches='tight')
        exit()

    for di, dataset in enumerate(datasets):

        config = load_config(f'peter.yaml')  # .stages[0]
        model_c = getattr(model_module, config.model.name)
        complete_cfg(config, dataset=dataset, model_tasks=model_c.TASKS, seq_mode=model_c.SEQ_MODE, test_flag=False)

        data = DebugDataset(config.data, config.model, torch.device('cuda:0'))
        labels = data.get_gen_labels(0, len(data), data.batch_size, [Task.EXPLANATION], 'test')

        if len(set(sum(labels[Task.FEAT], start=[]))) != len(data.mappers[FEAT_COL]):
            print(colored(f'Not all features are covered in the test set', Colors.RED))

        tst_feats = pd.Series(labels[Task.FEAT])
        # Compute popularity of features in train set
        unrolled = data.data.loc[data.split_ixs['train'], [I_COL, FEAT_COL]].explode(FEAT_COL).explode(FEAT_COL)
        f2i_trn = unrolled.groupby(FEAT_COL)[I_COL].agg(set)
        unrolled = data.data[[I_COL, FEAT_COL]].explode(FEAT_COL).explode(FEAT_COL)
        f2i = unrolled.groupby(FEAT_COL)[I_COL].agg(set)

        # Feature Popularity as a measure of item popularity
        fpop = f2i_trn.apply(len) / data.data[I_COL].nunique()
        fpop.sort_values(ascending=False, inplace=True)
        fpop = fpop.to_frame(name='trn_pop')
        fpop['rank'] = range(fpop.shape[0])

        tst_if = tst_feats.to_frame(name='gt_f')
        tst_if[I_COL] = labels[f'metadata-{I_COL}']

        # Compute hallucination and hit ratios
        i2f = unrolled.groupby(I_COL)[FEAT_COL].agg(set)
        i2f = i2f.loc[tst_if[I_COL].unique()]

        fpop['tst_pop'] = 0
        fpop['tst_pop'].update(
            tst_if[[I_COL, 'gt_f']].explode('gt_f').dropna().groupby('gt_f')[I_COL].agg(set).apply(len) /
            tst_if[I_COL].nunique())

        fpop['tst_r_pop'] = 0
        fpop['tst_r_pop'].update(tst_if['gt_f'].explode().dropna().value_counts() / tst_if.shape[0])

        # Feature Popularity as a measure of review occurrence
        tmp = data.data.loc[data.split_ixs['train'], [I_COL, FEAT_COL]]
        i_occ = tmp[I_COL].value_counts()
        tmp[FEAT_COL] = tmp[FEAT_COL].apply(lambda ll: list(set(sum(ll, start=[]))))  # .explode(FEAT_COL)
        tmp2 = tmp.explode(FEAT_COL).groupby([I_COL, FEAT_COL]).agg(len).to_frame('freq').reset_index()

        tmp2['freq'] = tmp2['freq'] / tmp2[I_COL].map(i_occ)
        fpop['trn_r_pop'] = 0
        fpop['trn_r_pop'].update(tmp[FEAT_COL].explode().value_counts() / len(data.split_ixs['train']))

        # Maximum Feature frequency at item level (maximum ratio of reviews containing this feature for an item)
        fpop['trn_max_ffreq'] = 0
        fpop['trn_max_ffreq'].update(tmp2.groupby(FEAT_COL)['freq'].agg('max'))

        fpop.sort_values('trn_r_pop', inplace=True, ascending=False)
        fpop['r_rank'] = range(fpop.shape[0])

        # Compute feature groups on feature review popularity
        g_f = pd.qcut(fpop['r_rank'], n_groups, labels=False)
        g_f = g_f.to_frame('group').reset_index().groupby('group')[FEAT_COL].agg(set)[list(range(n_groups))].to_dict()
        f_g = pd.Series(sum([[i] * v for i, v in enumerate(map(len, g_f.values()))], start=[]),
                        index=fpop.index.values)

        # Compute the feature group popularity at item level as the feature ratio of each group (in the training set)
        trn_df = data.data.loc[data.split_ixs['train'], [FEAT_COL]]
        trn_df[FEAT_COL] = trn_df[FEAT_COL].apply(lambda l: list(set(sum(l, start=[]))))
        trn_df = trn_df.explode(FEAT_COL)
        trn_df['group'] = trn_df[FEAT_COL].map(f_g).fillna(-1).astype(int)

        g_occ = trn_df[['group']].reset_index(names='index').pivot_table(index='index', columns='group',
                                                                         aggfunc=len, fill_value=0)
        if trn_df.isna().values.any():
            g_occ = g_occ.drop(-1, axis=1)
        g_occ = g_occ.reindex(columns=g_cols, fill_value=0)
        g_occ = (g_occ / g_occ.values.sum(axis=1, keepdims=True))

        # Group popularity in the training set
        g_pop = g_occ.mean(axis=0).to_frame(name='train')

        tst_df = tst_if[['gt_f']].explode('gt_f').dropna()
        tst_df['group'] = tst_df['gt_f'].map(f_g)

        g_occ = tst_df[['group']].reset_index(names='index').pivot_table(index='index', columns='group',
                                                                         aggfunc=len, fill_value=0)
        g_occ = g_occ.reindex(columns=g_cols, fill_value=0)
        g_occ = (g_occ / g_occ.values.sum(axis=1, keepdims=True))
        g_pop['test'] = g_occ.mean(axis=0)  # .to_frame(name='test')

        dist = np.array(tst_if.apply(count_groups, axis=1).tolist())
        trn_pop = g_pop[['train']].loc[list(range(n_groups))].values.T
        dist = dist * trn_pop
        dist = np.divide(dist, dist.sum(axis=1, keepdims=True))
        g_pop['Baseline_hallu'] = dist.mean(0)
        g_pop['Baseline_unbiased'] = dist.mean(0)
        g_pop['Baseline_bias'] = 0

        for model in models:
            config = load_config(f'{model}.yaml')  # .stages[0]
            model_n = (config.model.name + config.model.suffix).replace('-best', '')
            if model not in models_n:
                models_n[model] = model_n

            for seed in seeds:
                pred_f = f'logs/pred/{dataset}_{fold}_{model}_{seed}_{DATA_MODE}.txt'
                print(colored(f'Processing {pred_f}', Colors.GREEN))
                if not os.path.isfile(pred_f):
                    print(colored(f'Unable to find {pred_f}', Colors.RED))
                    continue

                d = pd.read_json(pred_f)
                d['True'] = d['True'].apply(literal_eval)

                tst_if['pred'] = list(map(data.mappers[FEAT_COL].get_idxs,
                                          map(list, feature_detect(d['Pred'].tolist(), data.feature_set))))

                # NOTE: It is probably useful to check that no new features are used in the test (unseen during training)
                fpop[f'tst_{model_n}'] = 0
                fpop[f'tst_{model_n}'].update(
                    tst_if[[I_COL, 'pred']].explode('pred').dropna().groupby('pred')[I_COL].agg(set).apply(len) /
                    tst_if[I_COL].nunique())

                print(f'Number of features predicted by the model: {(fpop[f"tst_{model_n}"] != 0).sum()} / {fpop.shape[0]}')

                # Bias as a difference
                fpop[f'bias_{model_n}'] = fpop[f'tst_{model_n}'] - fpop['trn_pop']
                # Bias as a ratio -- Ratio for unpredicted features is infinity
                fpop[f'r_bias_{model_n}'] = fpop['trn_pop'] / fpop[f'tst_{model_n}']

                fpop[f'tst_r_{model_n}'] = 0
                fpop[f'tst_r_{model_n}'].update(tst_if['pred'].explode().dropna().value_counts() / tst_if.shape[0])

                # Bias as a difference
                fpop[f'bias_r_{model_n}'] = fpop[f'tst_r_{model_n}'] - fpop['trn_r_pop']
                # Bias as a ratio -- Ratio for unpredicted features is infinity
                fpop[f'r_bias_r_{model_n}'] = fpop['trn_pop'] / fpop[f'tst_{model_n}']

                tst_if['hallu'] = tst_if.apply(lambda r: list(set(r['pred']).difference(i2f.loc[r[I_COL]])), axis=1)
                mask = (tst_if['hallu'].str.len() > 0).values

                fpop['hallu'] = 0
                # tst_if.apply(lambda r: list(set(r['pred']).difference(i2f.loc[r[I_COL]])), axis=1)
                fpop['hallu'].update(tst_if['hallu'].explode().dropna().value_counts())
                # hallu_fs = fpop.index.values[fpop['hallu'] > 0]
                fpop['hallu_norm'] = 1
                fpop['hallu_norm'].update(
                    pd.Series({f: sum(~tst_if[I_COL].isin(f2i[f])) for f in set(tst_feats.explode().tolist())}))
                fpop[f'hallu_{model_n}'] = fpop['hallu'] / fpop['hallu_norm']
                fpop.drop(['hallu', 'hallu_norm'], axis=1, inplace=True)

                def f_hitrate(gdata):
                    assert isinstance(gdata, pd.DataFrame)
                    hits = (gdata['pred'] == gdata['gt_f']).groupby(level=0).agg(any)
                    return hits.sum() / len(hits)

                fpop[f'hit_{model_n}'] = float('nan')
                fpop[f'hit_{model_n}'].update(
                    tst_if[['pred', 'gt_f']].explode('gt_f').explode('pred').groupby('gt_f')[['pred', 'gt_f']].apply(
                        f_hitrate))

                # Aggregate results over all seeds
                tst_df = tst_if[['pred']].explode('pred').dropna()
                tst_df['group'] = tst_df['pred'].map(f_g)

                g_occ = tst_df[['group']].reset_index(names='index').pivot_table(index='index', columns='group',
                                                                                 aggfunc=len, fill_value=0)
                g_occ = g_occ.reindex(columns=g_cols, fill_value=0)
                g_occ = (g_occ / g_occ.values.sum(axis=1, keepdims=True))
                if model_n in g_pop.columns:
                    g_pop[model_n] += g_occ.mean(axis=0)  # .to_frame(name=model_n)
                else:
                    g_pop[model_n] = g_occ.mean(axis=0)  # .to_frame(name=model_n)

                tst_df = tst_if[['hallu']].explode('hallu').dropna()
                tst_df['group'] = tst_df['hallu'].map(f_g)

                g_occ = tst_df[['group']].reset_index(names='index').pivot_table(index='index', columns='group',
                                                                                 aggfunc=len, fill_value=0)
                g_occ = g_occ.reindex(columns=g_cols, fill_value=0)
                g_occ = (g_occ / g_occ.values.sum(axis=1, keepdims=True))

                if f'{model_n}_hallu' in g_pop.columns:
                    g_pop[f'{model_n}_hallu'] += g_occ.mean(axis=0)  # .to_frame(name=f'{model_n}_hallu')
                    g_pop[f'{model_n}_unbiased'] += dist[mask].mean(0)
                else:
                    g_pop[f'{model_n}_hallu'] = g_occ.mean(axis=0)  # .to_frame(name=f'{model_n}_hallu')
                    g_pop[f'{model_n}_unbiased'] = dist[mask].mean(0)

            g_pop[model_n] /= len(seeds)
            g_pop[f'{model_n}_hallu'] /= len(seeds)
            g_pop[f'{model_n}_bias'] = (g_pop[model_n] - g_pop['train']) / g_pop['train']
            g_pop[f'{model_n}_unbiased'] /= len(seeds)

        g_pop.to_csv(os.path.join(LOG_PATH, 'tmp', f'{dataset}_hallu_pop.csv'), index=False)

    # plt.savefig('images/fhallu.png', bbox_inches='tight')
    feature_hallucination(recompute=False)
    print('Finished')


# def check_examples():
#     data_mode = 'feat_only'
#     dataset = 'amazon-beauty'
#     tok = '_bpe'
#     usef = ''
#     baselines = [f'peter-l{usef}{tok}', f'nrt{tok}', f'att2seq{tok}', f'pepler{usef}']
#     tgt_models = [f'sawer-lm-google-flan-t5-base{usef}']  # f'sawer-fmlp-l-bce{usef}{tok}']
#     models = baselines + tgt_models
#
#     fs = [f for f in os.listdir(os.path.join(CKPT_PATH, data_mode)) if dataset in f and 'generated' in f]
#     lookup_fs = {m: f for f in fs for m in models if f.endswith(m + '.txt')}
#     assert len(models) == len(lookup_fs)
#
#     # Load Review data
#     data_path = os.path.join(DATA_PATHS[dataset], DATA_MODE, 'reviews_new.pickle')
#     index_dir = os.path.join(DATA_PATHS[dataset], DATA_MODE, '0')
#     data = pd.DataFrame.from_records(pd.read_pickle(data_path))
#     ixs = load_partition(index_dir)
#     data = data.iloc[ixs['test']]
#     data.rename({EXP_COL: 'GT'}, axis=1, inplace=True)
#     # print(data[RAT_COL].mean())
#
#     data['GT'] = Colors.BLUE + data['GT'] + Colors.ENDC
#     feat_avg_rat = data[[FEAT_COL, RAT_COL]].groupby(FEAT_COL).mean().applymap('{:.2f}'.format)
#     data[f'Avg-{FEAT_COL}'] = colored('(' + feat_avg_rat.loc[data[FEAT_COL]] + ')', Colors.LIGHT_GRAY).values
#     docs = data[[U_COL, I_COL, HIST_FEAT_COL, f'Avg-{FEAT_COL}', FEAT_COL, RAT_COL, 'GT']].to_dict(orient='records')
#
#     # Create Docs structure with target model
#     for m, f in lookup_fs.items():
#         d = pd.read_json(os.path.join(CKPT_PATH, data_mode, f), orient="records")
#         if 'Rating' not in d.columns:
#             d['Rating'] = -1
#         for i, r in d[['Pred', 'Rating']].iterrows():
#             # If hist. feature in review, color it with Yellow
#             for f in set(docs[i][HIST_FEAT_COL]):
#                 r['Pred'] = r['Pred'].replace(f, colored(f, Colors.YELLOW))
#
#             # If cand. feature in review, color it with Red
#             text = r['Pred'].replace(docs[i][FEAT_COL], colored(docs[i][FEAT_COL], Colors.RED))
#             docs[i][m.upper()] = f'{text} ' + colored(f'(Rating: {r["Rating"]:.2f})', Colors.LIGHT_GRAY)
#
#     for i in range(len(docs)):
#         docs[i][FEAT_COL] += f' {docs[i][f"Avg-{FEAT_COL}"]}'
#         docs[i].pop(f"Avg-{FEAT_COL}")
#
#     i = 0
#     print_colored_dict(docs[i])
#     while input('\nDo you want to print the next doc? (y/n)\n') == 'y' and i < len(docs) - 1:
#         i += 1
#         print_colored_dict(docs[i])


# def transformer_analysis(attn_flag=True, logits_flag=False):
#     import torch
#     from bertviz import head_view, model_view
#     from src.utils.data import GenDataset
#     import seaborn as sns
#     sns.set()
#
#     args = SimpleNamespace(**{
#         'dataset': 'amazon-toys',
#         'model_name': f'sawer',
#         'model_suffix': f'-l+_bpe',
#         'seed': RNG_SEED,
#         'fold': '0',
#         'hist_len': DEFAULT_HIST_LEN
#     })
#
#     pred_file = f'{args.dataset}_{args.fold}_{args.model_name}{args.model_suffix}_{DATA_MODE}.txt'
#     prediction_path = os.path.join(LOG_PATH, 'pred', pred_file)
#
#     with open(os.path.join(CONFIG_PATH, f'{args.model_name}{args.model_suffix}.yaml'), 'r') as f:
#         cfg = yaml.load(f, yaml.FullLoader)
#
#     # Set the random seed manually for reproducibility.
#     init_seed(args.seed, reproducibility=True)
#     device = torch.device('cpu')
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#
#     seq_mode = SeqMode(cfg.get('seq_mode', 0))
#     if seq_mode.requires_past_info():
#         f_name = f'{args.model_name}{args.model_suffix}_{args.dataset}_{args.fold}_{args.hist_len}_{args.seed}'
#         model_path = os.path.join(CKPT_PATH, DATA_MODE, f'{f_name}.pth')
#     else:
#         f_name = f'{args.model_name}{args.model_suffix}_{args.dataset}_{args.fold}_{args.seed}'
#         model_path = os.path.join(CKPT_PATH, DATA_MODE, f'{f_name}.pth')
#
#     tok = cfg.get('tokenizer', 'default')
#     use_segment_ids = (args.model_name.lower() in ['sequer_v2', 'sequer_v3', 'sequer_v4'])
#     use_feature = cfg.get('use_feature', False)
#     mod_context_flag = cfg.get('mod_context_flag', False)
#     sent_emb_model = cfg.get('sent_embedder', 'all-MiniLM-L6-v2')
#     hist_len = cfg.get('hist_len', DEFAULT_HIST_LEN)
#     data = GenDataset(args.dataset, args.fold, None, seq_mode, word_tokenizer=tok, add_segment_ids=use_segment_ids,
#                       batch_size=128, mod_context_flag=mod_context_flag, batch_first=False,
#                       sent_emb_model=sent_emb_model, hist_len=hist_len)
#     predictions = pd.read_json(prediction_path, orient="records")
#     _, _, tst_data = data.get_dataloaders()
#
#     with open(model_path, 'rb') as f:
#         model = torch.load(f, map_location=device)
#
#     model.eval()
#     with torch.no_grad():
#         for bix, batch in enumerate(tst_data):
#             batch.pop(NEG_EVAL_COL, None)
#
#             if Task.MLM in model.tasks:
#                 batch[I_COL] = torch.cat((batch[I_COL], batch[I_COL][-1:]), dim=0)
#                 batch[I_COL][-2] = data.toks[I_COL].mask_token_id
#
#             batch[EXP_COL] = batch[EXP_COL][:-1]
#             if use_feature:
#                 batch[EXP_COL] = torch.cat([batch[FEAT_COL], batch[EXP_COL]], 0)  # (src_len + tgt_len - 2, batch_size)
#                 if use_segment_ids:
#                     f_seg_id = torch.ones_like(batch[FEAT_COL]) * batch[I_COL][-1:]
#                     batch[SEG_EXP_COL] = torch.cat([f_seg_id, batch[SEG_EXP_COL]], 0)
#             for k, v in batch.items():
#                 batch[k] = v.to(device)
#
#             res = model(batch)
#             attention = res['attns'].detach().cpu()
#             sentence_b_start = batch[I_COL].shape[0] + batch[U_COL].shape[0]
#
#             for k, v in batch.items():
#                 batch[k] = v.detach().cpu()
#
#             for six in range(batch[U_COL].shape[1]):
#                 item_tokens = data.toks[I_COL].convert_ids_to_tokens(batch[I_COL][:, six])
#                 user_token = data.toks[U_COL].batch_decode(batch[U_COL][:, six].numpy()).tolist()
#                 exp_tokens = data.toks[EXP_COL].convert_ids_to_tokens(batch[EXP_COL][:, six])
#
#                 tokens = item_tokens + user_token + exp_tokens
#
#                 # fig, axes = plt.subplots(1, attention.shape[0] + 1, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1, 0.08]})
#                 # axes[0].get_shared_y_axes().join(*axes[1:])
#                 # axes[0].set_yticks(range(len(tokens)))
#                 # axes[0].set_yticklabels(tokens)
#                 # for i, att in enumerate(attention[:, six]):
#                 #     axes[i].set_title(f'Layer {i}')
#                 #     sns.heatmap(att.squeeze(0), ax=axes[i], linewidth=0.5)
#                 #     axes[i].set_xticks(range(len(tokens)))
#                 #     axes[i].set_xticklabels(tokens, rotation=90)
#                 #
#                 # plt.show()
#
#                 # out = head_view(attention[:, six].unsqueeze(1).unsqueeze(2), tokens, sentence_b_start,
#                 #                 html_action='return')
#                 assert False


if __name__ == '__main__':
    feature_hallucination()
