import copy
import json
import logging
import math
import shutil
from types import SimpleNamespace

import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import funcs
from .constants import *


def load_config(config_file, args=None, prefix_path='', as_dict=False, ho_space=False):
    # Load and update model specific config
    with open(os.path.join(CONFIG_PATH, prefix_path, config_file), 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    model_config = {} if model_config is None else model_config

    # Load default config files
    config = model_config.copy()
    if not ho_space:
        for cfg_f in os.listdir(os.path.join(CONFIG_PATH, 'default')):
            if os.path.isfile(os.path.join(CONFIG_PATH, 'default', cfg_f)) and cfg_f.endswith('.yaml'):
                key = cfg_f.split('.')[0]
                with open(os.path.join(CONFIG_PATH, 'default', cfg_f), 'r') as f:
                    def_cfg = yaml.load(f, Loader=yaml.FullLoader)
                def_cfg = {} if def_cfg is None else def_cfg
                config[key] = funcs.update_dict(def_cfg, model_config.get(key, {}))

        for k in model_config:
            if k not in config:
                config[k] = model_config[k]

        if args:
            config['data']['dataset'] = args.dataset
            config['data']['fold'] = args.fold
            config['data']['test_flag'] = args.test
            config['data']['txt_len'] = TXT_LEN.get(args.dataset, 35)
            config['model']['txt_len'] = TXT_LEN.get(args.dataset, 35)
            config['gen_flag'] = not args.no_generate
            config['save_results'] = args.save_results
            config['seed'] = args.seed
            config['device'] = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

            cfg_model_name = (config['model']['name'] + config['model']['suffix']).lower()
            if cfg_model_name != config_file.split('.')[0].lower():
                logging.warning(f'Config. file name ({config_file.split(".")[0].lower()}) does not match model name+suffix '
                                f'specified inside ({cfg_model_name})')

            # NOTE: This may no longer be necessary
            if 'lm_model' in config['model']:
                config['model']['suffix'] = '-' + config['model'].get('lm_alias', config['model']['lm_model'].replace('/', '%')) + config['model']['suffix']

            if args.test:
                config['train']['epochs'] = 1
                config['gen_flag'] = False
                config['save_results'] = False

    if as_dict:
        return config

    return funcs.dict2namespace(config)


def load_partition(ixs_dir):
    assert os.path.exists(ixs_dir)
    train_ixs = sorted(np.load(os.path.join(ixs_dir, 'train.npy')).tolist())
    valid_ixs = sorted(np.load(os.path.join(ixs_dir, 'validation.npy')).tolist())
    test_ixs = sorted(np.load(os.path.join(ixs_dir, 'test.npy')).tolist())
    return {
        'train': train_ixs,
        'valid': valid_ixs,
        'test': test_ixs
    }


def print_exp_analysis(analysis):
    def dict2str(results, color=None):
        # return f"{{{', '.join([f'{colored(k, color)}: {v:.2f}'for k, v in results])}}}"
        out_str = f"{{{', '.join([f'{k}: {v}' for k, v in results.items()])}}}"
        if color is not None:
            out_str = colored(out_str, color)
        return out_str

    config_key = 'config'
    results_key = 'val'
    for tid, trial in analysis.results.items():
        if results_key in trial:
            if tid == analysis.best_trial.trial_id:
                print(f'{colored("Results:", Colors.BLUE)} {dict2str(trial[results_key], Colors.GREEN)}, '
                      f'{colored("Config:", Colors.BLUE)} {dict2str(trial[config_key], Colors.GREEN)}')
            else:
                print(f'{colored("Results:", Colors.BLUE)} {dict2str(trial[results_key])}, '
                      f'{colored("Config:", Colors.BLUE)} {dict2str(trial[config_key])}')
        else:
            print(f'Trial {tid} does not contain {results_key} argument')


def save_results(curr_res):
    filename = f'{DATA_MODE}_results.csv'
    is_new_file = not os.path.isfile(os.path.join(RES_PATH, filename))
    if not is_new_file:
        results = pd.read_csv(os.path.join(RES_PATH, filename))
    else:
        results = pd.DataFrame(columns=curr_res.keys())
    missing_cols = set(curr_res.keys()).difference(results.columns)
    for c in missing_cols:
        results[c] = np.nan
    missing_cols = set(results.columns).difference(curr_res.keys())
    for c in missing_cols:
        curr_res[c] = np.nan
    # results = results.append(pd.DataFrame().from_records([curr_res]))
    results = pd.concat((results, pd.DataFrame().from_records([curr_res])))
    results.to_csv(os.path.join(RES_PATH, filename), index=False)


def plot_mask(mask):
    plt.imshow(mask, cmap='Greys', interpolation='nearest')
    plt.xticks(np.arange(0, mask.shape[1], 1))
    plt.yticks(np.arange(0, mask.shape[0], 1))
    plt.grid()
    plt.show()


class Colors:
    BLACK = '\033[1;30m'
    RED = '\033[1;31m'
    GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[1;34m'
    PURPLE = '\033[1;35m'
    CYAN = '\033[1;36m'
    WHITE = '\033[1;37m'
    LIGHT_GRAY = '\033[0;37m'
    ENDC = '\033[0m'


def colored(text, color):
    return color + text + Colors.ENDC


def print_colored_dict(d):
    if isinstance(d, list):
        for subd in d:
            print_colored_dict(subd)
    else:
        print()
        for k, v in d.items():
            print(f'{k}: {v}')


def print_losses(loss, device, suffix=None, color=None, logger=None):
    losses = loss.copy()
    # for task in [Task.CONTEXT, Task.EXPLANATION]:
    #     if task in losses:
    #         losses[task] = math.exp(losses[task])

    out_str = ' | '.join([f'{task} {v:4.4f}' for task, v in losses.items()])
    if suffix is not None:
        out_str += f' | {suffix}'
    if device.type == 'cuda':
        out_str += f' | {funcs.get_gpu_usage(device)}'
    if color is not None:
        out_str = colored(out_str, color)

    if logger is None:
        logger = logging.getLogger(PROJECT_NAME)
    logger.info(out_str)


def print_namespace(ns, skipkeys=None):
    d = funcs.namespace2dict(ns)
    logging.info(json.dumps(d, indent=4, skipkeys=skipkeys))


def handle_dirs(args):
    if not os.path.exists(os.path.join(CKPT_PATH, DATA_MODE, 'best')):
        os.makedirs(os.path.join(CKPT_PATH, DATA_MODE, 'best'))
    if not os.path.exists(os.path.join(LOG_PATH, 'raw')):
        os.makedirs(os.path.join(LOG_PATH, 'raw'))
    if not os.path.exists(os.path.join(LOG_PATH, 'tensorboard')):
        os.mkdir(os.path.join(LOG_PATH, 'tensorboard'))
    if not os.path.exists(os.path.join(LOG_PATH, 'tmp')):
        os.mkdir(os.path.join(LOG_PATH, 'tmp'))
    if not os.path.exists(os.path.join(LOG_PATH, 'pred')):
        os.mkdir(os.path.join(LOG_PATH, 'pred'))

    if not os.path.exists(RES_PATH):
        os.makedirs(RES_PATH)

    # Dataset specific
    if not os.path.exists(os.path.join(DATA_PATHS[args.dataset], DATA_MODE, args.fold, 'tokenizers')):
        os.makedirs(os.path.join(DATA_PATHS[args.dataset], DATA_MODE, args.fold, 'tokenizers'))
    if not os.path.exists(os.path.join(DATA_PATHS[args.dataset], DATA_MODE, args.fold, 'aux')):
        os.makedirs(os.path.join(DATA_PATHS[args.dataset], DATA_MODE, args.fold, 'aux'))
    if not os.path.exists(os.path.join(DATA_PATHS[args.dataset], DATA_MODE, 'mappers')):
        os.makedirs(os.path.join(DATA_PATHS[args.dataset], DATA_MODE, 'mappers'))
