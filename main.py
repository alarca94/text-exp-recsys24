import argparse
import logging
import os
import sys
import warnings

import torch

warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
# NOTE: Comment this line the first time you run this code
# os.environ['HF_HUB_OFFLINE'] = 'true'

from src.models import BaseTrainer
from src.utils.in_out import load_config, print_namespace, handle_dirs, Colors, colored
from src.utils.funcs import now_time, init_seed
from src.utils.constants import *


def update_logger(args):
    if args.log_to_file:
        if not os.path.isdir(os.path.join(LOG_PATH, 'raw')):
            os.makedirs(os.path.join(LOG_PATH, 'raw'))
        mode = 'a' if args.eval_only or args.resume_training else 'w'
        log_f = f'{args.dataset}_{args.fold}_{args.model_name}_{args.seed}_{DATA_MODE}.txt'
        handlers = [logging.FileHandler(filename=os.path.join(LOG_PATH, 'raw', log_f), mode=mode)]
    else:
        handlers = [logging.StreamHandler(stream=sys.stdout)]

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s - %(message)s',
                        handlers=handlers)


def main(args):
    update_logger(args)

    cfg = load_config(f'{args.model_name}.yaml', args)

    if not args.resume_training:
        # Set the random seed manually for reproducibility.
        init_seed(args.seed, reproducibility=True)
    else:
        logging.info(colored('-' * 40 + ' RESUME TRAINING ' + '-' * 40, Colors.BLUE))
        logging.warning('--resume-training does not reset the RNG state to one of the last checkpoint, thus, results'
                        ' will not be reproducible')

    device = torch.device('cpu')
    if torch.cuda.is_available():
        if not args.cuda:
            logging.warning(now_time() + 'You have a CUDA device, so you should probably run with --cuda')
        else:
            device = torch.device('cuda')

    handle_dirs(args)

    # NOTE: the pred_file name does not contain the RNG_SEED as we only need a single output for qualitative analysis
    cfg_model_name = (cfg.model.name + cfg.model.suffix).lower()
    pred_file = f'{args.dataset}_{args.fold}_{cfg_model_name}_{args.seed}_{DATA_MODE}.txt'
    prediction_path = os.path.join(LOG_PATH, 'pred', pred_file)

    logging.info('-' * 40 + ' ARGUMENTS ' + '-' * 40)
    print_namespace(cfg)
    logging.info('-' * 40 + ' ARGUMENTS ' + '-' * 40)

    trainer = BaseTrainer(cfg, device, prediction_path, args.eval_only, args.resume_training)

    if not os.path.exists(trainer.ckpt_path) and (args.eval_only or args.resume_training):
        logging.warning(f'Pretrained model checkpoint not found with path: {trainer.ckpt_path}. Training from scratch')
        args.eval_only = False
        args.resume_training = False

    if not args.eval_only:
        trainer.train(resume=args.resume_training)
    if not args.train_only:
        trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text-based Explainable Recommender Systems')
    parser.add_argument('--model-name', type=str, default='peter',
                        help='Name of the model configuration (peter, sequer, etc)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset name (amazon_movies, yelp, tripadvisor)')
    parser.add_argument('--fold', type=str, default="0",
                        help='data partition index')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--test', action='store_true',
                        help='use a small fraction of the data')
    parser.add_argument('--no-save-results', dest="save_results", default=True, action='store_false',
                        help='whether results will be saved or not')
    parser.add_argument('--no-generate', action='store_true',
                        help='whether generated text will be saved to log file or not')
    parser.add_argument('--train-only', action='store_true',
                        help='Whether to train a model and skip evaluation on the test set')
    parser.add_argument('--eval-only', action='store_true',
                        help='Whether to load an existing checkpoint and evaluate it (skipping training)')
    parser.add_argument('--resume-training', action='store_true',
                        help='Whether to resume training')
    parser.add_argument('--log-to-file', action='store_true',
                        help='Whether to load an existing checkpoint and evaluate it')
    parser.add_argument('--seed', type=int, default=RNG_SEED,
                        help='seed for reproducibility')

    args = parser.parse_args()
    if args.dataset is None:
        parser.error('--dataset should be provided for loading data')
    elif args.dataset not in DATA_PATHS:
        parser.error(
            f'--dataset supported values are: {", ".join(list(DATA_PATHS.keys()))} -- Provided value: {args.dataset}')

    if args.resume_training and args.eval_only:
        parser.error('Set one of --resume-training or --eval-only to True.')

    main(args)
