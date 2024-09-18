import os

from enum import Enum, auto

DATE_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
PROJECT_NAME = 'text_exp'

BASE_PATH = f"{os.path.expanduser('~')}/text-exp-recsys24/"
DATA_MODE = 'feat_only'
DATA_PATHS = {
    'yelp': os.path.join(BASE_PATH, 'data/Yelp'),
    'tripadvisor': os.path.join(BASE_PATH, 'data/TripAdvisor'),
    'ratebeer': os.path.join(BASE_PATH, 'data/RateBeer'),
}
WE_PATH = os.path.join(BASE_PATH, 'data', 'word_embeddings')
RES_PATH = os.path.join(BASE_PATH, 'results')
CKPT_PATH = os.path.join(BASE_PATH, 'checkpoints')
LOG_PATH = os.path.join(BASE_PATH, 'logs')
CONFIG_PATH = 'config'

CKPT_EXT = '.ckpt'

UNK_TOK = '<unk>'
PAD_TOK = '<pad>'
BOS_TOK = '<bos>'
EOS_TOK = '<eos>'
SEP_TOK = '<sep>'
MASK_TOK = '<mask>'
CLS_TOKEN = '<cls>'
SPECIAL_TOKENS = {
    'pad_token': PAD_TOK,
    'bos_token': BOS_TOK,
    'eos_token': EOS_TOK,
    'unk_token': UNK_TOK,
    'sep_token': SEP_TOK,
    'mask_token': MASK_TOK,
    'cls_token': CLS_TOKEN
}

ITEM_PREFIX = 'item_'
USER_PREFIX = 'user_'
TEST_SUFFIX = '_mini'

FEAT_COL = 'feature'
ADJ_COL = 'adj'
ASP_COL = 'aspect'
SCO_COL = 'sco'
REV_COL = 'review'
EXP_COL = 'text'
U_COL = 'user'
I_COL = 'item'
SEQ_LEN_COL = 'seq_length'
RAT_COL = 'rating'
TIMESTAMP_COL = 'timestamp'
TIMESTEP_COL = 'timestep'
HIST_PREFIX = 'hist'
EMB_PREFIX = 'emb'
U_HIST_I_COL = f'{U_COL}_{HIST_PREFIX}_{I_COL}'
U_HIST_FEAT_COL = f'{U_COL}_{HIST_PREFIX}_{FEAT_COL}'
U_HIST_EXP_COL = f'{U_COL}_{HIST_PREFIX}_{EXP_COL}'
U_HIST_RAT_COL = f'{U_COL}_{HIST_PREFIX}_{RAT_COL}'
I_HIST_EXP_COL = f'{I_COL}_{HIST_PREFIX}_{EXP_COL}'
I_HIST_FEAT_COL = f'{I_COL}_{HIST_PREFIX}_{FEAT_COL}'
HIST_FEAT_COL = f'{HIST_PREFIX}_{FEAT_COL}'
HIST_EXP_COL = f'{HIST_PREFIX}_{EXP_COL}'
REV_LEN_COL = EXP_COL + 'length'
CONTEXT_COL = 'context'
MASK_COL = 'mask'
POS_OPT_COL = 'pos_answer'
NEG_OPT_COL = 'neg_answer'
NEG_EVAL_COL = 'neg_samples'

RNG_SEED = 1111
ALL_SEEDS = [1111, 24, 53, 126, 675]

HIST_I_MODE = 1
HIST_REV_MODE = 3

DEFAULT_HIST_LEN = 10
MAX_HIST_LEN = 400

# NOTE: To get the text length per dataset, run check_text_length() from eda.py
TXT_LEN = {
    'tripadvisor': 45,
    'yelp': 35,
    'ratebeer': 25
}
TOP_KS = [1, 5, 10, 20]
VOCAB_SIZE = 5000

ENC_DEC_MODELS = ['t5']

METRICS = ['RMSE', 'MAE', 'HR@20', 'MRR@20', 'NDCG@20', 'DIV', 'FCR', 'FP', 'FR', 'FMR', 'FHR', 'D-FHR', 'USR', 'USN',
           'avg_len', 'w_idf', 'rep/l', 'seq_rep_2', 'BLEU-1', 'BLEU-2', 'BLEU-4',
           'rouge_1/f_score', 'rouge_1/p_score', 'rouge_1/r_score', 'rouge_2/f_score', 'rouge_2/p_score',
           'rouge_2/r_score', 'rouge_l/f_score', 'rouge_l/p_score', 'rouge_l/r_score', 'BERTScore/p_score',
           'BERTScore/r_score', 'BERTScore/f_score', 'PROR', 'PSOS', 'Cand2Ref_RMSE', 'Cand2True_RMSE',
           'Cand2Pred_RMSE', 'Ref2True_RMSE']


class SeqMode(Enum):
    NO_SEQ = auto()
    # Get Item interaction history
    HIST_ITEM = auto()
    # Get Item interaction history but rating prediction only occurs for the cand.item
    HIST_ITEM_RATING_LAST = auto()
    # Get Item interaction history + User Review History
    HIST_ITEM_U_EXP = auto()
    # Get Item interaction history + User Review History (in the form of sentence embeddings)
    HIST_ITEM_U_EXP_EMB = auto()
    # Get Item interaction history (in the form of corresponding review sentence embeddings)
    HIST_REV2ITEM_EMB = auto()
    # Get User Review History + Item Review History
    HIST_UI_EXP = auto()

    # # Get User Review History + Item Review History (in the form of sentence embeddings)
    # HIST_ITEM_UI_EXP_EMB = 7

    def requires_past_info(self) -> bool:
        return self.value >= self.HIST_ITEM.value

    def requires_item_seq(self) -> bool:
        return self.requires_past_info() and self.value >= self.HIST_UI_EXP.value

    def requires_past_user_exp(self) -> bool:
        return self.value >= self.HIST_ITEM_U_EXP.value

    def requires_past_ui_exp(self) -> bool:
        return self.value >= self.HIST_UI_EXP.value


class Task(Enum):
    RATING = 'rating'
    CONTEXT = 'context'
    EXTRACTION = 'extraction'
    EXPLANATION = 'explanation'
    NEXT_ITEM = 'next_item'
    NEXT_ITEM_SAMPLE = 'next_item_sample'
    FEAT = 'feature'
    TOPN = 'top-n'
    ASPECT = 'aspect'
    NSP = 'next_sentence_prediction'
    L2 = 'l2'
    MLM = 'masked_language_modelling'

    def __str__(self):
        return self.value


class InputType(Enum):
    REGULAR = auto()
    SEQUENTIAL = auto()
    TEMPLATE = auto()
    CUSTOM = auto()


class ModelType(Enum):
    GENERATIVE = auto()
    EXTRACTIVE = auto()
    HYBRID = auto()
    RECOMMENDER = auto()


CFG_ENTRY_TYPES = ('data', 'model', 'train')
