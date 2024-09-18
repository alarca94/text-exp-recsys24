""" Code adapted from https://github.com/reinaldncku/ESCOFILT """

import json
import logging
import queue
from functools import partial

import torch
import numpy as np
import sentence_transformers as st
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.util import cos_sim
from tqdm import tqdm
from packaging.version import parse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from torch import nn
import torch.multiprocessing as mp
from typing import List, Optional, Tuple, Union, Dict, Iterable
from spacy.lang.en import English
from transformers import AutoModel, AutoTokenizer, AutoConfig

from ..base import BaseModel
from src.utils.constants import Task, InputType, ModelType, SeqMode, U_COL, I_COL, EXP_COL, RAT_COL
from ...utils.funcs import multigpu_embed, chunk_embed


class ESCOFILT(BaseModel):
    TASKS = [Task.RATING, Task.EXPLANATION]
    INPUT_TYPE = InputType.CUSTOM
    MODEL_TYPE = ModelType.EXTRACTIVE
    SEQ_MODE = SeqMode.NO_SEQ

    def __init__(self, data_info, cfg, **kwargs):
        super().__init__(data_info, cfg, **kwargs)

        # NOTE: Diff. between ALL and trad user/item_embeddings? ALL embeddings have 1024 emb_dim and trads have 32 (reduce_dim)
        # NOTE: Also, ALL embeddings are pretrained (probably from the NCF model).
        # self.ALL_user_embeddings = users
        # self.ALL_item_embeddings = items
        self.reduce_dim = getattr(cfg, 'reduce_dim', 32)
        num_layers = getattr(cfg, 'num_layers', 2)

        self.trad_user_embeddings = nn.Embedding(self.data_info.n_users, self.reduce_dim)
        self.trad_item_embeddings = nn.Embedding(self.data_info.n_items, self.reduce_dim)

        # NOTE: We do not use the projection layer as the ESCOFILT layers
        emsize = self.user_emb.weight.shape[-1]
        self.compress_u = nn.Linear(emsize, self.reduce_dim)
        self.compress_i = nn.Linear(emsize, self.reduce_dim)

        self.mlp = nn.Sequential()

        ctr = 0
        curr_in = self.reduce_dim * 2
        for ctr in range(num_layers):
            self.mlp.add_module("mlp" + str(ctr), nn.Linear(curr_in, int(curr_in / 2)))
            self.mlp.add_module("relu" + str(ctr), nn.ReLU())
            curr_in = int(curr_in / 2)

        self.mlp.add_module("last_dense", nn.Linear(curr_in, 1))
        # self.mlp.add_module("last_relu", nn.ReLU()) 
        self.dropper = nn.Dropout(0.5)  # Dont forget!!!! 0.5 default

    def get_loss_fns(self):
        return {Task.RATING: nn.MSELoss()}

    def prepare_xy(self, batch, generation=False):
        inputs = {k: v for k, v in batch.items() if k in [U_COL, I_COL, EXP_COL]}
        labels = {Task.RATING: batch[RAT_COL]}
        return inputs, labels

    def unpack_batch(self, batch):
        return batch[U_COL].squeeze(1), batch[I_COL].squeeze(1)

    def forward(self, batch):
        u, i = self.unpack_batch(batch)
        # emp_u = []
        # emp_i = []

        # emp_u = self.ALL_user_embeddings(us)
        # emp_i = self.ALL_item_embeddings(it)
        emp_u = self.user_emb(u)
        emp_i = self.item_emb(i)

        emp_u = self.compress_u(emp_u)
        emp_i = self.compress_i(emp_i)

        trad_u = self.trad_user_embeddings(u)
        trad_i = self.trad_item_embeddings(i)

        emp_u += trad_u
        emp_i += trad_i

        cat_features = torch.cat((emp_u, emp_i), 1)
        cat_features = self.dropper(cat_features)
        out = self.mlp(cat_features)

        return {Task.RATING: out}

    def generate(self, batch):
        # Generate only takes the explanations closest to the cluster centroids (invariant explanations)
        inputs, labels = self.prepare_xy(batch)
        out = self(inputs)
        out[Task.RATING] = out[Task.RATING].cpu().numpy().flatten().tolist()
        out[Task.EXPLANATION] = inputs[EXP_COL].cpu().numpy().tolist()
        return out, labels

#########################


''' #########################################################
#
# Other utilities/tools
#
######################################################### '''


def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LayerSelect(nn.Module):
    """
    Token embeddings are weighted mean of their different hidden layer representations
    """

    def __init__(self, word_embedding_dimension, start_layer: int = -1, end_layer: int = None, concat_dim: int = -1,
                 layer_ixs: Iterable[int] = None, layer_stack: bool = False, reduce_opt: str = 'mean'):
        super(LayerSelect, self).__init__()
        self.config_keys = ["word_embedding_dimension", "layer_start", "num_hidden_layers"]
        self.word_embedding_dimension = word_embedding_dimension
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layer_ixs = layer_ixs
        self.concat_dim = concat_dim
        self.layer_stack = layer_stack
        self.reduce_opt = reduce_opt

        if end_layer is not None:
            assert self.end_layer > self.start_layer
            self.select_fn = self.start_end_select
        elif layer_ixs is not None:
            self.select_fn = self.ixs_select
        else:
            self.select_fn = self.start_select

        if self.layer_stack:
            # NOTE: This is like HuggingFace implementation (reduce at layer level, then reduce at token level)
            self.reduce_fn = self.stack_reduce
        else:
            # NOTE: This is like ESCOFILT implementation (Merge layers at token level via concat. and then reduce)
            self.reduce_fn = partial(torch.cat, dim=self.concat_dim)

    def stack_reduce(self, out_layers: List[torch.Tensor]) -> torch.Tensor:
        out_layers = torch.stack(out_layers)
        if self.reduce_opt == 'mean':
            return out_layers.mean(dim=0, dtype=torch.float32)
        elif self.reduce_opt == 'max':
            return out_layers.max(dim=0)[0]
        elif self.reduce_opt == 'median':
            return out_layers.median(dim=0)[0]
        raise ValueError('Invalid reduce_opt at LayerSelect')

    def start_end_select(self, all_features: List[torch.Tensor]) -> List[torch.Tensor]:
        return all_features[self.start_layer:self.end_layer]

    def start_select(self, all_features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [all_features[self.start_layer]]

    def ixs_select(self, all_features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [all_features[i] for i in self.layer_ixs]

    def forward(self, features: Dict[str, torch.Tensor]):
        ft_all_layers = features["all_layer_embeddings"]
        selected_layers = self.select_fn(ft_all_layers)

        features.update({"token_embeddings": self.reduce_fn(selected_layers)})
        return features

    def get_word_embedding_dimension(self):
        return self.word_embedding_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = LayerSelect(**config)
        model.load_state_dict(
            torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
        )
        return model


class BertParent(object):
    """
    Base handler for BERT models.
    """

    # MODELS = {
    #     'bert-base-uncased': (BertModel, BertTokenizer),
    #     'bert-large-uncased': (BertModel, BertTokenizer),
    #     'xlnet-base-cased': (XLNetModel, XLNetTokenizer),
    #     'xlm-mlm-enfr-1024': (XLMModel, XLMTokenizer),
    #     'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer),
    #     'albert-base-v1': (AlbertModel, AlbertTokenizer),
    #     'albert-large-v1': (AlbertModel, AlbertTokenizer)
    # }

    def __init__(
            self,
            model: str,
            # custom_model: PreTrainedModel=None,
            # custom_tokenizer: PreTrainedTokenizer=None
    ):
        """
        :param model: Model is the string path for the bert weights. If given a keyword, the s3 path will be used.
        :param custom_model: This is optional if a custom bert model is used.
        :param custom_tokenizer: Place to use custom tokenizer.
        """
        # base_model, base_tokenizer = self.MODELS.get(model, (None, None))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(model, output_hidden_states=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.model.eval()

    def tokenize_input(self, text: str) -> torch.tensor:
        """
        Tokenizes the text input.

        :param text: Text to tokenize.
        :return: Returns a torch tensor.
        """

        ret = self.tokenizer.encode_plus(text=text, add_special_tokens=False, max_length=512, padding=False,
                                         truncation=True)

        # tokenized_text = self.tokenizer.tokenize(text)
        # indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = ret['input_ids']

        # print (len(indexed_tokens))
        # print (indexed_tokens)
        # print ("\n\n")

        return torch.tensor([indexed_tokens]).to(self.device)

    def _pooled_handler(self, hidden: torch.Tensor, reduce_option: str) -> torch.Tensor:
        """
        Handles torch tensor.

        :param hidden: The hidden torch tensor to process.
        :param reduce_option: The reduce option to use, such as mean, etc.
        :return: Returns a torch tensor.
        """

        if reduce_option == 'max':
            return hidden.max(dim=1)[0].squeeze()

        elif reduce_option == 'median':
            return hidden.median(dim=1)[0].squeeze()

        return hidden.mean(dim=1).squeeze()

    def extract_embeddings(
            self,
            text: str,
            hidden: Union[List[int], int] = -2,
            reduce_option: str = 'mean',
            hidden_concat: bool = False
    ) -> torch.Tensor:

        """
        Extracts the embeddings for the given text.

        :param text: The text to extract embeddings for.
        :param hidden: The hidden layer(s) to use for a readout handler.
        :param squeeze: If we should squeeze the outputs (required for some layers).
        :param reduce_option: How we should reduce the items.
        :param hidden_concat: Whether or not to concat multiple hidden layers.
        :return: A torch vector.
        """
        tokens_tensor = self.tokenize_input(text)
        pooled, hidden_states = self.model(tokens_tensor)[-2:]

        # deprecated temporary keyword functions.
        if reduce_option == 'concat_last_4':
            last_4 = [hidden_states[i] for i in (-1, -2, -3, -4)]
            cat_hidden_states = torch.cat(tuple(last_4), dim=-1)
            return torch.mean(cat_hidden_states, dim=1).squeeze()

        elif reduce_option == 'reduce_last_4':
            last_4 = [hidden_states[i] for i in (-1, -2, -3, -4)]
            return torch.cat(tuple(last_4), dim=1).mean(axis=1).squeeze()

        elif type(hidden) == int:
            hidden_s = hidden_states[hidden]
            return self._pooled_handler(hidden_s, reduce_option)

        elif hidden_concat:
            last_states = [hidden_states[i] for i in hidden]
            cat_hidden_states = torch.cat(tuple(last_states), dim=-1)
            return torch.mean(cat_hidden_states, dim=1).squeeze()

        last_states = [hidden_states[i] for i in hidden]
        hidden_s = torch.cat(tuple(last_states), dim=1)

        return self._pooled_handler(hidden_s, reduce_option)

    def create_matrix(
            self,
            content: List[str],
            hidden: Union[List[int], int] = -2,
            reduce_option: str = 'mean',
            hidden_concat: bool = False
    ) -> np.ndarray:
        """
        Create matrix from the embeddings.

        :param content: The list of sentences.
        :param hidden: Which hidden layer to use.
        :param reduce_option: The reduce option to run.
        :param hidden_concat: Whether or not to concat multiple hidden layers.
        :return: A numpy array matrix of the given content.
        """

        return np.asarray([
            np.squeeze(self.extract_embeddings(
                t, hidden=hidden, reduce_option=reduce_option, hidden_concat=hidden_concat
            ).data.cpu().numpy()) for t in tqdm(content)
        ])

    def __call__(
            self,
            content: List[str],
            hidden: int = -2,
            reduce_option: str = 'mean',
            hidden_concat: bool = False
    ) -> np.ndarray:
        """
        Create matrix from the embeddings.

        :param content: The list of sentences.
        :param hidden: Which hidden layer to use.
        :param reduce_option: The reduce option to run.
        :param hidden_concat: Whether or not to concat multiple hidden layers.
        :return: A numpy array matrix of the given content.
        """
        return self.create_matrix(content, hidden, reduce_option, hidden_concat)


class ClusterFeatures(object):
    """
    Basic handling of clustering features.
    """
    def __init__(self, features: np.ndarray, algorithm: str = 'kmeans', pca_k: int = None, random_state: int = None):
        """
        :param features: the embedding matrix created by bert parent.
        :param algorithm: Which clustering algorithm to use.
        :param pca_k: If you want the features to be ran through pca, this is the components number.
        :param random_state: Random state.
        """
        from sklearn.decomposition import PCA

        if pca_k:
            self.features = PCA(n_components=pca_k).fit_transform(features)
        else:
            self.features = features

        self.algorithm = algorithm
        self.pca_k = pca_k
        self.random_state = random_state

    def __get_model(self, k: int):
        """
        Retrieve clustering model.

        :param k: amount of clusters.
        :return: Clustering model.
        """
        if self.algorithm == 'gmm':
            return GaussianMixture(n_components=k, random_state=self.random_state)

        return KMeans(n_clusters=k, random_state=self.random_state)

    def __get_centroids(self, model):
        """
        Retrieve centroids of model.

        :param model: Clustering model.
        :return: Centroids.
        """
        if self.algorithm == 'gmm':
            return model.means_
        return model.cluster_centers_

    def __find_closest_args(self, centroids: np.ndarray) -> Dict:
        """
        Find the closest arguments to centroid.

        :param centroids: Centroids to find closest.
        :return: Closest arguments.
        """
        centroid_min = 1e10
        cur_arg = -1
        args = {}
        used_idx = []

        for j, centroid in enumerate(centroids):

            for i, feature in enumerate(self.features):
                value = np.linalg.norm(feature - centroid)

                if value < centroid_min and i not in used_idx:
                    cur_arg = i
                    centroid_min = value

            used_idx.append(cur_arg)
            args[j] = cur_arg
            centroid_min = 1e10
            cur_arg = -1

        return args

    def calculate_elbow(self, k_max: int) -> List[float]:
        """
        Calculates elbow up to the provided k_max.

        :param k_max: K_max to calculate elbow for.
        :return: The inertias up to k_max.
        """
        inertias = []

        for k in range(1, min(k_max, len(self.features))):
            model = self.__get_model(k).fit(self.features)

            inertias.append(model.inertia_)

        return inertias

    def calculate_optimal_cluster(self, k_max: int):
        """
        Calculates the optimal cluster based on Elbow.

        :param k_max: The max k to search elbow for.
        :return: The optimal cluster size.
        """
        delta_1 = []
        delta_2 = []

        max_strength = 0
        k = 1

        inertias = self.calculate_elbow(k_max)

        for i in range(len(inertias)):
            delta_1.append(inertias[i] - inertias[i - 1] if i > 0 else 0.0)
            delta_2.append(delta_1[i] - delta_1[i - 1] if i > 1 else 0.0)

        for j in range(len(inertias)):
            strength = 0 if j <= 1 or j == len(inertias) - 1 else delta_2[j + 1] - delta_1[j + 1]

            if strength > max_strength:
                max_strength = strength
                k = j + 1

        return k

    def cluster(self, ratio: float = 0.1, num_sentences: int = None) -> List[int]:
        """
        Clusters sentences based on the ratio.

        :param ratio: Ratio to use for clustering.
        :param num_sentences: Number of sentences. Overrides ratio.
        :param min_sents: Minimum number of sentences. Works with ratio and overrides it if necessary (and possible).
        :return: Sentences index that qualify for summary.
        """
        if num_sentences is not None:
            if num_sentences == 0:
                return []

            k = min(num_sentences, len(self.features))
        else:
            k = max(int(len(self.features) * ratio), 1)

        model = self.__get_model(k).fit(self.features)

        centroids = self.__get_centroids(model)
        cluster_args = self.__find_closest_args(centroids)

        sorted_values = sorted(cluster_args.values())
        return sorted_values

    def __call__(self, ratio: float = 0.1, num_sentences: int = None) -> List[int]:
        """
        Clusters sentences based on the ratio.

        :param ratio: Ratio to use for clustering.
        :param num_sentences: Number of sentences. Overrides ratio.
        :return: Sentences index that qualify for summary.
        """
        return self.cluster(ratio)


class SentenceHandler(object):
    def __init__(self, language=English):
        """
        Base Sentence Handler with Spacy support.

        :param language: Determines the language to use with spacy.
        """
        self.nlp = language()

        try:
            # Supports spacy 2.0
            self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
            self.is_spacy_3 = False
        except:
            # Supports spacy 3.0
            self.nlp.add_pipe("sentencizer")
            self.is_spacy_3 = True

    def sentence_processor(self, doc, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes a given spacy document and turns them into sentences.

        :param doc: The document to use from spacy.
        :param min_length: The minimum length a sentence should be to be considered.
        :param max_length: The maximum length a sentence should be to be considered.
        :return: Sentences.
        """
        to_return = []

        for c in doc.sents:
            if max_length > len(c.text.strip()) > min_length:

                if self.is_spacy_3:
                    to_return.append(c.text.strip())
                else:
                    to_return.append(c.string.strip())

        return to_return

    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences.

        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        doc = self.nlp(body)
        return self.sentence_processor(doc, min_length, max_length)

    def __call__(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences.

        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        return self.process(body, min_length, max_length)


class ModelProcessor(object):
    aggregate_map = {
        'mean': np.mean,
        'min': np.min,
        'median': np.median,
        'max': np.max
    }

    def __init__(
            self,
            model_name: str = 'bert-large-uncased',
            layers: Union[List[int], int, str] = -2,
            reduce_opt: str = 'mean',
            batch_size: int = 64,
            chunk_size: int = 20000,
            sentence_handler: SentenceHandler = SentenceHandler(),
            random_state: int = None
    ):
        """
        This is the parent Bert Summarizer model. New methods should implement this class.

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer(s) of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param sentence_handler: The handler to process sentences. If want to use coreference, instantiate and pass.
        CoreferenceHandler instance
        :param random_state: The random state to reproduce summarizations.
        :param hidden_concat: Whether or not to concat multiple hidden layers.
        """
        # np.random.seed(random_state)
        self.name = model_name

        if parse(st.__version__) >= parse('2.3.0'):
            from sentence_transformers.util import is_sentence_transformer_model
        else:
            from src.utils.funcs import is_sentence_transformer_model

        if is_sentence_transformer_model(model_name):
            # logging.info('Loading existing sentence transformer...')
            self.model = SentenceTransformer(model_name)
        else:
            logging.info('Building sentence transformer from scratch...')
            transformer = models.Transformer(model_name, max_seq_length=512)
            config = AutoConfig.from_pretrained(model_name, **{"output_hidden_states": True})
            kwargs = {}
            if isinstance(layers, int):
                kwargs['layer_ixs'] = [layers]  # The embedding layer output is in output_hidden_states as index 0
            elif isinstance(layers, str) and layers.startswith('reduce_last_'):
                n_layers = int(layers.split('_')[-1])
                kwargs['start_layer'] = -n_layers
                kwargs['concat_dim'] = 1
            elif isinstance(layers, str) and layers.startswith('concat_last_'):
                n_layers = int(layers.split('_')[-1])
                kwargs['start_layer'] = -n_layers
            else:
                raise ValueError('')
            # NOTE: With this option, selected layers are averaged and then tokens are aggregated with reduce_opt.
            #  However, in ESCOFILT implementation, layers are concatenated and then reduced altogether with reduce_opt.
            #  Nevertheless, default values (hidden = -2) behaves equally with both approaches
            transformer.auto_model = AutoModel.from_pretrained(model_name, config=config)
            transformer.auto_model.config.tokenizer_class = transformer.tokenizer.__class__.__name__
            layer_select = LayerSelect(transformer.get_word_embedding_dimension(), **kwargs)
            pooling_model = models.Pooling(transformer.get_word_embedding_dimension(), reduce_opt)
            self.model = SentenceTransformer(modules=[transformer, layer_select, pooling_model])
        # self.model = BertParent(model)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.sentence_handler = sentence_handler
        self.random_state = random_state

    def cluster_runner(
            self,
            content: List[str],
            ratio: float = 0.2,
            algorithm: str = 'kmeans',
            use_first: bool = True,
            num_sentences: int = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Runs the cluster algorithm based on the hidden state. Returns both the embeddings and sentences.

        :param content: Content list of sentences.
        :param ratio: The ratio to use for clustering.
        :param algorithm: Type of algorithm to use for clustering.
        :param use_first: Whether to use first sentence (helpful for news stories, etc).
        :param num_sentences: Number of sentences to use for summarization.
        :param min_sents: Minimum number of sentences to use for summarization (if possible).
        :return: A tuple of summarized sentences and embeddings
        """
        if num_sentences is not None:
            num_sentences = num_sentences if use_first else num_sentences

        if len(content) > self.chunk_size:
            n_chunks = max(np.ceil(len(content) / self.chunk_size), torch.cuda.device_count())
            chunk_size = int(np.ceil(len(content) / n_chunks))
            if torch.cuda.device_count() > 1:
                hidden = multigpu_embed(self.model, content, self.batch_size, chunk_size)
            else:
                hidden = chunk_embed(self.model, content, self.batch_size, chunk_size, show_progress_bar=False)
        else:
            hidden = self.model.encode(content, self.batch_size, show_progress_bar=False)

        # NOTE: Unable to pre-compute distances to save time (CPU allocation memory error for some users)

        # logging.info('Starting clustering...')
        # hidden = self.model(content, self.hidden, self.reduce_option, hidden_concat=self.hidden_concat)
        hidden_args = ClusterFeatures(hidden, algorithm, random_state=self.random_state).cluster(ratio, num_sentences)

        if use_first:

            if not hidden_args:
                hidden_args.append(0)

            elif hidden_args[0] != 0:
                hidden_args.insert(0, 0)

        sentences = [content[j] for j in hidden_args]
        embeddings = np.asarray([hidden[j] for j in hidden_args])

        return sentences, embeddings

    def __run_clusters(
            self,
            content: List[str],
            ratio: float = 0.2,
            algorithm: str = 'kmeans',
            use_first: bool = True,
            num_sentences: int = None
    ) -> List[str]:
        """
        Runs clusters and returns sentences.

        :param content: The content of sentences.
        :param ratio: Ratio to use for for clustering.
        :param algorithm: Algorithm selection for clustering.
        :param use_first: Whether to use first sentence
        :param num_sentences: Number of sentences. Overrides ratio.
        :return: summarized sentences
        """
        sentences, _ = self.cluster_runner(content, ratio, algorithm, use_first, num_sentences)
        return sentences

    def __retrieve_summarized_embeddings(
            self,
            content: List[str],
            ratio: float = 0.2,
            algorithm: str = 'kmeans',
            use_first: bool = True,
            num_sentences: int = None,
            min_sents: int = None,
            max_sents: int = 20000
    ) -> Tuple[List, np.ndarray]:
        """
        Retrieves embeddings of the summarized sentences.

        :param content: The content of sentences.
        :param ratio: Ratio to use for clustering.
        :param algorithm: Algorithm selection for clustering.
        :param use_first: Whether to use first sentence
        :return: Summarized embeddings
        """
        # NOTE: Added by alarca to account for top-k minimum explanations and cap the maximum number of sentences
        if num_sentences is None:
            k = int(len(content) * ratio)
            if k > max_sents:
                num_sentences = max_sents
            elif min_sents is not None and k < min_sents:
                num_sentences = min_sents

        sentences, embeddings = self.cluster_runner(content, ratio, algorithm, use_first, num_sentences)
        return sentences, embeddings

    def calculate_elbow(
            self,
            body: str,
            algorithm: str = 'kmeans',
            min_length: int = 40,
            max_length: int = 600,
            k_max: int = None,
    ) -> List[float]:
        """
        Calculates elbow across the clusters.

        :param body: The input body to summarize.
        :param algorithm: The algorithm to use for clustering.
        :param min_length: The min length to use.
        :param max_length: The max length to use.
        :param k_max: The maximum number of clusters to search.
        :return: List of elbow inertia values.
        """
        sentences = self.sentence_handler(body, min_length, max_length)

        if k_max is None:
            k_max = len(sentences) - 1

        # hidden = self.model(sentences, self.hidden, self.reduce_option, hidden_concat=self.hidden_concat)
        if len(sentences) > self.chunk_size:
            if torch.cuda.device_count() > 1:
                hidden = multigpu_embed(self.model, sentences, self.batch_size, self.chunk_size)
            else:
                hidden = chunk_embed(self.model, sentences, self.batch_size, self.chunk_size)
        else:
            hidden = self.model.encode(sentences, self.batch_size)

        elbow = ClusterFeatures(hidden, algorithm, random_state=self.random_state).calculate_elbow(k_max)

        return elbow

    def calculate_optimal_k(
            self,
            body: str,
            algorithm: str = 'kmeans',
            min_length: int = 40,
            max_length: int = 600,
            k_max: int = None
    ):
        """
        Calculates the optimal Elbow K.

        :param body: The input body to summarize.
        :param algorithm: The algorithm to use for clustering.
        :param min_length: The min length to use.
        :param max_length: The max length to use.
        :param k_max: The maximum number of clusters to search.
        :return:
        """
        sentences = self.sentence_handler(body, min_length, max_length)

        if k_max is None:
            k_max = len(sentences) - 1

        # hidden = self.model(sentences, self.hidden, self.reduce_option, hidden_concat=self.hidden_concat)
        if len(sentences) > self.chunk_size:
            if torch.cuda.device_count() > 1:
                hidden = multigpu_embed(self.model, sentences, self.batch_size, self.chunk_size)
            else:
                hidden = chunk_embed(self.model, sentences, self.batch_size, self.chunk_size)
        else:
            hidden = self.model.encode(sentences, self.batch_size)

        optimal_k = ClusterFeatures(hidden, algorithm, random_state=self.random_state).calculate_optimal_cluster(k_max)

        return optimal_k

    def run_embeddings(
            self,
            body: Union[str, list],
            sentences: list = None,
            ratio: float = 0.2,
            min_length: int = 40,
            max_length: int = 600,
            use_first: bool = True,
            algorithm: str = 'kmeans',
            num_sentences: int = None,
            min_sents: int = None,
            max_sents: int = 20000,
            aggregate: str = None
    ) -> Tuple[List, Optional[np.ndarray]]:
        """
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the embeddings.

        :param body: The raw string body to process
        :param ratio: Ratio of sentences to use
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param num_sentences: Number of sentences to use. Overrides ratio.
        :param min_sents: Minimum number of sentences to use. If necessary, it overrides ratio.
        :param max_sents: Maximum number of sentences to use. If surpassed, it overrides ratio.
        :param aggregate: One of mean, median, max, min. Applied on zero axis
        :return: A summary embedding
        """
        if isinstance(body, str):
            sentences = self.sentence_handler(body, min_length, max_length)

        if sentences:
            sents, embs = self.__retrieve_summarized_embeddings(sentences, ratio, algorithm, use_first, num_sentences,
                                                                min_sents, max_sents)

            if aggregate is not None:
                assert aggregate in ['mean', 'median', 'max', 'min'], "aggregate must be mean, min, max, or median"
                embs = self.aggregate_map[aggregate](embs, axis=0)

            return sents, embs

        return [], None

    def run(
            self,
            body: str,
            ratio: float = 0.2,
            min_length: int = 40,
            max_length: int = 600,
            use_first: bool = True,
            algorithm: str = 'kmeans',
            num_sentences: int = None,
            return_as_list: bool = False
    ) -> Union[List, str]:
        """
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process
        :param ratio: Ratio of sentences to use
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param num_sentences: Number of sentences to use (overrides ratio).
        :param return_as_list: Whether or not to return sentences as list.
        :return: A summary sentence
        """
        sentences = self.sentence_handler(body, min_length, max_length)

        if sentences:
            sentences = self.__run_clusters(sentences, ratio, algorithm, use_first, num_sentences)

        if return_as_list:
            return sentences
        else:
            return ' '.join(sentences)

    def __call__(
            self,
            body: str,
            ratio: float = 0.2,
            min_length: int = 40,
            max_length: int = 600,
            use_first: bool = True,
            algorithm: str = 'kmeans',
            num_sentences: int = None,
            return_as_list: bool = False,
    ) -> str:
        """
        (utility that wraps around the run function)
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process.
        :param ratio: Ratio of sentences to use.
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary.
        :param use_first: Whether or not to use the first sentence.
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param Number of sentences to use (overrides ratio).
        :param return_as_list: Whether or not to return sentences as list.
        :return: A summary sentence.
        """
        return self.run(
            body, ratio, min_length, max_length, algorithm=algorithm, use_first=use_first,
            num_sentences=num_sentences,
            return_as_list=return_as_list
        )
