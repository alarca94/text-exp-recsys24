""" Code adapted from https://github.com/HCDM/XRec/tree/main/GREENer """

import copy
import gc
import os
import queue
import time
from itertools import groupby
from typing import Tuple, Callable, Optional

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GATConv
import torch.multiprocessing as mp

from ..base import BaseModel
from src.utils.constants import *

import logging

logging.getLogger('pyomo.core').setLevel(logging.ERROR)

# import gurobipy as gp
import pyomo.environ as pe
import pyomo.opt as po


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def activation_layer(act_name):
    """Construct activation layers
    Args:
        act_name: str or nn.Module, name of activation function
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, layer_num=2, parameterization='vector'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = nn.Parameter(
                torch.Tensor(self.layer_num, in_features, 1)
            )
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(
                torch.Tensor(self.layer_num, in_features, in_features)
            )
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

    def forward(self, inputs):
        # inputs: (batch_size, units)
        # x_0: (batch_size, units, 1)
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)   # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]                  # W * xi + b
                x_l = x_0 * dot_ + x_l                      # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        # x_l: (batch_size, units)
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class DNN(nn.Module):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.
        The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``.
        For instance, for a 2D input with shape ``(batch_size, input_dim)``,
        the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **inputs_dim**: input feature dimension.
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **init_seed**: float. Used for initialization of the linear layer.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, seed=1024):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc

        return deep_input


class DCN(nn.Module):
    """Instantiates the Deep&Cross Network architecture. Including DCN-V (parameterization='vector')
    and DCN-M (parameterization='matrix').

    :param node_embed_size: Node embedding size. User/Item/Sentence should have the same node embedding size.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param cross_num: positive integet,cross layer number
    :param cross_parameterization: str, ``"vector"`` or ``"matrix"``, how to parameterize the cross network.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_cross: float. L2 regularizer strength applied to cross net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
    :param dnn_activation: Activation function to use in DNN
    :param device: str, ``"cpu"`` or ``"cuda:0"`` (or other cuda, if we have)
    :return: A PyTorch model instance.

    """
    def __init__(self, node_embed_size, cross_num=2, cross_parameterization='vector',
                 dnn_hidden_units=[128, 128], l2_reg_linear=0.00001, l2_reg_cross=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False):
        super(DCN, self).__init__()
        self.node_embed_size = node_embed_size
        self.input_embed_size = self.node_embed_size * 3    # concat user/item/sentence node embeddings
        # after DNN and CrossNet, we concat the 2 feature embeddings
        self.dcn_output_size = self.input_embed_size + dnn_hidden_units[-1]
        # init DNN
        self.dnn = DNN(
            inputs_dim=self.input_embed_size,
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            use_bn=dnn_use_bn,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            init_std=init_std)
        # init CrossNet
        self.crossnet = CrossNet(
            in_features=self.input_embed_size,
            layer_num=cross_num,
            parameterization=cross_parameterization)
        self.dcn_linear = nn.Linear(
            self.dcn_output_size, 1, bias=False)

    def forward(self, x):
        deep_out = self.dnn(x)
        cross_out = self.crossnet(x)
        stack_out = torch.cat((cross_out, deep_out), dim=-1)
        logit = self.dcn_linear(stack_out)
        return logit


class GATNET(torch.nn.Module):
    def __init__(self, in_dim, out_dim, head_num, dropout_rate):
        super(GATNET, self).__init__()

        self.gat_layer_1 = GATConv(in_dim, out_dim, heads=head_num, dropout=dropout_rate)
        # default, concat all attention head
        self.gat_layer_2 = GATConv(head_num*out_dim, out_dim, heads=1, concat=False, dropout=dropout_rate)

    def forward(self, x, edge_index, attention_flag=False):
        x_0 = F.dropout(x, p=0.6, training=self.training)

        if attention_flag:
            # e_1, e_2: edge attention weights (edge_index, attention_weights)
            x_1, e_1 = self.gat_layer_1(x_0, edge_index, return_attention_weights=True)
            x_1 = F.elu(x_1)
            x_1 = F.dropout(x_1, p=0.6, training=self.training)
            x_2, e_2 = self.gat_layer_2(x_1, edge_index, return_attention_weights=True)

            return x_2, e_1, e_2
        else:
            x_1 = self.gat_layer_1(x_0, edge_index)
            x_1 = F.elu(x_1)
            x_1 = F.dropout(x_1, p=0.6, training=self.training)
            x_2 = self.gat_layer_2(x_1, edge_index)

            return x_2


class GREENer(BaseModel):
    TASKS = [Task.EXPLANATION, Task.FEAT]
    INPUT_TYPE = InputType.CUSTOM
    MODEL_TYPE = ModelType.EXTRACTIVE
    SEQ_MODE = SeqMode.HIST_UI_EXP

    def __init__(self, data_info, cfg, **kwargs):
        super().__init__(data_info, cfg, **kwargs)

        # gat_layers = getattr(cfg, 'gat_layers', 2)
        gat_num_heads = getattr(cfg, 'gat_num_heads', [4, 1])
        # gat_jk = getattr(cfg, 'gat_jk', 'last')
        gat_dropout = getattr(cfg, 'gat_dropout', 0.6)
        cross_num = getattr(cfg, 'cross_num', 2)
        cross_type = getattr(cfg, 'cross_type', 'vector')
        dnn_hidden_units = getattr(cfg, 'dnn_hidden_units', [128, 128])
        init_std = getattr(cfg, 'init_std', 0.0001)
        dnn_dropout = getattr(cfg, 'dnn_dropout', 0)
        dnn_activation = getattr(cfg, 'dnn_activation', 'relu')
        dnn_use_bn = getattr(cfg, 'dnn_use_bn', 0)

        ilp_topk = getattr(cfg, 'topk', 5)
        if getattr(cfg, 'use_ilp', True):
            ilp_alpha = getattr(cfg, 'alpha', 2.0)
            ilp_pre_topk = getattr(cfg, 'pre_topk', 100)
            tfidf_exps = getattr(data_info, f'tfidf_{EXP_COL}_emb')
            postprocess_path = os.path.join(BASE_PATH, 'logs', 'tmp', f'GREENER_postprocess_{data_info.dataset}_{kwargs["seed"]}')
            self.ilp_processor = ILPPostprocessor(tfidf_exps, ilp_pre_topk, ilp_topk, ilp_alpha, postprocess_path)
            self.postprocess = self.postprocess_ilp
        else:
            self.topk = ilp_topk
            self.postprocess = self.postprocess_naive

        self.valid_sent_labels = ['hard', 'bleu']  #, 'bert']

        try:
            self.sent_label = self.valid_sent_labels.index(cfg.labels.lower())
        except ValueError:
            raise ValueError(f'Invalid label in the config file: {cfg.labels.lower()} --> {self.valid_sent_labels}')

        # act_fn = nn.LeakyReLU()
        # act_fn = nn.ELU()

        self.feat_emb = self.get_embeddings(FEAT_COL, data_info.n_feats)
        self.sent_emb = self.get_embeddings(EXP_COL, data_info.n_sents)

        # self.gat = CustomGAT(in_channels=self.hsize, out_channels=self.hsize, num_layers=gat_layers,
        #                      heads=gat_num_heads, act=act_fn, jk=gat_jk)

        self.gat = GATNET(self.hsize, self.hsize, gat_num_heads[0], dropout_rate=gat_dropout)

        self.dcn = DCN(node_embed_size=self.hsize, cross_num=cross_num, cross_parameterization=cross_type,
                       dnn_hidden_units=dnn_hidden_units, init_std=init_std, dnn_dropout=dnn_dropout,
                       dnn_activation=dnn_activation, dnn_use_bn=dnn_use_bn)

        self.feat_output = nn.Linear(self.hsize, 1)

        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

        # Delete Dataclass attributes to avoid them copied over to the workers in multiprocessing postprocess
        delattr(self, 'data_info')
        delattr(self, 'tok')
        delattr(self, 'mappers')

    @staticmethod
    def unpack_data(data):
        g = data['graph']
        bsz = data['size']
        cols = [U_COL, I_COL, EXP_COL, FEAT_COL]
        return [g[c] for c in cols] + [g[c, FEAT_COL].edge_index for c in cols[:-1]] + [bsz]

    def get_loss_fns(self):
        if self.cfg.labels.lower() == 'hard':
            exp_loss = self.bce_loss
        else:
            self.loss_args['bpr_eps'] = self.loss_args.get('bpr_eps', 1e-5)
            exp_loss = self.pairwise_ranking_loss

        return {
            Task.EXPLANATION: exp_loss,
            Task.FEAT: self.bce_loss
        }

    def get_from_batch(self, batch, col):
        return batch['graph'][col].x

    def bce_loss(self, preds, labels, **kwargs):
        return self.bce(preds, labels.float())

    def pairwise_ranking_loss(self, preds, labels, **kwargs):
        batch = kwargs.get('sent_batch', torch.ones(labels.shape))
        mask = ((batch - batch.T) == 0)

        sign = (labels - labels.T).triu(1)
        sign[sign.abs() <= kwargs.get('bpr_eps', 1e-5)] = 0
        sign = sign.sign()  # * batch_mask

        mask |= (sign != 0)
        diffs = torch.masked_select(sign * (preds - preds.T), mask)
        return - F.logsigmoid(diffs).sum() / mask.sum()

    @staticmethod
    def is_monotonic(x: torch.Tensor, max_dif: int = 1):
        check = (x[1:] - x[:-1])
        return torch.bitwise_and(check.gt(-1), check.lt(max_dif + 1)).all()

    def forward(self, data):
        u, i, s, f, uf_edge_index, if_edge_index, sf_edge_index, bsz = self.unpack_data(data)

        # print(f'The batch size is: {bsz}')

        # Embed and project to equal dim. latent space
        ux = self.user_emb(u.x)
        ix = self.item_emb(i.x)
        sx = self.sent_emb(s.x)
        fx = self.feat_emb(f.x)

        # Concatenate all nodes
        x = torch.cat((ux, ix, sx, fx), dim=0)

        # Concatenate all indexes (add respective index offsets and make it undirectional)
        i_offset = ux.shape[0]
        s_offset = i_offset + ix.shape[0]
        f_offset = s_offset + sx.shape[0]

        if_edge_index[0, :] += i_offset
        sf_edge_index[0, :] += s_offset
        edge_index = torch.cat((uf_edge_index, if_edge_index, sf_edge_index), dim=1)
        edge_index[1, :] += f_offset

        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)

        ## Contextualize node embeddings through GAT
        hidden = self.gat(x, edge_index)

        # print(f'The shape of hidden after GAT is: {hidden.shape}')
        assert self.is_monotonic(u.batch) & self.is_monotonic(i.batch) & self.is_monotonic(s.batch)

        # num_repeats = torch_scatter.scatter_add(torch.ones(sx.shape[0]), index=s.batch, dim_size=bsz)
        dcn_x = torch.cat((hidden[:i_offset], hidden[i_offset:s_offset]), dim=-1)
        dcn_x = dcn_x[s.batch]  # Works as a repeat_interleave of users and items
        dcn_x = torch.cat((dcn_x, hidden[s_offset: f_offset]), dim=-1)

        assert dcn_x.shape[0] == sx.shape[0]
        # print(f'The shape of dcn_x is: {dcn_x.shape}')

        logits_s = self.dcn(dcn_x)
        logits_f = self.feat_output(hidden[f_offset:])

        assert logits_f.shape[0] == fx.shape[0]
        # print(f'The shape of logits_s is: {logits_s.shape}')
        # print(f'The shape of logits_f is: {logits_f.shape}')

        return {
            Task.EXPLANATION: logits_s,
            Task.FEAT: logits_f
        }

    def set_batchwise_loss_args(self, inputs, labels):
        self.loss_args['sent_batch'] = inputs['graph'][EXP_COL].batch.unsqueeze(1)

    def prepare_xy(self, batch, generation=False):
        if generation:
            # NOTE: For generation, EXP and FEAT labels are taken directly from the batches
            labels = {}
        else:
            labels = {
                Task.EXPLANATION: batch['graph'][EXP_COL].y[:, self.sent_label].unsqueeze(1),
                Task.FEAT: batch['graph'][FEAT_COL].y.unsqueeze(1)
            }
            if self.sent_label < 2:
                labels[Task.EXPLANATION] = labels[Task.EXPLANATION].to(torch.int)

        return batch, labels

    def generate(self, batch):
        inputs, labels = self.prepare_xy(batch, generation=True)

        preds = self(inputs)

        # If we were to return the current predictions (EXP and FEAT logits):
        # Option 1: Return the batch index of the sentences as well --> Problem with repeated ixs in different batches.
        # Option 2: convert to list of lists with itertools.groupby --> Need to change some of postprocess logic
        # Sticking with option 2
        # NOTE: This grouping works because batch_ixs are sorted
        preds.pop(Task.FEAT)
        groupby_iter = iter(inputs['graph'][EXP_COL].batch.cpu().numpy().flatten())
        preds[Task.EXPLANATION] = preds[Task.EXPLANATION].detach().cpu().numpy().flatten().tolist()
        preds[Task.EXPLANATION] = [list(g) for k, g in groupby(preds[Task.EXPLANATION],
                                                               lambda _, it=groupby_iter: next(it))]
        groupby_iter = iter(inputs['graph'][EXP_COL].batch.cpu().numpy().flatten())
        preds[EXP_COL] = [list(g) for k, g in groupby(inputs['graph'][EXP_COL].x.cpu().numpy().flatten().tolist(),
                                                      lambda _, it=groupby_iter: next(it))]
        return preds, labels

    def postprocess_naive(self, preds, labels, **kwargs):
        # Empty memory as it is no longer needed
        del self.feat_emb, self.sent_emb, self.user_emb, self.item_emb
        choices = []
        for logits, exps in zip(preds[Task.EXPLANATION], preds[EXP_COL]):
            if len(exps) <= self.topk:
                choices.append(np.array(exps)[np.argsort(logits)[::-1]].tolist())
            else:
                topk_ixs = np.argpartition(logits, -self.topk)[-self.topk:]
                sorted_topk_ixs = topk_ixs[np.argsort(np.array(logits)[topk_ixs])[::-1]]
                choices.append(np.array(exps)[sorted_topk_ixs].tolist())
        return {Task.EXPLANATION: choices}

    def postprocess_ilp(self, preds, labels, max_chunk_sz=5000, parallel=True, max_n_proc=10):
        self.eval()
        del self.feat_emb, self.sent_emb, self.user_emb, self.item_emb, self.gat
        torch.cuda.empty_cache()
        gc.collect()
        return self.ilp_processor.run(preds, labels, max_chunk_sz, parallel, max_n_proc)


class ILPPostprocessor:
    def __init__(self, tfidf_exps, ilp_pre_topk, ilp_topk, ilp_alpha, postprocess_path):
        self.tfidf_exps = tfidf_exps
        self.ilp_pre_topk = ilp_pre_topk
        self.ilp_topk = ilp_topk
        self.ilp_alpha = ilp_alpha
        self.postprocess_path = postprocess_path

        self.threads = None

        if not os.path.isdir(self.postprocess_path):
            os.mkdir(self.postprocess_path)

        assert self.ilp_pre_topk >= self.ilp_topk

    def _setup_pool(self, n_proc):
        # Launch processes and create input/output queues
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []
        for _ in range(n_proc):
            p = ctx.Process(
                target=ILPPostprocessor._postprocess_chunk,
                args=(self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return input_queue, output_queue, processes

    @staticmethod
    def _close_pool(processes, input_queue, output_queue):
        # Close all processes and queues
        for p in processes:
            p.terminate()

        for p in processes:
            p.join()
            p.close()

        input_queue.close()
        output_queue.close()

    def run(self, preds, labels, max_chunk_sz=5000, parallel=False, max_n_proc=10):
        data = list(zip(preds[Task.EXPLANATION], preds[EXP_COL],
                        labels[f'metadata-{U_COL}'], labels[f'metadata-{I_COL}']))
        # Clean past temporal logs
        for f in os.listdir(self.postprocess_path):
            if f.endswith('.txt'):
                os.remove(os.path.join(self.postprocess_path, f))
        # Determine number of nodes. Estimate each new process will require half the memory allocated for the
        # current process and reserve one core for other tasks.
        process = psutil.Process(os.getpid())
        n_proc = int((100 - psutil.virtual_memory().percent) / (process.memory_percent() / 2))
        if parallel:
            if n_proc < 2:
                logging.warning('GREENer postprocess will not be run in parallel due to lack of CPU memory')
                return self.run(preds, labels, max_chunk_sz, parallel=False)

            n_proc = min([os.cpu_count() - 1, max_n_proc, n_proc])  # , int(np.ceil(len(data) / max_chunk_sz))
            logging.info(f'GREENer postprocess will be run in {n_proc} parallel processes')

            self.threads = len(os.sched_getaffinity(0)) - 1 // n_proc
            input_queue, output_queue, processes = self._setup_pool(n_proc)
            choices = self._postprocess_mp(data, n_proc, max_chunk_sz, input_queue, output_queue)
            self._close_pool(processes, input_queue, output_queue)
            # return {Task.EXPLANATION: choices}
        else:
            write_f = open(os.path.join(self.postprocess_path, f'global.txt'), 'w+')
            self.threads = min([os.cpu_count() - 1, n_proc])
            choices = [self.rerank_and_select(data[i:i + max_chunk_sz], write_f)
                       for i in range(0, len(data), max_chunk_sz)]
            choices = sum(choices, start=[])

        assert len(choices) == len(data)
        return {Task.EXPLANATION: choices}

    def _optimize_ilp(self, solver, p_b, sims, b_size, bix, user, item, write_f=None):
        # ILP Objective: Maximize.
        # First argument: Weighted sum over logits_s prediction (+1) (x)
        # Second argument: Weighted sum over sentence similarities (-1) (y)
        # Constraints:
        #   1_ Sum of x add up to K.
        #   2_ x is binary.
        #   3_ y is binary.
        #   4_ x_si + x_sj is leq than y_ij +1
        #   5_ Sum of ys = K * (K - 1)

        ILP_m = pe.ConcreteModel()
        if solver == 'glpk':
            solver = po.SolverFactory('glpk')
        elif solver == 'gurobi':
            solver = po.SolverFactory('gurobi', solver_io="python")
            if self.threads is not None:
                solver.options['threads'] = self.threads
                # solver.options['solutionlimit'] = 1
        else:
            raise ValueError(f'Solver ({solver}) is not supported by this framework')

        sims = {(i + 1, j + 1): sims[i, j] for i in range(b_size) for j in range(i + 1, b_size)}

        ILP_m.Xset = pe.RangeSet(1, b_size)
        ILP_m.gs = pe.Param(ILP_m.Xset, initialize={i + 1: v for i, v in enumerate(p_b)})
        ILP_m.sim = pe.Param(ILP_m.Xset, ILP_m.Xset, initialize=sims)  # , default=0.0)
        ILP_m.alpha = pe.Param(initialize=self.ilp_alpha)
        ILP_m.topk = pe.Param(initialize=self.ilp_topk)
        ILP_m.X = pe.Var(ILP_m.Xset, domain=pe.Binary)
        ILP_m.Y = pe.Var(ILP_m.Xset, ILP_m.Xset, domain=pe.Binary)

        left_obj = sum(ILP_m.gs[i] * ILP_m.X[i] for i in ILP_m.Xset)
        right_obj = - ILP_m.alpha * sum(
            ILP_m.sim[i, j] * ILP_m.Y[i, j] for i in range(1, b_size + 1) for j in range(i + 1, b_size + 1))
        ILP_m.obj = pe.Objective(expr=(left_obj + right_obj), sense=pe.maximize)

        constr1 = sum(ILP_m.X[x] for x in ILP_m.Xset) == ILP_m.topk
        ILP_m.constr1 = pe.Constraint(expr=constr1)

        ILP_m.constr4 = pe.ConstraintList()
        for i in range(1, b_size + 1):
            for j in range(i + 1, b_size + 1):
                ILP_m.constr4.add((ILP_m.X[i] + ILP_m.X[j]) <= (ILP_m.Y[i, j] + 1))

        constr5 = sum(ILP_m.Y[i, j] for i in range(1, b_size + 1) for j in range(i + 1, b_size + 1)) == (
                    ILP_m.topk * (ILP_m.topk - 1) / 2)
        ILP_m.constr5 = pe.Constraint(expr=constr5)

        solver.solve(ILP_m, tee=False)
        obj_score = pe.value(ILP_m.obj, exception=False)
        if obj_score is not None:
            # Found optimal solution
            solution = np.array(list(ILP_m.X.get_values().values())).nonzero()

            y_values = np.array(list(ILP_m.Y.get_values().values())).reshape(b_size, b_size).tolist()
            self._check_inconsistencies(list(ILP_m.X.get_values().values()), y_values, b_size, bix, user, item, write_f)
        else:
            # No feasible solution found. Defaulting to choice based on g(s_i)
            solution = np.sort(np.argpartition(p_b, -min(len(p_b), self.ilp_topk))[-self.ilp_topk:])

        return obj_score, solution

    @staticmethod
    def _check_inconsistencies(X, Y, b_size, idx, user, item, write_f=None):
        try:
            for i in range(b_size):
                for j in range(i + 1, b_size):
                    if X[i] == 1.0 and X[j] == 1.0:
                        assert Y[i][j] == 1.0
                    else:
                        assert Y[i][j] == 0.0
        except AssertionError:
            # Add log of Y not aligned with X
            msg = f"At Batch Index {idx}, Y not aligned with X for user {user}  and item {item}, for i {i} and j {j}"
            if write_f is not None:
                write_f.write(msg)
            else:
                logging.warning(msg)

    @staticmethod
    def _postprocess_mp(data, n_proc, max_chunk_sz, input_queue, output_queue):
        # Divide in chunks and send it to the input_queue
        n_samples = len(data)
        # laps = np.ceil(n_samples / max_chunk_sz / n_proc)
        # chunk_sz = int(np.ceil(n_samples / n_proc / laps))
        chunk_sz = int(np.ceil(n_samples / n_proc))
        n_chunks = int(np.ceil(n_samples / chunk_sz))
        for i in range(n_chunks):
            chunk = data[i * chunk_sz: (i + 1) * chunk_sz]
            input_queue.put([i, chunk])

        # Get output from all processes
        results_list = sorted([output_queue.get() for _ in range(n_chunks)], key=lambda x: x[0])
        choices = sum([result[1] for result in results_list], start=[])
        return choices

    @torch.no_grad()
    def rerank_and_select(self, samples, write_f=None):
        ILP_scores = []
        choices = []
        total = len(samples)
        for rix, (logits, exps, user, item) in enumerate(samples):
            msg = f'Processing row {rix + 1} / {total} for user {user} and item ' \
                  f'{item} with {len(exps)} explanation sentences.\n'
            if write_f is not None:
                write_f.write(f'{time.asctime(time.localtime())} -- {msg}')
            else:
                logging.debug(msg)
            assert len(exps) == len(logits)
            logits = torch.FloatTensor(logits)
            exps = torch.LongTensor(exps)

            if len(exps) > self.ilp_pre_topk:
                _, ixs = torch.topk(logits, k=self.ilp_pre_topk)
                exps = exps[ixs]
                logits = logits[ixs]

            if len(exps) <= self.ilp_topk:
                ixs = logits.argsort()
                choices.append(exps[ixs].numpy()[::-1].tolist())
                ILP_scores.append(None)
                continue

            p_b = logits.numpy()
            # s_b = getattr(model.data_info, f'tfidf_{EXP_COL}_emb')[exps.numpy()].toarray()
            s_b = self.tfidf_exps[exps].toarray()

            s_b = torch.FloatTensor(s_b)
            sims = F.cosine_similarity(s_b[None, :, :], s_b[:, None, :], dim=-1, eps=1e-8).numpy()
            del s_b, logits

            obj_score, solution = self._optimize_ilp('gurobi', p_b, sims, len(exps), rix, user, item, write_f)
            ILP_scores.append(obj_score)
            choices.append(exps.numpy()[solution].tolist())

            if write_f is not None:
                write_f.write(f'{time.asctime(time.localtime())} -- Selected sentences: {choices[-1]}\n')
                write_f.flush()
            else:
                logging.debug(f'Selected sentences: {choices[-1]}')

        return choices

    @staticmethod
    def _postprocess_chunk(model, input_queue, results_queue):
        while True:
            try:
                chunk_id, chunk = input_queue.get()
                write_f = open(os.path.join(model.postprocess_path, f'chunk_{chunk_id}.txt'), 'w+')
                write_f.write(f'Running in process PID: {os.getpid()}\n')

                choices = model.rerank_and_select(chunk, write_f)

                write_f.close()
                results_queue.put([chunk_id, choices])
                # TOOD: Check if CUDA OOM happens inside the try-catch
            except queue.Empty:
                break
            except Exception as e:
                if "write_f" not in locals() or write_f.closed:
                    write_f = open(os.path.join(model.postprocess_path, f'chunk_{chunk_id}.txt'), 'w+')
                write_f.write(str(e))
                write_f.close()
                break
