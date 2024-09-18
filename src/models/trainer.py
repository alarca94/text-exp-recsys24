import importlib
import logging
import gc
import os
import shutil
import torch
import time
import numpy as np

from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timezone, timedelta

from ..utils import funcs
from ..utils.in_out import colored, Colors, print_losses
from ..utils.callbacks import EarlyStopping
from ..utils.evaluation import Evaluator
from ..utils.constants import *


class DummyScheduler:
    def __init__(self, optimizer):
        self.lr = [group['lr'] for group in optimizer.param_groups]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self):
        pass

    def get_last_lr(self):
        return self.lr


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def __call__(self, *args, **kwargs):
        if not self.shadow:
            self.register()
        self.update()
        self.apply_shadow()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class BaseTrainer:
    def __init__(self, cfg, device, prediction_path, eval_only=False, resume_training=False, logger=None):
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(PROJECT_NAME)
        self.device = device
        self.cfg = cfg
        self.max_epochs = self.cfg.train.epochs
        self._init_training()

        self.ckpt_path, self.ckpt_syml_path = self.get_ckpt_path()
        if not os.path.exists(self.ckpt_path):
            eval_only, resume_training = False, False
        self._init_tensorboard(eval_only, resume_training)

        self.start_epoch = 0
        self.early_stop = EarlyStopping(monitor='loss', ckpt_path=self.ckpt_path, syml_path=self.ckpt_syml_path,
                                        **vars(self.cfg.train.early_stop))
        self.evaluator = Evaluator(self.model.TASKS, self.data, prediction_path, self.get_exp_metadata(),
                                   cfg.data.test_flag, cfg.gen_flag, cfg.save_results, self.model.MODEL_TYPE,
                                   self.device)

    def _init_training(self):
        self.max_iters_per_epoch = getattr(self.cfg.train, 'max_iters_per_epoch', float('inf'))

        self.model_class = self.get_model_class()
        self.data = self.load_data()
        # self.trn_data, self.val_data, self.tst_data = self.data.get_dataloaders()

        self.model = self.build_model()
        assert self.model is not None

        self.set_optim_sched()

        # Initialize Exponential Moving Average (EMA) for stable training
        if hasattr(self.cfg.train, 'ema'):
            self.ema = EMA(self.model, getattr(self.cfg.train.ema, 'decay', 0.95))

    def _init_tensorboard(self, eval_only, resume_training, tfb_folder=None):
        if not eval_only:
            if tfb_folder is None:
                tfb_folder = os.path.split(self.ckpt_path)[-1][:-len(CKPT_EXT)]
            if self.cfg.data.test_flag:
                tensorboard_path = os.path.join(LOG_PATH, 'tmp', tfb_folder)
            else:
                tensorboard_path = os.path.join(LOG_PATH, 'tensorboard', tfb_folder)
            # NOTE: Need to ignore_errors as it sometimes fails to remove the entire directory
            if os.path.isdir(tensorboard_path) and not resume_training:
                shutil.rmtree(tensorboard_path, ignore_errors=True)
                os.makedirs(tensorboard_path, exist_ok=True)
            self.tensorboard = SummaryWriter(tensorboard_path)

    def get_exp_metadata(self):
        cfg = self.cfg
        cfg_txt = funcs.dict2txt(funcs.flatten_dict(funcs.namespace2dict(cfg), key_sep='-'), entry_sep='|')
        return {'model_name': cfg.model.name + cfg.model.suffix, 'dataset': cfg.data.dataset, 'seed': cfg.seed,
                'split_ix': cfg.data.fold, 'config': cfg_txt, 'date': datetime.now(timezone.utc).strftime(DATE_FORMAT)}

    @staticmethod
    def get_optimizer(model_params, cfg):
        optim_name = cfg.__dict__.pop('name').lower()
        if optim_name == 'sgd':
            cfg.lr = getattr(cfg, 'lr', 1.0)
            return torch.optim.SGD(model_params, **vars(cfg))
        elif optim_name == 'rmsprop':
            cfg.lr = getattr(cfg, 'lr', 0.02)
            cfg.alpha = getattr(cfg, 'alpha', 0.95)
            torch.optim.RMSprop(model_params, **vars(cfg))
        elif optim_name == 'adam':
            cfg.lr = getattr(cfg, 'lr', 1e-4)
            return torch.optim.Adam(model_params, **vars(cfg))
        elif optim_name == 'adamw':
            cfg.lr = getattr(cfg, 'lr', 1e-3)
            return torch.optim.AdamW(model_params, **vars(cfg))

    @staticmethod
    def get_scheduler(optim, cfg):
        if cfg.name.lower() == 'steplr':
            return torch.optim.lr_scheduler.StepLR(optim, step_size=getattr(cfg, 'step_size', 1),
                                                   gamma=getattr(cfg, 'gamma', 0.25))
        return DummyScheduler(optim)

    def set_optim_sched(self):
        self.optim = self.get_optimizer(self.model.parameters(), cfg=self.cfg.train.optim)
        self.sched = self.get_scheduler(self.optim, cfg=self.cfg.train.scheduler)

    def set_stage(self, stage):
        pass

    def load_best_model(self, ckpt=None, resume_training=False):
        self.logger.info(f'Loading best checkpoint ({self.ckpt_path})...')
        if ckpt is None:
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.start_epoch = ckpt['epoch'] + 1
        if resume_training:
            self.logger.info(f'Training will be resumed at epoch {self.start_epoch}...')
            self.optim.load_state_dict(ckpt['optimizer_state_dict'])
            self.sched.load_state_dict(ckpt['scheduler_state_dict'])
            self.early_stop.setup_checkpoint(ckpt.get('trn_res', {}), ckpt['val_res'], ckpt['epoch'])
        self.model.load_state_dict(ckpt['model_state_dict'])

    def get_ckpt_path(self):
        test_str = '_test' if self.cfg.data.test_flag else ''
        if getattr(self.cfg.model, 'ckpt_ignore_suffix', False):
            model_id = self.cfg.model.name
        else:
            model_id = f'{self.cfg.model.name}{self.cfg.model.suffix}'

        if self.model_class.SEQ_MODE.requires_past_info():
            f_name = f'{model_id}_{self.cfg.data.dataset}_{self.cfg.data.fold}_{self.cfg.data.hist_len}_' \
                     f'{self.cfg.seed}{test_str}'
        else:
            f_name = f'{model_id}_{self.cfg.data.dataset}_{self.cfg.data.fold}_{self.cfg.seed}{test_str}'

        return os.path.join(CKPT_PATH, DATA_MODE, f'{f_name}{CKPT_EXT}'), \
               os.path.join(CKPT_PATH, DATA_MODE, 'best', f'{f_name}{CKPT_EXT}')

    def get_model_class(self):
        module_path = getattr(self.cfg.model, 'path', 'src.models')

        try:
            model_module = importlib.import_module(module_path)
            return getattr(model_module, self.cfg.model.name)
        except Exception:
            raise ValueError(f'Model "{module_path}" could not be found.')

    def build_model(self):
        self.cfg.model.device = self.device
        return self.model_class(self.data.data_info, self.cfg.model, seed=self.cfg.seed, logger=self.logger).to(self.device)

    def load_data(self):
        self.cfg.data.requires_exp = (
                    Task.EXPLANATION in self.model_class.TASKS or getattr(self.cfg.data, 'requires_exp', False))
        self.cfg.data.requires_feat = (
                    Task.FEAT in self.model_class.TASKS or getattr(self.cfg.data, 'requires_feat', False))
        self.cfg.data.requires_rating = (
                    Task.RATING in self.model_class.TASKS or getattr(self.cfg.data, 'requires_rating', False))
        self.cfg.data.requires_context = (
                    Task.CONTEXT in self.model_class.TASKS or getattr(self.cfg.data, 'requires_context', False))
        self.cfg.data.requires_nextitem = (
                    Task.NEXT_ITEM in self.model_class.TASKS or getattr(self.cfg.data, 'requires_nextitem', False))
        self.cfg.data.seq_mode = self.model_class.SEQ_MODE

        if self.model_class.INPUT_TYPE == InputType.REGULAR:
            data_path = 'src.utils.data'
            data_class_name = 'GenDataset'
        elif self.model_class.INPUT_TYPE == InputType.TEMPLATE:
            data_path = 'src.utils.data'
            data_class_name = 'TemplateDataset'
        elif self.model_class.INPUT_TYPE == InputType.SEQUENTIAL:
            data_path = 'src.utils.data'
            data_class_name = 'SequentialDataset'
        elif self.model_class.INPUT_TYPE == InputType.CUSTOM:
            data_path = getattr(self.cfg.data, 'path', 'src.utils.data')
            data_class_name = f'{self.cfg.model.name}Dataset'
        else:
            raise NotImplementedError('The selected model does not have an implemented input type')

        try:
            data_module = importlib.import_module(data_path)
            data_class = getattr(data_module, data_class_name)
        except Exception:
            raise ValueError(f'Unable to find the data class "{data_path}.{data_class_name}" could not be found.')

        return data_class(self.cfg.data, self.cfg.model, device=self.device, seed=self.cfg.seed, logger=self.logger)

    def log_tensorboard(self, results, step):
        for desc, res in results.items():
            if isinstance(res, dict):
                for task, score in res.items():
                    self.tensorboard.add_scalar(f'{desc}/{task}', score, step)
            else:
                self.tensorboard.add_scalar(f'{desc}', res, step)

    def train_epoch(self, trn_data):
        # Turn on training mode which enables dropout.
        self.model.train()
        self.data.train()

        global_loss = defaultdict(float)  # {task: 0 for task in self.model.TASKS}
        total_samples = defaultdict(int)  # {task: 0 for task in self.model.TASKS}
        for bix, batch in enumerate(trn_data):
            # NOTE: CompExp implements a max_iters training argument to limit the number of iterations per epoch.
            if bix >= self.max_iters_per_epoch:
                break

            batch_size = batch['size']

            # Send all tensors in the batch to self.device
            for k, v in batch.items():
                if hasattr(v, 'is_cuda'):
                    batch[k] = v.to(self.device)

            self.optim.zero_grad()
            losses = self.model.compute_loss(batch)
            losses['loss'].backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem.
            if self.cfg.train.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_norm)
            self.optim.step()

            for task in losses:
                # if task != 'loss':
                global_loss[task] += batch_size * losses[task].item()
                total_samples[task] += batch_size

            step = (bix + 1)
            if step % self.cfg.train.log_interval == 0:  # or step == len(trn_data):
                print_losses({task: v / total_samples[task] for task, v in global_loss.items()}, self.device,
                             suffix=f'{step:5d}/{len(trn_data):5d} batches', logger=self.logger)

        global_loss = {task: v / total_samples[task] for task, v in global_loss.items()}
        print_losses(global_loss, self.device, suffix=f'{step:5d}/{len(trn_data):5d} batches', logger=self.logger)

        return global_loss

    def train(self, resume=False):
        if resume:
            if os.path.islink(self.ckpt_syml_path):
                logging.warning(f'Attempting to resume training on an already trained model! Exiting training...')
                return
            self.resume_training()

        epoch_times = {'Train': [], 'Valid': []}
        val_task_res = None
        trn_data = self.data.get_dataloader('train')

        if hasattr(self.cfg.train, 'epoch_log_steps'):
            epoch_steps = min(len(trn_data), self.max_iters_per_epoch)
            self.cfg.train.log_interval = max(int(np.ceil(epoch_steps / self.cfg.train.epoch_log_steps)), 1)

        if self.data.do_eval:
            val_data = self.data.get_dataloader('valid')

        for epoch in range(self.start_epoch, self.max_epochs):
            self.logger.info('Epoch {}'.format(epoch))

            st = time.time()
            trn_task_res = self.train_epoch(trn_data)
            epoch_times['Train'].append(time.time() - st)

            epoch_info = {'Train': trn_task_res, 'LR': self.sched.get_last_lr()[0]}

            if self.data.do_eval:
                st = time.time()
                if hasattr(self.cfg.train, 'ema'):
                    self.ema()

                val_task_res = self.evaluate(val_data, 'valid')
                # Some models compute a validation loss that is different from the training aggregated loss
                if callable(getattr(self.model, 'get_val_loss', None)):
                    val_task_res['agg_loss'] = val_task_res.pop('loss')
                    val_task_res['loss'] = self.model.get_val_loss(val_task_res)

                epoch_times['Valid'].append(time.time() - st)
                epoch_info['Valid'] = val_task_res

                print_losses(val_task_res, self.device, suffix=f'valid loss {val_task_res["loss"]:4.4f}',
                             color=Colors.GREEN, logger=self.logger)

                if hasattr(self.cfg.train, 'ema'):
                    self.ema.restore()

            self.log_tensorboard(epoch_info, epoch)
            torch.cuda.empty_cache()
            if self.data.do_eval:
                if self.early_stop(epoch, trn_task_res, val_task_res, self.model, self.optim, self.sched):
                    break
            elif self.early_stop(epoch, {}, trn_task_res, self.model, self.optim, self.sched):
                break

            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            self.sched.step()
            self.logger.info(colored('Learning rate set to {:2.8f}'.format(self.sched.get_last_lr()[0]), Colors.YELLOW))

        self.logger.info(f'Avg. Train Time: {timedelta(seconds=np.mean(epoch_times["Train"]))} ||| '
                         f'Avg. Valid Time: {timedelta(seconds=np.mean(epoch_times["Valid"]))}')

        return self.early_stop.get_best_results()

    def test(self):
        self.logger.info('=' * 89)
        if not os.path.isfile(self.ckpt_syml_path):
            logging.warning('Attempting to evaluate a model that was not fully trained!')
        self.load_best_model()

        # Run on test data.
        tst_data = self.data.get_dataloader('test')
        test_task_losses = self.evaluate(tst_data, 'test')
        print_losses(test_task_losses, self.device, suffix='End of training', logger=self.logger)

        self.logger.info('Generating text')
        self.generate(tst_data)

    @torch.no_grad()
    def evaluate(self, data, phase):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        self.data.eval(phase)

        global_loss = defaultdict(float)  # {task: 0 for task in self.model.TASKS}
        global_loss['loss'] = 0
        total_sample = 0
        for bix, batch in enumerate(data):
            # if bix == 55:
            #     break
            batch_size = batch['size']

            # Send all tensors in the batch to self.device
            for k, v in batch.items():
                if hasattr(v, 'is_cuda'):
                    batch[k] = v.to(self.device)

            # self.optim.zero_grad()
            task_loss = self.model.compute_loss(batch)

            for task in task_loss:
                global_loss[task] += batch_size * task_loss[task].item()

            total_sample += batch_size

        return {task: v / total_sample for task, v in global_loss.items()}

    @torch.no_grad()
    def generate(self, data, gen_steps=0, phase='test'):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        self.data.eval(phase)

        preds = defaultdict(list)
        labels = defaultdict(list)
        for bix, batch in enumerate(data):
            # Send all tensors in the batch to self.device
            for k, v in batch.items():
                if hasattr(v, 'is_cuda'):
                    batch[k] = v.to(self.device)

            # NOTE: The results returned by mode.generate must be casted to list
            res, m_labels = self.model.generate(batch)

            for task in res:
                preds[task].extend(res[task])

            if gen_steps != 0 and (bix + 1) >= gen_steps:
                break

        labels.update(self.data.get_gen_labels(gen_steps, len(data), data.batch_size, preds.keys(), phase))

        # Free some memory
        del batch, data
        torch.cuda.empty_cache()
        gc.collect()

        if hasattr(self.model, 'postprocess'):
            self.logger.info('Starting Post-processing step...')
            preds = self.model.postprocess(preds, labels)

        if gen_steps == 0:
            self.evaluator.evaluate(labels, preds)
        else:
            self.evaluator.print_generated(labels, preds)

    def resume_training(self):
        self.load_best_model(resume_training=True)
