import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import logging
import logging.handlers
import copy
import math
import numpy as np
import random
import time

from calflops import calculate_flops

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.amp.grad_scaler import GradScaler

from tools.muon_optimizer import MuonAdamW, MuonSGD


def parse_args_example():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--string-variable',
                        type=str,
                        default='string',
                        help='explain variable')
    parser.add_argument('--float-variable',
                        type=float,
                        default=0.01,
                        help='explain variable')
    parser.add_argument('--int-variable',
                        type=int,
                        default=10,
                        help='explain variable')
    parser.add_argument('--list-variable',
                        type=list,
                        default=[1, 10, 100],
                        help='explain variable')
    # store_true即命令行有这个参数时，值为True,没有这个参数时，默认值为False
    # store_false即命令行有这个参数时，值为False,没有这个参数时，默认值为True
    parser.add_argument('--bool-variable',
                        default=False,
                        action='store_true',
                        help='explain variable')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='LOCAL_PROCESS_RANK in DistributedDataParallel model')

    return parser.parse_args()


def get_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_name = os.path.join(log_dir, '{}.info.log'.format(name))
    file_handler = logging.handlers.TimedRotatingFileHandler(file_name,
                                                             when='D',
                                                             interval=365,
                                                             backupCount=0,
                                                             encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def worker_seed_init_fn(worker_id, num_workers, global_rank, seed):
    worker_seed = num_workers * global_rank + worker_id + seed + int(
        time.time())
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_macs_and_params(config, model):
    assert isinstance(config.input_image_size, int) == True or isinstance(
        config.input_image_size,
        list) == True, 'Illegal input_image_size type!'

    if isinstance(config.input_image_size, int):
        macs_input = torch.randn(1, 3, config.input_image_size,
                                 config.input_image_size).cpu()
    elif isinstance(config.input_image_size, list):
        macs_input = torch.randn(1, 3, config.input_image_size[0],
                                 config.input_image_size[1]).cpu()

    model = model.cpu()

    flops, macs, params = calculate_flops(model=model,
                                          args=[
                                              macs_input,
                                          ],
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)

    return flops, macs, params


class EmaModel(nn.Module):

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        super(EmaModel, self).__init__()
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            if hasattr(model, '_orig_mod'):
                model = model._orig_mod

            model_requires_grad = {}
            for name, param in model.named_parameters():
                model_requires_grad[name] = param.requires_grad

            # Use key-based indexing instead of zip to avoid order assumptions
            model_state_dict = model.state_dict()
            for name, ema_v in self.ema_model.state_dict().items():
                model_v = model_state_dict[name]
                assert ema_v.shape == model_v.shape, 'wrong ema model!'
                # For non-floating-point types, directly copy instead of EMA
                if not ema_v.dtype.is_floating_point:
                    ema_v.copy_(model_v.detach())
                # For buffers (not in named_parameters), directly copy
                elif name not in model_requires_grad:
                    ema_v.copy_(model_v.detach())
                # For frozen parameters (requires_grad=False), directly copy
                elif not model_requires_grad[name]:
                    ema_v.copy_(model_v.detach())
                else:
                    ema_v.mul_(d).add_((1 - d) * model_v.detach())


class DeepSpeedEmaModel:

    def __init__(self,
                 model_engine,
                 config,
                 decay=0.9999,
                 tau=2000,
                 updates=0):
        self.zero_stage = config.deepspeed_zero_stage
        self.use_compile = config.use_compile
        self.updates = updates
        self.decay = decay
        self.tau = tau

        module = model_engine.module._orig_mod if self.use_compile else model_engine.module

        if self.zero_stage == 3:
            # ZeRO-3: store EMA of each parameter's local shard
            self.ema_params = {}
            for name, param in module.named_parameters():
                # param.ds_tensor is the local shard held by this rank
                self.ema_params[name] = param.ds_tensor.detach().clone().float(
                )
        else:
            # ZeRO-0/1/2: store full state_dict as EMA
            self.ema_state_dict = {
                k: v.detach().clone().float()
                for k, v in module.state_dict().items()
            }

    @torch.no_grad()
    def update(self, model_engine):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))

        module = model_engine.module._orig_mod if self.use_compile else model_engine.module

        if self.zero_stage == 3:
            # ZeRO-3: only named_parameters() are tracked (no buffers),
            # and all EMA values are stored as float32, so no need to
            # check is_floating_point or buffer membership.
            for name, param in module.named_parameters():
                ema_v = self.ema_params[name]
                local_data = param.ds_tensor
                # For frozen parameters (requires_grad=False), directly copy
                if not param.requires_grad:
                    ema_v.copy_(local_data.detach().float())
                else:
                    ema_v.mul_(d).add_((1 - d) * local_data.detach().float())
        else:
            model_requires_grad = {}
            for name, param in module.named_parameters():
                model_requires_grad[name] = param.requires_grad

            model_state_dict = module.state_dict()
            for key, ema_v in self.ema_state_dict.items():
                model_v = model_state_dict[key]
                assert ema_v.shape == model_v.shape, 'wrong ema model!'
                # For non-floating-point types, directly copy instead of EMA
                # Check model_v's dtype (not ema_v) since ema_v is always
                # float32 after .float() conversion in __init__
                if not model_v.dtype.is_floating_point:
                    ema_v.copy_(model_v.detach().float())
                # For buffers (not in named_parameters), directly copy
                elif key not in model_requires_grad:
                    ema_v.copy_(model_v.detach().float())
                # For frozen parameters (requires_grad=False), directly copy
                elif not model_requires_grad[key]:
                    ema_v.copy_(model_v.detach().float())
                else:
                    ema_v.mul_(d).add_((1 - d) * model_v.detach().float())

    @torch.no_grad()
    def get_ema_model_state_dict(self, model_engine):
        module = model_engine.module._orig_mod if self.use_compile else model_engine.module

        if self.zero_stage == 3:
            ema_full_state_dict = {}
            world_size = torch.distributed.get_world_size()

            # Maps a parameter's unique id to the first key that holds its
            # gathered EMA tensor. Tied weights (e.g. a VLM's lm_head.weight
            # and embed_tokens.weight) share one nn.Parameter, so without
            # this they would be all_gathered twice and stored as two
            # independent tensors, silently doubling both the communication
            # and the host memory for the largest weight in the model.
            #
            # Note that named_parameters() itself de-duplicates by default
            # (remove_duplicate=True) and would therefore only yield the
            # FIRST name of a tied group, leaving the other key missing from
            # the saved state_dict entirely. remove_duplicate=False restores
            # every key, and the map below keeps them sharing one tensor --
            # exactly what nn.Module.state_dict() produces.
            shared_params = {}

            for name, param in module.named_parameters(remove_duplicate=False):
                param_key = getattr(param, 'ds_id', None)
                if param_key is None:
                    param_key = id(param)

                if param_key in shared_params:
                    ema_full_state_dict[name] = ema_full_state_dict[
                        shared_params[param_key]]
                    continue

                ema_local = self.ema_params[name]
                # all_gather local EMA shards from all ranks
                gathered = [
                    torch.zeros_like(ema_local) for _ in range(world_size)
                ]
                torch.distributed.all_gather(gathered, ema_local)
                full_param = torch.cat(gathered, dim=0)
                # ZeRO-3 pads shards to equal size; truncate to original numel
                ema_full_state_dict[
                    name] = full_param[:param.ds_numel].reshape(
                        param.ds_shape).cpu().clone()
                shared_params[param_key] = name

            # Also include non-parameter buffers from the model.
            # Buffers are replicated (never partitioned) under ZeRO-3, so they
            # can be read directly. Non-persistent buffers are skipped, exactly
            # as nn.Module.state_dict() does, so that the produced key
            # namespace stays byte-for-byte the one module.state_dict()
            # yields. A plain named_buffers() call cannot do this: it returns
            # dotted names, while _non_persistent_buffers_set holds the bare
            # name on each owning submodule. Walking the modules one by one
            # keeps both sides in the same namespace.
            for per_module_name, per_module in module.named_modules():
                per_module_prefix = f'{per_module_name}.' if per_module_name else ''
                for k, v in per_module.named_buffers(recurse=False):
                    if v is not None and k not in per_module._non_persistent_buffers_set:
                        ema_full_state_dict[per_module_prefix +
                                            k] = v.detach().cpu().clone()

            return ema_full_state_dict
        else:
            return {k: v.cpu().clone() for k, v in self.ema_state_dict.items()}

    def state_dict(self):
        state = {
            'updates': self.updates,
            'decay': self.decay,
            'tau': self.tau,
        }
        if self.zero_stage == 3:
            state['ema_params'] = {
                k: v.cpu().clone()
                for k, v in self.ema_params.items()
            }
        else:
            state['ema_state_dict'] = {
                k: v.cpu().clone()
                for k, v in self.ema_state_dict.items()
            }

        return state

    def load_state_dict(self, state):
        self.updates = state['updates']
        self.decay = state['decay']
        self.tau = state['tau']
        if self.zero_stage == 3:
            for k, v in state['ema_params'].items():
                if k in self.ema_params:
                    self.ema_params[k].copy_(v.to(self.ema_params[k].device))
        else:
            for k, v in state['ema_state_dict'].items():
                if k in self.ema_state_dict:
                    self.ema_state_dict[k].copy_(
                        v.to(self.ema_state_dict[k].device))


def build_training_mode(config, model):
    ema_model, scaler = None, None
    if config.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()

    if hasattr(config, 'find_unused_parameters'):
        find_unused_parameters = config.find_unused_parameters
    else:
        find_unused_parameters = False

    local_rank = config.local_rank
    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        ema_model = EmaModel(model,
                             decay=config.ema_model_decay,
                             tau=config.ema_model_tau)
        ema_model.ema_model = nn.parallel.DistributedDataParallel(
            ema_model.ema_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True)

    if hasattr(config, 'use_amp') and config.use_amp:
        if config.amp_type == torch.bfloat16:
            scaler = GradScaler(enabled=False)
        else:
            scaler = GradScaler(enabled=True)

    return model, ema_model, scaler


class Scheduler:

    def __init__(self, config, optimizer):
        self.scheduler_name = config.scheduler[0]
        self.scheduler_parameters = config.scheduler[1]
        self.warm_up_epochs = self.scheduler_parameters['warm_up_epochs']
        self.epochs = config.epochs
        self.optimizer_parameters = config.optimizer[1]
        self.lr = self.optimizer_parameters['lr']
        self.current_lr = self.lr

        self.init_param_groups_lr = [
            param_group["lr"] for param_group in optimizer.param_groups
        ]

        assert self.scheduler_name in ['MultiStepLR', 'CosineLR',
                                       'PolyLR'], 'Unsupported scheduler!'
        assert self.warm_up_epochs >= 0, 'Illegal warm_up_epochs!'
        assert self.epochs > 0, 'Illegal epochs!'

    def step(self, optimizer, epoch):
        if self.scheduler_name == 'MultiStepLR':
            gamma = self.scheduler_parameters['gamma']
            milestones = self.scheduler_parameters['milestones']
        elif self.scheduler_name == 'CosineLR':
            min_lr = 0. if 'min_lr' not in self.scheduler_parameters.keys(
            ) else self.scheduler_parameters['min_lr']
        elif self.scheduler_name == 'PolyLR':
            power = self.scheduler_parameters['power']
            min_lr = 0. if 'min_lr' not in self.scheduler_parameters.keys(
            ) else self.scheduler_parameters['min_lr']

        assert len(self.init_param_groups_lr) == len(optimizer.param_groups)

        for idx, param_group in enumerate(optimizer.param_groups):
            param_group_init_lr = self.init_param_groups_lr[idx]

            # Warmup phase: only when warm_up_epochs > 0 and epoch < warm_up_epochs
            if self.warm_up_epochs > 0 and epoch < self.warm_up_epochs:
                param_group_current_lr = (
                    epoch / self.warm_up_epochs) * param_group_init_lr
            else:
                if self.scheduler_name == 'MultiStepLR':
                    param_group_current_lr = gamma**len([
                        m for m in milestones if m <= epoch
                    ]) * param_group_init_lr
                elif self.scheduler_name == 'CosineLR':
                    param_group_current_lr = 0.5 * (math.cos(
                        (epoch - self.warm_up_epochs) /
                        (self.epochs - self.warm_up_epochs) * math.pi) + 1) * (
                            param_group_init_lr - min_lr) + min_lr
                elif self.scheduler_name == 'PolyLR':
                    param_group_current_lr = (
                        (1 - (epoch - self.warm_up_epochs) /
                         (self.epochs - self.warm_up_epochs))**
                        power) * (param_group_init_lr - min_lr) + min_lr

            param_group["lr"] = param_group_current_lr

        # Update self.current_lr for logging (using the global base lr)
        if self.warm_up_epochs > 0 and epoch < self.warm_up_epochs:
            self.current_lr = (epoch / self.warm_up_epochs) * self.lr
        else:
            if self.scheduler_name == 'MultiStepLR':
                self.current_lr = gamma**len(
                    [m for m in milestones if m <= epoch]) * self.lr
            elif self.scheduler_name == 'CosineLR':
                self.current_lr = 0.5 * (math.cos(
                    (epoch - self.warm_up_epochs) /
                    (self.epochs - self.warm_up_epochs) * math.pi) +
                                         1) * (self.lr - min_lr) + min_lr
            elif self.scheduler_name == 'PolyLR':
                self.current_lr = ((1 - (epoch - self.warm_up_epochs) /
                                    (self.epochs - self.warm_up_epochs))**
                                   power) * (self.lr - min_lr) + min_lr

    def state_dict(self):
        return {
            'scheduler_name': self.scheduler_name,
            'warm_up_epochs': self.warm_up_epochs,
            'epochs': self.epochs,
            'lr': self.lr,
            'current_lr': self.current_lr,
            'init_param_groups_lr': self.init_param_groups_lr,
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def build_optimizer(config, model):
    optimizer_name = config.optimizer[0]
    optimizer_parameters = config.optimizer[1]
    assert optimizer_name in ['SGD', 'AdamW', 'MuonAdamW',
                              'MuonSGD'], 'Unsupported optimizer!'

    lr = optimizer_parameters['lr']
    weight_decay = optimizer_parameters['weight_decay']

    # if global_weight_decay = False,set 1d parms weight decay = 0.
    global_weight_decay = True if 'global_weight_decay' not in optimizer_parameters.keys(
    ) else optimizer_parameters['global_weight_decay']

    # if global_weight_decay = True,no_weight_decay_layer_name_list can't be set.
    no_weight_decay_layer_name_list = []
    if 'no_weight_decay_layer_name_list' in optimizer_parameters.keys(
    ) and isinstance(optimizer_parameters['no_weight_decay_layer_name_list'],
                     list):
        no_weight_decay_layer_name_list = optimizer_parameters[
            'no_weight_decay_layer_name_list']

    param_layer_name_list = []
    param_layer_weight_dict = {}
    param_layer_decay_dict, param_layer_lr_dict = {}, {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_layer_name_list.append(name)
        param_layer_weight_dict[name] = param

        if global_weight_decay is False:
            if param.ndim <= 1 or any(no_weight_decay_layer_name in name
                                      for no_weight_decay_layer_name in
                                      no_weight_decay_layer_name_list):
                param_layer_decay_dict[name] = 0.
            else:
                per_layer_weight_decay = weight_decay
                if 'sub_layer_weight_decay' in optimizer_parameters.keys(
                ) and isinstance(
                        optimizer_parameters['sub_layer_weight_decay'], dict):
                    for per_sub_layer_name_prefix, per_sub_layer_weight_decay in optimizer_parameters[
                            'sub_layer_weight_decay'].items():
                        if per_sub_layer_name_prefix in name:
                            per_layer_weight_decay = per_sub_layer_weight_decay
                            break
                param_layer_decay_dict[name] = per_layer_weight_decay
        else:
            param_layer_decay_dict[name] = weight_decay

        per_layer_lr = lr
        if 'sub_layer_lr' in optimizer_parameters.keys() and isinstance(
                optimizer_parameters['sub_layer_lr'], dict):
            for per_sub_layer_name_prefix, per_sub_layer_lr in optimizer_parameters[
                    'sub_layer_lr'].items():
                if per_sub_layer_name_prefix in name:
                    per_layer_lr = per_sub_layer_lr
                    break
        param_layer_lr_dict[name] = per_layer_lr

    assert len(param_layer_name_list) == len(param_layer_weight_dict) == len(
        param_layer_decay_dict) == len(param_layer_lr_dict)

    unique_decays = list(set(param_layer_decay_dict.values()))
    unique_lrs = list(set(param_layer_lr_dict.values()))

    lr_weight_decay_combination = []
    for per_decay in unique_decays:
        for per_lr in unique_lrs:
            lr_weight_decay_combination.append([per_decay, per_lr])

    model_params_weight_decay_list = []
    model_layer_weight_decay_list = []
    for per_decay, per_lr in lr_weight_decay_combination:
        per_decay_lr_param_list, per_decay_lr_name_list = [], []
        for per_layer_name in param_layer_name_list:
            per_layer_weight = param_layer_weight_dict[per_layer_name]
            per_layer_weight_decay = param_layer_decay_dict[per_layer_name]
            per_layer_lr = param_layer_lr_dict[per_layer_name]

            if per_layer_weight_decay == per_decay and per_layer_lr == per_lr:
                per_decay_lr_param_list.append(per_layer_weight)
                per_decay_lr_name_list.append(per_layer_name)

        assert len(per_decay_lr_param_list) == len(per_decay_lr_name_list)

        if len(per_decay_lr_param_list) > 0:
            model_params_weight_decay_list.append({
                'params': per_decay_lr_param_list,
                'weight_decay': per_decay,
                'lr': per_lr,
            })
            model_layer_weight_decay_list.append({
                'name': per_decay_lr_name_list,
                'weight_decay': per_decay,
                'lr': per_lr,
            })

    assert len(model_params_weight_decay_list) == len(
        model_layer_weight_decay_list)

    if optimizer_name == 'SGD':
        momentum = 0.9 if 'momentum' not in optimizer_parameters.keys(
        ) else optimizer_parameters['momentum']
        nesterov = False if 'nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['nesterov']
        return torch.optim.SGD(
            model_params_weight_decay_list,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov), model_layer_weight_decay_list

    elif optimizer_name == 'MuonSGD':
        # Note: MuonSGD uses unified lr for all parameters.
        # Per-layer lr settings from optimizer_parameters are not applied.
        # MuonSGD optimizer don't support sub_layer_lr/sub_layer_weight_decay

        exclude_muon_layer_name_list = [
            'position_encoding',
            'cls_token',
            'patch_embedding',
            'embed',
            'lm_head',
            'merger',
        ]
        if 'exclude_muon_layer_name_list' in optimizer_parameters.keys(
        ) and isinstance(optimizer_parameters['exclude_muon_layer_name_list'],
                         list):
            exclude_muon_layer_name_list = exclude_muon_layer_name_list + optimizer_parameters[
                'exclude_muon_layer_name_list']

        # Separate parameters into muon_params, sgd_params and sgd_nowd_params
        muon_param_list, muon_param_names = [], []
        sgd_param_list, sgd_param_names = [], []
        sgd_nowd_param_list, sgd_nowd_param_names = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Muon is used for 2D parameters that are not in exclude list.
            # The name is lowercased before matching, exactly like
            # deepspeed.set_optimizer_flags does.
            use_muon = (
                param.ndim >= 2
                and not any(exclude_name in name.lower()
                            for exclude_name in exclude_muon_layer_name_list))

            if use_muon:
                muon_param_list.append(param)
                muon_param_names.append(name)
            elif global_weight_decay is False and (param.ndim <= 1 or any(
                    no_weight_decay_layer_name in name
                    for no_weight_decay_layer_name in
                    no_weight_decay_layer_name_list)):
                sgd_nowd_param_list.append(param)
                sgd_nowd_param_names.append(name)
            else:
                sgd_param_list.append(param)
                sgd_param_names.append(name)

        # Create summary for model_layer_weight_decay_list
        model_layer_weight_decay_list = []
        if len(muon_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': muon_param_names,
                'optimizer': 'MuonSGD(Muon)',
                'lr': lr,
                'weight_decay': weight_decay,
            })
        if len(sgd_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': sgd_param_names,
                'optimizer': 'MuonSGD(SGD)',
                'lr': lr,
                'weight_decay': weight_decay,
            })
        if len(sgd_nowd_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': sgd_nowd_param_names,
                'optimizer': 'MuonSGD(SGD)',
                'lr': lr,
                'weight_decay': 0.,
            })

        momentum = 0.95 if 'momentum' not in optimizer_parameters.keys(
        ) else optimizer_parameters['momentum']
        nesterov = True if 'nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['nesterov']
        ns_steps = 5 if 'ns_steps' not in optimizer_parameters.keys(
        ) else optimizer_parameters['ns_steps']

        sgd_momentum = 0.9 if 'sgd_momentum' not in optimizer_parameters.keys(
        ) else optimizer_parameters['sgd_momentum']
        sgd_nesterov = False if 'sgd_nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['sgd_nesterov']

        return MuonSGD(
            lr=lr,
            wd=weight_decay,
            muon_params=muon_param_list,
            sgd_params=sgd_param_list,
            sgd_nowd_params=sgd_nowd_param_list,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            sgd_momentum=sgd_momentum,
            sgd_nesterov=sgd_nesterov), model_layer_weight_decay_list

    elif optimizer_name == 'AdamW':
        beta1 = 0.9 if 'beta1' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta1']
        beta2 = 0.999 if 'beta2' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta2']
        eps = 1e-08 if 'eps' not in optimizer_parameters.keys(
        ) else optimizer_parameters['eps']
        return torch.optim.AdamW(model_params_weight_decay_list,
                                 lr=lr,
                                 betas=(beta1, beta2),
                                 eps=eps), model_layer_weight_decay_list

    elif optimizer_name == 'MuonAdamW':
        # Note: MuonAdamW uses unified lr for all parameters.
        # Per-layer lr settings from optimizer_parameters are not applied.
        # MuonAdamW optimizer don't support sub_layer_lr/sub_layer_weight_decay

        exclude_muon_layer_name_list = [
            'position_encoding',
            'cls_token',
            'patch_embedding',
            'embed',
            'lm_head',
            'merger',
        ]
        if 'exclude_muon_layer_name_list' in optimizer_parameters.keys(
        ) and isinstance(optimizer_parameters['exclude_muon_layer_name_list'],
                         list):
            exclude_muon_layer_name_list = exclude_muon_layer_name_list + optimizer_parameters[
                'exclude_muon_layer_name_list']

        # Separate parameters into muon_params, adamw_params and adamw_nowd_params
        muon_param_list, muon_param_names = [], []
        adamw_param_list, adamw_param_names = [], []
        adamw_nowd_param_list, adamw_nowd_param_names = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Muon is used for 2D parameters that are not in exclude list.
            # The name is lowercased before matching, exactly like
            # deepspeed.set_optimizer_flags does.
            use_muon = (
                param.ndim >= 2
                and not any(exclude_name in name.lower()
                            for exclude_name in exclude_muon_layer_name_list))

            if use_muon:
                muon_param_list.append(param)
                muon_param_names.append(name)
            elif global_weight_decay is False and (param.ndim <= 1 or any(
                    no_weight_decay_layer_name in name
                    for no_weight_decay_layer_name in
                    no_weight_decay_layer_name_list)):
                adamw_nowd_param_list.append(param)
                adamw_nowd_param_names.append(name)
            else:
                adamw_param_list.append(param)
                adamw_param_names.append(name)

        # Create summary for model_layer_weight_decay_list
        model_layer_weight_decay_list = []
        if len(muon_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': muon_param_names,
                'optimizer': 'MuonAdamW(Muon)',
                'lr': lr,
                'weight_decay': weight_decay,
            })
        if len(adamw_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': adamw_param_names,
                'optimizer': 'MuonAdamW(AdamW)',
                'lr': lr,
                'weight_decay': weight_decay,
            })
        if len(adamw_nowd_param_names) > 0:
            model_layer_weight_decay_list.append({
                'name': adamw_nowd_param_names,
                'optimizer': 'MuonAdamW(AdamW)',
                'lr': lr,
                'weight_decay': 0.,
            })

        momentum = 0.95 if 'momentum' not in optimizer_parameters.keys(
        ) else optimizer_parameters['momentum']
        nesterov = True if 'nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['nesterov']
        ns_steps = 5 if 'ns_steps' not in optimizer_parameters.keys(
        ) else optimizer_parameters['ns_steps']

        adamw_beta1 = 0.9 if 'adamw_beta1' not in optimizer_parameters.keys(
        ) else optimizer_parameters['adamw_beta1']
        adamw_beta2 = 0.999 if 'adamw_beta2' not in optimizer_parameters.keys(
        ) else optimizer_parameters['adamw_beta2']
        adamw_eps = 1e-08 if 'adamw_eps' not in optimizer_parameters.keys(
        ) else optimizer_parameters['adamw_eps']

        return MuonAdamW(lr=lr,
                         wd=weight_decay,
                         muon_params=muon_param_list,
                         adamw_params=adamw_param_list,
                         adamw_nowd_params=adamw_nowd_param_list,
                         momentum=momentum,
                         nesterov=nesterov,
                         ns_steps=ns_steps,
                         adamw_betas=(adamw_beta1, adamw_beta2),
                         adamw_eps=adamw_eps), model_layer_weight_decay_list
