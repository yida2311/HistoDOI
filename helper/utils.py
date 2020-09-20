import torch
import time
from collections import OrderedDict
from torch import nn


def create_model_load_weights(model, device, distributed=False, local_rank=0, evaluation=False, ckpt_path=None):
    if evaluation and ckpt_path:  # load checkpoint
        state_dict = torch.load(ckpt_path)
        if 'module' in next(iter(state_dict)):
            state_dict = Parallel2Single(state_dict)
        state = model.state_dict()
        state.update(state_dict)
        model.load_state_dict(state)
    
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model.to(device)

    return model


def create_model_load_weights_v2(model, device, distributed=False, local_rank=0, evaluation=False, ckpt_path=None):
    if evaluation and ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
        if 'module' in next(iter(state_dict)):
            state_dict = Parallel2Single(state_dict)
        state = model.state_dict()
        state.update(state_dict)
        model.load_state_dict(state)
    
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model.to(device)

    return model


def get_optimizer(model, learning_rate=2e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    return optimizer

def struct_time():
    # 格式化成2020-08-07 16:56:32
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return cur_time

def simple_time():
    cur_time = time.strftime("[%m-%d-%H]", time.localtime())
    return cur_time


def Parallel2Single(original_state):
    converted = OrderedDict()
    for k, v in original_state.items():
        name = k[7:]
        converted[name] = v
    return converted


def load_state_dict(src, target):
    # pdb.set_trace()
    for k,v in src.items():
        if 'bn' in k:
            continue
        if k in target.state_dict().keys():
            try:
                v = v.numpy()
            except RuntimeError:
                v = v.detach().numpy()
            try:
                target.state_dict()[k].copy_(torch.from_numpy(v))
            except:
                print("{} skipped".format(k))
                continue   
    set_requires_grad(target, True)
    return target