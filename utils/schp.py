import os
import torch
from torch.nn import SyncBatchNorm

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, SyncBatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, SyncBatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, SyncBatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, SyncBatchNorm):
        module.momentum = momenta[module]


def bn_re_estimate(loader, model):
    if not check_bn(model):
        print('No batch norm layer detected')
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for i_iter, batch in enumerate(loader):
        # images, labels, _ = batch
        # b = images.data.size(0)
        images = batch['image']
        b = images.size(0)
        
        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum
        model(images)
        n += b
    model.apply(lambda module: _set_momenta(module, momenta))


def save_schp_checkpoint(states, is_best_parsing, output_dir, filename='schp_checkpoint.pth'):
    save_path = os.path.join(output_dir, filename)
    if os.path.exists(save_path):
        os.remove(save_path)
    torch.save(states, save_path)
    if is_best_parsing and 'state_dict' in states:
        best_save_path = os.path.join(output_dir, 'model_parsing_best.pth')
        # if os.path.exists(best_save_path):
        #     os.remove(best_save_path)
        torch.save(states, best_save_path)