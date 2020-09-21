import torch
import time
from collections import OrderedDict
from torch import nn
import numpy as np 


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


class ConfusionMatrixSeg(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        if not isinstance(label_preds, list):
            self.confusion_matrix += self._fast_hist(label_trues.flatten(), label_preds.flatten(), self.n_classes)
        else:
            for lt, lp in zip(label_trues, label_preds):
                tmp = self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
                self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - IoU
            - mean IoU
            - dice
            - mean dice
            - IoU based on frequency
        """
        hist = self.confusion_matrix
        # accuracy is recall/sensitivity for each class, predicted TP / all real positives
        # axis in sum: perform summation along
        acc = np.nan_to_num(np.diag(hist) / hist.sum(axis=1))
        acc_mean = np.mean(np.nan_to_num(acc[1:]))
        
        intersect = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iou = intersect / union
        mean_iou = np.mean(np.nan_to_num(iou[1:]))
        
        dice = 2 * intersect / (hist.sum(axis=1) + hist.sum(axis=0))
        mean_dice = np.mean(np.nan_to_num(dice[1:]))

        freq = hist.sum(axis=1) / hist.sum() # freq of each target
        # fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        freq_iou = (freq * iou).sum()

        return {'accuracy': acc,
                'accuracy_mean': acc_mean,
                'iou': iou, 
                'iou_mean': mean_iou, 
                'dice': dice,
                'dice_mean': mean_dice,
                'freqw_iou': freq_iou,
                }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def class_to_RGB(label):
    h, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(h, w, 3)).astype(np.uint8)

    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 0]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 0]
    indices = np.where(label == 3)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 255]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]

    return colmap

def collate(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [b[key] for b in batch]
    batch_dict['image'] = torch.stack(batch_dict['image'], dim=0)
    if 'mask' in batch_dict.keys():
        batch_dict['mask'] = torch.stack(batch_dict['mask'], dim=0)

    return batch_dict