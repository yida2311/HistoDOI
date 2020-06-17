from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from models.backbone import resnet50, resnext50_32x4d, efficientnet_b0, efficientnet_b2
from models.utils import Parallel2Single
from utils.metrics import AverageMeter, ConfusionMatrixCls, ConfusionMatrixSeg


def get_optimizer(model, learning_rate=1e-2):
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95, weight_decay=1e-4)
    return optimizer

def create_model_load_weights(model, evaluation=False, ckpt_path=None):
    if evaluation and ckpt_path:
        state_dict = torch.load(ckpt_path) #()
        model_dict = model.state_dict()
        state_dict = {k:v for k, v in state_dict.items() if k in model_dict}
        model.load_state_dict(state_dict)

    model = model.cuda()

    return model


class Trainer(object):
    def __init__(self, criterion, optimizer, n_class):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = ConfusionMatrixCls(n_class)
        self.n_class = n_class
    
    def get_scores(self):
        return self.metrics.get_scores()
    
    def reset_metrics(self):
        self.metrics.reset()
    
    def train(self, sample, model):
        imgs = sample['image']
        labels = sample['label']
        model.train()

        imgs = imgs.cuda()
        labels_npy = np.array(labels)
        labels = labels.cuda()

        preds = model.forward(imgs)
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        outputs = preds.cpu().detach().numpy()
        predictions = np.argmax(outputs, axis=1)

        self.metrics.update(labels_npy, predictions)

        return loss 


class Evaluator(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrixCls(n_class)
        self.n_class = n_class
    
    def get_scores(self):
        return self.metrics.get_scores()
    
    def reset_metrics(self):
        self.metrics.reset()
    
    def eval(self, sample, model):
        imgs = sample['image']
        labels = sample['label']
        with torch.no_grad():
            imgs = imgs.cuda()
            labels_npy = np.array(labels)

            preds = model.forward(imgs)
            outputs = preds.cpu().detach().numpy()
            predictions = np.argmax(outputs, axis=1)

            self.metrics.update(labels_npy, predictions)
        
        return predictions
    
    def test(self, sample, model):
        imgs = sample['image']
        with torch.no_grad():
            imgs = imgs.cuda()
            preds = model.forward(imgs)
            outputs = preds.cpu().detach().numpy()
            predictions = np.argmax(outputs, axis=1)
        
        return predictions


class SlideEvaluator(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrixSeg(n_class+1)
        self.n_class = n_class
    
    def get_scores(self):
        return self.metrics.get_scores()
    
    def reset_metrics(self):
        self.metrics.reset()
    
    def update_scores(self, mask, output):
        self.metrics.update(mask, output)
    
    def eval(self, sample, model, output):
        imgs = sample['image']
        coord = sample['coord']
        with torch.no_grad():
            imgs = imgs.cuda()
            preds = model.forward(imgs)
            outputs = preds.cpu().detach().numpy()
            predictions = np.argmax(outputs, axis=1)

            for i in range(imgs.shape[0]):
                output[coord[i][0], coord[i][1]] = predictions[i] + 1
        
        return output
    
    
    

    

