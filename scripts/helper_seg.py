from __future__ import absolute_import, division, print_function

import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from models.segmentor.fpn import fpn_bilinear_resnet50
from models.utils import Parallel2Single
from utils.metrics import ConfusionMatrixSeg, AverageMeter


def create_model_load_weights(model, evaluation=False, ckpt_path=None):
    if evaluation and ckpt_path:
            state = model.state_dict()
            state.update(torch.load(ckpt_path))
            model.load_state_dict(state)
    # model = nn.DataParallel(model)
    model = model.cuda()

    return model

def get_optimizer(model, learning_rate=2e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    return optimizer


class Trainer(object):
    def __init__(self, criterion, optimizer, n_class):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = ConfusionMatrixSeg(n_class)
        self.n_class = n_class

    def get_scores(self):
        score_train = self.metrics.get_scores()
        
        return score_train
    def reset_metrics(self):
        self.metrics.reset()
    
    def train(self, sample, model):
        model.train()
        imgs = sample['image']
        masks = sample['mask'].squeeze(1)

        imgs = imgs.cuda()
        masks_npy = np.array(masks)
        masks = masks.cuda()

        preds = model.forward(imgs)
        loss = self.criterion(preds, masks)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        outputs = preds.cpu().detach().numpy()
        predictions = np.argmax(outputs, axis=1)
        self.metrics.update(masks_npy, predictions)
        
        return loss


class Evaluator(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrixSeg(n_class)
        self.n_class = n_class

    def get_scores(self):
        score_train = self.metrics.get_scores()

        return score_train
    
    def reset_metrics(self):
        self.metrics.reset()
    
    def eval(self, sample, model):
        imgs = sample['image']
        masks = sample['mask']
        with torch.no_grad():
            imgs = imgs.cuda()
            masks_npy = np.array(masks)

            preds = model.forward(imgs)
            outputs = preds.cpu().detach().numpy()
            predictions = np.argmax(outputs, axis=1)

            self.metrics.update(masks_npy, predictions)
        
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
        self.metrics = ConfusionMatrixSeg(n_class)
        self.n_class = n_class
    
    def get_scores(self):
        return self.metrics.get_scores()
    
    def reset_metrics(self):
        self.metrics.reset()
    
    def update_scores(self, mask, output):
        self.metrics.update(mask, output)
    
    def eval(self, sample, model, output, template, step):
        imgs = sample['image']
        coord = sample['coord']
        with torch.no_grad():
            imgs = imgs.cuda()
            preds = model.forward(imgs)
            outputs = preds.cpu().detach().numpy()
            # predictions = np.argmax(outputs, axis=1)
            _, _, h, w = outputs.shape

            for i in range(imgs.shape[0]):
                x = math.floor(coord[i][0] * step[0])
                y = math.floor(coord[i][1] * step[1])
                output[:, x:x+h, y:y+w] += outputs[i]
                template[x:x+h, y:y+w] += np.ones((h, w), dtype='uint8')
        
        return output, template



















