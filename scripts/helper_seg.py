from __future__ import absolute_import, division, print_function

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.utils import Parallel2Single
from utils.metrics import ConfusionMatrixSeg, AverageMeter
from dataset.dataset_seg import collate
from utils.data import class_to_RGB


def create_model_load_weights(model, evaluation=False, ckpt_path=None):
    if evaluation and ckpt_path:
        state_dict = torch.load(ckpt_path)
        if 'module' in next(iter(state_dict)):
            state_dict = Parallel2Single(state_dict)
        state = model.state_dict()
        state.update(state_dict)
        model.load_state_dict(state)
    return model


def create_model_load_weights_v2(model, evaluation=False, ckpt_path=None):
    if evaluation and ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
        if 'module' in next(iter(state_dict)):
            state_dict = Parallel2Single(state_dict)
        state = model.state_dict()
        state.update(state_dict)
        model.load_state_dict(state)
    return model


def get_optimizer(model, learning_rate=2e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
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
    
    def train_acc(self, sample, model, i, acc_step, maxLen):
        imgs = sample['image']
        masks = sample['mask'].squeeze(1)

        imgs = imgs.cuda()
        masks_npy = np.array(masks)
        masks = masks.cuda()

        preds = model.forward(imgs)
        loss = self.criterion(preds, masks) / acc_step
        loss.backward()

        if i%acc_step == 0 or i==maxLen-1:
            self.optimizer.step()
            self.optimizer.zero_grad()

        outputs = preds.cpu().detach().numpy()
        predictions = np.argmax(outputs, axis=1)
        self.metrics.update(masks_npy, predictions)
        
        return loss
    
    def train_schp(self, sample, model, schp_model, cycle_n):
        model.train()
        imgs = sample['image']
        masks = sample['mask'].squeeze(1)

        imgs = imgs.cuda()
        masks_npy = np.array(masks)
        masks = masks.cuda()

        preds = model.forward(imgs)
        # Online self correction cycle with label refinement
        if cycle_n >= 1:
            with torch.no_grad():
                soft_preds = schp_model.forward(imgs)
        else:
            soft_preds = None 

        loss = self.criterion(preds, soft_preds, masks, cycle_n)
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


class SlideInference(object):
    def __init__(self, n_class, num_workers, batch_size):
        # self.metrics = ConfusionMatrixSeg(n_class)
        self.n_class = n_class
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.metrics = ConfusionMatrixSeg(n_class)
    
    def update_scores(self, gt, pred):
        self.metrics.update(gt, pred)

    def get_scores(self):
        scores = self.metrics.get_scores()
        return scores
    
    def reset_metrics(self):
        self.metrics.reset()

    def inference(self, dataset, model, scale=4):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate, shuffle=False, pin_memory=True)
        output = np.zeros((self.n_class, dataset.slide_size[0], dataset.slide_size[1])) # n_class x H x W
        template = np.zeros(dataset.slide_size, dtype='uint8') # H x W
        step = dataset.slide_step

        for sample in dataloader:
            imgs = sample['image']
            coord = sample['coord']
            with torch.no_grad():
                imgs = imgs.cuda()
                preds = model.forward(imgs)
                preds_np = preds.cpu().detach().numpy()
            _, _, h, w = preds_np.shape

            for i in range(imgs.shape[0]):
                x = math.floor(coord[i][0] * step[0])
                y = math.floor(coord[i][1] * step[1])
                output[:, x:x+h, y:y+w] += preds_np[i]
                template[x:x+h, y:y+w] += np.ones((h, w), dtype='uint8')
    
        template[template==0] = 1
        output = output / template
        prediction = np.argmax(output, axis=0)

        output = torch.from_numpy(output).unsqueeze(0)
        output = F.interpolate(output, size=(dataset.slide_size[0]//4, dataset.slide_size[1]//4), mode='bilinear')
        
        return prediction, class_to_RGB(prediction), output.squeeze(0).numpy()

def struct_time():
    # 格式化成2020-08-07 16:56:32
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return cur_time

















