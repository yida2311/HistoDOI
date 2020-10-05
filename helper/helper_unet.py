from __future__ import absolute_import, division, print_function
import os 
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .utils import Parallel2Single, ConfusionMatrixSeg, AverageMeter, class_to_RGB, collate


def save_ckpt_model(model, cfg, scores, best_pred, epoch):
    if scores['iou_mean'] > best_pred:
        best_pred = scores['iou_mean']
        save_path = os.path.join(cfg.model_path, "%s-%d-%.5f.pth"%(cfg.model+'-'+cfg.encoder, epoch, best_pred))
        torch.save(model.state_dict(), save_path)
    
    return best_pred


def update_log(f_log, cfg, scores_train, scores_val, epoch):
    log = ""
    log = log + 'epoch [{}/{}] mIoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, cfg.num_epochs, scores_train['iou_mean'], scores_val['iou_mean']) + "\n"
    log = log + "[train] IoU = " + str(scores_train['iou']) + "\n"
    log = log + "[train] Dice = " + str(scores_train['dice']) + "\n"
    log = log + "[train] Dice_mean = " + str(scores_train['dice_mean']) + "\n"
    log = log + "[train] Accuracy = " + str(scores_train['accuracy'])  + "\n"
    log = log + "[train] Accuracy_mean = " + str(scores_train['accuracy_mean'])  + "\n"
    log = log + "------------------------------------ \n"
    log = log + "[val] IoU = " + str(scores_val['iou']) + "\n"
    log = log + "[val] Dice = " + str(scores_val['dice']) + "\n"
    log = log + "[val] Dice_mean = " + str(scores_val['dice_mean']) + "\n"
    log = log + "[val] Accuracy = " + str(scores_val['accuracy'])  + "\n"
    log = log + "[val] Accuracy_mean = " + str(scores_val['accuracy_mean'])  + "\n"
    log += "================================\n"
    print(log)
    f_log.write(log)
    f_log.flush()

def update_writer(writer, writer_info, epoch):
    for k, v in writer_info.items():
        if isinstance(v, dict):
            writer.add_scalars(k, v, epoch)
        elif isinstance(v, torch.Tensor):
            writer.add_image(k, v, epoch)
        else:
            writer.add_scalar(k, v, epoch)


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
        preds = F.interpolate(preds, size=(masks.size(1), masks.size(2)), mode='bilinear')
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
        preds = F.interpolate(preds, size=(masks.size(1), masks.size(2)), mode='bilinear')
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
        preds = F.interpolate(preds, size=(masks.size(1), masks.size(2)), mode='bilinear')
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
            preds = F.interpolate(preds, size=(masks.size(1), masks.size(2)), mode='bilinear')
            outputs = preds.cpu().detach().numpy()
            predictions = np.argmax(outputs, axis=1)

            self.metrics.update(masks_npy, predictions)
        
        return predictions
    
    def test(self, sample, model):
        imgs = sample['image']
        with torch.no_grad():
            imgs = imgs.cuda()
            preds = model.forward(imgs)
            preds = F.interpolate(preds, size=(imgs.size(2), imgs.size(3)), mode='bilinear')
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
                preds = F.interpolate(preds, size=(imgs.size(2), imgs.size(3)), mode='bilinear')
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



















