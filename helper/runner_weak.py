import os
import time
import random
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist 

from utils.metrics import AverageMeter
from utils.lr_scheduler import LR_Scheduler
from utils.data import class_to_RGB
from .helper_weak_unet import save_ckpt_model, update_log, update_writer


def argParser():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    return args


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True


class Runner:
    def __init__(
        self,
        cfg,
        model,
        model_loader_func,
        distributed=False,
    ):
        if distributed:
            # DPP 1
            dist.init_process_group('nccl')
            # DPP 2
            self.local_rank = dist.get_rank()
            print(local_rank)
            torch.cuda.set_device(local_rank)
            self.device = torch.device('cuda', local_rank)
        else:
            self.device = torch.device("cuda:0")
            self.local_rank = 0
    
        if self.local_rank == 0:
            if not os.path.exists(cfg.model_path): 
                os.makedirs(cfg.model_path)
            if not os.path.exists(cfg.log_path): 
                os.makedirs(cfg.log_path)
            if not os.path.exists(cfg.writer_path):
                os.makedirs(cfg.writer_path)
            if not os.path.exists(cfg.output_path): 
                os.makedirs(cfg.output_path)

        print(cfg.task_name)
        self.cfg = cfg
        self.model = model 
        self.model_loader = model_loader_func
        self.distributed = distributed


    def train(self, 
            dataset_train, 
            dataset_val, 
            criterion, 
            optimizer_func, 
            trainer_func, 
            evaluator_func, 
            collate):
        if self.distributed:
            sampler_train = DistributedSampler(dataset_train, shuffle=True)
            dataloader_train = DataLoader(dataset_train, num_workers=self.cfg.num_workers, batch_size=self.cfg.batch_size, collate_fn=collate, sampler=sampler_train, pin_memory=True)
        else:
            dataloader_train = DataLoader(dataset_train, num_workers=self.cfg.num_workers, batch_size=self.cfg.batch_size, collate_fn=collate, shuffle=True, pin_memory=True)
        dataloader_val = DataLoader(dataset_val, num_workers=self.cfg.num_workers, batch_size=self.cfg.batch_size, collate_fn=collate, shuffle=False, pin_memory=True)

        ###################################
        print("creating models......")
        model = self.model_loader(self.model, self.device, distributed=self.distributed, local_rank=self.local_rank, evaluation=True, ckpt_path=self.cfg.ckpt_path)

        ###################################
        num_epochs = self.cfg.num_epochs
        learning_rate = self.cfg.lr
        data_time = AverageMeter("DataTime", ':3.3f')
        batch_time = AverageMeter("BatchTime", ':3.3f')

        optimizer = optimizer_func(model, learning_rate=learning_rate)
        scheduler = LR_Scheduler(self.cfg.scheduler, learning_rate, num_epochs, len(dataloader_train))
        ##################################
        trainer = trainer_func(criterion, optimizer, self.cfg.n_class)
        evaluator = evaluator_func(self.cfg.n_class)
        evaluation = self.cfg.evaluation
        val_vis = self.cfg.val_vis
        best_pred = 0.0
        print("start training......")

        # log
        if self.local_rank == 0:
            f_log = open(self.cfg.log_path + self.cfg.task_name + ".log", 'w')
            log = self.cfg.task_name + '\n'
            for k, v in self.cfg.__dict__.items():
                log += str(k) + ' = ' + str(v) + '\n'
            print(log)
            f_log.write(log)
            f_log.flush()
        # writer
        if self.local_rank == 0:
            writer = SummaryWriter(log_dir=self.cfg.writer_path)
        writer_info = {}

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            num_batch = len(dataloader_train)
            tbar = tqdm(dataloader_train)
            train_loss = 0
        
            start_time = time.time()
            model.train()
            for i_batch, sample in enumerate(tbar):
                data_time.update(time.time()-start_time)
                scheduler(optimizer, i_batch, epoch, best_pred)
                # loss = trainer.train(sample, model)
                if self.distributed:
                    loss = trainer.train(sample, model)
                else:
                    loss = trainer.train_acc(sample, model, i_batch, 2, num_batch)
                
                train_loss += loss.item()
                scores_train = trainer.get_scores()

                batch_time.update(time.time()-start_time)
                start_time = time.time()
            
                if i_batch % 20 == 0 and self.local_rank == 0:
                    tbar.set_description('Train loss: %.4f; mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                                (train_loss / (i_batch + 1), scores_train["iou_mean"], data_time.avg, batch_time.avg))
                
            trainer.reset_metrics()
            data_time.reset()
            batch_time.reset()

            train_fr_values = list(trainer.fr_dict.values())
            train_fr = np.sum(np.array(train_fr_values), axis=0) / len(train_fr_values)

            if evaluation and epoch % 1 == 0 and self.local_rank == 0:
                with torch.no_grad():
                    model.eval()
                    print("evaluating...")
                    tbar = tqdm(dataloader_val)

                    start_time = time.time()
                    for i_batch, sample in enumerate(tbar):
                        data_time.update(time.time()-start_time)
                        predictions = evaluator.eval(sample, model)
                        scores_val = evaluator.get_scores()
                    
                        batch_time.update(time.time()-start_time)
                        if i_batch % 20 == 0 and self.local_rank==0:
                            tbar.set_description('mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                                        (scores_val["iou_mean"], data_time.avg, batch_time.avg))

                        if val_vis and (1+epoch) % 10 == 0: # val set result visualize
                            for i in range(len(sample['id'])):
                                name = sample['id'][i] + '.png'
                                slide = name.split('_')[0] 
                                slide_dir = os.path.join(self.cfg.output_path, slide)
                                if not os.path.exists(slide_dir):
                                    os.makedirs(slide_dir)
                                predictions_rgb = class_to_RGB(predictions[i])
                                predictions_rgb = cv2.cvtColor(predictions_rgb, cv2.COLOR_BGR2RGB)
                                cv2.imwrite(os.path.join(slide_dir, name), predictions_rgb)
                                # writer_info.update(mask=mask_rgb, prediction=predictions_rgb)
                        start_time = time.time()
            
                    
                    data_time.reset()
                    batch_time.reset()
                    scores_val = evaluator.get_scores()
                    evaluator.reset_metrics()

                    val_fr_values = list(evaluator.fr_dict.values())
                    val_fr = np.sum(np.array(val_fr_values), axis=0) / len(val_fr_values)

                    # save model
                    best_pred = save_ckpt_model(model, self.cfg, scores_val, best_pred, epoch)
                    # log 
                    update_log(f_log, self.cfg, scores_train, scores_val, epoch)   
                    # writer\
                    if self.cfg.n_class == 4:
                        writer_info.update(
                            loss=train_loss/len(tbar),
                            lr=optimizer.param_groups[0]['lr'],
                            mIOU={
                                "train": scores_train["iou_mean"],
                                "val": scores_val["iou_mean"],
                            },
                            mucosa_iou={
                                "train": scores_train["iou"][2],
                                "val": scores_val["iou"][2],
                            },
                            tumor_iou={
                                "train": scores_train["iou"][3],
                                "val": scores_val["iou"][3],
                            },
                            mucosa_fr={
                                "train": train_fr[0],
                                "val": val_fr[0],
                            },
                            tumor_fr={
                                "train": train_fr[1],
                                "val": val_fr[1],
                            }
                        )
                    else:
                        writer_info.update(
                            loss=train_loss/len(tbar),
                            lr=optimizer.param_groups[0]['lr'],
                            mIOU={
                                "train": scores_train["iou_mean"],
                                "val": scores_val["iou_mean"],
                            },
                            merge_iou={
                                "train": scores_train["iou"][2],
                                "val": scores_val["iou"][2],
                            },
                            merge_fr={
                                "train": train_fr[0],
                                "val": val_fr[0],
                            }
                        )
                    update_writer(writer, writer_info, epoch)
        if self.local_rank == 0:     
            f_log.close()
        
    
    def eval_slide(self, dataset, evaluator_func):
        print("preparing datasets and dataloaders......")
        evaluation = self.cfg.slideset_cfg["label"]
        slide_time = AverageMeter("DataTime", ':3.3f')
        
        model = self.model_loader(self.model, device=self.device, evaluation=True, ckpt_path=self.cfg.ckpt_path)
        model = model.cuda()
        f_log = open(self.cfg.log_path + self.cfg.task_name +  "_slide_test.log", 'w')
        #######################################
        evaluator = evaluator_func(self.cfg.n_class, self.cfg.num_workers, self.cfg.batch_size)
        num_slides = len(dataset.slides)
        tbar = tqdm(range(num_slides))

        for i in tbar:
            start_time = time.time()
            dataset.get_patches_from_index(i)
            prediction, output, _ = evaluator.inference(dataset, model)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(self.cfg.output_path, dataset.slide+'.png'), output)
            slide_time.update(time.time()-start_time)
        
            if evaluation:
                mask = dataset.get_slide_mask_from_index(i)
                evaluator.update_scores(mask, prediction)
                scores = evaluator.get_scores()
                tbar.set_description('Slide: {}'.format(dataset.slide) + ', mIOU: %.5f; slide time: %.2f' % (scores['iou_mean'], slide_time.avg))
            else:
                tbar.set_description('Slide: {}'.format(dataset.slide) + ', slide time: %.2f' % (slide_time.avg))
    
        if evaluation:
            scores = evaluator.get_scores()
            print(evaluator.metrics.confusion_matrix)

        log = ""
        log = log + str(task_name) + '   slide inference \n'
        if evaluation:
            log = log + "mIOU = " + str(scores['iou_mean']) + '\n'
            log = log + "IOU: " + str(scores['iou']) + '\n'
            log = log + "Dice: " + str(scores['dice']) + '\n'
    
        log = log + "[Time consuming %.2fs][%.2fs per slide]" % (slide_time.sum, slide_time.avg) + '\n'
        log += "================================\n"
        print(log)

        f_log.write(log)
        f_log.close()




















