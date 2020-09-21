import os
import sys
import time
import random
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.distributed as dist 

from models.segmentor.GLNet import GLNet
from dataset.transformer_seg import TransformerSeg, TransformerSegVal, TransformerSegGL, TransformerSegGLVal
from dataset.dataset_seg import OralDatasetSeg, collate, collateGL
from utils.metrics import AverageMeter
from utils.lr_scheduler import LR_Scheduler
from utils.seg_loss import FocalLoss, SymmetricCrossEntropyLoss, NormalizedSymmetricCrossEntropyLoss
from utils.data import class_to_RGB
from helper.helper_glnet import Trainer, Evaluator, get_optimizer, create_model_load_weights, save_ckpt_model, update_writer, update_log
from helper.config_glnet import Config


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


def main(cfg, distributed=True):
    if distributed:
        # DPP 1
        dist.init_process_group('nccl')
        # DPP 2
        local_rank = dist.get_rank()
        print(local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device("cuda:0")
        local_rank = 0

    ###################################################
    mode = cfg.mode
    n_class = cfg.n_class
    model_path = cfg.model_path # save model
    log_path = cfg.log_path
    output_path = cfg.output_path

    if local_rank == 0:
        if not os.path.exists(model_path): 
            os.makedirs(model_path)
        if not os.path.exists(log_path): 
            os.makedirs(log_path)
        if not os.path.exists(output_path): 
            os.makedirs(output_path)

    task_name = cfg.task_name
    print(task_name)

    ###################################
    print("preparing datasets and dataloaders......")
    batch_size = cfg.batch_size
    sub_batch_size = cfg.sub_batch_size
    size_g = (cfg.size_g, cfg.size_g)
    size_p = (cfg.size_p, cfg.size_p)
    num_workers = cfg.num_workers
    trainset_cfg = cfg.trainset_cfg
    valset_cfg = cfg.valset_cfg

    data_time = AverageMeter("DataTime", ':3.3f')
    batch_time = AverageMeter("BatchTime", ':3.3f')

    transformer_train = TransformerSegGL(crop_size=cfg.size_g)
    dataset_train = OralDatasetSeg(
        trainset_cfg["img_dir"],
        trainset_cfg["mask_dir"],
        trainset_cfg["meta_file"], 
        label=trainset_cfg["label"], 
        transform=transformer_train,
    )
    if distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        dataloader_train = DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, collate_fn=collateGL, sampler=sampler_train, pin_memory=True)
    else:
        dataloader_train = DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, collate_fn=collateGL, shuffle=True, pin_memory=True)
    transformer_val = TransformerSegGLVal()
    dataset_val = OralDatasetSeg(
        valset_cfg["img_dir"],
        valset_cfg["mask_dir"],
        valset_cfg["meta_file"],
        label=valset_cfg["label"], 
        transform=transformer_val
    )
    dataloader_val = DataLoader(dataset_val, num_workers=num_workers, batch_size=batch_size, collate_fn=collateGL, shuffle=False, pin_memory=True)

    ###################################
    print("creating models......")
    path_g = cfg.path_g
    path_g2l = cfg.path_g2l
    path_l2g = cfg.path_l2g
    model = GLNet(n_class, cfg.encoder, **cfg.model_cfg)
    if mode == 3:
        global_fixed = GLNet(n_class, cfg.encoder, **cfg.model_cfg)
    else:
        global_fixed = None
    model, global_fixed = create_model_load_weights(model, global_fixed, device, mode=mode, local_rank=local_rank, evaluation=False, path_g=path_g, path_g2l=path_g2l, path_l2g=path_l2g)
 
    ###################################
    num_epochs = cfg.num_epochs
    learning_rate = cfg.lr

    optimizer = get_optimizer(model, mode, learning_rate=learning_rate)
    scheduler = LR_Scheduler(cfg.scheduler, learning_rate, num_epochs, len(dataloader_train))
    ##################################
    if cfg.loss == "ce":
        criterion = nn.CrossEntropyLoss(reduction='mean')
    elif cfg.loss == "sce":
        criterion = SymmetricCrossEntropyLoss(alpha=cfg.alpha, beta=cfg.beta, num_classes=cfg.n_class)
        # criterion4 = NormalizedSymmetricCrossEntropyLoss(alpha=cfg.alpha, beta=cfg.beta, num_classes=cfg.n_class)
    elif cfg.loss == "focal":
        criterion = FocalLoss(gamma=3)
    elif cfg.loss == "ce-dice":
        criterion = nn.CrossEntropyLoss(reduction='mean')
        # criterion2 = 
    
    #######################################
    trainer = Trainer(criterion, optimizer, n_class, size_g, size_p, sub_batch_size, mode, cfg.lamb_fmreg)
    evaluator = Evaluator(n_class, size_g, size_p, sub_batch_size, mode)
    evaluation = cfg.evaluation
    val_vis = cfg.val_vis

    best_pred = 0.0
    print("start training......")

    # log
    if local_rank == 0:
        f_log = open(os.path.join(log_path, ".log"), 'w')
        log = task_name + '\n'
        for k, v in cfg.__dict__.items():
            log += str(k) + ' = ' + str(v) + '\n'
        f_log.write(log)
        f_log.flush()
    # writer
    if local_rank == 0:
        writer = SummaryWriter(log_dir=log_path)
    writer_info = {}

    for epoch in range(num_epochs):
        trainer.set_train(model)
        optimizer.zero_grad()
        tbar = tqdm(dataloader_train)
        train_loss = 0

        start_time = time.time()
        for i_batch, sample in enumerate(tbar):
            data_time.update(time.time()-start_time)
            scheduler(optimizer, i_batch, epoch, best_pred)
            # loss = trainer.train(sample, model)
            loss = trainer.train(sample, model, global_fixed)
            train_loss += loss.item()
            score_train, score_train_global, score_train_local = trainer.get_scores()

            batch_time.update(time.time()-start_time)
            start_time = time.time()

            if i_batch % 20 == 0 and local_rank == 0:
                if mode == 1:
                    tbar.set_description('Train loss: %.4f;global mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                                (train_loss / (i_batch + 1), score_train_global["iou_mean"], data_time.avg, batch_time.avg))
                elif mode == 2:
                    tbar.set_description('Train loss: %.4f;agg mIoU: %.4f; local mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                                (train_loss / (i_batch + 1), score_train["iou_mean"], score_train_local["iou_mean"], data_time.avg, batch_time.avg))
                else:
                    tbar.set_description('Train loss: %.4f;agg mIoU: %.4f; global mIoU: %.4f; local mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                                (train_loss / (i_batch + 1), score_train["iou_mean"], score_train_global["iouu_mean"], score_train_local["iou_mean"], data_time.avg, batch_time.avg))

        score_train, score_train_global, score_train_local = trainer.get_scores()        
        trainer.reset_metrics()
        data_time.reset()
        batch_time.reset()

        if evaluation and epoch % 1 == 0 and local_rank == 0:
            with torch.no_grad():
                model.eval()
                print("evaluating...")
                tbar = tqdm(dataloader_val)

                start_time = time.time()
                for i_batch, sample in enumerate(tbar):
                    data_time.update(time.time()-start_time)
                    predictions, predictions_global, predictions_local = evaluator.eval(sample, model, global_fixed)
                    score_val, score_val_global, score_val_local = evaluator.get_scores()
                    
                    batch_time.update(time.time()-start_time)
                    if i_batch % 20 == 0 and local_rank==0:
                        if mode == 1:
                            tbar.set_description('global mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                                        (score_val_global["iou_mean"], data_time.avg, batch_time.avg))
                        elif mode == 2:
                            tbar.set_description('agg mIoU: %.4f; local mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                                        (score_val["iou_mean"], score_val_local["iou_mean"], data_time.avg, batch_time.avg))
                        else:
                            tbar.set_description('agg mIoU: %.4f; global mIoU: %.4f; local mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                                        (score_val["iou_mean"], score_val_global["iou_mean"], score_val_local["iou_mean"], data_time.avg, batch_time.avg))

                    
                    if val_vis and i_batch == len(tbar)//2: # val set result visualize
                            mask_rgb = class_to_RGB(sample['mask'][1].numpy())
                            mask_rgb = ToTensor()(mask_rgb)
                            writer_info.update(mask=mask_rgb, prediction_global=ToTensor()(class_to_RGB(predictions_global[1])))
                            if mode == 2 or mode == 3:
                                writer.update(prediction=ToTensor()(class_to_RGB(predictions[1])), prediction_local=ToTensor()(class_to_RGB(predictions_local[1])))
                                
                    start_time = time.time()
                    
                data_time.reset()
                batch_time.reset()
                score_val, score_val_global, score_val_local = evaluator.get_scores()
                evaluator.reset_metrics()

                # save model
                best_pred = save_ckpt_model(model, cfg, score_val, score_val_global, best_pred, epoch)
                # log 
                update_log(f_log, cfg, [score_train, score_train_global, score_train_local], [score_val, score_val_global, score_val_local], epoch)   
                # writer
                if mode == 1:
                    writer_info.update(
                        loss=trainset_cfg/len(tbar),
                        lr=optimizer.param_groups[0]['lr'],
                        mIOU={
                            "train": score_train_global["iou_mean"],
                            "val": score_val_global["iou_mean"],
                        },
                        global_mIOU={
                            "train": score_train_global["iou_mean"],
                            "val": score_val_global["iou_mean"],
                        },
                        mucosa_iou={
                            "train": score_train_global["iou"][2],
                            "val": score_val_global["iou"][2],
                        },
                        tumor_iou={
                            "train": score_train_global["iou"][3],
                            "val": score_val_global["iou"][3],
                        },
                    )
                else:
                    writer_info.update(
                        loss=trainset_cfg/len(tbar),
                        lr=optimizer.param_groups[0]['lr'],
                        mIOU={
                            "train": score_train["iou_mean"],
                            "val": score_val["iou_mean"],
                        },
                        global_mIOU={
                            "train": score_train_global["iou_mean"],
                            "val": score_val_global["iou_mean"],
                        },
                        local_mIOU={
                            "train": score_train_local["iou_mean"],
                            "val": score_val_local["iou_mean"],
                        },
                        mucosa_iou={
                            "train": score_train["iou"][2],
                            "val": score_val["iou"][2],
                        },
                        tumor_iou={
                            "train": score_train["iou"][3],
                            "val": score_val["iou"][3],
                        },
                    )

                update_writer(writer, writer_info, epoch)
    if local_rank == 0:     
        f_log.close()


args = argParser()
distributed = False
if torch.cuda.device_count() > 1:
    distributed = True

# seed
SEED = 233
seed_everything(SEED)
cfg = Config(train=True)
main(cfg, distributed=distributed)











