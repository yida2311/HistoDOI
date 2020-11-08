import os
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

from models.segmentor.dynamicUNet import Unet
from dataset.transformer_seg import TransformerSeg, TransformerSegVal
from dataset.dataset_seg import OralDatasetSeg, collate
from utils.metrics import AverageMeter
from utils.lr_scheduler import LR_Scheduler
from utils.seg_loss import FocalLoss, SymmetricCrossEntropyLoss, DecoupledSegLoss_v1, DecoupledSegLoss_v2
from utils.data import class_to_RGB
from helper.helper_unet import Trainer, Evaluator, save_ckpt_model, update_log, update_writer
from helper.utils import get_optimizer, create_model_load_weights
from helper.config_unet import Config


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


def main(cfg, distributed=False):
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
    num_workers = cfg.num_workers
    trainset_cfg = cfg.trainset_cfg
    valset_cfg = cfg.valset_cfg

    data_time = AverageMeter("DataTime", ':3.3f')
    batch_time = AverageMeter("BatchTime", ':3.3f')

    transformer_train = TransformerSeg()
    dataset_train = OralDatasetSeg(
        trainset_cfg["img_dir"],
        trainset_cfg["mask_dir"],
        trainset_cfg["meta_file"], 
        label=trainset_cfg["label"], 
        transform=transformer_train,
    )
    if distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        dataloader_train = DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, collate_fn=collate, sampler=sampler_train, pin_memory=True)
    else:
        dataloader_train = DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, collate_fn=collate, shuffle=True, pin_memory=True)
    transformer_val = TransformerSegVal()
    dataset_val = OralDatasetSeg(
        valset_cfg["img_dir"],
        valset_cfg["mask_dir"],
        valset_cfg["meta_file"],
        label=valset_cfg["label"], 
        transform=transformer_val
    )
    dataloader_val = DataLoader(dataset_val, num_workers=num_workers, batch_size=batch_size, collate_fn=collate, shuffle=False, pin_memory=True)

    ###################################
    print("creating models......")
    model = Unet(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.model_cfg)
    model = create_model_load_weights(model, device, distributed=distributed, local_rank=local_rank, evaluation=True, ckpt_path=cfg.ckpt_path)
 
    ###################################
    num_epochs = cfg.num_epochs
    learning_rate = cfg.lr

    optimizer = get_optimizer(model, learning_rate=learning_rate)
    scheduler = LR_Scheduler(cfg.scheduler, learning_rate, num_epochs, len(dataloader_train))
    ##################################
    if cfg.loss == "ce":
        criterion = nn.CrossEntropyLoss(reduction='mean')
    elif cfg.loss == "sce":
        criterion = SymmetricCrossEntropyLoss(alpha=cfg.loss_cfg['sce']['alpha'], beta=cfg.loss_cfg['sce']['beta'], num_classes=cfg.n_class)
        # criterion4 = NormalizedSymmetricCrossEntropyLoss(alpha=cfg.alpha, beta=cfg.beta, num_classes=cfg.n_class)
    elif cfg.loss == "focal":
        criterion = FocalLoss(gamma=cfg.loss_cfg['focal']['gamma'])
    elif cfg.loss == "ce-dice":
        criterion = nn.CrossEntropyLoss(reduction='mean')
        # criterion2 = 
    elif cfg.loss == 'decouple-v1':
        criterion = DecoupledSegLoss_v1(alpha=cfg.loss_cfg['sce']['alpha'], beta=cfg.loss_cfg['sce']['beta'], num_classes=cfg.n_class)
    elif cfg.loss == 'decouple-v2':
        criterion = DecoupledSegLoss_v2(alpha=cfg.loss_cfg['sce']['alpha'], beta=cfg.loss_cfg['sce']['beta'], gamma=cfg.loss_cfg['focal']['gamma'], num_classes=cfg.n_class)
    
    #######################################
    trainer = Trainer(criterion, optimizer, n_class)
    evaluator = Evaluator(n_class)
    evaluation = cfg.evaluation
    val_vis = cfg.val_vis

    best_pred = 0.0
    print("start training......")

    # log
    if local_rank == 0:
        f_log = open(log_path + ".log", 'w')
        log = task_name + '\n'
        for k, v in cfg.__dict__.items():
            log += str(k) + ' = ' + str(v) + '\n'
        print(log)
        f_log.write(log)
        f_log.flush()
    # writer
    if local_rank == 0:
        writer = SummaryWriter(log_dir=log_path)
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
            if distributed:
                loss = trainer.train(sample, model)
            else:
                loss = trainer.train_acc(sample, model, i_batch, 4, num_batch)
            train_loss += loss.item()
            scores_train = trainer.get_scores()

            batch_time.update(time.time()-start_time)
            start_time = time.time()

            if i_batch % 20 == 0 and local_rank == 0:
                tbar.set_description('Train loss: %.4f; mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                                (train_loss / (i_batch + 1), scores_train["iou_mean"], data_time.avg, batch_time.avg))
        
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
                    predictions = evaluator.eval(sample, model)
                    scores_val = evaluator.get_scores()
                    
                    batch_time.update(time.time()-start_time)
                    if i_batch % 20 == 0 and local_rank==0:
                        tbar.set_description('mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                                        (scores_val["iou_mean"], data_time.avg, batch_time.avg))

                    
                    if val_vis and i_batch == len(tbar)//2: # val set result visualize
                            mask_rgb = class_to_RGB(sample['mask'][1].numpy())
                            mask_rgb = ToTensor()(mask_rgb)
                            predictions_rgb = class_to_RGB(predictions[1])
                            predictions_rgb = ToTensor()(predictions_rgb)
                            writer_info.update(mask=mask_rgb, prediction=predictions_rgb)
                    start_time = time.time()
                    
                data_time.reset()
                batch_time.reset()
                scores_val = evaluator.get_scores()
                evaluator.reset_metrics()

                # save model
                best_pred = save_ckpt_model(model, cfg, scores_val, best_pred, epoch)
                # log 
                update_log(f_log, cfg, scores_train, scores_val, epoch)   
                # writer
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











