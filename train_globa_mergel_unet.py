import os
import torch
import torch.nn as nn

from models.segmentor.dynamicUNet import Unet
from dataset.transformer import TransformerMerge, TransformerMergeVal   # **
from dataset.dataset import OralDataset, collate
from utils.seg_loss import FocalLoss, SymmetricCrossEntropyLoss, DecoupledSegLoss_v1, DecoupledSegLoss_v2
from helper.helper_unet import Trainer, Evaluator, save_ckpt_model, update_log, update_writer, get_optimizer, create_model_load_weights
from configs.config_global_merge_unet import Config   # ** 
from helper.runner import argParser, seed_everything, Runner


args = argParser()
distributed = False
# if torch.cuda.device_count() > 1:
#     distributed = True

# seed
SEED = 233
seed_everything(SEED)
cfg = Config(mode='global-merge', train=True)  # **
model = Unet(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.model_cfg)
runner = Runner(cfg, model, create_model_load_weights, distributed=distributed)

###################################
print("preparing datasets......")
batch_size = cfg.batch_size
num_workers = cfg.num_workers
trainset_cfg = cfg.trainset_cfg
valset_cfg = cfg.valset_cfg

transformer_train = TransformerMerge()   # **
dataset_train = OralDataset(
    trainset_cfg["img_dir"],
    trainset_cfg["mask_dir"],
    trainset_cfg["meta_file"], 
    label=trainset_cfg["label"], 
    transform=transformer_train,
)
transformer_val = TransformerMergeVal()  #**
dataset_val = OralDataset(
    valset_cfg["img_dir"],
    valset_cfg["mask_dir"],
    valset_cfg["meta_file"],
    label=valset_cfg["label"], 
    transform=transformer_val
)

if cfg.loss == "ce":
    criterion = nn.CrossEntropyLoss(reduction='mean')
elif cfg.loss == "sce":
    criterion = SymmetricCrossEntropyLoss(alpha=cfg.loss_cfg['sce']['alpha'], beta=cfg.loss_cfg['sce']['beta'], num_classes=cfg.n_class)
    # criterion4 = NormalizedSymmetricCrossEntropyLoss(alpha=cfg.alpha, beta=cfg.beta, num_classes=cfg.n_class)
elif cfg.loss == "focal":
    criterion = FocalLoss(gamma=cfg.loss_cfg['focal']['gamma'])
elif cfg.loss == "ce-dice":
    criterion = nn.CrossEntropyLoss(reduction='mean')
elif cfg.loss == 'decouple-v1':
    criterion = DecoupledSegLoss_v1(alpha=cfg.loss_cfg['sce']['alpha'], beta=cfg.loss_cfg['sce']['beta'], num_classes=cfg.n_class)
elif cfg.loss == 'decouple-v2':
    criterion = DecoupledSegLoss_v2(alpha=cfg.loss_cfg['sce']['alpha'], beta=cfg.loss_cfg['sce']['beta'], gamma=cfg.loss_cfg['focal']['gamma'], num_classes=cfg.n_class)
    
runner.train(dataset_train, dataset_val, criterion, get_optimizer, Trainer, Evaluator, collate)











