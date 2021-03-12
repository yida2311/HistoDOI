import os
import torch
import torch.nn as nn

from helper.runner_weak import argParser, seed_everything, Runner
from dataset.dataset_local import OralDatasetLocal, collate
from dataset.dataset import OralSlide
from utils.seg_loss import *
from helper.helper_weak_unet import Trainer, Evaluator, SlideInference, save_ckpt_model, update_log, update_writer, get_optimizer, create_model_load_weights
from dataset.transformer import Transformer, TransformerVal
from configs.config_local_merge_weak_unet import Config
from models.segmentor.weakUNet import WeakUnet


args = argParser()
distributed = False
# if torch.cuda.device_count() > 1:
#     distributed = True

# seed
SEED = 233
seed_everything(SEED)
cfg = Config(mode='local', train=True)
model = WeakUnet(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.model_cfg)
runner = Runner(cfg, model, create_model_load_weights, distributed=distributed)
# print(model)
###################################
print("preparing datasets......")
batch_size = cfg.batch_size
num_workers = cfg.num_workers
trainset_cfg = cfg.trainset_cfg
valset_cfg = cfg.valset_cfg
testset_cfg = cfg.testset_cfg

transformer_train = Transformer()
dataset_train = OralDatasetLocal(
    trainset_cfg["img_dir"],
    trainset_cfg["mask_dir"],
    trainset_cfg["meta_file"], 
    label=trainset_cfg["label"], 
    transform=transformer_train,
)
transformer_val = TransformerVal()
dataset_val = OralDatasetLocal(
    valset_cfg["img_dir"],
    valset_cfg["mask_dir"],
    valset_cfg["meta_file"],
    label=valset_cfg["label"], 
    transform=transformer_val
)
transformer_test = TransformerMergeVal()
slide_list = sorted(os.listdir(testset_cfg["img_dir"]))
dataset_test = OralSlide(
    slide_list,
    testset_cfg["img_dir"],
    testset_cfg["meta_file"],
    slide_mask_dir=testset_cfg["mask_dir"],
    label=testset_cfg["label"], 
    transform=transformer_test,
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
elif cfg.loss == 'fr':
    criterion = FRSegLoss(cfg.n_class, alpha=cfg.loss_cfg['fr']['alpha'], beta=cfg.loss_cfg['fr']['beta'], momentum=cfg.loss_cfg['fr']['momentum'], reduction='mean')

runner.train(dataset_train, 
            dataset_val, 
            criterion, 
            get_optimizer, 
            Trainer, 
            Evaluator, 
            collate,
            dataset_test=dataset_test,
            tester_func=SlideInference)










