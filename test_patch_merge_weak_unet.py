import os
import json

from models.segmentor.weakUNet import WeakUnet
from dataset.transformer import TransformerVal
from dataset.dataset import OralSlide, collate
from helper.helper_weak_unet import SlideInference, create_model_load_weights
from helper.runner_weak import Runner
from configs.config_patch_merge_weak_unet import Config


distributed = False
cfg = Config(mode='patch-merge', train=False)
model = WeakUnet(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.model_cfg)
runner = Runner(cfg, model, create_model_load_weights, distributed=distributed)

###################################
print("preparing datasets......")
slideset_cfg = cfg.testset_cfg
slide_list = sorted(os.listdir(slideset_cfg["img_dir"]))
transformer = TransformerVal()
dataset = OralSlide(
    slide_list, 
    slideset_cfg["img_dir"], 
    slideset_cfg["meta_file"], 
    slide_mask_dir=slideset_cfg["mask_dir"], 
    label=slideset_cfg['label'], 
    transform=transformer,
    )

runner.eval_slide(dataset, SlideInference, cfg.test_output_path)





















