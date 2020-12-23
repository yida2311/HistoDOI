import os
import json

from models.segmentor.dynamicUNet import Unet
from dataset.transformer import TransformerVal
from dataset.dataset import OralSlide, collate
from helper.helper_unet import SlideInference, create_model_load_weights
from helper.runner import Runner
from configs.config_patch_unet import Config


distributed = False
cfg = Config(mode='patch', train=False)
model = Unet(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.model_cfg)
runner = Runner(cfg, model, create_model_load_weights, distributed=distributed)

###################################
print("preparing datasets......")
with open('/media/ldy/7E1CA94545711AE6/OSCC/train_val_part.json', 'r') as f:
    slide_list = json.load(f)['val']
slideset_cfg = cfg.slideset_cfg
transformer = TransformerVal()
dataset = OralSlide(
    slide_list, 
    slideset_cfg["img_dir"], 
    slideset_cfg["meta_file"], 
    slide_mask_dir=slideset_cfg["mask_dir"], 
    label=slideset_cfg['label'], 
    transform=transformer,
    )

runner.eval_slide(dataset, SlideInference)





















