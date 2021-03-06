import os
from .utils import simple_time

class Config:
    def __init__(self, train=True):
        # model config
        self.model = "glnet"
        self.encoder = "resnet18"  
        self.n_class = 4
        self.mode = 1 # 1:g 2:g2l 3:l2g
        self.model_cfg = {
            'decoder_channels': (512, 256, 128, 64),
            'attention_type': 'scse',
        }
        self.crop_size = 4000
        self.size_g = 832
        self.size_p = 832

        root = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile-v2/'
        # data config
        self.trainset_cfg = {
            "img_dir": root + "5x_5000/train/",
            "mask_dir": root + "5x_5000/train_mask/",
            "meta_file": root + "5x_5000/train_5000.csv",
            "label": True,
        }
        self.valset_cfg = {
            "img_dir": root + "5x_4000/val/",
            "mask_dir": root + "5x_4000/val_mask/",
            "meta_file": root + "5x_4000/val_4000.csv",
            "label": True,
        }
        self.slideset_cfg = {  # for slide level inference
            "img_dir": root + "5x_4000/val/",
            "meta_file": root + "5x_4000/tile_info_val_4000.json",
            "mask_dir": root + "5x_4000/val_mask/",
            "label": True,
        }

        # train config
        self.scheduler = 'poly' # ['cos', 'poly', 'step', 'ym']
        self.lr = 1e-4
        self.num_epochs = 150
        self.warmup_epochs = 2
        self.batch_size = 8
        self.sub_batch_size = 8
        ckpt_path = "" # pretrained model
        self.path_g = os.path.join(ckpt_path, "")
        self.path_g2l = os.path.join(ckpt_path, "")
        self.path_l2g = os.path.join(ckpt_path, "")
        self.num_workers = 2
        self.evaluation = True  # evaluatie val set
        self.val_vis = True # val result visualization

        # loss config
        self.loss = "ce" # ["ce", "sce", 'ce-dice]
        self.loss_cfg = {
            "sce": {
                "alpha": 1.0,
                "beta": 1.0,
            },
            "ce-dice": {
                "alpha": 0.1,
            },
            "focal": {
                "gamma": 2,
            }

        }
        self.lamb_fmreg = 0.5

        # task name
        self.mode_str = ["g", "g2l", "l2g", "l"]
        self.task_name = "-".join([self.model, self.encoder, self.mode_str[self.mode], self.loss, self.scheduler, str(self.lr), str(self.num_epochs), simple_time()])
        if train:
            self.task_name += "-" + "train"
        else:
            self.task_name += "-" + "test"
        
        # output config
        out_root = "results-v2/"
        self.model_path = out_root + "saved_models/" + self.task_name
        self.log_path = out_root + "logs/" + self.task_name
        self.output_path = out_root + "predictions/" + self.task_name

        
