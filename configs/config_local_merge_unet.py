from helper.utils import simple_time

class Config:
    def __init__(self, mode, train=True):
        # model config
        self.model = "unet"
        self.encoder = "resnet34"  
        self.n_class = 3
        self.model_cfg = {
            'encoder_depth': 5,
            'encoder_weights': 'imagenet',
            'decoder_use_batchnorm': True,
            'decoder_channels': (512, 256, 128, 64),
            'decoder_attention_type': 'scse',
            'in_channels': 3,
        }
        self.mode = mode

        # data config
        root = '/disk2/ldy/5x_gl_3020/'
        self.trainset_cfg = {
            "img_dir": root + "patch/",
            "mask_dir": root + "std_mask/",
            "meta_file": root + "train_local.csv",
            "label": True,
        }
        self.valset_cfg = {
            "img_dir": root +  "patch/",
            "mask_dir": root + "std_mask/",
            "meta_file": root + "val_local.csv",
            "label": True,
        }
        test_root = '/disk2/ldy/fine/5x_tile/5x_800/'
        self.testset_cfg = {
            "img_dir": root +  "patch/",
            "mask_dir": root + "std_mask/",
            "meta_file": root + "tile_info.json",
            "label": True,
        }
        self.slideset_cfg = {  # for slide level inference
            "img_dir": root + "patch/",
            "meta_file": root + "tile_info.json",
            "mask_dir": '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile-v3/5x_mask/std_mask/',
            "label": True,
        }

        # train config
        self.scheduler = 'poly' # ['cos', 'poly', 'step', 'ym']
        self.lr = 1e-4
        self.num_epochs = 120
        self.warmup_epochs = 2
        self.batch_size = 4
        self.ckpt_path = None #"/home/ldy/HistoDOI/results-v3/saved_models/unet-resnet34-sce-poly-0.0001-150-[10-13-01]-train/unet-resnet34-129-0.88217.pth" # pretrained model
        self.num_workers = 4
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

        # task name
        self.task_name = "-".join([self.model, self.mode, self.encoder, self.loss, str(self.lr), str(self.num_epochs), simple_time()])
        if train:
            self.task_name += "-" + "train"
        else:
            self.task_name += "-" + "test"
        # output config
        out_root = "results-v3/"
        self.model_path = out_root + "saved_models/" + self.task_name
        self.log_path = out_root + "logs/" 
        self.writer_path = out_root + 'writers/' + self.task_name
        self.val_output_path = out_root + "predictions/" + self.task_name + '/val'
        self.test_output_path = out_root + "predictions/" + self.task_name + '/test'
        
        
