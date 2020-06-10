###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn 
# Copyright (c) 2018
###########################################################################

import os
import argparse
import torch

# path_g = os.path.join(model_path, "cityscapes_global.800_4.5.2019.lr5e5.pth")
# # path_g = os.path.join(model_path, "fpn_global.804_nonorm_3.17.2019.lr2e5" + ".pth")
# path_g2l = os.path.join(model_path, "fpn_global2local.508_deep.cat.1x_fmreg_ensemble.p3.0.15l2_3.19.2019.lr2e5.pth")
# path_l2g = os.path.join(model_path, "fpn_local2global.508_deep.cat.1x_fmreg_ensemble.p3_3.19.2019.lr2e5.pth")
class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=4, help='segmentation classes')
        parser.add_argument('--img_path_train', type=str, help='path to train dataset where images store')
        parser.add_argument('--mask_path_train', type=str, help='path to train dataset where masks store')
        parser.add_argument('--meta_path_train', type=str, help='path to train meta_file where images name store')
        parser.add_argument('--img_path_val', type=str, help='path to val dataset where images store')
        parser.add_argument('--mask_path_val', type=str, help='path to train dataset where masks store')
        parser.add_argument('--meta_path_val', type=str, help='path to val meta_file where images name store')
        parser.add_argument('--model_path', type=str, help='path to store trained model files, no need to include task specific name')
        parser.add_argument('--output_path', type=str, help='path to store output files, no need to include task specific name')
        parser.add_argument('--log_path', type=str, help='path to store tensorboard log files, no need to include task specific name')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        # parser.add_argument('--mode', type=int, default=1, choices=[1, 2, 3], help='mode for training procedure. 1: train global branch only. 2: train local branch with fixed global branch. 3: train global branch with fixed local branch')
        parser.add_argument('--evaluation', action='store_true', default=False, help='evaluation only')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for image pair')
        # parser.add_argument('--sub_batchsize', type=int, default=2, help=' sub batch size for origin local image (without downsampling)')
        parser.add_argument('--num_workers', type=int, default=4, help='num of workers for dataloader')
        parser.add_argument('--epochs', type=int, default=100, help='num of epochs for training')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        # parser.add_argument('--size_crop', type=int, default=3200, help='size (in pixel) for cropped subslide global image')
        parser.add_argument('--ckpt_path', type=str, default="", help='name for seg model path')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args
