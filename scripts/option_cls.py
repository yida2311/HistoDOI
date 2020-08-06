import os
import argparse

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='HistoDOI Classification')
        
        parser.add_argument('--n_class', type=int, default=3, help='classification classes')
        parser.add_argument('--scheduler', type=str, default='poly', help='learning rate scheduler')
        parser.add_argument('--warmup_epochs', type=float, default=0, help='warmup epochs')
        parser.add_argument('--data_path_train', type=str, help='path to train dataset ')
        parser.add_argument('--meta_path_train', type=str, help='path to train meta_file')
        parser.add_argument('--data_path_val', type=str, help='path to val dataset ')
        parser.add_argument('--meta_path_val', type=str, help='path to val meta_file')
        parser.add_argument('--slide_file', type=str, help='path to slide file')
        parser.add_argument('--model_path', type=str, help='path to store trained model files, no need to include task specific name')
        parser.add_argument('--log_path', type=str, help='path to store tensorboard log files, no need to include task specific name')
        parser.add_argument('--output_path', type=str, help='path to store output files, no need to include task specific name')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--evaluation', action='store_true', default=False, help='evaluation only')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for image pair')
        parser.add_argument('--num_workers', type=int, default=4, help='num of workers for dataloader')
        parser.add_argument('--epochs', type=int, default=120, help='num of epochs for traing')
        parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
        parser.add_argument('--ckpt_path', type=str, help='checkpoint path for test')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()

        return args
        