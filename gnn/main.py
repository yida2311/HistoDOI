import os
import sys
sys.path.append('../')
import random
import time
import argparse
import torch
import torch_geometric

from tqdm import tqdm
import numpy as np 
from torch import nn
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import AverageMeter
from utils.lr_scheduler import LR_Scheduler
from scripts.helper_seg import create_model_load_weights, get_optimizer
from data.graph_dataset import DoiDataset
from network.gcn import DoiNet
from helper import Trainer, Evaluator, seed_everything, save_model, write_log, write_summaryWriter


class Args():
    def __init__(self):
        parser = argparse.ArgumentParser(description='OSCC EdgeGAT for DOI measurement')
        parser.add_argument('--n_class', type=int, default=3, help='segmentation classes')
        parser.add_argument('--num_normal', type=int, default=10, help='maximum number of nodes for normal')
        parser.add_argument('--num_mucosa', type=int, default=10, help='maximum number of nodes for normal')
        parser.add_argument('--num_tumor', type=int, default=30, help='maximum number of nodes for tumor')
        parser.add_argument('--min_node_area', type=int, default=30, help='minimum area w.r.t connnected component to extract node')
        parser.add_argument('--num_edges_per_class', type=int, default=4, help='number of edges for each small(not in topk) node in graph')
        parser.add_argument('--node_resize', type=int, default=16, help='size of node after resize from connected components')
        parser.add_argument('--scheduler', type=str, default='poly', help='learning rate scheduler')
        parser.add_argument('--img_path_train', type=str, help='path to train dataset where images store')
        parser.add_argument('--mask_path_train', type=str, help='path to train dataset where masks store')
        parser.add_argument('--img_path_val', type=str, help='path to val dataset where images store')
        parser.add_argument('--mask_path_val', type=str, help='path to train dataset where masks store')
        parser.add_argument('--model_path', type=str, help='path to store trained model files, no need to include task specific name')
        parser.add_argument('--output_path', type=str, help='path to store output files, no need to include task specific name')
        parser.add_argument('--log_path', type=str, help='path to store tensorboard log files, no need to include task specific name')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--evaluation', action='store_true', default=False, help='evaluation only')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for image pair')
        parser.add_argument('--num_workers', type=int, default=4, help='num of workers for dataloader')
        parser.add_argument('--epochs', type=int, default=100, help='num of epochs for training')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--ckpt_path', type=str, default="", help='name for seg model path')
        parser.add_argument('--alpha', type=float, default=1.0, help='weight for edge criterion w.r.t node criterion')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        config = {}
        config['max_num_nodes'] = [args.num_normal, args.num_mucosa, args.num_tumor]
        config['min_node_area'] = args.min_node_area
        config['num_edges_per_class'] = args.num_edges_per_class
        config['node_resize'] = args.node_resize
        args.config = config

        return args


def main(seed=25):
    seed_everything(25)
    device = torch.device('cuda:0')

    # arguments
    args = Args().parse()
    n_class = args.n_class

    img_path_train = args.img_path_train
    mask_path_train = args.mask_path_train
    img_path_val = args.img_path_val
    mask_path_val = args.mask_path_val

    model_path = os.path.join(args.model_path, args.task_name) # save model
    log_path = args.log_path
    output_path = args.output_path

    if not os.path.exists(model_path): 
        os.makedirs(model_path)
    if not os.path.exists(log_path): 
        os.makedirs(log_path)
    if not os.path.exists(output_path): 
        os.makedirs(output_path)

    task_name = args.task_name
    print(task_name)
    ###################################
    evaluation = args.evaluation
    test = evaluation and False
    print("evaluation:", evaluation, "test:", test)

    ###################################
    print("preparing datasets and dataloaders......")
    batch_size = args.batch_size
    num_workers = args.num_workers
    config = args.config

    data_time = AverageMeter("DataTime", ':3.3f')
    batch_time = AverageMeter("BatchTime", ':3.3f')

    dataset_train = DoiDataset(img_path_train, config, train=True, root_mask=mask_path_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataset_val = DoiDataset(img_path_val, config, train=True, root_mask=mask_path_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    ###################################
    print("creating models......")
    model = DoiNet(n_class)
    model = create_model_load_weights(model, evaluation=False, ckpt_path=args.ckpt_path)
    model.to(device)

    ###################################
    num_epochs = args.epochs
    learning_rate = args.lr

    optimizer = get_optimizer(model, learning_rate=learning_rate)
    scheduler = LR_Scheduler(args.scheduler, learning_rate, num_epochs, len(dataloader_train))
    ##################################
    criterion_node = nn.CrossEntropyLoss(reduction='mean')
    criterion_edge = nn.BCELoss(reduction='mean')
    alpha = args.alpha

    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')
    #######################################
    trainer = Trainer(criterion_node, criterion_edge, optimizer, n_class, device, alpha=alpha)
    evaluator = Evaluator(n_class, device)

    best_pred = 0.0
    print("start training......")
    log = task_name + '\n'
    for k, v in args.__dict__.items():
        log += str(k) + ' = ' + str(v) + '\n'
    print(log)
    f_log.write(log)
    f_log.flush()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        tbar = tqdm(dataloader_train)
        train_loss = 0
        train_loss_edge = 0
        train_loss_node = 0

        start_time = time.time()
        for i_batch, sample in enumerate(tbar):
            data_time.update(time.time()-start_time)

            if evaluation:  # evaluation pattern: no training
                break
            scheduler(optimizer, i_batch, epoch, best_pred)
            loss, loss_node, loss_edge = trainer.train(sample, model)
            train_loss += loss.item()
            train_loss_node += loss_node.item()
            train_loss_edge += loss_edge.item()
            train_scores_node, train_scores_edge = trainer.get_scores()

            batch_time.update(time.time()-start_time)
            start_time = time.time()

            if i_batch % 2 == 0:
                tbar.set_description('Train loss: %.4f (loss_node=%.4f  loss_edge=%.4f); F1 node: %.4f  F1 edge: %.4f; data time: %.2f; batch time: %.2f' % 
                                    (train_loss / (i_batch + 1), train_loss_node / (i_batch + 1), train_loss_edge / (i_batch + 1), 
                                    train_scores_node["macro_f1"], train_scores_edge["macro_f1"], data_time.avg, batch_time.avg))
    
        trainer.reset_metrics()
        data_time.reset()
        batch_time.reset()

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                print("evaluating...")

                tbar = tqdm(dataloader_val)
                start_time = time.time()
                for i_batch, sample in enumerate(tbar):
                    data_time.update(time.time()-start_time)
                    pred_node, pred_edge = evaluator.eval(sample, model)
                    val_scores_node, val_scores_edge = evaluator.get_scores()
                
                    batch_time.update(time.time()-start_time)
                    tbar.set_description('F1 node: %.4f  F1 edge: %.4f; data time: %.2f; batch time: %.2f' % 
                                        (val_scores_node["macro_f1"], val_scores_edge["macro_f1"], data_time.avg, batch_time.avg))
                    start_time = time.time()
            
            data_time.reset()
            batch_time.reset()
            val_scores_node, val_scores_node = evaluator.get_scores()
            evaluator.reset_metrics()

            best_pred = save_model(model, model_path, val_scores_node, val_scores_edge, alpha, task_name, epoch, best_pred)
            write_log(f_log, train_scores_node, train_scores_edge, val_scores_node, val_scores_edge, epoch, num_epochs)
            write_summaryWriter(writer, train_loss/len(dataloader_train), optimizer, train_scores_node, train_scores_edge, val_scores_node, val_scores_edge, epoch)
            
    f_log.close()


if __name__ == '__main__':
    main(seed=22)






