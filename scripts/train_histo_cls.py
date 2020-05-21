import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset.transformer_cls import TransformerCls, TransformerClsVal, TransformerClsTTA
from dataset.dataset_cls import OralDatasetCls, collate
from utils.metrics import AverageMeter
from utils.lr_scheduler import LR_Scheduler
from helper_cls import get_optimizer, Trainer, Evaluator, create_model_load_weights
from option_cls import Options


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# seed
SEED = 233
seed_everything(SEED)

# arguments
args = Options().parse()
n_class = args.n_class

data_path_train = args.data_path_train
meta_path_train = args.meta_path_train
data_path_val = args.data_path_val
meta_path_val = args.meta_path_val
model_path = args.model_path # save model
log_path = args.log_path

if not os.path.exists(model_path): 
    os.makedirs(model_path)
if not os.path.exists(log_path): 
    os.makedirs(log_path)

task_name = args.task_name
print(task_name)

save_dir = os.path.join(model_path, task_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
###################################
evaluation = args.evaluation
test = evaluation and False
print("evaluation:", evaluation, "test:", test)

###################################
print("preparing datasets and dataloaders......")
batch_size = args.batch_size
num_workers = args.num_workers

data_time = AverageMeter("DataTime", ':3.3f')
batch_time = AverageMeter("BatchTime", ':3.3f')

transformer_train = TransformerCls
dataset_train = OralDatasetCls(data_path_train, meta_path_train, label=True, transform=transformer_train)
dataloader_train = torch.utils.data.DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, collate_fn=collate, shuffle=True, pin_memory=True)
transformer_val = TransformerClsVal
dataset_val = OralDatasetCls(data_path_val, meta_path_val, label=True, transform=transformer_val)
dataloader_val = torch.utils.data.DataLoader(dataset_val, num_workers=num_workers, batch_size=batch_size, collate_fn=collate, shuffle=False, pin_memory=True)

###################################
print("creating models......")
model = create_model_load_weights(n_class, evaluation=False, ckpt_path=args.ckpt_path)

###################################
num_epochs = args.epochs
learning_rate = args.lr

optimizer = get_optimizer(model, learning_rate=learning_rate)
scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train))
##################################

# criterion1 = FocalLoss(gamma=3)
criterion = nn.CrossEntropyLoss(reduction='mean')
# criterion = lambda x, y: criterion1(x, y)

if not evaluation:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')
#######################################

trainer = Trainer(criterion, optimizer, n_class)
evaluator = Evaluator(n_class)

best_pred = 0.0
print("start training......")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    tbar = tqdm(dataloader_train)
    train_loss = 0

    start_time = time.time()
    for i_batch, sample in enumerate(tbar):
        # print(i_batch)
        data_time.update(time.time()-start_time)
        if evaluation:  # evaluation pattern: no training
            break
        scheduler(optimizer, i_batch, epoch, best_pred)
        loss = trainer.train(sample, model)
        train_loss += loss.item()
        scores_train = trainer.get_scores()

        batch_time.update(time.time()-start_time)
        start_time = time.time()

        tbar.set_description('Train loss: %.4f; macro F1: %.4f; data time: %.2f; batch time: %.2f' % 
                            (train_loss / (i_batch + 1), scores_train['macro_f1'], data_time.avg, batch_time.avg))

    writer.add_scalar('loss', train_loss/len(tbar), epoch)
    trainer.reset_metrics()
    data_time.reset()
    batch_time.reset()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            if test: 
                tbar = tqdm(dataloader_test)
            else: 
                tbar = tqdm(dataloader_val)

            start_time = time.time()
            for i_batch, sample in enumerate(tbar):
                data_time.update(time.time()-start_time)
                predictions = evaluator.eval(sample, model)
                scores_val = evaluator.get_scores()
                
                batch_time.update(time.time()-start_time)
                tbar.set_description('macro F1: %.4f; data time: %.2f; batch time: %.2f' % 
                                    (scores_val['macro_f1'], data_time.avg, batch_time.avg))

                start_time = time.time()

            data_time.reset()
            batch_time.reset()

            scores_val = evaluator.get_scores()
            evaluator.reset_metrics()

            if scores_val['macro_f1'] > best_pred:
                best_pred = scores_val['macro_f1']
        
                save_path = os.path.join(save_dir, task_name + '-' + str(epoch) + '-' + str(best_pred) + '.pth')
                torch.save(model.state_dict(), save_path)
                
            log = ""
            log = log + 'epoch [{}/{}] macro_f1: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, scores_train['macro_f1'], scores_val['macro_f1']) + "\n"
            log = log + "[train] F1 = " + str(scores_train['f1']) + "\n"
            log = log + "[train] precision = " + str(scores_train['precision'])  + "\n"
            log = log + "[train] recall = " + str(scores_train['recall']) + "\n"
            log = log + "[train] macro_precision = {:.3f}, macro_recall = {:.3f}".format(scores_train['macro_precision'], scores_train['macro_recall']) + "\n"
            log = log + "------------------------------------ \n"
            log = log + "[val] F1 = " + str(scores_val['f1']) + "\n"
            log = log + "[val] precision = " + str(scores_val['precision'])  + "\n"
            log = log + "[val] recall = " + str(scores_val['recall']) + "\n"
            log = log + "[val] macro_precision = {:.3f}, macro_recall = {:.3f}".format(scores_val['macro_precision'], scores_val['macro_recall']) + "\n"
            log = log + "----------------------------------- \n"
            log += "================================\n"
            print(log)
            if evaluation: break  # one peoch

            f_log.write(log)
            f_log.flush()
            writer.add_scalars('macro_f1', {'train f1': scores_train['macro_f1'], 'validation f1': scores_val['macro_f1']}, epoch)

if not evaluation: 
    f_log.close()