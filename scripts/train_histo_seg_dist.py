import os
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor
import torch.distributed as dist

from models.segmentation_models_pytorch.seg_generator import generate_unet
from dataset.transformer_seg import TransformerSeg, TransformerSegVal
from dataset.dataset_seg import OralDatasetSeg, collate
from utils.metrics import AverageMeter
from utils.lr_scheduler import LR_Scheduler
from utils.seg_loss import FocalLoss, SymmetricCrossEntropyLoss, NormalizedSymmetricCrossEntropyLoss
from utils.data import class_to_RGB
from helper_seg import Trainer, Evaluator, get_optimizer, create_model_load_weights
from option_seg import Options


# DPP 1
dist.init_process_group('nccl')
# DPP 2
local_rank = dist.get_rank()
print(local_rank)
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

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

img_path_train = args.img_path_train
mask_path_train = args.mask_path_train
meta_path_train = args.meta_path_train
img_path_val = args.img_path_val
mask_path_val = args.mask_path_val
meta_path_val = args.meta_path_val
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

data_time = AverageMeter("DataTime", ':3.3f')
batch_time = AverageMeter("BatchTime", ':3.3f')

transformer_train = TransformerSeg
dataset_train = OralDatasetSeg(img_path_train, mask_path_train, meta_path_train, label=True, transform=transformer_train)
# dataloader_train = DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, collate_fn=collate, shuffle=True, pin_memory=True)
sampler_train = DistributedSampler(dataset_train, shuffle=True)
dataloader_train = DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, collate_fn=collate, pin_memory=True, sampler=sampler_train)
transformer_val = TransformerSegVal
dataset_val = OralDatasetSeg(img_path_val, mask_path_val, meta_path_val, label=True, transform=transformer_val)
# dataloader_val = DataLoader(dataset_val, num_workers=num_workers, batch_size=batch_size, collate_fn=collate, shuffle=False, pin_memory=True)
# sampler_val = DistributedSampler(dataset_val, shuffle=False)
dataloader_val = DataLoader(dataset_val, num_workers=num_workers*2, batch_size=batch_size, collate_fn=collate, pin_memory=True)

###################################
print("creating models......")
model = generate_unet(num_classes=n_class, encoder_name='resnet34')
model = create_model_load_weights(model, evaluation=False, ckpt_path=args.ckpt_path)
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

###################################
num_epochs = args.epochs
learning_rate = args.lr

optimizer = get_optimizer(model, learning_rate=learning_rate)
scheduler = LR_Scheduler(args.scheduler, learning_rate, num_epochs, len(dataloader_train))
##################################
criterion1 = FocalLoss(gamma=3)
criterion2 = nn.CrossEntropyLoss(reduction='mean')
criterion3 = SymmetricCrossEntropyLoss(alpha=args.alpha, beta=args.beta, num_classes=n_class)
criterion4 = NormalizedSymmetricCrossEntropyLoss(alpha=args.alpha, beta=args.beta, num_classes=n_class)
criterion = lambda x,y: criterion4(x, y)

if not evaluation and local_rank==0:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')
#######################################
trainer = Trainer(criterion, optimizer, n_class)
evaluator = Evaluator(n_class)

best_pred = 0.0
print("start training......")

if local_rank == 0:
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

    start_time = time.time()
    for i_batch, sample in enumerate(tbar):
        data_time.update(time.time()-start_time)

        if evaluation:  # evaluation pattern: no training
            break
        scheduler(optimizer, i_batch, epoch, best_pred)
        loss = trainer.train(sample, model)
        train_loss += loss.item()
        scores_train = trainer.get_scores()

        batch_time.update(time.time()-start_time)
        start_time = time.time()

        if i_batch % 10 == 0 and local_rank==0:
            tbar.set_description('Train loss: %.4f; mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                            (train_loss / (i_batch + 1), scores_train["iou_mean"], data_time.avg, batch_time.avg))
    
    if local_rank == 0:
        writer.add_scalar('loss', train_loss/len(tbar), epoch)
    trainer.reset_metrics()
    data_time.reset()
    batch_time.reset()

    if epoch % 1 == 0 and local_rank==0:
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
                tbar.set_description('mIoU: %.4f; data time: %.2f; batch time: %.2f' % 
                                    (scores_val["iou_mean"], data_time.avg, batch_time.avg))

                if not test: # has label
                    masks = sample['mask'] # PIL images

                if test: # save predictions
                    output_save_path = os.path.join(output_path, task_name)
                    if not os.path.isdir(output_save_path): os.makedirs(output_save_path)
                    for i in range(len(sample['id'])):
                        transforms.functional.to_pil_image(class_to_RGB(predictions[i])).save(os.path.join(output_save_path, sample['id'][i] + "_mask.png"))
                        
                if not evaluation and not test: # train:val
                    if i_batch * batch_size + len(sample['id']) > (epoch % len(dataloader_val)) and i_batch * batch_size <= (epoch % len(dataloader_val)):
                    # writer.add_image('image', transforms.ToTensor()(images[(epoch % len(dataloader_val)) - i_batch * batch_size]), epoch)
                        if not test:
                            mask_rgb = class_to_RGB(masks[epoch % len(dataloader_val) - i_batch * batch_size].numpy())
                            mask_rgb = ToTensor()(mask_rgb)
                            predictions_rgb = class_to_RGB(predictions[(epoch % len(dataloader_val)) - i_batch * batch_size])
                            predictions_rgb = ToTensor()(predictions_rgb)
                            writer.add_image('mask', mask_rgb, epoch)
                            writer.add_image('prediction', predictions_rgb, epoch)
                  
                start_time = time.time()

            data_time.reset()
            batch_time.reset()
            scores_val = evaluator.get_scores()
            evaluator.reset_metrics()

            # save model
            if scores_val['iou_mean'] > best_pred or scores_val['iou_mean'] > 0.81:
                best_pred = scores_val['iou_mean']
                save_path = os.path.join(model_path, "%s-%d-%.5f.pth"%(task_name, epoch, best_pred))
                torch.save(model.state_dict(), save_path)
            
            log = ""
            log = log + 'epoch [{}/{}] mIoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, scores_train['iou_mean'], scores_val['iou_mean']) + "\n"
            log = log + "[train] IoU = " + str(scores_train['iou']) + "\n"
            log = log + "[train] Dice = " + str(scores_train['dice']) + "\n"
            log = log + "[train] Dice_mean = " + str(scores_train['dice_mean']) + "\n"
            log = log + "[train] Accuracy = " + str(scores_train['accuracy'])  + "\n"
            log = log + "[train] Accuracy_mean = " + str(scores_train['accuracy_mean'])  + "\n"
            log = log + "------------------------------------ \n"
            log = log + "[val] IoU = " + str(scores_val['iou']) + "\n"
            log = log + "[val] Dice = " + str(scores_val['dice']) + "\n"
            log = log + "[val] Dice_mean = " + str(scores_val['dice_mean']) + "\n"
            log = log + "[val] Accuracy = " + str(scores_val['accuracy'])  + "\n"
            log = log + "[val] Accuracy_mean = " + str(scores_val['accuracy_mean'])  + "\n"
            log += "================================\n"
            print(log)

            if evaluation: break  # one peoch

            f_log.write(log)
            f_log.flush()
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalars('mIoU', {'train mIoU': scores_train['iou_mean'], 'validation mIoU': scores_val['iou_mean']}, epoch)
            writer.add_scalars('mucosa-iou', {'train mucosa-iou': scores_train['iou'][2], 'validation mucosa-iou': scores_val['iou'][2]}, epoch)
            writer.add_scalars('tumor-iou', {'train tumor-iou': scores_train['iou'][3], 'validation tumor-iou': scores_val['iou'][3]}, epoch)

if not evaluation and local_rank==0: 
    f_log.close()














