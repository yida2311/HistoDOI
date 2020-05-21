import os
import time
import numpy as np 
import torch 
import cv2
from tqdm import tqdm

from dataset.transformer_cls import TransformerClsVal
from dataset.dataset_cls import OralSlideCls, collate
from utils.metrics import AverageMeter
from helper_cls import create_model_load_weights, SlideEvaluator
from option_cls import Options
from utils.data import class_to_RGB


# arguments
args = Options().parse()
n_class = args.n_class

task_name = args.task_name
print(task_name)

data_path = args.data_path_val
meta_path = args.meta_path_val
log_path = args.log_path
output_path = os.path.join(args.output_path, task_name)
ckpt_path = args.ckpt_path

if not os.path.exists(log_path): 
    os.makedirs(log_path)
if not os.path.exists(output_path): 
    os.makedirs(output_path)

###################################
evaluation = args.evaluation
test = evaluation and True
print("evaluation:", evaluation, "test:", test)

###################################
print("preparing datasets and dataloaders......")
batch_size = args.batch_size
num_workers = args.num_workers

slide_time = AverageMeter("DataTime", ':3.3f')

transformer = TransformerClsVal
dataset = OralSlideCls(data_path, meta_path, label=True, transform=transformer)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate, shuffle=False, pin_memory=True)

###################################
print("creating models......")
model = create_model_load_weights(n_class, evaluation=True, ckpt_path=args.ckpt_path)

if not evaluation:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + "_test.log", 'w')
#######################################
num_slides = len(dataset.slides)
tbar = tqdm(range(num_slides))
for i in tbar:
    dataset.get_patches_from_index(i)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate, shuffle=False, pin_memory=True)

    output = np.zeros_like(dataset.slide_mask, dtype='uint8')
    evaluator = SlideEvaluator(n_class, output)

    start_time = time.time()
    for sample in dataloader:
        print(dataset.slide_mask)
        evaluator.eval(sample, model)
    
    slide_time.update(time.time()-start_time)
    start_time = time.time()

    slide = dataset.slide
    output = evaluator.output
    mask = dataset.slide_mask
    evaluator.update_scores(mask, output)
    scores = evaluator.get_scores()

    # save result
    
    output_rgb = class_to_RGB(output)
    mask_rgb = class_to_RGB(mask)
    output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_path, slide+'_output.png'), output_rgb)
    cv2.imwrite(os.path.join(output_path, slide+'_mask.png'), mask_rgb)

    tbar.set_description('Slide: {}'.format(slide) + ', mIOU: %.5f; slide time: %.2f' % (np.mean(np.nan_to_num(scores["iou"][1:])), slide_time.avg))

log = ""
log = log + str(task_name) + 'slide inference \n'
log = log + "mIOU = " + str(np.mean(np.nan_to_num(scores["iou"][1:]))) + '\n'
log = log + "IOU: " + str(scores['iou']) + '\n'








