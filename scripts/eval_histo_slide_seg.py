import os
import time
import numpy as np 
import torch 
import cv2
from tqdm import tqdm

# from models.segmentor.unet import UNet
from models.segmentation_models_pytorch.seg_generator import generate_unet
from dataset.transformer_seg import TransformerSegVal
from dataset.dataset_seg import OralSlideSeg, collate
from utils.metrics import AverageMeter
from helper_seg import create_model_load_weights, SlideInference
from option_seg import Options 
from utils.data import class_to_RGB


# arguments
args = Options().parse()
n_class = args.n_class

task_name = args.task_name
print(task_name)

img_path = args.img_path_val
# mask_path = args.mask_path_val
meta_path = args.meta_path_val
slide_mask_path = args.slide_mask_path
log_path = args.log_path
output_path = os.path.join(args.output_path, task_name)
npy_path = os.path.join(args.npy_path, task_name)
ckpt_path = args.ckpt_path

if not os.path.exists(log_path): 
    os.makedirs(log_path)
if not os.path.exists(output_path): 
    os.makedirs(output_path)
if not os.path.exists(npy_path): 
    os.makedirs(npy_path)

###################################
evaluation = args.evaluation
test = evaluation and True
print("evaluation:", evaluation, "test:", test)

###################################
print("preparing datasets and dataloaders......")
batch_size = args.batch_size
num_workers = args.num_workers

slide_time = AverageMeter("DataTime", ':3.3f')

transformer = TransformerSegVal
slide_list = os.listdir(img_path)
# slide_list = [slide_list[45]]
dataset = OralSlideSeg(slide_list, img_path, meta_path, slide_mask_dir=slide_mask_path, label=True, transform=transformer)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate, shuffle=False, pin_memory=True)

###################################
print("creating models......")
# model = UNet(n_channels=3, n_classes=n_class)
model = generate_unet(num_classes=n_class, encoder_name='resnet34')
model = create_model_load_weights(model, evaluation=True, ckpt_path=ckpt_path)
model.cuda()

f_log = open(log_path + task_name + "_test.log", 'w')
#######################################
evaluator = SlideInference(n_class, num_workers, batch_size)

num_slides = len(dataset.slides)
tbar = tqdm(range(num_slides))
for i in tbar:
    dataset.get_patches_from_index(i)
    
    start_time = time.time()
    prediction, output_rgb, output = evaluator.inference(dataset, model)
    slide_time.update(time.time()-start_time)
    start_time = time.time()

    slide = dataset.slide
    # mask = dataset.slide_mask
    mask = dataset.get_slide_mask_from_index(i)
    evaluator.update_scores(mask, prediction)
    scores = evaluator.get_scores()

    # save result
    print(output.shape)
    # mask_rgb = class_to_RGB(mask)
    output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_BGR2RGB)
    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_path, slide+'_output.png'), output_rgb)
    # cv2.imwrite(os.path.join(output_path, slide+'_mask.png'), mask_rgb)
    np.save(os.path.join(npy_path, slide+'.npy'), output)
    tbar.set_description('Slide: {}'.format(slide) + ', mIOU: %.5f; slide time: %.2f' % (scores['iou_mean'], slide_time.avg))

scores = evaluator.get_scores()
print(evaluator.metrics.confusion_matrix)
log = ""
log = log + str(task_name) + '   slide inference \n'
log = log + "mIOU = " + str(scores['iou_mean']) + '\n'
log = log + "tmIOU = " + str(scores['iou_tm']) + '\n'
log = log + "IOU: " + str(scores['iou']) + '\n'
log = log + "Dice: " + str(scores['dice']) + '\n'
log += "================================\n"
print(log)

f_log.write(log)
f_log.close()




















