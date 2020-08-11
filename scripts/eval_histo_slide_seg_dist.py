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
from helper_seg import create_model_load_weights, SlideEvaluator, create_model_load_weights_v2
from option_seg import Options 
from utils.data import class_to_RGB


# # DPP 1
# dist.init_process_group('nccl')
# # DPP 2
# local_rank = dist.get_rank()
# print(local_rank)
# torch.cuda.set_device(local_rank)
# device = torch.device('cuda', local_rank)

def main(slide_list):
    # arguments
    args = Options().parse()
    n_class = args.n_class

    task_name = args.task_name
    print(task_name)

    img_path = args.img_path_val
    mask_path = args.mask_path_val
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
    transformer = TransformerSegVal
    dataset = OralSlideSeg(slide_list, img_path, meta_path, mask_dir=mask_path, label=False, transform=transformer)

    ###################################
    print("creating models......")
    model = generate_unet(num_classes=n_class, encoder_name='resnet34')
    # model = create_model_load_weights_v2(model, evaluation=True, ckpt_path=ckpt_path)
    model = create_model_load_weights(model, evaluation=True, ckpt_path=ckpt_path)
    model.cuda()

    f_log = open(log_path + task_name + "_test.log", 'w')
    #######################################
    evaluator = SlideInference(n_class, num_workers, batch_size)
    num_slides = len(dataset.slides)
    tbar = tqdm(range(num_slides))

    for i in tbar:
        start_time = time.time()
        dataset.get_patches_from_index(i)
        prediction, output = evaluator.inference(dataset, model)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_path, dataset.slide+'_ouput.png'), output)
        slide_time.update(time.time()-start_time)
        
        if evaluation:
            mask = dataset.get_slide_mask_from_index(i)
            evaluator.update_scores(mask, prediction)
            scores = evaluator.get_scores()
            tbar.set_description('Slide: {}'.format(slide) + ', mIOU: %.5f; slide time: %.2f' % (scores['iou_mean'], slide_time.avg))
        else:
            tbar.set_description('Slide: {}'.format(slide) + ', slide time: %.2f' % (slide_time.avg))
    
    if evaluation:
        scores = evaluator.get_scores()
        print(evaluator.metrics.confusion_matrix)
    log = ""
    log = log + str(task_name) + '   slide inference \n'
    if evaluation:
        log = log + "mIOU = " + str(scores['iou_mean']) + '\n'
        log = log + "IOU: " + str(scores['iou']) + '\n'
        log = log + "Dice: " + str(scores['dice']) + '\n'
    
    log = log + "[Time consuming %.2fs][%.2fs per slide]" % (slide_time.sum, slide_time.avg) + '\n'
    log += "================================\n"
    print(log)

    f_log.write(log)
    f_log.close()


slide_list = []
main(slide_list)



















