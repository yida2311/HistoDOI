import os
import time
import torch 
import cv2
from tqdm import tqdm

from .segmentor.dynamicUNet import UNet
from .dataset.transformer_seg import TransformerSegVal
from .dataset.dataset_seg import OralSlideSeg, collate
from .utils.metrics import AverageMeter
from .utils.data import class_to_RGB
from .helper.helper_unet import SlideInference
from .helper.utils import create_model_load_weights, create_model_load_weights_v2
from .helper.config_unet import Config

# # DPP 1
# dist.init_process_group('nccl')
# # DPP 2
# local_rank = dist.get_rank()
# print(local_rank)
# torch.cuda.set_device(local_rank)
# device = torch.device('cuda', local_rank)

def main(cfg, slide_list):
    print(cfg.task_name)

    if not os.path.exists(cfg.log_path): 
        os.makedirs(cfg.log_path)
    if not os.path.exists(cfg.output_path): 
        os.makedirs(cfg.output_path)

    ###################################
    print("preparing datasets and dataloaders......")
    slideset_cfg = cfg.slideset_cfg
    evaluation = slideset_cfg["label"]
    slide_time = AverageMeter("DataTime", ':3.3f')
    transformer = TransformerSegVal
    dataset = OralSlideSeg(
        slide_list, 
        slideset_cfg["img_dir"], 
        slideset_cfg["meta_file"], 
        mask_dir=slideset_cfg["mask_dir"], 
        label=slideset_cfg['label'], 
        transform=transformer
    )

    ###################################
    print("creating models......")
    model = UNet(cfg.n_class, encoder_name=cfg.encoder, **cfg.model_cfg)
    # model = create_model_load_weights_v2(model, evaluation=True, ckpt_path=ckpt_path)
    model = create_model_load_weights(model, evaluation=True, ckpt_path=cfg.ckpt_path)
    model.cuda()

    f_log = open(os.path.join(cfg.log_path, "_test.log"), 'w')
    #######################################
    evaluator = SlideInference(cfg.n_class, cfg.num_workers, cfg.batch_size)
    num_slides = len(dataset.slides)
    tbar = tqdm(range(num_slides))

    for i in tbar:
        start_time = time.time()
        dataset.get_patches_from_index(i)
        prediction, output = evaluator.inference(dataset, model)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(cfg.output_path, dataset.slide+'.png'), output)
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


cfg = Config(train=False)
slide_list = sorted(os.listdir(cfg.slideset_cfg["img_dir"]))
main(cfg, slide_list)



















