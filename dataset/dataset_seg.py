import os
import random
import torch 
import math
import cv2
import json
import numpy as np
import pandas as pd
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import Dataset


def cv2_image_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img 

def cv2_mask_loader(path):
    mask = cv2.imread(path, 0)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return mask


class OralDatasetSeg(Dataset):
    """ OralDataset for classification"""
    def __init__(self,
                img_dir,
                mask_dir,
                meta_file,
                label=True,  # True: has label
                transform=None):
        super(OralDatasetSeg, self).__init__()
        self.img_dir = img_dir
        self.label = label 
        if self.label:
            self.mask_dir = mask_dir
        self.transform = transform

        df = pd.read_csv(meta_file)
        self.samples = df  
    
    def __getitem__(self, index):
        info = self.samples.iloc[index]
        img_path = os.path.join(os.path.join(self.img_dir, info['slide_id']), info['image_id'])
        img = cv2_image_loader(img_path)
        sample = {}
        
        if self.transform:
            if self.label:
                mask_path = os.path.join(os.path.join(self.mask_dir, info['slide_id']), info['image_id'])
                mask = cv2_mask_loader(mask_path)
                sample = self.transform(image=img, mask=mask)
            else:
                sample = self.transform(image=img)
        sample['id'] = info['image_id'].split('.')[0]

        return sample
    
    def __len__(self):
        return len(self.samples)


class OralSlideSeg(Dataset):
    """OralSlide for segmentation"""
    def __init__(self,
                img_dir, 
                mask_dir,
                meta_file,
                label=True,
                transform=None):
        super(OralSlideSeg, self).__init__()
        self.img_dir = img_dir
        self.label = label 
        if self.label:
            self.mask_dir = mask_dir
        self.transform = transform

        self.slides = sorted(os.listdir(self.img_dir))
        with open(meta_file, 'r') as f:
            cnt = json.load(f)
        self.info = cnt  # {"slide": {"size":[h, w], "tiles": [x, y], "step":[step_x, step_y]}}

        self.slide = ""
        self.slide_mask = None
        self.slide_size = None
        self.slide_step = None
        self.slide_template = None
        self.samples = []

    def get_slide_mask_from_index(self, index):
        slide = self.slides[index]
        slide_dir = os.path.join(self.img_dir, slide)
        samples = os.listdir(slide_dir)

        size = self.info[slide]['size']
        step = self.info[slide]['step']
        slide_size = tuple(size)
        slide_mask = np.zeros(slide_size, dtype='uint8')

        for c in samples:
            ver, col = self._parse_patch_name(c)
            mask_path = os.path.join(os.path.join(self.mask_dir, self.slide), c)
            mask = cv2_mask_loader(mask_path)
            h, w = mask.shape
            x = math.floor(ver*step[0]); y = math.floor(col*step[1])
            slide_mask[x:x+h, y:y+w] = mask
        
        return slide_mask

    def get_patches_from_index(self, index):
        self.slide = self.slides[index]
        
        slide_dir = os.path.join(self.img_dir, self.slide)
        self.samples = os.listdir(slide_dir)

        size = self.info[self.slide]['size']
        step = self.info[self.slide]['step']
        self.slide_size = tuple(size)
        self.slide_step = step
        self.slide_mask = np.zeros(self.slide_size, dtype='uint8')
        self.slide_template = np.zeros(self.slide_size, dtype='uint8')
    
    def __getitem__(self, index):
        patch = self.samples[index]
        img_path = os.path.join(os.path.join(self.img_dir, self.slide), patch)
        img = cv2_image_loader(img_path)
        sample = {}

        if self.transform:
            if self.label:
                mask_path = os.path.join(os.path.join(self.mask_dir, self.slide), patch)
                mask = cv2_mask_loader(mask_path)
                sample = self.transform(image=img, mask=mask)
            else:
                sample = self.transform(image=img)
        
        ver, col = self._parse_patch_name(patch)
        sample['coord'] = (ver, col)

        return sample
    
    def __len__(self):
        return len(self.samples)

    def _parse_patch_name(self, patch):
        sp = patch.split('_')
        ver = int(sp[2])
        col = int(sp[3])
        
        return ver, col 


def collate(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [b[key] for b in batch]
    
    batch_dict['image'] = torch.stack(batch_dict['image'], dim=0)
    batch_dict['mask'] = torch.stack(batch_dict['mask'], dim=0)

    return batch_dict








