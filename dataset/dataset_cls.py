import os
import torch
import cv2 
import random
import json

import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader


def cv2_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img 


class OralDatasetCls(Dataset):
    """ OralDataset for classification"""
    def __init__(self,
                data_dir,
                meta_file,
                label=True,  # True: has label
                transform=None):
        super(OralDatasetCls, self).__init__()

        self.data_dir = data_dir
        self.label = label 
        self.transform = transform

        df = pd.read_csv(meta_file)
        self.samples = df  
    
    def __getitem__(self, index):
        info = self.samples.iloc[index]
        img_path = os.path.join(os.path.join(self.data_dir, info['slide_id']), info['image_id'])
        img = cv2_loader(img_path)

        sample = {}
        if self.transform:
            img = self.transform(image=img)['image']
        sample['image'] = img

        if self.label:
            label = info['target'] - 1
            sample['label'] = label
        
        return sample

    def __len__(self):
        return len(self.samples)


class OralSlideCls(Dataset):
    """ OralSlide for classification"""
    def __init__(self,
                data_dir, 
                meta_file,
                label=True,
                transform=None):
        super(OralSlideCls, self).__init__()
        self.data_dir = data_dir
        self.label = label 
        self.transform = transform

        self.slides = sorted(os.listdir(self.data_dir))

        with open(meta_file, 'r') as f:
            cnt = json.load(f)
        self.info = cnt  # {"slide": {"size":[h, w], "tiles": [x, y], "step":[step_x, step_y]}}

        self.slide = ""
        self.slide_mask = None
        self.slide_size = None
        self.samples = []
    
    def get_patches_from_index(self, index):
        self.slide = self.slides[index]
        
        slide_dir = os.path.join(self.data_dir, self.slide)
        self.samples = os.listdir(slide_dir)

        tiles = self.info[self.slide]['tiles']
        self.slide_size = tuple(tiles)
        self.slide_mask = np.zeros(tuple(tiles), dtype='uint8')
    
    def get_patches_from_name(self, name):
        self.slide = name 
        
        slide_dir = os.path.join(self.data_dir, self.slide)
        self.samples = os.listdir(slide_dir)

        tiles = self.info[self.slide]['tiles']
        self.slide_size = tuple(tiles)
        self.slide_mask = np.zeros(tuple(tiles), dtype='uint8')
    
    def __getitem__(self, index):
        patch = self.samples[index]
        img_path = os.path.join(os.path.join(self.data_dir, self.slide), patch)
        img = cv2_loader(img_path)

        sample = {}
        if self.transform:
            img = self.transform(image=img)['image']
        sample['image'] = img

        if self.label:
            ver, col, target = self._parse_patch_name(patch)
            sample['coord'] = (ver, col)
            sample['label'] = target
            self.slide_mask[ver, col] = target 
        else:
            ver, col = self._parse_patch_name(patch)
            sample['coord'] = (ver, col)

        return sample
    
    def __len__(self):
        return len(self.samples)

    def _parse_patch_name(self, patch):
        sp = patch.split('_')
        ver = int(sp[2])
        col = int(sp[3])
        if self.label:
            target = int(sp[4])
            return ver, col, target
        
        return ver, col 


def collate(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [b[key] for b in batch]
    batch_dict['image'] = torch.stack(batch_dict['image'], dim=0)
    batch_dict['label'] = torch.tensor(batch_dict['label'], dtype=torch.long)

    return batch_dict
