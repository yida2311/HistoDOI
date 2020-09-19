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
                train=False,
                label=True,  # True: has label
                transform=None):
        super(OralDatasetCls, self).__init__()

        self.data_dir = data_dir
        self.train = train
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
            if self.train:
                label = self.parse_ratio(info['new_ratio'])
                # print(label)
            else:
                # label = 0 if info['target']==1 else 1
                label = int(info['target'] - 1)
            sample['label'] = label
        return sample

    def __len__(self):
        return len(self.samples)
    
    def parse_ratio(self, ratio: str):
        ratio = ratio[1:].split(' ')
        assert len(ratio) == 3
        res = [float(c[:-1]) for c in ratio]
        return torch.tensor(res, dtype=torch.float)



class OralSlideCls(Dataset):
    """ OralSlide for classification"""
    def __init__(self,
                data_dir, 
                meta_file,
                slide_file,
                label=True,
                transform=None):
        super(OralSlideCls, self).__init__()
        self.data_dir = data_dir
        self.label = label 
        self.transform = transform

        self.slides = sorted(os.listdir(self.data_dir))

        with open(slide_file, 'r') as f:
            cnt = json.load(f)
        self.info = cnt  # {"slide": {"size":[h, w], "tiles": [x, y], "step":[step_x, step_y]}}

        df = pd.read_csv(meta_file)
        self.df = df.set_index('image_id')

        self.slide = ""
        self.slide_mask = None
        self.slide_size = None
        self.samples = []

    def get_slide_mask_from_index(self, index):
        slide = self.slides[index]
        slide_dir = os.path.join(self.data_dir, slide)
        samples = os.listdir(slide_dir)

        tiles = self.info[slide]['tiles']
        slide_size = tuple(tiles)
        slide_mask = np.zeros(tuple(tiles), dtype='uint8')

        for c in samples:
            info = self.df.loc[c]
            row = info['row']
            col = info['col']
            target = info['target']
            slide_mask[row, col] = target
        
        return slide_mask
    
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
        info = self.df.loc[patch]
        sample = {}
        if self.transform:
            img = self.transform(image=img)['image']
        sample['image'] = img

        if self.label:
            sample['coord'] = (info['row'], info['col'])
            sample['label'] = int(info['target'])
            self.slide_mask[info['row'], info['col']] = int(info['target'])
        else:
            sample['coord'] = (info['row'], info['col'])

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
    if isinstance(batch_dict['label'][0], int):
        batch_dict['label'] = torch.tensor(batch_dict['label'], dtype=torch.long)
    else:
        batch_dict['label'] = torch.stack(batch_dict['label'], dim=0)

    return batch_dict

