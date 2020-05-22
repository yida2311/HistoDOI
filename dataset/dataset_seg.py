import os
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.autograd import Variable
import torch 
import cv2
import pandas as pd 

ImageFile.LOAD_TRUNCATED_IMAGES = True


def cv2_image_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img 

def cv2_mask_loader(path):
    mask = cv2.imread(path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return mask


# def resize(image, shape, label=False):
#     '''
#     resize PIL image
#     shape: (w, h)
#     '''
#     if label:
#         return image.resize(shape, Image.NEAREST)
#     else:
#         return image.resize(shape, Image.BILINEAR)

class OralDatasetSeg(Dataset):
    """ OralDataset for classification"""
    def __init__(self,
                data_dir,
                meta_file,
                label=True,  # True: has label
                img_suffix='train',
                mask_suffix='train_mask',
                transform=None):
        super(OralDatasetCls, self).__init__()

        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, img_suffix)
        self.label = label 
        if self.label:
            self.mask_dir = os.path.join(self.data_dir, mask_suffix)
        self.transform = transform

        df = pd.read_csv(meta_file)
        self.samples = df  
    
    def __getitem__(self, index):
        info = self.samples.iloc[index]
        img_path = os.path.join(os.path.join(self.img_dir, info['slide_id']), info['image_id'])
        img = cv2_image_loader(img_path)

        if self.transform:
            if self.label:
                mask_path = os.path.join(os.path.join(self.mask_dir, info['slide_id']), info['image_id'])
                mask = cv2_mask_loader(mask_path)
                sample = self.transform(image=img, mask=mask)
            else:
                sample = self.transform(image=img)
        
        return sample
    
    def __len__(self):
        return len(self.samples)










