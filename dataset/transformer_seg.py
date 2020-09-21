import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import random
import albumentations
from albumentations.pytorch import ToTensor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TransformerSeg:
    def __init__(self, crop_size=1024):
        self.master = albumentations.Compose([
            albumentations.RandomCrop(crop_size, crop_size),
            albumentations.RandomRotate90(p=0.5),
            albumentations.Transpose(p=0.5),
            albumentations.Flip(p=0.5),
            albumentations.OneOf([
                # albumentations.CLAHE(clip_limit=2),
                # albumentations.IAASharpen(),
                # albumentations.IAAEmboss(),
                # albumentations.JpegCompression(),
                # albumentations.Blur(),
                # albumentations.GaussNoise(),
                albumentations.RandomBrightness(),
                albumentations.RandomContrast(),
                albumentations.HueSaturationValue(),
            ], p=0.5),
            albumentations.ElasticTransform(),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            albumentations.Normalize(mean=[0.798, 0.621, 0.841], std=[0.125, 0.228, 0.089]),
        ])
        self.to_tensor = ToTensor()

    def __call__(self, image=None, mask=None):
        result = self.master(image=image, mask=mask)
        result['image'] = self.to_tensor(image=result['image'])['image']
        result['mask'] = torch.tensor(result['mask'], dtype=torch.long)
        return result
    

class TransformerSegVal:
    def __init__(self):
        self.master = img_trans = albumentations.Compose([
            albumentations.Normalize(mean=[0.798, 0.621, 0.841], std=[0.125, 0.228, 0.089]),
            ToTensor(),
        ])
    
    def __call__(self, image=None, mask=None):
        result = self.master(image=image)
        if mask is not None:
            result['mask'] = torch.tensor(mask, dtype=torch.long)
        return result
    
    
class TransformerSegGL:
    def __init__(self, crop_size=1024):
        self.master = albumentations.Compose([
            albumentations.RandomCrop(crop_size, crop_size),
            albumentations.RandomRotate90(p=0.5),
            albumentations.Transpose(p=0.5),
            albumentations.Flip(p=0.5),
            albumentations.OneOf([
                # albumentations.CLAHE(clip_limit=2),
                # albumentations.IAASharpen(),
                # albumentations.IAAEmboss(),
                # albumentations.JpegCompression(),
                # albumentations.Blur(),
                # albumentations.GaussNoise(),
                albumentations.RandomBrightness(),
                albumentations.RandomContrast(),
                albumentations.HueSaturationValue(),
            ], p=0.5),
            albumentations.ElasticTransform(),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            albumentations.Normalize(mean=[0.798, 0.621, 0.841], std=[0.125, 0.228, 0.089]),
        ])

    def __call__(self, image=None, mask=None):
        result = self.master(image=image, mask=mask)
        result['image'] = np2pil(result['image'])
        result['mask'] = np2pil(result['mask'], mode='L')
        return result
    

class TransformerSegGLVal:
    def __init__(self):
        self.master = img_trans = albumentations.Compose([
            albumentations.Normalize(mean=[0.798, 0.621, 0.841], std=[0.125, 0.228, 0.089]),
        ])
    
    def __call__(self, image=None, mask=None):
        result = self.master(image=image)
        result['image'] = np2pil(result['image'])
        if mask is not None:
            result['mask'] = np2pil(mask, mode='L')
        return result
    

def np2pil(np_arr, mode="RGB"):
    return Image.fromarray(np_arr, mode=mode)
