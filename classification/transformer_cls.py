import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import random
import albumentations
from albumentations.pytorch import ToTensor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


TransformerCls = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.OneOf([
        albumentations.CLAHE(clip_limit=2),
        albumentations.IAASharpen(),
        albumentations.IAAEmboss(),
        albumentations.RandomBrightness(),
        albumentations.RandomContrast(),
        albumentations.JpegCompression(),
        albumentations.Blur(),
        albumentations.GaussNoise(),
    ], p=0.5),
    albumentations.HueSaturationValue(p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    albumentations.Normalize(mean=[0.798, 0.621, 0.841], std=[0.125, 0.228, 0.089]),
    ToTensor(),
])

TransformerClsVal = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.Normalize(mean=[0.798, 0.621, 0.841], std=[0.125, 0.228, 0.089]),
    ToTensor(),
])

TransformerClsTTA = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.RandomRotate90(p=1),
    albumentations.Normalize(mean=[0.798, 0.621, 0.841], std=[0.125, 0.228, 0.089]),
    ToTensor(),
])




# torchvision transformer
# class TransformerCls(object):
#     def __init__(self, size):
#         # self.crop = RandomCrop(size=size)
#         # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#         #                               std=[0.229, 0.224, 0.225])
#         # self.color_jitter = transforms.ColorJitter(brightness=16.0/255, contrast=0.15, saturation=0.2, hue=0.04)

#         self.crop = transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1))
#         self.normalize = transforms.Normalize(mean=[0.798, 0.621, 0.841],
#                                             std=[0.125, 0.228, 0.089])
#         self.color_jitter = transforms.RandomChoice([transforms.ColorJitter(brightness=0.3),
#                                                     transforms.ColorJitter(contrast=0.3), 
#                                                     transforms.ColorJitter(saturation=0.3),
#                                                     transforms.ColorJitter(hue=0.3),
#                                                     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
#                                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3), 
#                                                     ])
#         self.space_trans = transforms.RandomChoice(transforms.RandomRotation((0, 0)),
#                                                     transforms.RandomHorizontalFlip(p=1),
#                                                     transforms.RandomVerticalFlip(p=1),
#                                                     transforms.RandomRotation((90, 90)),
#                                                     transforms.RandomRotation((180, 180)),
#                                                     transforms.RandomRotation((270, 270)),
#                                                     transforms.Compose([
#                                                         transforms.RandomHorizontalFlip(p=1),
#                                                         transforms.RandomRotation((90, 90)),
#                                                     ]),
#                                                     transforms.Compose([
#                                                         transforms.RandomHorizontalFlip(p=1),
#                                                         transforms.RandomRotation((270, 270)),
#                                                     ]),
#                                                     )
        
#         self.to_tensor = transforms.ToTensor()

#     def __call__(self, img):
#         img = self.crop(imgt)
#         img = self.space_trans(img)
#         img = self.color_jitter(img.convert('RGB'))
#         img = self.to_tensor(img)
#         img = self.normalize(img)

#         return img


# class ValTransformerCls(object):
#     def __init__(self):
#         self.normalize = transforms.Normalize(mean=[0.798, 0.621, 0.841],
#                                             std=[0.125, 0.228, 0.089])
#         self.to_tensor = transforms.ToTensor()
    
#     def __call__(self, img):
#         img = self.to_tensor(img)
#         img = self.normalize(img)

#         return img