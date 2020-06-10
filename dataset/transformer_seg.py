import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import random
import albumentations
from albumentations.pytorch import ToTensor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def TransformerSeg(image=None, mask=None):
    master = albumentations.Compose([
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
    ])

    result = master(image=image, mask=mask)
    result['image'] = ToTensor()(image=result['image'])['image']
    result['mask'] = torch.tensor(result['mask'], dtype=torch.long)

    return result


def TransformerSegVal(image=None, mask=None):
    img_trans = albumentations.Compose([
        albumentations.Normalize(mean=[0.798, 0.621, 0.841], std=[0.125, 0.228, 0.089]),
        ToTensor(),
    ])
    result = img_trans(image=image)
    if mask is not None:
        result['mask'] = torch.tensor(mask, dtype=torch.long)
        # print(result['mask'].size())
        # print(result['mask'].mean())
        # print(result['mask'].max())

    return result




