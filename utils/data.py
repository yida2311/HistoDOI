import os
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import torch 
import cv2
import pandas as pd 

ImageFile.LOAD_TRUNCATED_IMAGES = True


def RGB_mapping_to_class(label):
    h, w = label.shape[0], label.shape[1]
    classmap = np.zeros(shape=(h, w))

    indices = np.where(np.all(label == (255, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 1
    indices = np.where(np.all(label == (0, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 2
    indices = np.where(np.all(label == (0, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 3
    indices = np.where(np.all(label == (0, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 0

    return classmap


def class_to_RGB(label):
    h, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(h, w, 3)).astype(np.uint8)

    indices = np.where(label == 1)
    print(indices)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 0]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 0]
    indices = np.where(label == 3)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 255]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]

    return colmap


def class_to_target(inputs, numClass):
    batchSize, h, w = inputs.shape[0], inputs.shape[1], inputs.shape[2]
    target = np.zeros(shape=(batchSize, h, w, numClass), dtype=np.float32)
    for index in range(numClass):
        indices = np.where(inputs == index)
        temp = np.zeros(shape=numClass, dtype=np.float32)
        temp[index] = 1
        target[indices[0].tolist(), indices[1].tolist(), indices[2].tolist(), :] = temp
    return target.transpose(0, 3, 1, 2) 


def label_bluring(inputs):
    batchSize, numClass, height, width = inputs.shape
    outputs = np.ones((batchSize, numClass, height, width), dtype=np.float)
    for batchCnt in range(batchSize):
        for index in range(numClass):
            outputs[batchCnt, index, ...] = cv2.GaussianBlur(inputs[batchCnt, index, ...].astype(np.float), (7, 7), 0)
    return outputs


def one_hot_gaussian_blur(index, classes):
    '''
    index: numpy array b, h, w
    classes: int
    '''
    mask = np.transpose((np.arange(classes) == index[..., None]).astype(float), (0, 3, 1, 2))
    b, c, _, _ = mask.shape
    for i in range(b):
        for j in range(c):
            mask[i][j] = cv2.GaussianBlur(mask[i][j], (0, 0), 8)

    return mask
    

def collate(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [b[key] for b in batch]
    for key in batch_dict.keys():
        if not (key=='output_size' or key=='id'):
            batch_dict[key] = torch.stack(batch_dict[key], dim=0)

    return batch_dict


def boundary_patch_parser(mask, width=3):
    tumor = np.array(mask==3, dtype='uint8')
    mucosa = np.array(mask==2, dtype='uint8')
    # get contours
    tumor_contour = get_contour(tumor, thresh=0)
    mucosa_contour = get_contour(mucosa, thresh=0)
    # poly contours
    target = np.zeros_like(mask)
    target = cv2.polylines(target, tumor_contour, True, 1, 1)
    target = cv2.polylines(target, mucosa_contour, True, 1, 1)
    # dilate contours and get indices
    target = cv2.dilate(target, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1)
    indices = np.where(target==1)
    indices = [(h, w) for h, w in zip(indices[0], indices[1])]

    return indices


def get_contour(img, thresh=1):
    """
    img: 单通道图像
    save_dir: optional
    """
    _, thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours