import os 
import shutil
from PIL import Image
import numpy as np 
import sys
import pandas as pd 
sys.path.append("../")
from utils.data import boundary_patch_parser


def boundary_patch_getter(slide, mask, src_dir, src_mask_dir, save_dir, save_mask_dir):
    slide_dir = os.path.join(src_dir, slide)
    slide_mask_dir = os.path.join(src_mask_dir, slide)
    slide_save_dir = os.path.join(save_dir, slide)
    slide_save_mask_dir = os.path.join(save_mask_dir, slide)
    if not os.path.exists(slide_save_dir):
        os.makedirs(slide_save_dir)
    if not os.path.exists(slide_save_mask_dir):
        os.makedirs(slide_save_mask_dir)

    indices = boundary_patch_parser(mask, width=3)
    num = 0
    for indice in indices:
        patch_name = slide + '_' + str(indice[0]) + '_' + str(indice[1]) + '_.png'

        src = os.path.join(slide_dir, patch_name)
        dst = os.path.join(slide_save_dir, patch_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            num += 1
        else:
            continue

        src_mask = os.path.join(slide_mask_dir, patch_name)
        dst_mask = os.path.join(slide_save_mask_dir, patch_name)
        shutil.copy(src_mask, dst_mask)
        
    return num


def boundary_patch_getter_all(mask_dir, src_dir, src_mask_dir, save_dir, save_mask_dir):
    num = 0
    for c in os.listdir(mask_dir):
        slide = c.split('.')[0]
        print(slide)
        mask = np.array(Image.open(os.path.join(mask_dir, c)), dtype='uint8')
        
        num += boundary_patch_getter(slide, mask, src_dir, src_mask_dir, save_dir, save_mask_dir)
    
    print('Num of patches for seg: {}'.format(num))

def boundary_patch_meta(src_meta_path, dst_meta_path, data_dir):
    src_meta = pd.read_csv(src_csv_path)
    dst_meta = pd.DataFrame(columns=src_meta.columns)
    for slide in os.listdir(data_dir):
        print(slide)
        for patch in os.listdir(os.path.join(data_dir, slide)):
            index = src_meta[src_meta['image_id']==patch].index
            # print(index)
            dst_meta = dst_meta.append(src_meta.iloc[index])
            # print(dst_meta)
    print(dst_meta)
    dst_meta.to_csv(dst_meta_path)


if __name__ == '__main__':
    size = 224
    pattern = 'val'

    mask_dir = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/slide/mask/' + pattern + '_' + str(size) + '_mask_5x'
    root = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/5x_' + str(size) + '/'
    src_dir = root + pattern + '_' + str(size)
    src_mask_dir = root + pattern + '_mask_' + str(size)
    save_dir = root + pattern + '_bd_' + str(size)
    save_mask_dir = root + pattern + '_bd_mask_' + str(size)
    src_csv_path  = root + pattern + '_' + str(size) + '.csv'
    dst_csv_path  = root + pattern + '_bd_' + str(size) + '.csv'

    boundary_patch_getter_all(mask_dir, src_dir, src_mask_dir, save_dir, save_mask_dir)
    boundary_patch_meta(src_csv_path, dst_csv_path, save_dir)

