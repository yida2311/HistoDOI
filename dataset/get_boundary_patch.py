import os 
import shutil
from PIL import Image
import numpy as np 
import sys
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




if __name__ == '__main__':
    mask_dir = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/slide/mask/val_512_mask_5x'
    src_dir = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/seg_5x/val_512'
    src_mask_dir = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/seg_5x/val_mask_512'
    save_dir = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/seg_5x/val_bd_512'
    save_mask_dir = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/seg_5x/val_bd_mask_512'

    boundary_patch_getter_all(mask_dir, src_dir, src_mask_dir, save_dir, save_mask_dir)

