import os
import csv 
import random
import pandas as pd 


def trainval_split(root, suffix, save_dir):
    content = []
    num = 0
    suf_root = os.path.join(root, suffix)
    for slide in os.listdir(suf_root):
        slide_dir = os.path.join(suf_root, slide)
        subset = random.randint(0, 4)
        for subslide in os.listdir(slide_dir):
            subslide_dir = slide + '/' + subslide
            content.append(dict(subslide=subslide_dir, subset=subset))
            num += 1
    random.shuffle(content)

    df = pd.DataFrame(content, columns=['subslide', 'subset'])
    df.to_csv(os.path.join(save_dir, 'trainval_t.csv'), index=False)
    print('Number of training patches: {}'.format(num))
    print('Finished!!')


def make_trainval_mini(csv_dir, save_dir, num):
    df = pd.read_csv(csv_dir)
    df = list(df.itertuples(index=False))
    random.shuffle(df)
    new_df = df[:num]
    gf = pd.DataFrame(new_df, columns=['subslide', 'subset'])
    gf.to_csv(os.path.join(save_dir, 'trainval_t_mini.csv'), index=False)


if __name__ == '__main__':
    suffix = 'train'
    root = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_sf/subslide/'
    save_dir = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_sf/'
    csv_dir = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_sf/trainval_t.csv'

    trainval_split(root, suffix, save_dir)
    make_trainval_mini(csv_dir, save_dir, num=200)