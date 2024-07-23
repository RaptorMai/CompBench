import json
import argparse
import os
from collections import defaultdict, Counter
import copy
import shutil
import sys
import mat73
import numpy as np
from tqdm import tqdm


############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract images from CompCars')
parser.add_argument('--root_dir',type=str,default="dataset/comp_cars", help='directory of CompCars dataset')
parser.add_argument('--split',type=str,default="test", help='name of dataset split')


config = parser.parse_args()


def get_imgs_split(config):
 
    imgs_split = []
    with open(os.path.join(config.root_dir, "data", "train_test_split", "classification", f"{config.split}.txt"))  as fp:
        for line in fp:
            imgs_split.append(line)
        
    return imgs_split
    

def save_imgs(imgs_split, config):


    # folder storing the original image
    org_f = os.path.join(config.root_dir, "data", "image")   

    # output folder of {split} images
    out_f = os.path.join(config.root_dir, "data", f"{config.split}_image")
    for img_id in tqdm(imgs_split):
        img_id = img_id.strip()
        org_img_pth = os.path.join(org_f, img_id)

        make_model_year, img_name = "/".join(img_id.split("/")[:-1]), img_id.split("/")[-1]
        
        # store the image
        img_out_f = os.path.join(out_f, make_model_year)
        if not os.path.exists(img_out_f):
            os.makedirs(img_out_f)
        shutil.copy2(org_img_pth, os.path.join(img_out_f, img_name))

    print(f"# images {config.split}: {len(imgs_split)}")

def main():
    # Group images by split
    imgs_split = get_imgs_split(config)

    # Save images
    save_imgs(imgs_split, config)


if __name__ == '__main__':
    main()

