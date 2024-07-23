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
parser=argparse.ArgumentParser(description='Extract images from CelebA')
parser.add_argument('--root_dir',type=str,default="dataset/celeba", help='directory of CelebA dataset')
parser.add_argument('--split',type=str,default="test", help='name of dataset split')


config = parser.parse_args()


def emotional_imgs(config):
    
    if config.split == "test":
        split_id = 2
    elif config.split == "val":
        split_id = 1
    elif config.split == "train":
        split_id = 0
    
    imgs_split = []
    with open(os.path.join(config.root_dir, "list_eval_partition.txt"))  as fp:
        for line in fp:
            img_id, img_split = line.split(" ")[0].strip(), line.split(" ")[1].strip()
            if img_split == split_id:
                imgs_split.append(img_id)

    # emotional attribute
    emo_atts  = ["Smiling"]
    emo_atts_idxs = []
    
    att_lst = None
    
    targ_imgs_ids = []
    targ_imgs_emo_atts = []
    with open(os.path.join(config.root_dir, "list_attr_celeba.txt"))  as fp:
        for idx, line in enumerate(fp):
            line = line.strip()
            if idx == 1:
                att_lst = line.split(" ")
                for eatt in emo_atts:
                    emo_atts_idxs.append(att_lst.index(eatt))
            elif idx > 1:
                
                img_id, img_att_vec = line.split(" ")[0], line.split(" ")[1:]
                
                # remove empty string if necessary
                img_att_vec = list(filter(None, img_att_vec))
                assert len(img_att_vec) == len(att_lst)

                for eai in emo_atts_idxs:
                    if int(img_att_vec[eai]) == 1:
                        targ_imgs_ids.append(img_id)
                        targ_imgs_emo_atts.append(att_lst[eai])
            
    return targ_imgs_ids, targ_imgs_emo_atts
    

def save_imgs(targ_imgs_ids, targ_imgs_emo_atts, config):

    cnt=0
    # save images by att
    for img_id, img_att in tqdm(zip(targ_imgs_ids, targ_imgs_emo_atts), total=len(targ_imgs_ids)):
        # output directory 
        out_f = os.path.join(config.root_dir, "adj_images", config.split, img_att)
        if not os.path.exists(out_f):
            os.makedirs(out_f)
        
        # Get the image path
        img_pth = os.path.join(config.root_dir, "img_align_celeba", f"{img_id}")
        # store the image
        shutil.copy2(img_pth, os.path.join(out_f, img_id))
        cnt+=1
    print(f"# images: {cnt}")

def main():
    # extract images related to emotional attributes
    targ_imgs_ids, targ_imgs_emo_atts = emotional_imgs(config)

    # Save images
    save_imgs(targ_imgs_ids, targ_imgs_emo_atts, config)


if __name__ == '__main__':
    main()

