import json
import argparse
import os
from collections import defaultdict, Counter
import copy
import shutil
import sys
import mat73
import numpy as np


############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract images from NYU Depth v2')
parser.add_argument('--root_dir',type=str,default="dataset/nyu_depth_v2", help='directory of NYU Depth v2 dataset')
parser.add_argument('--split',type=str,default="test", help='name of dataset split')


config = parser.parse_args()



def group_by_object(config):

    # Annotations
    anns = mat73.loadmat(os.path.join(config.root_dir, "nyu_depth_v2_labeled.mat"))["labels"]
    label_lst = mat73.loadmat(os.path.join(config.root_dir, "nyu_depth_v2_labeled.mat"))["names"]

    # list of index of label which is hard to compare its sptial distance. e.g., 3: "ceiling", 10: "floor"
    # 65535 is geneerated when convert matlabe file to np array. Ignore this index
    neg_lbls = [3,10,19,39,40,
                56,61,68,74,129,
                196,212,219,551,558,
                718,723,860,866,867,
                868,871,890,65535]
    
    # split file containg image ids
    split_f = open(os.path.join(config.root_dir,f"list_{config.split}.txt"), "r")
    img_ids_split = split_f.read()
    img_ids_split = img_ids_split.split("\n")
    split_f.close()

    # HxWxN -> NxHxW
    anns = np.array(anns)
    anns = np.transpose(anns, (2,0,1))
    
    obj2img = defaultdict(list)
    for img_id, ann in enumerate(anns):
        img_id = str(img_id)
        if img_id in img_ids_split:
            lbls = np.unique(ann)
            
            # make annotation's lbel index same as label_lst's label index
            lbls = lbls-1

            for lbl in lbls:
                if lbl not in neg_lbls:
                    lbl_name = label_lst[lbl][0]
                    obj2img[lbl_name].append(img_id)
        
    return obj2img
    

def save_imgs(obj2smps, config):
    
    cnt=0
    # save images by object
    for _, obj in enumerate(obj2smps):
        # only consider a case having >= 2 images
        if len(obj2smps[obj]) >=2:
            # output directory 
            out_f = os.path.join(config.root_dir, "obj_images", config.split, obj)
            if not os.path.exists(out_f):
                os.makedirs(out_f)
            
            # Iterate samples in the object
            for img_id in (obj2smps[obj]):
                img_id_lZero = "%04d" % int(img_id)
                # Get the image path
                img_pth = os.path.join(config.root_dir, "image", f"{img_id_lZero}.jpg")
                # store the image
                shutil.copy2(img_pth, os.path.join(out_f, img_id+".jpg"))
            cnt+=1
    print(f"# objects: {cnt}")

def main():
    # Group samples by object
    obj2img = group_by_object(config)

    # Save images
    save_imgs(obj2img, config)


if __name__ == '__main__':
    main()

