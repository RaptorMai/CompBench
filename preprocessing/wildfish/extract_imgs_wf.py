import json
import argparse
import os
from collections import defaultdict, Counter
import copy
import shutil
import sys

############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract images from wildfish++')
parser.add_argument('--root_dir',type=str,default="dataset/wildfish", help='directory of wildfish dataset')
parser.add_argument('--split',type=str,default="val", help='name of dataset split')


config = parser.parse_args()

def save_imgs(config):
    

    # description of difference between two classes
    cls_diff_info_lst = []
    with open(os.path.join(config.root_dir, "fine_grained", "pairwise_info.txt"))  as fp:
        for line in fp:
            line = line.strip()
            cls1, cls2, cls_diff_txt = line.split(" ")[0], line.split(" ")[2], " ".join(line.split("\t")[1:])
            info_smp = {
                "cls1": cls1,
                "cls2": cls2,
                "cls_diff": cls_diff_txt
            }
            cls_diff_info_lst.append(info_smp)
    
    # save images by object_attribute
    cnt=0
    for _, cdi in enumerate(cls_diff_info_lst):

        cls1_id = cdi["cls1"]
        cls2_id = cdi["cls2"]
        cls_diff = cdi["cls_diff"]

        # Get images of two classes
        imgs_split = defaultdict(list)
        with open(os.path.join(config.root_dir, "fine_grained", f"{config.split}.txt"))  as fp:
            for line in fp:
                line = line.strip()
                img_name, img_id = line.split(" ")[0].split(".jpg")[0], line.split(" ")[1]
                cls_name = "_".join(img_name.split("_")[:2])
                if cls1_id == str(img_id): 
                    imgs_split[cls_name].append(img_name)
                elif cls2_id == str(img_id):
                    imgs_split[cls_name].append(img_name)
        
        
        # Store images of two classes
        cls_names = list(imgs_split.keys())
        out_f = os.path.join(config.root_dir, "diff_images", config.split, f"{cls_names[0]}-{cls_names[1]}")
        if not os.path.exists(out_f):
            os.makedirs(out_f)

        for cls_name, imgs_name in imgs_split.items():

            out_cls_f = os.path.join(out_f, cls_name)
            if not os.path.exists(out_cls_f):
                os.makedirs(out_cls_f)
            for img_name in imgs_name:
                img_pth = os.path.join(config.root_dir, f"Pair_images", f"{img_name}.jpg")

                 # store two images
                shutil.copy2(img_pth, os.path.join(out_cls_f, f"{img_name}.jpg"))

        info = {
            "cls1": cls_names[0],
            "cls2": cls_names[1],
            "cls_diff": cls_diff
        }


        
        with open(os.path.join(out_f, "info.json"), "w") as fp:
            json.dump(info, fp)

        cnt+=1


    print(f"# pairs: {cnt}")

def main():

   
    # Save two images into a folder
    save_imgs(config)


if __name__ == '__main__':
    main()

