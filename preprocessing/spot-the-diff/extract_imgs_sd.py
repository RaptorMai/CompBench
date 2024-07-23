import json
import argparse
import os
from collections import defaultdict, Counter
import copy
import shutil
import sys

############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract pair of images from spot-the-diff')
parser.add_argument('--root_dir',type=str,default="dataset/spot-the-diff", help='directory of spot-the-diff dataset')
parser.add_argument('--split',type=str,default="test", help='name of dataset split')


config = parser.parse_args()



def save_imgs(config):
    

    # Annotations
    anns = json.load(open(os.path.join(config.root_dir, f"{config.split}.json")))


    # save images by object_attribute
    cnt=0
    for _, ann in enumerate(anns):

        
        img1_id = ann["img_id"]
        img2_id = img1_id + "_2"
        img_dif = img1_id + "_diff"
        sents = ann["sentences"]

        out_f = os.path.join(config.root_dir, "pair_images", config.split, str(img1_id))
        if not os.path.exists(out_f):
            os.makedirs(out_f)

        
        # Iterate samples in the object_attribute
        img1_pth = os.path.join(config.root_dir, "resized_images", f"{img1_id}.png")
        img2_pth = os.path.join(config.root_dir, "resized_images", f"{img2_id}.png")
        imgDif_pth = os.path.join(config.root_dir, "resized_images", f"{img_dif}.jpg")

        # store two images
        shutil.copy2(img1_pth, os.path.join(out_f, f"{img1_id}.png")) 
        shutil.copy2(img2_pth, os.path.join(out_f, f"{img2_id}.png")) 
        shutil.copy2(imgDif_pth, os.path.join(out_f, f"{img_dif}.jpg")) 
        
        with open(os.path.join(out_f, "sents.json"), "w") as fp:
            json.dump(sents, fp)
        cnt+=1


    print(f"# pairs: {cnt}")

def main():
    # Save two images into a folder
    save_imgs(config)


if __name__ == '__main__':
    main()

