import json
import argparse
import os
from collections import defaultdict, Counter
import copy
import shutil
import sys
from tqdm import tqdm
import numpy as np

############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract two images from cub_200_2011')
parser.add_argument('--root_dir',type=str,default="dataset/cub_200_2011", help='directory of CUB_200_2011 dataset')
parser.add_argument('--split',type=str,default="test", help='split')
parser.add_argument('--mode',type=str,default="att_cls", help='att_cls: group images by attribute_class, diff: group images by having only one attribute different')

config = parser.parse_args()


def group_by_att_cls(config):

    # class list
    clsName2id = {}
    with open(os.path.join(config.root_dir, "classes.txt"))  as fp:
        for line in fp:
            cls_id, cls_name = line.split(" ")[0], line.split(" ")[1].strip()
            clsName2id[cls_name] = cls_id


    # selected comparable attributes
    selectAttNames = []
    with open(os.path.join(config.root_dir, "comp_attributes.txt"))  as fp:
        for line in fp:
            sel_att_id, sel_att_name = line.split(" ")[0], line.split(" ")[1].strip()
            selectAttNames.append(sel_att_name)

    # selected classes
    selectClss = []
    with open(os.path.join(config.root_dir, "comp_classes.txt"))  as fp:
        for line in fp:
            sel_cls_id, sel_cls_name = line.split(" ")[0], line.split(" ")[1].strip()
            selectClss.append(sel_cls_name)

    
    ann_file = json.load(open(os.path.join(config.root_dir, "imgs_grp_by_attrs.json")))
    for ele in tqdm(ann_file):
        att_name = ele["attributes"]["name"]
        att_id = ele["attributes"]["id"]
        
        if not att_name in selectAttNames:
            continue

        # group images by class label
        for _, img in ele["images"].items():
            img_cls, img_name = img["path"].split("/")[0], img["path"].split("/")[1].split(".jpg")[0]

            if not img_cls in selectClss:
                continue

            if not img_cls in clsName2id:
                print("ERROR!! class not found")
                sys.exit(1)

            out_f = os.path.join(config.root_dir, "att_cls_images", config.split, f"{att_id}_{clsName2id[img_cls]}")
            if not os.path.exists(out_f):
                os.makedirs(out_f)

                # store info
                info = {
                    "attribute": att_name,
                    "class": img_cls,
                    "images": [f"{img_name}.jpg"]
                }

                with open(os.path.join(out_f, "info.json"), "w") as fp:
                    json.dump(info, fp)
            else:
                # update info
                info = json.load(open(os.path.join(out_f, "info.json")))
                upd_imgs = copy.deepcopy(info["images"])
                upd_imgs.append(f"{img_name}.jpg")

                info.update({"images": upd_imgs})
                with open(os.path.join(out_f, "info.json"), "w") as fp:
                    json.dump(info, fp)


            # load image
            img_pth = os.path.join(config.root_dir, "images", img["path"])
            shutil.copy2(img_pth, os.path.join(out_f, f"{img_name}.jpg"))            

    cnt_folder = 0
    # filter out folders having only one image
    out_f = os.path.join(config.root_dir, "att_cls_images", config.split)
    for folder in os.scandir(out_f):
        #print("folder: ", folder.name)
        folder_pth = os.path.join(out_f, folder)

        n_imgs = 0
        for file in os.scandir(folder_pth):
            #print("file: ", file.name)
            if file.name.endswith(".jpg"):
                n_imgs+=1
                
        # remove folders having only one image
        if n_imgs <= 1:
            shutil.rmtree(folder_pth)
        else:
            cnt_folder+=1
    print("cnt_folder: ", cnt_folder) 

    
            
# find two images having one attribute different
def find_diff_att(config):

    # Get the split's image ids
    img_ids_split = []
    with open(os.path.join(config.root_dir, "train_test_split.txt"))  as fp:
        for line in fp:
            img_id, is_training = line.split(" ")[0], int(line.split(" ")[1])
            if config.split == "test":
                if not is_training:
                    img_ids_split.append(img_id)
            elif config.split == "train":
                if is_training:
                    img_ids_split.append(img_id)


    # attId2name
    attId2name = {}
    with open(os.path.join(config.root_dir, "attributes.txt"))  as fp:
        for line in fp:
            att_id, att_name = line.split(" ")[0], line.split(" ")[1]
            attId2name[att_id] = att_name
    

    # cls2imgIds = {}
    cls2imgIds = defaultdict(list)
    img_cls_lbl_f = os.path.join(config.root_dir, "image_class_labels.txt")
    with tqdm(total=os.path.getsize(img_cls_lbl_f)) as pbar:
        with open(img_cls_lbl_f)  as fp:
            for line in fp:
                pbar.update(len(line))
                img_id, cls_id = line.split(" ")[0], line.split(" ")[1]
                cls2imgIds[cls_id].append(img_id)

    # imgId2name
    imgId2name = {}
    with open(os.path.join(config.root_dir, "images.txt"))  as fp:
        for line in fp:
            img_id, img_name = line.split(" ")[0], line.split(" ")[1]
            imgId2name[img_id] = img_name

    # imgId2atts
    imgId2atts = {}
    img_att_lbl_f = os.path.join(config.root_dir, "attributes", "image_attribute_labels.txt")
    with tqdm(total=os.path.getsize(img_att_lbl_f)) as pbar:
        with open(img_att_lbl_f)  as fp:
            for line in fp:
                pbar.update(len(line))

                img_id, att_id, att_is_present, att_certain = line.split(" ")[0], int(line.split(" ")[1]), int(line.split(" ")[2]), int(line.split(" ")[3])

                # Attribute certaintiy level:
                ## 1 not visible
                ## 2 guessing
                ## 3 probably
                ## 4 definitely
                if img_id in img_ids_split:
                    if not img_id in imgId2atts:
                        # there are 312 attributes
                        imgId2atts[img_id] = [0] * 312

                    # If the attribute is annotated as present with certaintiy level >=3, consider that the attribute exists.
                    if att_is_present and att_certain >=3:
                        imgId2atts[img_id][att_id-1] = 1
                        
    for cls in cls2imgIds:
        cls_img_ids = cls2imgIds[cls]
        tgt_img_ids = list(set(cls_img_ids) & set(img_ids_split))

        # Get two birds of the same species
        for idx, img1 in enumerate(tgt_img_ids):
            for img2 in tgt_img_ids[idx + 1:]:
                print(f"img1_img2: {img1}_{img2}")
                img1_atts = np.array(imgId2atts[img1])
                

                img2_atts = np.array(imgId2atts[img2])


                diff_two_imgs = img1_atts == img2_atts

                # Indicies of different present attributes between two images
                # if two images have only one attrbiute different
                if np.sum(diff_two_imgs) == len(img1_atts)-1:
                    #print("img1_atts: ", img1_atts)
                    #print("img1id: ", img1)
                    #print(f"img1: {imgId2name[img1]}")

                    img1_name = imgId2name[img1]
                    
                    diff_att_idx = np.where(diff_two_imgs==0)[0][0]
                    diff_att_name = attId2name[str(diff_att_idx+1)]
                    #print(f"differnt attribute: {attId2name[str(diff_att_idx+1)]}")
                    
                    
                    #print("img2_atts: ", img2_atts)
                    #print("img2id: ", img2)
                    #print(f"img2: {imgId2name[img2]}")

                    img2_name = imgId2name[img2]

                    if img1_atts[diff_att_idx] == 1:
                        answer = imgId2name[img1]
                    elif img2_atts[diff_att_idx] == 1:
                        answer = imgId2name[img2]
                    else:
                        print("ERROR!!!")
                        sys.exit(1)
                    
                    save_imgs_find_diff(img1, img1_atts, img2, img2_atts, imgId2name, diff_att_idx, attId2name, answer, config)
                     
def save_imgs_find_diff(img1id, img1_atts, img2id, img2_atts, imgId2name, diff_att_idx, attId2name, answer, config):


    diff_att_name = attId2name[str(diff_att_idx+1)]

    img1_class, img1_name = imgId2name[img1id].split("/")[0], imgId2name[img1id].split("/")[1].strip().split(".jpg")[0]
    img2_class, img2_name = imgId2name[img2id].split("/")[0], imgId2name[img2id].split("/")[1].strip().split(".jpg")[0]

    # output directory 
    out_f = os.path.join(config.root_dir, "find_diff", config.split, f"{str(img1id)}_{str(img2id)}_{diff_att_idx}")
    if not os.path.exists(out_f):
        os.makedirs(out_f)

    # save images
    img1_pth = os.path.join(config.root_dir, "images", img1_class, f"{img1_name}.jpg")
    img2_pth = os.path.join(config.root_dir, "images", img2_class, f"{img2_name}.jpg")
    shutil.copy2(img1_pth, os.path.join(out_f, f"{img1_name}.jpg")) 
    shutil.copy2(img2_pth, os.path.join(out_f, f"{img2_name}.jpg"))

    print(f"img1_id: {str(img1id)}")
    print(f"img1_name: {img1_name}")
    print(f"img1_atts: {list(img1_atts)}")
    print(f"img2_id: {str(img2id)}")
    print(f"img2_name: {img2_name}")
    print(f"img2_atts: {list(img2_atts)}")

    info = {
        "diff_att_name": diff_att_name,
        "diff_att_idx:": str(diff_att_idx),
        "img_with_diff_att": answer,
        "img1_id": str(img1id),
        "img1_name": img1_name,
        "img1_atts": img1_atts.tolist(),
        "img2_id": str(img2id),
        "img2_name": img2_name,
        "img2_atts": img2_atts.tolist()
    }
    with open(os.path.join(out_f, "info.json"), "w") as fp:
        json.dump(info, fp)


def main():

    if config.mode=="diff":
        # Group images by having only one attribute different
        obj2img = find_diff_att(config)
    elif config.mode=="att_cls":
        # Group images by attribute_class
        obj2img = group_by_att_cls(config)


    

if __name__ == '__main__':
    main()