import json
import argparse
import os
from collections import defaultdict, Counter
import copy
import shutil
import sys

############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract images from VAW')
parser.add_argument('--root_dir',type=str,default="dataset/vaw", help='directory of vaw dataset')
parser.add_argument('--split',type=str,default="val", help='name of dataset split')


config = parser.parse_args()


def group_by_attr_object(config):
    # Images
    img_vg1 = []
    # part#1 of visual genome
    for entry in os.scandir(os.path.join(config.root_dir, "images", "VG_100K")):
        img_vg1.append(entry.name.split(".jpg")[0])

    img_vg2 = []
    # part#2 of visual genome
    for entry in os.scandir(os.path.join(config.root_dir, "images", "VG_100K_2")):
        img_vg2.append(entry.name.split(".jpg")[0])


    # Annotations
    anns = json.load(open(os.path.join(config.root_dir, "data", f"{config.split}.json")))

    # # Group samples by the object name
    # obj2smps = defaultdict(list)
    # for ann in anns:
    #     image_id = ann["image_id"]
    #     obj = ann["object_name"]
    #     atts = ann["positive_attributes"]
    #     smp = {
    #         "image_id": image_id,
    #         "object_name": obj,
    #         "positive_attributes": atts
    #     }
    #     obj2smps[obj].append(smp)
    # print(f"# objects: {len(obj2smps)}")


    # # Group samples by the attribute
    # att2smps = defaultdict(list)
    # for ann in anns:
    #     image_id = ann["image_id"]
    #     obj = ann["object_name"]
    #     atts = ann["positive_attributes"]
    #     for att in atts:
    #         smp = {
    #             "image_id": image_id,
    #             "object_name": obj,
    #         }
    #         att2smps[att].append(smp)
    # print(f"# attributes: {len(att2smps)}")


    # Group samples by attribute_object
    attObj2smps = defaultdict(list)
    for ann in anns:
        image_id = ann["image_id"]
        obj = ann["object_name"]
        atts = ann["positive_attributes"]
        for att in atts:
            att_obj = att + "_" + obj
            attObj2smps[att_obj].append(image_id)

    #return obj2smps, att2smps, img_vg1, img_vg2
    return {}, {}, attObj2smps, img_vg1, img_vg2
    

def save_imgs(obj2smps, att2smps, attObj2smps, img_vg1, img_vg2, config):
    
    # # save images by object
    # for _, obj in enumerate(obj2smps):
    #     # output directory 
    #     out_f = os.path.join(config.root_dir, "obj_images", config.split, obj)
    #     if not os.path.exists(out_f):
    #         os.makedirs(out_f)
        
    #     # Iterate samples in the object
    #     for smp in (obj2smps[obj]):
    #         img_id = str(smp["image_id"])
            
    #         # Get the image path
    #         if img_id in img_vg1:
    #             img_pth = os.path.join(config.root_dir, "images", "VG_100K", f"{img_id}.jpg")
    #         elif img_id in img_vg2:
    #             img_pth = os.path.join(config.root_dir, "images", "VG_100K_2", f"{img_id}.jpg")
    #         # store the image
    #         shutil.copy2(img_pth, os.path.join(out_f, img_id+".jpg"))
        
    #     # store the object json file
    #     info = obj2smps[obj]
    #     with open(os.path.join(out_f, "info.json"), "w") as fp:
    #         json.dump(info, fp)

    # # save images by attributes
    # for _, att in enumerate(att2smps):
    #     # output directory 
    #     out_f = os.path.join(config.root_dir, "att_images", config.split, att)
    #     if not os.path.exists(out_f):
    #         os.makedirs(out_f)
        
    #     # Iterate samples in the attrbitues
    #     for smp in (att2smps[att]):
    #         img_id = str(smp["image_id"])
            
    #         # Get the image path
    #         if img_id in img_vg1:
    #             img_pth = os.path.join(config.root_dir, "images", "VG_100K", f"{img_id}.jpg")
    #         elif img_id in img_vg2:
    #             img_pth = os.path.join(config.root_dir, "images", "VG_100K_2", f"{img_id}.jpg")
    #         # store the image
    #         shutil.copy2(img_pth, os.path.join(out_f, img_id+".jpg"))
        
    #     # store the attribute json file
    #     info = att2smps[att]
    #     with open(os.path.join(out_f, "info.json"), "w") as fp:
    #         json.dump(info, fp)

    # save images by object_attribute
    cnt=0
    for _, att_obj in enumerate(attObj2smps):
        # remove duplicate image_ids
        image_lst = set(attObj2smps[att_obj])

        # only consider a case having >= 2 images
        if len(image_lst) >= 2:
            # output directory 
            out_f = os.path.join(config.root_dir, "att_obj_images", config.split, att_obj)
            if not os.path.exists(out_f):
                os.makedirs(out_f)
            
            # Iterate samples in the object_attribute
            for img_id in image_lst:
                # Get the image path
                if img_id in img_vg1:
                    img_pth = os.path.join(config.root_dir, "images", "VG_100K", f"{img_id}.jpg")
                elif img_id in img_vg2:
                    img_pth = os.path.join(config.root_dir, "images", "VG_100K_2", f"{img_id}.jpg")
                # store the image
                shutil.copy2(img_pth, os.path.join(out_f, img_id+".jpg")) 
            cnt+=1
    print(f"# attribute_object: {cnt}")

def main():
    # Group samples by object, attribute, or object_attribute
    obj2smps, att2smps, attObj2smps, img_vg1, img_vg2 = group_by_attr_object(config)

    # Save images
    save_imgs(obj2smps, att2smps, attObj2smps, img_vg1, img_vg2, config)


if __name__ == '__main__':
    main()

