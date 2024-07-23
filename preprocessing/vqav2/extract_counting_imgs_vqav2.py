import json
import argparse
import os
from collections import defaultdict, Counter
import copy
import shutil
import sys

############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract images from VQAv2')
parser.add_argument('--root_dir',type=str,default="dataset/vqav2", help='directory of vqav2 dataset')
parser.add_argument('--split',type=str,default="val2014", help='name of dataset split')

config = parser.parse_args()


def extract_counting_images(config):
    # Get annotation file
    ann_fname = "v2_mscoco_{}_annotations".format(config.split)
    ann_f = json.load(open(os.path.join(config.root_dir, ann_fname + ".json")))["annotations"]

    # Get question file
    que_fname = "v2_OpenEnded_mscoco_{}_questions".format(config.split)
    que_f = json.load(open(os.path.join(config.root_dir, que_fname + ".json")))["questions"]
    qid2ele = {}
    for ele in que_f:
        qid = ele["question_id"]
        qid2ele[qid] = ele

    # Extract question types
    qtype_lst = []
    for ele in ann_f:
        qtype = ele["question_type"]
        if qtype not in qtype_lst:
            qtype_lst.append(qtype)

    # Extract the counting question types
    cnting_qtype_lst = []
    for qtype in qtype_lst:
        if "how many" in qtype:
            cnting_qtype_lst.append(qtype)
    print("The counting questions: ", cnting_qtype_lst)

    # Group samples by the counting questions
    cnting_smps = defaultdict(list)
    for ele in ann_f:
        qtype = ele["question_type"]
        if "how many" in qtype:
            qid = ele["question_id"]
            que = qid2ele[qid]["question"]
            ele_copy = copy.deepcopy(ele)
            ele_copy["question"] = que

            cnting_smps[que].append(ele_copy)

    # Count an instance per the counting question
    cnting_smps_cnt = {}
    for key in cnting_smps:
        val = cnting_smps[key]
        cnting_smps_cnt[key] = len(val)
    cnting_smps_cnt = dict(sorted(cnting_smps_cnt.items(), key=lambda x:x[1], reverse=True))

    # Exclude the counting question only having one sample
    for key in list(cnting_smps):
        if cnting_smps_cnt[key] == 1:
            del cnting_smps[key]
    
    return cnting_smps

def save_imgs(all_cnt_imgs, config):
    
    for i, que in enumerate(all_cnt_imgs):

        # directory that the question's samples will be stored 
        out_f = os.path.join(config.root_dir, "counting_images", config.split, str(i))
        if not os.path.exists(out_f):
            os.makedirs(out_f)
        
        info = {
            "question": que,
            "imgId2ans": []
        }

        # Iterate samples in the question
        for smp in (all_cnt_imgs[que]):
            img_id = str(smp["image_id"])
            img_id_lZero = "%012d" % int(img_id)
            ans = smp["multiple_choice_answer"]
            
            # Image_Id: GT Answer
            info["imgId2ans"].append({img_id:ans})
            
            img_pth = os.path.join(config.root_dir, "images", config.split, f"COCO_{config.split}_{img_id_lZero}.jpg")
            
            # store the image
            shutil.copy2(img_pth, os.path.join(out_f, img_id+".jpg"))
        
        # store the info file
        with open(os.path.join(out_f, "info.json"), "w") as fp:
            json.dump(info, fp)

def main():
    # Extract all images related to the counting questions
    all_cnt_imgs = extract_counting_images(config)

    # Save images
    save_imgs(all_cnt_imgs, config)


if __name__ == '__main__':
    main()
