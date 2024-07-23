import numpy as np
import os
import sys
import argparse
import json
import shutil

from fashionpedia.fp import Fashionpedia

############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract adjective_object from Fashionpedia')
parser.add_argument('--root_dir',type=str,default="dataset/fashionpedia/", help='directory of Fashionpedia dataset')
parser.add_argument('--split',type=str,default="val", help='name of dataset split')



def group_by_adj_obj(anno_f_pth, comp_adjs, config, imgId2name):

    if config.split == "val":
        img_split = "test"
    else:
        img_split = config.split
    
    # initialize Fashionpedia api
    fp = Fashionpedia(anno_f_pth)


    # Images by category ids
    #cat_ids = fp.getCatIds(catNms=['pants','sleeve'])
    #img_ids = fp.getImgIds(catIds=cat_ids)
    #selected = img_ids[np.random.randint(0, len(img_ids))]

    # Images by category_attribute ids
    cats = fp.loadCats(fp.getCatIds())
    cat_names =[cat['name'] for cat in cats]
    
    cat2simpleCat = {
        "shirt, blouse": "blouse",
        "top, t-shirt, sweatshirt": "sweatshirt"
    }



    #cat_ids = fp.getCatIds(catNms=['pants','sleeve'])
    cat_ids = fp.getCatIds(catNms=cat_names)
    #att_ids = fp.getAttIds(attNms=['regular (collar)'])
    att_ids = fp.getAttIds(attNms=comp_adjs)


    from collections import Counter, defaultdict
    import collections
    import matplotlib.pyplot as plt

    cnt_dict = {}
    folder_cnt = 0

    cat_compAdj_imgs = defaultdict(dict)
    

    # cat_compAdj_imgs: 
    # {
    #   {'sweater': {'symmetrical': [15676, ...], 
    #                'curved (fit)': [13077], ...
    #   }, ...
    # }
    for cat_name, cat_id in zip(cat_names,cat_ids):
        if cat_name in cat2simpleCat:
            cat_name = cat2simpleCat[cat_name]
        
        for comp_adj, att_id in zip(comp_adjs, att_ids):
            
            cat_att_name = f"{comp_adj}_{cat_name}"
            cat_att_id = f"{cat_id}_{att_id}"
            img_ids = fp.getImgIds(catAttId=cat_att_id)
            img_names = []
            for img_id in img_ids:
                img_name = imgId2name[img_id]
                img_names.append(img_name)

            if len(img_ids) > 0:
                #cat_dict[cat_name] +=1
                #cnt_dict[cat_att_name] = len(img_ids)

                #save_imgs(cat_att_name, img_names, config)

                cat_compAdj_imgs[cat_name][comp_adj] = img_names

                folder_cnt+=1

    # Generate folder pairs
    cnt_pairs = 0
    cnt_cat = 0
    for cat_name in cat_compAdj_imgs:
        comp_adjs_per_cat = list(cat_compAdj_imgs[cat_name].keys())
        #print("comp_adjs_per_cat: ", comp_adjs_per_cat)

        if len(comp_adjs_per_cat) > 1: 
            cnt_cat+=1
            for idx, capc in enumerate(comp_adjs_per_cat):
                for capc_oth in comp_adjs_per_cat[idx + 1:]:
                    #print(f"{capc}_{cat_name}, {capc_oth}_{cat_name}")
                    cnt_pairs+=1

                    # output directory 
                    out_f = os.path.join(config.root_dir, "adj_obj_folder_pairs_images", config.split, f"{capc}_{cat_name}-{capc_oth}_{cat_name}")
                    if not os.path.exists(out_f):
                        os.makedirs(out_f)
                    
                    # store first folder of pair
                    out_fir_fol = os.path.join(out_f, f"{capc}_{cat_name}")
                    if not os.path.exists(out_fir_fol):
                        os.makedirs(out_fir_fol)
                    # store images of the first folder
                    fir_fol_imgs = cat_compAdj_imgs[cat_name][capc]
                    for fir_fol_img in fir_fol_imgs:
                        img_pth = os.path.join(config.root_dir, img_split, f"{fir_fol_img}.jpg")
                        shutil.copy2(img_pth, os.path.join(out_fir_fol, fir_fol_img+".jpg"))

                    
                     # store second folder of pair
                    out_sec_fol = os.path.join(out_f, f"{capc_oth}_{cat_name}")
                    if not os.path.exists(out_sec_fol):
                        os.makedirs(out_sec_fol)
                    # store images of the second folder
                    sec_fol_imgs = cat_compAdj_imgs[cat_name][capc_oth]
                    for sec_fol_img in sec_fol_imgs:
                        img_pth = os.path.join(config.root_dir, img_split, f"{sec_fol_img}.jpg")
                        shutil.copy2(img_pth, os.path.join(out_sec_fol, sec_fol_img+".jpg"))

    print("cnt_pairs: ", cnt_pairs)
    print("cnt_cat: ", cnt_cat)
            
   


    
    # cnt_dict = Counter(cnt_dict)
    # cnt_dict = dict(sorted(cnt_dict.items(), key=lambda item: item[1], reverse=True))
    # plt.bar(cnt_dict.keys(), cnt_dict.values())
    # plt.savefig("n_imgs_per_adj_obj.png")


    # cat_dict = Counter(cat_dict)
    # cat_dict = dict(sorted(cat_dict.items(), key=lambda item: item[1], reverse=True))
    # plt.bar(cat_dict.keys(), cat_dict.values())
    # print(cat_dict)
    # plt.savefig("n_atts_per_obj.png")

    # print(f"# adj_obj: {folder_cnt}")
            

def save_imgs(cat_att_name, img_names, config):
    
    
    # output directory 
    out_f = os.path.join(config.root_dir, "adj_obj_images", config.split, cat_att_name)
    if not os.path.exists(out_f):
        os.makedirs(out_f)

    # Iterate image
    for img_name in img_names:
        # Get the image path
        if config.split == "val":
            split = "test"
        else:
            split = config.split
        img_pth = os.path.join(config.root_dir, split, f"{img_name}.jpg")
        # store the image
        shutil.copy2(img_pth, os.path.join(out_f, img_name+".jpg"))


def main():
    config = parser.parse_args()

    anno_f_pth = os.path.join(config.root_dir, f"instances_attributes_{config.split}2020.json")
    ann = json.load(open(anno_f_pth))
    
    imgId2name = {}
    for _, ele in enumerate(ann["images"]):
        img_id = ele["id"]
        img_name = ele["file_name"].split(".jpg")[0]
        imgId2name[img_id] = img_name


    # load comparative adjectives (if already have)
    #comp_adjs = json.load(open(os.path.join(config.root_dir, f"comp_adjs_{config.split}.json")))

    #Define comparative adjectives and store them
    # 37 comparative adjectives
    comp_adjs = ["asymmetrical", "symmetrical", "straight", "baggy", "oversized",
                 "curved (fit)", "tight (fit)", "regular (fit)", "loose (fit)", "micro (length)", 
                 "mini (length)", "maxi (length)", "short (length)", "asymmetric (collar)", "regular (collar)", 
                 "oversized (collar)", "oversized (lapel)", "asymmetric (neckline)", "round (neck)", "oval (neck)",
                 "square (neckline)", "plunging (neckline)", "curved (pocket)", "washed", "distressed", 
                 "embossed", "frayed", "ruched", "quilted", "tiered", 
                 "slit", "perforated", "plain (pattern)", "floral",
                 "geometric", "paisley", "argyle"
    ]
    with open(os.path.join(config.root_dir, f"comp_adjs_{config.split}.json"), "w") as fp:
        json.dump(comp_adjs, fp)
    

    # Group images by adjective_object
    group_by_adj_obj(anno_f_pth, comp_adjs, config, imgId2name)


if __name__ == '__main__':
    main()