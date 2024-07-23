import argparse
import json
import os
import sys
import copy
from tqdm import tqdm

from typing import List
from io import BytesIO
from PIL import Image
import requests


LEAGUE = ["england_epl", "europe_uefa-champions-league", "france_ligue-1", "germany_bundesliga", "italy_serie-a"]


def merge_left_right_imgs():
    args = setup_parser().parse_args()
    
    data_folder = args.data_folder
    output_dir = "/".join(data_folder.split("/")[:-1])
    print("output_dir: ", output_dir)
    
    results = []

    # save merged images
    out_merged_imgs = os.path.join(args.image_folder+"_mergedImgs")
    os.makedirs(out_merged_imgs, exist_ok=True)
    

    source_json = json.load(open(data_folder))

    results = []
    for idx, ele in enumerate(tqdm(source_json)):
        ele_cpy = copy.deepcopy(ele)
        left_img = ele_cpy["image_1"]
        left_img = "/".join(left_img.split("/")[9:])
        

        right_img = ele_cpy["image_2"]
        right_img = "/".join(right_img.split("/")[9:])


        two_img_links = [os.path.join(args.image_folder, left_img), os.path.join(args.image_folder, right_img)]


        merged_img = merge_images(two_img_links)
        
        left_imgid = left_img.split("/")[-1].split(".jpg")[0]

        
        right_imgid = right_img.split("/")[-1].split(".jpg")[0]
        
        # save merged images
        merged_img.save(os.path.join(out_merged_imgs, str(left_imgid) + "_" + str(right_imgid) + ".jpg"))

        que = ele_cpy["question"]
        answer = ele_cpy["answer"]
        ele_cpy["id"] = str(left_imgid) + "_" + str(right_imgid)
        ele_cpy["image"] = str(left_imgid) + "_" + str(right_imgid) + ".jpg"
        ele_cpy["conversations"] = [
            {
                "from": "human",
                "value": f"{que} If the second image has more, return Right. If the first image has more, return Left. If both images have the same number, return Same. Please only return either Left or Right or Same without any other words, spaces or punctuation."
            },
            {
                "from": "gpt",
                "value": answer
            }

        ]
        results.append(ele_cpy)
    
        
    filename = os.path.join(output_dir, f"vqa_train_pairs_mergedImgs.json")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fout:
        json.dump(results, fout)

def setup_parser():
    parser = argparse.ArgumentParser(description='VQAv2.')
    parser.add_argument('--data_folder', type=str, default='VL_fine-grained/Labeled_data/vqav2_label/vqa_train_pairs.json')
    parser.add_argument('--image_folder', type=str, default='comp_imgs/dataset/vqav2/counting_images/train2014')
    return parser



# Merge two images from Mantis.
# https://github.com/TIGER-AI-Lab/Mantis/blob/89d34077bd87b66eaadc13117add553e3a3d4c0b/mantis/mllm_tools/mllm_utils.py#L26

def load_image(image_file):
    if image_file.startswith("http"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        import os
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        if isinstance(image_file, Image.Image):
            image = image_file.convert("RGB")
        else:
            image = load_image(image_file)
        out.append(image)
    return out


def merge_images(image_links: List = []):
        """Merge multiple images into one image

        Args:
            image_links (List, optional): List of image links. Defaults to [].

        Returns:
            [type]: [description]
        """
        if len(image_links) == 0:
            return None
        images = load_images(image_links)
        if len(images) == 1:
            return images[0]
        widths, heights = zip(*(i.size for i in images))
        average_height = sum(heights) // len(heights)
        for i, im in enumerate(images):
            # scale in proportion
            images[i] = im.resize((int(im.size[0] * average_height / im.size[1]), average_height))
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new("RGB", (total_width + 10 * (len(images) - 1), max_height))
        x_offset = 0
        for i, im in enumerate(images):
            if i > 0:
                # past a column of 1 pixel starting from x_offset width being black, 8 pixels being white, and 1 pixel being black
                new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
                x_offset += 1
                new_im.paste(Image.new("RGB", (8, max_height), (255, 255, 255)), (x_offset, 0))
                x_offset += 8
                new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
                x_offset += 1
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        return new_im


if __name__ == '__main__':
    merge_left_right_imgs()