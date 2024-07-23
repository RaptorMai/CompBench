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
    
    results = []
    league_idx = 0

    # save merged images
    out_merged_imgs = os.path.join(args.image_folder+"_mergedImgs")
    os.makedirs(out_merged_imgs, exist_ok=True)
    

    for league in os.scandir(data_folder):
        league_name = league.name.split(".json")[0]

        if league_name in LEAGUE:
            source_json = json.load(open(os.path.join(data_folder, league.name)))

            for idx, ele in enumerate(tqdm(source_json)):
                ele_cpy = copy.deepcopy(ele)
                left_img = ele_cpy["left"]
                right_img = ele_cpy["right"]

                two_img_links = [os.path.join(args.image_folder, left_img), os.path.join(args.image_folder, right_img)]


                merged_img = merge_images(two_img_links)
                
                left_is = left_img.split("/")
                left_lea, left_sea, left_mat, left_fra, left_act, left_imgid = left_is[0], left_is[1], left_is[2], left_is[3], left_is[4], left_is[5]
                left_imgid = left_imgid.split(".png")[0]

                right_is = right_img.split("/")
                right_lea, right_sea, right_mat, right_fra, right_act, right_imgid = right_is[0], right_is[1], right_is[2], right_is[3], right_is[4], right_is[5]
                right_imgid = right_imgid.split(".png")[0]
               
                # save merged images
                merged_img.save(os.path.join(out_merged_imgs, str(league_idx) + "_" + str(idx) + ".png"))

                action = ele_cpy["action"]
                answer = ele_cpy["answer"]
                ele_cpy["id"] = str(league_idx) + "_" + str(idx)
                ele_cpy["image"] = str(league_idx) + "_" + str(idx) + ".png"
                ele_cpy["conversations"] = [
                    {
                        "from": "human",
                        "value": f"<image>\nThese are two frames related to {action} in a soccer match. Which frame happens first? Please only return one option from (Left, Right, None) without any other words. If these two frames are exactly the same, select None. Otherwise, choose Left if the first frame happens first and select Right if the second frame happens first."
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }

                ]
                results.append(ele_cpy)
            
            league_idx +=1
        
    filename = os.path.join(data_folder, f"all_leagues_mergedImgs.json")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fout:
        json.dump(results, fout)

def setup_parser():
    parser = argparse.ArgumentParser(description='SoccerNet.')
    parser.add_argument('--data_folder', type=str, default='soccernet/train')
    parser.add_argument('--image_folder', type=str, default='soccernet/train')
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