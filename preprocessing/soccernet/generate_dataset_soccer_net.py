import sys
import os
sys.path.insert(0, '../..')
import argparse
import json
from setup import compute_clip_similarity
import clip
import random
import time


FOLDER = ('1_frames_actions', '2_frames_actions')
ACTION = ["Indirect free-kick", "Throw-in", "Foul", "Shots off target", "Shots on target", "Goal", "Corner", "Direct free-kick", "Penalty"]
INTERVAL = 2
CLIP_THRESHOLD = 0.78

def generate_dataset():
    args = setup_parser().parse_args()
    
    data_folder = args.data_folder


    clip_L, clip_L_preprocess = clip.load("ViT-L/14", device='cuda')

    league = data_folder.split("/")[-1]
    data_split = data_folder.split("/")[-2]
    
    
    results = []
    for season in os.scandir(data_folder):
        if season.is_dir():
            print(f"Running {season.name}...")
            for match in os.scandir(season):
                if match.is_dir():
                    for folder in os.scandir(match):
                        if folder.name in FOLDER:
                            pair_list = list(os.scandir(folder))
                            for pair in pair_list:
                                if pair.name.split("_")[1] in ACTION:
                                    
                                    tmp = {}
                                    first = f'{pair.path}/{pair.name.split("_")[0]}.png'
                                    first_fname = first.split("/")
                                    first_fname = "/".join(first_fname[first_fname.index(league):])

                                    second_index = "{:0{width}d}".format(int(pair.name.split("_")[0]) + INTERVAL,
                                                                        width=len(pair.name.split("_")[0]))
                                    second = f'{pair.path}/{second_index}.png'
                                    second_fname = second.split("/")
                                    second_fname = "/".join(second_fname[second_fname.index(league):])


                                    if not os.path.exists(second):
                                        continue
                                    c_score = compute_clip_similarity(first, second, clip_L, clip_L_preprocess)
                                    if c_score < CLIP_THRESHOLD:
                                        continue

                                    rand_1 = random.randint(0, 1)

                                    if rand_1 == 0:
                                        tmp['left'] = first_fname
                                        tmp['right'] = second_fname
                                        tmp['answer'] = 'Left'
                                    else:
                                        tmp['left'] = second_fname
                                        tmp['right'] = first_fname
                                        tmp['answer'] = 'Right'
                                    tmp['CLIP'] = c_score
                                    tmp['time'] = pair.name.split("_")[0]
                                    tmp['action'] = pair.name.split("_")[1]

                                    tmp['match'] = match.name
                                    tmp['season'] = season.name
                                    tmp['league'] = league
                                    
                                    
                                    results.append(tmp)
                                    #print(tmp)

    out_dir =data_folder.split("/")[-2]                                   
    filename = f'../../Labeled_data/soccernet_label/{data_split}/{league}22.json'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fout:
        json.dump(results, fout)

def setup_parser():
    parser = argparse.ArgumentParser(description='SoccerNet.')
    parser.add_argument('--data_folder', type=str, default='england_epl')
    return parser

if __name__ == '__main__':
    generate_dataset()