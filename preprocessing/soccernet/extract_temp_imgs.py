import os
import argparse
import os
import ffmpeg
import json
import sys
import shutil
from tqdm import tqdm

############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract two temporal images')
parser.add_argument('--root_dir',type=str,default="dataset/soccernet", help='directory of video dataset')
parser.add_argument('--dataset',type=str,default="soccernet", help='name of dataset')
parser.add_argument('--action_duration',type=int,default=6, help='Duration of an action')

config = parser.parse_args()



def extract_actions_soccernet(root_dir, dataset):
    actions = []
    ignore_actions = ["Kick-off", "Ball out of play"]
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if dataset == "soccernet":
                if file.endswith(".json"):
                    print(os.path.join(root, file))
                    anns = json.load(open(os.path.join(root, "Labels-v2.json")))
                    anns = anns["annotations"]
                    # Extract all actions in the annotation file
                    for ann in anns:
                        if ann["visibility"] == "visible":
                            label = ann["label"]
                            if label not in actions and label not in ignore_actions:
                                actions.append(label)

    with open(os.path.join(root_dir, "actions.json"), "w") as fp:
        json.dump(actions, fp)

def iterate_directory(root_dir, dataset):
    file_count = sum(len(files) for _, _, files in os.walk(root_dir))  # Get the number of files
    with tqdm(total=file_count) as pbar:
        for root, dirs, files in os.walk(root_dir):
            print(root)
            for file in files:
                pbar.update(1)  # Increment the progress bar
                if dataset == "soccernet":
                    # load all actions in the soccernet
                    actions = json.load(open(os.path.join(root_dir, "actions.json")))
                    video2frame_soccernet(root, file, actions)


def video2frame_soccernet(root, file, actions):

    if file.endswith(".mkv"):
        # 1st or 2nd half
        half_id = file.split("_224p.mkv")[0]

        anns = json.load(open(os.path.join(root, "Labels-v2.json")))
        anns = anns["annotations"]

        # Extract all actions and their executed time in the annotation file
        time_actions = {}
        for ann in anns:
            label = ann["label"]
            if label in actions and ann["visibility"] == "visible":
                min_sec_t = ann["gameTime"].split("-")[1].strip()
                # action start time in seconds
                sec_t = (60 * int(min_sec_t.split(":")[0])) + int(min_sec_t.split(":")[1])        
                # action start time in seconds with leading zeros
                sec_t = "%06d" % sec_t
                half_id_ann = ann["gameTime"].split("-")[0].strip()
                
                # only conisder annotation for the current half video
                if half_id_ann == half_id:
                    # store action and its time
                    time_actions[sec_t] = label
       

        video = os.path.join(root, file)
        out_fn = os.path.join(root, half_id + "_frames")
        if not os.path.exists(out_fn):
            os.makedirs(out_fn)
        # Extract frames and store them
        ffmpeg.input(video).filter('fps', fps=1).output(os.path.join(out_fn, "%06d.png"), start_number=0).overwrite_output().run(quiet=True)

        # Extract action frames from all frames
        out_fn_actions = out_fn + "_actions"
        for f_subdir, f_dirs, f_files in os.walk(out_fn):
            for f_file in f_files:
                
                # skip first frame
                if f_file == "000000.png":
                    continue

                # remove .png
                f_file = f_file.split(".png")[0]
                
                if f_file in time_actions:
                    # directory that actions are stored 
                    out_fn_act = os.path.join(out_fn_actions, str(f_file) + "_" + time_actions[f_file])
                    if not os.path.exists(out_fn_act):
                        os.makedirs(out_fn_act)

                    # remove leading zeros
                    f_file = int(f_file.lstrip("0"))
                    for i in range(config.action_duration):
                        act_fr = f_file + i
                        act_fr = "%06d.png" % act_fr
                        if os.path.exists(os.path.join(out_fn,act_fr)):
                            shutil.copy2(os.path.join(out_fn,act_fr), out_fn_act)
                     

def main():

    # extract all actions in soccernet
    extract_actions_soccernet(config.root_dir, config.dataset)

    # Get frames per video
    iterate_directory(config.root_dir, config.dataset)

    

if __name__ == '__main__':
    main()