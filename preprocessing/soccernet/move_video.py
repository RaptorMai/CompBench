import os
import argparse
import os
import ffmpeg
import json
import sys
import shutil
from tqdm import tqdm


# sourcepath='C:/Users/kevinconnell/Desktop/Test_Folder/'
# sourcefiles = os.listdir(sourcepath)
# destinationpath = 'C:/Users/kevinconnell/Desktop/Test_Folder/Archive'
# for file in sourcefiles:
#     if file.endswith('.png'):
#         shutil.move(os.path.join(sourcepath,file), os.path.join(destinationpath,file))


############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract two temporal images')
parser.add_argument('--root_dir',type=str,default="/local/scratch_2/jihyung/comp_imgs/dataset/soccernet/train", help='directory of video dataset')
parser.add_argument('--dataset',type=str,default="soccernet", help='name of dataset')

config = parser.parse_args()

def iterate_directory(root_dir):
    file_count = sum(len(files) for _, _, files in os.walk(root_dir))  # Get the number of files
    with tqdm(total=file_count) as pbar:
        for root, dirs, files in os.walk(root_dir):
            #print(root)
            for file in files:
                pbar.update(1)  # Increment the progress bar
                video2frame_soccernet(root, file)

def video2frame_soccernet(root, file):

    if file.endswith(".mkv"):

        out_dir = "/".join(root.split("train")[1:])
        out_dir = os.path.join("/local/scratch_2/jihyung/comp_imgs/dataset/soccernet/train_video" + out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        #print("source: ", os.path.join(root,file))
        #print("target: ", os.path.join(out_dir,file))
        shutil.move(os.path.join(root,file), os.path.join(out_dir,file))


def main():    
    iterate_directory(config.root_dir)
if __name__ == '__main__':
    main()