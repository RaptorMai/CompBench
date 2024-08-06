# CompBench: A Comparative Reasoning Benchmark for Multimodal LLMs


Official dataset and codes for <a href="https://www.arxiv.org/abs/2407.16837">CompBench</a>. 

## Benchmark Summary

CompBench is a benchmark designed to evaluate the comparative reasoning capability of multimodal large language models (MLLMs).
CompBench mines and pairs images through visually oriented questions covering eight dimensions of relative comparison: visual attribute, existence, state, emotion, temporality, spatiality, quantity, and quality. CompBench comprises around 40K image pairs collected from a broad array of visual domains, including animals, fashion, sports, and both outdoor and indoor scenes. The questions are carefully crafted to discern relative characteristics between two images and are labeled by human annotators for accuracy and relevance.

Check our <a href="https://compbench.github.io/">project page</a> and <a href="https://www.arxiv.org/abs/2407.16837">paper</a> for key contributions and findings.

## Release Process
- [x] Dataset
  - [x] Preparing images
  - [x] Preparing question-answer pairs
- [ ] Model evaluation

## Preparing images
Images in CompBench are collected from fourteen publicly available datasets.

### MIT-States

Download images from <a href="https://web.mit.edu/phillipi/Public/states_and_transformations/index.html">[here]</a>.

Store the images into <code>dataset/transformed_states</code>.


    dataset
    ├── transformed_states
    │ ├── release_dataset
    │ │ ├── images
    │ │ │ ├── deflated ball

Reference: <a href="https://openaccess.thecvf.com/content_cvpr_2015/papers/Isola_Discovering_States_and_2015_CVPR_paper.pdf">[link]</a>

### Fashionpedia
Download images from <a href="https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip">[here]</a>.

Download annotations from <a href="https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json">[here]</a>.

Store the images and the annotations into <code>dataset/fashionpedia</code>.

Run <code>preprocessing/fashionpedia/extract_adj_obj.py</code> to group images by the object (e.g., dress) with adjective (argyle). 

Output directory: <code>dataset/fashionpedia/adj_obj_folder_pairs_images</code>.

    dataset
    ├── fashionpedia
    │ ├── test
    │ ├── instances_attributes_val2020.json
    │ ├── adj_obj_folder_pairs_images
    │ │ ├── val
    │ │ │ ├── straight_pants-loose (fit)_pants

Reference: <a href="https://arxiv.org/pdf/2004.12276">[link]</a>

### VAW

Follow <a href="https://github.com/SHTUPLUS/PySGG/blob/main/DATASET.md#:~:text=Download%20the%20VG%20images%20part1%20part2">[here]</a> to download Visual Genome (VG) part1 and part2 images.

Store its part1 images into <code>dataset/vaw/images/VG_100K</code>.
Store its part2 images into <code>dataset/vaw/images/VG_100K_2</code>.

Download a Val annotation file (i.e., val.json) from <a href="https://github.com/adobe-research/vaw_dataset/tree/main/data">[here]</a>.

Store the annotation file into <code>dataset/vaw/data</code>.

Run <code>preprocessing/vaw/extract_imgs_vaw.py</code> to group images by the object (e.g., hair) with adjective (wet). 

Output directory: <code>dataset/vaw/att_obj_images</code>.

    dataset
    ├── vaw
    │ ├── images
    │ ├── data
    │ ├── att_obj_images
    │ │ ├── val
    │ │ │ ├── wet hair

Reference: <a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Pham_Learning_To_Predict_Visual_Attributes_in_the_Wild_CVPR_2021_paper.pdf">[link]</a>

### CUB-200-2011

Download images and annotations from <a href="https://data.caltech.edu/records/20098">[here]</a>.

Store them into <code>dataset/cub_200_2011</code>.

Run <code>preprocessing/cub_200_2011/extract_imgs.py</code> to group images by the bird species (e.g., Laysan_Albatross) with adjective (curved bill). 

Output directory: <code>dataset/cub_200_2011/att_cls_images</code>.

    dataset
    ├── cub_200_2011
    │ ├── images
    │ ├── parts
    │ ├── segmentations
    │ ├── attributes.txt
    │ ├── att_cls_images
    │ │ ├── test
    │ │ │ ├── 1_2

Reference: <a href="https://www.vision.caltech.edu/datasets/cub_200_2011/">[link]</a>

 ### Wildfish
 Please ask the author of  <a href="https://github.com/PeiqinZhuang/WildFish">Wildfish++</a> to download pairs of images (i.e., Pair_images) and its annotations (i.e., fine_grained).

 Store the annotation file into <code>dataset/wildfish</code>.

 Run <code>preprocessing/wildfish/extract_imgs_wf.py</code> to group images by two similar fish species (e.g., Amphiprion_akindynos-Amphiprion_chrysopterus).

 Output directory: <code>dataset/wildfish/diff_images</code>.

    dataset
    ├── wildfish
    │ ├── Pair_images
    │ ├── fine_grained
    │ ├── diff_images
    │ │ ├── val
    │ │ │ ├── Amphiprion_akindynos-Amphiprion_chrysopterus

  Reference: <a href="https://ieeexplore.ieee.org/document/9211789">[link]</a>

  ### MagicBrush
  Download Dev images from <a href="https://osu-nlp-group.github.io/MagicBrush/">[here]</a>.

  Store them into <code>dataset/magic_brush</code>.

    dataset
    ├── magic_brush
    │ ├── dev
    │ │ ├── images

  Reference: <a href="https://arxiv.org/pdf/2306.10012">[link]</a>

  ### Spot-the-diff
  Download scenes from <a href="https://drive.google.com/file/d/1OVb4_3Uec_xbyUk90aWC6LFpKsIOtR7v/view?usp=sharing">[here]</a>.

  Download the test annotation file (i.e., test.json) from <a href="https://github.com/harsh19/spot-the-diff/tree/master/data/annotations">[here]</a>.
  
  Store the scenes and the annotation into <code>dataset/spot-the-diff</code>.
  
  Run <code>preprocessing/spot-the-diff/extract_imgs_sd.py</code> to generate a pair of two similar scenes.

  Output directory: <code>dataset/spot-the-diff/pair_images</code>.

    dataset
    ├── spot-the-diff
    │ ├── resized_images
    │ ├── test.json
    │ ├── pair_images
    │ │ ├── test

  Reference: <a href="https://aclanthology.org/D18-1436.pdf">[link]</a>

  ### CelebA
  Download img_align_celeba.zip from <a href="https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ">[here]</a>.

  Unzip the file and store it into <code>dataset/celeba</code>.
  
  Download list_eval_partition.txt from <a href="https://drive.google.com/drive/folders/0B7EVK8r0v71pdjI3dmwtNm5jRkE?resourcekey=0-TD_RXHhlG6LPvwHReuw6IA">[here]</a>.
  
  Store it into <code>dataset/celeba</code>.

  Download list_attr_celeba.txt from <a href="https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ">[here]</a>.

  Store it into <code>dataset/celeba</code>.

  Run <code>preprocessing/celeba/extract_imgs.py</code> to group images by adjectives (e.g., smiling).

  Output directory: <code>dataset/celeba/adj_images</code>.

    dataset
    ├── celeba
    │ ├── img_align_celeba
    │ ├── list_eval_partition.txt
    │ ├── list_attr_celeba.txt
    │ ├── adj_images
    │ │ ├── test
    │ │ │ ├── Smiling

  Reference: <a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">[link]</a>

  ### FER-2013
  Download test images from <a href="https://www.kaggle.com/datasets/msambare/fer2013?resource=download">[here]</a>.

  Store them into <code>dataset/fer-2013</code>.

    dataset
    ├── fer-2013
    │ ├── test
    │ │ ├── angry

  Reference: <a href="https://arxiv.org/abs/1307.0414">[link]</a>

  ### SoccerNet
  Check <code>preprocessing/soccernet/download_soccernet.py</code> to download videos and their labels.
  Store them into <code>dataset/soccernet</code>.

  Run <code>preprocessing/soccernet/extract_temp_imgs.py</code> to extract a pair of frames from the action (e.g., corner-kick).
  
  Output directory: <code>\*/*_frames_actions</code>.
    
    dataset
    ├── soccernet
    │ ├── val
    │ │ ├── england_epl
    │ │ │ ├── 2014-2015
    │ │ │ │ ├── 2015-04-11 - 19-30 Burnley 0 - 1 Arsenal
    │ │ │ │ │ ├── 1_frames
    │ │ │ │ │ ├── 1_frames_actions
    │ │ │ │ │ ├── 2_frames
    │ │ │ │ │ ├── 2_frames_actions
    │ │ │ │ │ ├── Labels-v2.json
    │ │ │ │ │ ├── 1_224p.mkv
    │ │ │ │ │ ├── 2_224p.mkv

  Reference: <a href="https://www.soccer-net.org/home">[link]</a>

  ### CompCars
  Follow <a href="https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/">[here]</a> to download images.
  
  Run <code>preprocessing/comp_cars/extract_objects_comp_cars.py</code> to group vehicle images by the make, model, and released year.
  
  Output directory: <code>dataset/comp_cars/test_images</code>.
    
    dataset
    ├── comp_cars
    │ ├── data
    │ │ ├── image
    │ │ ├── label
    │ │ ├── misc
    │ │ ├── part
    │ │ ├── train_test_split
    │ ├── test_image
    │ │ ├── make_id
    │ │ │ ├── model_id
    │ │ │ │ ├── released_year
  
  Reference: <a href="https://arxiv.org/pdf/1506.08959">[link]</a>

  ### NYU-Depth V2
  Download Labeled dataset from <a href="https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html#raw_parts">[here]</a>.
  
  Follow <a href="https://github.com/JaminJeong/Convert_to_image_from_nyuv2">[here]</a> to convert from mat to image.
  
  Download list_test.txt from <a href="https://github.com/davidstutz/nyu-depth-v2-tools/tree/master">[here]</a>.


  Run <code>preprocessing/nyu_depth_v2/extract_objects_nyu.py</code> to group images by the object (e.g., air conditioner).

  Output directory: <code>dataset/nyu_depth_v2/obj_images</code>.

    dataset
    ├── nyu_depth_v2
    │ ├── image
    │ ├── list_test.txt
    │ ├── obj_images
    │ │ ├── test
    │ │ │ ├── air_conditioner

  Reference: <a href="https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html">[link]</a>

  ### VQAv2
  Download VQAv2 data from <a href="https://visualqa.org/download.html">[here]</a>

  Run <code>preprocessing/vqav2/extract_counting_imgs_vqav2.py</code> to group samples by the counting questions.

  Output directory: <code>dataset/vqav2/counting_images</code>.

    dataset
    ├── vqav2
    │ ├── images
    │ │ ├── train2014
    │ │ ├── val2014
    │ ├── v2_mscoco_train2014_annotations.json
    │ ├── v2_mscoco_val2014_annotations.json
    │ ├── v2_OpenEnded_mscoco_train2014_questions.json
    │ ├── v2_OpenEnded_mscoco_val2014_questions.json
    │ ├── counting_images
    │ │ ├── train2014
    │ │ ├── val2014

  Reference: <a href="https://arxiv.org/pdf/1612.00837">[link]</a>

  ### Q-Bench2
  Download Dev data from <a href="https://huggingface.co/zhangzicheng/q-bench2/tree/main">[here]</a>.

    dataset
    ├── q-bench2
    │ ├── q-bench2-a1-dev.jsonl
    │ ├── llvisionqa_compare_dev

  Reference: <a href="https://arxiv.org/pdf/2402.07116">[link]</a>

  ## Preparing question-answer pairs

  All annotated pairs are available under <a href="https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/kil_5_buckeyemail_osu_edu/ElJ8rIa0lJ9MlX-_OcSQdhEBM_cf-buvzu-Jgr8_ffys8Q">[here]</a>. Concretely,
  
  -  MIT-States: <code>st_label</code>
  -  Fashionpedia: <code>fashion_label</code>
  -  VAW: <code>vaw_label</code>
  -  CUB-200-2011: <code>cub_label</code>
  -  Wildfish: <code>Wildfish_label</code>
  -  MagicBrush: <code>mb_label</code>
  -  Spot-the-diff: <code>spot_difference_label</code>
  -  CelebA: <code>celebA_label</code>
  -  FER-2013: <code>fer2013_label</code>
  -  SoccerNet: <code>soccernet_label</code>
  -  CompCars: <code>car_label</code>
  -  NYU-Depth V2: <code>depth_label</code>
  -  VQAv2: <code>vqav2_label</code>
  -  Q-bench2: <code>qbench2_label</code>

Each dataset has one annotated JSON file, which contains a list of dictionaries. Each dictionary represents the annotation for a pair of images. 

**CompCars**, **CelebA**, **CUB-200-2011**, **NYU-Depth V2**, **Fashionpedia**, **FER-2013**, **MIT-States**, **VAW** and **VQAv2** have the following keys in the annotation:
  - `image_1`: First image
  - `image_2`: Second image
  - `question`: Question about relativity between the two images
  - `answer`: Correct answer. "Right" indicates that the second image is correct and "Left" means that the first image is correct

Note: All questions in **CompCars** are the same, so it does not have the `question` key. The question is "Based on these images, which car is newer in terms of its model year or release year?" 
The term "newer" is related to the year each car was manufactured or released, not its current condition or usage."

Note: **MIT-States** and **VAW** have an additional key `type`, which can be 'Size', 'Color', 'Pattern', 'Texture', 'Shape', or 'State'. 
The 'Size', 'Color', 'Pattern', 'Texture', and 'Shape' are common visual attributes. 

 **MagicBrush** and **Spot-the-diff** consist of multi-choice questions where models need to select one of the provided options.
  - `image_1`: First image
  - `image_2`: Second image
  - `options`: Options related to images
  - `answer`: Correct answer

Question for **MagicBrush** and **Spot-the-diff**: "What is the most obvious difference between two images? Choose from the following options. If there is no obvious difference, choose None. Options: None,`{pair['options']}`. Please only return one of the options without any other words."

**MagicBrush** has additional keys:
  -  `image_1_caption`: Caption for the first image
  -  `image_2_caption`: Caption for the second image
  -  `CLIP_similarity`: CLIP similarity between the two images

**Q-bench2** contains multi-choice questions that models need to select one of the provided options. The authors of Q-bench2 have already combined two images into one single image.
  - `image`: Combined image
  - `question`: Question about relativity between the two images
  - `options`: Options related to images
  - `answer`: Correct answer

**SoccerNet** has the following keys:
  - `image_1`: First frame
  - `image_2`: Second frame
  - `answer`: Correct answer. "Right" indicates that the second frame is correct and "Left" means that the first frame is correct
  - `CLIP_similarity`: CLIP similarity between the two frames
  - `action`: Soccer action related to the two frames
  - `match`: Soccer match related to the two frames
  - `league`: Soccer League related to the two frames

Question: "These are two frames related to `{pair['action']}` in a soccer match. Which frame happens first? Please only return one option from (Left, Right, None) without any other words. If these two frames are exactly the same, select None. Otherwise, choose Left if the first frame happens first and select Right if the second frame happens first."




## Citation
If you find the code and data useful, please cite the following paper:
```
@article{kil2024compbench,
  title={CompBench: A Comparative Reasoning Benchmark for Multimodal LLMs},
  author={Kil, Jihyung and Mai, Zheda and Lee, Justin and Wang, Zihe and Cheng, Kerrie and Wang, Lemeng and Liu, Ye and Chowdhury, Arpita and Chao, Wei-Lun},
  journal={arXiv preprint arXiv:2407.16837},
  year={2024}
}
```
