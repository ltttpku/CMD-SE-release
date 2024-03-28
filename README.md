# CMD-SE


This repository contains the official PyTorch implementation for the paper: 

> Ting Lei, Shaofeng Yin, Yang Liu; Towards Open-Vocabulary HOI Detection via Conditional Multi-level Decoding and Fine-grained Semantic Enhancement; In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024  


## Overview

> Open-vocabulary human-object interaction (HOI) detection, which is concerned with the problem of detecting novel HOIs guided by natural language, is crucial for understanding human-centric scenes. 
However, prior zero-shot HOI detectors often employ the same levels of feature maps to model HOIs with varying distances, leading to suboptimal performance in scenes containing human-object pairs with a wide range of distances.
In addition, these detectors primarily rely on category names and overlook the rich contextual information that language can provide, which is essential for capturing open vocabulary concepts that are typically rare and not well-represented by category names alone.
In this paper, we introduce a novel end-to-end open vocabulary HOI detection framework with conditional multi-level decoding and fine-grained semantic enhancement~(CMD-SE), harnessing the potential of Visual-Language Models (VLMs). Specifically, we propose to model human-object pairs with different distances with different levels of feature maps by incorporating a soft constraint during the bipartite matching process. 
Furthermore, by leveraging large language models (LLMs) such as GPT models, we exploit their extensive world knowledge to generate descriptions of human body part states for various interactions. Then we integrate the generalizable and fine-grained semantics of human body parts to improve interaction recognition.
Experimental results on two datasets, SWIG-HOI and HICO-DET, demonstrate that our proposed method achieves state-of-the-art results in open vocabulary HOI detection.

## Preparation

### Installation

Our code is built upon [CLIP](https://github.com/openai/CLIP). This repo requires to install [PyTorch](https://pytorch.org/get-started/locally/) and torchvision, as well as small additional dependencies.

```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install ftfy regex tqdm numpy Pillow matplotlib
```

### Dataset

The experiments are mainly conducted on **HICO-DET** and **SWIG-HOI** dataset. We follow [this repo](https://github.com/YueLiao/PPDM) to prepare the HICO-DET dataset. And we follow [this repo](https://github.com/scwangdyd/large_vocabulary_hoi_detection) to prepare the SWIG-HOI dataset.

#### HICO-DET

HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory. We use the annotation files provided by the [PPDM](https://github.com/YueLiao/PPDM) authors. We re-organize the annotation files with additional meta info, e.g., image width and height. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1lqmevkw8fjDuTqsOOgzg07Kf6lXhK2rg). The downloaded files have to be placed as follows. Otherwise, please replace the default path to your custom locations in [datasets/hico.py](./datasets/hico.py).

``` plain
 |─ data
 │   └─ hico_20160224_det
 |       |- images
 |       |   |─ test2015
 |       |   |─ train2015
 |       |─ annotations
 |       |   |─ trainval_hico_ann.json
 |       |   |─ test_hico_ann.json
 :       :
```

#### SWIG-DET

SWIG-DET dataset can be downloaded [here](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip). After finishing downloading, unpack the `images_512.zip` to the `data` directory. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1GxNP99J0KP6Pwfekij_M1Z0moHziX8QN). The downloaded files to be placed as follows. Otherwise, please replace the default path to your custom locations in [datasets/swig.py](./datasets/swig.py).

``` plain
 |─ data
 │   └─ swig_hoi
 |       |- images_512
 |       |─ annotations
 |       |   |─ swig_train_1000.json
 |       |   |- swig_val_1000.json
 |       |   |─ swig_trainval_1000.json
 |       |   |- swig_test_1000.json
 :       :
```

## Training

Run this command to train the model in HICO-DET dataset

``` bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port 3990 --use_env main.py \
    --batch_size 64 \
    --output_dir [path to save checkpoint] \
    --epochs 80 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 25 \
    --enable_dec \
    --dataset_file hico --multi_scale true --f_idxs 5 8 11 --set_cost_hoi_type 5 --use_aux_text true \
    --enable_focal_loss --description_file_path hico_hoi_descriptions.json
```

Run this command to train the model in SWIG-HOI dataset

``` bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port 4990 --use_env main.py \
    --batch_size 64 \
    --output_dir [path to save checkpoint] \
    --epochs 70 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 10 \
    --enable_dec \
    --dataset_file swig --multi_scale true --f_idxs 5 8 11 --set_cost_hoi_type 5 --use_aux_text true \
    --enable_focal_loss --description_file_path swig_hoi_descriptions_6bodyparts.json
```


## Inference

Run this command to evaluate the model on HICO-DET dataset

``` bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 3990 --use_env main.py \
    --batch_size 64 \
    --output_dir [path to save checkpoint] \
    --epochs 80 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 25 \
    --enable_dec \
    --dataset_file hico --multi_scale true --f_idxs 5 8 11 --set_cost_hoi_type 5 --use_aux_text true \
    --enable_focal_loss --description_file_path hico_hoi_descriptions.json \
    --eval --pretrained [path to ckpt]
```

Run this command to evaluate the model on SWIG-HOI dataset

``` bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 4990 --use_env main.py \
    --batch_size 64 \
    --output_dir [path to save results] \
    --epochs 70 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 10 \
    --enable_dec \
    --dataset_file swig --multi_scale true --f_idxs 5 8 11 --set_cost_hoi_type 5 --use_aux_text true \
    --enable_focal_loss --description_file_path swig_hoi_descriptions_6bodyparts.json \
    --eval --pretrained [path to ckpt]
```


## Models

| dataset | HOI Tokens | Unseen | Seen | Full | Checkpoint |
| :-----: | :-----: | :-----: | :-----: | -----: | :-----: |
| HICO-DET | 25 | 16.70 | 23.95 | 22.35 | [params](https://disk.pku.edu.cn/link/AACBAE087FC3C7479EBD955966748D6F3F)|


| dataset | HOI Tokens | Non-rare | Rare | Unseen | Full |  Checkpoint |
| :-----: | :-----: | :-----: | :-----: |:-----: | :-----: | :-----: |
| SWIG-HOI | 10 | 21.46 | 14.64 | 10.70 | 15.26 | [params](https://disk.pku.edu.cn/link/AA32F27FDBAF1D4DAF90FD1D3F01E5B881)|


## Citing

Please consider citing our paper if it helps your research.

```
@inproceedings{CMD-SE_2024_CVPR,
 title={Towards Open-Vocabulary HOI Detection via Conditional Multi-level Decoding and Fine-grained Semantic Enhancement},
 author={Ting Lei, Shaofeng Yin, and Yang Liu},
 year={2024},
 booktitle={CVPR},
 organization={IEEE}
}
```

## Acknowledgement
We thank [THID](https://github.com/scwangdyd/promting_hoi) for open-sourcing their code.
We would also like to thank the anonymous reviewers for their constructive feedback.