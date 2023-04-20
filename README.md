# SmallObjectDetection

This repository includes the codes used in the [challenge](http://www.mva-org.jp/mva2023/challenge).

It is built on MMDetection V2.24.1 (released on Apr 30, 2022, source code is downloaded from [here](https://github.com/open-mmlab/mmdetection/releases/tag/v2.24.1)).


## Hardware Requirements
NVIDIA-SMI 510.47.03    
Driver Version: 510.47.03    
CUDA Version: 11.6 

We performed our executions in google cloud. Below are the machine configurations used:

GPU Type- NVIDIA T4 and NVIDIA V100


## Software Requirements

Python version -3.7
torch - 1.12.1
torchvision - 0.13.0
mmvc-full -1.6.0

## Dataset
**[Download Link](https://drive.google.com/drive/folders/1vTHiIelagbzPO795yhOdNUFh9u2XxZP-?usp=share_link)**  

Dataset Directory Structure
```
data
 ├ drone2021 (62.4GB)
 │  ├ images
 │  └ annotations
 ├ mva2023_sod4bird_train (9.4GB)
 │  ├ images
 │  └ annotations
 ├ mva2023_sod4bird_pub_test (8.7GB)
 │  ├ images
 │  └ annotations(including an empty annotation file)
 └ mva2023_sod4bird_private_test (4kB)
    ├ images(empty)
    └ annotations(empty)
```

## Pretained Weights

Weights ResNet18: https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth

