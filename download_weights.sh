#!/bin/bash

mkdir -p /scratch/aneesh/
wget -O /scratch/aneesh/ram_swin_large_14m.pth https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth
wget -O /scratch/aneesh/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth