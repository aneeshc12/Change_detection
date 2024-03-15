import os, sys, time

print("Starting imports")
start_time = time.time()

sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "recognize-anything"))
sys.path.append(os.path.join(os.getcwd(), "Objectron"))
sys.path.append(os.path.join(os.getcwd(), "FourDNet-wrapper"))
sys.path.append(os.path.join(os.getcwd(), "FourDNet-wrapper", "config"))
sys.path.append(os.path.join(os.getcwd(), "FourDNet-wrapper", "datasets"))
sys.path.append(os.path.join(os.getcwd(), "FourDNet-wrapper", "loss"))
sys.path.append(os.path.join(os.getcwd(), "FourDNet-wrapper", "model"))
sys.path.append(os.path.join(os.getcwd(), "FourDNet-wrapper", "processor"))
sys.path.append(os.path.join(os.getcwd(), "FourDNet-wrapper", "solver"))
sys.path.append(os.path.join(os.getcwd(), "FourDNet-wrapper", "utils"))

# fourDNet
from test_heatmap import load_reid_model, get_reid_emb

import os, sys, time
import tyro
import argparse
import copy
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
from ram.models import ram
from ram import inference_ram
from ram import get_transform as get_transform_ram
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
import supervision as sv
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
import open3d as o3d
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    ToPILImage
)
from tqdm import tqdm
from transformers import ViTConfig, ViTModel, ViTForImageClassification
from transformers import AutoImageProcessor, CLIPVisionModel
from peft import LoraConfig, get_peft_model
from GroundingDINO.groundingdino.util.inference import annotate as gd_annotate 
from GroundingDINO.groundingdino.util.inference import load_image as gd_load_image
from GroundingDINO.groundingdino.util.inference import predict as gd_predict
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy import arctan2
import torch.nn.functional as F
import json
from dataclasses import dataclass, field
from jcbb import JCBB
from fpfh.fpfh_register import register_point_clouds
from similarity_volume import *

from objectron.dataset import box, iou

# copy pasted from test_heatmap
import os
import torch
from config import cfg
import argparse
# from datasets import make_dataloader
from model import make_model
# from processor import do_inference
# from utils.logger import setup_logger
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np
import cv2
import os
import shutil
import os.path as osp
from tqdm import tqdm
import torch.nn.functional as F
import pickle
from dataclasses import dataclass
import tyro

end_time = time.time()
print(f"Imports completed in {end_time - start_time} seconds")

NUM_CLASSES = 69
NUM_VIEWS = 5 

@dataclass
class LocalArgs:
    """
    Class to hold local configuration arguments.
    """
    reid_config_file: str = "./FourDNet-wrapper/config.yml"
    reid_model: str = "/scratch/vineeth.bhat/FourDNet/procthor_final.pth"
    reid_num_classes: int = 69
    test_folder: str = "/scratch/vineeth.bhat/FourDNet/data/procthor_final/val"
    reid_model_pretrain_path: str = "/scratch/vineeth.bhat/FourDNet/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth"

if __name__ == "__main__":
    largs = tyro.cli(LocalArgs, description=__doc__)
    print(largs)

    model = load_reid_model(largs)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_path = largs.test_folder
    test_classes = []
    for classname in os.listdir(test_path):
        test_classes.append(classname)


    test_images = [os.listdir(os.path.join(test_path, c)) for c in test_classes]
    
    model_inputs = []
    for class_idx, classname in enumerate(test_classes):
        model_inputs.append([])
        for img_idx, img in enumerate(test_images[class_idx]):
            if img.find(".npy") != -1:
                # if it is a depth image, we do not take it here, it would be taken along with it's rgb counterpart
                continue

            img_name = test_images[class_idx][img_idx]
            rgb_path = osp.join(test_path, classname, img)
            depth_path = osp.join(test_path, classname, img.split(".")[0] + ".npy")

            model_inputs[-1].append((rgb_path, depth_path))


    assert len(model_inputs) == NUM_CLASSES
    for i in range(NUM_CLASSES):
        assert len(model_inputs[i]) == NUM_VIEWS

    model.eval()
    w = []
    with torch.no_grad():
        with tqdm(NUM_CLASSES * NUM_VIEWS) as bar:
            for ctg in model_inputs:
                r = []
                for rgb, depth in ctg:
                    k = get_reid_emb(model, rgb, depth)
                    r.append(k)
                    bar.update(1)
                w.append(torch.stack(r))
    w = torch.stack(w)

    print(f"We have {NUM_VIEWS} views for {NUM_CLASSES} classes.")

    print(f"w.shape before reshaping = {w.shape}")
    w = w.reshape((-1, w.shape[-1]))

    scores = torch.zeros((w.shape[0], w.shape[0])).cpu().numpy()
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            scores[i][j] = w[i] @ w[j] / (torch.norm(w[i]) * torch.norm(w[j]))

    print(f"w.shape = {w.shape}")
    print(f"scores.shape = {scores.shape}")

    plt.figure(figsize=(15, 15))
    plt.imshow(scores, cmap="hot")
    plt.colorbar()

    num_instances = NUM_VIEWS

    x_axis_titles = [
        f"{test_classes[i//num_instances]}"
        for i in range(
            num_instances // 2, num_instances // 2 + len(scores), num_instances
        )
    ]
    y_axis_titles = [
        f"{test_classes[i//num_instances]}"
        for i in range(
            num_instances // 2, num_instances // 2 + len(scores), num_instances
        )
    ]

    plt.xticks(
        range(num_instances // 2, num_instances // 2 + len(scores), num_instances),
        x_axis_titles,
        fontsize=6,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        range(num_instances // 2, num_instances // 2 + len(scores), num_instances),
        y_axis_titles,
        fontsize=6,
        va="center",
    )

    for i in range(1, len(scores)):
        if i % num_instances == 0:
            plt.axvline(x=i - 0.5, color="blue", linestyle="-", linewidth=0.5)
            plt.axhline(y=i - 0.5, color="blue", linestyle="-", linewidth=0.5)

    # Show the heatmap
    plt.title("FourDNet")
    plt.savefig("FourDNet-full-heatmap.jpg")

    print("Saved plot")