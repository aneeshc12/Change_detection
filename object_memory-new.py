import os, sys, time

print("Starting imports")
start_time = time.time()

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
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot
import json
from dataclasses import dataclass, field
from jcbb import JCBB
from fpfh.fpfh_register import register_point_clouds

end_time = time.time()
print(f"Imports completed in {end_time - start_time} seconds")


@dataclass
class LocalArgs:
    """
    Class to hold local configuration arguments.
    """
    lora_path: str='models/vit_finegrained_5x40_procthor.pt'
    poses_json_path: str='/home2/aneesh.chavan/Change_detection/360_zip/json_poses.json'
    device: str='cuda'
    sam_checkpoint_path: str = '/scratch/aneesh/sam_vit_h_4b8939.pth'
    ram_pretrained_path: str = '/scratch/aneesh/ram_swin_large_14m.pth'


"""
#################
        Object Detection Classes
#################
"""


class LoraRevolver:
    """
    Loads a base ViT and a set of LoRa configs, allows loading and swapping between them.
    """
    def __init__(self, device, model_checkpoint="google/vit-base-patch16-224-in21k"):
        """
        Initializes the LoraRevolver object.
        
        Parameters:
        - device (str): Device to be used for compute.
        - model_checkpoint (str): Checkpoint for the base ViT model.
        """
        self.device = device
        
        # self.base_model will be augmented with a saved set of lora_weights
        # self.lora_model is the augmented model (NOTE)
        self.base_model = ViTModel.from_pretrained(
            model_checkpoint,
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.lora_model = self.base_model

        # image preprocessors the ViT needs
        self.image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
        self.normalize = Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)
        self.train_transforms = Compose(
            [
                RandomResizedCrop(self.image_processor.size["height"]),
                RandomHorizontalFlip(),
                ToTensor(),
                self.normalize,
            ]
        )
        self.test_transforms = Compose(
            [
                Resize(self.image_processor.size["height"]),
                CenterCrop(self.image_processor.size["height"]),
                ToTensor(),
                self.normalize,
            ]
        )

        # stored lora_configs, ready to be swapped in
        # only expects store lora_checkpoints.pt objects created by this class
        self.ckpt_library = {}


    def load_lora_ckpt_from_file(self, config_path, name):
        """
        Load a LoRa config from a saved file.
        
        Parameters:
        - config_path (str): Path to the LoRa config file.
        - name (str): Name to associate with the loaded config.
        """
        ckpt = torch.load(config_path)
        try:
            self.ckpt_library[str(name)] = ckpt
            del self.lora_model
            self.lora_model = get_peft_model(self.base_model,
                                                ckpt["lora_config"]).to(self.device)
            self.lora_model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        except:
            print("Lora checkpoint invalid")
            raise IndexError

    def encode_image(self, imgs):
        """
        Use the current LoRa model to encode a batch of images.
        
        Parameters:
        - imgs (list): List of images to encode.
        
        Returns:
        - emb (torch.Tensor): Encoded embeddings for the input images.
        """
        with torch.no_grad():
            if isinstance(imgs[0], np.ndarray):
                img_batch = torch.stack([Compose([ToPILImage(),
                                                  self.test_transforms])(i) for i in imgs])
            else:
                img_batch = torch.stack([self.test_transforms(i) for i in imgs])
            # if len(img.shape) == 3:
            #     img = img.unsqueeze(0)    # if the image is unbatched, batch it
            emb = self.lora_model(img_batch.to(self.device), output_hidden_states=True).last_hidden_state[:,0,:]
        
        return emb
    
    def train_current_lora_model(self):
        """
        Train the current LoRa model.
        """
        pass

    def save_lora_ckpt(self):
        """
        Save the current LoRa model checkpoint.
        """
        pass



class ObjectFinder:
    """
    Class that detects objects through segmentation.
    """
    def __init__(self, device, box_threshold=0.35, text_threshold=0.55):
        """
        Initializes the ObjectFinder object.
        
        Parameters:
        - device (str): Device to be used for compute.
        - box_threshold (float): Threshold for bounding box detection.
        - text_threshold (float): Threshold for text detection.
        """
        self.device =  device
        
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold


    def _load_models(self, ram_pretrained_path):
        """
        Load RAM and Grounding Dino models.
        
        Parameters:
        - ram_pretrained_path (str): Path to the pretrained RAM model.
        """
        # ram
        self.ram_model = ram(pretrained=ram_pretrained_path, image_size=384, vit='swin_l')
        self.ram_model.eval()
        self.ram_model.to(self.device)
        self.ram_transform = get_transform_ram(image_size=384)

        # grounding dino
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

        cache_config_file = hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        args.device = self.device
        self.groundingdino_model = build_model(args)

        cache_file = hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_filenmae)
        checkpoint = torch.load(cache_file, map_location=self.device)
        log = self.groundingdino_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        self.groundingdino_model.eval()


    def _load_sam(self, sam_checkpoint_path):
        """
        Load SAM model from checkpoint.
        
        Parameters:
        - sam_checkpoint_path (str): Path to the SAM model checkpoint.
        """
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint_path).to(self.device).eval())
        # self.sam_predictor.to(self.device)
        # self.sam_predictor.eval()
        print("SAM loaded")


    def _getIoU(self, rect1, rect2):
        """
        Calculate the intersection over union (IoU) between two rectangles.
        
        Parameters:
        - rect1 (tuple): Coordinates of the first rectangle (x, y, width, height).
        - rect2 (tuple): Coordinates of the second rectangle (x, y, width, height).
        
        Returns:
        - percent_overlap (float): Percentage of overlap between the rectangles.
        """
        area_rect1 = rect1[2]*rect1[3]
        area_rect2 = rect2[2]*rect2[3]

        overlap_top_left = (max(rect1[0], rect2[0]), max(rect1[1], rect2[1]))
        overlap_bottom_right = (min(rect1[0] + rect1[2], rect2[0] + rect2[2]), min(rect1[1] + rect1[3], rect2[1] + rect2[3]))

        if (overlap_bottom_right[0] <= overlap_top_left[0]) or (overlap_bottom_right[1] <= overlap_top_left[1]):
            return 0.0  # No overlap, return 0% overlap

        # Calculate the area of the overlap rectangle
        overlap_area = abs((overlap_bottom_right[0] - overlap_top_left[0]) * (overlap_bottom_right[1] - overlap_top_left[1]))
        percent_overlap = (overlap_area / min(area_rect1, area_rect2))

        return percent_overlap


    def _compSize(self, rect1, rect2):
        """
        Compare the sizes of two rectangles.
        
        Parameters:
        - rect1 (tuple): Coordinates of the first rectangle (x, y, width, height).
        - rect2 (tuple): Coordinates of the second rectangle (x, y, width, height).
        
        Returns:
        - diff (float): Size difference between the rectangles.
        """
        area_rect1 = rect1[2]*rect1[3]
        area_rect2 = rect2[2]*rect2[3]

        diff = min(area_rect1, area_rect2)/max(area_rect1, area_rect2)
        return diff


    def getBoxes(self, image, text_prompt, show=False, intersection_threshold=0.7, size_threshold=0.75):
        """
        Given a phrase, filter and get all boxes and phrases.
        
        Parameters:
        - image (np.ndarray): Input image.
        - text_prompt (str): Phrase for object detection.
        - show (bool): Whether to display intermediate results.
        - intersection_threshold (float): Threshold for box intersection.
        - size_threshold (float): Threshold for box size difference.
        
        Returns:
        - boxes (torch.Tensor): Detected bounding boxes.
        - phrases (list): List of detected phrases.
        """
        keywords = [k.strip() for k in text_prompt.split('.')]

        with torch.no_grad():
            boxes = []
            phrases = []
            unique_boxes_num = 0

            for i, word in enumerate(keywords):
                # af, detected, detected_phrases = self._detect(image, image_source=image_source, text_prompt=str(word))

                detected, _, detected_phrases = gd_predict(
                    model=self.groundingdino_model,
                    image=image,
                    caption=str(word),
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold
                )

                # print("FIND: ", word, detected_phrases, detected)
                if show:
                    print(i)
                unique_enough = True

                if detected != None and len(detected) != 0:
                    if unique_boxes_num == 0:
                        for box in detected:
                            boxes.append(box)
                            phrases.append(word)
                            unique_boxes_num += 1
                    else:
                        # print("boxes: ", boxes)
                        for box in detected:
                            unique_enough = True

                            # if show:
                            #     print("detected: ", detected)

                            for prev in boxes[:unique_boxes_num]:
                                iou = self._getIoU(box, prev)
                                diff = self._compSize(box, prev)

                                if (iou > intersection_threshold and diff > size_threshold):
                                    # bounding box is not unique enough to be added
                                    unique_enough = False

                                    # if show:
                                    #     print("failed")
                                    #     break

                            if unique_enough:
                                boxes.append(box)
                                phrases.append(word)
                                unique_boxes_num += 1

            return torch.stack(boxes), phrases
        
    def segment(self, image, boxes):
        """
        Segment objects in the image based on provided bounding boxes.
        
        Parameters:
        - image (np.ndarray): Input image.
        - boxes (torch.Tensor): Bounding boxes for object segmentation.
        
        Returns:
        - boxes_xyxy (torch.Tensor): Transformed bounding boxes.
        - masks (torch.Tensor): Segmentation masks.
        """
        with torch.no_grad():
            self.sam_predictor.set_image(image)
            H, W, _ = image.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image.shape[:2])
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
                )
            return boxes_xyxy, masks


    def find(self, image_path=None, caption=None):
        """
        Find and ground objects in the given image.
        
        Parameters:
        - image_path (str): Path to the input image.
        - caption (str): Caption for object detection.
        
        Returns:
        - grounded_objects (list): List of grounded object images.
        - boxes (torch.Tensor): Detected bounding boxes.
        - masks (torch.Tensor): Segmentation masks.
        - phrases (list): List of detected phrases.
        """
        if type(image_path) == None:
            raise NotImplementedError
        else:
            image_source, image = gd_load_image(image_path)
        
        # get object names
        if caption == None:
            img_ram = self.ram_transform(PIL.Image.fromarray(image_source)).unsqueeze(0).to(self.device)
            caption = inference_ram(img_ram, self.ram_model)[0].split("|")
            
            words_to_ignore = ["carpet", "living room", "ceiling", "room", "curtain", "den", "window", "floor", "wall", "red", "yellow", "white", "blue", "green", "brown"]

            filtered_caption = ""
            for c in caption:
                if c.strip() in words_to_ignore:
                    continue
                else:
                    filtered_caption += c
                    filtered_caption += " . "
            filtered_caption = filtered_caption[:-2]

            print("caption post ram: ", filtered_caption)
        
        # ground them, get associated phrases
        cxcy_boxes, phrases = self.getBoxes(image, filtered_caption)

        boxes, masks = self.segment(image_source, cxcy_boxes)

        # ground objects
        grounded_objects = [image_source[int(bb[1]):int(bb[3]),
                                         int(bb[0]):int(bb[2]), :] for bb in boxes]

        return grounded_objects, boxes, masks, phrases
    

    def _show_detections(self, image_path=None, caption=None):
        """
        Display object detections on the given image.
        
        Parameters:
        - image_path (str): Path to the input image.
        - caption (str): Caption for object detection.
        """
        if type(image_path) == None:
            raise NotImplementedError
        else:
            image_source, image = gd_load_image(image_path)

        ## TODO implement RAM
        if caption==None:
            caption = "sofa . chair . table"

        Image.fromarray(image_source)
        b, l, p = gd_predict(model=self.groundingdino_model, 
                                           image=image, caption=caption,
                                           box_threshold=0.35,
                                           text_threshold=0.55)
        af = gd_annotate(image_source=image_source, boxes=b, logits=l, phrases=p)[...,::-1]
        Image.fromarray(af)
        plt.imshow(af)

    # TODO determine whether outliers need to be filtered here
    def getDepth(self, depth_image_path, masks, f=300):
        """
        Returns a 3D point cloud corresponding to each object based on depth information.

        Parameters:
        - depth_image_path (str): Path to the depth image file.
        - masks (torch.Tensor): Binary segmentation masks for each object.
        - f (float): Focal length for depth-to-distance conversion.

        Returns:
        - all_pointclouds (list): List of 3D point clouds for each segmented object.
        """
        if depth_image_path is None:
            raise NotImplementedError
        else:
            depth_image = np.load(depth_image_path)
            
            w, h = depth_image.shape
            num_objs = masks.shape[0]

            stacked_depth = np.tile(depth_image, (num_objs, 1, 1))  # Get all centroids/point clouds together
            stacked_depth[masks.squeeze(dim=1).cpu() == False] = 0  # Remove the depth channel from the masks

            horizontal_distance = np.tile(np.linspace(-h/2, h/2, h, dtype=np.float32), (num_objs, w, 1))
            vertical_distance = np.tile(np.linspace(w/2, -w/2, w, dtype=np.float32).reshape(-1, 1), (num_objs, 1, h))

            X = horizontal_distance * stacked_depth / f
            Y = vertical_distance * stacked_depth / f
            Z = stacked_depth

            # Combine calculated X, Y, Z points
            all_pointclouds = np.stack([X, Y, Z], 1).reshape((num_objs, 3, -1))

            # Filter out [0,0,0]
            all_pointclouds = [pcd[:, pcd[2, :] != 0] for pcd in all_pointclouds]
            
            return all_pointclouds
        

"""
#################
        Utility Functions
#################
"""


def transform_pcd_to_global_frame(pcd, pose):
    """
    Transforms a point cloud into the global frame based on a given pose.

    Parameters:
    - pcd (numpy.ndarray): 3D point cloud represented as a 3xN array.
    - pose (numpy.ndarray): Pose of the camera frame with respect to the world frame
      represented as [x, y, z, qw, qx, qy, qz].

    Returns:
    - transformed_pcd (numpy.ndarray): Transformed point cloud in the global frame.
    """
    t = pose[:3]
    q = pose[3:]

    q /= np.linalg.norm(q)
    R = Rotation.from_quat(q).as_matrix()

    transformed_pcd = R @ pcd
    transformed_pcd += t.reshape(3, 1)

    return transformed_pcd

def calculate_3d_IoU(pcd1, pcd2):
    """
    Calculates the 3D Intersection over Union (IoU) between two 3D point clouds.

    Parameters:
    - pcd1 (numpy.ndarray): First 3D point cloud represented as a 3xN array.
    - pcd2 (numpy.ndarray): Second 3D point cloud represented as a 3xN array.

    Returns:
    - IoU (float): 3D Intersection over Union between the two point clouds.
    """
    bb1_min = pcd1.min(axis=-1)
    bb1_max = pcd1.max(axis=-1)

    bb2_min = pcd2.min(axis=-1)
    bb2_max = pcd2.max(axis=-1)

    overlap_min_corner = np.stack([bb1_min, bb2_min], axis=0).max(axis=0)
    overlap_max_corner = np.stack([bb1_max, bb2_max], axis=0).min(axis=0)

    if (overlap_min_corner > overlap_max_corner).any():
        return 0
    else:
        v = overlap_max_corner - overlap_min_corner
        overlap_volume = v[0] * v[1] * v[2]
        
        bb1 = bb1_max - bb1_min
        bb2 = bb2_max - bb2_min

        v1 = bb1[0] * bb1[1] * bb1[2]
        v2 = bb2[0] * bb2[1] * bb2[2]

        IoU = overlap_volume / (v1 + v2 - overlap_volume)

        return IoU

def calculate_strict_overlap(pcd1, pcd2):
    """
    Calculates the strict overlap between two 3D point clouds.

    Parameters:
    - pcd1 (numpy.ndarray): First 3D point cloud represented as a 3xN array.
    - pcd2 (numpy.ndarray): Second 3D point cloud represented as a 3xN array.

    Returns:
    - overlap (float): Strict overlap between the two point clouds.
    """
    bb1_min = pcd1.min(axis=-1)
    bb1_max = pcd1.max(axis=-1)

    bb2_min = pcd2.min(axis=-1)
    bb2_max = pcd2.max(axis=-1)

    overlap_min_corner = np.stack([bb1_min, bb2_min], axis=0).max(axis=0)
    overlap_max_corner = np.stack([bb1_max, bb2_max], axis=0).min(axis=0)

    if (overlap_min_corner > overlap_max_corner).any():
        return 0
    else:
        v = overlap_max_corner - overlap_min_corner
        overlap_volume = v[0] * v[1] * v[2]
        
        bb1 = bb1_max - bb1_min
        bb2 = bb2_max - bb2_min

        v1 = bb1[0] * bb1[1] * bb1[2]
        v2 = bb2[0] * bb2[1] * bb2[2]

        overlap = overlap_volume / min(v1, v2)

        return overlap



"""
#################
        Object Memory Classes
#################
"""


class ObjectInfo:
    """
    Bundles together object information for distinct objects.

    Attributes:
    - id (int): Object ID.
    - names (list): List of object names.
    - embeddings (list): List of embeddings associated with the object.
    - pcd (numpy.ndarray): Point cloud data for the object.
    - mean_emb (numpy.ndarray): Mean embedding of the object.
    - centroid (numpy.ndarray): Centroid of the object in 3D space.

    Methods:
    - addInfo(name, embedding, pcd): Adds information for the object, including name, embedding, and point cloud data.
    - computeMeans(): Computes the mean embedding and centroid for the object.
    - __repr__(): Returns a string representation of the object information.
    """

    def __init__(self, id, name, emb, pcd):
        """
        Initializes ObjectInfo with the given ID, name, embedding, and point cloud data.

        Parameters:
        - id (int): Object ID.
        - name (str): Object name.
        - emb (numpy.ndarray): Object embedding.
        - pcd (numpy.ndarray): Object point cloud data.
        """
        self.id = id
        self.names = [name]
        self.embeddings = [emb]
        self.pcd = pcd

        self.mean_emb = None
        self.centroid = None

    def addInfo(self, name, embedding, pcd):
        """
        Adds information for the object, including name, embedding, and point cloud data.

        Parameters:
        - name (str): Object name to be added.
        - embedding (numpy.ndarray): Object embedding to be added.
        - pcd (numpy.ndarray): Object point cloud data to be added.
        """
        if name not in self.names:
            self.names.append(name)
        self.embeddings.append(embedding)
        self.pcd = np.concatenate([self.pcd, pcd], axis=-1)

    def computeMeans(self):
        """
        Computes the mean embedding and centroid for the object.
        """
        # TODO messy, clean this up
        self.mean_emb = np.mean(np.asarray(
            [e.cpu() for e in self.embeddings]), axis=0)
        self.centroid = np.mean(self.pcd, axis=-1)

    def __repr__(self):
        """
        Returns a string representation of the object information.
        """
        return(f"ID: %d | Names: [%s] |  Num embs: %d | Pcd size: " % \
              (self.id, " ".join(self.names), len(self.embeddings)) + str(self.pcd.shape))

