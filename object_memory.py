import os, sys, time

print("Starting imports")
start_time = time.time()

sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "recognize-anything"))

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
            
            words_to_ignore = ["image", "floor", "wood floor", "wall", "hardwood floor", "carpet", "living room", "ceiling", "room", "curtain", "den", "window", "floor", "wall", "red", "yellow", "white", "blue", "green", "brown"]

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

    def __add__(self, info):
        self.names += info.names
        self.embeddings += info.embeddings
        self.pcd = np.concatenate([self.pcd, info.pcd], axis=-1)

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


class ObjectMemory:
    def __init__(self, device, ram_pretrained_path, sam_checkpoint_path, lora_path=None):
        self.device = device

        self.objectFinder = ObjectFinder(self.device)
        self.loraModule = LoraRevolver(self.device)

        self.objectFinder._load_models(ram_pretrained_path)
        self.objectFinder._load_sam(sam_checkpoint_path)

        if lora_path != None:
            self.loraModule.load_lora_ckpt_from_file(lora_path, "5x40")

        self.num_objects_stored = 0
        self.memory = dict() # store ObjectInfo classes here

    # visualisation and utility
    def view_memory(self):
        print("Objects stored in memory:")
        for _, info in self.memory.items():
            print(info.names)
            print(info)
        print()

    def clear_memory(self):
        self.num_objects_stored = 0
        self.memory = dict()


    """
    Takes in an image and depth_image path, returns the following:
        object phrases
        embeddings of grounded images containing that object
        backprojected pointclouds

        there are an equal number of each, phrase_i <=> emb_i <=> pcd_i <=>
    """
    def _get_object_info(self, image_path, depth_image_path):
        if image_path == None or depth_image_path == None:
            raise NotImplementedError
        else:
            # segment objects, get (grounded_image bounding boxes, segmentation mask and label) per box
            # TODO RAM
            obj_grounded_imgs, obj_bounding_boxes, obj_masks, obj_phrases = self.objectFinder.find(image_path)
            
            # get ViT+LoRA embeddings, use bounding boxes and the image to get grounded images
            embs = self.loraModule.encode_image(obj_grounded_imgs)
            
            # filter and transform pcds to the global frame
            obj_pointclouds = self.objectFinder.getDepth(depth_image_path, obj_masks)

            # check that all info recovered
            assert(len(obj_grounded_imgs) == len(obj_bounding_boxes) \
                    and len(obj_bounding_boxes) == len(obj_masks) \
                    and len(obj_masks) == len(obj_phrases) \
                    and len(embs) == len(obj_phrases))

            # can return (obj_grounded_imgs, obj_bounding_boxes) if needed

            return obj_phrases, embs, obj_pointclouds


    """
    takes in an rgb-d image and associated pose, detects all objects present, stores into memory
     
    image_path: path to png file containing the rgb image (3,W,H)
    depth_image_path: path to .npy file containing the depth img in npy format
    pose: [x, y, z, qw, qx, qy, qz]
    bounding_box_threshold: lax condition that checks for 3d bounding box IoU
    occlusion_overlap_threshold: very strict condition that only checks for overlap
                                 included to make sure heavily occluded objects are aggregated properly
    """
    def process_image(self, image_path=None, depth_image_path=None, pose=None, bounding_box_threshold=0.3,  occlusion_overlap_threshold=0.9, testname="", outlier_removal_config=None):
        if outlier_removal_config == None:
            outlier_removal_config = {
                "radius_nb_points": 8,
                "radius": 0.05,
            }
        
        if image_path == None or depth_image_path == None:
            raise NotImplementedError
        else:
            obj_phrases, embs, obj_pointclouds = self._get_object_info(image_path, depth_image_path)
            
            filtered_pointclouds = []
            for points in obj_pointclouds:  # filter
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.T)
                inlier_pcd, _ = pcd.remove_radius_outlier(nb_points=outlier_removal_config["radius_nb_points"],
                                                              radius=outlier_removal_config["radius"])
                filtered_pointclouds.append(np.asarray(inlier_pcd.points).T)

            transformed_pointclouds = [transform_pcd_to_global_frame(pcd, pose) for pcd in filtered_pointclouds]


            # for each tuple, consult already stored memory, match tuples to stored memory (based on 3d IoU)
                # TODO optimise and batch, fetch all memory bounding boxes once
                # remove double loop
            print(obj_phrases)
            
            for i, (obj_phrase, emb, q_pcd) in enumerate(zip(obj_phrases, embs, transformed_pointclouds)):
                obj_exists = False

                # np.save("pcds/" + testname + "_" + str(i) + ".npy", q_pcd)

                print("Detected: ", obj_phrase)

                for obj_id, info in self.memory.items():
                    object_pcd = info.pcd
                    IoU3d = calculate_3d_IoU(q_pcd, object_pcd)
                    overlap3d = calculate_strict_overlap(q_pcd, object_pcd)
                    print("\tFound in mem (info, iou, strict_overlap): ", info, IoU3d, overlap3d)

                    # if the iou is above the threshold, consider it to be the same object/instance
                    if IoU3d > bounding_box_threshold or overlap3d > occlusion_overlap_threshold:
                        info.addInfo(obj_phrase ,emb, q_pcd)
                        obj_exists = True
                        break

                # new object detected
                if not obj_exists:
                    new_obj_info = ObjectInfo(self.num_objects_stored,
                                                obj_phrase,
                                                emb,
                                                q_pcd)
                    
                    print('Object added\n', obj_phrase, '\n', new_obj_info, '\n')
                    
                    self.memory[self.num_objects_stored] = new_obj_info
                    self.num_objects_stored += 1
                else:
                    print('Object exists, aggregated to\n', info, '\n')
            
            # TODO consider downsampling points (optimisation)

    """
    Runs through all objects stored in memory, consolidates objects that have a sufficient overlap
    Performs uniform downsampling on all pointclouds as well

    Consolidates memory in place
    """
    def consolidate_memory(self, bounding_box_threshold=0.3,  occlusion_overlap_threshold=0.9, downsample_voxel_size=0.01):
        new_memory = dict()
        for obj_id, obj_info in self.memory.items():
            obj_pcd = obj_info.pcd

            # check all objects in new_memory to try and match them
            match_found = False
            for new_id, new_obj_info in new_memory.items():
                IoU3d = calculate_3d_IoU(new_obj_info.pcd, obj_pcd)
                overlap3d = calculate_strict_overlap(new_obj_info.pcd, obj_pcd)

                # object overlaps enough to be consolidated with the object in new memory
                if IoU3d > bounding_box_threshold and overlap3d > occlusion_overlap_threshold:
                    match_found = True
                    break
            
            if match_found:
                new_memory[new_id] += obj_info
            else:
                new_memory[len(new_memory)] = self.memory[obj_id]

        del self.memory
        self.memory = new_memory

        # downsample all pcds
        tempPcd = o3d.geometry.PointCloud()
        for obj_id in self.memory:
            tempPcd.points = o3d.utility.Vector3dVector(self.memory[obj_id].pcd.T)
            tempPcd = tempPcd.voxel_down_sample(downsample_voxel_size)
            self.memory[obj_id].pcd = np.array(tempPcd.points).T



    """
    Given an image and a corresponding depth image in an unknown frame, consult the stored memory
    and output a pose in the world frame of the pcds stored in memory

    pose returned as [x, y, z, qw, qx, qy, qz]
    """
    def localise(self, image_path, depth_image_path, icp_threshold=.2, testname=""):
        localized_pose = np.zeros(7, dtype=np.float32)      # default pose, no translation or rotation
        localized_pose[3] = 1.

        # extract all objects currently seen, get embeddings, pcds in the local unknown frame
        if image_path == None or depth_image_path == None:
            raise NotImplementedError
        else:
            # get relevant obj info
            _, detected_embs, detected_pointclouds = self._get_object_info(image_path, depth_image_path)

            # correlate embeddings with objects in memory for all seen objects
            # TODO maybe a KNN search will do better?
            for _, m in self.memory.items(): m.computeMeans()       # update object info means
            memory_embs = torch.Tensor([m.mean_emb for _, m in self.memory.items()]).to(self.device)

            if len(detected_embs) > len(memory_embs):
                detected_embs = detected_embs[:len(memory_embs)]

            # Detected x Mem x Emb sized
            cosine_similarities = F.cosine_similarity(detected_embs.unsqueeze(1), memory_embs.unsqueeze(0), axis=-1).cpu()



            # TODO optimisations may be possible
            # run ICP/FPFH loop closure to get an estimated transform for each seen object
            R_matrices = np.zeros((len(detected_pointclouds), len(memory_embs), 3, 3), dtype=np.float32)
            t_vectors = np.zeros((len(detected_pointclouds), len(memory_embs), 3), dtype=np.float32)


            # save pcds
            for i, d in enumerate(detected_pointclouds): np.save("pcds/detected_pcd" + str(i) + ".npy", d)
            for j, (_, m) in enumerate(self.memory.items()): np.save(f"pcds/%smemory_pcd" % str(testname) + str(j) + ".npy", m.pcd)
            print("Pcds saved")

            detected_pcd = o3d.geometry.PointCloud()
            memory_pcd = o3d.geometry.PointCloud()
            # use centered pcds to get a better ICP initialisation
            # with tqdm(total=len(detected_pointclouds) * len(self.memory)) as pbar:
            #     for i, d in enumerate(detected_pointclouds):
            #         detected_mean = np.mean(d, axis=-1)
            #         detected_pcd.points = o3d.utility.Vector3dVector(d.T - detected_mean)

            #         for j, (_, m) in enumerate(self.memory.items()):
            #             memory_mean = np.mean(m.pcd, axis=-1)
            #             memory_pcd.points = o3d.utility.Vector3dVector(m.pcd.T - memory_mean)

            #             # voxel downsample for equal point density
            #             # registration = o3d.pipelines.registration.registration_icp(
            #             #     detected_pcd.voxel_down_sample(0.05), 
            #             #     memory_pcd.voxel_down_sample(0.05),
            #             #     icp_threshold,
            #             #     np.eye(4),
            #             #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            #             #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
            #             # )
            #             # transform = registration.transformation

            #             transform = register_point_clouds(detected_pcd, memory_pcd, voxel_size=0.1)
                        
            #             R = transform[:3,:3]
            #             t = transform[:3, 3]
                        
            #             R_matrices[i, j] = R
            #             t_vectors[i, j] = t
                        
            #             # adjust transformation to account for centered pcds
            #             # M = [R|t]D + (mean_M - R @ mean_D)
            #             t_vectors[i,j] += memory_mean - R@detected_mean

            #             pbar.update(1)

            # jcbb association using lora embeddings, estimated rotation/transform for each object to encode cost
                    
            # precalculation done, begin assigning objects
            # TODO unseen objects in detected objects are not being dealt with, assuming that all detected objects can be assigned to mem objs
            j = JCBB(cosine_similarities, R_matrices)
            assns = j.get_assignments()

            # calculate paths for all assingments, pick the best one
            best_assignment = assns[0]
            min_cost = 1e11

            # print()
            # print()
            # print(cosine_similarities)
            # print()
            # print()
            for assn in assns:
                # for now, normalized product of cosine differences
                cost = 0

                ### COST FUNCTION ###
                for i,j in enumerate(assn):
                    cost += (1 - cosine_similarities[i,j])      
                cost = np.log(cost) * 1./len(assn)      # normalized product of cosine DIFFERENCES

                # get min cost
                if cost < min_cost:
                    min_cost = cost
                    best_assignment = assn
            
            #TODO use best assignment

            # use ALL object pointclouds together
            all_detected_points = []
            all_memory_points = []
            for i,j in enumerate(best_assignment):
                all_detected_points.append(detected_pointclouds[i])
                all_memory_points.append(self.memory[i].pcd)
            all_detected_points = np.concatenate(all_detected_points, axis=-1)
            all_memory_points = np.concatenate(all_memory_points, axis=-1)

            detected_mean = np.mean(all_detected_points, axis=-1)
            memory_mean = np.mean(all_memory_points, axis=-1)
            
            all_detected_pcd = o3d.geometry.PointCloud()
            all_detected_pcd.points = o3d.utility.Vector3dVector(all_detected_points.T - detected_mean)
            
            all_memory_pcd = o3d.geometry.PointCloud()
            all_memory_pcd.points = o3d.utility.Vector3dVector(all_memory_points.T - memory_mean)

            transform = register_point_clouds(all_detected_pcd, all_memory_pcd, voxel_size=0.1)

            R = copy.copy(transform[:3,:3])
            t = copy.copy(transform[:3, 3])
            
            tAvg = t + memory_mean - R@detected_mean
            qAvg = Rotation.from_matrix(R).as_quat()

            localised_pose = np.concatenate((tAvg, qAvg))


            """
            # object wise alignment

            # using https://math.stackexchange.com/questions/61146/averaging-quaternions direct/fast averaging
            qAvg = np.zeros(4)
            q0 = None
            tAvg = np.zeros(3)

            print(best_assignment)

            for i,j in enumerate(best_assignment):
                # roughly avg rotation
                q = Rot.from_matrix(R_matrices[i,j])
                q = q.as_quat()

                print("Rotation: ", R_matrices[i,j])
                print(i,j)
                print()

                print("Quaternion obtained: ", i, " | ", q, '\n', R_matrices[i,j], "\n")

                if i > 0:
                    if np.dot(q, q0) < 0:
                        q = -q
                else:
                    q0 = q
                qAvg += q

                # avg translation
                tAvg += t_vectors[i,j]

            qAvg = qAvg / np.linalg.norm(qAvg)
            tAvg /= len(best_assignment)
            
            localised_pose = np.concatenate((tAvg, qAvg))
            """
            return localised_pose



if __name__ == "__main__":
    largs = tyro.cli(LocalArgs, description=__doc__)
    print(largs)

    tgt = []
    pred = []

    print("Begin")
    mem = ObjectMemory(device = largs.device, 
                       ram_pretrained_path=largs.ram_pretrained_path,
                       sam_checkpoint_path = largs.sam_checkpoint_path,
                       lora_path=largs.lora_path)
    print("\nMemory Init'ed")

    with open(largs.poses_json_path, 'r') as f:
        poses = json.load(f)

    # for target in range(1,9): # all of them
    # for target in [6]: # sanity check
    for target in [1]:
        target_num = target
        target_pose = None
        for i, view in enumerate(poses["views"]):
            num = i+1

            # view 6 is our unseen view, skip
            print(f"Processing img %d" % num)
            q = Rotation.from_euler('zyx', [r for _, r in view["rotation"].items()], degrees=True).as_quat()
            t = np.array([x for _, x in view["position"].items()])
            pose = np.concatenate([t, q])
            if num == target_num:
                target_pose = pose
                continue
            else:
                print("Pose: ", pose)
            
            mem.process_image(testname=f"view%d" % num, image_path=f"360_zip/view%d/view%d.png" % (num, num), depth_image_path=f"360_zip/view%d/view%d.npy" % (num, num), pose=pose)
            print("Processed\n")

        mem.view_memory()

        print("Consolidating memory")
        mem.consolidate_memory()
        mem.view_memory()

        estimated_pose = mem.localise(image_path=f"360_zip/view%d/view%d.png" % (target_num, target_num), depth_image_path=f"360_zip/view%d/view%d.npy" % (target_num, target_num))

        print("Target pose: ", target_pose)
        print("Estimated pose: ", estimated_pose)

        mem.clear_memory()

        tgt.append(target_pose)
        pred.append(estimated_pose)

        # for _, m in mem.memory.items():
        #     np.save(f"pcds/new%d.npy" % m.id, m.pcd)
        torch.cuda.empty_cache()

    for i, t, p in zip(range(1,9), tgt, pred):
        print("Pose: ", i)
        print("Target pose:", t)
        print("Estimated pose:", p)
        print()
