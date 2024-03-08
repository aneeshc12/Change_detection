import os, sys, time

print("Starting imports")
start_time = time.time()

sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "recognize-anything"))
sys.path.append(os.path.join(os.getcwd(), "Objectron"))

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

from objectron.dataset import box, iou

end_time = time.time()
print(f"Imports completed in {end_time - start_time} seconds")


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

            try:
                return torch.stack(boxes), phrases
            except:
                return None, None

            
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
            
            words_to_ignore = [
                                "living room", 
                                "ceiling", 
                                "room", 
                                "curtain", 
                                "den", 
                                "window", 
                                "floor", 
                                "wall", 
                                "red", 
                                "yellow", 
                                "white", 
                                "blue", 
                                "green", 
                                "brown", # new additions start in the next line
                                "corridor",
                                "image",
                                "picture frame",
                                "mat",
                                "wood floor",
                                "shadow",
                                "hardwood",
                                "plywood",
                                "waiting room",
                                "lead to",
                                "belly",
                                "person",
                                "chest",
                                "black",
                                "accident",
                                "act",
                                "doorway",
                                "illustration",
                                "animal",
                                "mountain",
                                "table top", # since we don't want a flat object as an instance
                                "pen",
                                "pencil",
                                "corner",
                                "notepad",
                                "flower",
                                "man",
                                "pad",
                                "lead",
                                "ramp",
                                "plank",
                                "scale",
                                "beam",
                                "pink",
                                "tie",
                                "crack",
                                "mirror",
                                "square",
                                "rectangle",
                                "woman",
                                "tree",
                                "umbrella",
                                "hat",
                                "salon",
                                "beach",
                                "open",
                                "closet",
                                "blanket",
                                "circle",
                                "furniture",
                                "balustrade",
                                "cube",
                                "dress",
                                "ladder",
                                "briefcase",
                                "marble",
                                "pillar",
                                "dark",
                                "sea"
            ]
            sub_phrases_to_ignore = [
                                "room",
                                "floor",
                                "wall",
                                "frame",
                                "image",
                                "building",
                                "ceiling"
                                "lead",
                                "paint",
                                "shade",
                                "snow",
                                "rain",
                                "cloud",
                                "frost",
                                "fog",
                                "sky",
                                "carpet",
                                "view",
                                "scene",
                                "mat",
                                "window",
                                "vase",
                                "bureau",
            ]


            def check_whether_in_sub_phrases(text):
                for sub_phrase in sub_phrases_to_ignore:
                    if sub_phrase in text:
                        return True

                return False

            filtered_caption = ""
            for c in caption:
                if c.strip() in words_to_ignore:
                    continue
                if check_whether_in_sub_phrases(c.strip()):
                    continue
                else:
                    filtered_caption += c
                    filtered_caption += " . "
            filtered_caption = filtered_caption[:-2]

            print("caption post ram: ", filtered_caption)
        
        # ground them, get associated phrases
        cxcy_boxes, phrases = self.getBoxes(image, filtered_caption)

        # no objects considered
        if cxcy_boxes is None:
            return None, None, None, None

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

class QuaternionOps:
    @staticmethod
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])

    @staticmethod
    def quaternion_conjugate(q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    # https://math.stackexchange.com/a/3573308
    @staticmethod
    def quaternion_error(q1, q2): # returns orientation angle between the two
        q_del = QuaternionOps.quaternion_multiply(QuaternionOps.quaternion_conjugate(q1), q2)
        q_del_other_way = QuaternionOps.quaternion_multiply(QuaternionOps.quaternion_conjugate(q1), -q2)
        return min(np.abs(arctan2(np.linalg.norm(q_del[1:]), q_del[0])),
                   np.abs(arctan2(np.linalg.norm(q_del_other_way[1:]), q_del_other_way[0])))

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

def calculate_obj_aligned_3d_IoU(pcd1, pcd2):
    """
    Calculates the 3D Intersection over Union (IoU) between two 3D point clouds. Using Objectrons algorithm
    Uses object aligned bounding boxes isntead of axis aligned

    Parameters:
    - pcd1 (numpy.ndarray): First 3D point cloud represented as a 3xN array.
    - pcd2 (numpy.ndarray): Second 3D point cloud represented as a 3xN array.

    Returns:
    - IoU (float): 3D Intersection over Union between the two point clouds.
    """
    def conv_to_objectron_ordering(v):
        v = sorted(v, key=lambda v: v[2])
        v = sorted(v, key=lambda v: v[1])
        v = sorted(v, key=lambda v: v[0])
        return v

    bb1 = o3d.geometry.OrientedBoundingBox.create_from_points(
        points=o3d.utility.Vector3dVector(pcd1.T) #, robust=True
    )
    bb2 = o3d.geometry.OrientedBoundingBox.create_from_points(
        points=o3d.utility.Vector3dVector(pcd2.T) #, robust=True
    )

    bb1_vertices = np.zeros((9,3), dtype=np.float32)
    bb1_vertices[0, :] = bb1.get_center()
    bb1c = np.array(bb1.get_box_points())
    bb1_vertices[1:,:] = conv_to_objectron_ordering(bb1c)

    bb2_vertices = np.zeros((9,3), dtype=np.float32)
    bb2_vertices[0, :] = bb2.get_center()
    bb2c = np.array(bb2.get_box_points())
    bb2_vertices[1:,:] = conv_to_objectron_ordering(bb2c)

    w1 = box.Box(vertices=bb1_vertices)
    w2 = box.Box(vertices=bb2_vertices)

    loss = iou.IoU(w1, w2)
    try:
        iou3d = loss.iou()
    except:
        iou3d = 0.

    return iou3d

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

    def downsample(self, voxel_size, use_external_mesh):
        temp_pc = o3d.geometry.PointCloud()
        temp_pc.points = o3d.utility.Vector3dVector(self.pcd.T)

        if use_external_mesh:
            alpha = 0.03

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(temp_pc.points, alpha)
            mesh.compute_vertex_normals()

            o3d.visualization.draw_geometries([mesh])

            temp_pc.points = mesh.sample_points_uniformly(number_of_points=50000)

        temp_pc = temp_pc.voxel_down_sample(voxel_size=voxel_size)
        self.pcd = np.asarray(temp_pc.points).T 

    def __add__(self, info):
        self.names += info.names
        self.embeddings += info.embeddings
        self.pcd = np.concatenate([self.pcd, info.pcd], axis=-1)

    def addInfo(self, name, embedding, pcd, align=True, max_iteration=30, max_correspondence_distance=0.01):
        """
        Adds information for the object, including name, embedding, and point cloud data.
        Added point cloud data is aligned with a fine-grained point-to-point ICP if the align flag is true
        Added point cloud data is aligned with a fine-grained point-to-point ICP if the align flag is true

        Parameters:
        - name (str): Object name to be added.
        - embedding (numpy.ndarray): Object embedding to be added.
        - pcd (numpy.ndarray): Object point cloud data to be added
        - align (bool): Should the new point information be ailgned to the existing points.
        - pcd (numpy.ndarray): Object point cloud data to be added
        - align (bool): Should the new point information be ailgned to the existing points.
        """
        if name not in self.names:
            self.names.append(name)
        self.embeddings.append(embedding)

        if not align:
            self.pcd = np.concatenate([self.pcd, pcd], axis=-1)
        else:
            memPcd = o3d.geometry.PointCloud()
            newPcd = o3d.geometry.PointCloud()

            memPcd.points = o3d.utility.Vector3dVector(self.pcd.T)
            newPcd.points = o3d.utility.Vector3dVector(pcd.T)

            # Perform ICP registration
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source=newPcd,
                target=memPcd,
                max_correspondence_distance=max_correspondence_distance,  # Adjust as needed based on your data
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )


    def computeMeans(self):
        """
        Computes the mean embedding and centroid for the object.
        """
        # TODO messy, clean this up
        # self.mean_emb = np.mean(np.asarray(
        #     [e.cpu() for e in self.embeddings]), axis=0)
        self.mean_emb = np.mean(np.array(self.embeddings), axis=0)
        self.centroid = np.mean(self.pcd, axis=-1)

    def __repr__(self):
        """
        Returns a string representation of the object information.
        """
        return(f"ID: %d | Names: [%s] |  Num embs: %d | Pcd size: " % \
              (self.id, " ,".join(self.names), len(self.embeddings)) + str(self.pcd.shape))


class ObjectMemory:
    def __init__(self, device, ram_pretrained_path, sam_checkpoint_path, lora_path=None):
        """
        Initializes the ObjectMemory instance.

        Parameters:
        - device (str): Device to be used for computation (e.g., 'cuda' or 'cpu').
        - ram_pretrained_path (str): Path to the pre-trained RAM model checkpoint.
        - sam_checkpoint_path (str): Path to the SAM model checkpoint.
        - lora_path (str, optional): Path to the LoRA checkpoint file. Default is None.
        """
        self.device = device

        self.objectFinder = ObjectFinder(self.device)
        self.loraModule = LoraRevolver(self.device)

        self.objectFinder._load_models(ram_pretrained_path)
        self.objectFinder._load_sam(sam_checkpoint_path)

        if lora_path != None:
            self.loraModule.load_lora_ckpt_from_file(lora_path, "5x40")

        self.num_objects_stored = 0
        self.memory = [] # store ObjectInfo classes here

    def view_memory(self):
        """
        Prints information about the objects stored in memory.
        """
        print("Objects stored in memory:")
        for info in self.memory:
            print(info)
        print()

    def clear_memory(self):
        """
        Clears the memory by resetting the number of stored objects and the memory dictionary.
        """
        self.num_objects_stored = 0
        self.memory = []

    def _get_object_info(self, image_path, depth_image_path):
        """
        Processes an RGB-D image and depth image to obtain object information.

        Parameters:
        - image_path (str): Path to the PNG file containing the RGB image.
        - depth_image_path (str): Path to the NPY file containing the depth image.

        Returns:
        Tuple containing phrases, embeddings, and point clouds of detected objects.
        """
        if image_path == None or depth_image_path == None:
            raise NotImplementedError

        # segment objects, get (grounded_image bounding boxes, segmentation mask and label) per box
        obj_grounded_imgs, obj_bounding_boxes, obj_masks, obj_phrases = self.objectFinder.find(image_path)

        if obj_grounded_imgs is None:
            return None, None, None
        
        # get ViT+LoRA embeddings, use bounding boxes and the image to get grounded images
        embs = np.array(self.loraModule.encode_image(obj_grounded_imgs).cpu())
        
        # filter out the pointclouds. NOTE: pointclouds are transformed to global pose later.
        obj_pointclouds = self.objectFinder.getDepth(depth_image_path, obj_masks)

        # check that all info recovered
        assert(len(obj_grounded_imgs) == len(obj_bounding_boxes) \
                and len(obj_bounding_boxes) == len(obj_masks) \
                and len(obj_masks) == len(obj_phrases) \
                and len(embs) == len(obj_phrases))

        # can return (obj_grounded_imgs, obj_bounding_boxes) if needed

        return obj_phrases, embs, obj_pointclouds

    def process_image(self, image_path=None, depth_image_path=None, pose=None, verbose=True, add_noise=True,
                      bounding_box_threshold=0.3,  occlusion_overlap_threshold=0.9, testname="", 
                      outlier_removal_config=None, min_points = 500, pose_noise = {'trans': 0.0005, 'rot': 0.0005},
                      depth_noise = 0.003, lora_threshold = 0.5):
        """
        Processes an RGB-D image, detects objects within and updates the object memory.

        Parameters:
        - image_path (str): Path to the PNG file containing the RGB image.
        - depth_image_path (str): Path to the NPY file containing the depth image.
        - pose (list or np.ndarray): Pose information [x, y, z, qw, qx, qy, qz].
        - bounding_box_threshold (float): IoU threshold for bounding boxes. Default is 0.3.
        - occlusion_overlap_threshold (float): Overlap threshold for heavily occluded objects. Default is 0.9.
        - testname (str): Test name for saving point clouds. Default is an empty string.
        - outlier_removal_config (dict, optional): Configuration for outlier removal. Default is None.
        - min_points (int): Minimum number of points under which the object is ignored
        """

        if image_path == None or depth_image_path == None:
            raise NotImplementedError

        # Default outlier removal config
        if outlier_removal_config == None:
            outlier_removal_config = {
                "radius_nb_points": 12,
                "radius": 0.05,
            }

        # Detect all objects within the config
        obj_phrases, embs, obj_pointclouds = self._get_object_info(image_path, depth_image_path)

        if obj_phrases is None:
            if verbose:
                print("No Objects found")
            return
        
        # Outlier removal
        filtered_pointclouds = []
        for points in obj_pointclouds:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.T)
            inlier_pcd, _ = pcd.remove_radius_outlier(nb_points=outlier_removal_config["radius_nb_points"],
                                                            radius=outlier_removal_config["radius"])
            filtered_pointclouds.append(np.asarray(inlier_pcd.points).T)

        if pose is None:
            raise NotImplementedError # TODO: mapping without pose :)
        
        def add_point_noise(array, noise_level):
            noise = np.random.normal(0, noise_level, array.shape)
            noisy_array = array + noise
            return noisy_array
        
        # adding noise to pose
        if add_noise:
            pose[:3] = add_point_noise(pose[:3], pose_noise['trans'])
            pose[3:] = add_point_noise(pose[3:], pose_noise['rot'])

        # normalizing quaternion
        def normalize_quaternion(quaternion):
            norm = np.linalg.norm(quaternion)
            if norm == 0:
                return quaternion
            return quaternion / norm
        pose[3:] = normalize_quaternion(pose[3:])

        transformed_pointclouds = [transform_pcd_to_global_frame(pcd, pose) for pcd in filtered_pointclouds]
        
        # Adding noise to depth (pointclouds)
        if add_noise:
            transformed_pointclouds = [add_point_noise(pcd, depth_noise) for pcd in transformed_pointclouds]

        # for each tuple, consult already stored memory, match tuples to stored memory (based on 3d IoU)
            # TODO optimise and batch, fetch all memory bounding boxes once
            # remove double loop
        if verbose:
            print("Object phrases detected in RGBD input of ObjectMemory.process_image", obj_phrases)
        
        # Loop over every object detected and add to memory
        for i, (obj_phrase, emb, q_pcd) in enumerate(zip(obj_phrases, embs, transformed_pointclouds)):
            if verbose:
                print("\tCurrent Object Phrase under consideration", obj_phrase)

            if q_pcd.shape[-1] < min_points:
                if verbose:
                    print(f"\tObject has {q_pcd.shape[-1]} points which is under {min_points}. Ignored.")
                continue

            obj_exists = False
            count = 0
            for info in self.memory:

                object_pcd = info.pcd

                # replace iou with the objectron implementation
                IoU3d_o = calculate_3d_IoU(q_pcd, object_pcd)
                IoU3d = calculate_obj_aligned_3d_IoU(q_pcd, object_pcd)
                
                if IoU3d_o > 0.:
                    print("Old iou: ", IoU3d_o, IoU3d)

                if IoU3d_o > bounding_box_threshold:
                    # np.save(f"./temp/{count}_qpcd.npy", q_pcd)
                    # np.save(f"./temp/{count}_objectpcd.npy", object_pcd)
                    count += 1

                overlap3d = calculate_strict_overlap(q_pcd, object_pcd)

                if verbose:
                    print("\tFound in mem (info, iou, strict_overlap): ", info, IoU3d, overlap3d)

                # if the iou is above the threshold, consider it to be the same object/instance

                info.computeMeans()

                lora_cos_sim = np.dot(info.mean_emb, emb)/(np.linalg.norm(info.mean_emb) * np.linalg.norm(emb))

                if (IoU3d > bounding_box_threshold or overlap3d > occlusion_overlap_threshold):
                    if lora_cos_sim > 0: # NOTE; not using LoRA here
                        info.addInfo(obj_phrase ,emb, q_pcd, align=False)
                        obj_exists = True
                        break


            # new object detected
            if not obj_exists:
                new_obj_info = ObjectInfo(self.num_objects_stored,
                                            obj_phrase,
                                            emb,
                                            q_pcd)
                
                if verbose:
                    print('\tObject added\n\t\t', obj_phrase, '\n\t\t', new_obj_info, '\n')
                
                self.memory.append(new_obj_info)
                self.num_objects_stored += 1
            else:
                if verbose:
                    print('\tObject exists, aggregated to\n', info, '\n')

        for m in self.memory:
            m.computeMeans()  # Update object info means
        
        # TODO consider downsampling points (optimisation)
                    
    def downsample_all_objects(self, voxel_size = 0.001, use_external_mesh = False):
        for info in self.memory:
            info.downsample(voxel_size, use_external_mesh)
    
    def remove_object_floors(self, floor_thickness=0.1):
        floor_height = 1e8
        for info in self.memory:
            low = np.min(info.pcd[1,:])
            floor_height = min(low, floor_height)

        for info in self.memory:
            info.pcd = (info.pcd.T[(info.pcd[1,:] > floor_height + floor_thickness)]).T

            if len(info.pcd) == 0:
                self.memory.remove(info)
            
            if len(info.pcd[0]) == 0:
                self.memory.remove(info)
                continue

    """
    Runs through all objects stored in memory, consolidates objects that have a sufficient overlap
    Performs uniform downsampling on all pointclouds as well

    Consolidates memory in place
    Thresholds are lower 
    """

    def consolidate_memory(self, bounding_box_threshold=0.2,  occlusion_overlap_threshold=0.6, downsample_voxel_size=0.01, verbose=False):
        if verbose:
            print("Pre consolidation")
            self.view_memory()
        
        new_memory = []
        for obj_id, obj_info in enumerate(self.memory):
            obj_pcd = obj_info.pcd

            # check all objects in new_memory to try and match them
            match_found = False
            for new_id, new_obj_info in enumerate(new_memory):
                IoU3d = calculate_obj_aligned_3d_IoU(new_obj_info.pcd, obj_pcd)
                overlap3d = calculate_strict_overlap(new_obj_info.pcd, obj_pcd)
                if verbose:
                    print(f"{obj_id}, {obj_info.names} -- {new_obj_info.names} | {IoU3d}, {overlap3d}")

                # object overlaps enough to be consolidated with the object in new memory
                if IoU3d > bounding_box_threshold or overlap3d > occlusion_overlap_threshold:
                    match_found = True
                    break
            
            if match_found:
                new_obj_info += obj_info

            else:
                new_memory.append(self.memory[obj_id])


        del self.memory
        self.memory = new_memory

        if verbose:
            print("Post consolidation")
            self.view_memory()



    def localise(self, image_path, depth_image_path, testname="", save_point_clouds=False,
                 outlier_removal_config=None, 
                 fpfh_global_dist_factor = 2, fpfh_local_dist_factor = 0.4, 
                 fpfh_voxel_size = 0.05, topK=5):
        """
        Given an image and a corresponding depth image in an unknown frame, consult the stored memory
        and output a pose in the world frame of the point clouds stored in memory.

        Args:
        - image_path (str): Path to the RGB image file.
        - depth_image_path (str): Path to the depth image file in .npy format.
        - icp_threshold (float): Threshold for ICP (Iterative Closest Point) algorithm.
        - testname (str): Name for test-specific files.

        Returns:
        - np.ndarray: Localized pose in the world frame as [x, y, z, qw, qx, qy, qz].
        """

        # NOTE: removed redundant code - refer to older commits

        # Default outlier removal config
        if outlier_removal_config == None:
            outlier_removal_config = {
                "radius_nb_points": 8,
                "radius": 0.05,
            }

        # Extract all objects currently seen, get embeddings, point clouds in the local unknown frame
        detected_phrases, detected_embs, detected_pointclouds = self._get_object_info(image_path, depth_image_path)

        # Correlate embeddings with objects in memory for all seen objects
        # TODO maybe a KNN search will do better?
        for m in self.memory:
            m.computeMeans()  # Update object info means

        memory_embs = np.array([m.mean_emb for m in self.memory])

        # TODO deal with no objects detected
        if detected_embs is None:
            return np.array([0.,0.,0.,0.,0.,0.,1.]), [[],[]]

        if len(detected_embs) > len(memory_embs):
            detected_embs = detected_embs[:len(memory_embs)]

        detected_embs /= np.linalg.norm(detected_embs, axis=-1, keepdims=True)
        memory_embs /= np.linalg.norm(memory_embs, axis=-1, keepdims=True)

        # Detected x Mem x Emb sized
        cosine_similarities = np.dot(detected_embs, memory_embs.T)

        # Run ICP/FPFH loop closure to get an estimated transform for each seen object
        R_matrices = np.zeros((len(detected_pointclouds), len(memory_embs), 3, 3), dtype=np.float32)
        t_vectors = np.zeros((len(detected_pointclouds), len(memory_embs), 3), dtype=np.float32)

        # Save point clouds
        if save_point_clouds:
            for i, d in enumerate(detected_pointclouds):
                np.save("pcds/%s_detected_pcd" % str(testname) + str(i) + ".npy", d)
            for j, m in enumerate(self.memory):
                np.save(f"pcds/%s_memory_pcd" % str(testname) + str(j) + ".npy", m.pcd)
            print("Point clouds saved")

        # TODO unseen objects in detected objects are not being dealt with, 
        # assuming that all detected objects can be assigned to mem objs
        # TODO calculate rotation matrices
        j = JCBB(cosine_similarities, R_matrices)
        # assns = j.get_candidate_assignments(min_length=max(1, len(detected_embs)-1))
        print("Getting assignments")
        assns = j.get_candidate_assignments(max_length=3)
        del j

        # only consider the top K assignments based on net cosine similarity
        # assns_to_consider = [assn[0] for assn in assns[:topK]]

        assns_to_consider = [assn[0] for assn in assns]

        print("Phrases: ", detected_phrases)
        print(cosine_similarities)
        print("Assignments being considered: ", assns_to_consider)

        assn_data = [ [assn, None, None] for assn in assns_to_consider ]

        # go through all top K assingments, record ICP costs
        for assn_num, assn in tqdm(enumerate(assns_to_consider)):
            # use ALL object pointclouds together
            all_detected_points = []
            all_memory_points = []
            for i,j in assn:
                all_detected_points.append(detected_pointclouds[i])
                all_memory_points.append(self.memory[j].pcd)
            all_detected_points = np.concatenate(all_detected_points, axis=-1)
            all_memory_points = np.concatenate(all_memory_points, axis=-1)

            # centering all the pointclouds
            detected_mean = np.mean(all_detected_points, axis=-1)
            memory_mean = np.mean(all_memory_points, axis=-1)
            all_detected_pcd = o3d.geometry.PointCloud()
            all_detected_pcd.points = o3d.utility.Vector3dVector(all_detected_points.T - detected_mean)
            all_memory_pcd = o3d.geometry.PointCloud()
            all_memory_pcd.points = o3d.utility.Vector3dVector(all_memory_points.T - memory_mean)
            
            # remove outliers from detected pcds
            all_detected_pcd_filtered, _ = all_detected_pcd.remove_radius_outlier(nb_points=outlier_removal_config["radius_nb_points"],
                                                            radius=outlier_removal_config["radius"])

            all_memory_pcd.paint_uniform_color([0,1,0])
            all_detected_pcd_filtered.paint_uniform_color([1,0,0])
            # o3d.io.write_point_cloud(f"./temp/{str(assn)}-{testname}-detmem.ply", all_memory_pcd + all_detected_pcd_filtered)

            transform, rmse = register_point_clouds(all_detected_pcd_filtered, all_memory_pcd, 
                                            voxel_size = fpfh_voxel_size, global_dist_factor = fpfh_global_dist_factor, 
                                            local_dist_factor = fpfh_local_dist_factor)

            assn_data[assn_num] = [assn, transform, rmse]

            # o3d.io.write_point_cloud(f"./temp/{str(assn)}-{testname}-trns.ply", all_memory_pcd + 
            #                         all_detected_pcd_filtered.transform(transform))
            # import pdb; pdb.set_trace()

        best_assn = min(assn_data, key=lambda x: x[-1])

        assn = best_assn[0]
        transform = best_assn[1]

        moved_objs = [n for n in range(len(detected_pointclouds)) if n not in assn]

        R = copy.copy(transform[:3,:3])
        t = copy.copy(transform[:3, 3])
        
        tAvg = t + memory_mean - R@detected_mean
        qAvg = Rotation.from_matrix(R).as_quat()

        localised_pose = np.concatenate((tAvg, qAvg))

        # moved objects will have indices that are not present in the first row of assn

        print(best_assn)
        return localised_pose, [assn, moved_objs]

@dataclass
class LocalArgs:
    """
    Class to hold local configuration arguments.
    """
    lora_path: str='models/vit_finegrained_5x40_procthor.pt'
    test_folder_path: str='/home2/aneesh.chavan/Change_detection/360_zip/'
    device: str='cuda'
    sam_checkpoint_path: str = '/scratch/aneesh.chavan/sam_vit_h_4b8939.pth'
    ram_pretrained_path: str = '/scratch/aneesh.chavan/ram_swin_large_14m.pth'
    mem_save_dir: str = ''
    save_point_clouds: bool = True

if __name__ == "__main__":
    start_time = time.time()

    largs = tyro.cli(LocalArgs, description=__doc__)
    print(largs)

    poses_json_path = os.path.join(largs.test_folder_path, "json_poses.json")

    tgt = []
    pred = []

    print("\nBegin Memory Initialization")
    mem = ObjectMemory(device = largs.device, 
                       ram_pretrained_path=largs.ram_pretrained_path,
                       sam_checkpoint_path = largs.sam_checkpoint_path,
                       lora_path=largs.lora_path)
    print("Memory Init'ed in {} seconds\n".format(time.time() - start_time))

    test_start_time = time.time()

    with open(poses_json_path, 'r') as f:
        poses = json.load(f)

    # tests = [i for i in range(1,9)] # all of them
    # tests = [6,7,8]  # sanity check
    tests = [6, 7]  # sanity check
    for target in tests:
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
                print("This is the target pose")
                continue
            else:
                print("Pose: ", pose)
            
            mem.process_image(testname=f"view%d" % num, 
                              image_path = os.path.join(largs.test_folder_path, f"view%d/view%d.png" % (num, num)), 
                              depth_image_path=os.path.join(largs.test_folder_path,f"view%d/view%d.npy" % (num, num)), 
                              pose=pose,
                              verbose=False)
            print("Processed\n")

        print("Consolidating memory")
        mem.consolidate_memory()
        mem.view_memory()

        estimated_pose, assignment = mem.localise(image_path=os.path.join(largs.test_folder_path,f"view%d/view%d.png" % 
                                                              (target_num, target_num)), 
                                      depth_image_path=(os.path.join(largs.test_folder_path,"view%d/view%d.npy" % 
                                                                     (target_num, target_num))),
                                      save_point_clouds=largs.save_point_clouds,
                                      testname="pose_"+str(target)+"_",)

        print("Target pose: ", target_pose)
        print("Estimated pose: ", estimated_pose)
        print("----\n")

        # saving memory to scratch
        if largs.mem_save_dir != "":
            pcd_list = []
            for obj_id, info in mem.memory.items():
                object_pcd = info.pcd
                pcd_list.append(object_pcd)

            combined_pcd = o3d.geometry.PointCloud()

            for pcd_np in pcd_list:
                pcd_vec = o3d.utility.Vector3dVector(pcd_np.T)
                pcd = o3d.geometry.PointCloud()
                pcd.points = pcd_vec
                combined_pcd += pcd

            os.makedirs(largs.mem_save_dir, exist_ok=True)

            save_path = f"{largs.mem_save_dir}/mem_{target}.pcd"
            o3d.io.write_point_cloud(save_path, combined_pcd)
            print("Pointcloud saved to", save_path)

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

    end_time = time.time()
    test_start_time
    print(f"Memory construction and localisation for {len(tests)} tests done in {(end_time - test_start_time)//60} minutes, {(end_time - start_time)%60} seconds")
    print(f"360zip test completed in {(end_time - start_time)//60} minutes, {(end_time - start_time)%60} seconds")
