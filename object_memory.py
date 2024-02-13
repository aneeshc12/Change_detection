import os, sys

# add gsam, gdino, ram to python path
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "recognize-anything"))

import argparse
import copy

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# recognise anything
from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline


from huggingface_hub import hf_hub_download

import open3d as o3d

import os

# load models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    print()
    print()
    _ = model.eval()
    return model

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL
import os
import random

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

# %%

# ! mkdir -p /scratch/aneesh/
# ! wget -O /scratch/aneesh/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# %%
# loads a base ViT and a set of LoRa configs, allows loading and swapping between them
class LoraRevolver:
    # load base ViT, its preprocessing functions
    def __init__(self, model_checkpoint="google/vit-base-patch16-224-in21k"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # self.base_model will be augmented with a saved set of lora_weights
        # self.lora_model is the augmented model
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

    # load a config into the config library from a saved file
    def load_lora_ckpt_from_file(self, config_path, name):
        ckpt = torch.load(config_path)
        try:
            self.ckpt_library[name] = ckpt
            del self.lora_model
            self.lora_model = self.get_peft_model(self.base_model,
                                                ckpt["lora_config"]).to(self.device)
            self.lora_model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        except:
            print("Lora checkpoint invalid")
            raise IndexError

        self.ckpt_library[str(name): ckpt]
        
    def train_current_lora_model(self):
        pass

    def save_lora_ckpt(self):
        pass

    # use the current lora_model to encode a batch of images
    def encode_image(self, imgs):
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

# detector and segmenter class

from GroundingDINO.groundingdino.util.inference import annotate as gd_annotate 
from GroundingDINO.groundingdino.util.inference import load_image as gd_load_image
from GroundingDINO.groundingdino.util.inference import predict as gd_predict

class ObjectFinder:
    def __init__(self, sam_checkpoint_path='/scratch/aneesh/sam_vit_h_4b8939.pth', box_threshold=0.35, text_threshold=0.55):
        self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # self._load_models(sam_checkpoint_path)


    # loads GROUNDINGDINO
    #       SAM
    #       RAM
    #       
    def _load_models(self):
        # ram
        # TODO

        # grounding dino
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

        cache_config_file = hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        args.device = device
        self.groundingdino_model = build_model(args)

        cache_file = hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_filenmae)
        checkpoint = torch.load(cache_file, map_location=device)
        log = self.groundingdino_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = self.groundingdino_model.eval()

    def _load_sam(self, sam_checkpoint_path):
        # segment anything
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint_path).to(self.device).eval())
        # self.sam_predictor.to(self.device)
        # self.sam_predictor.eval()
        print("SAM loaded")

    def _getIoU(self, rect1, rect2):
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

    def _compSize(sefl, rect1, rect2):
        area_rect1 = rect1[2]*rect1[3]
        area_rect2 = rect2[2]*rect2[3]

        diff = min(area_rect1, area_rect2)/max(area_rect1, area_rect2)
        return diff

    # given a phrase, filter and get all boxes and phrases
    def getBoxes(self, image, text_prompt, show=False, intersection_threshold=0.7, size_threshold=0.75):
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
        with torch.no_grad():
            self.sam_predictor.set_image(image)
            H, W, _ = image.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
                )
            return boxes_xyxy, masks

    # # use the saved model to return grounded images, bounding boxes, masks and phrases
    def find(self, image_path=None, caption=None):
        if type(image_path) == None:
            raise NotImplementedError
        else:
            image_source, image = gd_load_image(image_path)
        
        # get object names
        if caption == None:
            caption = "sofa . chair . table"    # TODO replace with RAM
        
        # ground them, get associated phrases
        cxcy_boxes, phrases = self.getBoxes(image, caption)

        boxes, masks = self.segment(image_source, cxcy_boxes)


        # ground objects
        grounded_objects = [image_source[int(bb[1]):int(bb[3]),
                                         int(bb[0]):int(bb[2]), :] for bb in boxes]

        return grounded_objects, boxes, masks, phrases
    
    # return a 3xN pointcloud corresponding to each object
    # TODO determine whether outliers need to be filtered here
    def getDepth(self, depth_image_path, masks, f=300):
        if depth_image_path == None:
            raise NotImplementedError
        else:
            # TODO convert to torch
            depth_image = np.load(depth_image_path)
            
            w, h = depth_image.shape
            num_objs = masks.shape[0]

            stacked_depth = np.tile(depth_image, (num_objs, 1, 1)) # get all centroids/pcds together
            stacked_depth[masks.squeeze(dim=1).cpu() == False] = 0    # remove the depth channel from the masks

            horizontal_distance = np.tile(np.linspace(-h/2, h/2, h, dtype=np.float32), (num_objs, w,1))
            vertical_distance =   np.tile(np.linspace(w/2, -w/2, w, dtype=np.float32).reshape(-1,1), (num_objs, 1, h))

            X = horizontal_distance * stacked_depth/f
            Y = vertical_distance * stacked_depth/f
            Z = stacked_depth

            # combine caluclated X,Y,Z points
            all_pointclouds = np.stack([X, Y, Z], 1).reshape((num_objs, 3, -1))

            # filter out [0,0,0]
            all_pointclouds = [pcd[:, pcd[2, :] != 0] for pcd in all_pointclouds]
            
            return all_pointclouds


    def _show_detections(self, image_path=None, caption=None):
        if type(image_path) == None:
            raise NotImplementedError
        else:
            image_source, image = gd_load_image(image_path)

        ## TODO implement RAM
        if caption==None:
            caption = "sofa . chair . table"

        Image.fromarray(image_source)
        b, l, p = gd_predict(model=self.groundingdino_model, 
                                           image=image, caption="sofa . chair . table",
                                           box_threshold=0.35,
                                           text_threshold=0.55)
        af = gd_annotate(image_source=image_source, boxes=b, logits=l, phrases=p)[...,::-1]
        Image.fromarray(af)
        plt.imshow(af)

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import torch.nn.functional as F

from jcbb import JCBB

# Utility functions
"""
Given a point cloud and an [x y z qw qx qy qz] pose for the camera frame wrt
world frame, 
transform a pcd into the world frame
"""
def transform_pcd_to_global_frame(pcd, pose):
    t = pose[:3]
    q = pose[3:]

    q /= np.linalg.norm(q)                  # normalise
    R = Rotation.from_quat(q).as_matrix()

    transformed_pcd = R @ pcd
    transformed_pcd += t.reshape(3, 1)

    return transformed_pcd

"""
Assume 3xN pcds, 
"""
def calculate_3d_IoU(pcd1, pcd2):
    # get [min_X, min_Y, min_Z, max_X, max_Y, max_Z] for both pcds
    bb1_min = pcd1.min(axis=-1)
    bb1_max = pcd1.max(axis=-1)

    bb2_min = pcd2.min(axis=-1)
    bb2_max = pcd2.max(axis=-1)

    overlap_min_corner = np.stack([bb1_min, bb2_min], axis=0).max(axis=0)
    overlap_max_corner = np.stack([bb1_max, bb2_max], axis=0).min(axis=0)

    # no overlap case
    if (overlap_min_corner > overlap_max_corner).any():
        return 0
    else:
        v = overlap_max_corner - overlap_min_corner
        overlap_volume = v[0] * v[1] * v[2]
        
        bb1 = bb1_max - bb1_min
        bb2 = bb2_max - bb2_min

        v1 = bb1[0] * bb1[1] * bb1[2]
        v2 = bb2[0] * bb2[1] * bb2[2]

        v = overlap_volume/(v1 + v2 - overlap_volume)

        return v

def calculate_strict_overlap(pcd1, pcd2):
    # get [min_X, min_Y, min_Z, max_X, max_Y, max_Z] for both pcds
    bb1_min = pcd1.min(axis=-1)
    bb1_max = pcd1.max(axis=-1)

    bb2_min = pcd2.min(axis=-1)
    bb2_max = pcd2.max(axis=-1)

    overlap_min_corner = np.stack([bb1_min, bb2_min], axis=0).max(axis=0)
    overlap_max_corner = np.stack([bb1_max, bb2_max], axis=0).min(axis=0)

    # no overlap case
    if (overlap_min_corner > overlap_max_corner).any():
        return 0
    else:
        v = overlap_max_corner - overlap_min_corner
        overlap_volume = v[0] * v[1] * v[2]
        
        bb1 = bb1_max - bb1_min
        bb2 = bb2_max - bb2_min

        v1 = bb1[0] * bb1[1] * bb1[2]
        v2 = bb2[0] * bb2[1] * bb2[2]

        v = overlap_volume/(min(v1,v2))

        return v

# Classes
"""
Bundles together object information for distinct objects
"""
class ObjectInfo:
    def __init__(self, id, name, emb, pcd):
        self.id = id
        self.names = [name]
        self.embeddings = [emb]
        self.pcd = pcd
        # self.centroid = np.empty(3, np.float64)

        self.mean_emb = None
        self.centroid = None

    def addInfo(self, name, embedding, pcd):
        if name not in self.names:
            self.names.append(name)
        self.embeddings.append(embedding)
        self.pcd = np.concatenate([self.pcd, pcd], axis=-1)

    def computeMeans(self):
        # TODO messy, clean this up
        self.mean_emb = np.mean(np.asarray(
            [e.cpu() for e in self.embeddings]), axis=0)
        self.centroid = np.mean(self.pcd, axis=-1)

    def __repr__(self):
        return(f"ID: %d | Names: [%s] |  Num embs: %d | Pcd size: " % \
              (self.id, " ".join(self.names), len(self.embeddings)) + str(self.pcd.shape))

# main object recorder class

# stores all object information, point clouds, embeddings etc
"""

"""
class ObjectMemory:
    def __init__(self, lora_path=None):
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'

        self.objectFinder = ObjectFinder()
        self.loraModule = LoraRevolver()

        self.objectFinder._load_models()
        self.objectFinder._load_sam('/scratch/aneesh/sam_vit_h_4b8939.pth')

        if lora_path != None:
            self.loraModule.load_lora_ckpt_from_file(lora_path)

        self.num_objects_stored = 0
        self.memory = dict() # store ObjectInfo classes here

        return

    # visualisation and utility
    def view_memory(self):
        print("Objects stored in memory:")
        for _, info in self.memory.items():
            print(info.names)
            print(info)
        print()


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
            obj_grounded_imgs, obj_bounding_boxes, obj_masks, obj_phrases = self.objectFinder.find(image_path, caption="sofa . chair. table")
            
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
            
            for obj_phrase, emb, q_pcd in zip(obj_phrases, embs, transformed_pointclouds):
                obj_exists = False

                # np.save("pcds/" + testname + "_" + obj_phrase + ".npy", q_pcd)

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
    Given an image and a corresponding depth image in an unknown frame, consult the stored memory
    and output a pose in the world frame of the pcds stored in memory

    pose returned as [x, y, z, qw, qx, qy, qz]
    """
    def localise(self, image_path, depth_image_path, icp_threshold=.2):
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

            assert(len(detected_embs) <= len(memory_embs))

            # Detected x Mem x Emb sized
            cosine_similarities = F.cosine_similarity(detected_embs.unsqueeze(1), memory_embs.unsqueeze(0), axis=-1).cpu()

            # TODO optimisations may be possible
            # run ICP/FPFH loop closure to get an estimated transform for each seen object
            R_matrices = np.zeros((len(detected_pointclouds), len(memory_embs), 3, 3), dtype=np.float32)
            t_vectors = np.zeros((len(detected_pointclouds), len(memory_embs), 3), dtype=np.float32)


            # save pcds
            for i, d in enumerate(detected_pointclouds): np.save("pcds/detected_pcd" + str(i) + ".npy", d)
            for j, (_, m) in enumerate(self.memory.items()): np.save("pcds/memory_pcd" + str(j) + ".npy", m.pcd)
            print("Pcds saved")

            detected_pcd = o3d.geometry.PointCloud()
            memory_pcd = o3d.geometry.PointCloud()
            # use centered pcds to get a better ICP initialisation
            for i, d in enumerate(detected_pointclouds):
                detected_mean = np.mean(d, axis=-1)
                detected_pcd.points = o3d.utility.Vector3dVector(d.T - detected_mean)

                for j, (_, m) in enumerate(self.memory.items()):
                    memory_mean = np.mean(m.pcd, axis=-1)
                    memory_pcd.points = o3d.utility.Vector3dVector(m.pcd.T - memory_mean)

                    # voxel downsample for equal point density
                    registration = o3d.pipelines.registration.registration_icp(
                        detected_pcd.voxel_down_sample(0.05), 
                        memory_pcd.voxel_down_sample(0.05),
                        icp_threshold,
                        np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
                    )

                    transform = registration.transformation
                    R = transform[:3,:3]
                    t = transform[:3, 3]
                    
                    R_matrices[i, j] = R
                    t_vectors[i, j] = t
                    
                    # adjust transformation to account for centered pcds
                    # M = [R|t]D + (mean_M - R @ mean_D)
                    t_vectors[i,j] += memory_mean - R@detected_mean

            # jcbb association using lora embeddings, estimated rotation/transform for each object to encode cost
                    
            # precalculation done, begin assigning objects
            # TODO unseen objects in detected objects are not being dealt with, assuming that all detected objects can be assigned to mem objs
            j = JCBB(cosine_similarities, R_matrices)
            assns = j.get_assignments()

            # calculate paths for all assingments, pick the best one
            best_assignment = assns[0]
            min_cost = 1e11
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
            
            # using https://math.stackexchange.com/questions/61146/averaging-quaternions direct/fast averaging
            qAvg = np.zeros(4)
            q0 = None
            tAvg = np.zeros(3)

            print(best_assignment)

            for i,j in enumerate(best_assignment):
                # roughly avg rotation
                q = Rot.from_matrix(R_matrices[i,j])
                q = q.as_quat()

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
            return localised_pose

# Main func
from scipy.spatial.transform import Rotation as Rot
import json

if __name__ == "__main__":
    print("Begin")
    mem = ObjectMemory()
    print("Memory Init'ed")

    # q = R.from_euler('zyx', [0, 135, 0], degrees=True).as_quat()
    # t = np.array([-4.5, 0.9, 6.25, ])
    # pose = np.concatenate([t, q])
    # print("pose: ", pose)

    # print("Processing img 2")
    # mem.process_image(testname="view2", image_path='360_zip/view2/view2.png', depth_image_path='360_zip/view2/view2.npy', pose=pose)
    # print("Processed image\n")

    # q = R.from_euler('zyx', [0, 90, 0], degrees=True).as_quat()
    # t = np.array([-4.5, 0.9, 3.25, ])
    # pose = np.concatenate([t, q])

    # print("Processing img 3")
    # mem.process_image(testname="view3", image_path='360_zip/view3/view3.png', depth_image_path='360_zip/view3/view3.npy', pose=pose)
    # print("Processed image\n")

    with open('/home2/aneesh.chavan/Change_detection/360_zip/json_poses.json', 'r') as f:
        poses = json.load(f)

    target_pose = None
    for i, view in enumerate(poses["views"]):
        num = i+1

        # view 6 is our unseen view, skip

        print(f"Processing img %d" % num)
        q = Rot.from_euler('zyx', [r for _, r in view["rotation"].items()], degrees=True).as_quat()
        t = np.array([x for _, x in view["position"].items()])
        pose = np.concatenate([t, q])
        if num == 6:
            target_pose = pose
            continue
        else:
            print("Pose: ", pose)

        

        
        mem.process_image(testname=f"view%d" % num, image_path=f"360_zip/view%d/view%d.png" % (num, num), depth_image_path=f"360_zip/view%d/view%d.npy" % (num, num), pose=pose)
        print("Processed\n")

    mem.view_memory()

    # localise image 6
    target_num = 6
    estimated_pose = mem.localise(image_path=f"360_zip/view%d/view%d.png" % (target_num, target_num), depth_image_path=f"360_zip/view%d/view%d.npy" % (target_num, target_num))

    print("Target pose: ", target_pose)
    print("Estimated pose: ", estimated_pose)

    # for _, m in mem.memory.items():
    #     np.save(f"pcds/new%d.npy" % m.id, m.pcd)