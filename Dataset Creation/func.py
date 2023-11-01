import os, sys, time

sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "recognize-anything"))

import argparse
import copy

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont

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

from torch import nn
import torch.nn.functional as F
import einops
import torchvision.transforms as T

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from huggingface_hub import hf_hub_download

# detect object using grounding DINO
def detect(image, text_prompt, model, image_source=None, box_threshold = 0.3, text_threshold = 0.25, remove_combined=False):
  boxes, logits, phrases = predict(
      model=model,
      image=image,
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )

  if type(image_source) == None:
    annotated_frame = None
  else:
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

  return annotated_frame, boxes


# return a tensor containing all bounding boxes
# filter for duplicates by calculating
# [TODO] speedup

def getIoU(rect1, rect2):
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

def compSize(rect1, rect2):
  area_rect1 = rect1[2]*rect1[3]
  area_rect2 = rect2[2]*rect2[3]

  diff = min(area_rect1, area_rect2)/max(area_rect1, area_rect2)
  return diff


def getAllDetectedBoxes(image, groundingdino_model, image_source=None, keywords=[], show=False, intersection_threshold=0.7, size_threshold=0.75):
  total_time = 0

  with torch.no_grad():
    boxes = []
    unique_boxes_num = 0

    for i, word in enumerate(keywords):
      af, detected = detect(image, image_source=image_source, text_prompt=str(word), model=groundingdino_model)
      cnt_time = time.time()

      if show:
        print(i)
      unique_enough = True

      if detected != None and len(detected) != 0:
        if unique_boxes_num == 0:
          for box in detected:
            boxes.append(box)
            unique_boxes_num += 1

          if show and type(image_source) != None:
            Image.fromarray(af).show()

            if show:
              print("detected", detected)

        else:
          print("boxes: ", boxes)
          for box in detected:
            unique_enough = True

            if show:
              print("detected: ", detected)

            for prev in boxes[:unique_boxes_num]:

              iou = getIoU(box, prev)
              diff = compSize(box, prev)

              if show:
                print("comparing; -- ", prev, box)
                print("iou: ", iou)
                print("diff: ", diff)

              if (iou > intersection_threshold and diff > size_threshold):
                # bounding box is not unique enough to be added
                unique_enough = False

                if show:
                  print("failed")
                break

            if unique_enough:
              boxes.append(box)
              unique_boxes_num += 1

              if show:
                print("         success!")
                print(boxes)

          if show and type(image_source) != None:
            Image.fromarray(af).show()

      total_time += (time.time() - cnt_time)

    print(total_time)
    return torch.stack(boxes)

def decide_uniqueness(candidate_boxes, stored_boxes, intersection_threshold=0.7, size_threshold=0.75):
  # get area difference
  candidate_areas = 4 * candidate_boxes[:,2] * candidate_boxes[:,3]
  stored_areas = 4 * stored_boxes[:,2] * stored_boxes[:,3]
  minimum_areas = np.minimum(candidate_areas.unsqueeze(1), stored_areas)

  area_diff = candidate_areas.unsqueeze(1)/stored_areas
  area_diff[area_diff >= 1.] = 1/area_diff[area_diff >= 1.]

  conv_cb = candidate_boxes.clone()
  conv_sb = stored_boxes.clone()

  conv_cb[:, :2] -= conv_cb[:, 2:]
  conv_cb[:, 2:] = 2 * conv_cb[:, 2:] + conv_cb[:, :2]
  conv_cb = np.expand_dims(conv_cb, axis=1)

  conv_sb[:, :2] -= conv_sb[:, 2:]
  conv_sb[:, 2:] = 2 * conv_sb[:, 2:] + conv_sb[:, :2]

  overlap_boxes = np.concatenate([np.maximum(conv_cb[...,:2], conv_sb[...,:2]),
                                  np.minimum(conv_cb[...,2:], conv_sb[...,2:])],
                                 axis=-1)

  iou = np.where(np.logical_and((overlap_boxes[..., 2] > overlap_boxes[..., 0]), (overlap_boxes[..., 3] > overlap_boxes[..., 1])),
                 (overlap_boxes[..., 3] - overlap_boxes[..., 1]) * (overlap_boxes[..., 2] - overlap_boxes[..., 0])/minimum_areas,
                 -np.inf)

  boxes_comparison = np.where(
      np.logical_and(np.logical_and(iou > intersection_threshold, area_diff > size_threshold), iou != -np.inf),
      False,
      True
  )

  unique_enough = np.logical_and.reduce(boxes_comparison, 1)

  return unique_enough


def eff_getAllDetectedBoxes(image, image_source=None, keywords=[], show=False, intersection_threshold=0.7, size_threshold=0.75):
  with torch.no_grad():
    boxes = None
    unique_boxes_num = 0

    total_time = 0

    for i, word in enumerate(keywords):
      af, detected = detect(image, image_source=image_source, text_prompt=str(word), model=groundingdino_model)

      cnt_time = time.time()

      if show:
        print(i)
      # unique_enough = True

      # sort through all detected boxes, add them if there is little enough overlap with all recorded bboxes, or it is small enough for overlap to not matter
      if detected != None and len(detected) != 0:
        if boxes == None:
          boxes = detected

          if show and type(image_source) != None:
            Image.fromarray(af).show()

            if show:
              print("detected", detected)

        else:
          if show:
            print("boxes:\n", boxes)

            if type(image_source) != None:
              Image.fromarray(af).show()


          unique_enough = decide_uniqueness(detected, boxes)
          boxes = torch.concat([boxes] + [detected[num].unsqueeze(0) for num, val in enumerate(unique_enough) if val])

          if show:
            for i, k in enumerate(unique_enough):
              print("Added " if k else "Failed ", sep='')
              print(detected[i])


          total_time += (time.time() - cnt_time)

    print(total_time)
    return boxes

def segment(image, sam_model, boxes, device):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return boxes_xyxy, masks.cpu()


def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


# supply an image and all keywords, returns bounding box cropped
def getAllSegmentedImages(image_path, keywords, show=False):
  image_source, image = load_image(image_path)
  cropped_images = []

  detected_boxes = getAllDetectedBoxes(image, keywords=keywords, show=True)
  bboxs, segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes, device=device)

  bboxs = bboxs.numpy().astype(int)
  for idx in range(detected_boxes.shape[0]):
    cropped_images.append(image_source[bboxs[idx][1]:bboxs[idx][3], bboxs[idx][0]:bboxs[idx][2]])

    Image.fromarray(image_source[bboxs[idx][1]:bboxs[idx][3], bboxs[idx][0]:bboxs[idx][2]]).show()

  return cropped_images

# supply an image and all keywords, returns bounding box cropped
def objectDetectionPipeline(image_path, ram_model, groundingdino_model, sam_predictor, show=False, device='cpu'):
  # get keywords
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ram_transform = get_transform(image_size=384)
  ram_image = ram_transform(Image.open(image_path)).unsqueeze(0).to(device)

  res = inference(ram_image, ram_model)
  keywords = res[0].split(' | ')

  print(keywords)

  keywords = [i for i in keywords if i not in ['room', 'living room', 'window', 'ceiling', 'curtain', 'curtains']]

  # load images for dino and sam
  image_source, image = load_image(image_path)
  cropped_images = []

  detected_boxes = getAllDetectedBoxes(image, groundingdino_model, image_source=image_source, keywords=keywords, show=show)
  bboxs, segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes, device=device)

  if len(bboxs) != len(detected_boxes):
    print("\n\n\n\nMISSING STUFF\nKHDSFKJHDSKFJLHDSKLJFHKLSDFH\nKHDSFKJHDSKFJLHDSKLJFHKLSDFH\n\n\n\n")

  H, W = image_source.shape[:2]
  detected_boxes = detected_boxes * torch.Tensor([W, H, W, H])
  # from xywh to xyxy
  detected_boxes[:, :2] -= detected_boxes[:, 2:] / 2
  detected_boxes[:, 2:] += detected_boxes[:, :2]

  detected_boxes = detected_boxes.numpy().astype(int)

  bboxs = bboxs.numpy().astype(int)
  for idx in range(detected_boxes.shape[0]):
    cropped_images.append(image_source[bboxs[idx][1]:bboxs[idx][3], bboxs[idx][0]:bboxs[idx][2]])

  return cropped_images, bboxs