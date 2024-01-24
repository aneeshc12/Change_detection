# %% [markdown]
# # Run through one sequence, use the sequence class

# %%
!mkdir -p /scratch/aneesh

# %%
import os, sys

sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "recognize-anything"))

print(os.getcwd(), os.path.join(os.getcwd(), "Grounded-Segment-Anything", "GroundingDINO"))
print(sys.path)

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

import os

# %%
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
    _ = model.eval()
    return model

# %%
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"


groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

# %%
import os

# if not os.path.isfile("/scratch/aneesh/ram_swin_large_14m.pth"):
# !wget -O /scratch/aneesh/ram_swin_large_14m.pth https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth

# r = ram(image_size=384, vit='swin_l')
ram_model = ram(pretrained='/scratch/aneesh/ram_swin_large_14m.pth', image_size=384, vit='swin_l')
# ram_model.eval()
# ram_model.to(device)

# %% [markdown]
# 

# %%
import matplotlib.pyplot as plt

test_img_path = '/home2/aneesh.chavan/Change_detection/360_zip/view2/view2.png'
transform = get_transform(image_size=384)
image = transform(Image.open(test_img_path)).unsqueeze(0).to(device)

res = inference(image, ram_model)
print("Image Tags: ", res[0])

plt.imshow(Image.open(test_img_path));


# %%
! wget -O /scratch/aneesh/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

sam_checkpoint = '/scratch/aneesh/sam_vit_h_4b8939.pth'

sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

# %% [markdown]
# ### Grounding DINO for detection

# %%
# detect object using grounding DINO
def detect(image, text_prompt, model, image_source=None, box_threshold = 0.35, text_threshold = 0.55, remove_combined=False):
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

# %%
# download_image(image_url, local_image_path)
test_img_path = '/home2/aneesh.chavan/Change_detection/360_zip/view2/view2.png'
image_source, image = load_image(test_img_path)
Image.fromarray(image_source)

annotated_frame, detected_boxes = detect(image, text_prompt="sofa . chair . table",
                                         model=groundingdino_model,
                                         image_source=image_source)
Image.fromarray(annotated_frame)

# %%
# object retrieval functions

# return a tensor containing all bounding boxes
# filter for duplicates by calculating

# [TODO] speedup

import time

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

def getAllDetectedBoxes(image, image_source=None, keywords=[], show=False, intersection_threshold=0.7, size_threshold=0.75):
  total_time = 0

  with torch.no_grad():
    boxes = []
    unique_boxes_num = 0

    for i, word in enumerate(keywords):
      af, detected = detect(image, image_source=image_source, text_prompt=str(word), model=groundingdino_model)

      cnt_time = time.time()

      # # limit edges
      # for d in detected:
      #   if d[0] + d[2] >= 1:
      #     d[2] = 1 - d[0]

      #   if d[1] + d[3] >= 1:
      #     d[3] = 1 - d[1]

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
            plt.imshow(af)

      total_time += (time.time() - cnt_time)

    # print(total_time)
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
  
# segmentation code

# THERE IS SPACE TO BATCH SEGMENTATIONS

def segment(image, sam_model, boxes):
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


# %%
%matplotlib inline

# %%
detected_boxes = getAllDetectedBoxes(image, image_source, [l for l in "sofa | chair | table".split('|')],
                                     show=False)

# %%
bboxs, masks = segment(image_source, sam_predictor, boxes=detected_boxes)

for idx in range(masks.shape[0]):
  annotated_frame_with_mask = draw_mask(masks[idx][0], annotated_frame)
  plt.figure()
  plt.imshow(annotated_frame_with_mask)
#   Image.fromarray(annotated_frame_with_mask).show()

# %%
bboxs

# %%
masks.shape

# %%
mt = np.copy(image_source)
mt[masks[0,0] == False] = 0
plt.imshow(mt[290:385, 370:500,  :])
plt.figure()
plt.imshow(mt)

# %%
depth_img = np.load("360_zip/view2/view2.npy")

plt.imshow(depth_img, cmap='gray')
plt.colorbar()  # Adds a colorbar to show the depth values
plt.title("Depth Image")
plt.show()

# print(depth_img, np.max(depth_img), np.min(depth_img))



# %%
depth_test = np.copy(depth_img)
depth_test[(masks[0] == False).squeeze()] = 0
plt.imshow(depth_test, cmap='gray')
plt.show()

# %%
a = [1,2,3]
b = [1,2,3]
c = [1,2,3]
d = [1,2,3]
e = [1,2,3]

x = [[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]]
y = [[2,2,2,2,2], [2,2,2,2,2], [2,2,2,2,2], [2,2,2,2,2], [2,2,2,2,2]]
z = [[4,4,4,4,4], [4,4,4,4,4], [4,4,4,4,4], [4,4,4,4,4], [4,4,4,4,4]]

np.stack([x,y,z]).reshape(3, -1).shape

# %%
f = 300

w, h = depth_test.shape

row_wise = np.tile(np.linspace(-h/2, h/2, h, dtype=np.float32), (w, 1))
col_wise = np.tile(np.linspace(w/2, -w/2, w, dtype=np.float32).reshape(1,-1).T, (1, h))

X = row_wise * depth_img/f
Y = col_wise * depth_img/f
Z = depth_img

# zeroth object centroid
centroid0 = np.array([
    np.where(masks[0] == True, X, 0).sum(),
    np.where(masks[0] == True, Y, 0).sum(),
    np.where(masks[0] == True, depth_img, 0).sum()
]) /np.where(masks[0] == True, 1, 0).sum()

centroid1 = np.array([
    np.where(masks[1] == True, X, 0).sum(),
    np.where(masks[1] == True, Y, 0).sum(),
    np.where(masks[1] == True, depth_img, 0).sum()
]) /np.where(masks[1] == True, 1, 0).sum()

centroid2 = np.array([
    np.where(masks[2] == True, X, 0).sum(),
    np.where(masks[2] == True, Y, 0).sum(),
    np.where(masks[2] == True, depth_img, 0).sum()
]) /np.where(masks[2] == True, 1, 0).sum()

# for i in range(w):
#   for h in range(h):
#     # X = z*x/f

X1 = np.where(masks[0] == True, X, 0).reshape(600,600)
Y1 = np.where(masks[0] == True, Y, 0).reshape(600,600)
Z1 = np.where(masks[0] == True, depth_img, 0).reshape(600,600)

# pcd1 = np.stack([X1, Y1, Z1]).reshape(3, -1).T
pcd1 = np.stack([X, Y, Z]).reshape(3, -1).T
print(pcd1.shape)

print(pcd1)

# pcd1 = pcd1[np.logical_and(pcd1[:,0] != 0. , pcd1[:,1] != 0. , pcd1[:,2] != 0.)]
# print(pcd1.shape)

# plt.figure()
# plt.title('X')
# plt.imshow(X1[:,200:])
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.title('Y')
# plt.imshow(Y1[:,200:])
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.title('Z')
# plt.imshow(Z1[:,200:])
# plt.colorbar()
# plt.show()


# %%
import open3d as o3d
import plotly.graph_objects as go


# %%
pcd1_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd1))


fig = go.Figure(
    data=[
        go.Scatter3d(
            x=pcd1[:,0], y=pcd1[:,1], z=pcd1[:,2], 
            mode='markers',
            marker=dict(size=1)
        )
    ],
    layout=dict(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
)
fig.show()


