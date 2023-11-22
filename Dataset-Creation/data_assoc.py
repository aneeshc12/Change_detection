import torch
from func import *
from huggingface_hub import hf_hub_download

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict

import shutil
from func import objectDetectionPipeline

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

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
  
def chooseCroppedImages(img1_cropped, img2_cropped, choice1, choice2, show=True):
  img_array_1 = []
  c = 0
  for i in choice1:
    img_array_1.append(img1_cropped[c][i])
    c+=1

  img_array_2 = []
  c = 0
  for i in choice2:
    img_array_2.append(img2_cropped[c][i])
    c+=1

  if show:
    fig1, axs1 = plt.subplots(1, len(img_array_1), figsize=(2*len(img_array_1), 2))
    # if len(choice1) == 1:
    for i, img in enumerate(img_array_1):
      axs1[i].set_title(str(i))
      axs1[i].imshow(img)
      # print(bboxes1[i])
    plt.subplots_adjust(wspace=0.4)
    plt.show()

    fig2, axs2 = plt.subplots(1, len(img_array_2), figsize=(2*len(img_array_2), 2))
    # if len(choice2) == 1:
    for i, img in enumerate(img_array_2):
      axs2[i].set_title(str(i))
      axs2[i].imshow(img)
    plt.subplots_adjust(wspace=0.4)
    plt.show()

  return img_array_1, img_array_2

# filenames and filepaths
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

# load Grounding DINO model
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

# ! wget https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth
# load RAM model
ram_model = ram(pretrained='/scratch/sarthak.chittawar/checkpoint/ram_swin_large_14m.pth', image_size=384, vit='swin_l')
ram_model.eval()
ram_model.to(device)

# ! wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

sam_checkpoint = '/scratch/sarthak.chittawar/checkpoint/sam_vit_h_4b8939.pth'

# load SAM predictor model
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

x = os.listdir('/scratch/sarthak.chittawar/dataset')
try:
  shutil.rmtree('./Dataset')
except:
  pass
os.mkdir("./Dataset")
for orientation in sorted(x, key=lambda x: int(x[len('orientation'):])):
  # Loading the AI2-THOR images from drive onto local memory
  img_cropped_images = [0 for i in range(8)]
  bboxes = [0 for i in range(8)]
  for i in range(8):
    img_cropped_images[i], bboxes[i] = objectDetectionPipeline("/scratch/sarthak.chittawar/dataset/{}/view{}/view{}.png".format(orientation, i+1, i+1), ram_model, groundingdino_model, sam_predictor, False, device)

  try:
    os.mkdir("./Dataset/{}".format(orientation))
  except:
    pass

  fig = [0 for _ in range(8)]
  axs = [0 for _ in range(8)]
  c = 1
  for i in range(8):
    os.mkdir("./Dataset/{}/view{}".format(orientation, i+1))
    fig[i], axs[i] = plt.subplots(1, len(img_cropped_images[i]), figsize=(2*len(img_cropped_images[i]), 2))
    for j, img in enumerate(img_cropped_images[i]):
      # axs[i][j] = plt.subplot(8, len(img_cropped_images[i]), c)
      axs[i][j].imshow(img)
      c += 1
      Image.fromarray(img).save("./Dataset/{}/view{}/{}.png".format(orientation, i+1, j+1))
    plt.subplots_adjust(wspace=0.4)
    print()
  plt.show()