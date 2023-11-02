from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, AutoProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, AutoImageProcessor, ViTConfig, ViTModel, ViTForImageClassification
from peft import LoraConfig, get_peft_model
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# vit
model_checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

# create dataloaders to create and store triplets
class ObjectTriplets(Dataset):
  def __init__(self, dataset_path, transforms, num_triplets_per_class=None, difficult_triplet_percentage=None):
    if transforms == None:
      self.transform = ToTensor()
    else:
      self.transform = transforms

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.triplets = self.get_triplets(dataset_path, num_triplets_per_class=num_triplets_per_class)

  def pick_random(self, arr, val):
    if not arr:
      return None

    k = val
    while k == val:
      k = random.choice(arr)
    return k

  # no negative mining so far, address class balance by creating a fixed number of triplets for each class
  def get_triplets(self, dataset_path, num_triplets_per_class=None, difficult_triplet_percentage=None):
    triplets = []

    categories = os.listdir(dataset_path)
    classes = []

    for ctg in categories:
      for i in os.listdir(os.path.join(dataset_path, ctg)):
        classes.append(ctg + '/' + i)

    print(classes)

    image_paths = [[os.path.join(
                        dataset_path,
                        c,
                        n,
                    ) for n in os.listdir(os.path.join(dataset_path, c))]
                   for c in classes]
    random.shuffle(image_paths)

    if num_triplets_per_class == None:
      self.num_triplets_per_class = max([len(c) for c in image_paths])
    else:
      self.num_triplets_per_class = num_triplets_per_class

    with tqdm(total = self.num_triplets_per_class * len(classes)) as pbar:
      for cidx, c in enumerate(classes):

        coarse_ctg = c[:c.find('/')]
        easy_classes  = [i for i in range(len(classes)) if (i != cidx and coarse_ctg != classes[i][:classes[i].find('/')])]
        tough_classes = [i for i in range(len(classes)) if (i != cidx and coarse_ctg == classes[i][:classes[i].find('/')])]

        other_classes = [i for i in range(len(classes)) if i != cidx]
        for i in range(self.num_triplets_per_class):
          if i >= len(image_paths[cidx]):
            idx = random.randint(0, len(image_paths[cidx]) - 1)
          else:
            idx = i

          anchor_path = image_paths[cidx][idx]
          positive_path = self.pick_random(image_paths[cidx],
                                            image_paths[cidx][idx])


          if difficult_triplet_percentage == None:
            negative_path = self.pick_random(image_paths[random.choice(other_classes)],
                                              image_paths[cidx][idx])
          else:
            if i < int(difficult_triplet_percentage * self.num_triplets_per_class):
              negative_path = self.pick_random(image_paths[random.choice(tough_classes)], image_paths[cidx][idx])
            else:
              negative_path = self.pick_random(image_paths[random.choice(easy_classes)], image_paths[cidx][idx])

          triplet = [
              self.transform(PIL.Image.open(anchor_path)),
              self.transform(PIL.Image.open(positive_path)),
              self.transform(PIL.Image.open(negative_path))
          ]

          triplet = torch.stack(triplet)

          triplets.append(triplet)
          pbar.update(1)

    return triplets

  def __len__(self):
    return len(self.triplets)

  ##
  def __getitem__(self, idx):
    return self.triplets[idx]

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

test_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

# pretrained VIT
pretrained_model = ViTModel.from_pretrained(
    model_checkpoint,
    ignore_mismatched_sizes=True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# LoRA Config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,

    # vit
    target_modules=["query", "value"],

    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(pretrained_model, lora_config).to(device)