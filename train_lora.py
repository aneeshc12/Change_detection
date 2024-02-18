# %% [markdown]
# # LoRa on finegrained data

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# !mkdir -p /scratch/data/
# !cp /content/drive/MyDrive/Datasets/finegrained_dataset_mk2.zip /scratch/data
# !cp /content/drive/MyDrive/Datasets/multisequence_lora_with_depth.zip /scratch/data

# !cp /content/drive/MyDrive/Datasets/objaverse_dataset.zip /scratch/data

# %%
# !unzip /scratch/data/finegrained_dataset_mk2.zip -d /scratch;

# %%
# !pip install transformers accelerate evaluate datasets peft -q
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, AutoImageProcessor

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
)

from tqdm import tqdm
from transformers import AutoImageProcessor, CLIPVisionModel

# %%
### HYPERPARAMETERS

hyp = {
    "hard_triplet_percent": 0.5,
    "ignore_classes": True,
    "ignore_start": 40,
    "exp_name": "vit_finegrained_5x40"
}

hyp["ignored_classes"] = \
    [f"armchairs/armchair%d" % i for i in range(hyp["ignore_start"],60)] + \
    [f"beds/bed%d" % i for i in range(hyp["ignore_start"],60)] + \
    [f"chairs/chair%d" % i for i in range(hyp["ignore_start"],60)] + \
    [f"coffee_tables/coffee_table%d" % i for i in range(hyp["ignore_start"],60)] + \
    [f"dining_tables/dining_table%d" % i for i in range(hyp["ignore_start"],60)] + \
    [f"sofas/sofa%d" % i for i in range(hyp["ignore_start"],60)]

hyp["ignored_classes"] = []

# %%
# vit

model_checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

# clip

# model_checkpoint = "openai/clip-vit-base-patch32"
# image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# %%
#create dataloaders to create and store triplets

class ObjectTriplets(Dataset):
  def __init__(self, dataset_path, transforms, num_triplets_per_class=None, difficult_triplet_percentage=None):
    if transforms == None:
      self.transform = ToTensor()
    else:
      self.transform = transforms

    # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.device = 'cpu'
    self.triplets = self.get_triplets(dataset_path, num_triplets_per_class=num_triplets_per_class)

  def pick_random(self, arr, val):
    if not arr:
      return None

    k = val
    while k == val:
      k = random.choice(arr)
      # print(k,val)
    return k

  # no negative mining so far, address class balance by creating a fixed number of triplets for each class
  def get_triplets(self, dataset_path, num_triplets_per_class=None, difficult_triplet_percentage=None):
    triplets = []

    categories = os.listdir(dataset_path)
    classes = []

    for ctg in categories:
      for i in os.listdir(os.path.join(dataset_path, ctg)):
        if (not hyp["ignore_classes"]) or ((ctg + '/' + i) not in hyp["ignored_classes"]):
          classes.append(ctg + '/' + i)

    classes = sorted(classes)
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

          # triplet = [
          #     self.transform(PIL.Image.open(anchor_path).convert("RGB")),
          #     self.transform(PIL.Image.open(positive_path).convert("RGB")),
          #     self.transform(PIL.Image.open(negative_path).convert("RGB"))
          # ]
              
          triplet = [
              anchor_path,
              positive_path,
              negative_path,
          ]


          # print(triplet)
          # print([anchor_path, positive_path, negative_path])
          triplets.append(triplet)
          pbar.update(1)

    return triplets

  def __len__(self):
    return len(self.triplets)

  ##
  def __getitem__(self, idx):
    triplet = self.triplets[idx]
    t = [
      self.transform(PIL.Image.open(triplet[0]).convert("RGB")),
      self.transform(PIL.Image.open(triplet[1]).convert("RGB")),
      self.transform(PIL.Image.open(triplet[2]).convert("RGB"))
    ]
    t = torch.stack(t)
    return t


# %%
image_processor

# %%
# image_processor.image_processor

# %%
# vit
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

# clip

# normalize = Normalize(mean=image_processor.image_processor.image_mean, std=image_processor.image_processor.image_std)
# train_transforms = Compose(
#     [
#         RandomResizedCrop(image_processor.image_processor.crop_size["height"]),
#         RandomHorizontalFlip(),
#         ToTensor(),
#         normalize,
#     ]
# )

# val_transforms = Compose(
#     [
#         Resize(image_processor.image_processor.size["shortest_edge"]),
#         CenterCrop(image_processor.image_processor.crop_size["height"]),
#         ToTensor(),
#         normalize,
#     ]
# )

# test_transforms = Compose(
#     [
#         Resize(image_processor.image_processor.size["shortest_edge"]),
#         CenterCrop(image_processor.image_processor.crop_size["height"]),
#         ToTensor(),
#         normalize,
#     ]
# )

# %%
batch_size = 12

# train_dataset = ObjectTriplets('/home2/aneesh.chavan/Change_detection/condensed_procthor_images/train', train_transforms, num_triplets_per_class=100, difficult_triplet_percentage=hyp["hard_triplet_percent"])
# val_dataset = ObjectTriplets('/home2/aneesh.chavan/Change_detection/condensed_procthor_images/val', val_transforms, num_triplets_per_class=40, difficult_triplet_percentage=hyp["hard_triplet_percent"])
# test_dataset = ObjectTriplets('/home2/aneesh.chavan/Change_detection/condensed_procthor_images/test', test_transforms, num_triplets_per_class=20, difficult_triplet_percentage=hyp["hard_triplet_percent"])
# train_dataset = ObjectTriplets('/home2/aneesh.chavan/Change_detection/condensed_procthor_images/train', train_transforms, num_triplets_per_class=100, difficult_triplet_percentage=hyp["hard_triplet_percent"])

val_dataset = ObjectTriplets('/scratch/aneesh/random/splits/val', val_transforms, num_triplets_per_class=40, difficult_triplet_percentage=hyp["hard_triplet_percent"])
test_dataset = ObjectTriplets('/scratch/aneesh/random/splits/test', test_transforms, num_triplets_per_class=20, difficult_triplet_percentage=hyp["hard_triplet_percent"])
train_dataset = ObjectTriplets('/scratch/aneesh/random/splits/train', train_transforms, num_triplets_per_class=100, difficult_triplet_percentage=hyp["hard_triplet_percent"])

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# %%
test_triplet = next(iter(train_loader))
test_triplet

# %% [markdown]
# ## Set up model configs

# %%
from transformers import ViTConfig, ViTModel, ViTForImageClassification
from transformers import CLIPVisionModel

# vit

pretrained_model = ViTModel.from_pretrained(
    model_checkpoint,
    ignore_mismatched_sizes=True
)

# the first patch is the CLS token
pretrained_model


# clip

# pretrained_model = CLIPVisionModel.from_pretrained(
#     model_checkpoint,
#     ignore_mismatched_sizes=True
# )

# # the first patch is the CLS token
# pretrained_model

# %%
# print(len(output.last_hidden_state))
# print((output.last_hidden_state).shape)
# print()

# cls_token = output.last_hidden_state[:,0,:]
# print(cls_token.shape)

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# set up the lora config

# !pip install peft
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,

    # vit
    target_modules=["query", "value"],

    # clip
    # target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(pretrained_model, lora_config).to(device)

# print(lora_model)

# %%
# compare trainable param count
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

pretrained_model = ViTModel.from_pretrained(
    model_checkpoint,
    ignore_mismatched_sizes=True
)

print("No LoRa: ", end='')
print_trainable_parameters(pretrained_model)
print("With LoRa: ", end='')
print_trainable_parameters(lora_model)

# %% [markdown]
# ## Main training loop

# %%
# returns embeddings for anchor, positive and negative images
def get_embeddings(model, batch):
    # get cls tokens directly
    a_embs = model(batch[:, 0, ...], output_hidden_states=True).last_hidden_state[:,0,:]
    p_embs = model(batch[:, 1, ...], output_hidden_states=True).last_hidden_state[:,0,:]
    n_embs = model(batch[:, 2, ...], output_hidden_states=True).last_hidden_state[:,0,:]

    return a_embs, p_embs, n_embs

def get_triplet_loss(a_embs, p_embs, n_embs, bias_val=0.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bias = torch.ones(a_embs.shape[0]).float() * bias_val
    bias = bias.to(device)

    z = torch.zeros(a_embs.shape[0]).float().to(device)

    loss = torch.max(z,
               (
                  bias +
                  torch.linalg.norm(a_embs - p_embs, dim=1) -
                  torch.linalg.norm(a_embs - n_embs, dim=1)
               ))

    loss = torch.sum(loss)

    return loss

# %%
import copy
t = copy.copy(test_triplet).to('cuda' if torch.cuda.is_available() else 'cpu')

# print(test_triplet[0].shape, test_triplet[1].shape, test_triplet[2].shape)
# print(t[:,0,...])

# %%
a,p,n = get_embeddings(lora_model, t)

# %%
loss = get_triplet_loss(a,p,n, bias_val=2)

# %%
loss

# %% [markdown]
# ## Evaluate before finetuning

# %%
# Visualise test set images

import re

# Load all test images
# test_path =  '/home2/aneesh.chavan/Change_detection/condensed_procthor_images/test'
test_path =  '/scratch/aneesh/random/splits/test'

categories = os.listdir(test_path)
test_classes = []
for ctg in categories:
  for i in os.listdir(os.path.join(test_path, ctg)):
    test_classes.append(ctg + '/' + i)

test_classes = [i for i in sorted(test_classes) if (int(re.sub(r'[a-z_/]*','', i)) in [x for x in range(1,16)])]

test_images = [
    os.listdir(os.path.join(
        test_path,
        c
    )) for c in test_classes
]

for i in range(len(test_classes)):
  for j in range(len(test_images[i])):
    test_images[i][j] = os.path.join(test_path, test_classes[i], test_images[i][j])

# %%
for i in range(len(test_classes)):
  for j in range(len(test_images[i])):
    test_images[i][j] = PIL.Image.open(test_images[i][j])

# %%
len(test_classes)
# test_classes

# %%
import matplotlib.pyplot as plt

to_show = 10
# fig, axes = plt.subplots(len(test_classes), to_show, figsize=(to_show * 2.5, len(test_classes) * 2.5))
fig, axes = plt.subplots(30, to_show, figsize=(30 * 0.5, to_show * 0.5))

# for i in tqdm(range(len(test_classes))):
for i in tqdm(range(30)):
  for j, img in enumerate(test_images[i][:to_show]):
    ax = axes[i][j]
    ax.imshow(img)
    ax.axis('off')  # Hide axis

# plt.show()


# %%
# get all embeddings

w = []

with torch.no_grad():
  with tqdm(total=len(test_images)*len(test_images[0])) as bar:
    for row in test_images:
      r = []
      for i in row:
        im = test_transforms(i.convert("RGB"))
        k = lora_model(im.unsqueeze(0).cuda(), output_hidden_states=True).last_hidden_state[0,0,:]

        r.append(k)
        bar.update(1)
      w.append(torch.stack(r))
w = torch.stack(w).reshape(-1, 768)

# %%
# get heatmap

import torch.nn.functional as F

scores = (F.cosine_similarity(w.unsqueeze(0), w.unsqueeze(1), axis=-1)).detach().cpu().numpy()

plt.figure(figsize=(15,15))
plt.imshow(scores, cmap='hot')
plt.colorbar()

num_instances = 6

x_axis_titles = [f"{test_classes[i//num_instances]}" for i in range(num_instances//2, num_instances//2 + len(scores), num_instances)]
y_axis_titles = [f"{test_classes[i//num_instances]}" for i in range(num_instances//2, num_instances//2 + len(scores), num_instances)]

plt.xticks(range(num_instances//2, num_instances//2 + len(scores), num_instances), x_axis_titles, fontsize=6, rotation=45, ha='right')
plt.yticks(range(num_instances//2, num_instances//2 + len(scores), num_instances), y_axis_titles, fontsize=6, va='center')

for i in range(1, len(scores)):
    if i % num_instances == 0:
        plt.axvline(x=i - 0.5, color='blue', linestyle='-', linewidth=0.5)
        plt.axhline(y=i - 0.5, color='blue', linestyle='-', linewidth=0.5)


# Show the heatmap
plt.title("Base ViT vision encoder, no finetuning")

plt.show()

# %%
import numpy as np
exponent = 3
scaled_similarity = np.power(scores,exponent)

plt.plot(np.arange(0,1,0.01), np.power(np.arange(0,1,0.01),exponent))

plt.figure(figsize=(15,15))
plt.imshow(scaled_similarity, cmap='hot')
plt.colorbar()

num_instances = 6

x_axis_titles = [f"{test_classes[i//num_instances]}" for i in range(num_instances//2, num_instances//2 + len(scores), num_instances)]
y_axis_titles = [f"{test_classes[i//num_instances]}" for i in range(num_instances//2, num_instances//2 + len(scores), num_instances)]

plt.xticks(range(num_instances//2, num_instances//2 + len(scores), num_instances), x_axis_titles, fontsize=6, rotation=45, ha='right')
plt.yticks(range(num_instances//2, num_instances//2 + len(scores), num_instances), y_axis_titles, fontsize=6, va='center')

for i in range(1, len(scores)):
    if i % num_instances == 0:
        plt.axvline(x=i - 0.5, color='blue', linestyle='-', linewidth=0.5)
        plt.axhline(y=i - 0.5, color='blue', linestyle='-', linewidth=0.5)


# Show the heatmap
plt.title("Base ViT vision encoder, no finetuning")

plt.show()

# %% [markdown]
# ## Training loop

# %%
import torch
import torch.nn as nn
import torch.optim as optim

class train_config():
  def __init__(self,
               r=16,
               bias=1.0,
               batch_size=16,
               num_epochs=10,
               ):
    self.r = r
    self.bias = bias
    self.batch_size = batch_size
    self.num_epochs = num_epochs

def train(model, train_loader, val_loader, optimizer, config):
    """
    Generic training loop for PyTorch models.

    Args:
    model: The neural network model to train.
    train_loader: DataLoader for the training dataset.
    criterion: The loss function (e.g., nn.CrossEntropyLoss).
    optimizer: The optimizer (e.g., optim.SGD or optim.Adam).
    device: The device (CPU or GPU) to perform training.
    num_epochs: The number of training epochs.

    Returns:
    None
    """
    model.to(device)
    model.train()  # Set the model to training mode

    for epoch in range(config.num_epochs):
        print("Epoch: ", epoch+1)
        running_loss = 0.0

        for triplet in tqdm(train_loader):
            triplet = triplet.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            a,p,n = get_embeddings(lora_model, triplet)
            loss = get_triplet_loss(a,p,n, config.bias)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate the average loss for this epoch
        epoch_loss = running_loss / len(train_loader)

        # Calculate validation loss
        val_running_loss = 0.0
        with torch.no_grad():
          for val_triplet in val_loader:
            a,p,n = get_embeddings(lora_model, triplet)
            val_loss = get_triplet_loss(a,p,n, config.bias)

            val_running_loss += val_loss.item()

        val_epoch_loss = val_running_loss / len(val_loader)


        print(f"Epoch [{epoch + 1}/{config.num_epochs}] - Loss: {epoch_loss:.4f} - Validation loss: {val_epoch_loss:.4}")
        print()

    print("Training complete.")

# Example usage:
# train(model, train_loader, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001), "cuda", num_epochs=10)

# %%
optimizer = optim.Adam(lora_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

train_cfg = train_config(bias=2.5)
train(lora_model, train_loader, val_loader, optimizer, train_cfg)

# %% [markdown]
# ## Evaluate on test set

# %%
# Visualise test set images

import re

# Load all test images
# test_path =  '/home2/aneesh.chavan/Change_detection/condensed_procthor_images/test'
test_path =  '/scratch/aneesh/random/splits/test'

categories = os.listdir(test_path)
test_classes = []
for ctg in categories:
  for i in os.listdir(os.path.join(test_path, ctg)):
    test_classes.append(ctg + '/' + i)

test_classes = [i for i in sorted(test_classes) if (int(re.sub(r'[a-z_/]*','', i)) in [x for x in range(1,16)])]

test_images = [
    os.listdir(os.path.join(
        test_path,
        c
    )) for c in test_classes
]

for i in range(len(test_classes)):
  for j in range(len(test_images[i])):
    test_images[i][j] = os.path.join(test_path, test_classes[i], test_images[i][j])

# %%
for i in range(len(test_classes)):
  for j in range(len(test_images[i])):
    test_images[i][j] = PIL.Image.open(test_images[i][j])

# %%
import matplotlib.pyplot as plt

to_show = 10
fig, axes = plt.subplots(to_show, len(test_classes), figsize=(to_show * 2.5, len(test_classes) * 2.5))

for i in tqdm(range(len(test_classes))):
  for j, img in enumerate(test_images[i][:to_show]):
    ax = axes[j][i]
    ax.imshow(img)
    ax.axis('off')  # Hide axis

plt.show()


# %%
# get all embeddings

w = []

with torch.no_grad():
  for row in test_images:
    r = []
    for i in row:
      im = test_transforms(i.convert("RGB"))
      k = lora_model(im.unsqueeze(0).cuda(), output_hidden_states=True).last_hidden_state[0,0,:]

      r.append(k)
    w.append(torch.stack(r))
w = torch.stack(w).reshape(-1, 768)

# %%

# plt.plot(np.arange(0,1,0.01), np.power(np.arange(0,1,0.01),exponent))

plt.figure(figsize=(15,15))
plt.imshow(scores, cmap='hot')
plt.colorbar()

num_instances = 6

x_axis_titles = [f"{test_classes[i//num_instances]}" for i in range(num_instances//2, num_instances//2 + len(scores), num_instances)]
y_axis_titles = [f"{test_classes[i//num_instances]}" for i in range(num_instances//2, num_instances//2 + len(scores), num_instances)]

plt.xticks(range(num_instances//2, num_instances//2 + len(scores), num_instances), x_axis_titles, fontsize=6, rotation=45, ha='right')
plt.yticks(range(num_instances//2, num_instances//2 + len(scores), num_instances), y_axis_titles, fontsize=6, va='center')

for i in range(1, len(scores)):
    if i % num_instances == 0:
        plt.axvline(x=i - 0.5, color='blue', linestyle='-', linewidth=0.5)
        plt.axhline(y=i - 0.5, color='blue', linestyle='-', linewidth=0.5)


# Show the heatmap
plt.title("Base ViT vision encoder, no finetuning")

plt.show()

# %%
exponent = 3
scaled_similarity = np.power(scores,exponent)

import numpy as np
plt.plot(np.arange(0,1,0.01), np.power(np.arange(0,1,0.01),exponent))

plt.figure(figsize=(15,15))
plt.imshow(scaled_similarity, cmap='hot')
plt.colorbar()

num_instances = 6

x_axis_titles = [f"{test_classes[i//num_instances]}" for i in range(num_instances//2, num_instances//2 + len(scores), num_instances)]
y_axis_titles = [f"{test_classes[i//num_instances]}" for i in range(num_instances//2, num_instances//2 + len(scores), num_instances)]

plt.xticks(range(num_instances//2, num_instances//2 + len(scores), num_instances), x_axis_titles, fontsize=6, rotation=45, ha='right')
plt.yticks(range(num_instances//2, num_instances//2 + len(scores), num_instances), y_axis_titles, fontsize=6, va='center')

for i in range(1, len(scores)):
    if i % num_instances == 0:
        plt.axvline(x=i - 0.5, color='blue', linestyle='-', linewidth=0.5)
        plt.axhline(y=i - 0.5, color='blue', linestyle='-', linewidth=0.5)


# Show the heatmap
plt.title("Finetuned")

plt.show()

# %%
# plot within each coarse category
# armchairs

# get heatmap

import torch.nn.functional as F

scores = (F.cosine_similarity(w.unsqueeze(0), w.unsqueeze(1), axis=-1)).detach().cpu().numpy()[:40,:40]

plt.imshow(scores, cmap='hot')
plt.colorbar()


for i in range(1, len(scores)):
    if i % 10 == 0:
        plt.axvline(x=i - 0.5, color='blue', linestyle='-', linewidth=1.5)
        plt.axhline(y=i - 0.5, color='blue', linestyle='-', linewidth=1.5)


# Show the heatmap
plt.title("ViT, LoRa r=8, bias=2.5, 75% hard triplets")

plt.show()

# chairs
scores = (F.cosine_similarity(w.unsqueeze(0), w.unsqueeze(1), axis=-1)).detach().cpu().numpy()[40:90,40:90]

plt.imshow(scores, cmap='hot')
plt.colorbar()


for i in range(1, len(scores)):
    if i % 10 == 0:
        plt.axvline(x=i - 0.5, color='blue', linestyle='-', linewidth=1.5)
        plt.axhline(y=i - 0.5, color='blue', linestyle='-', linewidth=1.5)


# Show the heatmap
plt.title("ViT, LoRa r=8, bias=2.5, 75% hard triplets")

plt.show()



# tables
scores = (F.cosine_similarity(w.unsqueeze(0), w.unsqueeze(1), axis=-1)).detach().cpu().numpy()[90:,90:]

plt.imshow(scores, cmap='hot')
plt.colorbar()


for i in range(1, len(scores)):
    if i % 10 == 0:
        plt.axvline(x=i - 0.5, color='blue', linestyle='-', linewidth=1.5)
        plt.axhline(y=i - 0.5, color='blue', linestyle='-', linewidth=1.5)


# Show the heatmap
plt.title("ViT, LoRa r=8, bias=2.5, 75% hard triplets")

plt.show()

# %%
# hungarian tests
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np

def hungarian_test(model, num_trials, num_objects=5, disp=True):
  # Load all test images
  test_path =  '/scratch/merged/test'

  categories = os.listdir(test_path)
  test_classes = []
  for ctg in categories:
    for i in os.listdir(os.path.join(test_path, ctg)):
      test_classes.append(ctg + '/' + i)

  test_classes = sorted(test_classes)

  test_images = [
      os.listdir(os.path.join(
          test_path,
          c
      )) for c in test_classes
  ]

  for i in range(len(test_classes)):
    for j in range(len(test_images[i])):
      test_images[i][j] = os.path.join(test_path, test_classes[i], test_images[i][j])

  # for num_trials, load two sets of images, containing the same categories of images, from different views
  with torch.no_grad():
    total_associations = 0
    correct_associations = 0
    num_correct = [0]*(num_objects+1)

    for trial in tqdm(range(num_trials)):
      # choose categories, load images for both sets
      chosen_objects = random.sample([i for i in range(len(test_classes))], num_objects)

      set1 = []; set2 = []
      for o in chosen_objects:
        a,b = random.sample(test_images[o], 2)

        set1.append(PIL.Image.open(a))
        set2.append(PIL.Image.open(b))

      # display images if needed
      if disp and trial == 0:
        fig1, axs1 = plt.subplots(1, len(set1), figsize=(2*len(set1), 2))
        for i, img in enumerate(set1):
          axs1[i].set_title(str(i))
          axs1[i].imshow(img)
        plt.subplots_adjust(wspace=0.4)
        plt.show()

        fig2, axs2 = plt.subplots(1, len(set2), figsize=(2*len(set2), 2))
        for i, img in enumerate(set2):
          axs2[i].set_title(str(i))
          axs2[i].imshow(img)
        plt.subplots_adjust(wspace=0.4)
        plt.show()
        print()

      # get embeddings
      embs1 = []; embs2 = []
      for a,b in zip(set1, set2):
        ima = test_transforms(a)
        ka = model(ima.unsqueeze(0).cuda(), output_hidden_states=True).last_hidden_state[0,0,:]
        embs1.append(ka)

        imb = test_transforms(b)
        kb = model(imb.unsqueeze(0).cuda(), output_hidden_states=True).last_hidden_state[0,0,:]
        embs2.append(kb)
      embs1 = torch.stack(embs1)
      embs2 = torch.stack(embs2)

      # get cosine similarity, run hungarian
      scores = (F.cosine_similarity(embs1.unsqueeze(0), embs2.unsqueeze(1), axis=-1)).detach().cpu().numpy()

      cost_matrix = 1 - scores
      row_idx, col_idx = linear_sum_assignment(cost_matrix)

      # update statistics
      numcor = 0
      pairs = {}
      for i,j in zip(row_idx, col_idx):
        if i == j:
          numcor += 1
        pairs[i] = j


      total_associations += len(row_idx)
      correct_associations += numcor
      num_correct[numcor] += 1


      # display similarity heatmap if required, for the first iteration
      if disp and trial == 0:
        print("Hungarian algorithm")
        print(row_idx, col_idx)
        plt.imshow(scores, cmap='hot', interpolation='nearest')

        plt.colorbar()
        plt.xlabel('Img 2')
        plt.ylabel('Img 1')

        for i in range(scores.shape[0]):
          for j in range(scores.shape[1]):
            text = f'{scores[i, j]:.2f}'  # Format score to two decimal placesg

            if scores[i,j] > 0.5:
              text_color = 'black'
            else:
              text_color = 'white'

            if pairs[i] == j:
              text_color = 'blue'
              text_weight = 'bold'
            else:
              text_weight=None

            plt.text(j, i, text, ha='center', va='center', color=text_color, fontweight=text_weight)

        plt.clim(0, 1);
        plt.show()
        print()

        print(f"{numcor} correct out of {num_objects}")
        print()

  # display the final calculated statistics
  print(f"{num_trials} trials run, {100*float(correct_associations)/total_associations}% of objects correctly matched")


  print(num_correct)
  plt.plot(np.linspace(0, num_objects, num=(num_objects+1)), num_correct)
  plt.xlabel('Correctly matched objects per trial')
  plt.ylabel('Number of trials')
  plt.title('')
  plt.show()

# hungarian_test(lora_model, 200, disp=True)

# %%


# %% [markdown]
# ## Saving only LoRa params
# 

# %%
def save_lora_checkpoint(lora_model, lora_config, model_checkpoint, path):
  lora_params = {}
  sd = lora_model.state_dict()
  for k in lora_model.state_dict().keys():
    if(k.find("lora") != -1):
      lora_params[k] = sd[k]

  lora_ckpt = {"lora_config": lora_config ,"lora_state_dict": lora_params, "pretrained_model_checkpoint": model_checkpoint}
  torch.save(lora_ckpt, path)
  return lora_ckpt

# !!!!hardcoded to ViT for now!!!!
def load_lora_checkpoint(lora_ckpt):
  # load pretrained
  pretrained_model = ViTModel.from_pretrained(
      lora_ckpt["pretrained_model_checkpoint"],
      ignore_mismatched_sizes=True
  )

  # load config, add adapter layers to the model
  lora_model = get_peft_model(pretrained_model, lora_ckpt["lora_config"]).to(device)
  # restore finetuned lora weights
  lora_model.load_state_dict(lora_ckpt["lora_state_dict"], strict=False);

  return lora_model

# %%
# !mkdir -p /content/drive/MyDrive/change_detection_ckpts
# !ls -lh /content/drive/MyDrive/change_detection_ckpts

# %%
# save_name = hyp["exp_name"]
save_name = "procthor_1.pt"

# %%
lora_ckpt_vit = save_lora_checkpoint(lora_model, lora_config, model_checkpoint, save_name)

# %%
lora_model = load_lora_checkpoint(torch.load(save_name, map_location=torch.device('cuda')))

# %%
lora_model = load_lora_checkpoint(lora_ckpt_vit)

# %% [markdown]
# ## Multisequence

# %%
# !unzip /scratch/data/multisequence_lora_with_depth.zip

# %% [markdown]
# ### composite emb from hand sorted cropped images directly
# 1. assemble composite embeddings from cropped images
# 2. check if composite emb works with all images
# 
#   - ? how do we decide when to add a new object? multi-track?
#     - thresholding?
# 
# ### composite emb from all images one at a time
# 

# %%
# load finetuned model
# lora_model = load_lora_checkpoint(torch.load("/content/drive/MyDrive/change_detection_ckpts/vit_finegrained.pt", map_location=torch.device(device)))

# # %%
# import os, sys, glob

# class EnvObject():
#   def __init__(self, name):
#     self.name = ""
#     self.img_emb_pairs = []
#     self.composite_emb = None

#   def add_image(self, path):
#     self.img_emb_pairs.append({'path': path, 'emb': None})

#   def encode_images(self, model):
#     for ie in tqdm(self.img_emb_pairs):
#       with torch.no_grad():
#         img = PIL.Image.open(ie['path'])
#         img = test_transforms(img)
#         emb = model(img.unsqueeze(0).to(device), output_hidden_states=True).last_hidden_state[0,0,:]   #### TODO ADD A CUDA HERE AFTER UNSQUEEZE
#                                                                                           #### TRY BATCHING
#       ie['emb'] = emb

#   def get_composite_emb(self):
#     avg = None
#     for pair in self.img_emb_pairs:
#       try:
#         if avg == None:
#           avg = pair['emb']
#           continue

#         avg += pair['emb']
#       except:
#         avg = None
#         print('Not all images encoded')
#         return

#     avg /= len(self.img_emb_pairs)
#     self.composite_emb = avg

# class Sequence():
#   def __init__(self, path, model):
#     self.dir_path = path
#     self.model = model

#     for p in model.parameters():
#       p.requies_grad = False

#   def load_cropped_images(self):
#     self.objects = {}
#     for dir in sorted(os.listdir(os.path.join(self.dir_path, "cropped"))):

#       self.objects[dir] = EnvObject(dir)
#       for img in sorted(os.listdir(
#             os.path.join(os.path.join(self.dir_path, "cropped", dir)))
#       ):
#         self.objects[dir].add_image(os.path.join(os.path.join(self.dir_path, "cropped", dir, img)))

#       self.objects[dir].encode_images(self.model)
#     print(self.objects)

#   def get_composite_embs(self):
#     for k, env_obj in self.objects.items():
#       env_obj.get_composite_emb()



# # class

# # %%
# seq1 = Sequence("/content/Datasets/ArmChairs/orientation1", lora_model)

# seq1.load_cropped_images()
# seq1.get_composite_embs()

# # %%
# seq2 = Sequence("/content/Datasets/ArmChairs/orientation2", lora_model)
# seq2.load_cropped_images()
# seq2.get_composite_embs()

# # %%
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np

# def compare_composite_vs_composite(seq1, seq2, title='Compare aggregated embeddings across sequences'):
#   # for k, o in seq1.objects.items():
#   #   print(o.composite_emb.shape)

#   # print()
#   # for k, o in seq2.objects.items():
#   #   print(o.composite_emb.shape)

#   comp1 = torch.stack([o.composite_emb for k,o in seq1.objects.items()])
#   comp2 = torch.stack([o.composite_emb for k,o in seq2.objects.items()])

#   scores = (F.cosine_similarity(comp1.unsqueeze(0), comp2.unsqueeze(1), axis=-1)).detach().cpu().numpy()

#   plt.xlabel('Img 2')
#   plt.ylabel('Img 1')

#   for i in range(scores.shape[0]):
#     for j in range(scores.shape[1]):
#       text = f'{scores[i, j]:.2f}'  # Format score to two decimal placesg

#       if scores[i,j] > 0.5:
#         text_color = 'black'
#       else:
#         text_color = 'white'

#       # if pairs[i] == j:
#       #   text_color = 'blue'
#       #   text_weight = 'bold'
#       # else:
#       #   text_weight=None

#       plt.text(j, i, text, ha='center', va='center', color=text_color, fontweight=None)

#   plt.title(title)
#   plt.imshow(scores, cmap='hot')
#   # plt.clim(0, 1);

#   print("Comparing the assembled aggregated embedings across both sequences, we obtain correct associations even within classes\n")

#   plt.show()

#   rows = [i for i in range(1, len(scores) + 1)]
#   best_per_row = [np.argmax(scores[i]) + 1 for i in range(len(scores))]
#   correct = len([i for i in range(len(scores)) if rows[i] == best_per_row[i]])
#   print(f"%d/%d correct associations" % (correct, len(rows)))

# # %%
# compare_composite_vs_composite(seq1, seq2, title='Seq1 vs seq2 armchairs')

# # %%
# torch.cuda.empty_cache()

# # %%
# # check for chairs
# # del seq1
# # del seq2
# # # del seq1_chairs
# # # del seq2_chairs
# # torch.cuda.empty_cache()

# seq1_chairs = Sequence("/content/Datasets/Chairs/orientation1", lora_model)
# seq1_chairs.load_cropped_images()
# seq1_chairs.get_composite_embs()

# seq2_chairs = Sequence("/content/Datasets/Chairs/orientation2", lora_model)
# seq2_chairs.load_cropped_images()
# seq2_chairs.get_composite_embs()

# compare_composite_vs_composite(seq1_chairs, seq2_chairs, title='Seq1 vs seq2 chairs')

# # %%
# # check for chairs
# # del seq1
# # del seq2
# # del seq1_chairs
# # del seq2_chairs
# torch.cuda.empty_cache()

# seq1_tables = Sequence("/content/Datasets/Tables/orientation1", lora_model)
# seq1_tables.load_cropped_images()
# seq1_tables.get_composite_embs()
# seq2_tables = Sequence("/content/Datasets/Tables/orientation2", lora_model)
# seq2_tables.load_cropped_images()
# seq2_tables.get_composite_embs()

# compare_composite_vs_composite(seq1_tables, seq2_tables, title='Seq1 vs seq2 tables')

# # %% [markdown]
# # ## Check that the composite embedding correctly associates with each image`

# # %% [markdown]
# # `Definitions`

# # %%
# # assemble per image sets

# # class Sequence():
# #   def __init__(self, path, model):
# #     self.dir_path = path
# #     self.model = model

# #     self.model.eval()

# #   def load_cropped_images(self):
# #     self.objects = {}
# #     for dir in sorted(os.listdir(os.path.join(self.dir_path, "cropped"))):

# #       self.objects[dir] = EnvObject()
# #       for img in sorted(os.listdir(
# #             os.path.join(os.path.join(self.dir_path, "cropped", dir)))
# #       ):
# #         self.objects[dir].add_image(os.path.join(os.path.join(self.dir_path, "cropped", dir, img)))

# #       self.objects[dir].encode_images(self.model)
# #     print(self.objects)

# #   def get_composite_embs(self):
# #     for k, env_obj in self.objects.items():
# #       env_obj.get_composite_emb()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # sort all images by view
# def assemble_images_per_image(seq: Sequence):
#   per_image_sets = []

#   for i in range(len(glob.glob(os.path.join(seq.dir_path, "view*")))):
#     per_image_sets.append([])

#   for _, obj in seq.objects.items():
#     for pair in obj.img_emb_pairs:
#       p = pair['path']
#       num = int(p.split('/')[-1].split('.')[0])
#       per_image_sets[num - 1].append(p)

#   return per_image_sets

# # retrieve embeddings for each set of images sorted by view
# # if an object is not present in that view, replace it with a dummy embedding of zeros
# def view_vs_composite(seq, seq_per_img):
#   dummy_emb = torch.zeros((768)).to(device)
#   objs = seq.objects.keys()  # all objs

#   cropped = []
#   for view in seq_per_img:
#     per_view = []

#     present = [path.split('/')[-2] for path in view]    # stuff not present in this view

#     for i in objs:
#       if i not in present:
#         per_view.append(dummy_emb)
#       else:
#         for k, pair in enumerate(seq.objects[i].img_emb_pairs):   # find the emb from EnvObject list
#           if pair['path'] in view:                                # check if the emb has a path belonging to the current view
#             e = pair['emb']
#             per_view.append(e)
#             break

#     cropped.append(torch.stack(per_view))

#   return cropped

# # check composite emb per view
# def check_comp(seq, seq_embs_per_view):
#   comp_embs = []
#   for _, o in seq.objects.items():
#     comp_embs.append(o.composite_emb)
#   comp_embs = torch.stack(comp_embs)

#   for num, view in enumerate(seq_embs_per_view):
#     scores = (F.cosine_similarity(comp_embs.unsqueeze(0), view.unsqueeze(1), axis=-1)).detach().cpu().numpy()

#     plt.xlabel("View " + str(num))
#     plt.ylabel('Composite embs')

#     for i in range(scores.shape[0]):
#       for j in range(scores.shape[1]):
#         text = f'{scores[i, j]:.2f}'  # Format score to two decimal placesg

#         if scores[i,j] > 0.5:
#           text_color = 'black'
#         else:
#           text_color = 'white'

#         # if pairs[i] == j:
#         #   text_color = 'blue'
#         #   text_weight = 'bold'
#         # else:
#         #   text_weight=None

#         plt.text(j, i, text, ha='center', va='center', color=text_color, fontweight=None)

#     plt.title(f"View %d vs composite embeddings" % num)
#     plt.imshow(scores, cmap='hot')
#     plt.clim(0, 1);

#     plt.show()

#     rows = [i for i in range(1, len(scores) + 1)]
#     best_per_row = [np.argmax(scores[i]) + 1 for i in range(len(scores))]
#     correct = len([i for i in range(len(scores)) if rows[i] == best_per_row[i] and scores[i][i] != 0])
#     print(f"%d/%d correct associations" % (correct, len(rows) - len([i for i in range(len(scores)) if scores[i][i] == 0])))

# # combine all functions, directly compare seq_composites' comp. embs to seq_views
# def composite_vs_views(seq_views, seq_composite):
#   per_img = assemble_images_per_image(seq_views)
#   per_img_embs = view_vs_composite(seq_views, per_img)
#   check_comp(seq_composite, per_img_embs)


# seq1_per_img = assemble_images_per_image(seq1)


# # %% [markdown]
# # ### `Armchair sequences`

# # %%
# print("These experiments compare the composites from sequence 1 with all sequence 1 images")
# print("Composite vs each view is successful for all 8 views except one, (view 5, row 3)")
# print("The armchair in this case is very occluded and barely detected")
# print("These occlusions may skew our composite emb, a better strategy may be a library embeddings (memory concerns)")
# print()
# composite_vs_views(seq1, seq1)


# # %%
# # seq2 vs seq2 per view

# composite_vs_views(seq2, seq2)

# # %% [markdown]
# # 
# # 
# # ```
# # # Going across sequences
# # ```
# # 
# # 

# # %%
# # seq1 composite and seq2 per view

# print("As expeceted, worse results, given that occlusions prevent us from getting an accurate composite emb of each object")
# print("Memory buffers could solve these, but using just composite embeddings may be enough")
# print("I am concerned about how brittle this system is, composite vs composite is 100% accurate for an extreme case, (all sofas) but the margins are thin")
# print("Experimentation in more diverse datasets is required")
# composite_vs_views(seq1, seq2)

# # %%
# # seq2 composite and seq1 per view

# print("Now going the other way")
# composite_vs_views(seq2, seq1)

# # %% [markdown]
# # ### `Chair sequences`

# # %%
# composite_vs_views(seq1_chairs, seq1_chairs)


# # %%
# # seq2 vs seq2 per view

# composite_vs_views(seq2_chairs, seq2_chairs)

# # %% [markdown]
# # 
# # 
# # ```
# # # Going across sequences
# # ```
# # 
# # 

# # %%
# # seq1 composite and seq2 per view

# composite_vs_views(seq1_chairs, seq2_chairs)

# # %%
# # seq2 composite and seq1 per view

# print("Now going the other way")
# composite_vs_views(seq2_chairs, seq1_chairs)

# # %% [markdown]
# # ### `Table sequences`

# # %%
# composite_vs_views(seq1_tables, seq1_tables)


# # %%
# # seq2 vs seq2 per view

# composite_vs_views(seq2_tables, seq2_tables)

# # %% [markdown]
# # 
# # 
# # ```
# # # Going across sequences
# # ```
# # 
# # 

# # %%
# # seq1 composite and seq2 per view

# composite_vs_views(seq1_tables, seq2_tables)

# # %%
# # seq2 composite and seq1 per view

# print("Now going the other way")
# composite_vs_views(seq2_tables, seq1_tables)

# # %% [markdown]
# # ### Using k-means instead of composite embeddings


