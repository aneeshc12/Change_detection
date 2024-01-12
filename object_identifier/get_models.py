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


def get_configs(config=None):
    model_checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

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

    if config != None:
        return {
            "model_checkpoint": model_checkpoint,
            "image_processor": image_processor,
            "train_transforms": train_transforms,
            "val_transforms": val_transforms,
            "test_transforms": test_transforms,
        }
    else:
        config["model_checkpoint"] = model_checkpoint
        config["image_processor"] = image_processor
        config["train_transforms"] = train_transforms
        config["val_transforms"] = val_transforms
        config["test_transforms"] = test_transforms

        return config
