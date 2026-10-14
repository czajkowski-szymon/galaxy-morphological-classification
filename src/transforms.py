"""Image preprocessing pipelines for training and evaluation."""

import torch
from torchvision.transforms import v2

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = v2.Compose(
    [
        # Rotate the full 424x424 image before cropping: the 224x224 centre crop
        # (corner radius ~158 px) stays inside the area unaffected by rotation
        # (radius 212 px), so no black corners reach the model.
        v2.RandomRotation(180),
        v2.CenterCrop(224),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)

test_transform = v2.Compose(
    [
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)
