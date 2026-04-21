from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def build_transform(image_size: int = 224, crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
        ]
    )


def load_image_tensor(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return transform(img)


def l2_normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, eps)
