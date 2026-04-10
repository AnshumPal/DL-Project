"""
DeepFashion dataset loader.
Expects the following folder structure (standard DeepFashion Category split):
  data/deepfashion/
    img/
      <category>/
        <image>.jpg
    Anno/list_category_img.txt   (optional — used if flat structure)

If you have a flat img/ folder with list_category_img.txt, set
  USE_ANNOTATION_FILE = True below.
Otherwise organise images into class subfolders and set it to False.
"""

import os
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


USE_ANNOTATION_FILE = False   # flip to True if using raw DeepFashion layout


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(image_size: int, split: str) -> transforms.Compose:
    """
    Train: random crop, horizontal flip, colour jitter, normalize.
    Val/Test: centre crop only.
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats — used for all pretrained models
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class DeepFashionDataset(Dataset):
    """
    Loads DeepFashion images from a class-subfolder layout:
      root/img/<class_name>/<image>.jpg
    """
    def __init__(self, root: str, image_size: int, split: str = "train"):
        self.transform = get_transforms(image_size, split)
        img_root = Path(root) / "img"

        self.classes = sorted([
            d.name for d in img_root.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = img_root / cls
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((str(img_path), self.class_to_idx[cls]))
            for img_path in cls_dir.glob("*.png"):
                self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Build train/val/test DataLoaders from config.
    Returns (train_loader, val_loader, test_loader, class_names).
    """
    ds_cfg = cfg["dataset"]
    full_ds = DeepFashionDataset(ds_cfg["root"], ds_cfg["image_size"], split="train")

    n       = len(full_ds)
    n_test  = int(n * ds_cfg["test_split"])
    n_val   = int(n * ds_cfg["val_split"])
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Override transforms for val/test (random_split shares the same dataset object)
    val_ds.dataset  = _clone_with_transform(full_ds, ds_cfg["image_size"], "val")
    test_ds.dataset = _clone_with_transform(full_ds, ds_cfg["image_size"], "test")

    kwargs = dict(
        batch_size=ds_cfg["batch_size"],
        num_workers=ds_cfg["num_workers"],
        pin_memory=True,
    )

    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
        full_ds.classes,
    )


def _clone_with_transform(ds: DeepFashionDataset, image_size: int, split: str):
    """Return a shallow copy of the dataset with a different transform."""
    import copy
    clone = copy.copy(ds)
    clone.transform = get_transforms(image_size, split)
    return clone
