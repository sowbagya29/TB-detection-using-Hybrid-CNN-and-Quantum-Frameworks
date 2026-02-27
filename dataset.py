"""
Dataset and preprocessing for TB Chest X-Ray (CLAHE, augmentations).
Supports Kaggle TB Chest X-Ray Dataset by Tawsifur Rahman.
"""
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torch

try:
    from torchvision import transforms
except ImportError:
    transforms = None

from config import (
    IMG_SIZE,
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE,
    AUGMENT_BRIGHTNESS,
    AUGMENT_CONTRAST,
    AUGMENT_ROTATION,
    CLASSES,
)


def apply_clahe(image: np.ndarray, clip_limit: float = CLAHE_CLIP_LIMIT, grid_size: tuple = CLAHE_GRID_SIZE) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (from base paper)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(gray)
    return enhanced


def get_transforms(is_training: bool, img_size: int = IMG_SIZE):
    """Train vs inference transforms."""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(AUGMENT_ROTATION),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=AUGMENT_BRIGHTNESS, contrast=AUGMENT_CONTRAST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])


class TBChestXRayDataset(Dataset):
    """
    TB Chest X-Ray dataset with CLAHE preprocessing.
    Expects folder structure:
      DATA_ROOT/Normal/*.png
      DATA_ROOT/Tuberculosis/*.png
    or:
      DATA_ROOT/Train/Normal, DATA_ROOT/Train/Tuberculosis
      DATA_ROOT/Test/...
    """
    def __init__(self, image_paths, labels, transform=None, use_clahe=True):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.use_clahe:
            img = apply_clahe(img)
            img = np.stack([img] * 3, axis=-1)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


def find_image_paths_and_labels(data_root: Path):
    """
    Discover (path, label) from DATA_ROOT.
    Supports: data_root/Normal, data_root/Tuberculosis
    or data_root/Train/Normal, data_root/Train/Tuberculosis.
    If not found, tries data_root/TB_Chest_Radiography_Database/...
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}. Download from Kaggle and set DATA_ROOT in config.py.")
    # If root has no Normal/Tuberculosis, try common Kaggle extract folder
    candidates = [data_root]
    inner = data_root / "TB_Chest_Radiography_Database"
    if inner.exists():
        candidates.append(inner)
    for root in candidates:
        paths, labels = _collect_paths_labels(root, exts={".png", ".jpg", ".jpeg", ".bmp"})
        if paths:
            return paths, labels
    raise FileNotFoundError(
        f"No images found under {data_root}. "
        f"Expected folders: .../Normal and .../Tuberculosis, "
        "or .../Train/Normal and .../Train/Tuberculosis."
    )


def _collect_paths_labels(data_root: Path, exts: set):
    paths, labels = [], []
    # Support multiple possible folder names for each class, to match this dataset:
    #   - "Normal" or "Normal Chest X-rays"
    #   - "Tuberculosis" or "TB Chest X-rays"
    alt_names = {
        0: ["Normal", "Normal Chest X-rays"],
        1: ["Tuberculosis", "TB Chest X-rays"],
    }
    for class_idx, class_name in enumerate(CLASSES):
        for folder_name in alt_names.get(class_idx, [class_name]):
            folder = data_root / folder_name
            if folder.exists():
                for f in folder.iterdir():
                    if f.suffix.lower() in exts and f.is_file():
                        paths.append(str(f))
                        labels.append(class_idx)
    if paths:
        return paths, labels
    for split in ("Train", "train", "Training"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for class_idx, class_name in enumerate(CLASSES):
            folder = split_dir / class_name
            if folder.exists():
                for f in folder.iterdir():
                    if f.suffix.lower() in exts and f.is_file():
                        paths.append(str(f))
                        labels.append(class_idx)
    return paths, labels


def get_train_val_test_splits(data_root: Path, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, seed=42):
    """Stratified split into train/val/test."""
    paths, labels = find_image_paths_and_labels(data_root)
    from sklearn.model_selection import train_test_split
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    idx = np.arange(len(paths))
    train_idx, rest_idx, train_y, rest_y = train_test_split(
        idx, labels, test_size=(1 - train_ratio), stratify=labels, random_state=seed
    )
    val_ratio_rest = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx, _, _ = train_test_split(
        rest_idx, [labels[i] for i in rest_idx], test_size=(1 - val_ratio_rest), stratify=[labels[i] for i in rest_idx], random_state=seed
    )
    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_paths = [paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
