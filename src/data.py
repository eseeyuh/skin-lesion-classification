"""Dataset loading, splitting and augmentation."""

import logging
from typing import Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import Config

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_metadata(cfg: Config) -> pd.DataFrame:
    """Read the HAM10000 metadata and resolve each image_id to a file path."""
    df = pd.read_csv(cfg.data_dir / cfg.metadata_csv)
    img_dirs = list(cfg.data_dir.glob("HAM10000_images*")) or [cfg.data_dir]
    image_id_to_path = {f.stem: str(f) for d in img_dirs for f in d.glob("*.jpg")}

    df["path"] = df["image_id"].map(image_id_to_path)
    df["label"] = df["dx"].map(cfg.label2idx)

    missing = int(df["path"].isna().sum())
    if missing:
        logger.warning("%d images not found on disk, dropping them", missing)
        df = df.dropna(subset=["path"]).reset_index(drop=True)

    logger.info("Loaded %d images across %d unique lesions", len(df), df["lesion_id"].nunique())
    return df


def group_split(df: pd.DataFrame, cfg: Config,
                seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/val/test split grouped by lesion_id so no lesion leaks across splits."""
    seed = cfg.seed if seed is None else seed

    gss1 = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=df["lesion_id"]))
    df_trainval = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    val_relative = cfg.val_size / (1 - cfg.test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
    train_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval["lesion_id"]))
    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    assert not (set(df_train["lesion_id"]) & set(df_val["lesion_id"]))
    assert not (set(df_train["lesion_id"]) & set(df_test["lesion_id"]))
    assert not (set(df_val["lesion_id"]) & set(df_test["lesion_id"]))
    return df_train, df_val, df_test


def build_transforms(image_size: int) -> Tuple[A.Compose, A.Compose]:
    """Return (train, eval) augmentation pipelines."""
    train_tfm = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    eval_tfm = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tfm, eval_tfm


class HAMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["path"]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, int(row["label"])


def make_dataloaders(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                     cfg: Config, balanced_sampling: bool = True
                     ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders; the train loader can rebalance classes by sampling."""
    train_tfm, eval_tfm = build_transforms(cfg.image_size)
    train_ds = HAMDataset(df_train, train_tfm)
    val_ds = HAMDataset(df_val, eval_tfm)
    test_ds = HAMDataset(df_test, eval_tfm)

    if balanced_sampling:
        class_counts = df_train["label"].value_counts().sort_index().values
        weights_per_class = 1.0 / class_counts
        sample_weights = weights_per_class[df_train["label"].values]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(df_train), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                                  num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def compute_class_weights(df_train: pd.DataFrame, num_classes: int,
                          device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights (kept available for weighted-loss experiments)."""
    weights = compute_class_weight(class_weight="balanced",
                                   classes=np.arange(num_classes),
                                   y=df_train["label"].values)
    return torch.tensor(weights, dtype=torch.float32, device=device)
