"""Skin lesion classification — comparing CNN and ViT architectures on HAM10000."""

from .config import Config
from .utils import set_seed, get_device, describe_device, setup_logging
from .data import (load_metadata, group_split, build_transforms, make_dataloaders,
                   compute_class_weights, HAMDataset, IMAGENET_MEAN, IMAGENET_STD)
from .models import build_model, count_trainable_params
from .engine import train_one_epoch, evaluate, train_model, lr_sweep
from .metrics import (compute_metrics, mcnemar_test, significance_stars,
                      expected_calibration_error)
from .ensemble import soft_vote

# viz and interpretability are imported explicitly (`from src import viz`,
# `from src.interpretability import gradcam_for_model`) to avoid pulling
# matplotlib / pytorch-grad-cam on every `import src`.

__version__ = "0.1.0"

__all__ = [
    "Config",
    "set_seed", "get_device", "describe_device", "setup_logging",
    "load_metadata", "group_split", "build_transforms", "make_dataloaders",
    "compute_class_weights", "HAMDataset", "IMAGENET_MEAN", "IMAGENET_STD",
    "build_model", "count_trainable_params",
    "train_one_epoch", "evaluate", "train_model", "lr_sweep",
    "compute_metrics", "mcnemar_test", "significance_stars", "expected_calibration_error",
    "soft_vote",
]
