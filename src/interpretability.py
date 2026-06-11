"""Grad-CAM visual explanations for the convolutional models."""

import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .config import Config
from .data import IMAGENET_MEAN, IMAGENET_STD, HAMDataset, build_transforms
from .models import build_model

logger = logging.getLogger(__name__)


def _target_layers(model, model_name: str):
    if "resnet" in model_name:
        return [model.layer4[-1]]
    if "efficientnet" in model_name:
        return [model.conv_head]
    return None  # ViT has no single conv layer to hook here


def gradcam_for_model(model_name: str, cfg: Config, df_test, results: Dict, device,
                      n_examples: int = 3, save: bool = True):
    """Overlay Grad-CAM heatmaps on the first `n_examples` correctly classified test images.

    Returns the matplotlib Figure, or None for architectures we skip (e.g. ViT).
    """
    model = build_model(model_name, cfg.num_classes, device,
                        freeze_backbone=False, pretrained=False)
    model.load_state_dict(torch.load(cfg.models_dir / f"{model_name}_main.pt"))
    model.eval()

    target_layers = _target_layers(model, model_name)
    if target_layers is None:
        logger.info("Skipping Grad-CAM for %s (no conv target layer)", model_name)
        return None

    cam = GradCAM(model=model, target_layers=target_layers)
    _, eval_tfm = build_transforms(cfg.image_size)
    ds = HAMDataset(df_test, eval_tfm)
    preds = np.array(results[model_name]["test_preds"])
    labels = np.array(results[model_name]["test_labels"])
    correct_idx = np.where(preds == labels)[0][:n_examples]

    fig, axes = plt.subplots(2, n_examples, figsize=(4 * n_examples, 8))
    for i, idx in enumerate(correct_idx):
        img_t, label = ds[idx]
        img_np = img_t.permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img_np = np.clip(img_np, 0, 1)
        grayscale_cam = cam(input_tensor=img_t.unsqueeze(0).to(device))[0]
        overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"True: {cfg.class_names[label]}")
        axes[0, i].axis("off")
        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis("off")
    plt.suptitle(f"Grad-CAM — {cfg.model_display[model_name]}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(cfg.plots_dir / f"11_gradcam_{model_name}.png", dpi=150, bbox_inches="tight")
    return fig
