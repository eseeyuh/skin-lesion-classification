"""Model construction (timm backbones with a trainable classifier head)."""

from typing import Tuple

import timm
import torch


def build_model(name: str, num_classes: int, device: torch.device,
                freeze_backbone: bool = True, pretrained: bool = True) -> torch.nn.Module:
    """Create a timm model with `num_classes` outputs.

    When `freeze_backbone` is True only the classifier head is trainable, which
    keeps the comparison across architectures fair under a fixed compute budget.
    """
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
    return model.to(device)


def count_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
