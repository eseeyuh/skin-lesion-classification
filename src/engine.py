"""Training and evaluation loops."""

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import balanced_accuracy_score

from .config import Config
from .data import make_dataloaders
from .metrics import compute_metrics
from .models import build_model
from .utils import set_seed

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    """One training pass. Uses AMP plus gradient clipping for stability at higher LR."""
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, criterion=None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """Evaluate in full fp32 (no autocast) so the loss stays finite under AMP training."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(probs.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    avg_loss = total_loss / n if criterion is not None else None
    return preds, labels, probs, avg_loss


def train_model(cfg: Config, model_name: str, df_train, df_val, df_test, device,
                lr: float = None, num_epochs: int = None, tag: str = "main") -> Dict:
    """Train one architecture end to end and evaluate it on the test set.

    Returns the test metrics, training history, raw predictions and the
    checkpoint path. The best epoch is selected by validation balanced accuracy.
    """
    set_seed(cfg.seed)
    lr = lr or cfg.learning_rate
    num_epochs = num_epochs or cfg.num_epochs
    logger.info("Training %s | lr=%g | tag=%s", cfg.model_display[model_name], lr, tag)

    train_loader, val_loader, test_loader = make_dataloaders(df_train, df_val, df_test, cfg)
    model = build_model(model_name, cfg.num_classes, device, freeze_backbone=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()

    history = {"train_loss": [], "val_loss": [], "val_bal_acc": [], "epoch_time": []}
    best_bal_acc, patience = -1.0, 0
    ckpt_path = cfg.models_dir / f"{model_name}_{tag}.pt"

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_preds, val_labels, _, val_loss = evaluate(model, val_loader, device, criterion)
        val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
        scheduler.step()
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bal_acc"].append(val_bal_acc)
        history["epoch_time"].append(epoch_time)
        logger.info("Epoch %02d  train_loss=%.4f  val_loss=%.4f  val_bal_acc=%.4f  time=%.1fs",
                    epoch, train_loss, val_loss, val_bal_acc, epoch_time)

        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            torch.save(model.state_dict(), ckpt_path)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                logger.info("Early stopping at epoch %d (best=%.4f)", epoch, best_bal_acc)
                break

    model.load_state_dict(torch.load(ckpt_path))
    t_start = time.time()
    test_preds, test_labels, test_probs, _ = evaluate(model, test_loader, device)
    inference_time = time.time() - t_start

    metrics = compute_metrics(test_labels, test_preds, test_probs, cfg.class_names)
    metrics.update({
        "best_val_bal_acc": best_bal_acc,
        "total_train_time_s": float(np.sum(history["epoch_time"])),
        "test_inference_s": inference_time,
        "test_throughput_img_s": len(df_test) / inference_time,
        "model": model_name,
        "tag": tag,
        "lr": lr,
    })
    return {"metrics": metrics, "history": history,
            "test_preds": test_preds.tolist(), "test_labels": test_labels.tolist(),
            "test_probs": test_probs.tolist(), "ckpt_path": str(ckpt_path)}


def lr_sweep(cfg: Config, model_name: str, df_train, df_val, df_test, device,
             lrs=(1e-3, 1e-4, 1e-5), num_epochs: int = 8) -> pd.DataFrame:
    """Train one architecture at several learning rates and tabulate the scores."""
    rows = []
    for lr in lrs:
        res = train_model(cfg, model_name, df_train, df_val, df_test, device,
                          lr=lr, num_epochs=num_epochs, tag=f"sweep_lr{lr:g}")
        m = res["metrics"]
        rows.append({
            "model": cfg.model_display[model_name],
            "lr": lr,
            "val_bal_acc": m["best_val_bal_acc"],
            "test_bal_acc": m["balanced_accuracy"],
            "test_f1_weighted": m["f1_weighted"],
            "test_auc_macro": m["auc_macro"],
        })
    return pd.DataFrame(rows)
