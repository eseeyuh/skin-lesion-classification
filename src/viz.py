"""Plots and summary tables for the analysis."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from .config import Config
from .metrics import expected_calibration_error

try:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
except OSError:  # pragma: no cover - older matplotlib
    pass


def best_model_name(results: Dict) -> str:
    """Name of the model with the highest test balanced accuracy."""
    return max(results, key=lambda n: results[n]["metrics"]["balanced_accuracy"])


def _save(fig, cfg: Config, filename: str, save: bool) -> None:
    if save:
        fig.savefig(cfg.plots_dir / filename, dpi=150, bbox_inches="tight")


def plot_class_distribution(df: pd.DataFrame, cfg: Config, save: bool = True):
    counts = df["dx"].value_counts().reindex(cfg.class_names)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("HAM10000 — Class Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Diagnosis code")
    ax.set_ylabel("Number of images")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 60, f"{v}\n({v / len(df) * 100:.1f}%)", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, cfg, "01_class_distribution.png", save)
    return fig


def plot_demographics(df: pd.DataFrame, cfg: Config, save: bool = True):
    age = df["age"].fillna(df["age"].median())
    age_bin = pd.cut(age, bins=[0, 30, 45, 60, 75, 100],
                     labels=["<30", "30-45", "45-60", "60-75", "75+"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    age_dx = pd.crosstab(age_bin, df["dx"], normalize="index") * 100
    age_dx.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab10")
    axes[0].set_title("Diagnosis Distribution by Age Group (%)")
    axes[0].set_xlabel("Age group")
    axes[0].set_ylabel("% of cases")
    axes[0].legend(title="dx", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[0].tick_params(axis="x", rotation=0)

    sex_dx = pd.crosstab(df["sex"], df["dx"])
    sex_dx.plot(kind="bar", ax=axes[1], colormap="tab10")
    axes[1].set_title("Diagnosis Counts by Sex")
    axes[1].set_xlabel("Sex")
    axes[1].set_ylabel("Count")
    axes[1].legend(title="dx", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, cfg, "02_demographics.png", save)
    return fig


def plot_images_per_lesion(df: pd.DataFrame, cfg: Config, save: bool = True):
    per_lesion = df.groupby("lesion_id").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    per_lesion.value_counts().sort_index().plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Number of Images per Lesion")
    ax.set_xlabel("Images per lesion")
    ax.set_ylabel("Number of lesions")
    fig.tight_layout()
    _save(fig, cfg, "03_images_per_lesion.png", save)
    return fig


def plot_sample_per_class(df: pd.DataFrame, cfg: Config, save: bool = True):
    fig, axes = plt.subplots(1, len(cfg.class_names), figsize=(20, 3.5))
    for ax, cls in zip(axes, cfg.class_names):
        sample = df[df["dx"] == cls].sample(1, random_state=cfg.seed).iloc[0]
        ax.imshow(Image.open(sample["path"]))
        ax.set_title(f"{cls}\n{cfg.class_full[cls]}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample image per class", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, cfg, "04_sample_per_class.png", save)
    return fig


def plot_lr_sweep(sweep_df: pd.DataFrame, cfg: Config, model_name: str = "", save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep_df["lr"], sweep_df["val_bal_acc"], "o-", label="Val balanced accuracy")
    ax.plot(sweep_df["lr"], sweep_df["test_bal_acc"], "s-", label="Test balanced accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Balanced accuracy")
    title = "Learning Rate Sweep" if not model_name else f"{cfg.model_display[model_name]} — Learning Rate Sweep"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, cfg, "05_lr_sweep.png", save)
    return fig


def build_comparison_table(results: Dict, cfg: Config) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append({
            "Model": cfg.model_display[name],
            "Accuracy": m["accuracy"],
            "Balanced Accuracy": m["balanced_accuracy"],
            "F1 (weighted)": m["f1_weighted"],
            "F1 (macro)": m["f1_macro"],
            "AUC (macro)": m["auc_macro"],
            "Train time (s)": m["total_train_time_s"],
            "Throughput (img/s)": m["test_throughput_img_s"],
        })
    return pd.DataFrame(rows)


def plot_model_comparison(comp_df: pd.DataFrame, cfg: Config, save: bool = True):
    metrics = ["Accuracy", "Balanced Accuracy", "F1 (weighted)", "AUC (macro)"]
    x, width = np.arange(len(comp_df)), 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, comp_df[m], width, label=m)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(comp_df["Model"])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    _save(fig, cfg, "06_model_comparison.png", save)
    return fig


def plot_confusion_matrices(results: Dict, cfg: Config, save: bool = True):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    axes = np.atleast_1d(axes)
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(np.array(res["test_labels"]), np.array(res["test_preds"]),
                              labels=np.arange(cfg.num_classes)).astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=cfg.class_names,
                    yticklabels=cfg.class_names, cmap="Blues", ax=ax, cbar=False)
        ax.set_title(cfg.model_display[name])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Confusion Matrices (normalised)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, cfg, "07_confusion_norm.png", save)
    return fig


def per_class_table(results: Dict, cfg: Config) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        for cls, m in res["metrics"]["per_class"].items():
            rows.append({"Model": cfg.model_display[name], "Class": cls,
                         "Class (full)": cfg.class_full[cls],
                         "Precision": m["precision"], "Recall": m["recall"],
                         "F1": m["f1"], "AUC": m.get("auc"), "Support": m["support"]})
    return pd.DataFrame(rows)


def plot_per_class_f1(pc_df: pd.DataFrame, cfg: Config, save: bool = True):
    pivot = pc_df.pivot(index="Class", columns="Model", values="F1").reindex(cfg.class_names)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title("Per-Class F1 Score — Model Comparison")
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Class")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, cfg, "08_per_class_f1.png", save)
    return fig


def plot_misclassified(results: Dict, df_test: pd.DataFrame, cfg: Config,
                       model_name: str = None, n: int = 8, save: bool = True):
    """Most confidently wrong predictions for one model (defaults to the best one)."""
    model_name = model_name or best_model_name(results)
    res = results[model_name]
    labels = np.array(res["test_labels"])
    preds = np.array(res["test_preds"])
    probs = np.array(res["test_probs"])
    mis_idx = np.where(preds != labels)[0]
    confidences = probs[mis_idx, preds[mis_idx]]
    order = np.argsort(-confidences)[:n]

    df_test = df_test.reset_index(drop=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, k in zip(axes.flat, order):
        i = mis_idx[k]
        ax.imshow(Image.open(df_test.iloc[i]["path"]))
        ax.set_title(f"True: {cfg.class_names[labels[i]]}\n"
                     f"Pred: {cfg.class_names[preds[i]]} ({confidences[k]:.2f})",
                     color="darkred", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Most confidently wrong predictions — {cfg.model_display[model_name]}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, cfg, "09_misclassified.png", save)
    return fig


def plot_calibration(results: Dict, cfg: Config, save: bool = True):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (name, res) in zip(axes, results.items()):
        y_true = np.array(res["test_labels"])
        y_prob = np.array(res["test_probs"])
        confidences = y_prob.max(axis=1)
        correct = (y_prob.argmax(axis=1) == y_true).astype(int)
        fop, mpv = calibration_curve(correct, confidences, n_bins=10, strategy="quantile")
        ece = expected_calibration_error(y_true, y_prob)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(mpv, fop, "o-", label=f"Model (ECE={ece:.3f})")
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(cfg.model_display[name])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
    fig.suptitle("Reliability Diagrams", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, cfg, "10_calibration.png", save)
    return fig
