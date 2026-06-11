"""Evaluation metrics and statistical tests."""

import logging
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from statsmodels.stats.contingency_tables import mcnemar

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                    class_names: Sequence[str]) -> Dict:
    """Headline + per-class metrics for a multi-class classifier."""
    num_classes = len(class_names)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), zero_division=0)
    metrics["per_class"] = {
        class_names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                         "f1": float(f[i]), "support": int(s[i])}
        for i in range(num_classes)
    }
    try:
        y_true_oh = np.eye(num_classes)[y_true]
        aucs = []
        for i in range(num_classes):
            if y_true_oh[:, i].sum() > 0:
                auc_i = roc_auc_score(y_true_oh[:, i], y_prob[:, i])
                metrics["per_class"][class_names[i]]["auc"] = float(auc_i)
                aucs.append(auc_i)
        metrics["auc_macro"] = float(np.mean(aucs))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("AUC computation failed: %s", e)
        metrics["auc_macro"] = None
    return metrics


def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray) -> Dict:
    """McNemar's test on the disagreements between two classifiers."""
    a_correct = (preds_a == labels)
    b_correct = (preds_b == labels)
    b1 = int((a_correct & ~b_correct).sum())
    b2 = int((~a_correct & b_correct).sum())
    res = mcnemar([[0, b1], [b2, 0]], exact=False, correction=True)
    return {"b1_a_only": b1, "b2_b_only": b2,
            "statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def significance_stars(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error using equal-width confidence bins."""
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum():
            ece += (mask.sum() / len(y_true)) * abs(
                confidences[mask].mean() - accuracies[mask].mean())
    return float(ece)
