"""Soft-voting ensemble over trained models."""

from typing import Dict, Sequence, Tuple

import numpy as np


def soft_vote(results: Dict[str, dict], model_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Average the per-model test probabilities and return (probs, predictions).

    `results` maps model name -> the dict returned by `engine.train_model`.
    All models must have been evaluated on the same ordered test set.
    """
    probs = np.mean([np.array(results[name]["test_probs"]) for name in model_names], axis=0)
    preds = probs.argmax(axis=1)
    return probs, preds
