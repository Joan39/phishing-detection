"""Threshold calibration helpers for phishing probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Return confusion-matrix metrics for phishing-positive labels."""
    scores = np.asarray(probabilities, dtype=float)
    target = np.asarray(labels, dtype=int)
    if scores.ndim != 1 or target.ndim != 1 or len(scores) != len(target):
        raise ValueError("probabilities and labels must be matching 1D arrays.")
    if not np.isfinite(scores).all() or not np.isin(target, [0, 1]).all():
        raise ValueError("probabilities must be finite and labels must be binary.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    predictions = (scores >= threshold).astype(int)
    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    )
    legitimate_count = true_negatives + false_positives
    phishing_count = false_negatives + true_positives
    return {
        "threshold": float(threshold),
        "false_positives": int(false_positives),
        "false_positive_rate": (
            float(false_positives / legitimate_count)
            if legitimate_count
            else None
        ),
        "phishing_recall": (
            float(true_positives / phishing_count) if phishing_count else None
        ),
        "confusion_matrix": [
            [int(true_negatives), int(false_positives)],
            [int(false_negatives), int(true_positives)],
        ],
    }


def select_zero_false_positive_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Select the smallest finite threshold with no observed false positives."""
    scores = np.asarray(probabilities, dtype=float)
    target = np.asarray(labels, dtype=int)
    if scores.ndim != 1 or target.ndim != 1 or len(scores) != len(target):
        raise ValueError("probabilities and labels must be matching 1D arrays.")
    if not np.isfinite(scores).all() or not np.isin(target, [0, 1]).all():
        raise ValueError("probabilities must be finite and labels must be binary.")

    legitimate_scores = scores[target == 0]
    if not len(legitimate_scores):
        raise ValueError("At least one legitimate validation example is required.")
    maximum_legitimate = float(np.max(legitimate_scores))
    threshold = float(np.nextafter(maximum_legitimate, np.inf))
    if threshold > 1.0:
        raise ValueError("No zero-false-positive threshold exists in [0, 1].")
    return threshold
