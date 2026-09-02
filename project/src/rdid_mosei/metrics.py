from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from rdid_mosei.mosei import round_sentiment_class


def _arrays(targets: Iterable[float], predictions: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(list(targets), dtype=np.float64)
    y_pred = np.asarray(list(predictions), dtype=np.float64)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("targets and predictions must be equally sized one-dimensional arrays")
    if y_true.size == 0:
        raise ValueError("metrics require at least one sample")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("targets and predictions must be finite")
    return y_true, y_pred


def weighted_binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = y_true.size
    score = 0.0
    for label in (False, True):
        support = int(np.sum(y_true == label))
        if support == 0:
            continue
        true_positive = int(np.sum((y_true == label) & (y_pred == label)))
        false_positive = int(np.sum((y_true != label) & (y_pred == label)))
        false_negative = int(np.sum((y_true == label) & (y_pred != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else (2 * true_positive) / denominator
        score += support / total * f1
    return score


def sentiment_metrics(targets: Iterable[float], predictions: Iterable[float]) -> dict[str, float | int]:
    y_true, y_pred = _arrays(targets, predictions)
    centered_true = y_true - y_true.mean()
    centered_pred = y_pred - y_pred.mean()
    denominator = math.sqrt(float(np.sum(centered_true**2) * np.sum(centered_pred**2)))
    correlation = float(np.sum(centered_true * centered_pred) / denominator) if denominator else float("nan")

    nonzero_mask = y_true != 0.0
    if np.any(nonzero_mask):
        nonzero_true = y_true[nonzero_mask] > 0.0
        # Both official policies use zero as the non-negative decision boundary for predictions.
        nonzero_pred = y_pred[nonzero_mask] >= 0.0
        acc2_nonzero = float(np.mean(nonzero_true == nonzero_pred))
        f1_nonzero = weighted_binary_f1(nonzero_true, nonzero_pred)
        nonzero_count = int(np.sum(nonzero_mask))
    else:
        acc2_nonzero = float("nan")
        f1_nonzero = float("nan")
        nonzero_count = 0

    has_zero_true = y_true >= 0.0
    has_zero_pred = y_pred >= 0.0
    rounded_true = np.asarray([round_sentiment_class(float(value)) for value in y_true])
    rounded_pred = np.asarray([round_sentiment_class(float(value)) for value in y_pred])
    return {
        "count": int(y_true.size),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "pearson": correlation,
        "acc2_nonzero": acc2_nonzero,
        "f1_weighted_nonzero": f1_nonzero,
        "nonzero_count": nonzero_count,
        "acc2_has_zero": float(np.mean(has_zero_true == has_zero_pred)),
        "f1_weighted_has_zero": weighted_binary_f1(has_zero_true, has_zero_pred),
        "acc7": float(np.mean(rounded_true == rounded_pred)),
    }
