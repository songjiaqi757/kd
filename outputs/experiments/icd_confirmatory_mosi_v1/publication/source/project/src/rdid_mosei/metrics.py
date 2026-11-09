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


def polarity_metrics(
    targets: Iterable[float], binary_logits: Iterable[Iterable[float]]
) -> dict[str, float | int]:
    """Metrics for the non-zero MOSEI polarity protocol from a binary head."""

    y_true = np.asarray(list(targets), dtype=np.float64)
    logits = np.asarray(list(binary_logits), dtype=np.float64)
    if y_true.ndim != 1 or logits.ndim != 2 or logits.shape != (len(y_true), 2):
        raise ValueError("binary logits must have shape [samples, 2] and match targets")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(logits)):
        raise ValueError("targets and binary logits must be finite")
    keep = y_true != 0.0
    if not np.any(keep):
        return {
            "binary_count": 0,
            "binary_acc2": float("nan"),
            "binary_f1_weighted": float("nan"),
            "binary_f1_macro": float("nan"),
            "binary_negative_precision": float("nan"),
            "binary_negative_recall": float("nan"),
            "binary_positive_precision": float("nan"),
            "binary_positive_recall": float("nan"),
            "binary_nll": float("nan"),
            "binary_brier": float("nan"),
            "binary_ece_15bin": float("nan"),
        }

    labels = (y_true[keep] > 0.0).astype(np.int64)
    kept_logits = logits[keep]
    shifted = kept_logits - np.max(kept_logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = np.argmax(kept_logits, axis=1)

    per_class_f1 = []
    class_stats: dict[int, tuple[float, float]] = {}
    for label in (0, 1):
        true_positive = int(np.sum((labels == label) & (predictions == label)))
        false_positive = int(np.sum((labels != label) & (predictions == label)))
        false_negative = int(np.sum((labels == label) & (predictions != label)))
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1_denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1.append(0.0 if not f1_denominator else 2 * true_positive / f1_denominator)
        class_stats[label] = (precision, recall)

    chosen = probabilities[np.arange(len(labels)), labels]
    one_hot = np.eye(2, dtype=np.float64)[labels]
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, 16)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if np.any(in_bin):
            ece += float(np.mean(in_bin) * abs(np.mean(correct[in_bin]) - np.mean(confidence[in_bin])))

    return {
        "binary_count": int(len(labels)),
        "binary_acc2": float(np.mean(correct)),
        "binary_f1_weighted": weighted_binary_f1(labels.astype(bool), predictions.astype(bool)),
        "binary_f1_macro": float(np.mean(per_class_f1)),
        "binary_negative_precision": float(class_stats[0][0]),
        "binary_negative_recall": float(class_stats[0][1]),
        "binary_positive_precision": float(class_stats[1][0]),
        "binary_positive_recall": float(class_stats[1][1]),
        "binary_nll": float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))),
        "binary_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "binary_ece_15bin": ece,
    }


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
