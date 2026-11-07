from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from rdid_sims.mosei import round_sentiment_class


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


def sentiment_metrics(targets, predictions):
    y, p = _arrays(targets, predictions)
    if np.any(np.abs(y) > 1.000001):
        raise ValueError("Native SIMS labels must remain in [-1,1]")
    y, p = np.clip(y, -1., 1.), np.clip(p, -1., 1.)
    ct, cp = y - y.mean(), p - p.mean()
    denom = math.sqrt(float(np.sum(ct**2) * np.sum(cp**2)))
    truth, predicted = y > 0, p > 0
    nz = y != 0
    acc2 = float(np.mean(truth == predicted))
    f1 = weighted_binary_f1(truth, predicted)
    result = {
        "count": int(y.size), "mae": float(np.abs(y-p).mean()),
        "pearson": float(np.sum(ct*cp)/denom) if denom else float("nan"),
        "acc2": acc2, "f1_weighted": f1,
        "acc2_has_zero": acc2, "f1_weighted_has_zero": f1,
        "acc2_nonzero": float(np.mean(truth[nz] == predicted[nz])) if nz.any() else float("nan"),
        "f1_weighted_nonzero": weighted_binary_f1(truth[nz], predicted[nz]) if nz.any() else float("nan"),
        "nonzero_count": int(nz.sum()),
    }
    for count, boundaries in [(3, [-.1, .1]), (5, [-.7, -.1, .1, .7])]:
        result[f"acc{count}"] = float(np.mean(np.searchsorted(boundaries, y, side="left") == np.searchsorted(boundaries, p, side="left")))
    return result
