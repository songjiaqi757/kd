"""Numerical diagnostics for RDID-v2 teacher interaction targets."""
from __future__ import annotations

import numpy as np

from .student import SUBSETS


QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("summary values must be nonempty and finite")
    quantiles = np.quantile(values, QUANTILES)
    return {
        "mean": float(values.mean()),
        "median": float(quantiles[2]),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("Pearson inputs must be equally sized vectors")
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(np.square(left).sum() * np.square(right).sum())
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, matching the standard Spearman definition."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pearson(rankdata(left), rankdata(right))


def utility(interactions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    interactions = np.asarray(interactions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if interactions.ndim != 2 or interactions.shape[1] != len(SUBSETS):
        raise ValueError("interactions must have shape [samples, 7]")
    if labels.shape != interactions.shape[:1]:
        raise ValueError("labels must match the interaction sample dimension")
    return np.asarray([abs(pearson(interactions[:, index], labels)) for index in range(len(SUBSETS))])


def weight_arrays(mean: np.ndarray, variance: np.ndarray, utility_values: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    utility_values = np.asarray(utility_values, dtype=np.float64)
    if mean.shape != variance.shape or mean.ndim != 2 or mean.shape[1] != len(SUBSETS):
        raise ValueError("mean and variance must have shape [samples, 7]")
    if utility_values.shape != (len(SUBSETS),):
        raise ValueError("utility must have shape [7]")
    amplitude = np.abs(mean)
    inverse_std = 1.0 / np.sqrt(np.maximum(variance, 0.0) + 1e-4)
    raw_reliability = amplitude * inverse_std
    normalized_reliability = raw_reliability / np.maximum(raw_reliability.mean(1, keepdims=True), 1e-8)
    reliability = np.clip(normalized_reliability, 0.25, 4.0)
    product = reliability * utility_values[None, :]
    q = product / np.maximum(product.mean(1, keepdims=True), 1e-8)
    probabilities = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(1)
    effective = 1.0 / np.square(probabilities).sum(1)
    return {
        "amplitude": amplitude,
        "inverse_std": inverse_std,
        "raw_reliability": raw_reliability,
        "normalized_reliability": normalized_reliability,
        "reliability": reliability,
        "product": product,
        "q": q,
        "probabilities": probabilities,
        "entropy": entropy,
        "effective_coordinate_count": effective,
    }


def split_diagnostics(mean: np.ndarray, variance: np.ndarray, utility_values: np.ndarray) -> tuple[dict, dict]:
    arrays = weight_arrays(mean, variance, utility_values)
    maximum = arrays["q"].argmax(1)
    sharpness = {
        "samples": int(len(mean)),
        "entropy": summarize(arrays["entropy"]),
        "effective_coordinate_count": summarize(arrays["effective_coordinate_count"]),
        "coordinate_statistics": {},
    }
    for index, coordinate in enumerate(SUBSETS):
        sharpness["coordinate_statistics"][coordinate] = {
            "mean_reliability": float(arrays["reliability"][:, index].mean()),
            "utility": float(utility_values[index]),
            "mean_r_times_u": float(arrays["product"][:, index].mean()),
            "median_r_times_u": float(np.median(arrays["product"][:, index])),
            "p90_r_times_u": float(np.quantile(arrays["product"][:, index], 0.90)),
            "mean_normalized_q": float(arrays["q"][:, index].mean()),
            "maximum_weight_frequency": float(np.mean(maximum == index)),
        }
    decomposition = {
        "samples": int(len(mean)),
        "pooled_correlation": {
            "reliability_vs_abs_mean": pearson(arrays["raw_reliability"], arrays["amplitude"]),
            "reliability_vs_inverse_std": pearson(arrays["raw_reliability"], arrays["inverse_std"]),
        },
        "per_coordinate_correlation": {
            coordinate: {
                "reliability_vs_abs_mean": pearson(arrays["raw_reliability"][:, index], arrays["amplitude"][:, index]),
                "reliability_vs_inverse_std": pearson(arrays["raw_reliability"][:, index], arrays["inverse_std"][:, index]),
            }
            for index, coordinate in enumerate(SUBSETS)
        },
        "clip_fraction": {
            "at_0.25": float(np.mean(arrays["normalized_reliability"] <= 0.25)),
            "at_4.0": float(np.mean(arrays["normalized_reliability"] >= 4.0)),
        },
    }
    return sharpness, decomposition


def utility_stability(split_values: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    utilities = {
        split: utility(interactions, labels)
        for split, (interactions, labels) in split_values.items()
    }
    result = {
        "coordinate_order": list(SUBSETS),
        "utility_raw_abs_pearson": {
            split: values.tolist() for split, values in utilities.items()
        },
        "comparisons": {},
    }
    for right in ("valid", "test"):
        if "train" in utilities and right in utilities:
            result["comparisons"][f"train_vs_{right}"] = {
                "pearson": pearson(utilities["train"], utilities[right]),
                "spearman": spearman(utilities["train"], utilities[right]),
            }
    return result
