"""Decision-space supervision on the existing scalar sentiment output.

The teacher target is sigmoid(mean probe score / temperature), not a
calibrated probability or the mean of per-probe sigmoid probabilities.
All three controls use the same target, Huber distance and scalar weight.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F

from .interaction import mobius_transform
from .metrics import sentiment_metrics
from .student import SUBSETS

DECISION_MODES = ("none", "full", "subset", "interaction")


def decision_distillation_loss(outputs, teacher_scores, sample_weights,
                               empty_baseline, *, mode, temperature=1.0):
    if mode not in DECISION_MODES:
        raise ValueError(f"unknown decision mode: {mode}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scores = torch.stack([outputs[s]["regression"].float() for s in SUBSETS], -1)
    if teacher_scores.shape != scores.shape:
        raise ValueError("teacher scores must have shape [batch, 7]")
    if sample_weights.shape != scores.shape[:1]:
        raise ValueError("sample weights must have shape [batch]")
    if mode == "none":
        return scores.sum() * 0.0
    student = torch.sigmoid(scores / temperature)
    teacher = torch.sigmoid(teacher_scores.detach().float() / temperature)
    if mode == "full":
        student, teacher = student[:, -1:], teacher[:, -1:]
    elif mode == "interaction":
        empty = torch.sigmoid(torch.as_tensor(
            empty_baseline, dtype=scores.dtype, device=scores.device
        ) / temperature)
        student = mobius_transform(student, empty)
        teacher = mobius_transform(teacher, empty)
    per_sample = F.smooth_l1_loss(student, teacher, reduction="none").mean(-1)
    weights = sample_weights.to(per_sample)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def error_strata(predictions, boundary=0.5):
    """Predeclared |GT| bins; neutral excluded only from polarity groups.

    MAE contributions sum across the three disjoint error groups plus neutral.
    These are descriptive diagnostics, not independent held-out estimates.
    """
    if boundary <= 0 or not predictions:
        raise ValueError("positive boundary and nonempty predictions required")
    if any(p.get("split") != "valid" for p in predictions):
        raise ValueError("development diagnostics accept validation only")
    y = np.array([p["target_sentiment"] for p in predictions], dtype=float)
    pred = np.array([p["prediction"] for p in predictions], dtype=float)
    if not (np.isfinite(y).all() and np.isfinite(pred).all()):
        raise ValueError("nonfinite predictions")
    nonzero = y != 0
    correct = (pred >= 0) == (y > 0)
    masks = {
        "neutral": ~nonzero,
        "near_zero": nonzero & (np.abs(y) <= boundary),
        "away_from_zero": np.abs(y) > boundary,
        "near_zero_wrong_sign": nonzero & (np.abs(y) <= boundary) & ~correct,
        "away_from_zero_wrong_sign": (np.abs(y) > boundary) & ~correct,
        "correct_sign": nonzero & correct,
    }
    result = {"boundary_abs_target": boundary, "count": len(y), "groups": {}}
    for name, mask in masks.items():
        n = int(mask.sum())
        result["groups"][name] = {
            "count": n,
            "mae": float(np.abs(pred[mask] - y[mask]).mean()) if n else None,
            "total_mae_contribution": float(np.abs(pred[mask] - y[mask]).sum() / len(y)),
            "acc2_nonzero": float(correct[mask].mean()) if n and name != "neutral" else None,
        }
    result["metrics"] = {
        k: v if not isinstance(v, float) or math.isfinite(v) else None
        for k, v in sentiment_metrics(y, pred).items()
    }
    return result


def clustered_paired_delta(reference, candidate, *, repetitions=2000, seed=20261003):
    """Paired source-video bootstrap; positive MAE delta means degradation."""
    ref = {r["parent_sample_id"]: r for r in reference}
    cand = {r["parent_sample_id"]: r for r in candidate}
    if len(ref) != len(reference) or len(cand) != len(candidate) or ref.keys() != cand.keys():
        raise ValueError("paired predictions must have identical unique utterances")
    # Reuse strict validation-only and finite-value checks.
    error_strata(reference)
    error_strata(candidate)
    groups = {}
    for key in sorted(ref):
        a, b = ref[key], cand[key]
        if a["target_sentiment"] != b["target_sentiment"] or a["video_id"] != b["video_id"]:
            raise ValueError("paired target/video mismatch")
        y, pa, pb = a["target_sentiment"], a["prediction"], b["prediction"]
        record = groups.setdefault(a["video_id"], np.zeros(4))
        record += [abs(pb-y)-abs(pa-y), 1,
                   int((pb >= 0) == (y > 0))-int((pa >= 0) == (y > 0)) if y != 0 else 0,
                   int(y != 0)]
    values = np.array(list(groups.values()))
    if not len(values) or repetitions <= 0:
        raise ValueError("nonempty clusters and positive repetitions required")
    sums = values.sum(0)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        draw = values[rng.integers(len(values), size=len(values))].sum(0)
        draws.append([draw[0]/draw[1], draw[2]/draw[3] if draw[3] else np.nan])
    draws = np.array(draws)
    result = {"video_clusters": len(values), "repetitions": repetitions,
              "scope": "validation development; conditional on selected checkpoints, not selection-adjusted"}
    for i, (name, numerator, denominator) in enumerate((("mae", 0, 1), ("acc2_nonzero", 2, 3))):
        finite = draws[:, i][np.isfinite(draws[:, i])]
        result[name] = {
            "candidate_minus_reference": float(sums[numerator]/sums[denominator]) if sums[denominator] else None,
            "cluster_bootstrap_95_ci": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
        }
    return result
