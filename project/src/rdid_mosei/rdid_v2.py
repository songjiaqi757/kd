"""RDID-v2 interaction weighting and sample-gating primitives.

This module is deliberately separate from the frozen M0--M6 implementation.
It keeps the original reliability definition intact while making the strength
of coordinate selection explicit and testable.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from .interaction import mobius_transform
from .student import SUBSETS


RDID_V2_METHODS = (
    "uniform",
    "soft_ru_a025",
    "soft_ru_a050",
    "soft_ru_a075",
    "ru",
    "r_only",
    "u_only",
    "amplitude_u",
    "soft_ru_gate",
)

SOFT_RU_ALPHA = {
    "uniform": 0.0,
    "soft_ru_a025": 0.25,
    "soft_ru_a050": 0.50,
    "soft_ru_a075": 0.75,
    "ru": 1.0,
}


def _normalize_mean_one(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.shape[-1] != len(SUBSETS):
        raise ValueError("coordinate values must have shape [batch, 7]")
    if not torch.isfinite(values).all() or torch.any(values < 0):
        raise ValueError("coordinate values must be finite and non-negative")
    denominator = values.mean(-1, keepdim=True)
    if torch.any(denominator <= 0):
        raise ValueError("every sample must have positive total coordinate weight")
    return values / denominator


def reliability_weights(
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    epsilon: float = 1e-4,
    clip: tuple[float, float] = (0.25, 4.0),
) -> torch.Tensor:
    """Reproduce the frozen M6 reliability calculation exactly."""
    if mean.shape != variance.shape or mean.ndim != 2 or mean.shape[-1] != len(SUBSETS):
        raise ValueError("interaction mean/variance must have shape [batch, 7]")
    if epsilon <= 0 or clip[0] <= 0 or clip[0] > clip[1]:
        raise ValueError("invalid reliability settings")
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise ValueError("interaction statistics must be finite")
    raw = mean.abs() / torch.sqrt(variance.clamp_min(0) + epsilon)
    return _normalize_mean_one(raw).clamp(*clip)


def coordinate_weights(
    mean: torch.Tensor,
    variance: torch.Tensor,
    utility: torch.Tensor,
    method: str,
    *,
    soft_alpha: float | None = None,
) -> torch.Tensor:
    """Return detached per-sample coordinate weights with mean exactly one."""
    if method not in RDID_V2_METHODS:
        raise ValueError(method)
    utility = utility.to(mean)
    if utility.shape != (len(SUBSETS),):
        raise ValueError("utility must have shape [7]")
    if not torch.isfinite(utility).all() or torch.any(utility < 0) or utility.sum() <= 0:
        raise ValueError("utility must be finite, non-negative, and nonzero")
    expanded_utility = utility.expand_as(mean)
    reliability = reliability_weights(mean, variance)

    if method in SOFT_RU_ALPHA or method == "soft_ru_gate":
        alpha = SOFT_RU_ALPHA.get(method, soft_alpha)
        if alpha is None or not 0.0 <= alpha <= 1.0:
            raise ValueError("soft-RU alpha must be in [0, 1]")
        ru = _normalize_mean_one(reliability * expanded_utility)
        result = (1.0 - alpha) + alpha * ru
    elif method == "r_only":
        result = reliability
    elif method == "u_only":
        result = expanded_utility
    elif method == "amplitude_u":
        result = mean.abs() * expanded_utility
    else:  # pragma: no cover - guarded by the method set above.
        raise ValueError(method)
    return _normalize_mean_one(result).detach()


def interaction_strength_gate(mean: torch.Tensor, train_median: float) -> torch.Tensor:
    """Non-learned high-order interaction gate from the frozen teacher target."""
    if mean.ndim != 2 or mean.shape[-1] != len(SUBSETS):
        raise ValueError("interaction mean must have shape [batch, 7]")
    if not torch.isfinite(mean).all() or not train_median > 0:
        raise ValueError("finite means and a positive train median are required")
    strength = mean[:, 3:].abs().mean(-1)
    return (strength / train_median).clamp(0.0, 1.0).detach()


def interaction_loss(
    outputs: Mapping[str, Mapping[str, torch.Tensor]],
    mean: torch.Tensor,
    variance: torch.Tensor,
    utility: torch.Tensor,
    sample_weights: torch.Tensor,
    empty_baseline: float,
    method: str,
    *,
    soft_alpha: float | None = None,
    gate_train_median: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the RDID-v2 interaction loss, weights, and sample gate."""
    values = torch.stack([outputs[subset]["regression"].float() for subset in SUBSETS], -1)
    coordinates = mobius_transform(values, empty_baseline)
    weights = coordinate_weights(mean, variance, utility, method, soft_alpha=soft_alpha)
    gate = (
        interaction_strength_gate(mean, gate_train_median)
        if method == "soft_ru_gate"
        else torch.ones(mean.shape[0], dtype=mean.dtype, device=mean.device)
    )
    errors = F.smooth_l1_loss(coordinates, mean, reduction="none")
    per_sample = (weights * errors).mean(-1) * gate
    loss = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
    return loss, weights, gate


def weight_statistics(weights: torch.Tensor, gate: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Batch statistics used by the training-history audit."""
    if weights.ndim != 2 or weights.shape[-1] != len(SUBSETS):
        raise ValueError("weights must have shape [batch, 7]")
    probabilities = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    effective = probabilities.square().sum(-1).reciprocal()
    result = {
        "interaction_weight_mean": weights.mean(),
        "interaction_weight_std": weights.std(unbiased=False),
        "interaction_weight_min": weights.min(),
        "interaction_weight_max": weights.max(),
        "interaction_weight_entropy": entropy.mean(),
        "effective_coordinate_count": effective.mean(),
    }
    if gate is not None:
        result["interaction_gate_mean"] = gate.mean()
        result["interaction_gate_active_fraction"] = (gate > 0).to(weights).mean()
    return result
