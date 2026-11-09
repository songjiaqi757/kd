"""Predeclared exploratory ICD variants; no additional inference parameters."""
from __future__ import annotations

import torch

ROUTED_VARIANTS = ('order_routed', 'scrambled_routed', 'fusion_only', 'dgl_icd')
TARGET_VARIANTS = ('ensemble_source', 'target_uniform', 'target_variance')
VARIANTS = ROUTED_VARIANTS + TARGET_VARIANTS + ('late_decay',)


def encoder_objective(variant, common, coordinates, coefficient, unimodal=None):
    """Keep each coordinate's original 1/7 scale, including routing controls."""
    if coordinates.shape != (7,):
        raise ValueError('Expected seven reduced coordinate losses')
    if variant == 'order_routed':
        return common + coefficient * coordinates[:3].sum() / 7
    if variant == 'scrambled_routed':
        # A fixed three-coordinate control; no random draws during training.
        return common + coefficient * coordinates[[0, 3, 6]].sum() / 7
    if variant == 'fusion_only':
        return common
    if variant == 'dgl_icd' and unimodal is not None:
        return unimodal
    raise ValueError(variant)


def corrected_coordinates(mean, variance, labels, baseline, variant, alpha=0.5):
    """Allocate the same full-score label correction across seven teacher terms.

    Probe variance is a heuristic instability proxy, not calibrated uncertainty.
    This is a pilot of target correction, not the cross-fitted posterior proposal.
    """
    if mean.shape != variance.shape or mean.shape[-1] != 7 or labels.shape != mean.shape[:-1]:
        raise ValueError('Coordinate/variance/label dimensions do not agree')
    if not 0 <= alpha <= 1 or variant not in TARGET_VARIANTS:
        raise ValueError('Invalid target correction')
    mean, variance, labels = mean.detach().float(), variance.detach().float(), labels.detach().float()
    if not all(torch.isfinite(value).all() for value in (mean, variance, labels)):
        raise ValueError('Nonfinite target input')
    if (variance < 0).any():
        raise ValueError('Negative probe variance')
    if variant == 'ensemble_source':
        return mean
    weights = torch.ones_like(mean)
    if variant == 'target_variance':
        weights = variance + 0.1 * variance.mean(-1, keepdim=True) + 1e-6
    weights = weights / weights.sum(-1, keepdim=True)
    residual = labels - baseline - mean.sum(-1)
    return mean + alpha * weights * residual.unsqueeze(-1)


def decay_coefficient(epoch, epochs):
    """Full ICD through the midpoint, linearly reduced to zero at final epoch."""
    if not 1 <= epoch <= epochs or epochs < 2:
        raise ValueError('Invalid epoch')
    midpoint = epochs // 2
    return 1.0 if epoch <= midpoint else (epochs - epoch) / (epochs - midpoint)


def assign_routed_gradients(encoder_loss, fusion_loss, encoder_parameters, fusion_parameters):
    enc = list(encoder_parameters)
    fus = list(fusion_parameters)
    if not enc or not fus or {id(p) for p in enc} & {id(p) for p in fus}:
        raise ValueError('Routing requires disjoint nonempty parameter groups')
    # autograd.grad only computes the requested group's update, with no hooks on shared features.
    ge = torch.autograd.grad(encoder_loss, enc, retain_graph=True, allow_unused=True)
    gf = torch.autograd.grad(fusion_loss, fus, allow_unused=True)
    for parameter, gradient in zip(enc + fus, list(ge) + list(gf)):
        parameter.grad = None if gradient is None else gradient.detach()
