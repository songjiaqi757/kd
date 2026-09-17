"""Losses and model additions for the RDID-MSA fixed-student comparison.

The implementations in this module are clean adaptations of the equations in
the frozen upstream repositories recorded in ``external/SOURCES.json``.  They
operate on the project's seven sentiment classes and preserve the common
regression distillation term outside the classification-specific methods.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .student import SUBSETS
from .tav_distillation import TAVDistillationStudent


MAIN_TABLE_METHODS = (
    "adapted_student",
    "full_kd",
    "projector",
    "cmad_cafd",
    "ea_kd",
    "ea_kd_full",
    "rld",
    "skd",
    "subset7",
    "uniform_interaction",
    "r_only_interaction",
    "u_only_interaction",
    "ru_interaction",
    "ensemble_full",
    "shuffled_r_u_interaction",
    "first_order_interaction",
    "coarse_u_interaction",
    "amplitude_interaction",
    "additive_r_u_interaction",
)

SUBSET_METHODS = frozenset(
    {
        "subset7",
        "uniform_interaction",
        "r_only_interaction",
        "u_only_interaction",
        "ru_interaction",
        "shuffled_r_u_interaction",
        "first_order_interaction",
        "coarse_u_interaction",
        "amplitude_interaction",
        "additive_r_u_interaction",
    }
)


def teacher_supervision_metadata(method: str) -> dict:
    """Describe teacher information exposed to one fixed-student method.

    ``probe_count`` counts distinct task Probes used by the method, not the
    number of large-model teachers. Interaction methods retain the public
    single-Probe TAV loss and use the three-Probe ensemble for their extra
    seven-subset target.
    """
    if method not in MAIN_TABLE_METHODS:
        raise ValueError(method)
    if method == "adapted_student":
        return {
            "teacher_subsets": [], "probe_count": 0,
            "base_tav_probe_count": 0, "additional_target_probe_count": 0,
            "teacher_hidden_features": False,
        }
    if method == "ensemble_full":
        return {
            "teacher_subsets": ["tav"], "probe_count": 3,
            "base_tav_probe_count": 0, "additional_target_probe_count": 3,
            "teacher_hidden_features": False,
        }
    if method in SUBSET_METHODS:
        return {
            "teacher_subsets": list(SUBSETS), "probe_count": 3,
            "base_tav_probe_count": 1, "additional_target_probe_count": 3,
            "teacher_hidden_features": False,
        }
    return {
        "teacher_subsets": ["tav"], "probe_count": 1,
        "base_tav_probe_count": 1, "additional_target_probe_count": 0,
        "teacher_hidden_features": method in {"projector", "cmad_cafd"},
    }


def requires_all_subset_outputs(method: str, training: bool) -> bool:
    if method not in MAIN_TABLE_METHODS:
        raise ValueError(method)
    return bool(training and method in SUBSET_METHODS)


def weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or sample_weights.shape != values.shape:
        raise ValueError("values and sample_weights must be equally sized vectors")
    return (values * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def kd_per_sample(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    calibration_temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0 or calibration_temperature <= 0:
        raise ValueError("temperatures must be positive")
    teacher_probabilities = F.softmax(
        teacher_logits.float() / (calibration_temperature * temperature), dim=-1
    )
    student_log_probabilities = F.log_softmax(student_logits.float() / temperature, dim=-1)
    return F.kl_div(
        student_log_probabilities, teacher_probabilities, reduction="none"
    ).sum(-1) * temperature**2


def probability_kd_per_sample(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL to an already calibrated/ensembled teacher probability target."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = teacher_probabilities.float()
    if probabilities.shape != student_logits.shape or not torch.isfinite(probabilities).all():
        raise ValueError("teacher probabilities must be finite and match student logits")
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-12)
    student_log_probabilities = F.log_softmax(student_logits.float() / temperature, dim=-1)
    # Preserve the ensemble target at its calibrated temperature.  The T^2
    # factor retains the public protocol's gradient scale for the student.
    return F.kl_div(student_log_probabilities, probabilities, reduction="none").sum(-1) * temperature**2


def entropy_adaptive_weights(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    entropy_temperature: float = 3.0,
) -> torch.Tensor:
    """EA-KD sample values with the paper-defined H upper bound log(C)."""
    if entropy_temperature <= 0:
        raise ValueError("entropy_temperature must be positive")
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher logits must have shape [batch, classes]")
    classes = student_logits.shape[-1]
    if classes < 2:
        raise ValueError("EA-KD requires at least two classes")

    def entropy(logits: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(logits.float() / entropy_temperature, dim=-1)
        return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)

    teacher_entropy = entropy(teacher_logits)
    student_entropy = entropy(student_logits)
    upper_bound = math.log(classes)
    return ((teacher_entropy + teacher_entropy * student_entropy / upper_bound) / 2.0).detach()


def rld_per_sample(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 8.0,
    temperature: float = 4.0,
    confidence_temperature: float = 1.0,
    standardize_logits: bool = False,
) -> torch.Tensor:
    """Per-sample Refined Logit Distillation (SCD + MCD)."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher logits must have shape [batch, classes]")
    if labels.shape != student_logits.shape[:1]:
        raise ValueError("labels must have shape [batch]")
    if temperature <= 0 or confidence_temperature <= 0:
        raise ValueError("temperatures must be positive")

    def normalize(logits: torch.Tensor) -> torch.Tensor:
        return (logits - logits.mean(-1, keepdim=True)) / logits.std(
            -1, keepdim=True, unbiased=True
        ).clamp_min(1e-7)

    student = normalize(student_logits.float()) if standardize_logits else student_logits.float()
    teacher = normalize(teacher_logits.float()) if standardize_logits else teacher_logits.float()
    classes = student.shape[-1]
    label_mask = F.one_hot(labels.long(), classes).bool()
    teacher_argmax_mask = F.one_hot(teacher.argmax(-1), classes).bool()

    student_probabilities = F.softmax(student / confidence_temperature, dim=-1)
    teacher_probabilities = F.softmax(teacher / confidence_temperature, dim=-1)
    student_binary = torch.stack(
        ((student_probabilities * label_mask).sum(-1),
         (student_probabilities * ~label_mask).sum(-1)), dim=-1
    )
    teacher_binary = torch.stack(
        ((teacher_probabilities * teacher_argmax_mask).sum(-1),
         (teacher_probabilities * ~teacher_argmax_mask).sum(-1)), dim=-1
    )
    scd = F.kl_div(
        student_binary.clamp_min(1e-12).log(), teacher_binary, reduction="none"
    ).sum(-1) * confidence_temperature**2

    ground_truth_teacher_value = teacher.gather(1, labels.long().unsqueeze(1))
    misleading = teacher >= ground_truth_teacher_value
    masked_student = (student / temperature).masked_fill(misleading, -1e9)
    masked_teacher = (teacher / temperature).masked_fill(misleading, -1e9)
    mcd = F.kl_div(
        F.log_softmax(masked_student, dim=-1),
        F.softmax(masked_teacher, dim=-1),
        reduction="none",
    ).sum(-1) * temperature**2
    return alpha * scd + beta * mcd


def skd_losses(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 4.0,
    tik_factor: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paper-faithful instance and direction knowledge for SKD.

    The upstream commit reduces KL and direction losses before multiplying by
    masks.  Here the low-confidence mask is applied to the per-instance KL as
    described by the paper.  Direction knowledge uses the full normalized
    Gram matrix.  A manual covariance with a defined batch-one limit avoids the
    NaNs produced by ``torch.cov`` for the project's possible final mini-batch.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher logits must have shape [batch, classes]")
    if tik_factor <= 0:
        raise ValueError("tik_factor must be positive")
    teacher_probabilities = F.softmax(teacher_logits.float(), dim=-1)
    confidence = teacher_probabilities.max(-1).values.detach()
    threshold = torch.quantile(confidence, 0.5)
    instance_mask = confidence <= threshold
    instance = kd_per_sample(student_logits, teacher_logits, temperature)

    student_direction = F.normalize(student_logits.float(), p=2, dim=-1, eps=1e-12)
    teacher_direction = F.normalize(teacher_logits.float(), p=2, dim=-1, eps=1e-12)
    classes = student_logits.shape[-1]
    difference = (
        student_direction @ student_direction.T - teacher_direction @ teacher_direction.T
    ) / classes
    batch = difference.shape[0]
    centered = difference - difference.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(batch - 1, 1)
    covariance = covariance + tik_factor * torch.eye(batch, device=difference.device)
    cholesky = torch.linalg.cholesky(covariance)
    whitened = torch.cholesky_solve(difference.T, cholesky).T
    squared_distance = (difference * whitened).sum(-1).clamp_min(0)
    # The epsilon makes the derivative defined when batch=1 yields an exactly
    # zero one-by-one Gram difference.
    direction = torch.sqrt(squared_distance + 1e-12)
    if not torch.isfinite(direction).all():
        raise FloatingPointError("non-finite SKD direction loss")
    return instance, direction, instance_mask


class ProjectorFeatureLoss(nn.Module):
    """Linear projector + non-affine BN + fourth-power LogSum distance."""

    def __init__(self, student_dim: int, teacher_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.projector = nn.Linear(student_dim, teacher_dim)
        self.student_norm = nn.BatchNorm1d(teacher_dim, eps=1e-4, affine=False)
        self.teacher_norm = nn.BatchNorm1d(teacher_dim, eps=1e-4, affine=False)
        self.eps = eps

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        if student_features.ndim != 2 or teacher_features.ndim != 2:
            raise ValueError("projector features must have shape [batch, dimension]")
        projected = self.projector(student_features.float())
        if student_features.shape[0] < 2 and self.training:
            # Defined smoke/final-remainder behavior; formal runs normally use
            # batch 8. Running statistics are checkpointed with the projector.
            projected = F.batch_norm(
                projected, self.student_norm.running_mean, self.student_norm.running_var,
                training=False, eps=self.student_norm.eps,
            )
            target = F.batch_norm(
                teacher_features.float(), self.teacher_norm.running_mean, self.teacher_norm.running_var,
                training=False, eps=self.teacher_norm.eps,
            )
        else:
            projected = self.student_norm(projected)
            target = self.teacher_norm(teacher_features.float())
        return torch.log((projected - target).abs().pow(4).sum() + self.eps)


class CAFDFeatureLoss(nn.Module):
    """CMAD-style correlation-aware feature distillation adaptation.

    The official CMAD student and teacher share a representation dimension.
    RDID-MSA adds a trainable student-only projection because its fused student
    and cached teacher representations are 512-D and 2048-D respectively.
    """

    def __init__(self, student_dim: int, teacher_dim: int, tau: float = 0.2) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("CAFD temperature must be positive")
        self.student_projector = nn.Linear(student_dim, teacher_dim)
        self.tau = tau

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        if student_features.ndim != 2 or teacher_features.ndim != 2:
            raise ValueError("CAFD features must have shape [batch, dimension]")
        student = self.student_projector(student_features.float())
        teacher = teacher_features.detach().float()
        if student.shape != teacher.shape:
            raise ValueError("projected student and teacher CAFD features must match")
        mse = F.mse_loss(student, teacher, reduction="none").mean(-1)
        student_teacher = F.normalize(student, dim=-1) @ F.normalize(teacher, dim=-1).T
        teacher_teacher = F.normalize(teacher, dim=-1) @ F.normalize(teacher, dim=-1).T

        def row_symmetric_kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            left_log = F.log_softmax(left / self.tau, dim=-1)
            right_log = F.log_softmax(right / self.tau, dim=-1)
            left_probability, right_probability = left_log.exp(), right_log.exp()
            return 0.5 * (
                F.kl_div(left_log, right_probability, reduction="none").sum(-1)
                + F.kl_div(right_log, left_probability, reduction="none").sum(-1)
            )

        correlation = row_symmetric_kl(student_teacher, teacher_teacher)
        difference = (teacher_teacher - student_teacher).abs()
        diagonal_ratio = difference.diagonal().sum() / difference.sum().clamp_min(1e-8)
        return (correlation * mse).mean() + diagonal_ratio


class MainTableKDStudent(TAVDistillationStudent):
    """M3-capacity student shared by every fixed-student comparison row."""

    def __init__(self, *args, comparison_method: str, teacher_feature_dim: int = 2048, cmad_tau: float = 0.2, **kwargs):
        if comparison_method not in MAIN_TABLE_METHODS:
            raise ValueError(comparison_method)
        # M3 supplies identical T/A/V LoRA scope without changing frozen P0 code.
        super().__init__(*args, method="M3", **kwargs)
        self.comparison_method = comparison_method
        if comparison_method == "projector":
            self.feature_loss = ProjectorFeatureLoss(self.fusion.hidden_size, teacher_feature_dim)
        elif comparison_method == "cmad_cafd":
            self.feature_loss = CAFDFeatureLoss(self.fusion.hidden_size, teacher_feature_dim, cmad_tau)
        else:
            self.feature_loss = None

    def forward(self, **kwargs):
        # Seven subset heads are a training-only supervision mechanism.  Valid
        # and test deployment execute the complete TAV path exactly once.
        kwargs["all_subsets"] = requires_all_subset_outputs(self.comparison_method, self.training)
        return super().forward(**kwargs)


def subset_regression_loss(
    outputs: Mapping[str, Mapping[str, torch.Tensor]],
    teacher_subset_scores: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    values = torch.stack([outputs[subset]["regression"].float() for subset in SUBSETS], -1)
    if teacher_subset_scores.shape != values.shape:
        raise ValueError("teacher subset scores must have shape [batch, 7]")
    per_sample = F.smooth_l1_loss(values, teacher_subset_scores.float(), reduction="none").mean(-1)
    return weighted_mean(per_sample, sample_weights)


def interaction_coordinate_weights(
    mean: torch.Tensor,
    variance: torch.Tensor,
    utility: torch.Tensor,
    method: str,
) -> torch.Tensor:
    if mean.shape != variance.shape or mean.shape[-1] != len(SUBSETS):
        raise ValueError("interaction statistics must have shape [batch, 7]")
    reliability = mean.abs() / torch.sqrt(variance.clamp_min(0) + 1e-4)
    reliability = (reliability / reliability.mean(-1, keepdim=True).clamp_min(1e-8)).clamp(0.25, 4.0)
    utility = utility.to(mean).expand_as(mean)
    if method == "uniform_interaction" or method == "first_order_interaction":
        result = torch.ones_like(mean)
    elif method == "r_only_interaction" or method == "shuffled_r_u_interaction":
        result = reliability
        if method == "shuffled_r_u_interaction":
            result = result * utility
    elif method == "u_only_interaction":
        result = utility
    elif method == "ru_interaction" or method == "coarse_u_interaction":
        result = reliability * utility
    elif method == "amplitude_interaction":
        result = mean.abs() * utility
    elif method == "additive_r_u_interaction":
        result = reliability + utility
    else:
        raise ValueError(method)
    return (result / result.mean(-1, keepdim=True).clamp_min(1e-8)).detach()


def adapted_interaction_loss(
    outputs: Mapping[str, Mapping[str, torch.Tensor]],
    mean: torch.Tensor,
    variance: torch.Tensor,
    utility: torch.Tensor,
    sample_weights: torch.Tensor,
    empty_baseline: float,
    method: str,
    weight_mean: torch.Tensor | None = None,
    weight_variance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .interaction import mobius_transform

    values = torch.stack([outputs[subset]["regression"].float() for subset in SUBSETS], -1)
    coordinates = mobius_transform(values, empty_baseline)
    weights = interaction_coordinate_weights(
        mean if weight_mean is None else weight_mean,
        variance if weight_variance is None else weight_variance,
        utility,
        method,
    )
    errors = F.smooth_l1_loss(coordinates, mean, reduction="none")
    if method == "first_order_interaction":
        errors = errors[:, :3]
        weights = weights[:, :3]
        weights = weights / weights.mean(-1, keepdim=True).clamp_min(1e-8)
    per_sample = (weights * errors).mean(-1)
    return weighted_mean(per_sample, sample_weights), weights
