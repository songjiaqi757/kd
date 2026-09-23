from pathlib import Path
import hashlib
import json
import math
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from rdid_mosei.interaction import inverse_mobius
from rdid_mosei.main_table_kd import (
    CAFDFeatureLoss,
    ProjectorFeatureLoss,
    adapted_interaction_loss,
    entropy_adaptive_weights,
    kd_per_sample,
    probability_kd_per_sample,
    random_orthogonal_loss,
    requires_all_subset_outputs,
    rld_per_sample,
    skd_losses,
    subset_regression_loss,
    teacher_supervision_metadata,
)
from analyze_main_tables import per_seed_cluster_bootstrap
from rdid_mosei.student import SUBSETS


def test_entropy_adaptive_uses_log_class_count_bound():
    uniform = torch.zeros(3, 7)
    weights = entropy_adaptive_weights(uniform, uniform)
    torch.testing.assert_close(weights, torch.full((3,), math.log(7.0)))
    assert not weights.requires_grad


def test_probability_and_logit_kd_are_zero_for_matching_targets():
    logits = torch.randn(4, 7)
    probabilities = torch.softmax(logits / 2.0, -1)
    assert torch.allclose(kd_per_sample(logits, logits, 2.0), torch.zeros(4), atol=1e-6)
    assert torch.allclose(probability_kd_per_sample(logits, probabilities, 2.0), torch.zeros(4), atol=1e-6)


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_skd_is_finite_for_small_batches_and_backpropagates(batch):
    student = torch.randn(batch, 7, requires_grad=True)
    teacher = torch.randn(batch, 7)
    instance, direction, mask = skd_losses(student, teacher)
    loss = instance.mean() + direction.mean()
    loss.backward()
    assert instance.shape == direction.shape == mask.shape == (batch,)
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()


def test_rld_matching_logits_is_finite_and_differentiable():
    student = torch.randn(5, 7, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 6])
    loss = rld_per_sample(student, student.detach(), labels).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()


@pytest.mark.parametrize("batch", [1, 4])
def test_projector_logsum_handles_formal_and_smoke_batches(batch):
    module = ProjectorFeatureLoss(5, 9).train()
    student = torch.randn(batch, 5, requires_grad=True)
    teacher = torch.randn(batch, 9)
    loss = module(student, teacher)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    assert module.projector.weight.grad.norm() > 0


@pytest.mark.parametrize("batch", [1, 4])
def test_cmad_style_cafd_is_finite_and_backpropagates(batch):
    module = CAFDFeatureLoss(5, 9, tau=0.2)
    student = torch.randn(batch, 5, requires_grad=True)
    teacher = torch.randn(batch, 9)
    loss = module(student, teacher)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    if batch > 1:
        assert module.student_projector.weight.grad.norm() > 0
    else:
        # The official cross-sample construction degenerates for a 1x1
        # relation matrix; the common task/Full-KD terms still train the model.
        assert module.student_projector.weight.grad.norm() == 0


def test_subset7_exact_match_is_zero():
    scores = torch.randn(3, 7)
    outputs = {subset: {"regression": scores[:, index]} for index, subset in enumerate(SUBSETS)}
    loss = subset_regression_loss(outputs, scores, torch.tensor([1.0, 0.5, 2.0]))
    assert loss.item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "method",
    [
        "uniform_interaction", "r_only_interaction", "u_only_interaction",
        "ru_interaction", "shuffled_r_u_interaction", "first_order_interaction",
        "first_second_order_interaction",
        "coarse_u_interaction", "amplitude_interaction", "additive_r_u_interaction",
    ],
)
def test_every_interaction_ablation_has_normalized_finite_weights(method):
    mean = torch.tensor([[1.0, 2.0, 3.0, -0.2, 0.4, 0.5, 0.1], [0.2] * 7])
    values = inverse_mobius(mean, 0.3)
    outputs = {subset: {"regression": values[:, index]} for index, subset in enumerate(SUBSETS)}
    loss, weights = adapted_interaction_loss(
        outputs, mean, torch.ones_like(mean), torch.arange(1, 8).float(),
        torch.ones(2), 0.3, method,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.mean(-1), torch.ones(2))


def test_first_second_order_ignores_only_third_order_error():
    mean = torch.tensor([[1.0, 2.0, 3.0, -0.2, 0.4, 0.5, 0.1]])
    values = inverse_mobius(mean, 0.3)
    changed = mean.clone()
    changed[:, -1] += 10.0
    outputs = {
        subset: {"regression": values[:, index]}
        for index, subset in enumerate(SUBSETS)
    }
    loss, weights = adapted_interaction_loss(
        outputs, changed, torch.ones_like(mean), torch.ones(7),
        torch.ones(1), 0.3, "first_second_order_interaction",
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert weights.shape == (1, 6)


def test_random_orthogonal_exact_match_is_zero_and_backpropagates():
    scores = torch.randn(3, 7)
    student = scores.clone().requires_grad_(True)
    outputs = {
        subset: {"regression": student[:, index]}
        for index, subset in enumerate(SUBSETS)
    }
    loss = random_orthogonal_loss(
        outputs, scores, torch.tensor([1.0, 0.5, 2.0]), 20260922
    )
    loss.backward()
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert torch.isfinite(student.grad).all()


def test_amplitude_ablation_replaces_reliability_but_retains_utility():
    mean = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    variance = torch.ones_like(mean)
    utility = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    from rdid_mosei.main_table_kd import interaction_coordinate_weights
    weights = interaction_coordinate_weights(mean, variance, utility, "amplitude_interaction")
    expected = mean.abs() * utility
    expected = expected / expected.mean(-1, keepdim=True)
    torch.testing.assert_close(weights, expected)


def test_frozen_source_commits_and_asset_protocol_are_complete():
    sources = json.loads((ROOT.parent / "external/SOURCES.json").read_text())
    assert {item["method"] for item in sources["repositories"]} == {
        "Projector-based feature KD", "EA-KD", "RLD", "SKD", "DLF", "GsiT", "DPDF-LQ",
        "Light-MER / SWD-H", "CMAD / CAFD",
    }
    assert all(len(item["commit"]) == 40 for item in sources["repositories"])
    for item in sources["repositories"]:
        if patch_name := item.get("local_protocol_patch"):
            patch = ROOT.parent / patch_name
            assert patch.is_file()
            assert hashlib.sha256(patch.read_bytes()).hexdigest() == item["local_protocol_diff_sha256"]
    assets = json.loads((ROOT.parent / "outputs/experiments/main_table_v1/mosei/assets/protocol.json").read_text())
    assert assets["ea_kd"]["entropy_upper_bound"] == "log(7)"
    assert assets["official_test_evaluated"] is False
    assert len(assets["utility_normalized"]) == len(assets["coarse_utility_normalized"]) == 7


def test_teacher_supervision_metadata_distinguishes_information_budget():
    assert teacher_supervision_metadata("adapted_student")["probe_count"] == 0
    assert teacher_supervision_metadata("full_kd")["teacher_subsets"] == ["tav"]
    assert teacher_supervision_metadata("projector")["teacher_hidden_features"] is True
    assert teacher_supervision_metadata("cmad_cafd")["teacher_hidden_features"] is True
    ours = teacher_supervision_metadata("ru_interaction")
    assert ours["probe_count"] == 3
    assert ours["base_tav_probe_count"] == 1
    assert len(ours["teacher_subsets"]) == 7
    assert teacher_supervision_metadata("random_orthogonal")["probe_count"] == 3
    assert requires_all_subset_outputs("ru_interaction", training=True)
    assert not requires_all_subset_outputs("ru_interaction", training=False)
    assert not requires_all_subset_outputs("full_kd", training=True)


def test_bootstrap_reports_per_seed_intervals_without_cross_seed_interval():
    identities = {
        "a": {"sentiment": 0.0, "video_id": "v1"},
        "b": {"sentiment": 1.0, "video_id": "v1"},
        "c": {"sentiment": -1.0, "video_id": "v2"},
    }
    seeds = (13, 42, 2026)
    candidate = {
        seed: {"predictions": {"a": 0.0, "b": 1.0, "c": -1.0}}
        for seed in seeds
    }
    baseline = {
        seed: {"predictions": {"a": 0.2, "b": 0.8, "c": -0.8}}
        for seed in seeds
    }
    result = per_seed_cluster_bootstrap(candidate, baseline, identities, seeds, 200)
    assert result["cross_seed_interval"] is None
    assert set(result["per_seed"]) == {"13", "42", "2026"}
    assert result["mean_candidate_minus_baseline_mae"] == pytest.approx(-0.2)
    assert all(len(row["ci95"]) == 2 for row in result["per_seed"].values())
