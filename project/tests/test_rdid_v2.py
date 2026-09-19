from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from rdid_mosei.interaction import inverse_mobius
from rdid_mosei.rdid_v2 import (
    coordinate_weights,
    interaction_loss,
    interaction_strength_gate,
    weight_statistics,
)
from rdid_mosei.rdid_v2_diagnostics import rankdata, utility_stability, weight_arrays
from rdid_mosei.student import SUBSETS
from train_rdid_v2 import finalize_weight_accumulator, update_weight_accumulator


def statistics():
    mean = torch.tensor([
        [1.0, 2.0, 3.0, -0.2, 0.4, 0.5, 0.1],
        [0.2, -0.1, 0.4, 0.8, -0.3, 0.7, 0.6],
    ])
    variance = torch.tensor([
        [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.5],
        [0.2, 0.1, 0.4, 0.2, 0.5, 0.3, 0.6],
    ])
    utility = torch.arange(1, 8).float()
    return mean, variance, utility


@pytest.mark.parametrize(
    "method",
    [
        "uniform", "soft_ru_a025", "soft_ru_a050", "soft_ru_a075", "ru",
        "r_only", "u_only", "amplitude_u", "soft_ru_gate",
    ],
)
def test_all_rdid_v2_coordinate_weights_are_finite_and_mean_one(method):
    mean, variance, utility = statistics()
    weights = coordinate_weights(
        mean, variance, utility, method,
        soft_alpha=0.5 if method == "soft_ru_gate" else None,
    )
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.mean(-1), torch.ones(len(mean)))


def test_soft_ru_endpoints_equal_uniform_and_original_ru():
    mean, variance, utility = statistics()
    uniform = coordinate_weights(mean, variance, utility, "uniform")
    ru = coordinate_weights(mean, variance, utility, "ru")
    torch.testing.assert_close(uniform, torch.ones_like(uniform))
    torch.testing.assert_close(
        coordinate_weights(mean, variance, utility, "soft_ru_gate", soft_alpha=0.0),
        uniform,
    )
    torch.testing.assert_close(
        coordinate_weights(mean, variance, utility, "soft_ru_gate", soft_alpha=1.0),
        ru,
    )


def test_soft_ru_is_exact_convex_interpolation_at_half():
    mean, variance, utility = statistics()
    uniform = coordinate_weights(mean, variance, utility, "uniform")
    ru = coordinate_weights(mean, variance, utility, "ru")
    half = coordinate_weights(mean, variance, utility, "soft_ru_a050")
    torch.testing.assert_close(half, 0.5 * uniform + 0.5 * ru)


def test_exact_teacher_coordinates_have_zero_gated_loss():
    mean, variance, utility = statistics()
    values = inverse_mobius(mean, 0.3)
    outputs = {subset: {"regression": values[:, index]} for index, subset in enumerate(SUBSETS)}
    loss, weights, gate = interaction_loss(
        outputs, mean, variance, utility, torch.ones(2), 0.3,
        "soft_ru_gate", soft_alpha=0.5, gate_train_median=0.4,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert weights.shape == (2, 7) and gate.shape == (2,)
    assert torch.all((0 <= gate) & (gate <= 1))


def test_gate_uses_only_four_high_order_coordinates():
    mean = torch.tensor([[100.0, 100.0, 100.0, 0.25, 0.5, 0.75, 1.0]])
    gate = interaction_strength_gate(mean, train_median=1.0)
    torch.testing.assert_close(gate, torch.tensor([0.625]))


def test_weight_statistics_report_entropy_and_effective_count():
    weights = torch.ones(3, 7)
    result = weight_statistics(weights, torch.ones(3))
    assert result["effective_coordinate_count"].item() == pytest.approx(7.0)
    assert result["interaction_weight_entropy"].item() == pytest.approx(np.log(7.0))


def test_epoch_weight_accumulator_uses_global_extrema_and_moments():
    accumulator = {
        "coordinate_mass": 0.0, "weight_sum": 0.0, "weight_square_sum": 0.0,
        "weight_min": float("inf"), "weight_max": -float("inf"), "sample_mass": 0.0,
        "entropy_sum": 0.0, "effective_sum": 0.0, "gate_sum": 0.0,
        "gate_active_sum": 0.0,
    }
    first = torch.ones(1, 7)
    second = torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 5.5]])
    update_weight_accumulator(accumulator, first, torch.ones(1), torch.tensor([1.0]))
    update_weight_accumulator(accumulator, second, torch.ones(1), torch.tensor([1.0]))
    result = finalize_weight_accumulator(accumulator)
    assert result["train_interaction_weight_min"] == 0.25
    assert result["train_interaction_weight_max"] == 5.5
    assert result["train_interaction_weight_mean"] == pytest.approx(1.0)
    assert result["train_interaction_weight_std"] > 0


def test_diagnostic_weight_formula_matches_normalization():
    mean, variance, utility_values = statistics()
    result = weight_arrays(mean.numpy(), variance.numpy(), utility_values.numpy())
    np.testing.assert_allclose(result["q"].mean(1), 1.0)
    np.testing.assert_allclose(result["probabilities"].sum(1), 1.0)
    assert np.all((result["effective_coordinate_count"] >= 1) & (result["effective_coordinate_count"] <= 7))


def test_rankdata_uses_average_tie_ranks_and_utility_reports_test():
    np.testing.assert_allclose(rankdata(np.asarray([3.0, 1.0, 1.0, 2.0])), [4.0, 1.5, 1.5, 3.0])
    labels = np.arange(8, dtype=np.float64)
    base = np.stack([labels * (index + 1) for index in range(7)], axis=1)
    report = utility_stability({
        "train": (base, labels),
        "valid": (base[::-1], labels[::-1]),
        "test": (base, labels),
    })
    assert set(report["comparisons"]) == {"train_vs_valid", "train_vs_test"}
