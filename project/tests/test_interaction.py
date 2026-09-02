from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.interaction import inverse_mobius, mobius_transform, random_orthogonal_matrix


def test_mobius_round_trip_batched() -> None:
    generator = torch.Generator().manual_seed(42)
    values = torch.randn(32, 7, generator=generator, dtype=torch.float64)
    baseline = torch.tensor(0.125, dtype=torch.float64)
    reconstructed = inverse_mobius(mobius_transform(values, baseline), baseline)
    assert torch.max(torch.abs(values - reconstructed)).item() < 1e-12


def test_known_three_way_interaction() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]])
    terms = mobius_transform(values, empty_baseline=0.0)
    assert torch.equal(terms, torch.tensor([[1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0]]))


def test_random_coordinate_is_deterministic_and_orthogonal() -> None:
    first = random_orthogonal_matrix(seed=13)
    second = random_orthogonal_matrix(seed=13)
    assert torch.equal(first, second)
    assert torch.allclose(first.T @ first, torch.eye(7, dtype=first.dtype), atol=1e-12)


def test_mobius_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="seven"):
        mobius_transform(torch.zeros(2, 6), 0.0)
