#!/usr/bin/env python3
"""Reproduce and verify the MOSI First geometry matrices.

This is a standalone copy of the exact construction used by
`mechanisms.py::transform()` in the frozen campaign.  It verifies:

* the 7x7 Anchored Moebius matrix and its singular values;
* the three matched-spectrum matrices and their SHA-256;
* the diagonal / weighted / normalized / orthogonal transforms;
* the recorded condition numbers.

Run from the report directory:
    python source/verify_geometry.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
NAMES = ("t", "a", "v", "ta", "tv", "av", "tav")
SUBSET_SETS = tuple(frozenset(name) for name in NAMES)


def mobius_transform(values: torch.Tensor, empty_baseline: torch.Tensor | float) -> torch.Tensor:
    if values.shape[-1] != 7:
        raise ValueError(f"expected seven subset values, got {values.shape}")
    baseline = torch.as_tensor(empty_baseline, dtype=values.dtype, device=values.device)
    terms = []
    for target in SUBSET_SETS:
        result = torch.zeros_like(values[..., 0])
        for index, source in enumerate(SUBSET_SETS):
            if source.issubset(target):
                result = result + ((-1) ** (len(target) - len(source))) * values[..., index]
        result = result + ((-1) ** len(target)) * baseline
        terms.append(result)
    return torch.stack(terms, dim=-1)


def mobius_matrix(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return mobius_transform(torch.eye(7, dtype=dtype), 0.0).T


def random_orthogonal_matrix(size: int = 7, seed: int = 2026, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(size, size, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    signs = torch.where(torch.diag(r) >= 0, 1.0, -1.0).to(dtype=dtype)
    return q * signs.unsqueeze(0)


def build(mechanism: str, matrix_seed: int, normalization: dict) -> torch.Tensor:
    a = mobius_matrix()
    if mechanism == "icd":
        return a
    if mechanism == "matched":
        sigma = torch.linalg.svdvals(a)
        q1 = random_orthogonal_matrix(seed=matrix_seed)
        q2 = random_orthogonal_matrix(seed=matrix_seed + 1000003)
        return q1 @ torch.diag(sigma) @ q2.T
    if mechanism == "orthogonal":
        return random_orthogonal_matrix(seed=matrix_seed)
    if mechanism in ("diagonal", "weighted"):
        weights = (a.T @ a).diag()
        if mechanism == "weighted":
            weights = weights / weights.mean()
        return torch.diag(weights.sqrt())
    if mechanism in ("normalized_icd", "normalized_subset"):
        key = "coordinate_std" if mechanism == "normalized_icd" else "subset_std"
        std = torch.tensor(normalization[key], dtype=torch.float64).clamp_min(normalization["epsilon"])
        return torch.diag(1.0 / std) @ (a if mechanism == "normalized_icd" else torch.eye(7, dtype=torch.float64))
    return torch.eye(7, dtype=torch.float64)


def digest(matrix: torch.Tensor) -> str:
    return hashlib.sha256(matrix.numpy().tobytes()).hexdigest()


def main() -> None:
    geometry = json.loads((BASE / "assets/geometry.json").read_text())
    matched = json.loads((BASE / "assets/matched_matrices.json").read_text())
    normalization = json.loads((BASE / "assets/normalization.json").read_text())
    fingerprints = json.loads((BASE / "fingerprints.json").read_text())

    a = mobius_matrix()
    expected_a = torch.tensor(geometry["matrix"], dtype=torch.float64)
    torch.testing.assert_close(a, expected_a, rtol=0.0, atol=0.0)
    torch.testing.assert_close(torch.linalg.svdvals(a), torch.tensor(geometry["singular_values"], dtype=torch.float64), rtol=0.0, atol=1e-10)
    assert abs(float(torch.linalg.cond(a)) - geometry["condition_number"]) < 1e-9
    assert float((torch.linalg.inv(a) @ a - torch.eye(7, dtype=torch.float64)).abs().max()) < 1e-12
    print(f"A: sha256={digest(a)} cond={float(torch.linalg.cond(a)):.12f}")

    for seed in ("20261111", "20261112", "20261113"):
        m = build("matched", int(seed), normalization)
        expected = torch.tensor(matched[seed]["matrix"], dtype=torch.float64)
        torch.testing.assert_close(m, expected, rtol=0.0, atol=1e-12)
        assert digest(m) == matched[seed]["sha256"]
        torch.testing.assert_close(torch.linalg.svdvals(m), torch.tensor(matched[seed]["singular_values"], dtype=torch.float64), rtol=0.0, atol=1e-10)
        assert abs(float(torch.linalg.cond(m)) - matched[seed]["condition_number"]) < 1e-9
        print(f"matched {seed}: sha256={digest(m)} cond={float(torch.linalg.cond(m)):.12f}")

    seen = {}
    for item in fingerprints["transforms"]:
        key = (item["mechanism"], item["matrix_seed"], item["loss_kind"])
        if key in seen:
            continue
        seen[key] = True
        m = build(item["mechanism"], int(item["matrix_seed"]), normalization)
        assert digest(m) == item["sha256"], (item["mechanism"], digest(m), item["sha256"])
        print(f"{item['mechanism']:18s} seed={item['matrix_seed']} sha256={digest(m)}")

    print("OK: all matrices reproduce the recorded hashes and spectra")


if __name__ == "__main__":
    main()
