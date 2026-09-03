from __future__ import annotations

import torch

SUBSET_NAMES = ("t", "a", "v", "ta", "tv", "av", "tav")
SUBSET_SETS = tuple(frozenset(name) for name in SUBSET_NAMES)


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


def inverse_mobius(terms: torch.Tensor, empty_baseline: torch.Tensor | float) -> torch.Tensor:
    if terms.shape[-1] != 7:
        raise ValueError(f"expected seven interaction terms, got {terms.shape}")
    baseline = torch.as_tensor(empty_baseline, dtype=terms.dtype, device=terms.device)
    values = []
    for target in SUBSET_SETS:
        result = torch.zeros_like(terms[..., 0]) + baseline
        for index, source in enumerate(SUBSET_SETS):
            if source.issubset(target):
                result = result + terms[..., index]
        values.append(result)
    return torch.stack(values, dim=-1)


def random_orthogonal_matrix(size: int = 7, seed: int = 2026, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(size, size, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    signs = torch.where(torch.diag(r) >= 0, 1.0, -1.0).to(dtype=dtype)
    return q * signs.unsqueeze(0)


def mobius_matrix(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return the linear part of the seven-subset Möbius transform."""
    return mobius_transform(torch.eye(7, dtype=dtype), 0.0).T


def random_conditioned_matrix(
    size: int = 7,
    seed: int = 2026,
    condition_number: float | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Create a deterministic random invertible matrix with a fixed condition number."""
    if size < 2:
        raise ValueError("size must be at least two")
    target = float(condition_number or torch.linalg.cond(mobius_matrix(dtype=dtype)).item())
    if not target >= 1.0:
        raise ValueError("condition_number must be at least one")
    left = random_orthogonal_matrix(size=size, seed=seed, dtype=dtype)
    right = random_orthogonal_matrix(size=size, seed=seed + 1_000_003, dtype=dtype)
    singular_values = torch.logspace(0.0, torch.log10(torch.tensor(target, dtype=dtype)).item(), size, dtype=dtype)
    return left @ torch.diag(singular_values) @ right.T
