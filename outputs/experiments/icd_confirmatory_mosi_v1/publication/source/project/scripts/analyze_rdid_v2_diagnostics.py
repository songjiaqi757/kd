#!/usr/bin/env python3
"""Compute RDID-v2 Phase-A diagnostics without training a student."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/src"))

from rdid_mosei.rdid_v2_diagnostics import split_diagnostics, utility_stability
from rdid_mosei.student import SUBSETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path,
        default=ROOT / "outputs/experiments/main_table_v1/mosei/assets/protocol.json",
    )
    parser.add_argument(
        "--test-interaction-targets", type=Path,
        help="Optional output of infer_rdid_v2_test_interactions.py",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/experiments/rdid_v2_mosei/diagnostics",
    )
    return parser.parse_args()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def read_targets(path: Path, allowed_splits: set[str]) -> dict[str, list[dict]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty interaction targets: {path}")
    if {row["split"] for row in rows} - allowed_splits:
        raise ValueError(f"unexpected split in {path}")
    identities = [row["parent_sample_id"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate parent interaction target in {path}")
    for row in rows:
        if len(row["mean"]) != len(SUBSETS) or len(row["variance"]) != len(SUBSETS):
            raise ValueError("interaction vectors must have seven coordinates")
    return {
        split: [row for row in rows if row["split"] == split]
        for split in sorted({row["split"] for row in rows})
    }


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray([row["mean"] for row in rows], dtype=np.float64)
    variance = np.asarray([row["variance"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["sentiment"] for row in rows], dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or not np.isfinite(labels).all():
        raise ValueError("diagnostic inputs contain non-finite values")
    return mean, variance, labels


def main() -> None:
    args = parse_args()
    assets = json.loads(args.assets.resolve().read_text())
    utility_values = np.asarray(assets["utility_normalized"], dtype=np.float64)
    grouped = read_targets(Path(assets["interaction_targets"]), {"train", "valid"})
    if set(grouped) != {"train", "valid"}:
        raise ValueError("base interaction targets must contain train and valid")

    sharpness = {
        "schema": "rdid-v2-weight-sharpness-v1",
        "coordinate_order": list(SUBSETS),
        "weight_definition": "Q=Normalize(clip(Normalize(abs(mean)/sqrt(variance+1e-4)),0.25,4)*U_train)",
        "splits": {},
    }
    decomposition = {
        "schema": "rdid-v2-reliability-decomposition-v1",
        "coordinate_order": list(SUBSETS),
        "reliability_definition": "abs(mean)/sqrt(variance+1e-4)",
        "splits": {},
    }
    utility_inputs = {}
    for split in ("train", "valid"):
        mean, variance, labels = arrays(grouped[split])
        sharpness["splits"][split], decomposition["splits"][split] = split_diagnostics(
            mean, variance, utility_values
        )
        utility_inputs[split] = (mean, labels)

    test_status = "not_provided"
    if args.test_interaction_targets is not None:
        test_group = read_targets(args.test_interaction_targets.resolve(), {"test"})
        if set(test_group) != {"test"}:
            raise ValueError("test interaction targets must contain exactly test")
        test_mean, _, test_labels = arrays(test_group["test"])
        utility_inputs["test"] = (test_mean, test_labels)
        test_status = "included_as_declared_mosei_development_information"

    stability = utility_stability(utility_inputs)
    stability.update({
        "schema": "rdid-v2-utility-split-stability-v1",
        "test_status": test_status,
        "method_selection_warning": (
            "Once test utility is read, MOSEI official-test is development information and cannot be claimed as blind confirmation."
        ),
    })
    output = args.output.resolve()
    atomic_json(sharpness, output / "weight_sharpness.json")
    atomic_json(decomposition, output / "reliability_decomposition.json")
    atomic_json(stability, output / "utility_split_stability.json")
    print(json.dumps({
        "status": "complete",
        "test_status": test_status,
        "outputs": [
            str(output / "weight_sharpness.json"),
            str(output / "reliability_decomposition.json"),
            str(output / "utility_split_stability.json"),
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
