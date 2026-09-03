#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.interaction import SUBSET_NAMES, mobius_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate probe-seed interaction means and variances")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, dict[str, dict]]:
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[str(item["parent_sample_id"])][str(item["subset"])] = item
    return rows


def main() -> int:
    args = parse_args()
    if len(args.inputs) < 2:
        raise ValueError("at least two probe seeds are required")
    probes = [load(path) for path in args.inputs]
    sample_ids = sorted(probes[0])
    expected = set(SUBSET_NAMES)
    output_rows = []
    for sample_id in sample_ids:
        interactions = []
        for probe in probes:
            items = probe.get(sample_id)
            if items is None or set(items) != expected:
                raise RuntimeError(f"incomplete probe targets for {sample_id}")
            values = torch.tensor([float(items[name]["probe_score"]) for name in SUBSET_NAMES])
            interactions.append(mobius_transform(values, 0.0))
        stacked = torch.stack(interactions)
        mean = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=True)
        output_rows.append({
            "sample_id": sample_id,
            "split": probes[0][sample_id]["tav"]["split"],
            "probe_seeds": [path.parent.name for path in args.inputs],
            "interaction_mean": {name: float(value) for name, value in zip(SUBSET_NAMES, mean)},
            "interaction_var": {name: float(value) for name, value in zip(SUBSET_NAMES, variance)},
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "samples": len(output_rows), "probe_count": len(probes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
