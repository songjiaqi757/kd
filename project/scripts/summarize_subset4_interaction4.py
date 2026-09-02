#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path("/home/wy/sjq/kd/outputs/student")
OUTPUT = ROOT / "comparisons" / "subset4_vs_interaction4_three_seed.json"
SEEDS = (2026, 2027, 2028)
METRICS = ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")


def load(method: str, seed: int) -> dict:
    path = ROOT / f"{method}_benchmark500_seed{seed}" / "report.json"
    return json.loads(path.read_text(encoding="utf-8"))["valid_metrics"]


def main() -> int:
    methods = {
        "subset4": {seed: load("subset_value_4", seed) for seed in SEEDS},
        "interaction4": {seed: load("high_order_interaction", seed) for seed in SEEDS},
    }
    summary = {}
    for method, seed_rows in methods.items():
        summary[method] = {
            metric: {
                "values": [float(seed_rows[seed][metric]) for seed in SEEDS],
                "mean": statistics.mean(float(seed_rows[seed][metric]) for seed in SEEDS),
                "sample_standard_deviation": statistics.stdev(float(seed_rows[seed][metric]) for seed in SEEDS),
            }
            for metric in METRICS
        }
    deltas = {
        metric: {
            str(seed): float(methods["interaction4"][seed][metric]) - float(methods["subset4"][seed][metric])
            for seed in SEEDS
        }
        for metric in METRICS
    }
    report = {
        "seeds": list(SEEDS),
        "delta_definition": "interaction4_minus_subset4; lower is better only for MAE",
        "summary": summary,
        "paired_seed_deltas": deltas,
        "high_order_gate_passed": (
            summary["interaction4"]["mae"]["mean"] < summary["subset4"]["mae"]["mean"]
            and summary["interaction4"]["pearson"]["mean"] > summary["subset4"]["pearson"]["mean"]
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
