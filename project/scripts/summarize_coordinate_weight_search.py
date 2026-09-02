#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/wy/sjq/kd/outputs/student")
OUTPUT = ROOT / "comparisons" / "subset4_interaction4_weight_search_seed2026.json"


def report(path: Path) -> dict:
    return json.loads((path / "report.json").read_text(encoding="utf-8"))["valid_metrics"]


def main() -> int:
    paths = {
        "subset4": {
            "0.25": ROOT / "tuning/subset4_lambda0.25_seed2026",
            "0.5": ROOT / "tuning/subset4_lambda0.5_seed2026",
            "1.0": ROOT / "subset_value_4_benchmark500_seed2026",
            "2.0": ROOT / "tuning/subset4_lambda2_seed2026",
        },
        "interaction4": {
            "0.25": ROOT / "tuning/interaction4_lambda0.25_seed2026",
            "0.5": ROOT / "tuning/interaction4_lambda0.5_seed2026",
            "1.0": ROOT / "high_order_interaction_benchmark500_seed2026",
            "2.0": ROOT / "tuning/interaction4_lambda2_seed2026",
        },
    }
    results = {
        method: {
            weight: {key: float(value) for key, value in report(path).items() if isinstance(value, (int, float))}
            for weight, path in method_paths.items()
        }
        for method, method_paths in paths.items()
    }
    best = {
        method: min(weight_rows, key=lambda weight: weight_rows[weight]["mae"])
        for method, weight_rows in results.items()
    }
    best_subset = results["subset4"][best["subset4"]]
    best_interaction = results["interaction4"][best["interaction4"]]
    output = {
        "seed": 2026,
        "selection_metric": "valid_mae",
        "equal_search_grid": [0.25, 0.5, 1.0, 2.0],
        "results": results,
        "best_weight": best,
        "best_interaction_minus_best_subset_mae": best_interaction["mae"] - best_subset["mae"],
        "high_order_gate_passed": best_interaction["mae"] < best_subset["mae"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
