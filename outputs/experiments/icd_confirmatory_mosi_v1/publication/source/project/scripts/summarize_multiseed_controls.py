#!/usr/bin/env python3
"""Audit and summarize valid-selected multi-seed controls."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


LABELS = {
    "full_kd": "Full KD",
    "subset7": "Subset-7",
    "first_second_order_interaction": "First+Second-order",
    "random_orthogonal": "Random Orthogonal",
    "uniform_interaction": "Uniform Interaction",
}


def read_json(path: Path):
    return json.loads(path.read_text())


def mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else math.nan,
        "count": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    base = args.base.resolve()
    plan = read_json(base / "plan.json")
    rows = []
    for seed in plan["seeds"]:
        for method in plan["methods"]:
            name = f"{method}_seed{seed}"
            run = base / "students" / name
            test = base / "official_test" / name
            required = [run / "status.json", run / "report.json", run / "history.json",
                        run / "checkpoint_inventory.json", test / "status.json", test / "report.json"]
            if not all(path.is_file() for path in required):
                if args.allow_incomplete:
                    continue
                raise FileNotFoundError(f"incomplete run: {name}")
            if read_json(run / "status.json").get("status") != "complete" or read_json(test / "status.json").get("status") != "complete":
                raise ValueError(f"non-complete status: {name}")
            report = read_json(run / "report.json")
            history = read_json(run / "history.json")
            inventory = read_json(run / "checkpoint_inventory.json")
            official = read_json(test / "report.json")
            epochs = [int(row["epoch"]) for row in history]
            inventory_epochs = [int(row["epoch"]) for row in inventory["epochs"]]
            if epochs != inventory_epochs or report["best_epoch"] != min(history, key=lambda row: (row["valid_metrics"]["mae"], row["epoch"]))["epoch"]:
                raise ValueError(f"checkpoint audit failed: {name}")
            if official.get("checkpoint_selection") != "valid_mae":
                raise ValueError(f"official test selection differs: {name}")
            rows.append({
                "method": method, "seed": seed, "best_epoch": report["best_epoch"],
                "epochs_run": len(history), "checkpoint_count": len(inventory_epochs),
                "valid_metrics": report["valid_metrics"], "test_metrics": official["metrics"],
            })
    aggregates = {}
    for method in plan["methods"]:
        selected = [row for row in rows if row["method"] == method]
        aggregates[method] = {
            "valid_mae": mean_std([row["valid_metrics"]["mae"] for row in selected]) if selected else None,
            "test_mae": mean_std([row["test_metrics"]["mae"] for row in selected]) if selected else None,
        }
    deltas = {}
    uniform = {row["seed"]: row for row in rows if row["method"] == "uniform_interaction"}
    for baseline in ("full_kd", "subset7", "random_orthogonal", "first_second_order_interaction"):
        other = {row["seed"]: row for row in rows if row["method"] == baseline}
        seeds = sorted(set(uniform) & set(other))
        if not seeds:
            continue
        values = [uniform[seed]["test_metrics"]["mae"] - other[seed]["test_metrics"]["mae"] for seed in seeds]
        deltas[f"uniform_minus_{baseline}"] = {"by_seed": dict(zip(map(str, seeds), values)), **mean_std(values)}
    payload = {
        "schema": "rdid-msa-multiseed-controls-summary-v1",
        "dataset": plan["dataset"], "checkpoint_selection": "validation_mae_minimum",
        "rows": rows, "aggregates": aggregates, "paired_test_mae_deltas": deltas,
    }
    output = args.output or base / "summary.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
