#!/usr/bin/env python3
"""Combine the seven MOSI controls under validation-MAE checkpoint selection."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "outputs/experiments/multiseed_controls_v1/mosi"
ADDITIONAL = ROOT / "outputs/experiments/multiseed_controls_v1/mosi_additional_controls"
OUTPUT = ROOT / "outputs/experiments/multiseed_controls_v1/mosi_seven_controls_valid_mae"
METHODS = (
    "full_kd",
    "ensemble_full",
    "subset7",
    "first_order_interaction",
    "first_second_order_interaction",
    "random_orthogonal",
    "uniform_interaction",
)
METRICS = (
    "mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero",
    "acc2_has_zero", "f1_weighted_has_zero", "acc7",
)
SEED13_RUNS = {
    method: ROOT / f"outputs/experiments/uniform_main_v1/mosi/students/{method}_seed13"
    for method in ("ensemble_full", "first_order_interaction")
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def additional_row(method: str, seed: int) -> dict:
    name = f"{method}_seed{seed}"
    run = SEED13_RUNS[method] if seed == 13 else ADDITIONAL / "students" / name
    train_report = read_json(run / "report.json")
    history = read_json(run / "history.json")
    epoch = int(train_report["best_epoch"])
    selected = next(row for row in history if int(row["epoch"]) == epoch)
    minimum = min(float(row["valid_metrics"]["mae"]) for row in history)
    if float(selected["valid_metrics"]["mae"]) > minimum + 1e-5:
        raise ValueError(f"recorded checkpoint is not validation-MAE best: {name}")
    if seed == 13:
        test_report = (
            ADDITIONAL / "official_test_reused_seed13" / name / "epochs"
            / f"epoch_{epoch:03d}" / "report.json"
        )
        test_metrics = read_json(test_report)["test_metrics"]
    else:
        test_report = ADDITIONAL / "official_test" / name / "report.json"
        report = read_json(test_report)
        if report.get("checkpoint_selection") != "valid_mae":
            raise ValueError(f"official test selection differs: {name}")
        test_metrics = report["metrics"]
    if test_metrics.get("count") != 686 or test_metrics.get("nonzero_count") != 656:
        raise ValueError(f"official test coverage differs: {name}")
    return {
        "run": name,
        "method": method,
        "seed": seed,
        "source_run": str(run.resolve()),
        "selected_epoch": epoch,
        "valid_metrics": selected["valid_metrics"],
        "test_metrics": test_metrics,
        "report": str(test_report.resolve()),
    }


def main() -> None:
    original = read_json(ORIGINAL / "official_test_valid_mae_3seed/summary.json")
    rows = list(original["rows"])
    if {(row["method"], row["seed"]) for row in rows} != {
        (method, seed)
        for method in METHODS
        if method not in {"ensemble_full", "first_order_interaction"}
        for seed in (13, 42, 2026)
    }:
        raise ValueError("the existing five-method summary has unexpected coverage")
    for method in ("ensemble_full", "first_order_interaction"):
        for seed in (13, 42, 2026):
            rows.append(additional_row(method, seed))
    rows.sort(key=lambda row: (METHODS.index(row["method"]), (13, 42, 2026).index(row["seed"])))
    expected = {(method, seed) for method in METHODS for seed in (13, 42, 2026)}
    if {(row["method"], row["seed"]) for row in rows} != expected or len(rows) != 21:
        raise ValueError("seven-method summary coverage differs")
    aggregates = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        aggregates[method] = {
            metric: {
                "mean": statistics.mean(row["test_metrics"][metric] for row in group),
                "sample_std": statistics.stdev(row["test_metrics"][metric] for row in group),
                "count": 3,
            }
            for metric in METRICS
        }
    payload = {
        "schema": "rdid-msa-mosi-seven-controls-valid-mae-summary-v1",
        "dataset": "mosi",
        "methods": list(METHODS),
        "seeds": [13, 42, 2026],
        "checkpoint_selection": "validation_mae_minimum",
        "official_test_policy": "evaluate_selected_checkpoint_only",
        "test_labels_used_for_selection": False,
        "rows": rows,
        "aggregates": aggregates,
    }
    atomic_json(payload, OUTPUT / "summary.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT / "summary.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "seed", "selected_epoch", "valid_mae",
            *[f"test_{metric}" for metric in METRICS],
        ])
        for row in rows:
            writer.writerow([
                row["method"], row["seed"], row["selected_epoch"],
                row["valid_metrics"]["mae"],
                *[row["test_metrics"][metric] for metric in METRICS],
            ])
    os.replace(temporary, OUTPUT / "summary.csv")
    print(json.dumps({"status": "complete", "rows": len(rows), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
