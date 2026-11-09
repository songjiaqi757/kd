#!/usr/bin/env python3
"""Aggregate the equal-modality method-extension validation runs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics


METHODS = (
    "uniform_interaction",
    "uniform_interaction_shared_dropout",
    "uniform_interaction_gradient_projected",
    "uniform_interaction_coordinate_compatible",
    "explicit_full_kd",
    "explicit_uniform_interaction",
)
SEEDS = (13, 42, 2026)
CHECKPOINT_SELECTIONS = ("valid_mae", "valid_acc2_nonzero")
METRICS = (
    "mae",
    "pearson",
    "acc2_nonzero",
    "f1_weighted_nonzero",
    "acc2_has_zero",
    "f1_weighted_has_zero",
    "acc7",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            for policy in CHECKPOINT_SELECTIONS:
                report_path = (
                    args.input_root / f"{method}_seed{seed}" / "test" / policy / "report.json"
                )
                if not report_path.is_file():
                    raise FileNotFoundError(report_path)
                report = json.loads(report_path.read_text())
                if report.get("method") != method or int(report.get("seed")) != seed:
                    raise ValueError(f"run identity mismatch: {report_path}")
                if report.get("checkpoint_selection") != policy:
                    raise ValueError(f"unexpected checkpoint policy: {report_path}")
                metrics = report["metrics"]
                rows.append({
                    "method": method,
                    "seed": seed,
                    "checkpoint_selection": policy,
                    "selected_epoch": int(report["selected_epoch"]),
                    **{metric: float(metrics[metric]) for metric in METRICS},
                })

    by_method = {}
    for policy in CHECKPOINT_SELECTIONS:
        by_method[policy] = {}
        for method in METHODS:
            method_rows = [
                row for row in rows
                if row["method"] == method and row["checkpoint_selection"] == policy
            ]
            by_method[policy][method] = {
                metric: {
                    "mean": statistics.fmean(row[metric] for row in method_rows),
                    "sample_std": statistics.stdev(row[metric] for row in method_rows),
                }
                for metric in METRICS
            }
    baseline = {
        (row["checkpoint_selection"], row["seed"]): row
        for row in rows if row["method"] == "uniform_interaction"
    }
    deltas = {}
    for policy in CHECKPOINT_SELECTIONS:
        deltas[policy] = {}
        for method in METHODS[1:]:
            method_rows = {
                row["seed"]: row for row in rows
                if row["method"] == method and row["checkpoint_selection"] == policy
            }
            deltas[policy][method] = {
                metric: {
                    "per_seed_candidate_minus_icd": {
                        str(seed): method_rows[seed][metric] - baseline[(policy, seed)][metric]
                        for seed in SEEDS
                    },
                    "mean_candidate_minus_icd": statistics.fmean(
                        method_rows[seed][metric] - baseline[(policy, seed)][metric]
                        for seed in SEEDS
                    ),
                }
                for metric in METRICS
            }
    payload = {
        "schema": "rdid-msa-equal-modality-method-extensions-summary-v1",
        "checkpoint_selections": list(CHECKPOINT_SELECTIONS),
        "seeds": list(SEEDS),
        "rows": rows,
        "aggregates": by_method,
        "paired_deltas": deltas,
        "metric_direction": {metric: "lower" if metric == "mae" else "higher" for metric in METRICS},
    }
    atomic_text(args.output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    temporary_csv.parent.mkdir(parents=True, exist_ok=True)
    with temporary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "seed", "checkpoint_selection", "selected_epoch", *METRICS),
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)
    print(json.dumps({"output": str(args.output), "csv": str(csv_path), "runs": len(rows)}))


if __name__ == "__main__":
    main()
