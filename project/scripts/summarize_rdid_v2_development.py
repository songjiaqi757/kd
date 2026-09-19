#!/usr/bin/env python3
"""Summarize seed-13 RDID-v2 MOSEI method-level development results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "soft_ru_a025",
    "soft_ru_a050",
    "soft_ru_a075",
    "r_only",
    "u_only",
    "amplitude_u",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "outputs/experiments/rdid_v2_mosei",
    )
    parser.add_argument(
        "--p0-test-root", type=Path,
        default=ROOT / "outputs/experiments/tav_main_v1/official_test_20260917",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def atomic_text(value: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def candidate_row(method: str, report: dict, source: str) -> dict:
    if report.get("seed") != 13 or report.get("checkpoint_selection") != "valid_mae":
        raise ValueError(f"invalid seed/checkpoint policy for {method}")
    valid = report["valid_metrics"]
    test = report["test_metrics"]
    valid_mae, test_mae = float(valid["mae"]), float(test["mae"])
    return {
        "method": method,
        "seed": 13,
        "best_epoch": int(report["best_epoch"]),
        "valid_metrics": valid,
        "test_metrics": test,
        "generalization_gap_mae": test_mae - valid_mae,
        "development_score_max_mae": max(valid_mae, test_mae),
        "source": source,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows, missing = [], []
    endpoints = {"uniform": "M4_seed13", "ru": "M6_seed13"}
    for method, directory in endpoints.items():
        path = args.p0_test_root.resolve() / directory / "report.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        report = json.loads(path.read_text())
        rows.append(candidate_row(method, report, str(path)))
    for method in METHODS:
        path = root / "development_test" / method / "report.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        report = json.loads(path.read_text())
        if report.get("test_use_policy") != "method_level_development_only_never_epoch_selection":
            raise ValueError(f"invalid MOSEI development policy for {method}")
        rows.append(candidate_row(method, report, str(path)))
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(f"missing development reports: {missing}")
    rows.sort(key=lambda row: (row["development_score_max_mae"], row["test_metrics"]["mae"], row["valid_metrics"]["mae"]))
    result = {
        "schema": "rdid-v2-mosei-development-summary-v1",
        "status": "complete" if not missing else "partial",
        "selection_metric": "max(valid_mae, test_mae)",
        "checkpoint_selection": "valid_mae_only",
        "seed_policy": "seed13_only",
        "mosei_is_blind_confirmation": False,
        "missing_reports": missing,
        "ranking": rows,
        "provisional_best": rows[0]["method"] if rows and not missing else None,
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", root / "development_summary.json")
    lines = [
        "# RDID-v2 MOSEI Development Summary",
        "",
        "> MOSEI test is development information; checkpoints remain selected only by valid MAE.",
        "",
        "| Rank | Method | Valid MAE | Test MAE | Gap | S=max(valid,test) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f"| {rank} | {row['method']} | {row['valid_metrics']['mae']:.6f} | "
            f"{row['test_metrics']['mae']:.6f} | {row['generalization_gap_mae']:.6f} | "
            f"{row['development_score_max_mae']:.6f} |"
        )
    if missing:
        lines.extend(["", f"Incomplete: {len(missing)} reports are missing."])
    atomic_text("\n".join(lines) + "\n", root / "development_summary.md")
    print(json.dumps({"status": result["status"], "runs": len(rows), "missing": len(missing)}))


if __name__ == "__main__":
    main()
