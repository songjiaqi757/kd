#!/usr/bin/env python3
"""Summarize teacher subset behavior and interaction-coordinate structure."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/src"))

from rdid_mosei.metrics import sentiment_metrics
from rdid_mosei.student import SUBSETS


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def subset_metrics(rows: list[dict], split: str) -> dict:
    selected = [row for row in rows if row["split"] == split]
    targets = [float(row["sentiment"]) for row in selected]
    return {
        subset: sentiment_metrics(
            targets,
            [float(row["subset_score_mean"][index]) for row in selected],
        )
        for index, subset in enumerate(SUBSETS)
    }


def interaction_statistics(rows: list[dict], split: str) -> dict:
    selected = [row for row in rows if row["split"] == split]
    values = np.asarray([row["mean"] for row in selected], dtype=np.float64)
    if values.shape != (len(selected), len(SUBSETS)) or not np.isfinite(values).all():
        raise ValueError("interaction target shape or values are invalid")
    return {
        subset: {
            "mean_absolute_magnitude": float(np.mean(np.abs(values[:, index]))),
            "median_absolute_magnitude": float(np.median(np.abs(values[:, index]))),
            "standard_deviation": float(np.std(values[:, index], ddof=1)),
            "signed_mean": float(np.mean(values[:, index])),
        }
        for index, subset in enumerate(SUBSETS)
    }


def markdown(dataset: str, split: str, subsets: dict, interactions: dict) -> str:
    lines = [
        f"# {dataset.upper()} teacher structure ({split})",
        "",
        "## Teacher subset behavior",
        "",
        "| Subset | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for subset in SUBSETS:
        row = subsets[subset]
        lines.append(
            f"| {subset.upper()} | {row['mae']:.6f} | {row['pearson']:.6f} | "
            f"{row['acc2_nonzero']:.6f} | {row['f1_weighted_nonzero']:.6f} | "
            f"{row['acc7']:.6f} |"
        )
    lines.extend([
        "",
        "## Teacher interaction-coordinate statistics",
        "",
        "| Coordinate | Mean |I| | Median |I| | Std. | Signed mean |",
        "|---|---:|---:|---:|---:|",
    ])
    for subset in SUBSETS:
        row = interactions[subset]
        lines.append(
            f"| I_{subset.upper()} | {row['mean_absolute_magnitude']:.6f} | "
            f"{row['median_absolute_magnitude']:.6f} | "
            f"{row['standard_deviation']:.6f} | {row['signed_mean']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mosei", "mosi"), required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = ROOT / f"outputs/experiments/main_table_v1/{args.dataset}/assets/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    ensemble_path = Path(protocol["ensemble_targets"])
    interaction_path = Path(protocol["interaction_targets"])
    ensemble = read_jsonl(ensemble_path)
    interactions = read_jsonl(interaction_path)
    ensemble_ids = {row["parent_sample_id"] for row in ensemble if row["split"] == args.split}
    interaction_ids = {row["parent_sample_id"] for row in interactions if row["split"] == args.split}
    if ensemble_ids != interaction_ids:
        raise ValueError("ensemble and interaction target coverage differs")
    payload = {
        "schema": "teacher-structure-analysis-v1",
        "dataset": args.dataset,
        "split": args.split,
        "utterances": len(ensemble_ids),
        "coordinate_order": list(SUBSETS),
        "subset_metrics": subset_metrics(ensemble, args.split),
        "interaction_statistics": interaction_statistics(interactions, args.split),
        "sources": {
            "protocol": str(protocol_path.resolve()),
            "ensemble_targets": str(ensemble_path.resolve()),
            "interaction_targets": str(interaction_path.resolve()),
        },
    }
    output = args.output.resolve()
    atomic_json(payload, output)
    output.with_suffix(".md").write_text(
        markdown(
            args.dataset,
            args.split,
            payload["subset_metrics"],
            payload["interaction_statistics"],
        )
    )
    print(json.dumps({"status": "complete", "output": str(output), "utterances": len(ensemble_ids)}))


if __name__ == "__main__":
    main()
