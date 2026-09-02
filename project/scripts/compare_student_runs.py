#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.metrics import sentiment_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison of two student runs")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load(path: Path, split: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {str(row["parent_sample_id"]): row for row in rows if row["split"] == split}


def main() -> int:
    args = parse_args()
    baseline = load(args.baseline, args.split)
    candidate = load(args.candidate, args.split)
    if set(baseline) != set(candidate):
        raise ValueError("runs do not contain the same sample IDs")
    sample_ids = sorted(baseline)
    targets = np.asarray([baseline[item]["target_sentiment"] for item in sample_ids])
    baseline_predictions = np.asarray([baseline[item]["prediction"] for item in sample_ids])
    candidate_predictions = np.asarray([candidate[item]["prediction"] for item in sample_ids])
    baseline_metrics = sentiment_metrics(targets, baseline_predictions)
    candidate_metrics = sentiment_metrics(targets, candidate_predictions)
    metric_names = ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")
    samples = {name: [] for name in metric_names}
    rng = np.random.default_rng(args.seed)
    for _ in range(args.repetitions):
        indices = rng.integers(0, len(targets), len(targets))
        left = sentiment_metrics(targets[indices], baseline_predictions[indices])
        right = sentiment_metrics(targets[indices], candidate_predictions[indices])
        for name in metric_names:
            delta = float(right[name]) - float(left[name])
            if math.isfinite(delta):
                samples[name].append(delta)
    comparison = {}
    for name in metric_names:
        values = np.asarray(samples[name])
        better = values < 0 if name == "mae" else values > 0
        comparison[name] = {
            "baseline": float(baseline_metrics[name]),
            "candidate": float(candidate_metrics[name]),
            "candidate_minus_baseline": float(candidate_metrics[name]) - float(baseline_metrics[name]),
            "delta_ci95": {
                "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
            },
            "bootstrap_probability_candidate_better": float(np.mean(better)),
        }
    report = {
        "split": args.split,
        "samples": len(sample_ids),
        "repetitions": args.repetitions,
        "seed": args.seed,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
