#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/wy/sjq/kd/dataset/cmu_mosei")
OUTPUT_ROOT = Path("/home/wy/sjq/kd/outputs/teacher_benchmark")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.metrics import sentiment_metrics  # noqa: E402
from rdid_mosei.subsets import SUBSETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the aggregated 500-sample teacher benchmark")
    parser.add_argument(
        "--input", type=Path, default=OUTPUT_ROOT / "teacher_benchmark500_aggregated.jsonl"
    )
    parser.add_argument(
        "--metadata", type=Path, default=DATASET_ROOT / "manifests/benchmark500_prepared.jsonl"
    )
    parser.add_argument(
        "--raw", type=Path, default=OUTPUT_ROOT / "teacher_benchmark500_windowed.jsonl"
    )
    parser.add_argument(
        "--output-json", type=Path, default=OUTPUT_ROOT / "teacher_benchmark500_metrics.json"
    )
    parser.add_argument(
        "--output-md", type=Path, default=OUTPUT_ROOT / "teacher_benchmark500_metrics.md"
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    return value


def bootstrap_intervals(
    targets: np.ndarray, predictions: np.ndarray, repetitions: int, seed: int
) -> dict[str, dict[str, float | None]]:
    rng = np.random.default_rng(seed)
    tracked = ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc2_has_zero", "f1_weighted_has_zero", "acc7")
    values: dict[str, list[float]] = {name: [] for name in tracked}
    for _ in range(repetitions):
        indices = rng.integers(0, len(targets), len(targets))
        result = sentiment_metrics(targets[indices], predictions[indices])
        for name in tracked:
            value = float(result[name])
            if math.isfinite(value):
                values[name].append(value)
    return {
        name: {
            "low": float(np.percentile(samples, 2.5)) if samples else None,
            "high": float(np.percentile(samples, 97.5)) if samples else None,
        }
        for name, samples in values.items()
    }


def paired_bootstrap(
    targets: np.ndarray,
    text_predictions: np.ndarray,
    full_predictions: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    metric_names = ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    observed_text = sentiment_metrics(targets, text_predictions)
    observed_full = sentiment_metrics(targets, full_predictions)
    for _ in range(repetitions):
        indices = rng.integers(0, len(targets), len(targets))
        left = sentiment_metrics(targets[indices], text_predictions[indices])
        right = sentiment_metrics(targets[indices], full_predictions[indices])
        for name in metric_names:
            delta = float(right[name]) - float(left[name])
            if math.isfinite(delta):
                samples[name].append(delta)
    return {
        name: {
            "observed_tav_minus_t": float(observed_full[name]) - float(observed_text[name]),
            "ci95": {
                "low": float(np.percentile(samples[name], 2.5)),
                "high": float(np.percentile(samples[name], 97.5)),
            },
            "bootstrap_probability_tav_better": float(
                np.mean(np.asarray(samples[name]) < 0.0 if name == "mae" else np.asarray(samples[name]) > 0.0)
            ),
        }
        for name in metric_names
    }


def grouped_metrics(
    records: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], field: str
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    groups: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        parent_id = str(record["parent_sample_id"])
        group = str(metadata[parent_id][field])
        groups[group][str(record["subset"])].append(
            (float(record["target_sentiment"]), float(record["teacher_score"]))
        )
    return {
        group: {
            subset: finite_or_none(sentiment_metrics([x for x, _ in pairs], [y for _, y in pairs]))
            for subset, pairs in sorted(subsets.items())
        }
        for group, subsets in sorted(groups.items())
    }


def main() -> int:
    args = parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    metadata_rows = [json.loads(line) for line in args.metadata.read_text(encoding="utf-8").splitlines() if line]
    metadata = {str(row["sample_id"]): row for row in metadata_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sample: dict[str, dict[str, float]] = defaultdict(dict)
    targets: dict[str, float] = {}
    for record in records:
        parent_id = str(record["parent_sample_id"])
        if parent_id not in metadata:
            raise ValueError(f"aggregated sample absent from metadata: {parent_id}")
        subset = str(record["subset"])
        grouped[subset].append(record)
        by_sample[parent_id][subset] = float(record["teacher_score"])
        targets[parent_id] = float(record["target_sentiment"])
    if set(grouped) != set(SUBSETS):
        raise ValueError(f"subset mismatch: {sorted(grouped)}")
    if any(set(values) != set(SUBSETS) for values in by_sample.values()):
        raise ValueError("one or more samples are missing subset predictions")

    overall = {}
    for subset_index, subset in enumerate(SUBSETS):
        subset_records = grouped[subset]
        y_true = np.asarray([float(row["target_sentiment"]) for row in subset_records])
        y_pred = np.asarray([float(row["teacher_score"]) for row in subset_records])
        overall[subset] = {
            "metrics": finite_or_none(sentiment_metrics(y_true, y_pred)),
            "ci95": bootstrap_intervals(
                y_true, y_pred, args.bootstrap_repetitions, args.seed + subset_index
            ),
            "prediction": {
                "mean": float(np.mean(y_pred)),
                "standard_deviation": float(np.std(y_pred)),
                "zero_fraction": float(np.mean(y_pred == 0.0)),
            },
        }

    sample_ids = sorted(by_sample)
    target_array = np.asarray([targets[item] for item in sample_ids])
    text_array = np.asarray([by_sample[item]["t"] for item in sample_ids])
    full_array = np.asarray([by_sample[item]["tav"] for item in sample_ids])
    within_ranges = [max(values.values()) - min(values.values()) for values in by_sample.values()]
    identical = sum(len(set(values.values())) == 1 for values in by_sample.values())

    train_targets = [float(row["sentiment"]) for row in metadata_rows if row["split"] == "train"]
    baseline = statistics.mean(train_targets)
    interaction_rows = []
    for sample_id, values in by_sample.items():
        interaction_rows.append(
            {
                "ta": values["ta"] - values["t"] - values["a"] + baseline,
                "tv": values["tv"] - values["t"] - values["v"] + baseline,
                "av": values["av"] - values["a"] - values["v"] + baseline,
                "tav": values["tav"] - values["ta"] - values["tv"] - values["av"]
                + values["t"] + values["a"] + values["v"] - baseline,
            }
        )
    interaction_summary = {
        name: {
            "mean": statistics.mean(row[name] for row in interaction_rows),
            "mean_absolute": statistics.mean(abs(row[name]) for row in interaction_rows),
            "zero_fraction": statistics.mean(row[name] == 0.0 for row in interaction_rows),
            "positive_fraction": statistics.mean(row[name] > 0.0 for row in interaction_rows),
        }
        for name in ("ta", "tv", "av", "tav")
    }

    raw_records = [json.loads(line) for line in args.raw.read_text(encoding="utf-8").splitlines() if line]
    inference = [float(row["inference_seconds"]) for row in raw_records]
    preprocessing = [float(row["preprocess_seconds"]) for row in raw_records]
    report = {
        "schema_version": "teacher-benchmark-metrics-v1",
        "metric_policy": {
            "acc2_nonzero": "exclude target == 0; prediction >= 0 is non-negative",
            "acc2_has_zero": "target >= 0 and prediction >= 0 are non-negative",
            "f1": "support-weighted binary F1 under the corresponding Acc-2 policy",
            "acc7": "round half away from zero, then clip to [-3, 3]",
            "bootstrap": f"paired/sample bootstrap, {args.bootstrap_repetitions} repetitions, seed {args.seed}",
        },
        "records": len(records),
        "samples": len(by_sample),
        "overall": overall,
        "tav_vs_t_paired": paired_bootstrap(
            target_array, text_array, full_array, args.bootstrap_repetitions, args.seed + 100
        ),
        "subset_diversity": {
            "all_seven_identical_count": identical,
            "all_seven_identical_fraction": identical / len(by_sample),
            "mean_within_sample_range": statistics.mean(within_ranges),
            "median_within_sample_range": statistics.median(within_ranges),
        },
        "interaction": {"empty_baseline_train_mean": baseline, "summary": interaction_summary},
        "by_split": grouped_metrics(records, metadata, "split"),
        "by_duration_bucket": grouped_metrics(records, metadata, "duration_bucket"),
        "by_sentiment_bucket": grouped_metrics(records, metadata, "sentiment_bucket"),
        "runtime": {
            "raw_window_jobs": len(raw_records),
            "inference_total_seconds": sum(inference),
            "inference_mean_seconds": statistics.mean(inference),
            "inference_p95_seconds": float(np.percentile(inference, 95)),
            "inference_max_seconds": max(inference),
            "preprocess_total_seconds": sum(preprocessing),
            "preprocess_mean_seconds": statistics.mean(preprocessing),
            "preprocess_max_seconds": max(preprocessing),
            "gpu_peak_allocated_gib": [
                max(row["gpu_memory"][index]["max_allocated_bytes"] for row in raw_records) / 2**30
                for index in (0, 1)
            ],
            "status_counts": dict(Counter(str(row["status"]) for row in raw_records)),
        },
    }
    report = finite_or_none(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Qwen3-Omni CMU-MOSEI Teacher Benchmark (500 samples)",
        "",
        f"Aggregated records: {len(records)}; samples: {len(by_sample)}; raw window jobs: {len(raw_records)}.",
        "",
        "| subset | MAE | Pearson | Acc-2 non-zero | F1 non-zero | Acc-2 has-zero | F1 has-zero | Acc-7 | zero output |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in SUBSETS:
        item = overall[subset]
        metric = item["metrics"]
        lines.append(
            f"| {subset} | {metric['mae']:.4f} | {metric['pearson']:.4f} | "
            f"{metric['acc2_nonzero']:.4f} | {metric['f1_weighted_nonzero']:.4f} | "
            f"{metric['acc2_has_zero']:.4f} | {metric['f1_weighted_has_zero']:.4f} | "
            f"{metric['acc7']:.4f} | {item['prediction']['zero_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired TAV minus text-only",
            "",
            "Negative MAE delta is better; positive deltas are better for other metrics.",
            "",
        ]
    )
    for name, item in report["tav_vs_t_paired"].items():
        lines.append(
            f"- {name}: delta={item['observed_tav_minus_t']:.4f}, "
            f"95% CI=[{item['ci95']['low']:.4f}, {item['ci95']['high']:.4f}], "
            f"P(TAV better)={item['bootstrap_probability_tav_better']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Key diagnostics",
            "",
            f"- All seven predictions identical: {identical}/{len(by_sample)}.",
            f"- Mean within-sample subset range: {statistics.mean(within_ranges):.4f}.",
            f"- Video-only zero-output fraction: {overall['v']['prediction']['zero_fraction']:.4f}.",
            f"- Peak allocated GPU memory: {report['runtime']['gpu_peak_allocated_gib'][0]:.2f} / {report['runtime']['gpu_peak_allocated_gib'][1]:.2f} GiB.",
            "",
            "Full confidence intervals and grouped metrics are stored in the JSON report.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md), "samples": len(by_sample)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
