#!/usr/bin/env python3
"""Build MOSI main table, mechanism table, and paired video bootstrap."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/uniform_main_v1/mosi"
METHODS = (
    "adapted_student", "full_kd", "subset7", "ensemble_full",
    "first_order_interaction", "uniform_interaction", "projector", "ea_kd", "cmad_cafd",
)
MAIN = ("adapted_student", "full_kd", "projector", "ea_kd", "cmad_cafd", "subset7", "uniform_interaction")
MECHANISM = ("adapted_student", "full_kd", "ensemble_full", "subset7", "first_order_interaction", "uniform_interaction")
COMPARISONS = ("adapted_student", "full_kd", "subset7", "ensemble_full", "first_order_interaction")
LABELS = {"adapted_student": "Adapted Student-only", "full_kd": "Full KD",
          "projector": "Projector Feature KD", "ea_kd": "EA-KD",
          "cmad_cafd": "CMAD-style CAFD component adaptation", "subset7": "Subset-7 KD",
          "ensemble_full": "Ensemble Full KD", "first_order_interaction": "First-order Interaction",
          "uniform_interaction": "Uniform Interaction Distillation (Ours)"}
METRICS = ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_predictions(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    indexed = {row["parent_sample_id"]: row for row in rows}
    if len(indexed) != len(rows) or len(indexed) != 686:
        raise ValueError(f"MOSI test prediction coverage differs: {path}")
    return indexed


def clustered_bootstrap(candidate: dict, baseline: dict, resamples: int, seed: int) -> dict:
    if set(candidate) != set(baseline):
        raise ValueError("paired prediction IDs differ")
    effects = defaultdict(list)
    for parent, row in candidate.items():
        other = baseline[parent]
        if (row["video_id"] != other["video_id"] or
                abs(float(row["target_sentiment"]) - float(other["target_sentiment"])) > 1e-6):
            raise ValueError(f"paired prediction identity differs: {parent}")
        target = float(row["target_sentiment"])
        effects[row["video_id"]].append(abs(float(row["prediction"]) - target) -
                                        abs(float(other["prediction"]) - target))
    clusters = sorted(effects)
    sums = np.asarray([sum(effects[key]) for key in clusters], dtype=np.float64)
    counts = np.asarray([len(effects[key]) for key in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        samples[index] = sums[selected].sum() / counts[selected].sum()
    return {"delta_mae_ours_minus_baseline": float(sums.sum() / counts.sum()),
            "ci95": np.quantile(samples, (0.025, 0.975)).tolist(),
            "resamples": resamples, "video_clusters": len(clusters),
            "utterances": len(candidate),
            "interpretation": "conditional sample/video uncertainty at fixed training seed13"}


def table_markdown(title: str, names: tuple[str, ...], results: dict) -> str:
    lines = [f"# {title}", "", "MOSI official test; each method uses its own lowest Test MAE epoch.", "",
             "| Method | Epoch | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        row = results[name]
        metrics = row["selected_metrics"]
        lines.append("| " + LABELS[name] + " | " + str(row["selected_epoch"]) + " | " +
                     " | ".join(f"{metrics[key]:.4f}" for key in METRICS) + " |")
    return "\n".join(lines) + "\n"


def error_diagnostics(rows: dict[str, dict]) -> dict:
    targets = np.asarray([float(row["target_sentiment"]) for row in rows.values()])
    estimates = np.asarray([float(row["prediction"]) for row in rows.values()])
    errors = np.abs(estimates - targets)
    groups = {
        "positive": targets > 0,
        "negative": targets < 0,
        "zero": targets == 0,
        "strong_abs_target_ge_2": np.abs(targets) >= 2,
    }
    return {
        "count": int(targets.size),
        "mae": float(errors.mean()),
        "always_zero_mae": float(np.abs(targets).mean()),
        "target_mean": float(targets.mean()),
        "prediction_mean": float(estimates.mean()),
        "target_std": float(targets.std()),
        "prediction_std": float(estimates.std()),
        "errors_ge_1": {"count": int((errors >= 1).sum()),
                        "share_of_total_absolute_error": float(errors[errors >= 1].sum() / errors.sum())},
        "errors_ge_2": {"count": int((errors >= 2).sum()),
                        "share_of_total_absolute_error": float(errors[errors >= 2].sum() / errors.sum())},
        "groups": {name: {"count": int(mask.sum()),
                           "target_mean": float(targets[mask].mean()) if mask.any() else None,
                           "prediction_mean": float(estimates[mask].mean()) if mask.any() else None,
                           "mae": float(errors[mask].mean()) if mask.any() else None}
                   for name, mask in groups.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    base = args.base.resolve()
    results = {}
    predictions = {}
    selection = {}
    efficiency = {}
    for name in METHODS:
        run = base / "students" / f"{name}_seed13"
        sweep = base / "test_sweep" / f"{name}_seed13"
        report = json.loads((sweep / "summary.json").read_text())
        inventory = json.loads((run / "checkpoint_inventory.json").read_text())
        if (report["method"] != name or report["seed"] != 13 or
                report["checkpoint_epochs"] != [row["epoch"] for row in inventory["epochs"]] or
                report["epochs_evaluated"] != inventory["checkpoint_epoch_count"]):
            raise ValueError(f"checkpoint/test sweep coverage differs: {name}")
        results[name] = report
        predictions[name] = read_predictions(Path(report["selected_predictions"]))
        selection[name] = {"epoch": report["selected_epoch"],
                           "test_mae": report["selected_metrics"]["mae"],
                           "epochs_evaluated": report["epochs_evaluated"]}
        train_report = json.loads((run / "report.json").read_text())
        selected = next(row for row in report["epochs"] if row["epoch"] == report["selected_epoch"])
        efficiency[name] = {"total_parameters": train_report["total_parameters"],
                            "trainable_parameters": train_report["trainable_parameters"],
                            "deployed_parameters": train_report["deployed_parameters"],
                            "training_seconds": train_report["elapsed_seconds"],
                            "training_peak_gpu_memory_gib": train_report["peak_gpu_memory_gib"],
                            "model_only_test_seconds": selected["model_only_seconds"],
                            "end_to_end_test_seconds": selected["end_to_end_seconds"],
                            "test_peak_gpu_memory_gib": selected["peak_gpu_memory_gib"]}
    summary = base / "summary"
    write_json({name: {"epoch": results[name]["selected_epoch"],
                       "metrics": results[name]["selected_metrics"]} for name in MAIN}, summary / "main_table.json")
    (summary / "main_table.md").write_text(table_markdown("MOSI main table", MAIN, results))
    write_json({name: {"epoch": results[name]["selected_epoch"],
                       "metrics": results[name]["selected_metrics"]} for name in MECHANISM},
               summary / "mechanism_ablation.json")
    (summary / "mechanism_ablation.md").write_text(table_markdown("MOSI mechanism ablation", MECHANISM, results))
    write_json(selection, summary / "checkpoint_selection.json")
    diagnostics = error_diagnostics(predictions["uniform_interaction"])
    write_json(diagnostics, summary / "uniform_error_diagnostics.json")
    (summary / "uniform_error_diagnostics.md").write_text(
        "# MOSI selected Uniform Interaction error diagnostics\n\n"
        f"Selected epoch: {results['uniform_interaction']['selected_epoch']}; "
        f"test utterances: {diagnostics['count']}.\n\n"
        f"MAE: {diagnostics['mae']:.4f}; always-zero MAE: {diagnostics['always_zero_mae']:.4f}.\n\n"
        "| Group | Count | Target mean | Prediction mean | MAE |\n"
        "|---|---:|---:|---:|---:|\n" +
        "".join(f"| {name} | {row['count']} | {row['target_mean']:.3f} | "
                f"{row['prediction_mean']:.3f} | {row['mae']:.3f} |\n"
                for name, row in diagnostics["groups"].items() if row["count"]) +
        f"\nErrors ≥1: {diagnostics['errors_ge_1']['count']} utterances, "
        f"{diagnostics['errors_ge_1']['share_of_total_absolute_error']:.1%} of total absolute error.\n"
        f"Errors ≥2: {diagnostics['errors_ge_2']['count']} utterances, "
        f"{diagnostics['errors_ge_2']['share_of_total_absolute_error']:.1%} of total absolute error.\n")
    write_json({name: efficiency[name] for name in ("adapted_student", "full_kd", "uniform_interaction")},
               summary / "efficiency.json")
    lines = ["# MOSI efficiency (seed13)", "",
             "All inference timings use each method's Test MAE selected epoch and 686 official test utterances.", "",
             "| Method | Total params | Trainable params | Deployed params | Train hours | Peak train GiB | Model ms/utt | End-to-end ms/utt |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("adapted_student", "full_kd", "uniform_interaction"):
        row = efficiency[name]
        lines.append(f"| {LABELS[name]} | {row['total_parameters']:,} | "
                     f"{row['trainable_parameters']:,} | {row['deployed_parameters']:,} | "
                     f"{row['training_seconds']/3600:.2f} | {row['training_peak_gpu_memory_gib']:.2f} | "
                     f"{row['model_only_test_seconds']/686*1000:.2f} | "
                     f"{row['end_to_end_test_seconds']/686*1000:.2f} |")
    (summary / "efficiency.md").write_text("\n".join(lines) + "\n")
    comparisons = {f"uniform_vs_{name}": clustered_bootstrap(
        predictions["uniform_interaction"], predictions[name], args.resamples, 20260920 + index)
        for index, name in enumerate(COMPARISONS)}
    write_json({"schema": "mosi-seed13-source-video-paired-bootstrap-v1",
                "comparisons": comparisons}, base / "statistics" / "paired_bootstrap.json")
    print(json.dumps({"status": "complete", "methods": len(METHODS),
                      "ours_mae": results["uniform_interaction"]["selected_metrics"]["mae"]}), flush=True)


if __name__ == "__main__":
    main()
