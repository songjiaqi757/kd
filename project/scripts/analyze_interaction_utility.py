#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.metrics import sentiment_metrics


SUBSETS = ("t", "a", "v", "ta", "tv", "av", "tav")
INTERACTIONS = ("ta", "tv", "av", "tav")
LEVELS = ("low", "middle", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose interaction fidelity versus downstream task utility")
    parser.add_argument("--root", type=Path, default=Path("/home/wy/sjq/kd"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_parent(path: Path, split: str | None = None) -> dict[str, dict]:
    return {
        str(row["parent_sample_id"]): row
        for row in read_jsonl(path)
        if split is None or row["split"] == split
    }


def interactions(values: dict[str, float]) -> dict[str, float]:
    return {
        "ta": values["ta"] - values["t"] - values["a"],
        "tv": values["tv"] - values["t"] - values["v"],
        "av": values["av"] - values["a"] - values["v"],
        "tav": values["tav"] - values["ta"] - values["tv"] - values["av"]
        + values["t"] + values["a"] + values["v"],
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(left.std()) == 0 or float(right.std()) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_standard_deviation": statistics.stdev(values)}


def tertiles(values: dict[str, float]) -> tuple[dict[str, str], list[float]]:
    cuts = np.quantile(np.asarray(list(values.values())), [1 / 3, 2 / 3])
    return {
        key: "low" if value <= cuts[0] else "middle" if value <= cuts[1] else "high"
        for key, value in values.items()
    }, [float(value) for value in cuts]


def subgroup_metrics(
    baseline: dict[str, dict], candidate: dict[str, dict], labels: dict[str, str]
) -> dict[str, dict]:
    output = {}
    for level in LEVELS:
        ids = [key for key in baseline if labels[key] == level]
        targets = [float(baseline[key]["target_sentiment"]) for key in ids]
        left = sentiment_metrics(targets, [float(baseline[key]["prediction"]) for key in ids])
        right = sentiment_metrics(targets, [float(candidate[key]["prediction"]) for key in ids])
        output[level] = {
            "count": len(ids),
            "full_kd": {name: float(left[name]) for name in ("mae", "pearson", "acc2_nonzero")},
            "pair_snr": {name: float(right[name]) for name in ("mae", "pearson", "acc2_nonzero")},
        }
    return output


def teacher_subset_ensemble(paths: list[Path]) -> dict[str, dict[str, float]]:
    collected: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        for row in read_jsonl(path):
            collected[str(row["parent_sample_id"])][str(row["subset"])].append(float(row["probe_score"]))
    output = {}
    for parent, values in collected.items():
        if set(values) != set(SUBSETS) or any(len(scores) != len(paths) for scores in values.values()):
            raise RuntimeError(f"incomplete teacher ensemble for {parent}")
        output[parent] = {subset: statistics.mean(values[subset]) for subset in SUBSETS}
    return output


def main() -> int:
    args = parse_args()
    root = args.root
    reliability_rows = read_jsonl(root / "outputs/probe/official_train_valid_interaction_reliability.jsonl")
    teacher = {str(row["sample_id"]): row for row in reliability_rows}
    teacher_subsets = teacher_subset_ensemble([
        root / f"outputs/probe/official_train_valid_seed{seed}/predictions.jsonl"
        for seed in (2026, 2027, 2028)
    ])

    label_source = by_parent(root / "outputs/student/fullscale_pair_snr_seed13/predictions.jsonl")
    train_ids = sorted(key for key, row in teacher.items() if row["split"] == "train")
    valid_ids = sorted(key for key, row in teacher.items() if row["split"] == "valid")
    if not set(teacher).issubset(label_source):
        raise RuntimeError("student predictions do not cover teacher rows")
    train_targets = np.asarray([float(label_source[key]["target_sentiment"]) for key in train_ids])
    global_utility = {}
    for name in INTERACTIONS[:3]:
        values = np.asarray([float(teacher[key]["interaction_mean"][name]) for key in train_ids])
        global_utility[name] = abs(correlation(values, train_targets))
    utility_mean = statistics.mean(global_utility.values())
    global_utility_normalized = {name: value / utility_mean for name, value in global_utility.items()}

    strength = {
        key: sum(abs(float(teacher[key]["interaction_mean"][name])) for name in INTERACTIONS)
        for key in valid_ids
    }
    uncertainty = {
        key: statistics.mean(float(teacher[key]["interaction_var"][name]) for name in INTERACTIONS)
        for key in valid_ids
    }
    conflict_by_type = {
        key: {
            "ta": abs(teacher_subsets[key]["t"] - teacher_subsets[key]["a"]),
            "tv": abs(teacher_subsets[key]["t"] - teacher_subsets[key]["v"]),
            "av": abs(teacher_subsets[key]["a"] - teacher_subsets[key]["v"]),
        }
        for key in valid_ids
    }
    conflict = {key: statistics.mean(values.values()) for key, values in conflict_by_type.items()}
    axes = {}
    for name, values in (("interaction_strength", strength), ("teacher_uncertainty", uncertainty), ("modality_conflict", conflict)):
        labels, cuts = tertiles(values)
        axes[name] = {"labels": labels, "tertile_cuts": cuts}

    per_seed = []
    for seed in (13, 42, 2026):
        full_kd = by_parent(
            root / f"outputs/student/fullscale_full_kd_seed{seed}/all_subset_predictions.jsonl", "valid"
        )
        pair_snr = by_parent(root / f"outputs/student/fullscale_pair_snr_seed{seed}/predictions.jsonl", "valid")
        if set(full_kd) != set(pair_snr) or set(full_kd) != set(valid_ids):
            raise RuntimeError(f"valid ID mismatch for seed {seed}")
        delta_task = np.asarray([
            abs(float(pair_snr[key]["prediction"]) - float(pair_snr[key]["target_sentiment"]))
            - abs(float(full_kd[key]["prediction"]) - float(full_kd[key]["target_sentiment"]))
            for key in valid_ids
        ])
        diagnostics = {}
        for name in INTERACTIONS:
            full_error = []
            pair_error = []
            for key in valid_ids:
                expected = float(teacher[key]["interaction_mean"][name])
                full_value = interactions(full_kd[key]["subset_predictions"])[name]
                pair_value = interactions(pair_snr[key]["subset_predictions"])[name]
                full_error.append(abs(full_value - expected))
                pair_error.append(abs(pair_value - expected))
            full_error_array = np.asarray(full_error)
            pair_error_array = np.asarray(pair_error)
            delta_interaction = pair_error_array - full_error_array
            improved = delta_interaction < 0
            diagnostics[name] = {
                "teacher_mean_snr": statistics.mean(
                    float(teacher[key]["interaction_snr"][name]) for key in valid_ids
                ),
                "full_kd_interaction_mae": float(full_error_array.mean()),
                "pair_snr_interaction_mae": float(pair_error_array.mean()),
                "delta_interaction_mae": float(delta_interaction.mean()),
                "corr_delta_interaction_delta_task": correlation(delta_interaction, delta_task),
                "interaction_improved_count": int(improved.sum()),
                "interaction_improved_rate": float(improved.mean()),
                "task_useful_rate": float(np.mean(delta_task[improved] < 0)) if np.any(improved) else math.nan,
                "mean_delta_task_when_interaction_improved": float(delta_task[improved].mean()) if np.any(improved) else math.nan,
            }
        bootstrap = json.loads((
            root / f"project/reports/stage_c0_pair_snr_vs_full_kd_seed{seed}_bootstrap.json"
        ).read_text(encoding="utf-8"))
        per_seed.append({
            "seed": seed,
            "mean_delta_task_mae": float(delta_task.mean()),
            "unconditional_task_improvement_rate": float(np.mean(delta_task < 0)),
            "diagnostics": diagnostics,
            "subgroups": {
                name: subgroup_metrics(full_kd, pair_snr, axis["labels"]) for name, axis in axes.items()
            },
            "paired_bootstrap": bootstrap["comparison"],
        })

    aggregate_diagnostics = {}
    for name in INTERACTIONS:
        aggregate_diagnostics[name] = {
            metric: mean_std([float(row["diagnostics"][name][metric]) for row in per_seed])
            for metric in (
                "delta_interaction_mae",
                "corr_delta_interaction_delta_task",
                "interaction_improved_rate",
                "task_useful_rate",
                "mean_delta_task_when_interaction_improved",
            )
        }
        aggregate_diagnostics[name]["teacher_mean_snr"] = per_seed[0]["diagnostics"][name]["teacher_mean_snr"]

    payload = {
        "scope": "official train/valid only; official test not used",
        "comparison": "pair-SNR minus B1 Full KD; negative error delta is better",
        "train_only_global_utility": global_utility,
        "train_only_global_utility_normalized": global_utility_normalized,
        "subgroup_definitions": {
            name: {"tertile_cuts": axis["tertile_cuts"]} for name, axis in axes.items()
        },
        "per_seed": per_seed,
        "aggregate_diagnostics": aggregate_diagnostics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pm(item: dict[str, float]) -> str:
        return f"{item['mean']:.4f} ± {item['sample_standard_deviation']:.4f}"

    lines = [
        "# Interaction Utility Diagnosis", "",
        "范围：official train/valid；official test 未使用。比较方向均为 pair-SNR − B1 Full KD。", "",
        "## Official-train global utility", "",
        "| Interaction | abs corr(I, y) | normalized utility |",
        "|---|---:|---:|",
    ]
    for name in INTERACTIONS[:3]:
        lines.append(f"| {name.upper()} | {global_utility[name]:.6f} | {global_utility_normalized[name]:.6f} |")
    lines += [
        "", "## Fidelity → task utility（三 seed）", "",
        "| Interaction | Teacher mean SNR | Δ interaction MAE | Corr(ΔE_I, ΔE_task) | Interaction improved rate | Task-useful rate | Δ task when I improves |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in INTERACTIONS:
        item = aggregate_diagnostics[name]
        lines.append(
            f"| {name.upper()} | {item['teacher_mean_snr']:.4f} | {pm(item['delta_interaction_mae'])} | "
            f"{pm(item['corr_delta_interaction_delta_task'])} | {pm(item['interaction_improved_rate'])} | "
            f"{pm(item['task_useful_rate'])} | {pm(item['mean_delta_task_when_interaction_improved'])} |"
        )
    lines += ["", "## Pair-SNR vs Full KD：MAE paired bootstrap", "",
              "| Seed | ΔMAE | 95% CI | P(pair-SNR better) |", "|---:|---:|---:|---:|"]
    for row in per_seed:
        item = row["paired_bootstrap"]["mae"]
        lines.append(
            f"| {row['seed']} | {item['candidate_minus_baseline']:+.6f} | "
            f"[{item['delta_ci95']['low']:+.6f}, {item['delta_ci95']['high']:+.6f}] | "
            f"{item['bootstrap_probability_candidate_better']:.4f} |"
        )
    for axis_name, title in (
        ("interaction_strength", "Interaction strength tertile"),
        ("teacher_uncertainty", "Teacher uncertainty tertile"),
        ("modality_conflict", "Modality conflict tertile"),
    ):
        lines += ["", f"## {title}", "",
                  "| Group | Method | MAE | Pearson | Acc-2 |", "|---|---|---:|---:|---:|"]
        for level in LEVELS:
            for method, label in (("full_kd", "B1 Full KD"), ("pair_snr", "pair-SNR")):
                values = {
                    metric: mean_std([
                        float(row["subgroups"][axis_name][level][method][metric]) for row in per_seed
                    ]) for metric in ("mae", "pearson", "acc2_nonzero")
                }
                lines.append(
                    f"| {level} | {label} | {pm(values['mae'])} | {pm(values['pearson'])} | "
                    f"{pm(values['acc2_nonzero'])} |"
                )
    lines += ["", "Utility 权重只由 official-train label 计算；valid label 只用于诊断和方法选择。", ""]
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_markdown)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
