#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_interaction_utility import (
    INTERACTIONS,
    by_parent,
    interactions,
    read_jsonl,
    subgroup_metrics,
    teacher_subset_ensemble,
    tertiles,
)
from rdid_mosei.metrics import sentiment_metrics


ROOT = Path("/home/wy/sjq/kd")
REPORTS = ROOT / "project/reports"
CANDIDATES = {
    "C1_uniform_ensemble_pair": ROOT / "outputs/student/fullscale_ensemble_pair_seed13/predictions.jsonl",
    "C2_snr_pair": ROOT / "outputs/student/fullscale_pair_snr_seed13/predictions.jsonl",
    "C3_utility_pair": ROOT / "outputs/student/fullscale_utility_pair_seed13/predictions.jsonl",
    "C4_reliability_utility_pair": ROOT / "outputs/student/fullscale_reliability_utility_pair_seed13/predictions.jsonl",
    "selective50_interaction4": ROOT / "outputs/student/fullscale_selective50_interaction4_seed13/predictions.jsonl",
}
BOOTSTRAP_NAMES = {
    "C1_uniform_ensemble_pair": "ensemble_pair",
    "C2_snr_pair": "pair_snr",
    "C3_utility_pair": "utility_pair",
    "C4_reliability_utility_pair": "reliability_utility_pair",
    "selective50_interaction4": "selective50_interaction4",
}


def interaction_mae(predictions: dict[str, dict], teacher: dict[str, dict]) -> dict[str, float]:
    values = {name: [] for name in INTERACTIONS}
    for sample_id, row in predictions.items():
        actual = interactions({key: float(value) for key, value in row["subset_predictions"].items()})
        for name in INTERACTIONS:
            values[name].append(abs(actual[name] - float(teacher[sample_id]["interaction_mean"][name])))
    result = {name: statistics.mean(items) for name, items in values.items()}
    result["pair"] = statistics.mean(result[name] for name in INTERACTIONS[:3])
    return result


def main() -> int:
    teacher = {
        str(row["sample_id"]): row
        for row in read_jsonl(ROOT / "outputs/probe/official_train_valid_interaction_reliability.jsonl")
        if row["split"] == "valid"
    }
    teacher_subsets = teacher_subset_ensemble([
        ROOT / f"outputs/probe/official_train_valid_seed{seed}/predictions.jsonl"
        for seed in (2026, 2027, 2028)
    ])
    valid_ids = sorted(teacher)
    strength = {
        key: sum(abs(float(teacher[key]["interaction_mean"][name])) for name in INTERACTIONS)
        for key in valid_ids
    }
    uncertainty = {
        key: statistics.mean(float(teacher[key]["interaction_var"][name]) for name in INTERACTIONS)
        for key in valid_ids
    }
    conflict = {
        key: statistics.mean((
            abs(teacher_subsets[key]["t"] - teacher_subsets[key]["a"]),
            abs(teacher_subsets[key]["t"] - teacher_subsets[key]["v"]),
            abs(teacher_subsets[key]["a"] - teacher_subsets[key]["v"]),
        ))
        for key in valid_ids
    }
    axes = {}
    for name, values in (("high_interaction", strength), ("high_uncertainty", uncertainty), ("high_conflict", conflict)):
        labels, cuts = tertiles(values)
        axes[name] = {"labels": labels, "tertile_cuts": cuts}

    baseline = by_parent(
        ROOT / "outputs/student/fullscale_full_kd_seed13/all_subset_predictions.jsonl", "valid"
    )
    targets = [float(baseline[key]["target_sentiment"]) for key in valid_ids]
    baseline_metrics = sentiment_metrics(targets, [float(baseline[key]["prediction"]) for key in valid_ids])
    baseline_interactions = interaction_mae(baseline, teacher)
    rows = []
    for method, path in CANDIDATES.items():
        candidate = by_parent(path, "valid")
        if set(candidate) != set(baseline):
            raise RuntimeError(f"valid IDs differ for {method}")
        metrics = sentiment_metrics(targets, [float(candidate[key]["prediction"]) for key in valid_ids])
        candidate_interactions = interaction_mae(candidate, teacher)
        subgroups = {
            name: subgroup_metrics(baseline, candidate, axis["labels"])["high"]
            for name, axis in axes.items()
        }
        high_deltas = {
            name: float(value["pair_snr"]["mae"] - value["full_kd"]["mae"])
            for name, value in subgroups.items()
        }
        bootstrap_name = BOOTSTRAP_NAMES[method]
        bootstrap_path = (
            REPORTS / "stage_c0_pair_snr_vs_full_kd_seed13_bootstrap.json"
            if method == "C2_snr_pair"
            else REPORTS / f"stage_c_seed13_{bootstrap_name}_vs_full_kd_bootstrap.json"
        )
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))["comparison"]
        delta_mae = float(metrics["mae"] - baseline_metrics["mae"])
        delta_pearson = float(metrics["pearson"] - baseline_metrics["pearson"])
        gates = {
            "G1_mae_at_least_0.003_better": delta_mae <= -0.003,
            "G2_pearson_not_down_more_than_0.003": delta_pearson > -0.003,
            "G3_not_fidelity_only": delta_mae < 0,
            "G4_high_subgroup_at_least_0.003_better": any(value <= -0.003 for value in high_deltas.values()),
        }
        rows.append({
            "method": method,
            "metrics": {name: float(metrics[name]) for name in ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")},
            "delta_vs_full_kd": {"mae": delta_mae, "pearson": delta_pearson},
            "interaction_mae": candidate_interactions,
            "delta_interaction_mae": {
                name: candidate_interactions[name] - baseline_interactions[name]
                for name in (*INTERACTIONS, "pair")
            },
            "high_subgroups": subgroups,
            "high_subgroup_delta_mae": high_deltas,
            "paired_bootstrap": bootstrap,
            "gates": gates,
            "promote_to_multiseed": all(gates.values()),
        })

    payload = {
        "scope": "official train/valid seed 13 screening; official test not used",
        "baseline": {
            "method": "B1 Full KD",
            "metrics": {name: float(baseline_metrics[name]) for name in ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")},
            "interaction_mae": baseline_interactions,
        },
        "subgroup_tertile_cuts": {name: axis["tertile_cuts"] for name, axis in axes.items()},
        "candidates": rows,
        "promoted": [row["method"] for row in rows if row["promote_to_multiseed"]],
    }
    output_json = REPORTS / "stage_c_seed13_screening.json"
    output_md = REPORTS / "stage_c_seed13_screening.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage C seed-13 screening", "",
        "范围：official train/valid；official test 未使用。Gate 基线为 B1 Full KD。", "",
        "| 方法 | MAE | ΔMAE | Pearson | ΔPearson | ΔE_pair | High-I ΔMAE | High-U ΔMAE | High-C ΔMAE | Bootstrap 95% CI | Promote |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        ci = row["paired_bootstrap"]["mae"]["delta_ci95"]
        lines.append(
            f"| {row['method']} | {row['metrics']['mae']:.6f} | {row['delta_vs_full_kd']['mae']:+.6f} | "
            f"{row['metrics']['pearson']:.6f} | {row['delta_vs_full_kd']['pearson']:+.6f} | "
            f"{row['delta_interaction_mae']['pair']:+.6f} | "
            f"{row['high_subgroup_delta_mae']['high_interaction']:+.6f} | "
            f"{row['high_subgroup_delta_mae']['high_uncertainty']:+.6f} | "
            f"{row['high_subgroup_delta_mae']['high_conflict']:+.6f} | "
            f"[{ci['low']:+.6f}, {ci['high']:+.6f}] | "
            f"{'YES' if row['promote_to_multiseed'] else 'NO'} |"
        )
    lines += ["", "## Gate details", ""]
    for row in rows:
        lines.append(f"- `{row['method']}`: " + ", ".join(f"{key}={value}" for key, value in row["gates"].items()))
    lines += ["", "只有 Promote=YES 的方法可补 seed 42/2026；该规则不使用 official-test 信息。", ""]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "promoted": payload["promoted"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
