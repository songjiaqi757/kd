#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

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


ROOT = Path("/home/wy/sjq/kd")
REPORTS = ROOT / "project/reports"
STUDENT = ROOT / "outputs/student"
SEEDS = (13, 42, 2026)
METHODS = {
    "full_kd": "B1 Full KD",
    "ensemble_pair": "C1 uniform ensemble-pair",
    "pair_snr": "C2 SNR pair",
    "reliability_utility_pair": "C4 Reliability × Utility",
    "selective50_interaction4": "selective-top50",
}
METRICS = ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_standard_deviation": statistics.stdev(values)}


def prediction_path(method: str, seed: int) -> Path:
    filename = "all_subset_predictions.jsonl" if method == "full_kd" else "predictions.jsonl"
    return STUDENT / f"fullscale_{method}_seed{seed}" / filename


def bootstrap_path(method: str, seed: int) -> Path:
    if method == "pair_snr":
        return REPORTS / f"stage_c0_pair_snr_vs_full_kd_seed{seed}_bootstrap.json"
    if seed == 13:
        return REPORTS / f"stage_c_seed13_{method}_vs_full_kd_bootstrap.json"
    return REPORTS / f"stage_c_{method}_vs_full_kd_seed{seed}_bootstrap.json"


def interaction_mae(predictions: dict[str, dict], teacher: dict[str, dict]) -> dict[str, float]:
    errors = {name: [] for name in INTERACTIONS}
    for sample_id, row in predictions.items():
        actual = interactions({key: float(value) for key, value in row["subset_predictions"].items()})
        for name in INTERACTIONS:
            errors[name].append(abs(actual[name] - float(teacher[sample_id]["interaction_mean"][name])))
    output = {name: statistics.mean(values) for name, values in errors.items()}
    output["pair"] = statistics.mean(output[name] for name in INTERACTIONS[:3])
    return output


def main() -> int:
    teacher = {
        str(row["sample_id"]): row
        for row in read_jsonl(STUDENT.parent / "probe/official_train_valid_interaction_reliability.jsonl")
        if row["split"] == "valid"
    }
    valid_ids = sorted(teacher)
    teacher_subsets = teacher_subset_ensemble([
        STUDENT.parent / f"probe/official_train_valid_seed{seed}/predictions.jsonl"
        for seed in (2026, 2027, 2028)
    ])
    axis_values = {
        "high_interaction": {
            key: sum(abs(float(teacher[key]["interaction_mean"][name])) for name in INTERACTIONS)
            for key in valid_ids
        },
        "high_uncertainty": {
            key: statistics.mean(float(teacher[key]["interaction_var"][name]) for name in INTERACTIONS)
            for key in valid_ids
        },
        "high_conflict": {
            key: statistics.mean((
                abs(teacher_subsets[key]["t"] - teacher_subsets[key]["a"]),
                abs(teacher_subsets[key]["t"] - teacher_subsets[key]["v"]),
                abs(teacher_subsets[key]["a"] - teacher_subsets[key]["v"]),
            )) for key in valid_ids
        },
    }
    axes = {}
    for name, values in axis_values.items():
        labels, cuts = tertiles(values)
        axes[name] = {"labels": labels, "tertile_cuts": cuts}

    per_seed = []
    for seed in SEEDS:
        predictions = {method: by_parent(prediction_path(method, seed), "valid") for method in METHODS}
        if any(set(rows) != set(valid_ids) for rows in predictions.values()):
            raise RuntimeError(f"valid ID mismatch for seed {seed}")
        baseline = predictions["full_kd"]
        item = {"seed": seed, "methods": {}}
        for method, rows in predictions.items():
            report = load(STUDENT / f"fullscale_{method}_seed{seed}" / "report.json")
            method_item = {
                "metrics": {metric: float(report["valid_metrics"][metric]) for metric in METRICS},
                "interaction_mae": interaction_mae(rows, teacher),
                "high_subgroups": {
                    axis: subgroup_metrics(baseline, rows, values["labels"])["high"]
                    for axis, values in axes.items()
                },
            }
            if method != "full_kd":
                method_item["paired_bootstrap"] = load(bootstrap_path(method, seed))["comparison"]
            item["methods"][method] = method_item
        per_seed.append(item)

    aggregate = {}
    for method in METHODS:
        aggregate[method] = {
            metric: mean_std([row["methods"][method]["metrics"][metric] for row in per_seed])
            for metric in METRICS
        }
        aggregate[method]["E_pair"] = mean_std([
            row["methods"][method]["interaction_mae"]["pair"] for row in per_seed
        ])
        for axis in axes:
            key = "full_kd" if method == "full_kd" else "pair_snr"
            aggregate[method][f"{axis}_mae"] = mean_std([
                row["methods"][method]["high_subgroups"][axis][key]["mae"] for row in per_seed
            ])

    gates = {}
    baseline_aggregate = aggregate["full_kd"]
    for method in METHODS:
        if method == "full_kd":
            continue
        mae_wins = sum(
            row["methods"][method]["metrics"]["mae"] < row["methods"]["full_kd"]["metrics"]["mae"]
            for row in per_seed
        )
        delta_mae = aggregate[method]["mae"]["mean"] - baseline_aggregate["mae"]["mean"]
        delta_pearson = aggregate[method]["pearson"]["mean"] - baseline_aggregate["pearson"]["mean"]
        high_wins = {
            axis: sum(
                row["methods"][method]["high_subgroups"][axis]["pair_snr"]["mae"]
                < row["methods"][method]["high_subgroups"][axis]["full_kd"]["mae"]
                for row in per_seed
            ) for axis in axes
        }
        gates[method] = {
            "mae_seed_wins": mae_wins,
            "mean_delta_mae": delta_mae,
            "mean_delta_pearson": delta_pearson,
            "high_subgroup_seed_wins": high_wins,
            "formal_multiseed_pass": mae_wins >= 2 and delta_mae < 0 and delta_pearson > -0.003,
            "recommended_effect_target_pass": delta_mae <= -0.005,
        }

    payload = {
        "scope": "official train/valid only; official test not used",
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "gates": gates,
        "formal_passed": [method for method, gate in gates.items() if gate["formal_multiseed_pass"]],
        "recommended_effect_passed": [method for method, gate in gates.items() if gate["recommended_effect_target_pass"]],
    }
    output_json = REPORTS / "stage_c_multiseed_results.json"
    output_md = REPORTS / "stage_c_multiseed_results.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pm(value: dict[str, float]) -> str:
        return f"{value['mean']:.4f} ± {value['sample_standard_deviation']:.4f}"

    lines = [
        "# Stage C 三 seed 结果", "",
        "official train/valid only；seeds 13、42、2026；official test 未使用。", "",
        "| 方法 | MAE | Pearson | Acc-2 | F1 | Acc-7 | E_pair | High-I MAE | High-U MAE | High-C MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, title in METHODS.items():
        item = aggregate[method]
        lines.append(
            f"| {title} | {pm(item['mae'])} | {pm(item['pearson'])} | {pm(item['acc2_nonzero'])} | "
            f"{pm(item['f1_weighted_nonzero'])} | {pm(item['acc7'])} | {pm(item['E_pair'])} | "
            f"{pm(item['high_interaction_mae'])} | {pm(item['high_uncertainty_mae'])} | "
            f"{pm(item['high_conflict_mae'])} |"
        )
    lines += ["", "## 相对 B1 的 Gate", "",
              "| 方法 | MAE wins | Mean ΔMAE | Mean ΔPearson | High-I wins | High-U wins | High-C wins | Formal pass | ΔMAE≤-0.005 |",
              "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for method, gate in gates.items():
        wins = gate["high_subgroup_seed_wins"]
        lines.append(
            f"| {METHODS[method]} | {gate['mae_seed_wins']}/3 | {gate['mean_delta_mae']:+.6f} | "
            f"{gate['mean_delta_pearson']:+.6f} | {wins['high_interaction']}/3 | "
            f"{wins['high_uncertainty']}/3 | {wins['high_conflict']}/3 | "
            f"{gate['formal_multiseed_pass']} | {gate['recommended_effect_target_pass']} |"
        )
    lines += ["", "## 逐 seed MAE paired bootstrap", "",
              "| 方法 | Seed | ΔMAE | 95% CI |", "|---|---:|---:|---:|"]
    for method in METHODS:
        if method == "full_kd":
            continue
        for row in per_seed:
            value = row["methods"][method]["paired_bootstrap"]["mae"]
            lines.append(
                f"| {METHODS[method]} | {row['seed']} | {value['candidate_minus_baseline']:+.6f} | "
                f"[{value['delta_ci95']['low']:+.6f}, {value['delta_ci95']['high']:+.6f}] |"
            )
    lines += ["", "Formal pass 要求平均 MAE 优于 B1、至少 2/3 seeds 获胜且 Pearson 平均下降不超过 0.003；方案建议的效果量目标为 ΔMAE≤-0.005。", ""]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "formal_passed": payload["formal_passed"], "recommended_effect_passed": payload["recommended_effect_passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
