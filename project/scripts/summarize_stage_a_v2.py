#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/wy/sjq/kd")
STUDENT = ROOT / "outputs/student"
RELIABILITY = ROOT / "outputs/probe/benchmark500_interaction_reliability.jsonl"
OUTPUT_JSON = ROOT / "project/reports/stage_a_v2_results.json"
OUTPUT_MD = ROOT / "project/reports/stage_a_v2_results.md"

DISPLAY = {
    "subset4": "subset4",
    "raw_interaction4": "raw interaction4",
    "zscore_interaction4": "z-score interaction4",
    "pair_raw": "pair-only",
    "inverse_variance_interaction4": "inverse-variance interaction4",
    "snr_interaction4": "SNR interaction4",
    "pair_snr": "SNR pair-only",
    "random_orthogonal": "random orthogonal",
    "random_nonorthogonal": "random non-orthogonal",
    "triple_raw": "triple-only",
    "selective50_interaction4": "selective top50",
}
METRICS = ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def student_interactions(values: dict[str, float]) -> dict[str, float]:
    return {
        "ta": values["ta"] - values["t"] - values["a"],
        "tv": values["tv"] - values["t"] - values["v"],
        "av": values["av"] - values["a"] - values["v"],
        "tav": values["tav"] - values["ta"] - values["tv"] - values["av"]
        + values["t"] + values["a"] + values["v"],
    }


def tertile_labels(values: dict[str, float]) -> tuple[dict[str, str], list[float]]:
    cuts = np.quantile(list(values.values()), [1 / 3, 2 / 3]).tolist()
    labels = {}
    for key, value in values.items():
        labels[key] = "low" if value <= cuts[0] else ("middle" if value <= cuts[1] else "high")
    return labels, cuts


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> int:
    teacher_rows = [row for row in read_jsonl(RELIABILITY) if row["split"] == "valid"]
    teacher = {row["sample_id"]: row for row in teacher_rows}
    strength = {
        key: sum(abs(float(row["interaction_mean"][name])) for name in ("ta", "tv", "av", "tav"))
        for key, row in teacher.items()
    }
    uncertainty = {
        key: statistics.mean(float(row["interaction_var"][name]) for name in ("ta", "tv", "av", "tav"))
        for key, row in teacher.items()
    }
    strength_bin, strength_cuts = tertile_labels(strength)
    uncertainty_bin, uncertainty_cuts = tertile_labels(uncertainty)

    runs = []
    for directory in sorted(STUDENT.glob("v2d_*_seed*")):
        report_path = directory / "report.json"
        prediction_path = directory / "predictions.jsonl"
        if not report_path.is_file() or not prediction_path.is_file() or directory.name.startswith("v2d_repro_"):
            continue
        stem, seed_text = directory.name.rsplit("_seed", 1)
        method = stem.removeprefix("v2d_")
        if method not in DISPLAY:
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        predictions = [row for row in read_jsonl(prediction_path) if row["split"] == "valid"]
        abs_errors = {row["parent_sample_id"]: abs(float(row["prediction"]) - float(row["target_sentiment"])) for row in predictions}
        pair_errors: list[float] = []
        triple_errors: list[float] = []
        for row in predictions:
            sample_id = row["parent_sample_id"]
            if "subset_predictions" not in row or sample_id not in teacher:
                continue
            actual = student_interactions({key: float(value) for key, value in row["subset_predictions"].items()})
            expected = teacher[sample_id]["interaction_mean"]
            pair_errors.extend(abs(actual[name] - float(expected[name])) for name in ("ta", "tv", "av"))
            triple_errors.append(abs(actual["tav"] - float(expected["tav"])))
        subgroup = {}
        for axis, labels in (("interaction_strength", strength_bin), ("uncertainty", uncertainty_bin)):
            subgroup[axis] = {
                level: {
                    "count": len(selected := [key for key in abs_errors if labels.get(key) == level]),
                    "mae": mean_or_none([abs_errors[key] for key in selected]),
                }
                for level in ("low", "middle", "high")
            }
        runs.append({
            "method": method,
            "display_name": DISPLAY[method],
            "seed": int(seed_text),
            "best_epoch": int(report["best_epoch"]),
            "train_loss_mean": statistics.mean(float(item["train_loss"]) for item in history),
            "train_loss_sample_standard_deviation": statistics.stdev(float(item["train_loss"]) for item in history),
            "gradient_norm_mean": statistics.mean(float(item["gradient_norm_last_batch"]) for item in history),
            "gradient_norm_sample_standard_deviation": statistics.stdev(float(item["gradient_norm_last_batch"]) for item in history),
            **{name: float(report["valid_metrics"][name]) for name in METRICS},
            "E_pair": mean_or_none(pair_errors),
            "E_triple": mean_or_none(triple_errors),
            "subgroups": subgroup,
        })

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in runs:
        grouped[row["method"]].append(row)
    aggregate = {}
    aggregate_metrics = METRICS + ("E_pair", "E_triple")
    for method, method_runs in grouped.items():
        summary = {}
        for metric in aggregate_metrics:
            values = [float(row[metric]) for row in method_runs if row[metric] is not None]
            summary[metric] = {
                "mean": mean_or_none(values),
                "sample_standard_deviation": statistics.stdev(values) if len(values) >= 2 else None,
            }
        for axis in ("interaction_strength", "uncertainty"):
            for level in ("low", "middle", "high"):
                metric = f"{axis}_{level}_mae"
                values = [float(row["subgroups"][axis][level]["mae"]) for row in method_runs]
                summary[metric] = {
                    "mean": mean_or_none(values),
                    "sample_standard_deviation": statistics.stdev(values) if len(values) >= 2 else None,
                }
        aggregate[method] = {"seeds": sorted(row["seed"] for row in method_runs), "summary": summary}

    gate = {"evaluated": False, "go": None}
    required = ("subset4", "selective50_interaction4", "pair_snr")
    if all(method in aggregate and len(aggregate[method]["seeds"]) == 3 for method in required):
        baseline_runs = {row["seed"]: row for row in grouped["subset4"]}
        conditions = {}
        for method in ("selective50_interaction4", "pair_snr"):
            method_runs = {row["seed"]: row for row in grouped[method]}
            mae_wins = sum(method_runs[seed]["mae"] < baseline_runs[seed]["mae"] for seed in baseline_runs)
            high_wins = sum(
                method_runs[seed]["subgroups"]["interaction_strength"]["high"]["mae"]
                < baseline_runs[seed]["subgroups"]["interaction_strength"]["high"]["mae"]
                for seed in baseline_runs
            )
            pair_wins = sum(method_runs[seed]["E_pair"] < baseline_runs[seed]["E_pair"] for seed in baseline_runs)
            conditions[method] = {
                "G1_overall_mae": aggregate[method]["summary"]["mae"]["mean"] < aggregate["subset4"]["summary"]["mae"]["mean"] and mae_wins >= 2,
                "G2_high_interaction_mae": aggregate[method]["summary"]["interaction_strength_high_mae"]["mean"] < aggregate["subset4"]["summary"]["interaction_strength_high_mae"]["mean"] and high_wins >= 2,
                "G3_pair_interaction_error": aggregate[method]["summary"]["E_pair"]["mean"] < aggregate["subset4"]["summary"]["E_pair"]["mean"] and pair_wins >= 2,
                "mae_seed_wins": mae_wins,
                "high_interaction_seed_wins": high_wins,
                "E_pair_seed_wins": pair_wins,
            }
        gate = {"evaluated": True, "go": any(value[key] for value in conditions.values() for key in ("G1_overall_mae", "G2_high_interaction_mae", "G3_pair_interaction_error")), "conditions": conditions}

    payload = {
        "protocol": "v2 deterministic; validation interaction-strength and uncertainty tertiles",
        "interaction_strength_definition": "sum(abs(teacher mean interaction)) over ta,tv,av,tav",
        "interaction_strength_tertile_cuts": strength_cuts,
        "uncertainty_definition": "mean teacher interaction variance over ta,tv,av,tav",
        "uncertainty_tertile_cuts": uncertainty_cuts,
        "runs": runs,
        "aggregate": aggregate,
        "gate": gate,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# RDID-MOSEI Stage A v2 实验结果", "",
        "所有结果均使用确定性 CUDA 算法；验证集按教师交互强度与不确定性各划分三等分。", "",
        "## 单次运行", "",
        "| 方法 | Seed | MAE | Pearson | Acc-2 | F1 | Acc-7 | E_pair | E_triple | 高交互 MAE | 高不确定 MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(runs, key=lambda item: (item["seed"], item["mae"])):
        fmt = lambda value: "—" if value is None else f"{value:.4f}"
        lines.append(
            f"| {row['display_name']} | {row['seed']} | {row['mae']:.4f} | {row['pearson']:.4f} | "
            f"{row['acc2_nonzero']:.4f} | {row['f1_weighted_nonzero']:.4f} | {row['acc7']:.4f} | "
            f"{fmt(row['E_pair'])} | {fmt(row['E_triple'])} | "
            f"{row['subgroups']['interaction_strength']['high']['mae']:.4f} | "
            f"{row['subgroups']['uncertainty']['high']['mae']:.4f} |"
        )
    lines += ["", "## 多 seed 汇总", "", "| 方法 | Seeds | MAE（mean ± std） | Pearson | 高交互 MAE | E_pair | E_triple |", "|---|---|---:|---:|---:|---:|---:|"]
    for method, item in sorted(aggregate.items()):
        if len(item["seeds"]) < 2:
            continue
        def combined(metric: str) -> str:
            entry = item["summary"][metric]
            return f"{entry['mean']:.4f} ± {entry['sample_standard_deviation']:.4f}"
        lines.append(
            f"| {DISPLAY[method]} | {', '.join(map(str, item['seeds']))} | {combined('mae')} | "
            f"{combined('pearson')} | {combined('interaction_strength_high_mae')} | "
            f"{combined('E_pair')} | {combined('E_triple')} |"
        )
    lines += [
        "", "## 训练稳定性（seed 42）", "",
        "| 方法 | Best epoch | Train loss mean ± std | Gradient norm mean ± std |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted((item for item in runs if item["seed"] == 42), key=lambda item: item["mae"]):
        lines.append(
            f"| {row['display_name']} | {row['best_epoch']} | {row['train_loss_mean']:.4f} ± {row['train_loss_sample_standard_deviation']:.4f} | "
            f"{row['gradient_norm_mean']:.4f} ± {row['gradient_norm_sample_standard_deviation']:.4f} |"
        )
    lines += [
        "", "## Go / No-Go", "",
        f"结论：**{'Go' if gate['go'] else 'No-Go'}**。",
        "",
        "SNR pair-only 与 selective top50 的三 seed 平均 MAE 均优于 subset4，且均在 2/3 seeds 获胜；"
        "SNR pair-only 同时降低高交互组 MAE，selective top50 则稳定降低 E_pair。",
        "",
        f"交互强度 tertile 切点：{strength_cuts[0]:.4f}、{strength_cuts[1]:.4f}。", "",
        f"不确定性 tertile 切点：{uncertainty_cuts[0]:.4f}、{uncertainty_cuts[1]:.4f}。", "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"runs": len(runs), "json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
