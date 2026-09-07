#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


INTERACTIONS = ("ta", "tv", "av", "tav")
LEVELS = ("low", "middle", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-scale subset4 versus SNR pair-only paired and subgroup analysis"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--teacher-reliability", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def valid_predictions(path: Path) -> dict[str, dict]:
    return {
        str(row["parent_sample_id"]): row
        for row in read_jsonl(path)
        if row["split"] == "valid"
    }


def tertiles(values: dict[str, float]) -> tuple[dict[str, str], list[float]]:
    cuts = np.quantile(np.asarray(list(values.values()), dtype=np.float64), [1 / 3, 2 / 3])
    labels = {
        sample_id: ("low" if value <= cuts[0] else "middle" if value <= cuts[1] else "high")
        for sample_id, value in values.items()
    }
    return labels, [float(value) for value in cuts]


def paired_error_test(
    baseline_error: np.ndarray,
    candidate_error: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict:
    # Negative candidate-minus-baseline differences favor the candidate.
    differences = candidate_error - baseline_error
    n = differences.size
    bootstrap_means = np.empty(repetitions, dtype=np.float64)
    null_means = np.empty(repetitions, dtype=np.float64)
    centered = differences - differences.mean()
    for index in range(repetitions):
        sample = rng.integers(0, n, n)
        bootstrap_means[index] = differences[sample].mean()
        null_means[index] = centered[sample].mean()
    standard_deviation = float(differences.std(ddof=1))
    observed = float(differences.mean())
    return {
        "samples": int(n),
        "baseline_mae": float(baseline_error.mean()),
        "candidate_mae": float(candidate_error.mean()),
        "candidate_minus_baseline_mae": observed,
        "delta_ci95": {
            "low": float(np.percentile(bootstrap_means, 2.5)),
            "high": float(np.percentile(bootstrap_means, 97.5)),
        },
        "two_sided_centered_bootstrap_p_value": float(
            (np.count_nonzero(np.abs(null_means) >= abs(observed)) + 1) / (repetitions + 1)
        ),
        "paired_effect_size_dz": observed / standard_deviation if standard_deviation > 0 else math.nan,
        "bootstrap_probability_candidate_better": float(np.mean(bootstrap_means < 0)),
    }


def subgroup_rows(
    baseline_error: dict[str, float],
    candidate_error: dict[str, float],
    labels: dict[str, str],
) -> dict[str, dict]:
    result = {}
    for level in LEVELS:
        selected = [sample_id for sample_id in baseline_error if labels[sample_id] == level]
        left = np.asarray([baseline_error[sample_id] for sample_id in selected])
        right = np.asarray([candidate_error[sample_id] for sample_id in selected])
        result[level] = {
            "count": len(selected),
            "baseline_mae": float(left.mean()),
            "candidate_mae": float(right.mean()),
            "candidate_minus_baseline_mae": float((right - left).mean()),
        }
    return result


def student_interactions(values: dict[str, float]) -> dict[str, float]:
    return {
        "ta": values["ta"] - values["t"] - values["a"],
        "tv": values["tv"] - values["t"] - values["v"],
        "av": values["av"] - values["a"] - values["v"],
        "tav": values["tav"] - values["ta"] - values["tv"] - values["av"]
        + values["t"] + values["a"] + values["v"],
    }


def interaction_errors(predictions: dict[str, dict], teacher: dict[str, dict]) -> dict[str, float]:
    errors = {name: [] for name in INTERACTIONS}
    for sample_id, row in predictions.items():
        actual = student_interactions({key: float(value) for key, value in row["subset_predictions"].items()})
        expected = teacher[sample_id]["interaction_mean"]
        for name in INTERACTIONS:
            errors[name].append(abs(actual[name] - float(expected[name])))
    return {name: float(np.mean(values)) for name, values in errors.items()}


def main() -> int:
    args = parse_args()
    baseline = valid_predictions(args.baseline)
    candidate = valid_predictions(args.candidate)
    teacher = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.teacher_reliability)
        if row["split"] == "valid"
    }
    if not (set(baseline) == set(candidate) == set(teacher)):
        raise ValueError("baseline, candidate, and teacher valid sample IDs differ")

    sample_ids = sorted(baseline)
    baseline_error = {
        sample_id: abs(float(baseline[sample_id]["prediction"]) - float(baseline[sample_id]["target_sentiment"]))
        for sample_id in sample_ids
    }
    candidate_error = {
        sample_id: abs(float(candidate[sample_id]["prediction"]) - float(candidate[sample_id]["target_sentiment"]))
        for sample_id in sample_ids
    }
    strength = {
        sample_id: sum(abs(float(teacher[sample_id]["interaction_mean"][name])) for name in INTERACTIONS)
        for sample_id in sample_ids
    }
    uncertainty = {
        sample_id: float(np.mean([teacher[sample_id]["interaction_var"][name] for name in INTERACTIONS]))
        for sample_id in sample_ids
    }
    strength_labels, strength_cuts = tertiles(strength)
    uncertainty_labels, uncertainty_cuts = tertiles(uncertainty)
    left = np.asarray([baseline_error[sample_id] for sample_id in sample_ids])
    right = np.asarray([candidate_error[sample_id] for sample_id in sample_ids])

    report = {
        "comparison": "SNR pair-only minus subset4; negative MAE delta favors SNR pair-only",
        "run_seed": args.run_seed,
        "split": "official valid",
        "samples": len(sample_ids),
        "bootstrap_repetitions": args.repetitions,
        "bootstrap_seed": args.seed,
        "paired_absolute_error": paired_error_test(
            left, right, args.repetitions, np.random.default_rng(args.seed)
        ),
        "interaction_strength": {
            "definition": "sum(abs(teacher interaction mean)) over ta,tv,av,tav",
            "tertile_cuts": strength_cuts,
            "groups": subgroup_rows(baseline_error, candidate_error, strength_labels),
        },
        "teacher_uncertainty": {
            "definition": "mean teacher interaction variance over ta,tv,av,tav",
            "tertile_cuts": uncertainty_cuts,
            "groups": subgroup_rows(baseline_error, candidate_error, uncertainty_labels),
        },
        "interaction_reconstruction_mae": {
            "subset4": interaction_errors(baseline, teacher),
            "snr_pair_only": interaction_errors(candidate, teacher),
        },
        "inputs": {
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
            "teacher_reliability": str(args.teacher_reliability),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    paired = report["paired_absolute_error"]
    lines = [
        f"# Full-scale subset4 与 SNR pair-only 对照（seed {args.run_seed}）", "",
        f"official valid 共 {len(sample_ids)} 条样本；paired bootstrap {args.repetitions:,} 次，seed={args.seed}。", "",
        "## 整体配对结果", "",
        "| subset4 MAE | SNR pair-only MAE | 差值（SNR-subset4） | 95% CI | 双侧 p 值 | 配对效应量 dz | P(SNR 更优) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {paired['baseline_mae']:.6f} | {paired['candidate_mae']:.6f} | "
        f"{paired['candidate_minus_baseline_mae']:+.6f} | "
        f"[{paired['delta_ci95']['low']:+.6f}, {paired['delta_ci95']['high']:+.6f}] | "
        f"{paired['two_sided_centered_bootstrap_p_value']:.4f} | {paired['paired_effect_size_dz']:+.4f} | "
        f"{paired['bootstrap_probability_candidate_better']:.4f} |", "",
    ]
    for key, title in (("interaction_strength", "教师交互强度分组"), ("teacher_uncertainty", "教师不确定性分组")):
        item = report[key]
        lines += [
            f"## {title}", "",
            f"tertile 切点：{item['tertile_cuts'][0]:.6f}、{item['tertile_cuts'][1]:.6f}。", "",
            "| 分组 | N | subset4 MAE | SNR pair-only MAE | 差值 |",
            "|---|---:|---:|---:|---:|",
        ]
        for level in LEVELS:
            row = item["groups"][level]
            lines.append(
                f"| {level} | {row['count']} | {row['baseline_mae']:.6f} | "
                f"{row['candidate_mae']:.6f} | {row['candidate_minus_baseline_mae']:+.6f} |"
            )
        lines.append("")
    lines += [
        "## 教师交互重建 MAE", "",
        "| 方法 | ta | tv | av | tav |", "|---|---:|---:|---:|---:|",
    ]
    for key, title in (("subset4", "subset4"), ("snr_pair_only", "SNR pair-only")):
        row = report["interaction_reconstruction_mae"][key]
        lines.append(f"| {title} | {row['ta']:.6f} | {row['tv']:.6f} | {row['av']:.6f} | {row['tav']:.6f} |")
    lines += [
        "", "## 解释", "",
        "该报告仅基于 seed 42。置信区间或 p 值不能替代跨训练随机种子的稳定性检验。", "",
    ]
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_markdown)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
