#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/wy/sjq/kd")
STUDENT = ROOT / "outputs/student"
SEEDS = (13, 42, 2026)
METHOD_PATHS = {
    "B0": "fullscale_student_seed{seed}",
    "B1": "fullscale_full_kd_seed{seed}",
    "B2": "fullscale_subset4_seed{seed}",
    "RU": "fullscale_reliability_utility_pair_seed{seed}",
}
COMPARISONS = (
    ("B0", "B1"),
    ("B1", "B2"),
    ("RU", "B2"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-seed polarity paired significance analysis")
    parser.add_argument("--student-root", type=Path, default=STUDENT)
    parser.add_argument("--output-json", type=Path, default=ROOT / "project/reports/p0_polarity_significance.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "project/reports/p0_polarity_significance.md")
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    selected = {
        str(row["parent_sample_id"]): row
        for row in rows
        if row["split"] == "valid" and float(row["target_sentiment"]) != 0.0
    }
    if len(selected) != 1438:
        raise RuntimeError(f"expected 1,438 non-zero valid utterances in {path}, found {len(selected)}")
    return selected


def weighted_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    value = 0.0
    for label in (False, True):
        support = int(np.sum(labels == label))
        true_positive = int(np.sum((labels == label) & (predictions == label)))
        false_positive = int(np.sum((labels != label) & (predictions == label)))
        false_negative = int(np.sum((labels == label) & (predictions != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        value += support / len(labels) * (0.0 if denominator == 0 else 2 * true_positive / denominator)
    return value


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(b, c) + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def compare(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise ValueError("paired runs do not contain identical valid sample IDs")
    sample_ids = sorted(baseline)
    baseline_targets = np.asarray([float(baseline[key]["target_sentiment"]) for key in sample_ids])
    candidate_targets = np.asarray([float(candidate[key]["target_sentiment"]) for key in sample_ids])
    if not np.array_equal(baseline_targets, candidate_targets):
        raise ValueError("paired runs disagree on target labels")
    labels = baseline_targets > 0.0
    baseline_predictions = np.asarray([float(baseline[key]["prediction"]) >= 0.0 for key in sample_ids])
    candidate_predictions = np.asarray([float(candidate[key]["prediction"]) >= 0.0 for key in sample_ids])
    baseline_acc = float(np.mean(baseline_predictions == labels))
    candidate_acc = float(np.mean(candidate_predictions == labels))
    baseline_f1 = weighted_f1(labels, baseline_predictions)
    candidate_f1 = weighted_f1(labels, candidate_predictions)

    acc_deltas = np.empty(repetitions, dtype=np.float64)
    f1_deltas = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        sampled_baseline = baseline_predictions[indices]
        sampled_candidate = candidate_predictions[indices]
        acc_deltas[repetition] = np.mean(sampled_candidate == sampled_labels) - np.mean(
            sampled_baseline == sampled_labels
        )
        f1_deltas[repetition] = weighted_f1(sampled_labels, sampled_candidate) - weighted_f1(
            sampled_labels, sampled_baseline
        )

    baseline_correct = baseline_predictions == labels
    candidate_correct = candidate_predictions == labels
    b = int(np.sum(baseline_correct & ~candidate_correct))
    c = int(np.sum(~baseline_correct & candidate_correct))
    return {
        "samples": len(sample_ids),
        "acc2": {
            "baseline": baseline_acc,
            "candidate": candidate_acc,
            "candidate_minus_baseline": candidate_acc - baseline_acc,
            "delta_ci95": {
                "low": float(np.percentile(acc_deltas, 2.5)),
                "high": float(np.percentile(acc_deltas, 97.5)),
            },
            "bootstrap_probability_candidate_better": float(np.mean(acc_deltas > 0.0)),
        },
        "f1_weighted": {
            "baseline": baseline_f1,
            "candidate": candidate_f1,
            "candidate_minus_baseline": candidate_f1 - baseline_f1,
            "delta_ci95": {
                "low": float(np.percentile(f1_deltas, 2.5)),
                "high": float(np.percentile(f1_deltas, 97.5)),
            },
            "bootstrap_probability_candidate_better": float(np.mean(f1_deltas > 0.0)),
        },
        "mcnemar": {
            "baseline_correct_candidate_wrong_b": b,
            "baseline_wrong_candidate_correct_c": c,
            "discordant": b + c,
            "exact_two_sided_p": exact_mcnemar_p(b, c),
        },
    }


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P0：Acc-2 / F1 三 seed 配对显著性",
        "",
        "仅使用 official valid 的 1,438 条非零标签 utterance；official test 未使用。",
        "每个比较执行 10,000 次 utterance-level paired bootstrap；Acc-2 同时使用 exact McNemar test。",
        "候选差值均为 `candidate - baseline`。",
        "",
        "| 对照 | Seed | ΔAcc-2 | 95% CI | ΔF1 | 95% CI | b | c | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in report["comparisons"]:
        for seed, item in comparison["seeds"].items():
            acc = item["acc2"]
            f1 = item["f1_weighted"]
            mc = item["mcnemar"]
            lines.append(
                f"| {comparison['baseline']} → {comparison['candidate']} | {seed} | "
                f"{acc['candidate_minus_baseline']:+.6f} | "
                f"[{acc['delta_ci95']['low']:+.6f}, {acc['delta_ci95']['high']:+.6f}] | "
                f"{f1['candidate_minus_baseline']:+.6f} | "
                f"[{f1['delta_ci95']['low']:+.6f}, {f1['delta_ci95']['high']:+.6f}] | "
                f"{mc['baseline_correct_candidate_wrong_b']} | "
                f"{mc['baseline_wrong_candidate_correct_c']} | {mc['exact_two_sided_p']:.6f} |"
            )
    lines.extend(["", "## 三 seed汇总", "", "| 对照 | Mean ΔAcc-2 | Mean ΔF1 | Acc-2 胜出 seeds |", "|---|---:|---:|---:|"])
    for comparison in report["comparisons"]:
        aggregate = comparison["aggregate"]
        lines.append(
            f"| {comparison['baseline']} → {comparison['candidate']} | "
            f"{aggregate['acc2_delta']['mean']:+.6f} ± {aggregate['acc2_delta']['sample_std']:.6f} | "
            f"{aggregate['f1_delta']['mean']:+.6f} ± {aggregate['f1_delta']['sample_std']:.6f} | "
            f"{aggregate['acc2_seed_wins']}/3 |"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            "- B1 相对 B0 的 Acc-2/F1 是方向性改善，但不是 3/3 seeds，逐 seed统计也不一致显著。",
            "- B2 相对 B1 平均约提升 0.49 个百分点 Acc-2，但仅 2/3 seeds 为正，所有 bootstrap CI 跨 0，所有 McNemar p > 0.05。",
            "- B2 相对 RU 为 3/3 seeds 点估计更高，但逐 seed bootstrap CI 和 McNemar 均未给出显著证据。",
            "- 因此 B2 的 polarity 优势应表述为方向性信号，不能表述为已证实的稳定提升。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    cache: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for method in METHOD_PATHS:
        for seed in SEEDS:
            path = args.student_root / METHOD_PATHS[method].format(seed=seed) / "predictions.jsonl"
            cache[(method, seed)] = load_predictions(path)

    comparisons = []
    for comparison_index, (baseline, candidate) in enumerate(COMPARISONS):
        seed_results = {}
        for seed in SEEDS:
            rng = np.random.default_rng(args.bootstrap_seed + comparison_index * 10_000 + seed)
            seed_results[str(seed)] = compare(
                cache[(baseline, seed)],
                cache[(candidate, seed)],
                repetitions=args.repetitions,
                rng=rng,
            )
        acc_deltas = [seed_results[str(seed)]["acc2"]["candidate_minus_baseline"] for seed in SEEDS]
        f1_deltas = [seed_results[str(seed)]["f1_weighted"]["candidate_minus_baseline"] for seed in SEEDS]
        comparisons.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "seeds": seed_results,
                "aggregate": {
                    "acc2_delta": mean_std(acc_deltas),
                    "f1_delta": mean_std(f1_deltas),
                    "acc2_seed_wins": sum(value > 0.0 for value in acc_deltas),
                    "f1_seed_wins": sum(value > 0.0 for value in f1_deltas),
                },
            }
        )

    report = {
        "schema_version": "rdid-p0-polarity-significance-v1",
        "split": "official-valid",
        "zero_label_policy": "exclude_true_zero",
        "prediction_policy": "regression_score_greater_than_or_equal_to_zero_is_positive",
        "seeds": list(SEEDS),
        "bootstrap_repetitions": args.repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "official_test_evaluated": False,
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
