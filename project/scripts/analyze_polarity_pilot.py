#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/wy/sjq/kd")
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from analyze_polarity_significance import exact_mcnemar_p, weighted_f1  # noqa: E402
from rdid_mosei.metrics import sentiment_metrics  # noqa: E402

SEEDS = (13, 42)
RUNS = {
    "B1": ("fullscale_full_kd_seed{seed}", "regression"),
    "B2": ("fullscale_subset4_seed{seed}", "regression"),
    "E1": ("polarity_e1_seed{seed}", "binary"),
    "E2": ({13: "polarity_e2_seed13_detached", 42: "polarity_e2_seed42"}, "binary"),
}
COMPARISONS = (("B1", "E1"), ("B2", "E2"), ("E1", "E2"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the two-seed E1/E2 polarity pilot")
    parser.add_argument("--student-root", type=Path, default=ROOT / "outputs/student")
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path, default=PROJECT / "reports/polarity_e1_e2_pilot.json")
    parser.add_argument("--output-md", type=Path, default=PROJECT / "reports/polarity_e1_e2_pilot.md")
    return parser.parse_args()


def directory_name(method: str, seed: int) -> str:
    specification = RUNS[method][0]
    return specification[seed] if isinstance(specification, dict) else specification.format(seed=seed)


def load_run(root: Path, method: str, seed: int) -> dict[str, Any]:
    directory = root / directory_name(method, seed)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    valid = {str(row["parent_sample_id"]): row for row in rows if row["split"] == "valid"}
    if len(valid) != 1871:
        raise RuntimeError(f"{method} seed {seed}: expected 1,871 valid utterances, found {len(valid)}")
    return {"directory": str(directory), "report": report, "rows": valid, "policy": RUNS[method][1]}


def predictions(run: dict[str, Any], sample_ids: list[str]) -> np.ndarray:
    if run["policy"] == "binary":
        return np.asarray([int(np.argmax(run["rows"][key]["binary_logits"])) == 1 for key in sample_ids])
    return np.asarray([float(run["rows"][key]["prediction"]) >= 0.0 for key in sample_ids])


def compare(
    baseline: dict[str, Any], candidate: dict[str, Any], repetitions: int, rng: np.random.Generator
) -> dict[str, Any]:
    if set(baseline["rows"]) != set(candidate["rows"]):
        raise ValueError("paired runs have different valid IDs")
    sample_ids = sorted(
        key for key, row in baseline["rows"].items() if float(row["target_sentiment"]) != 0.0
    )
    candidate_nonzero = {
        key for key, row in candidate["rows"].items() if float(row["target_sentiment"]) != 0.0
    }
    if set(sample_ids) != candidate_nonzero or len(sample_ids) != 1438:
        raise ValueError("paired runs have different non-zero valid IDs")
    targets = np.asarray([float(baseline["rows"][key]["target_sentiment"]) for key in sample_ids])
    candidate_targets = np.asarray([float(candidate["rows"][key]["target_sentiment"]) for key in sample_ids])
    if not np.array_equal(targets, candidate_targets):
        raise ValueError("paired target labels differ")
    labels = targets > 0.0
    left = predictions(baseline, sample_ids)
    right = predictions(candidate, sample_ids)
    left_acc, right_acc = float(np.mean(left == labels)), float(np.mean(right == labels))
    left_f1, right_f1 = weighted_f1(labels, left), weighted_f1(labels, right)
    acc_samples = np.empty(repetitions)
    f1_samples = np.empty(repetitions)
    for repetition in range(repetitions):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        sampled_left, sampled_right = left[indices], right[indices]
        acc_samples[repetition] = np.mean(sampled_right == sampled_labels) - np.mean(
            sampled_left == sampled_labels
        )
        f1_samples[repetition] = weighted_f1(sampled_labels, sampled_right) - weighted_f1(
            sampled_labels, sampled_left
        )
    left_correct, right_correct = left == labels, right == labels
    b = int(np.sum(left_correct & ~right_correct))
    c = int(np.sum(~left_correct & right_correct))
    baseline_regression = sentiment_metrics(
        [baseline["rows"][key]["target_sentiment"] for key in sorted(baseline["rows"])],
        [baseline["rows"][key]["prediction"] for key in sorted(baseline["rows"])],
    )
    candidate_regression = sentiment_metrics(
        [candidate["rows"][key]["target_sentiment"] for key in sorted(candidate["rows"])],
        [candidate["rows"][key]["prediction"] for key in sorted(candidate["rows"])],
    )
    return {
        "samples": len(labels),
        "acc2": {
            "baseline": left_acc,
            "candidate": right_acc,
            "delta": right_acc - left_acc,
            "ci95": [float(np.percentile(acc_samples, 2.5)), float(np.percentile(acc_samples, 97.5))],
        },
        "f1_weighted": {
            "baseline": left_f1,
            "candidate": right_f1,
            "delta": right_f1 - left_f1,
            "ci95": [float(np.percentile(f1_samples, 2.5)), float(np.percentile(f1_samples, 97.5))],
        },
        "mcnemar": {"b": b, "c": c, "exact_two_sided_p": exact_mcnemar_p(b, c)},
        "auxiliary": {
            "baseline_mae": baseline_regression["mae"],
            "candidate_mae": candidate_regression["mae"],
            "delta_mae": candidate_regression["mae"] - baseline_regression["mae"],
            "baseline_pearson": baseline_regression["pearson"],
            "candidate_pearson": candidate_regression["pearson"],
            "delta_pearson": candidate_regression["pearson"] - baseline_regression["pearson"],
        },
    }


def aggregate(seed_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def summary(path: tuple[str, ...]) -> dict[str, float]:
        values = []
        for seed in SEEDS:
            value: Any = seed_rows[str(seed)]
            for key in path:
                value = value[key]
            values.append(float(value))
        return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}

    return {
        "acc2_delta": summary(("acc2", "delta")),
        "f1_delta": summary(("f1_weighted", "delta")),
        "mae_delta": summary(("auxiliary", "delta_mae")),
        "pearson_delta": summary(("auxiliary", "delta_pearson")),
        "acc2_seed_wins": sum(seed_rows[str(seed)]["acc2"]["delta"] > 0 for seed in SEEDS),
        "f1_seed_wins": sum(seed_rows[str(seed)]["f1_weighted"]["delta"] > 0 for seed in SEEDS),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# E1 / E2 Polarity KD 两 seed Pilot",
        "",
        "official train/valid；seeds 13、42；official test 未使用。B1/B2 的 polarity 预测来自回归阈值，E1/E2 来自 binary head。",
        "每个比较使用 10,000 次 paired bootstrap 和 exact McNemar；差值为 candidate - baseline。",
        "",
        "| 对照 | Seed | ΔAcc-2 | 95% CI | ΔF1 | 95% CI | ΔMAE | ΔPearson | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in report["comparisons"]:
        for seed in map(str, SEEDS):
            row = comparison["seeds"][seed]
            lines.append(
                f"| {comparison['baseline']} → {comparison['candidate']} | {seed} | "
                f"{row['acc2']['delta']:+.6f} | [{row['acc2']['ci95'][0]:+.6f}, {row['acc2']['ci95'][1]:+.6f}] | "
                f"{row['f1_weighted']['delta']:+.6f} | [{row['f1_weighted']['ci95'][0]:+.6f}, {row['f1_weighted']['ci95'][1]:+.6f}] | "
                f"{row['auxiliary']['delta_mae']:+.6f} | {row['auxiliary']['delta_pearson']:+.6f} | "
                f"{row['mcnemar']['exact_two_sided_p']:.6f} |"
            )
    lines.extend(["", "## 两 seed 汇总", "", "| 对照 | Mean ΔAcc-2 | Mean ΔF1 | Mean ΔMAE | Mean ΔPearson | Acc/F1 wins |", "|---|---:|---:|---:|---:|---:|"])
    for comparison in report["comparisons"]:
        a = comparison["aggregate"]
        lines.append(
            f"| {comparison['baseline']} → {comparison['candidate']} | {a['acc2_delta']['mean']:+.6f} | "
            f"{a['f1_delta']['mean']:+.6f} | {a['mae_delta']['mean']:+.6f} | "
            f"{a['pearson_delta']['mean']:+.6f} | {a['acc2_seed_wins']}/2, {a['f1_seed_wins']}/2 |"
        )
    lines.extend(
        [
            "",
            "## Gate 判定",
            "",
            "- E1 相对 B1：一正一负，统计区间均跨 0，且 MAE 明显退化；No-Go。",
            "- E2 相对 B2：Acc-2 在 2/2 seeds 提高约 0.56–0.70 个百分点，但逐 seed CI/McNemar 未达到显著，且平均 MAE 退化约 0.025；不满足 regression safety gate。",
            "- E2 相对 E1：一正一负，不能证明 binary subset4 在 Binary Full KD 上稳定增加收益。",
            "- 结论：task-aligned binary supervision 对 Acc-2 有弱方向性价值，但当前 loss 造成明显连续任务代价。按预注册停止，不进入 Reliability、Boundary、LoRA 或权重搜索。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    runs = {(method, seed): load_run(args.student_root, method, seed) for method in RUNS for seed in SEEDS}
    comparisons = []
    for index, (baseline, candidate) in enumerate(COMPARISONS):
        seed_rows = {
            str(seed): compare(
                runs[(baseline, seed)],
                runs[(candidate, seed)],
                args.repetitions,
                np.random.default_rng(args.bootstrap_seed + index * 10_000 + seed),
            )
            for seed in SEEDS
        }
        comparisons.append(
            {"baseline": baseline, "candidate": candidate, "seeds": seed_rows, "aggregate": aggregate(seed_rows)}
        )
    report = {
        "schema_version": "rdid-polarity-e1-e2-pilot-v1",
        "seeds": list(SEEDS),
        "bootstrap_repetitions": args.repetitions,
        "official_test_evaluated": False,
        "runs": {f"{method}_seed{seed}": runs[(method, seed)]["directory"] for method in RUNS for seed in SEEDS},
        "comparisons": comparisons,
    }
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
