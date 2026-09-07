#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path("/home/wy/sjq/kd")
STUDENT = ROOT / "outputs/student"
REPORTS = ROOT / "project/reports"
SEEDS = (13, 42, 2026)
METHODS = {
    "student": "B0 Student-only",
    "full_kd": "B1 Full KD",
    "subset4": "B2 subset4",
}
METRICS = ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")
COMPARISONS = {
    "full_kd_vs_student": ("B1 Full KD", "B0 Student-only"),
    "subset4_vs_student": ("B2 subset4", "B0 Student-only"),
    "subset4_vs_full_kd": ("B2 subset4", "B1 Full KD"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def main() -> int:
    per_seed = []
    for seed in SEEDS:
        item = {"seed": seed}
        for method in METHODS:
            report = load(STUDENT / f"fullscale_{method}_seed{seed}" / "report.json")
            item[method] = {
                **{name: float(report["valid_metrics"][name]) for name in METRICS},
                "best_epoch": int(report["best_epoch"]),
                "epochs_run": int(report["epochs_run"]),
                "elapsed_seconds": float(report["elapsed_seconds"]),
            }
        per_seed.append(item)

    aggregate = {
        method: {
            metric: mean_std([row[method][metric] for row in per_seed])
            for metric in METRICS
        }
        for method in METHODS
    }
    comparisons = {}
    for name in COMPARISONS:
        rows = []
        for seed in SEEDS:
            report = load(REPORTS / f"stage_b_{name}_seed{seed}_bootstrap.json")
            rows.append({"seed": seed, **report["comparison"]})
        comparisons[name] = rows

    mae_wins = {
        "full_kd_vs_student": sum(row["full_kd"]["mae"] < row["student"]["mae"] for row in per_seed),
        "subset4_vs_student": sum(row["subset4"]["mae"] < row["student"]["mae"] for row in per_seed),
        "subset4_vs_full_kd": sum(row["subset4"]["mae"] < row["full_kd"]["mae"] for row in per_seed),
    }
    payload = {
        "protocol": "official train/valid only; deterministic; seeds 13/42/2026; no official test evaluation",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "paired_bootstrap": comparisons,
        "mae_seed_wins": mae_wins,
        "conclusion": "Full KD is the strongest Stage B baseline on official validation; subset4 does not improve over Full KD.",
    }
    output_json = REPORTS / "stage_b_baselines_three_seed.json"
    output_md = REPORTS / "stage_b_baselines_three_seed.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pm(value: dict[str, float]) -> str:
        return f"{value['mean']:.4f} ± {value['sample_standard_deviation']:.4f}"

    lines = [
        "# Stage B 主基线三 seed 汇总", "",
        "仅使用 official train/valid；确定性训练；seeds = 13、42、2026；未读取或评估 official test。", "",
        "## 单 seed", "",
        "| Seed | 方法 | MAE | Pearson | Acc-2 | F1 | Acc-7 | Best epoch |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        for method, title in METHODS.items():
            value = row[method]
            lines.append(
                f"| {row['seed']} | {title} | {value['mae']:.4f} | {value['pearson']:.4f} | "
                f"{value['acc2_nonzero']:.4f} | {value['f1_weighted_nonzero']:.4f} | "
                f"{value['acc7']:.4f} | {value['best_epoch']} |"
            )
    lines += [
        "", "## 三 seed mean ± sample std", "",
        "| 方法 | MAE | Pearson | Acc-2 | F1 | Acc-7 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, title in METHODS.items():
        value = aggregate[method]
        lines.append(
            f"| {title} | {pm(value['mae'])} | {pm(value['pearson'])} | "
            f"{pm(value['acc2_nonzero'])} | {pm(value['f1_weighted_nonzero'])} | {pm(value['acc7'])} |"
        )
    lines += [
        "", "## MAE paired bootstrap（candidate − baseline）", "",
        "| 对照 | Seed | ΔMAE | 95% CI | P(candidate 更优) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, (candidate, baseline) in COMPARISONS.items():
        for row in comparisons[name]:
            value = row["mae"]
            lines.append(
                f"| {candidate} vs {baseline} | {row['seed']} | {value['candidate_minus_baseline']:+.6f} | "
                f"[{value['delta_ci95']['low']:+.6f}, {value['delta_ci95']['high']:+.6f}] | "
                f"{value['bootstrap_probability_candidate_better']:.4f} |"
            )
    lines += [
        "", "## 结论", "",
        f"- B1 Full KD 的 MAE 在 {mae_wins['full_kd_vs_student']}/3 seeds 优于 B0。",
        f"- B2 subset4 的 MAE 在 {mae_wins['subset4_vs_student']}/3 seeds 优于 B0。",
        f"- B2 subset4 仅在 {mae_wins['subset4_vs_full_kd']}/3 seeds 优于 B1。",
        "- official-valid 上 B1 Full KD 是当前最强主基线；B2 subset4 没有证明能在 B1 之上继续提升。",
        "- 本报告不包含 official-test 结果，不能作为最终 test 结论。", "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "mae_seed_wins": mae_wins}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
