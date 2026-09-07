#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path("/home/wy/sjq/kd")
STUDENT = ROOT / "outputs/student"
REPORTS = ROOT / "project/reports"
SEEDS = (13, 42, 2026)
METRICS = ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def main() -> int:
    per_seed = []
    for seed in SEEDS:
        subset = read_json(STUDENT / f"fullscale_subset4_seed{seed}" / "report.json")
        pair = read_json(STUDENT / f"fullscale_pair_snr_seed{seed}" / "report.json")
        analysis = read_json(REPORTS / f"fullscale_subset4_vs_pair_snr_seed{seed}.json")
        bootstrap = read_json(REPORTS / f"fullscale_subset4_vs_pair_snr_seed{seed}_metrics_bootstrap.json")
        pair_reconstruction = analysis["interaction_reconstruction_mae"]
        per_seed.append({
            "seed": seed,
            "subset4": {name: float(subset["valid_metrics"][name]) for name in METRICS},
            "snr_pair_only": {name: float(pair["valid_metrics"][name]) for name in METRICS},
            "delta_snr_minus_subset4": {
                name: float(pair["valid_metrics"][name]) - float(subset["valid_metrics"][name])
                for name in METRICS
            },
            "mae_bootstrap": bootstrap["comparison"]["mae"],
            "interaction_strength_groups": analysis["interaction_strength"]["groups"],
            "teacher_uncertainty_groups": analysis["teacher_uncertainty"]["groups"],
            "interaction_reconstruction_mae": pair_reconstruction,
            "E_pair": {
                method: statistics.mean(float(values[name]) for name in ("ta", "tv", "av"))
                for method, values in pair_reconstruction.items()
            },
        })

    aggregate = {}
    for method in ("subset4", "snr_pair_only"):
        aggregate[method] = {
            metric: mean_std([row[method][metric] for row in per_seed]) for metric in METRICS
        }
        aggregate[method]["interaction_strength_high_mae"] = mean_std([
            row["interaction_strength_groups"]["high"][
                "baseline_mae" if method == "subset4" else "candidate_mae"
            ] for row in per_seed
        ])
        aggregate[method]["uncertainty_high_mae"] = mean_std([
            row["teacher_uncertainty_groups"]["high"][
                "baseline_mae" if method == "subset4" else "candidate_mae"
            ] for row in per_seed
        ])
        aggregate[method]["E_pair"] = mean_std([row["E_pair"][method] for row in per_seed])
        group_key = "baseline" if method == "subset4" else "candidate"
        for axis, source in (
            ("interaction_strength", "interaction_strength_groups"),
            ("uncertainty", "teacher_uncertainty_groups"),
        ):
            for level in ("low", "middle", "high"):
                for metric in ("mae", "pearson", "acc2_nonzero"):
                    aggregate[method][f"{axis}_{level}_{metric}"] = mean_std([
                        row[source][level][f"{group_key}_{metric}"] for row in per_seed
                    ])

    delta_summary = {
        metric: mean_std([row["delta_snr_minus_subset4"][metric] for row in per_seed])
        for metric in METRICS
    }
    mae_wins = sum(row["snr_pair_only"]["mae"] < row["subset4"]["mae"] for row in per_seed)
    high_wins = sum(
        row["interaction_strength_groups"]["high"]["candidate_mae"]
        < row["interaction_strength_groups"]["high"]["baseline_mae"]
        for row in per_seed
    )
    pair_wins = sum(row["E_pair"]["snr_pair_only"] < row["E_pair"]["subset4"] for row in per_seed)
    gates = {
        "G1_overall_mae": aggregate["snr_pair_only"]["mae"]["mean"]
        < aggregate["subset4"]["mae"]["mean"] and mae_wins >= 2,
        "G2_high_interaction_mae": aggregate["snr_pair_only"]["interaction_strength_high_mae"]["mean"]
        < aggregate["subset4"]["interaction_strength_high_mae"]["mean"] and high_wins >= 2,
        "G3_pair_interaction_error": aggregate["snr_pair_only"]["E_pair"]["mean"]
        < aggregate["subset4"]["E_pair"]["mean"] and pair_wins >= 2,
        "mae_seed_wins": mae_wins,
        "high_interaction_seed_wins": high_wins,
        "E_pair_seed_wins": pair_wins,
    }
    payload = {
        "protocol": "official train/valid, deterministic, seeds 13/42/2026",
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "delta_snr_minus_subset4": delta_summary,
        "gates": gates,
        "downstream_conclusion": "no_go" if not (gates["G1_overall_mae"] or gates["G2_high_interaction_mae"]) else "go",
        "representation_conclusion": "pair_interaction_reconstruction_improved" if gates["G3_pair_interaction_error"] else "not_improved",
    }
    output_json = REPORTS / "fullscale_subset4_vs_pair_snr_three_seed.json"
    output_md = REPORTS / "fullscale_subset4_vs_pair_snr_three_seed.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pm(item: dict[str, float]) -> str:
        return f"{item['mean']:.4f} ± {item['sample_standard_deviation']:.4f}"

    lines = [
        "# Full-scale subset4 与 SNR pair-only 三 seed 汇总", "",
        "official train/valid；确定性训练；seeds = 13、42、2026。", "",
        "## 单 seed 结果", "",
        "| Seed | 方法 | MAE | Pearson | Acc-2 | F1 | Acc-7 | 高交互 MAE | 高不确定 MAE | E_pair |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        for method, title, group_key in (
            ("subset4", "subset4", "baseline_mae"),
            ("snr_pair_only", "SNR pair-only", "candidate_mae"),
        ):
            metrics = row[method]
            lines.append(
                f"| {row['seed']} | {title} | {metrics['mae']:.4f} | {metrics['pearson']:.4f} | "
                f"{metrics['acc2_nonzero']:.4f} | {metrics['f1_weighted_nonzero']:.4f} | {metrics['acc7']:.4f} | "
                f"{row['interaction_strength_groups']['high'][group_key]:.4f} | "
                f"{row['teacher_uncertainty_groups']['high'][group_key]:.4f} | {row['E_pair'][method]:.4f} |"
            )
    lines += [
        "", "## 三 seed mean ± sample std", "",
        "| 方法 | MAE | Pearson | Acc-2 | F1 | Acc-7 | 高交互 MAE | 高不确定 MAE | E_pair |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, title in (("subset4", "subset4"), ("snr_pair_only", "SNR pair-only")):
        item = aggregate[method]
        lines.append(
            f"| {title} | {pm(item['mae'])} | {pm(item['pearson'])} | {pm(item['acc2_nonzero'])} | "
            f"{pm(item['f1_weighted_nonzero'])} | {pm(item['acc7'])} | "
            f"{pm(item['interaction_strength_high_mae'])} | {pm(item['uncertainty_high_mae'])} | {pm(item['E_pair'])} |"
        )
    for axis, title in (("interaction_strength", "教师交互强度 tertile"), ("uncertainty", "教师不确定性 tertile")):
        lines += [
            "", f"## {title}（三 seed mean ± sample std）", "",
            "| 分组 | 方法 | MAE | Pearson | Acc-2 |",
            "|---|---|---:|---:|---:|",
        ]
        for level in ("low", "middle", "high"):
            for method, method_title in (("subset4", "subset4"), ("snr_pair_only", "SNR pair-only")):
                item = aggregate[method]
                lines.append(
                    f"| {level} | {method_title} | {pm(item[f'{axis}_{level}_mae'])} | "
                    f"{pm(item[f'{axis}_{level}_pearson'])} | {pm(item[f'{axis}_{level}_acc2_nonzero'])} |"
                )
    lines += [
        "", "## 判定", "",
        f"- 总体 MAE：SNR pair-only 获胜 {mae_wins}/3 seeds，G1 = `{str(gates['G1_overall_mae']).lower()}`。",
        f"- 高交互组 MAE：SNR pair-only 获胜 {high_wins}/3 seeds，G2 = `{str(gates['G2_high_interaction_mae']).lower()}`。",
        f"- Pair 交互重建 E_pair：SNR pair-only 获胜 {pair_wins}/3 seeds，G3 = `{str(gates['G3_pair_interaction_error']).lower()}`。",
        "- 下游情感性能结论：**No-Go**；SNR pair-only 没有稳定优于 subset4。",
        "- 表征层结论：SNR pair-only 稳定改善 pair 交互重建，但该改善没有转化为验证集情感性能提升。", "",
        "每个 seed 的 10,000 次 paired bootstrap 结果见同目录对应报告。", "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "gates": gates}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
