#!/usr/bin/env python3
"""Partial/complete M0--M6 table and paired original-video clustered MAE CIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from train_video_adaptation_v2 import ROOT, atomic_json
from audit_tav_seed13_chain import CORE_METHODS

SEEDS = (13, 42, 2026)
METRICS = ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")
PAIRS = (("M0", "M1"), ("M1", "M3"), ("M3", "M4"), ("M3", "M6"), ("M4", "M6"))


def read_predictions(path):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len({r["parent_sample_id"] for r in rows}) != len(rows) or any(r["split"] != "valid" for r in rows):
        raise ValueError("unique valid-only predictions required")
    if len(rows) != 1871:
        raise ValueError("full official valid coverage required")
    return {r["parent_sample_id"]: r for r in rows}


def paired_errors(left, right):
    if set(left) != set(right):
        raise ValueError("paired sample sets differ")
    videos, errors = [], []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["target_sentiment"] != b["target_sentiment"] or a["video_id"] != b["video_id"]:
            raise ValueError("paired target/source mismatch")
        videos.append(a["video_id"])
        errors.append(abs(b["prediction"]-b["target_sentiment"]) - abs(a["prediction"]-a["target_sentiment"]))
    return videos, np.array(errors)


def clustered_mae(videos, errors, repetitions=10000, seed=2026):
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or len(errors) != len(videos) or not np.isfinite(errors).all():
        raise ValueError("finite per-utterance paired errors required")
    unique, inverse = np.unique(videos, return_inverse=True)
    sizes = np.bincount(inverse)
    sums = np.bincount(inverse, weights=errors)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repetitions, 256):
        chosen = rng.integers(0, len(unique), (min(256, repetitions-start), len(unique)))
        draws.extend((sums[chosen].sum(1) / sizes[chosen].sum(1)).tolist())
    return {"candidate_minus_baseline_mae": float(errors.mean()), "ci95": np.quantile(draws, [.025,.975]).tolist(),
            "video_clusters": len(unique), "utterances": len(errors), "repetitions": repetitions,
            "bootstrap_probability_delta_below_zero": float(np.mean(np.array(draws)<0))}


def summarize(base, repetitions=10000):
    if repetitions < 10000:
        raise ValueError("formal bootstrap requires at least 10000 draws")
    runs, predictions, table = {}, {}, {}
    for method in CORE_METHODS:
        selected = []
        for seed in SEEDS:
            name = f"{method}_seed{seed}"
            directory = base / "students" / name
            status = directory / "status.json"
            if not status.exists() or json.loads(status.read_text()).get("status") != "complete":
                continue
            report = json.loads((directory / "report.json").read_text())
            if report["method"] != method or report["seed"] != seed or report["official_test_evaluated"]:
                raise ValueError("run report identity differs")
            runs[name] = {"metrics": report["valid_metrics"], "best_epoch": report["best_epoch"],
                          "peak_gpu_memory_gib": report["peak_gpu_memory_gib"], "trainable_parameters": report["trainable_parameters"]}
            predictions[name] = read_predictions(directory / "predictions.jsonl")
            selected.append(report["valid_metrics"])
        table[method] = {"n": len(selected), "metrics": {k: {
            "mean": statistics.mean(r[k] for r in selected),
            "sample_sd": statistics.stdev(r[k] for r in selected) if len(selected)>1 else None,
        } for k in METRICS} if selected else {}}
    comparisons = {}
    for baseline, candidate in PAIRS:
        result, per_seed_errors = {}, []
        for seed in SEEDS:
            a, b = f"{baseline}_seed{seed}", f"{candidate}_seed{seed}"
            if a not in predictions or b not in predictions:
                continue
            videos, errors = paired_errors(predictions[a], predictions[b])
            result[str(seed)] = clustered_mae(videos, errors, repetitions)
            per_seed_errors.append(errors)
        if len(per_seed_errors) == 3:
            result["three_seed_mean"] = clustered_mae(videos, np.mean(per_seed_errors, axis=0), repetitions)
            result["three_seed_mean"]["interpretation"] = "video_sampling_CI_conditional_on_the_three_trained_seeds"
        if result:
            comparisons[f"{candidate}_vs_{baseline}"] = result
    diagnostics = {}
    for name, normal in predictions.items():
        directory = base / "students" / name
        for path in sorted(directory.glob("diagnostic_*.json")):
            rows = json.loads(path.read_text())
            perturbed = {r["parent_sample_id"]:r for r in rows}
            videos, errors = paired_errors(normal, perturbed)
            diagnostics[f"{name}/{path.stem}"] = clustered_mae(videos, errors, repetitions)
    report = {"complete": len(runs)==15, "completed_runs": len(runs), "expected_runs":15,
              "table": table, "runs": runs, "comparisons": comparisons, "diagnostics": diagnostics,
              "multiple_comparisons": "CIs are unadjusted; interpret primary comparisons and seed consistency jointly",
              "official_test_evaluated": False}
    atomic_json(report, base / "summary.json")
    lines = ["# TAV P0：M0/M1/M3/M4/M6", "", f"完成 {len(runs)}/15 组；official test 未使用。", "",
             "| Method | n | MAE | Pearson | Acc2 | F1 | Acc7 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, row in table.items():
        vals=[]
        for k in METRICS:
            value=row["metrics"].get(k)
            vals.append("—" if value is None else f"{value['mean']:.4f}" + (f" ± {value['sample_sd']:.4f}" if value["sample_sd"] is not None else ""))
        lines.append(f"| {method} | {row['n']} | " + " | ".join(vals) + " |")
    lines += ["", "MAE 为主指标；均值 ± 样本标准差。Acc2/F1 排除标签为零的样本，Acc7 由回归分数取整得到。", "",
              "## 视频聚类 paired bootstrap", "", "Δ = candidate − baseline，负值为改善；10,000 次以上按原视频重采样。三 seed 平均区间条件于这三个已训练模型，不估计新的训练随机性。"]
    for pair, result in comparisons.items():
        for seed, value in result.items():
            lo,hi=value["ci95"]
            lines.append(f"- {pair} / {seed}: ΔMAE {value['candidate_minus_baseline_mae']:+.6f}, 95% CI [{lo:+.6f}, {hi:+.6f}]")
    lines += ["", "各区间未作多重比较校正。不使用 0.005 硬停止门槛；不因某方法未提升而改写或隐藏结果。", "",
              "Utility 是训练集标签关联代理，并非因果效用。并发运行时间不用于独占设备效率结论。"]
    (base / "results.md").write_text("\n".join(lines)+"\n")
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/tav_main_v1")
    p.add_argument("--repetitions", type=int, default=10000)
    args=p.parse_args()
    print(json.dumps({"completed_runs": summarize(args.output, args.repetitions)["completed_runs"]}))
