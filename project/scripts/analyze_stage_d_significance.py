#!/usr/bin/env python3
"""Final three-seed Stage D comparisons with video-cluster bootstrap."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.metrics import sentiment_metrics

METRICS = ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")
SEEDS = (13, 42, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE_ROOT / "dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl",
    )
    parser.add_argument("--outputs", type=Path, default=WORKSPACE_ROOT / "outputs/student")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/stage_d_final_significance.json")
    parser.add_argument("--markdown", type=Path, default=PROJECT_ROOT / "reports/stage_d_final_significance.md")
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def manifest_identities(path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in read_jsonl(path):
        sample_id = str(row["parent_sample_id"])
        identity = {
            "video_id": str(row["video_id"]),
            "split": str(row["split"]),
            "target_sentiment": float(row["sentiment"]),
        }
        previous = result.setdefault(sample_id, identity)
        if previous != identity:
            raise ValueError(f"inconsistent manifest identity for {sample_id}")
    return result


def load_predictions(path: Path, identities: dict[str, dict], split: str = "valid") -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in read_jsonl(path):
        if row["split"] != split:
            continue
        sample_id = str(row["parent_sample_id"])
        if sample_id in result:
            raise ValueError(f"duplicate prediction ID in {path}: {sample_id}")
        if sample_id not in identities:
            raise ValueError(f"prediction ID missing from manifest: {sample_id}")
        identity = identities[sample_id]
        if identity["split"] != split or not math.isclose(
            float(row["target_sentiment"]), identity["target_sentiment"], abs_tol=1e-9
        ):
            raise ValueError(f"prediction identity/label mismatch: {sample_id}")
        result[sample_id] = {
            **row,
            "video_id": identity["video_id"],
        }
    if not result:
        raise ValueError(f"no {split} predictions in {path}")
    return result


def exact_mcnemar(targets: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, int | float]:
    keep = targets != 0.0
    truth = targets[keep] > 0.0
    left = baseline[keep] >= 0.0
    right = candidate[keep] >= 0.0
    baseline_only = int(np.sum((left == truth) & (right != truth)))
    candidate_only = int(np.sum((left != truth) & (right == truth)))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(baseline_only, candidate_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "baseline_correct_candidate_wrong": baseline_only,
        "baseline_wrong_candidate_correct": candidate_only,
        "discordant": discordant,
        "exact_two_sided_p_value": p_value,
    }


def cluster_comparison(
    baseline: dict[str, dict], candidate: dict[str, dict], repetitions: int, seed: int
) -> dict:
    if set(baseline) != set(candidate):
        raise ValueError("paired runs do not contain identical valid sample IDs")
    sample_ids = sorted(baseline)
    for sample_id in sample_ids:
        if (
            float(baseline[sample_id]["target_sentiment"])
            != float(candidate[sample_id]["target_sentiment"])
            or baseline[sample_id]["video_id"] != candidate[sample_id]["video_id"]
        ):
            raise ValueError(f"paired identity mismatch: {sample_id}")
    targets = np.asarray([baseline[x]["target_sentiment"] for x in sample_ids], dtype=np.float64)
    left = np.asarray([baseline[x]["prediction"] for x in sample_ids], dtype=np.float64)
    right = np.asarray([candidate[x]["prediction"] for x in sample_ids], dtype=np.float64)
    if not np.isfinite(np.concatenate((targets, left, right))).all():
        raise ValueError("non-finite target or prediction")

    video_ids = np.asarray([baseline[x]["video_id"] for x in sample_ids])
    videos = sorted(set(video_ids.tolist()))
    members = [np.flatnonzero(video_ids == video) for video in videos]
    baseline_metrics = sentiment_metrics(targets, left)
    candidate_metrics = sentiment_metrics(targets, right)
    samples = {name: [] for name in METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        chosen = rng.integers(0, len(videos), len(videos))
        indices = np.concatenate([members[index] for index in chosen])
        before = sentiment_metrics(targets[indices], left[indices])
        after = sentiment_metrics(targets[indices], right[indices])
        for name in METRICS:
            delta = float(after[name]) - float(before[name])
            if math.isfinite(delta):
                samples[name].append(delta)

    comparison = {}
    for name in METRICS:
        values = np.asarray(samples[name], dtype=np.float64)
        better = values < 0.0 if name == "mae" else values > 0.0
        comparison[name] = {
            "baseline": float(baseline_metrics[name]),
            "candidate": float(candidate_metrics[name]),
            "candidate_minus_baseline": float(candidate_metrics[name]) - float(baseline_metrics[name]),
            "cluster_ci95": np.quantile(values, (0.025, 0.975)).tolist(),
            "bootstrap_probability_candidate_better": float(np.mean(better)),
        }
    return {
        "utterances": len(sample_ids),
        "video_clusters": len(videos),
        "repetitions": repetitions,
        "comparison": comparison,
        "acc2_nonzero_mcnemar": exact_mcnemar(targets, left, right),
    }


def aggregate_reports(outputs: Path) -> dict:
    methods = {
        "B1_full_kd": "fullscale_full_kd_seed{seed}",
        "D1_full_kd_lora": "stage_d_d1_full_kd_lora_seed{seed}",
        "D2_ru_lora": "stage_d_d2_ru_lora_seed{seed}",
    }
    result = {}
    for name, template in methods.items():
        rows = [json.loads((outputs / template.format(seed=seed) / "report.json").read_text()) for seed in SEEDS]
        result[name] = {
            metric: {
                "mean": statistics.mean(float(row["valid_metrics"][metric]) for row in rows),
                "sample_std": statistics.stdev(float(row["valid_metrics"][metric]) for row in rows),
            }
            for metric in METRICS
        }
    return result


def markdown(report: dict) -> str:
    lines = [
        "# Stage D 三种子最终显著性分析",
        "",
        "仅使用 official valid；差值均为 candidate − baseline。Bootstrap 按原始 video ID 聚类重采样 10,000 次。",
        "",
        "## 三种子指标",
        "",
        "| 方法 | MAE | Pearson | Acc-2 | F1 | Acc-7 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["three_seed_metrics"].items():
        formatted = [f"{values[m]['mean']:.4f} ± {values[m]['sample_std']:.4f}" for m in METRICS]
        lines.append(f"| {name} | " + " | ".join(formatted) + " |")
    lines.extend(["", "## 逐 seed 聚类配对比较", "", "| 对照 | Seed | ΔMAE [95% CI] | ΔPearson | ΔAcc-2 | ΔF1 | ΔAcc-7 | McNemar p |", "|---|---:|---|---:|---:|---:|---:|---:|"])
    for name, by_seed in report["comparisons"].items():
        for seed, value in by_seed.items():
            c = value["comparison"]
            lo, hi = c["mae"]["cluster_ci95"]
            lines.append(
                f"| {name} | {seed} | {c['mae']['candidate_minus_baseline']:+.6f} "
                f"[{lo:+.6f}, {hi:+.6f}] | {c['pearson']['candidate_minus_baseline']:+.6f} | "
                f"{c['acc2_nonzero']['candidate_minus_baseline']:+.6f} | "
                f"{c['f1_weighted_nonzero']['candidate_minus_baseline']:+.6f} | "
                f"{c['acc7']['candidate_minus_baseline']:+.6f} | "
                f"{value['acc2_nonzero_mcnemar']['exact_two_sided_p_value']:.6g} |"
            )
    lines.extend(["", "Official test 未读取、未评估。", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    identities = manifest_identities(args.manifest)
    comparisons = {"D1_minus_B1": {}, "D2_minus_D1": {}}
    for index, seed in enumerate(SEEDS):
        b1 = load_predictions(args.outputs / f"fullscale_full_kd_seed{seed}/predictions.jsonl", identities)
        d1 = load_predictions(args.outputs / f"stage_d_d1_full_kd_lora_seed{seed}/predictions.jsonl", identities)
        d2 = load_predictions(args.outputs / f"stage_d_d2_ru_lora_seed{seed}/predictions.jsonl", identities)
        comparisons["D1_minus_B1"][str(seed)] = cluster_comparison(
            b1, d1, args.repetitions, args.bootstrap_seed + index
        )
        comparisons["D2_minus_D1"][str(seed)] = cluster_comparison(
            d1, d2, args.repetitions, args.bootstrap_seed + 10_000 + index
        )
    report = {
        "protocol": "official-valid video-cluster paired bootstrap",
        "manifest": str(args.manifest),
        "bootstrap_repetitions": args.repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "three_seed_metrics": aggregate_reports(args.outputs),
        "comparisons": comparisons,
        "official_test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.markdown)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
