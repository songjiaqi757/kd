#!/usr/bin/env python3
"""Aggregate three-seed tables and per-seed video-cluster paired intervals."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_predictions(path, identities):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {}
    for row in rows:
        sample_id = row.get("parent_sample_id", row.get("id"))
        if sample_id not in identities or sample_id in result:
            raise ValueError(f"unknown/duplicate prediction ID in {path}: {sample_id}")
        target = float(row.get("target_sentiment", row.get("target")))
        if abs(target - identities[sample_id]["sentiment"]) > 1e-5:
            raise ValueError(f"target mismatch in {path}: {sample_id}")
        result[sample_id] = float(row["prediction"])
    return result


def discover(root, identities, split):
    runs = defaultdict(dict)
    for report_path in root.rglob("report.json"):
        predictions_path = report_path.parent / "predictions.jsonl"
        if not predictions_path.is_file():
            continue
        report = json.loads(report_path.read_text())
        report_split = report.get("split", "valid" if "valid_metrics" in report else None)
        if report_split != split:
            continue
        name = report.get("method", report.get("system", report_path.parent.name.rsplit("_seed", 1)[0]))
        seed = int(report["seed"])
        if seed in runs[name]:
            raise ValueError(f"duplicate {name} seed {seed}")
        runs[name][seed] = {"report": report, "predictions": read_predictions(predictions_path, identities), "path": str(report_path.parent)}
    return runs


def metric_summary(runs, seeds):
    rows = []
    for method, seed_runs in sorted(runs.items()):
        if set(seed_runs) != set(seeds):
            continue
        metric_rows = [
            seed_runs[seed]["report"].get("valid_metrics") or seed_runs[seed]["report"]["metrics"]
            for seed in seeds
        ]
        keys = sorted(set.intersection(*(set(row) for row in metric_rows)) - {"count", "nonzero_count"})
        summary = {"method": method, "seeds": list(seeds)}
        for key in keys:
            values = np.asarray([row[key] for row in metric_rows], dtype=np.float64)
            summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        summary["trainable_parameters"] = [seed_runs[seed]["report"].get("trainable_parameters") for seed in seeds]
        summary["elapsed_seconds"] = [seed_runs[seed]["report"].get("elapsed_seconds") for seed in seeds]
        summary["peak_gpu_memory_gib"] = [seed_runs[seed]["report"].get("peak_gpu_memory_gib") for seed in seeds]
        supervision = {
            (
                tuple(seed_runs[seed]["report"].get("teacher_subsets", [])),
                seed_runs[seed]["report"].get("probe_count"),
                seed_runs[seed]["report"].get("teacher_hidden_features"),
            )
            for seed in seeds
        }
        if len(supervision) != 1:
            raise ValueError(f"teacher-supervision metadata differs across seeds: {method}")
        subsets, probe_count, hidden_features = supervision.pop()
        summary.update(
            teacher_subsets=list(subsets),
            probe_count=probe_count,
            teacher_hidden_features=hidden_features,
        )
        rows.append(summary)
    return rows


def per_seed_cluster_bootstrap(candidate, baseline, identities, seeds, resamples, random_seed=20260917):
    common = None
    for seed in seeds:
        ids = set(candidate[seed]["predictions"]) & set(baseline[seed]["predictions"])
        common = ids if common is None else common & ids
    if not common:
        raise ValueError("comparison has no common IDs")
    expected = set(identities)
    if common != expected:
        raise ValueError(f"comparison coverage differs: {len(common)} != {len(expected)}")
    videos = sorted({identities[sample_id]["video_id"] for sample_id in common})
    by_video = {video: [sample_id for sample_id in common if identities[sample_id]["video_id"] == video] for video in videos}

    def effect(seed, selected_videos):
        ids = [sample_id for video in selected_videos for sample_id in by_video[video]]
        target = np.asarray([identities[sample_id]["sentiment"] for sample_id in ids])
        candidate_error = np.abs(target - np.asarray([candidate[seed]["predictions"][sample_id] for sample_id in ids]))
        baseline_error = np.abs(target - np.asarray([baseline[seed]["predictions"][sample_id] for sample_id in ids]))
        return float(candidate_error.mean() - baseline_error.mean())

    per_seed = {}
    for seed in seeds:
        rng = np.random.default_rng(random_seed + int(seed))
        samples = np.empty(resamples)
        for index in range(resamples):
            selected_videos = rng.choice(videos, size=len(videos), replace=True).tolist()
            samples[index] = effect(seed, selected_videos)
        per_seed[str(seed)] = {
            "candidate_minus_baseline_mae": effect(seed, videos),
            "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        }
    return {
        "mean_candidate_minus_baseline_mae": float(np.mean([
            row["candidate_minus_baseline_mae"] for row in per_seed.values()
        ])),
        "per_seed": per_seed,
        "resamples": resamples,
        "seed_resampling": "none; each training seed is reported separately",
        "sample_resampling": "source-video clusters sampled with replacement independently within each seed",
        "cross_seed_interval": None,
        "common_utterances": len(common),
        "video_clusters": len(videos),
    }


def markdown(table, comparisons):
    lines = ["# RDID-MSA unified results", "", "| Method | Teacher subsets | Probes | Hidden feature | MAE | Corr | Acc-2 | F1 | Acc-7 |", "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in table:
        def cell(key):
            value = row.get(key)
            return "—" if value is None else f"{value['mean']:.4f} ± {value['std']:.4f}"
        subsets = ",".join(subset.upper() for subset in row["teacher_subsets"]) or "None"
        probes = "—" if row["probe_count"] is None else str(row["probe_count"])
        hidden = "—" if row["teacher_hidden_features"] is None else ("yes" if row["teacher_hidden_features"] else "no")
        lines.append(f"| {row['method']} | {subsets} | {probes} | {hidden} | {cell('mae')} | {cell('pearson')} | {cell('acc2_nonzero')} | {cell('f1_weighted_nonzero')} | {cell('acc7')} |")
    lines.extend(["", "## Paired source-video cluster bootstrap by seed", "", "| Comparison | Three-seed mean ΔMAE (no CI) | Seed 13 ΔMAE [95% CI] | Seed 42 ΔMAE [95% CI] | Seed 2026 ΔMAE [95% CI] |", "|---|---:|---:|---:|---:|"])
    for name, result in comparisons.items():
        seed_cells = []
        for seed in (13, 42, 2026):
            row = result["per_seed"][str(seed)]
            seed_cells.append(f"{row['candidate_minus_baseline_mae']:.6f} [{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}]")
        lines.append(f"| {name} | {result['mean_candidate_minus_baseline_mae']:.6f} | " + " | ".join(seed_cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--comparison", action="append", default=[], help="candidate:baseline")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    identities = {}
    for row in manifest:
        sample_id = row.get("parent_sample_id", row["sample_id"])
        identity = {"sentiment": float(row["sentiment"]), "video_id": row["video_id"], "split": row["split"]}
        if identities.setdefault(sample_id, identity) != identity:
            raise ValueError(f"conflicting manifest identity: {sample_id}")
    identities = {sample_id: row for sample_id, row in identities.items() if row["split"] == args.split}
    seeds = (13, 42, 2026)
    runs = discover(args.experiment_root, identities, args.split)
    table = metric_summary(runs, seeds)
    comparisons = {}
    for specification in args.comparison:
        candidate, baseline = specification.split(":", 1)
        if candidate not in runs or baseline not in runs:
            raise ValueError(f"comparison method unavailable: {specification}")
        comparisons[specification] = per_seed_cluster_bootstrap(runs[candidate], runs[baseline], identities, seeds, args.resamples)
    payload = {"split": args.split, "seeds": list(seeds), "table": table, "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(table, comparisons))
    print(json.dumps({"methods": len(table), "comparisons": len(comparisons), "output": str(args.output)}))


if __name__ == "__main__":
    main()
