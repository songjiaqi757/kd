#!/usr/bin/env python3
"""Freeze dataset-specific three-Probe interaction assets for MOSI or MOSEI."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]
from prepare_cv2_assets import mobius_numpy
from prepare_tav_protocol import utility_from_train
from train_video_adaptation_v2 import atomic_json, sha256

SUBSETS = ("t", "a", "v", "ta", "tv", "av", "tav")
PROBE_SEEDS = (2026, 2027, 2028)


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise FileExistsError("protocol already frozen; use a new output directory")
    manifest = args.manifest.resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate window ID")
    parents = {}
    for row in rows:
        identity = {key: row[key] for key in ("parent_sample_id", "split", "video_id", "sentiment", "class_7_index")}
        if parents.setdefault(row["parent_sample_id"], identity) != identity:
            raise ValueError("conflicting parent identity")
    counts = Counter(row["split"] for row in parents.values())
    if counts != {"train": args.expected_train, "valid": args.expected_valid}:
        raise ValueError(f"official parent counts differ: {counts}")
    split_videos = {split: {row["video_id"] for row in parents.values() if row["split"] == split} for split in ("train", "valid")}
    if split_videos["train"] & split_videos["valid"]:
        raise ValueError("train/valid source-video overlap")
    keys = sorted(parents)
    train_mask = np.asarray([parents[key]["split"] == "train" for key in keys])
    labels = np.asarray([parents[key]["sentiment"] for key in keys])

    features = args.features.resolve()
    feature_config = json.loads((features / "run_config.json").read_text())
    if feature_config["manifest_sha256"] != sha256(manifest) or feature_config.get("video_timing_policy") != "sampled_fps_v2":
        raise ValueError("teacher feature timing/manifest mismatch")
    complete = np.load(features / "completed.npy", mmap_mode="r")
    hidden = np.load(features / "features.npy", mmap_mode="r")
    if complete.shape != (len(rows) * len(SUBSETS),) or not complete.all() or hidden.shape[0] != complete.size:
        raise ValueError("incomplete teacher features")
    if not np.isfinite(hidden).all():
        raise ValueError("non-finite teacher features")

    source_paths = [manifest, Path(__file__), features / "run_config.json", features / "index.jsonl", features / "features.npy", features / "completed.npy"]
    arrays, baselines, temperatures = [], [], {}
    probe_predictions = {}
    for seed in PROBE_SEEDS:
        directory = args.probes_root.resolve() / f"seed{seed}"
        report_path = directory / "report.json"
        predictions_path = directory / "predictions.jsonl"
        report = json.loads(report_path.read_text())
        temperature = float(report["calibration"]["after"]["temperature"])
        if report["seed"] != seed or Path(report["features"]).resolve() != features or report["official_test_evaluated"] or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"invalid Probe: {seed}")
        temperatures[str(seed)] = temperature
        grouped = {}
        for line in predictions_path.open():
            item = json.loads(line)
            key = item["parent_sample_id"], item["subset"]
            if key in grouped or key[0] not in parents or key[1] not in SUBSETS:
                raise ValueError("unexpected/duplicate teacher prediction")
            grouped[key] = item
        expected = {(parent, subset) for parent in parents for subset in SUBSETS}
        if set(grouped) != expected:
            raise ValueError(f"Probe coverage differs for seed {seed}")
        raw = np.asarray([[grouped[key, subset]["probe_score"] for subset in SUBSETS] for key in keys])
        baseline = float(raw[train_mask, -1].mean())
        arrays.append(mobius_numpy(raw, baseline))
        baselines.append(baseline)
        probe_predictions[seed] = predictions_path
        source_paths.extend([report_path, predictions_path])
    stack = np.stack(arrays)
    mean, variance = stack.mean(0), stack.var(0, ddof=1)
    correlation, utility = utility_from_train(mean[train_mask], labels[train_mask])
    targets = output / "interaction_targets.jsonl"
    with targets.open("w") as handle:
        for index, key in enumerate(keys):
            handle.write(json.dumps({**parents[key], "mean": mean[index].tolist(), "variance": variance[index].tolist()}) + "\n")
    source_paths.append(targets)

    protocol = {
        "schema": "rdid-msa-three-probe-protocol-v1",
        "dataset": args.dataset,
        "teacher_probe_seed": 2026,
        "teacher_temperature": temperatures["2026"],
        "probe_temperatures": temperatures,
        "teacher_feature_manifest": str(features / "index.jsonl"),
        "teacher_feature_checksum": sha256(features / "features.npy"),
        "student_cache_manifest": None,
        "student_cache_checksum": None,
        "student_input_policy": "online_raw_media",
        "student_input_manifest": str(manifest),
        "student_input_checksum": sha256(manifest),
        "manifest": str(manifest),
        "teacher_targets": str(probe_predictions[2026]),
        "teacher_probe_report": str(args.probes_root.resolve() / "seed2026/report.json"),
        "interaction_targets": str(targets),
        "interaction_subsets": list(SUBSETS),
        "interaction_target_policy": "same_3_probe_mean",
        "probe_seeds": list(PROBE_SEEDS),
        "per_probe_empty_baseline": baselines,
        "student_empty_baseline": float(np.mean(baselines)),
        "empty_baseline_definition": "per_probe_official_train_TAV_utterance_mean",
        "utility_definition": "train_only_absolute_Pearson_floor_0.01_mean_normalized",
        "utility_is_task_association_proxy_not_causal_utility": True,
        "utility_raw_abs_correlation": correlation.tolist(),
        "utility_normalized": utility.tolist(),
        "reliability": {"epsilon_inside_sqrt": 1e-4, "clip_after_sample_mean_normalization": [0.25, 4.0]},
        "final_weight_normalization": "per_sample_mean_one",
        "lambda_interaction": 1.0,
        "alpha_ce": 0.5,
        "lambda_full": 1.0,
        "kd_temperature": 2.0,
        "seeds": [13, 42, 2026],
        "train_parents": counts["train"],
        "valid_parents": counts["valid"],
        "windows": len(rows),
        "official_test_evaluated": False,
    }
    protocol["input_sha256"] = {str(path.resolve()): sha256(path) for path in source_paths}
    atomic_json(protocol, output / "protocol.json")
    print(json.dumps({"status": "frozen", "dataset": args.dataset, "protocol": str(output / "protocol.json"), "utility": utility.tolist()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mosi", "mosei"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--probes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, required=True)
    parser.add_argument("--expected-valid", type=int, required=True)
    prepare(parser.parse_args())
