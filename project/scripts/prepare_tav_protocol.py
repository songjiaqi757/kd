#!/usr/bin/env python3
"""Freeze corrected teacher assets and train-only seven-coordinate statistics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from prepare_cv2_assets import mobius_numpy
from train_video_adaptation_v2 import ROOT, atomic_json, sha256

SUBSETS = ("t", "a", "v", "ta", "tv", "av", "tav")
SEEDS = (2026, 2027, 2028)


def utility_from_train(interactions, labels):
    """Historical RU utility proxy, estimated once using train utterances only."""
    centered = interactions - interactions.mean(0)
    y = labels - labels.mean()
    denom = np.sqrt((centered ** 2).sum(0) * (y ** 2).sum())
    corr = np.divide(np.abs((centered * y[:, None]).sum(0)), denom,
                     out=np.zeros(interactions.shape[-1]), where=denom > 0)
    # A predeclared floor avoids numerically eliminating a coordinate entirely.
    utility = np.maximum(corr, 0.01)
    return corr, utility / utility.mean()


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise FileExistsError("protocol already frozen; use a new output directory")
    manifest = ROOT / "dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl"
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate window ID")
    parents = {}
    for row in rows:
        identity = {k: row[k] for k in ("parent_sample_id", "split", "video_id", "sentiment", "class_7_index")}
        if parents.setdefault(row["parent_sample_id"], identity) != identity:
            raise ValueError("conflicting parent identity")
    if Counter(r["split"] for r in parents.values()) != {"train": 16326, "valid": 1871}:
        raise ValueError("official train/valid parent counts differ")
    videos = {s: {r["video_id"] for r in parents.values() if r["split"] == s} for s in ("train", "valid")}
    if videos["train"] & videos["valid"]:
        raise ValueError("train/valid source overlap")
    keys = sorted(parents)
    train_mask = np.array([parents[k]["split"] == "train" for k in keys])
    labels = np.array([parents[k]["sentiment"] for k in keys])
    paths = [manifest, Path(__file__), ROOT / "project/scripts/prepare_cv2_assets.py"]
    features = ROOT / "outputs/experiments/video_source_v2/teacher_features"
    feature_config = json.loads((features / "run_config.json").read_text())
    if feature_config["video_timing_policy"] != "sampled_fps_v2" or feature_config["manifest_sha256"] != sha256(manifest):
        raise ValueError("teacher timing/manifest mismatch")
    complete = np.load(features / "completed.npy", mmap_mode="r")
    hidden = np.load(features / "features.npy", mmap_mode="r")
    if complete.shape != (len(rows) * 7,) or not complete.all() or hidden.shape[0] != complete.size:
        raise ValueError("incomplete teacher features")
    if not np.isfinite(hidden).all():
        raise ValueError("nonfinite teacher features")
    paths += [features / name for name in ("run_config.json", "index.jsonl", "features.npy", "completed.npy", "video_preprocessing.jsonl")]
    arrays, baselines, temperatures = [], [], {}
    for seed in SEEDS:
        directory = ROOT / f"outputs/experiments/video_source_v2/probes/seed{seed}"
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text())
        temperature = float(report["calibration"]["after"]["temperature"])
        if (report["seed"] != seed or Path(report["features"]).resolve() != features.resolve()
                or report["video_timing_policy"] != "sampled_fps_v2" or report["official_test_evaluated"]
                or not np.isfinite(temperature) or temperature <= 0 or abs(temperature - 0.9259549975) < 1e-6):
            raise ValueError(f"invalid corrected Probe: {seed}")
        temperatures[str(seed)] = temperature
        grouped = {}
        for line in (directory / "predictions.jsonl").open():
            r = json.loads(line)
            key, subset = r["parent_sample_id"], r["subset"]
            if key not in parents or subset not in SUBSETS or (key, subset) in grouped:
                raise ValueError("unexpected/duplicate teacher target")
            identity = parents[key]
            if (r["split"] != identity["split"] or abs(r["target_sentiment"] - identity["sentiment"]) > 1e-6
                    or r["class_7_index"] != identity["class_7_index"]):
                raise ValueError("teacher/manifest target identity mismatch")
            if len(r["classification_logits"]) != 7 or not np.isfinite([r["probe_score"], *r["classification_logits"]]).all():
                raise ValueError("nonfinite teacher values")
            grouped[key, subset] = r["probe_score"]
        raw = np.array([[grouped[k, s] for s in SUBSETS] for k in keys])
        baseline = float(raw[train_mask, -1].mean())
        arrays.append(mobius_numpy(raw, baseline))
        baselines.append(baseline)
        paths += [report_path, directory / "predictions.jsonl"]
    stack = np.stack(arrays)
    mean, variance = stack.mean(0), stack.var(0, ddof=1)
    corr, utility = utility_from_train(mean[train_mask], labels[train_mask])
    targets = output / "interaction_targets.jsonl"
    with targets.open("w") as f:
        for i, key in enumerate(keys):
            f.write(json.dumps({**parents[key], "mean": mean[i].tolist(), "variance": variance[i].tolist()}) + "\n")
    paths.append(targets)
    # Online inputs deliberately have no student feature cache. Hash every
    # actual train/valid audio/video file once, rather than claiming old cache identity.
    media = output / "student_input_manifest.jsonl"
    media_paths = sorted({r[k] for r in rows for k in ("audio_segment_path", "silent_video_path")})
    with media.open("w") as f:
        for i, name in enumerate(media_paths):
            path = Path(name)
            stat = path.stat()
            f.write(json.dumps({"path": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                "sha256": sha256(path)}) + "\n")
            if i % 2000 == 0:
                print(json.dumps({"stage": "media_hashes", "completed": i, "total": len(media_paths)}), flush=True)
    paths.append(media)
    protocol = {
        "schema": "tav-m0-m6-v1", "teacher_probe_seed": 2026,
        "teacher_temperature": temperatures["2026"], "probe_temperatures": temperatures,
        "teacher_feature_manifest": str(features / "index.jsonl"),
        "teacher_feature_checksum": sha256(features / "features.npy"),
        "student_cache_manifest": None, "student_cache_checksum": None,
        "student_input_policy": "online_raw_media_identical_across_M0_M6_no_feature_cache",
        "student_input_manifest": str(media), "student_input_checksum": sha256(media),
        "manifest": str(manifest), "teacher_targets": str(ROOT / "outputs/experiments/video_source_v2/probes/seed2026/predictions.jsonl"),
        "teacher_probe_report": str(ROOT / "outputs/experiments/video_source_v2/probes/seed2026/report.json"),
        "interaction_targets": str(targets), "interaction_subsets": list(SUBSETS),
        "interaction_target_policy": "same_3_probe_mean_for_M4_M5_M6",
        "probe_seeds": list(SEEDS), "per_probe_empty_baseline": baselines,
        "student_empty_baseline": float(np.mean(baselines)),
        "empty_baseline_definition": "per_probe_official_train_TAV_utterance_mean",
        "utility_definition": "train_only_absolute_Pearson_of_ensemble_interaction_and_label_floor_0.01_mean_normalized",
        "utility_is_task_association_proxy_not_causal_utility": True,
        "utility_raw_abs_correlation": corr.tolist(), "utility_normalized": utility.tolist(),
        "reliability": {"epsilon_inside_sqrt": 1e-4, "clip_after_sample_mean_normalization": [0.25, 4.0]},
        "final_weight_normalization": "per_sample_mean_one_across_seven_coordinates",
        "lambda_interaction": 1.0, "alpha_ce": 0.5, "lambda_full": 1.0, "kd_temperature": 2.0,
        "seeds": [13, 42, 2026], "train_parents": 16326, "valid_parents": 1871,
        "windows": len(rows), "official_test_evaluated": False,
        "input_sha256": {str(p.resolve()): sha256(p) for p in paths},
    }
    atomic_json(protocol, output / "protocol.json")
    print(json.dumps({"status": "frozen", "protocol": str(output / "protocol.json"),
                      "temperature": protocol["teacher_temperature"], "utility": utility.tolist()}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/tav_main_v1/assets")
    prepare(p.parse_args().output.resolve())
