#!/usr/bin/env python3
"""Freeze ensemble/subset/feature targets for the fixed-student experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUBSETS = ("t", "a", "v", "ta", "tv", "av", "tav")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def prediction_paths(base: dict) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for seed in base["probe_seeds"]:
        suffix = f"/seed{seed}/predictions.jsonl"
        matches = [Path(name) for name in base["input_sha256"] if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected one frozen Probe prediction path for seed {seed}")
        paths[int(seed)] = matches[0]
    return paths


def load_probe(path: Path, expected_temperature: float) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for line in path.open():
        if not line.strip():
            continue
        item = json.loads(line)
        key = item["parent_sample_id"], item["subset"]
        if key in rows or item["subset"] not in SUBSETS:
            raise ValueError(f"duplicate/unknown Probe target: {key}")
        logits = np.asarray(item["classification_logits"], dtype=np.float64)
        if logits.shape != (7,) or not np.isfinite(logits).all() or not math.isfinite(item["probe_score"]):
            raise ValueError(f"invalid Probe target: {key}")
        scaled = logits / expected_temperature
        probabilities = np.exp(scaled - scaled.max())
        probabilities /= probabilities.sum()
        rows[key] = {**item, "calibrated_probabilities": probabilities.tolist()}
    return rows


def prepare(base_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"use a new asset directory: {output}")
    output.mkdir(parents=True)
    base = json.loads(base_path.read_text())
    if base.get("official_test_evaluated"):
        raise ValueError("asset preparation must not consume official test")
    manifest = Path(base["manifest"])
    windows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if {row["split"] for row in windows} != {"train", "valid"}:
        raise ValueError("main-table tuning assets require exactly train and valid")
    parent_rows: dict[str, dict] = {}
    for row in windows:
        identity = {key: row[key] for key in ("parent_sample_id", "split", "video_id", "sentiment", "class_7_index")}
        if parent_rows.setdefault(row["parent_sample_id"], identity) != identity:
            raise ValueError("conflicting window identity")

    paths = prediction_paths(base)
    probes = {
        seed: load_probe(path, float(base["probe_temperatures"][str(seed)]))
        for seed, path in paths.items()
    }
    expected = {(parent, subset) for parent in parent_rows for subset in SUBSETS}
    if any(set(probe) != expected for probe in probes.values()):
        raise ValueError("Probe coverage differs from manifest x seven subsets")

    ensemble_rows = []
    for parent in sorted(parent_rows):
        scores, probabilities = [], []
        for subset in SUBSETS:
            items = [probes[seed][parent, subset] for seed in sorted(probes)]
            scores.append(float(np.mean([item["probe_score"] for item in items])))
            probabilities.append(np.mean([item["calibrated_probabilities"] for item in items], axis=0).tolist())
        ensemble_rows.append(
            {
                **parent_rows[parent],
                "subsets": list(SUBSETS),
                "subset_score_mean": scores,
                "tav_calibrated_probability_mean": probabilities[-1],
                "probe_seeds": sorted(probes),
            }
        )
    ensemble_path = output / "ensemble_targets.jsonl"
    write_jsonl(ensemble_path, ensemble_rows)

    feature_manifest = Path(base["teacher_feature_manifest"])
    feature_rows = [json.loads(line) for line in feature_manifest.read_text().splitlines() if line.strip()]
    tav_features = {
        row["sample_id"]: row["job_index"] for row in feature_rows if row["subset"] == "tav"
    }
    if set(tav_features) != {row["sample_id"] for row in windows}:
        raise ValueError("TAV teacher-feature coverage differs from window manifest")
    feature_index_path = output / "teacher_feature_index.jsonl"
    write_jsonl(
        feature_index_path,
        [{"sample_id": row["sample_id"], "job_index": tav_features[row["sample_id"]]} for row in windows],
    )

    utility = np.asarray(base["utility_normalized"], dtype=np.float64)
    contains_video = np.asarray(["v" in subset for subset in SUBSETS])
    coarse_utility = np.where(contains_video, utility[contains_video].mean(), utility[~contains_video].mean())
    coarse_utility /= coarse_utility.mean()
    source_metadata = ROOT / "external/SOURCES.json"
    protocol = {
        "schema": "rdid-msa-main-table-assets-v1",
        "base_protocol": str(base_path),
        "manifest": str(manifest),
        "teacher_targets": base["teacher_targets"],
        "teacher_probe_report": base["teacher_probe_report"],
        "teacher_temperature": base["teacher_temperature"],
        "probe_seeds": base["probe_seeds"],
        "probe_temperatures": base["probe_temperatures"],
        "ensemble_targets": str(ensemble_path),
        "interaction_targets": base["interaction_targets"],
        "teacher_features": str(feature_manifest.parent / "features.npy"),
        "teacher_feature_index": str(feature_index_path),
        "teacher_feature_dimension": 2048,
        "student_empty_baseline": base["student_empty_baseline"],
        "utility_normalized": base["utility_normalized"],
        "coarse_utility_normalized": coarse_utility.tolist(),
        "lambda_interaction": 1.0,
        "lambda_subset": 1.0,
        "lambda_feature": 2.0,
        "alpha_ce": 0.5,
        "lambda_full": 1.0,
        "kd_temperature": 2.0,
        "rld": {"alpha": 1.0, "beta": 8.0, "temperature": 4.0, "confidence_temperature": 1.0, "warmup_epochs": 20, "standardize_logits": False},
        "skd": {"temperature": 4.0, "tik_factor": 0.1, "mask_policy": "per_sample_low_confidence_kl_and_unmasked_direction_loss"},
        "ea_kd": {"entropy_temperature": 3.0, "entropy_upper_bound": "log(7)"},
        "projector": {"student_dimension": 512, "teacher_dimension": 2048, "normalization": "BatchNorm1d(affine=False)", "distance": "log(sum(abs(delta)^4)+1e-5)"},
        "cmad_cafd": {"tau": 0.2, "lambda": 0.1, "minimum_relation_batch": 2, "adaptation": "student fused 512->2048 linear projection; official CAFD weighted MSE and cross-sample correlation terms; a final batch of one has zero CAFD gradient while common task/full-KD terms remain active"},
        "seeds": [13, 42, 2026],
        "official_test_evaluated": False,
        "upstream_sources": str(source_metadata),
    }
    inputs = [
        base_path,
        manifest,
        Path(base["teacher_targets"]),
        Path(base["teacher_probe_report"]),
        Path(base["interaction_targets"]),
        ensemble_path,
        feature_manifest,
        feature_manifest.parent / "features.npy",
        feature_index_path,
        source_metadata,
        *paths.values(),
    ]
    protocol["input_sha256"] = {str(path.resolve()): sha256(path) for path in inputs}
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "frozen", "output": str(output), "parents": len(parent_rows), "windows": len(windows)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "outputs/experiments/tav_main_v1/assets/protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/main_table_v1/mosei/assets")
    arguments = parser.parse_args()
    prepare(arguments.base.resolve(), arguments.output.resolve())
