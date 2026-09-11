#!/usr/bin/env python3
"""Build immutable C-v2 teacher targets, utility features, and cache identity sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl"
STUDENT_CACHE = ROOT / "outputs/student/features/official_train_valid"
TEACHER_CACHE = ROOT / "outputs/probe/features/official_train_valid"
ASSET_ROOT = ROOT / "outputs/audits/cv2_assets"
PROBE_SEEDS = (2026, 2027, 2028)
SUBSETS = ("t", "a", "v", "ta", "tv", "av", "tav")
PAIR_NAMES = ("ta", "tv", "av")
MODEL_DIRS = {
    "text": ROOT / "model/Qwen3-0.6B-Base",
    "audio": ROOT / "model/WavLM-Base-Plus",
    "video": ROOT / "model/VideoMAE-Base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("quick", "identity", "all"), default="all")
    return parser.parse_args()


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def prediction_path(seed: int) -> Path:
    return ROOT / f"outputs/probe/official_train_valid_seed{seed}/predictions.jsonl"


def shifted_prediction_path(seed: int) -> Path:
    return ROOT / f"outputs/probe/cv2_emptymean_seed{seed}/predictions.jsonl"


def train_empty_baseline(rows: list[dict]) -> float:
    values = [float(row["probe_score"]) for row in rows if row["split"] == "train" and row["subset"] == "tav"]
    parents = [str(row["parent_sample_id"]) for row in rows if row["split"] == "train" and row["subset"] == "tav"]
    if not values or len(values) != len(set(parents)):
        raise ValueError("official-train TAV predictions must contain one row per utterance")
    if not np.isfinite(values).all():
        raise ValueError("non-finite official-train TAV prediction")
    return float(np.mean(values))


def shift_rows(rows: list[dict], baseline: float) -> list[dict]:
    shifted = []
    for row in rows:
        if row["split"] not in {"train", "valid"}:
            raise ValueError("C-v2 teacher targets may contain only official train/valid")
        original = float(row["probe_score"])
        shifted.append(
            {
                **row,
                "probe_score": original - baseline,
                "probe_score_unshifted": original,
                "cv2_empty_baseline": baseline,
                "cv2_protocol": "subtract_per_probe_official_train_tav_utterance_mean",
            }
        )
    return shifted


def mobius_numpy(values: np.ndarray, empty: float = 0.0) -> np.ndarray:
    if values.shape[-1] != 7:
        raise ValueError("seven subset values required")
    result = np.empty_like(values, dtype=np.float64)
    t, a, v, ta, tv, av, tav = np.moveaxis(values, -1, 0)
    result[..., 0] = t - empty
    result[..., 1] = a - empty
    result[..., 2] = v - empty
    result[..., 3] = ta - t - a + empty
    result[..., 4] = tv - t - v + empty
    result[..., 5] = av - a - v + empty
    result[..., 6] = tav - ta - tv - av + t + a + v - empty
    return result


def build_shifted_targets() -> dict:
    reports = {}
    for seed in PROBE_SEEDS:
        source = prediction_path(seed)
        rows = jsonl(source)
        baseline = train_empty_baseline(rows)
        shifted = shift_rows(rows, baseline)
        output = shifted_prediction_path(seed)
        atomic_jsonl(shifted, output)
        source_report = json.loads((source.parent / "report.json").read_text(encoding="utf-8"))
        source_report["cv2_empty_baseline"] = {
            "value": baseline,
            "definition": "official-train utterance mean of TAV probe_score",
            "implementation": "subtract baseline from every non-empty subset before zero-baseline Mobius transform",
            "source_predictions_sha256": sha256_file(source),
            "shifted_predictions_sha256": sha256_file(output),
        }
        source_report["official_test_evaluated"] = False
        atomic_json(source_report, output.parent / "report.json")
        shifted_train_tav = [float(row["probe_score"]) for row in shifted if row["split"] == "train" and row["subset"] == "tav"]
        if not math.isclose(float(np.mean(shifted_train_tav)), 0.0, abs_tol=1e-7):
            raise RuntimeError("shifted official-train TAV mean is not zero")
        reports[str(seed)] = source_report["cv2_empty_baseline"]
    return reports


def grouped_probe_scores(seed: int) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in jsonl(prediction_path(seed)):
        parent = str(row["parent_sample_id"])
        subset = str(row["subset"])
        if subset in grouped[parent]:
            raise ValueError(f"duplicate teacher target {seed}/{parent}/{subset}")
        grouped[parent][subset] = row
    if any(set(items) != set(SUBSETS) for items in grouped.values()):
        raise ValueError(f"incomplete teacher subset coverage for seed {seed}")
    return grouped


def build_utility_features(baselines: dict[str, dict]) -> dict:
    manifest = jsonl(MANIFEST)
    train_rows: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        if row["split"] == "train":
            train_rows[str(row["parent_sample_id"])].append(row)
    folds = json.loads((ROOT / "project/configs/utility_crossfit_5fold_v1.json").read_text(encoding="utf-8"))
    video_to_fold = folds["video_to_fold"]
    probes = {seed: grouped_probe_scores(seed) for seed in PROBE_SEEDS}
    output_rows = []
    for parent in sorted(train_rows):
        windows = train_rows[parent]
        video_id = str(windows[0]["video_id"])
        raw = np.asarray(
            [[float(probes[seed][parent][subset]["probe_score"]) for subset in SUBSETS] for seed in PROBE_SEEDS],
            dtype=np.float64,
        )
        shifted = raw - np.asarray([baselines[str(seed)]["value"] for seed in PROBE_SEEDS])[:, None]
        interactions = mobius_numpy(shifted, 0.0)
        score_mean, score_var = raw.mean(axis=0), raw.var(axis=0, ddof=1)
        interaction_mean = interactions.mean(axis=0)
        interaction_var = interactions.var(axis=0, ddof=1)
        features = {
            **{f"teacher_score_mean_{name}": float(score_mean[i]) for i, name in enumerate(SUBSETS)},
            **{f"teacher_score_var_{name}": float(score_var[i]) for i, name in enumerate(SUBSETS)},
            **{f"interaction_mean_{name}": float(interaction_mean[i]) for i, name in enumerate(SUBSETS)},
            **{f"interaction_var_{name}": float(interaction_var[i]) for i, name in enumerate(SUBSETS)},
            **{
                f"interaction_snr_{name}": float(abs(interaction_mean[i]) / (math.sqrt(interaction_var[i]) + 1e-4))
                for i, name in enumerate(SUBSETS)
            },
            "single_modality_disagreement": float(np.std(score_mean[:3], ddof=0)),
            "pair_modality_disagreement": float(np.std(score_mean[3:6], ddof=0)),
            "utterance_duration": float(windows[0]["utterance_duration"]),
            "window_count": len(windows),
            "text_characters": len(str(windows[0]["text"])),
            "audio_padding_seconds": float(sum(float(row["audio_leading_padding_seconds"]) + float(row["audio_tail_padding_seconds"]) for row in windows)),
            "video_padding_seconds": float(sum(float(row["video_leading_padding_seconds"]) + float(row["video_tail_padding_seconds"]) for row in windows)),
        }
        if not np.isfinite(list(features.values())).all():
            raise ValueError(f"non-finite utility feature for {parent}")
        output_rows.append(
            {
                "parent_sample_id": parent,
                "video_id": video_id,
                "split": "train",
                "fold": int(video_to_fold[video_id]),
                "features": features,
            }
        )
    output = ROOT / "outputs/utility/crossfit_v1/features_train.jsonl"
    atomic_jsonl(output_rows, output)
    report = {
        "status": "pass",
        "rows": len(output_rows),
        "videos": len({row["video_id"] for row in output_rows}),
        "fold_counts": {str(fold): sum(row["fold"] == fold for row in output_rows) for fold in range(5)},
        "feature_count": len(output_rows[0]["features"]),
        "label_derived_features": False,
        "valid_or_test_rows": 0,
        "output_sha256": sha256_file(output),
        "official_test_evaluated": False,
    }
    atomic_json(report, output.parent / "report.json")
    return report


def existing_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("sidecar_index") != count - 1:
                raise ValueError(f"invalid resumable sidecar row {count - 1}: {path}")
    return count


def build_identity_sidecars() -> dict:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = ASSET_ROOT / "status.json"
    manifest = jsonl(MANIFEST)
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    manifest_hash = sha256_file(MANIFEST)

    student_rows = jsonl(STUDENT_CACHE / "index.jsonl")
    student_final = ASSET_ROOT / "student_index_v2.jsonl"
    student_partial = student_final.with_suffix(".jsonl.partial")
    if not student_final.exists():
        start = existing_count(student_partial)
        with student_partial.open("a", encoding="utf-8") as handle:
            for index in range(start, len(student_rows)):
                cached = student_rows[index]
                raw = manifest_by_id[str(cached["sample_id"])]
                feature_path = Path(cached["feature_path"])
                if not feature_path.is_absolute():
                    feature_path = ROOT / feature_path
                row = {
                    **cached,
                    "sidecar_index": index,
                    "video_id": raw["video_id"],
                    "window_index": raw["window_index"],
                    "window_start": raw["window_start"],
                    "window_end": raw["window_end"],
                    "manifest_sha256": manifest_hash,
                    "feature_sha256": sha256_file(feature_path),
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                if (index + 1) % 100 == 0:
                    handle.flush()
                    atomic_json({"status": "student_sidecar", "completed": index + 1, "total": len(student_rows)}, status_path)
        os.replace(student_partial, student_final)

    teacher_rows = jsonl(TEACHER_CACHE / "index.jsonl")
    features = np.load(TEACHER_CACHE / "features.npy", mmap_mode="r")
    teacher_final = ASSET_ROOT / "teacher_index_v2.jsonl"
    teacher_partial = teacher_final.with_suffix(".jsonl.partial")
    if not teacher_final.exists():
        start = existing_count(teacher_partial)
        with teacher_partial.open("a", encoding="utf-8") as handle:
            for index in range(start, len(teacher_rows)):
                cached = teacher_rows[index]
                if int(cached["job_index"]) != index:
                    raise ValueError(f"teacher job index mismatch at {index}")
                raw = manifest_by_id[str(cached["sample_id"])]
                digest = hashlib.sha256(np.ascontiguousarray(features[index]).tobytes()).hexdigest()
                row = {
                    **cached,
                    "sidecar_index": index,
                    "window_start": raw["window_start"],
                    "window_end": raw["window_end"],
                    "manifest_sha256": manifest_hash,
                    "feature_sha256": digest,
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                if (index + 1) % 1000 == 0:
                    handle.flush()
                    atomic_json({"status": "teacher_sidecar", "completed": index + 1, "total": len(teacher_rows)}, status_path)
        os.replace(teacher_partial, teacher_final)

    model_files = {}
    for name, directory in MODEL_DIRS.items():
        selected = sorted(
            path for path in directory.iterdir()
            if path.is_file() and (path.suffix in {".json", ".safetensors", ".bin"})
        )
        model_files[name] = {str(path.relative_to(ROOT)): sha256_file(path) for path in selected}
    atomic_json(model_files, ASSET_ROOT / "model_fingerprints.json")
    report = {
        "status": "pass",
        "manifest_sha256": manifest_hash,
        "student_rows": len(student_rows),
        "teacher_rows": len(teacher_rows),
        "student_sidecar_sha256": sha256_file(student_final),
        "teacher_sidecar_sha256": sha256_file(teacher_final),
        "model_fingerprints_sha256": sha256_file(ASSET_ROOT / "model_fingerprints.json"),
        "legacy_cache_modified": False,
        "official_test_evaluated": False,
    }
    atomic_json(report, ASSET_ROOT / "report.json")
    atomic_json({"status": "complete", **report}, status_path)
    return report


def main() -> int:
    args = parse_args()
    quick_report = None
    if args.stage in {"quick", "all"}:
        baselines = build_shifted_targets()
        utility = build_utility_features(baselines)
        quick_report = {
            "status": "pass",
            "empty_baselines": baselines,
            "utility_features": utility,
            "ensemble_target_paths": [str(shifted_prediction_path(seed)) for seed in PROBE_SEEDS],
            "official_test_evaluated": False,
        }
        atomic_json(quick_report, ASSET_ROOT / "quick_report.json")
    identity_report = build_identity_sidecars() if args.stage in {"identity", "all"} else None
    print(json.dumps({"quick": quick_report, "identity": identity_report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
