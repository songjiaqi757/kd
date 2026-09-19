#!/usr/bin/env python3
"""Apply frozen train/valid Probes to MOSEI test features for RDID-v2 analysis.

This script performs inference only. It does not fit or calibrate a Probe on
test labels, and it reuses each Probe's train-derived empty baseline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]

from prepare_cv2_assets import mobius_numpy
from rdid_mosei.probe import TeacherProbe
from rdid_mosei.student import SUBSETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Seven-subset official-test feature directory")
    parser.add_argument(
        "--probes-root", type=Path,
        default=ROOT / "outputs/experiments/video_source_v2/probes",
    )
    parser.add_argument(
        "--base-protocol", type=Path,
        default=ROOT / "outputs/experiments/tav_main_v1/assets/protocol.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/experiments/rdid_v2_mosei/diagnostics/test_teacher_interactions",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-mosei-test-development", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_mosei_test_development:
        parser.error("explicit --allow-mosei-test-development is required")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def validate_feature_rows(features: Path) -> tuple[list[dict], np.ndarray]:
    config = json.loads((features / "run_config.json").read_text())
    if config.get("video_timing_policy") != "sampled_fps_v2":
        raise ValueError("timing-v2 teacher features are required")
    rows = [json.loads(line) for line in (features / "index.jsonl").read_text().splitlines() if line]
    if not rows or {row["split"] for row in rows} != {"test"}:
        raise ValueError("feature index must contain only MOSEI official-test rows")
    if {row["subset"] for row in rows} != set(SUBSETS):
        raise ValueError("all seven subsets are required")
    keys = [(row["sample_id"], row["subset"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate feature sample/subset row")
    samples = {row["sample_id"] for row in rows}
    if set(keys) != {(sample_id, subset) for sample_id in samples for subset in SUBSETS}:
        raise ValueError("every official-test window must cover all seven subsets")
    completed = np.load(features / "completed.npy", mmap_mode="r")
    values = np.load(features / "features.npy", mmap_mode="r")
    if len(rows) != len(completed) or values.shape[0] != len(rows) or not completed.all():
        raise ValueError("official-test feature cache is incomplete")
    if not np.isfinite(values).all():
        raise ValueError("official-test feature cache contains non-finite values")
    return rows, values


@torch.inference_mode()
def predict_probe(checkpoint_path: Path, features: np.ndarray, device: torch.device, batch_size: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TeacherProbe(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        classes=int(checkpoint["classes"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scores, logits = [], []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(np.asarray(features[start:start + batch_size], dtype=np.float32)).to(device)
        output = model(batch)
        scores.append(output["regression"].float().cpu().numpy())
        logits.append(output["classification_logits"].float().cpu().numpy())
    return np.concatenate(scores), np.concatenate(logits), checkpoint


def aggregate(rows: list[dict], scores: np.ndarray, logits: np.ndarray) -> list[dict]:
    groups = defaultdict(list)
    for row, score, class_logits in zip(rows, scores, logits):
        groups[row["parent_sample_id"], row["subset"]].append((row, float(score), class_logits))
    result = []
    for (parent, subset), items in sorted(groups.items()):
        denominator = sum(float(item[0]["aggregation_weight"]) for item in items)
        if denominator <= 0:
            raise ValueError("non-positive aggregation weight")
        first = items[0][0]
        score = sum(item[1] * float(item[0]["aggregation_weight"]) for item in items) / denominator
        class_logits = sum(item[2] * float(item[0]["aggregation_weight"]) for item in items) / denominator
        result.append({
            "parent_sample_id": parent,
            "video_id": first["video_id"],
            "split": "test",
            "subset": subset,
            "target_sentiment": float(first["target_sentiment"]),
            "class_7_index": int(first["class_7_index"]),
            "probe_score": score,
            "classification_logits": class_logits.tolist(),
            "window_count": len(items),
        })
    return result


def write_jsonl(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    features = args.features.resolve()
    probes_root = args.probes_root.resolve()
    base_protocol_path = args.base_protocol.resolve()
    output = args.output.resolve()
    protocol = json.loads(base_protocol_path.read_text())
    seeds = [int(seed) for seed in protocol["probe_seeds"]]
    baselines = [float(value) for value in protocol["per_probe_empty_baseline"]]
    if len(seeds) != 3 or len(baselines) != len(seeds):
        raise ValueError("RDID-v2 requires the frozen three-Probe ensemble")
    rows, feature_values = validate_feature_rows(features)
    plan = {
        "schema": "rdid-v2-mosei-test-teacher-interaction-plan-v1",
        "features": str(features),
        "probes": [str(probes_root / f"seed{seed}" / "teacher_probe.pt") for seed in seeds],
        "probe_seeds": seeds,
        "train_derived_empty_baselines": baselines,
        "test_rows": len(rows),
        "test_role": "development_information_not_blind_confirmation",
        "fits_or_calibrates_on_test": False,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(plan, output / "run_config.json")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    probe_arrays = []
    identities = None
    probe_checksums = {}
    for seed in seeds:
        checkpoint_path = probes_root / f"seed{seed}" / "teacher_probe.pt"
        report_path = probes_root / f"seed{seed}" / "report.json"
        report = json.loads(report_path.read_text())
        if int(report["seed"]) != seed or report.get("official_test_evaluated") is not False:
            raise ValueError(f"Probe {seed} was not frozen from train/valid only")
        scores, logits, checkpoint = predict_probe(checkpoint_path, feature_values, device, args.batch_size)
        if int(checkpoint["seed"]) != seed:
            raise ValueError(f"Probe checkpoint seed mismatch: {seed}")
        predictions = aggregate(rows, scores, logits)
        write_jsonl(predictions, output / f"probe_seed{seed}_predictions.jsonl")
        grouped = {(row["parent_sample_id"], row["subset"]): row for row in predictions}
        parents = sorted({row["parent_sample_id"] for row in predictions})
        expected = {(parent, subset) for parent in parents for subset in SUBSETS}
        if set(grouped) != expected:
            raise ValueError(f"Probe {seed} test coverage differs")
        for parent in parents:
            signatures = {
                (
                    grouped[parent, subset]["video_id"],
                    grouped[parent, subset]["target_sentiment"],
                    grouped[parent, subset]["class_7_index"],
                )
                for subset in SUBSETS
            }
            if len(signatures) != 1:
                raise ValueError(f"Probe {seed} subset identity mismatch: {parent}")
        current_identities = {
            parent: {
                "parent_sample_id": parent,
                "video_id": grouped[parent, "tav"]["video_id"],
                "split": "test",
                "sentiment": grouped[parent, "tav"]["target_sentiment"],
                "class_7_index": grouped[parent, "tav"]["class_7_index"],
            }
            for parent in parents
        }
        if identities is not None and identities != current_identities:
            raise ValueError("Probe identity maps differ")
        identities = current_identities
        probe_arrays.append(np.asarray([
            [grouped[parent, subset]["probe_score"] for subset in SUBSETS]
            for parent in parents
        ]))
        probe_checksums[str(seed)] = {
            "checkpoint_sha256": sha256(checkpoint_path),
            "report_sha256": sha256(report_path),
        }

    interactions = np.stack([
        mobius_numpy(values, baseline) for values, baseline in zip(probe_arrays, baselines)
    ])
    mean = interactions.mean(0)
    variance = interactions.var(0, ddof=1)
    parents = sorted(identities)
    targets = [{
        **identities[parent],
        "mean": mean[index].tolist(),
        "variance": variance[index].tolist(),
    } for index, parent in enumerate(parents)]
    write_jsonl(targets, output / "interaction_targets.jsonl")
    report = {
        "schema": "rdid-v2-mosei-test-teacher-interactions-v1",
        "status": "complete",
        "test_parents": len(parents),
        "coordinate_order": list(SUBSETS),
        "probe_checksums": probe_checksums,
        "interaction_targets_sha256": sha256(output / "interaction_targets.jsonl"),
        "test_role": plan["test_role"],
        "fits_or_calibrates_on_test": False,
    }
    atomic_json(report, output / "report.json")
    atomic_json({"status": "complete"}, output / "status.json")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
