#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/wy/sjq/kd/dataset/cmu_mosei")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_smoke_manifest import build_candidates, video_is_decodable  # noqa: E402
from rdid_mosei.mosei import read_labels  # noqa: E402


# Fixed before observing teacher outputs. Total: train=450, valid=50.
QUOTAS: dict[tuple[str, str, str], int] = {
    ("train", "lt5", "negative"): 37,
    ("train", "lt5", "neutral"): 38,
    ("train", "lt5", "positive"): 37,
    ("train", "5to15", "negative"): 75,
    ("train", "5to15", "neutral"): 75,
    ("train", "5to15", "positive"): 75,
    ("train", "15to30", "negative"): 33,
    ("train", "15to30", "neutral"): 33,
    ("train", "15to30", "positive"): 33,
    ("train", "gt30", "negative"): 5,
    ("train", "gt30", "neutral"): 5,
    ("train", "gt30", "positive"): 4,
    ("valid", "lt5", "negative"): 4,
    ("valid", "lt5", "neutral"): 4,
    ("valid", "lt5", "positive"): 4,
    ("valid", "5to15", "negative"): 9,
    ("valid", "5to15", "neutral"): 8,
    ("valid", "5to15", "positive"): 8,
    ("valid", "15to30", "negative"): 4,
    ("valid", "15to30", "neutral"): 3,
    ("valid", "15to30", "positive"): 4,
    ("valid", "gt30", "neutral"): 1,
    ("valid", "gt30", "positive"): 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen 500-sample teacher benchmark")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/wy/sjq/kd/dataset/cmu_mosei_source"))
    parser.add_argument("--output", type=Path, default=DATASET_ROOT / "manifests/benchmark500.jsonl")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "reports/benchmark500_manifest.json")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def assert_official_video_splits_are_disjoint(dataset_root: Path) -> dict[str, int]:
    split_ids: dict[str, set[str]] = defaultdict(set)
    for row in read_labels(dataset_root / "label.csv"):
        split_ids[row["split"]].add(row["video_id"])
    split_names = sorted(split_ids)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise RuntimeError(f"official split leakage between {left}/{right}: {sorted(overlap)[:5]}")
    return {name: len(values) for name, values in sorted(split_ids.items())}


def select_samples(candidates: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for item in candidates:
        key = (str(item["split"]), str(item["duration_bucket"]), str(item["sentiment_bucket"]))
        groups[key].append(item)
    for values in groups.values():
        rng.shuffle(values)

    # Fill the scarcest strata first so unique-video selection cannot starve them.
    ordered_quotas = sorted(
        QUOTAS.items(),
        key=lambda pair: (len({str(x["video_id"]) for x in groups[pair[0]]}) / pair[1], pair[0]),
    )
    selected: list[dict[str, object]] = []
    used_video_ids: set[str] = set()
    used_sample_ids: set[str] = set()
    decode_failures: list[str] = []
    media_duration_cache: dict[str, tuple[float, float]] = {}

    def full_media_covers_interval(candidate: dict[str, object], tolerance: float = 0.05) -> bool:
        video_id = str(candidate["video_id"])
        if video_id not in media_duration_cache:
            capture = cv2.VideoCapture(str(candidate["full_video_path"]))
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                video_duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
            finally:
                capture.release()
            audio = sf.info(str(candidate["full_audio_path"]))
            audio_duration = audio.frames / audio.samplerate
            media_duration_cache[video_id] = (video_duration, audio_duration)
        video_duration, audio_duration = media_duration_cache[video_id]
        end = float(candidate["end"])
        return video_duration + tolerance >= end and audio_duration + tolerance >= end

    for key, quota in ordered_quotas:
        accepted = 0
        while groups[key] and accepted < quota:
            candidate = groups[key].pop()
            sample_id = str(candidate["sample_id"])
            video_id = str(candidate["video_id"])
            if sample_id in used_sample_ids or video_id in used_video_ids:
                continue
            if not video_is_decodable(Path(str(candidate["segmented_video_path"]))):
                decode_failures.append(sample_id)
                continue
            if not full_media_covers_interval(candidate):
                continue
            item = dict(candidate)
            item["manifest_version"] = "benchmark500-v1"
            item["selection_seed"] = seed
            selected.append(item)
            used_sample_ids.add(sample_id)
            used_video_ids.add(video_id)
            accepted += 1
        if accepted != quota:
            raise RuntimeError(f"quota not met for {key}: selected={accepted}, required={quota}")

    rng.shuffle(selected)
    if len(selected) != 500 or len(used_video_ids) != 500:
        raise RuntimeError("benchmark must contain 500 samples from 500 distinct source videos")
    return selected


def main() -> int:
    args = parse_args()
    official_video_counts = assert_official_video_splits_are_disjoint(args.dataset_root)
    candidates = build_candidates(args.dataset_root)
    selected = select_samples(candidates, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in selected)
    args.output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    checksum_path = args.output.with_suffix(args.output.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")

    actual = Counter(
        (str(x["split"]), str(x["duration_bucket"]), str(x["sentiment_bucket"])) for x in selected
    )
    summary = {
        "manifest_version": "benchmark500-v1",
        "seed": args.seed,
        "sha256": digest,
        "eligible_candidates": len(candidates),
        "selected_samples": len(selected),
        "unique_video_ids": len({str(x["video_id"]) for x in selected}),
        "official_video_counts": official_video_counts,
        "split": dict(Counter(str(x["split"]) for x in selected)),
        "duration": dict(Counter(str(x["duration_bucket"]) for x in selected)),
        "sentiment": dict(Counter(str(x["sentiment_bucket"]) for x in selected)),
        "strata": {"|".join(key): actual[key] for key in sorted(actual)},
        "duration_seconds": {
            "total": sum(float(x["duration"]) for x in selected),
            "min": min(float(x["duration"]) for x in selected),
            "max": max(float(x["duration"]) for x in selected),
        },
        "output": str(args.output.resolve()),
        "checksum_file": str(checksum_path.resolve()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
