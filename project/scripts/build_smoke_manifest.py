#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.mosei import (  # noqa: E402
    duration_bucket,
    match_transcript_segment,
    read_labels,
    read_transcript,
    round_sentiment_class,
    sentiment_bucket,
)


DURATION_SLOTS = ["lt5"] * 5 + ["5to15"] * 9 + ["15to30"] * 5 + ["gt30"]
SENTIMENT_SLOTS = ["negative", "neutral", "positive"] * 7
SPLIT_SLOTS = ["valid" if index in {3, 8, 13, 18} else "train" for index in range(20)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/wy/sjq/kd/dataset/cmu_mosei_source"))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/smoke20.jsonl",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def video_is_decodable(path: Path) -> bool:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None and frame.size)
    finally:
        capture.release()


def build_candidates(root: Path) -> list[dict[str, object]]:
    transcript_dir = root / "Transcript/Segmented/Combined"
    segmented_video_dir = root / "Videos/Segmented/Combined"
    full_video_dir = root / "Videos/Full/Combined"
    audio_dir = root / "Audio/Full/WAV_16000"
    transcript_cache: dict[str, list] = {}
    candidates: list[dict[str, object]] = []

    for row in read_labels(root / "label.csv"):
        if row["split"] not in {"train", "valid"}:
            continue
        video_id = row["video_id"]
        if video_id not in transcript_cache:
            transcript_cache[video_id] = read_transcript(transcript_dir / f"{video_id}.txt")
        start = float(row["start"])
        end = float(row["end"])
        segment = match_transcript_segment(transcript_cache[video_id], start, end)
        segmented_video = segmented_video_dir / f"{video_id}_{segment.index}.mp4"
        full_video = full_video_dir / f"{video_id}.mp4"
        full_audio = audio_dir / f"{video_id}.wav"
        if not (segmented_video.is_file() and full_video.is_file() and full_audio.is_file()):
            continue
        sentiment = float(row["sentiment"])
        duration = end - start
        candidates.append(
            {
                "sample_id": row["segment_id"],
                "video_id": video_id,
                "label_clip_index": int(row["clip_index"]),
                "transcript_segment_index": segment.index,
                "split": row["split"],
                "start": start,
                "end": end,
                "duration": duration,
                "text": segment.text,
                "sentiment": sentiment,
                "class_7_value": round_sentiment_class(sentiment),
                "class_7_index": round_sentiment_class(sentiment) + 3,
                "sentiment_bucket": sentiment_bucket(sentiment),
                "duration_bucket": duration_bucket(duration),
                "segmented_video_path": str(segmented_video.resolve()),
                "full_video_path": str(full_video.resolve()),
                "full_audio_path": str(full_audio.resolve()),
            }
        )
    return candidates


def select_samples(candidates: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for item in candidates:
        key = (str(item["split"]), str(item["duration_bucket"]), str(item["sentiment_bucket"]))
        groups[key].append(item)
    for values in groups.values():
        rng.shuffle(values)

    selected: list[dict[str, object]] = []
    used: set[str] = set()
    for split, duration, sentiment in zip(SPLIT_SLOTS, DURATION_SLOTS, SENTIMENT_SLOTS):
        fallback_keys = [
            (split, duration, sentiment),
            (split, duration, "negative"),
            (split, duration, "neutral"),
            (split, duration, "positive"),
        ]
        choice = None
        for key in fallback_keys:
            while groups[key]:
                candidate = groups[key].pop()
                if str(candidate["sample_id"]) in used:
                    continue
                if video_is_decodable(Path(str(candidate["segmented_video_path"]))):
                    choice = candidate
                    break
            if choice is not None:
                break
        if choice is None:
            raise RuntimeError(f"no candidate for split={split}, duration={duration}")
        used.add(str(choice["sample_id"]))
        selected.append(choice)
    return selected


def main() -> int:
    args = parse_args()
    candidates = build_candidates(args.dataset_root)
    selected = select_samples(candidates, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"eligible={len(candidates)} selected={len(selected)} output={args.output}")
    print("split", dict(Counter(str(x["split"]) for x in selected)))
    print("duration", dict(Counter(str(x["duration_bucket"]) for x in selected)))
    print("sentiment", dict(Counter(str(x["sentiment_bucket"]) for x in selected)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
