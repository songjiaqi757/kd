#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import av
import soundfile as sf

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a full CMU-MOSEI official-aligned inventory")
    parser.add_argument("--source-root", type=Path, default=Path("/home/wy/sjq/kd/dataset/cmu_mosei_source"))
    parser.add_argument("--output-root", type=Path, default=Path("/home/wy/sjq/kd/dataset/cmu_mosei"))
    parser.add_argument("--video-tolerance", type=float, default=0.55)
    parser.add_argument("--audio-tolerance", type=float, default=0.55)
    return parser.parse_args()


def video_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        if container.duration is not None:
            return float(container.duration / av.time_base)
        if not container.streams.video:
            raise ValueError("no video stream")
        stream = container.streams.video[0]
        if stream.duration is None or stream.time_base is None:
            raise ValueError("video duration unavailable")
        return float(stream.duration * stream.time_base)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def main() -> int:
    args = parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    labels_path = source / "label.csv"
    label_sha256 = sha256_file(labels_path)
    labels = read_labels(labels_path)
    transcript_dir = source / "Transcript/Segmented/Combined"
    video_dir = source / "Videos/Full/Combined"
    audio_dir = source / "Audio/Full/WAV_16000"
    legacy_video_dir = source / "Videos/Segmented/Combined"

    split_by_video: dict[str, str] = {}
    sample_ids: set[str] = set()
    duplicate_sample_ids: list[str] = []
    split_leakage: list[dict[str, str]] = []
    for row in labels:
        sample_id = row["segment_id"]
        if sample_id in sample_ids:
            duplicate_sample_ids.append(sample_id)
        sample_ids.add(sample_id)
        previous = split_by_video.setdefault(row["video_id"], row["split"])
        if previous != row["split"]:
            split_leakage.append(
                {"video_id": row["video_id"], "left": previous, "right": row["split"]}
            )
    if duplicate_sample_ids or split_leakage:
        raise RuntimeError(
            f"invalid official labels: duplicate_samples={len(duplicate_sample_ids)} "
            f"split_leakage={len(split_leakage)}"
        )

    media_cache: dict[str, dict[str, Any]] = {}
    transcript_cache: dict[str, list[Any]] = {}
    inventory: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for index, row in enumerate(labels, start=1):
        video_id = row["video_id"]
        full_video = video_dir / f"{video_id}.mp4"
        full_audio = audio_dir / f"{video_id}.wav"
        transcript = transcript_dir / f"{video_id}.txt"
        if video_id not in media_cache:
            media: dict[str, Any] = {
                "full_video_exists": full_video.is_file(),
                "full_audio_exists": full_audio.is_file(),
                "transcript_exists": transcript.is_file(),
            }
            try:
                media["full_video_duration"] = video_duration(full_video)
            except Exception as exc:  # noqa: BLE001 - inventory records every source failure
                media["video_probe_error"] = repr(exc)
            try:
                info = sf.info(str(full_audio))
                media["full_audio_duration"] = info.frames / info.samplerate
                media["full_audio_samplerate"] = info.samplerate
                media["full_audio_channels"] = info.channels
            except Exception as exc:  # noqa: BLE001
                media["audio_probe_error"] = repr(exc)
            media_cache[video_id] = media
            if transcript.is_file():
                transcript_cache[video_id] = read_transcript(transcript)
        media = media_cache[video_id]

        reasons: list[str] = []
        start = float(row["start"])
        end = float(row["end"])
        source_crop_start = max(0.0, start)
        if end <= start:
            reasons.append("invalid_label_interval")
        if not media["full_video_exists"]:
            reasons.append("missing_full_video")
        elif "full_video_duration" not in media:
            reasons.append("video_probe_failed")
        elif end > float(media["full_video_duration"]) + args.video_tolerance:
            reasons.append("video_does_not_cover_interval")
        if not media["full_audio_exists"]:
            reasons.append("missing_full_audio")
        elif "full_audio_duration" not in media:
            reasons.append("audio_probe_failed")
        elif end > float(media["full_audio_duration"]) + args.audio_tolerance:
            reasons.append("audio_does_not_cover_interval")
        if not media["transcript_exists"]:
            reasons.append("missing_segment_transcript")

        segment = None
        if media["transcript_exists"]:
            try:
                segment = match_transcript_segment(transcript_cache[video_id], start, end)
            except Exception:
                reasons.append("transcript_timestamp_mismatch")

        sentiment = float(row["sentiment"])
        clip_index = int(row["clip_index"])
        target_stem = f"{video_id}__{clip_index}"
        legacy_path = (
            legacy_video_dir / f"{video_id}_{segment.index}.mp4" if segment is not None else None
        )
        item: dict[str, Any] = {
            "manifest_version": "mosei-official-aligned-v1",
            "source_label_sha256": label_sha256,
            "sample_id": row["segment_id"],
            "video_id": video_id,
            "clip_index": clip_index,
            "transcript_segment_index": segment.index if segment is not None else None,
            "split": row["split"],
            "official_start": start,
            "official_end": end,
            "official_duration": end - start,
            "source_crop_start": source_crop_start,
            "duration": end - start,
            "video_leading_padding_seconds": max(0.0, -start),
            "audio_leading_padding_seconds": max(0.0, -start),
            "video_tail_padding_seconds": max(
                0.0, end - float(media.get("full_video_duration", end))
            ),
            "audio_tail_padding_seconds": max(
                0.0, end - float(media.get("full_audio_duration", end))
            ),
            "text": segment.text if segment is not None else None,
            "sentiment": sentiment,
            "class_7_value": round_sentiment_class(sentiment),
            "class_7_index": round_sentiment_class(sentiment) + 3,
            "sentiment_bucket": sentiment_bucket(sentiment),
            "duration_bucket": duration_bucket(end - start),
            "emotions": {
                name: float(row[name])
                for name in ("happy", "sad", "anger", "surprise", "disgust", "fear")
            },
            "full_video_path": str(full_video.resolve()),
            "full_audio_path": str(full_audio.resolve()),
            "source_video_duration": media.get("full_video_duration"),
            "source_audio_duration": media.get("full_audio_duration"),
            "legacy_segmented_video_path": str(legacy_path.resolve()) if legacy_path else None,
            "legacy_segmented_video_exists": bool(legacy_path and legacy_path.is_file()),
            "silent_video_path": str((output / "media/video_silent" / f"{target_stem}.mp4")),
            "audio_segment_path": str((output / "media/audio_16k_mono" / f"{target_stem}.wav")),
            "source_policy": "crop_full_media_by_official_label_timestamps",
            "status": "valid" if not reasons else "invalid",
            "failure_reasons": reasons,
        }
        inventory.append(item)
        if reasons:
            reason_counts.update(reasons)
        else:
            valid.append(item)
        if index % 500 == 0 or index == len(labels):
            print(f"inventory_progress={index}/{len(labels)}", flush=True)

    manifests = output / "manifests"
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    valid_sha = write_jsonl(manifests / "all.jsonl", valid)
    if len(valid) == len(inventory):
        inventory_sha = valid_sha
        (manifests / "all_inventory.jsonl").unlink(missing_ok=True)
        (manifests / "all_inventory.jsonl.sha256").unlink(missing_ok=True)
    else:
        inventory_sha = write_jsonl(manifests / "all_inventory.jsonl", inventory)
    split_shas = {}
    for split in ("train", "valid", "test"):
        split_rows = [row for row in valid if row["split"] == split]
        split_shas[split] = write_jsonl(manifests / f"{split}.jsonl", split_rows)

    summary = {
        "manifest_version": "mosei-official-aligned-v1",
        "source_root": str(source),
        "output_root": str(output),
        "source_label": str(labels_path),
        "source_label_sha256": label_sha256,
        "official_rows": len(labels),
        "official_videos": len(split_by_video),
        "official_split_rows": dict(Counter(row["split"] for row in labels)),
        "official_split_videos": dict(Counter(split_by_video.values())),
        "valid_rows": len(valid),
        "invalid_rows": len(inventory) - len(valid),
        "valid_split_rows": dict(Counter(row["split"] for row in valid)),
        "invalid_reasons": dict(reason_counts),
        "legacy_segmented_video_present": sum(
            bool(row["legacy_segmented_video_exists"]) for row in inventory
        ),
        "leading_padding_samples": sum(
            float(row["video_leading_padding_seconds"]) > 0 for row in inventory
        ),
        "tail_padding_samples": sum(
            float(row["video_tail_padding_seconds"]) > 0
            or float(row["audio_tail_padding_seconds"]) > 0
            for row in inventory
        ),
        "valid_duration_seconds": sum(float(row["duration"]) for row in valid),
        "inventory_sha256": inventory_sha,
        "valid_manifest_sha256": valid_sha,
        "split_manifest_sha256": split_shas,
    }
    (reports / "source_inventory.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
