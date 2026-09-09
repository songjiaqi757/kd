#!/usr/bin/env python3
"""Prepare segmented CMU-MOSI media for the RDID-MSA training pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import av
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FFMPEG = shutil.which("ffmpeg") or str(Path(sys.executable).with_name("ffmpeg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CMU-MOSI MP4/WAV media and manifests")
    parser.add_argument(
        "--source-root", type=Path, default=Path("/home/wy/sjq/kd/dataset/cmu_mosi_source")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("/home/wy/sjq/kd/dataset/cmu_mosi")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def round_sentiment_class(value: float) -> int:
    rounded = math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)
    return max(-3, min(3, rounded))


def sentiment_bucket(value: float) -> str:
    if value < -0.5:
        return "negative"
    if value > 0.5:
        return "positive"
    return "neutral"


def duration_bucket(duration: float) -> str:
    if duration < 5:
        return "lt5"
    if duration < 15:
        return "5to15"
    if duration <= 30:
        return "15to30"
    return "gt30"


def media_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        if container.duration is not None:
            return float(container.duration / av.time_base)
        stream = container.streams.video[0]
        if stream.duration is None or stream.time_base is None:
            raise ValueError(f"duration unavailable: {path}")
        return float(stream.duration * stream.time_base)


def prepare_media(source: Path, video_target: Path, audio_target: Path, force: bool) -> float:
    if not source.is_file():
        raise FileNotFoundError(source)
    video_target.parent.mkdir(parents=True, exist_ok=True)
    audio_target.parent.mkdir(parents=True, exist_ok=True)
    if force or not video_target.is_file() or not audio_target.is_file():
        video_tmp = video_target.with_name(f"{video_target.stem}.tmp{video_target.suffix}")
        audio_tmp = audio_target.with_name(f"{audio_target.stem}.tmp{audio_target.suffix}")
        video_tmp.unlink(missing_ok=True)
        audio_tmp.unlink(missing_ok=True)
        command = [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            str(video_tmp),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_tmp),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            video_tmp.unlink(missing_ok=True)
            audio_tmp.unlink(missing_ok=True)
            raise RuntimeError(result.stderr.strip() or f"ffmpeg exit {result.returncode}")
        video_tmp.replace(video_target)
        audio_tmp.replace(audio_target)
    duration = media_duration(video_target)
    samples, sample_rate = sf.read(str(audio_target), dtype="float32", always_2d=False)
    target_frames = max(1, round(duration * 16000))
    if sample_rate != 16000 or samples.ndim != 1:
        raise ValueError(f"invalid prepared audio: {audio_target}")
    if len(samples) != target_frames:
        normalized = np.zeros(target_frames, dtype=np.float32)
        available = min(len(samples), target_frames)
        normalized[:available] = samples[:available]
        audio_tmp = audio_target.with_name(f"{audio_target.stem}.normalize.tmp{audio_target.suffix}")
        audio_tmp.unlink(missing_ok=True)
        sf.write(str(audio_tmp), normalized, 16000, subtype="PCM_16")
        audio_tmp.replace(audio_target)
    audio = sf.info(str(audio_target))
    if audio.channels != 1 or audio.samplerate != 16000 or audio.frames == 0:
        raise ValueError(f"invalid prepared audio: {audio_target}")
    audio_duration = audio.frames / audio.samplerate
    if abs(audio_duration - duration) > 1 / 16000:
        raise ValueError(
            f"audio/video duration mismatch for {source}: {audio_duration:.3f} vs {duration:.3f}"
        )
    return duration


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    labels_path = source_root / "label.csv"
    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 2199:
        raise ValueError(f"expected 2199 label rows, got {len(source_rows)}")

    def prepare(row: dict[str, str]) -> dict[str, Any]:
        video_id = row["video_id"]
        source_clip_id = int(row["clip_id"])
        clip_index = source_clip_id - 1
        stem = f"{video_id}__{clip_index}"
        source_video = source_root / "Raw" / video_id / f"{source_clip_id}.mp4"
        silent_video = output_root / "media/video_silent" / f"{stem}.mp4"
        audio = output_root / "media/audio_16k_mono" / f"{stem}.wav"
        duration = prepare_media(source_video, silent_video, audio, args.force)
        sentiment = float(row["label"])
        class_value = round_sentiment_class(sentiment)
        return {
            "manifest_version": "mosi-segmented-media-v1",
            "sample_id": f"{video_id}[{clip_index}]",
            "video_id": video_id,
            "clip_index": clip_index,
            "source_clip_id": source_clip_id,
            "split": row["mode"],
            "text": row["text"].strip(),
            "sentiment": sentiment,
            "class_7_value": class_value,
            "class_7_index": class_value + 3,
            "sentiment_bucket": sentiment_bucket(sentiment),
            "duration_bucket": duration_bucket(duration),
            "duration": duration,
            "official_start": 0.0,
            "official_end": duration,
            "source_video_path": str(source_video.resolve()),
            "silent_video_path": str(silent_video.resolve()),
            "audio_segment_path": str(audio.resolve()),
            "annotation": row["annotation"],
            "source_policy": "community_mirror_segmented_media_with_cmu_standard_splits",
            "status": "valid",
        }

    prepared_by_index: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {
            executor.submit(prepare, row): index for index, row in enumerate(source_rows)
        }
        for completed, future in enumerate(as_completed(future_to_index), start=1):
            index = future_to_index[future]
            try:
                prepared_by_index[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append({"row": str(index), "error": repr(exc)})
            if completed % 250 == 0 or completed == len(source_rows):
                print(f"prepare_progress={completed}/{len(source_rows)} failures={len(failures)}", flush=True)

    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "preparation_errors.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"failed to prepare {len(failures)} MOSI samples")

    rows = [prepared_by_index[index] for index in range(len(source_rows))]
    manifests = output_root / "manifests"
    manifest_shas = {"all": write_jsonl(manifests / "all.jsonl", rows)}
    for split in ("train", "valid", "test"):
        selected = [row for row in rows if row["split"] == split]
        manifest_shas[split] = write_jsonl(manifests / f"{split}.jsonl", selected)
    summary = {
        "manifest_version": "mosi-segmented-media-v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_label": str(labels_path),
        "source_label_sha256": sha256_file(labels_path),
        "samples": len(rows),
        "videos": len({row["video_id"] for row in rows}),
        "split_rows": dict(Counter(row["split"] for row in rows)),
        "split_videos": dict(Counter({row["video_id"]: row["split"] for row in rows}.values())),
        "manifest_sha256": manifest_shas,
    }
    (reports / "source_inventory.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
