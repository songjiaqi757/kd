#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/wy/sjq/kd/dataset/cmu_mosei")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bounded-duration teacher manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATASET_ROOT / "manifests/benchmark500_prepared.jsonl",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DATASET_ROOT / "manifests/benchmark500_windowed.jsonl",
    )
    parser.add_argument(
        "--media-dir", type=Path, default=DATASET_ROOT / "derived/benchmark500_windows"
    )
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--overlap", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def window_intervals(duration: float, max_duration: float, overlap: float) -> list[tuple[float, float]]:
    if duration <= 0:
        raise ValueError("duration must be positive")
    if max_duration <= 0 or overlap < 0 or overlap >= max_duration:
        raise ValueError("require max_duration > overlap >= 0")
    if duration <= max_duration:
        return [(0.0, duration)]
    stride = max_duration - overlap
    count = math.ceil((duration - max_duration) / stride) + 1
    starts = [min(index * stride, duration - max_duration) for index in range(count)]
    # Anchoring the final window at the utterance end avoids a very short tail window.
    starts[-1] = duration - max_duration
    starts = list(dict.fromkeys(round(start, 9) for start in starts))
    return [(start, min(duration, start + max_duration)) for start in starts]


def coverage_weights(intervals: list[tuple[float, float]]) -> list[float]:
    """Allocate overlapped time equally so aggregation weights sum to utterance duration."""
    weights = [0.0] * len(intervals)
    boundaries = sorted({point for interval in intervals for point in interval})
    for left, right in zip(boundaries, boundaries[1:]):
        midpoint = (left + right) / 2
        active = [index for index, (start, end) in enumerate(intervals) if start <= midpoint < end]
        for index in active:
            weights[index] += (right - left) / len(active)
    return weights


def write_audio_window(source: Path, target: Path, start: float, end: float) -> None:
    audio, sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
    left = max(0, round(start * sample_rate))
    right = min(len(audio), round(end * sample_rate))
    if right <= left:
        raise RuntimeError(f"empty audio window: {source} [{start}, {end}]")
    sf.write(str(target), audio[left:right], sample_rate, subtype="PCM_16")


def write_video_window(source: Path, target: Path, start: float, end: float) -> None:
    ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).with_name("ffmpeg"))
    if not Path(ffmpeg).is_file():
        raise RuntimeError("ffmpeg is required to create long-sample windows")
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(source),
        "-t",
        f"{end - start:.6f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {source}: {result.stderr.strip()}")
    temporary.replace(target)


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    args.media_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    long_samples = 0

    for row in rows:
        duration = float(row["duration"])
        intervals = window_intervals(duration, args.max_duration, args.overlap)
        weights = coverage_weights(intervals)
        is_windowed = len(intervals) > 1
        long_samples += int(is_windowed)

        for index, ((start, end), weight) in enumerate(zip(intervals, weights)):
            item = dict(row)
            item.update(
                {
                    "parent_sample_id": row["sample_id"],
                    "utterance_duration": duration,
                    "window_index": index,
                    "window_count": len(intervals),
                    "window_start": start,
                    "window_end": end,
                    "window_duration": end - start,
                    "aggregation_weight": weight,
                    "window_policy": f"max{args.max_duration:g}_overlap{args.overlap:g}_v1",
                }
            )
            item["duration"] = end - start
            if is_windowed:
                item["sample_id"] = f"{row['sample_id']}::w{index:02d}"
                stem = f"{row['video_id']}__{row['label_clip_index']}__w{index:02d}"
                video_target = args.media_dir / f"{stem}.silent.mp4"
                audio_target = args.media_dir / f"{stem}.wav"
                if args.force or not video_target.is_file():
                    write_video_window(Path(row["silent_video_path"]), video_target, start, end)
                if args.force or not audio_target.is_file():
                    write_audio_window(Path(row["audio_segment_path"]), audio_target, start, end)
                item["silent_video_path"] = str(video_target.resolve())
                item["audio_segment_path"] = str(audio_target.resolve())
            output_rows.append(item)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "input_samples": len(rows),
                "long_samples": long_samples,
                "output_units": len(output_rows),
                "output_manifest": str(args.output_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
