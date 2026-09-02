#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import av
import cv2
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/smoke20.jsonl",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/smoke20_prepared.jsonl",
    )
    parser.add_argument("--media-dir", type=Path, default=PROJECT_ROOT / "data/smoke_media")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def audio_stream_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return len(container.streams.audio)


def write_silent_video(
    source: Path, target: Path, start: float | None = None, end: float | None = None
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
        temporary.unlink(missing_ok=True)
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        if start is not None and end is not None:
            command.extend(["-ss", f"{start:.6f}"])
        command.extend(["-i", str(source)])
        if start is not None and end is not None:
            command.extend(
                [
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
                ]
            )
        else:
            command.extend(["-map", "0:v:0", "-an", "-c:v", "copy"])
        command.append(str(temporary))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0:
            temporary.replace(target)
            return
        temporary.unlink(missing_ok=True)

    # Fallback for environments without ffmpeg or incompatible source containers.
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not fps or fps <= 0:
        fps = 25.0
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video geometry: {source}")
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create video: {target}")
    frames = 0
    if start is not None:
        capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    expected_frames = None
    if start is not None and end is not None:
        expected_frames = max(1, round((end - start) * fps))
    try:
        while True:
            if expected_frames is not None and frames >= expected_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
    finally:
        capture.release()
        writer.release()
    if frames == 0:
        raise RuntimeError(f"decoded zero frames: {source}")


def write_audio_segment(source: Path, target: Path, start: float, end: float) -> int:
    audio, sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
    start_frame = max(0, round(start * sample_rate))
    requested_frames = max(1, round((end - start) * sample_rate))
    available_end = min(len(audio), start_frame + requested_frames)
    if available_end <= start_frame:
        raise RuntimeError(f"empty audio interval: {source} [{start}, {end}]")
    mono = np.mean(audio[start_frame:available_end], axis=1)
    padding_frames = requested_frames - len(mono)
    if padding_frames > 0:
        mono = np.pad(mono, (0, padding_frames))
    sf.write(str(target), mono, sample_rate, subtype="PCM_16")
    return padding_frames


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    args.media_dir.mkdir(parents=True, exist_ok=True)
    prepared = []

    for index, row in enumerate(rows, start=1):
        stem = f"{row['video_id']}__{row['label_clip_index']}"
        silent_video = args.media_dir / f"{stem}.silent.mp4"
        audio_segment = args.media_dir / f"{stem}.wav"
        start = max(0.0, float(row["start"]))
        end = float(row["end"])
        if args.force or not silent_video.exists():
            source_video = Path(row.get("full_video_path", row["segmented_video_path"]))
            if "full_video_path" in row:
                write_silent_video(source_video, silent_video, start, end)
            else:
                write_silent_video(source_video, silent_video)
        padding_frames = 0
        if args.force or not audio_segment.exists():
            padding_frames = write_audio_segment(
                Path(row["full_audio_path"]),
                audio_segment,
                start,
                end,
            )
        if audio_stream_count(silent_video) != 0:
            raise RuntimeError(f"audio leakage detected: {silent_video}")
        item = dict(row)
        item["silent_video_path"] = str(silent_video.resolve())
        item["audio_segment_path"] = str(audio_segment.resolve())
        item["media_source"] = "full_video_timestamp_crop"
        item["audio_padding_frames"] = padding_frames
        prepared.append(item)
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
            print(f"prepared_progress={index}/{len(rows)}", flush=True)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for item in prepared:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"prepared={len(prepared)} output={args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
