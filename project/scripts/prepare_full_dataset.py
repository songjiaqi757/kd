#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FFMPEG = shutil.which("ffmpeg") or str(Path(sys.executable).with_name("ffmpeg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare all official CMU-MOSEI segments")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/wy/sjq/kd/dataset/cmu_mosei/manifests/all.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ffmpeg-threads", type=int, default=2)
    parser.add_argument("--mode", choices=("both", "audio", "video"), default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_audio(row: dict[str, Any], audio: np.ndarray, sample_rate: int, force: bool) -> str:
    target = Path(row["audio_segment_path"])
    if target.is_file() and target.stat().st_size > 44 and not force:
        return "skipped"
    if sample_rate != 16000:
        raise ValueError(f"expected 16 kHz source, got {sample_rate}: {row['full_audio_path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    requested_frames = max(1, round(float(row["duration"]) * sample_rate))
    output = np.zeros(requested_frames, dtype=np.float32)
    source_start = max(0, round(float(row["official_start"]) * sample_rate))
    source_end = min(len(audio), round(float(row["official_end"]) * sample_rate))
    destination_start = max(0, round(float(row["audio_leading_padding_seconds"]) * sample_rate))
    available = max(0, min(source_end - source_start, requested_frames - destination_start))
    if available:
        output[destination_start : destination_start + available] = audio[
            source_start : source_start + available
        ]
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    temporary.unlink(missing_ok=True)
    sf.write(str(temporary), output, sample_rate, subtype="PCM_16")
    temporary.replace(target)
    return "created"


def write_video(row: dict[str, Any], ffmpeg_threads: int, force: bool) -> str:
    target = Path(row["silent_video_path"])
    if target.is_file() and target.stat().st_size > 0 and not force:
        return "skipped"
    target.parent.mkdir(parents=True, exist_ok=True)
    source_start = max(0.0, float(row["official_start"]))
    source_duration = max(
        0.001,
        min(float(row["official_end"]), float(row["source_video_duration"])) - source_start,
    )
    lead = float(row["video_leading_padding_seconds"])
    tail = float(row["video_tail_padding_seconds"])
    filters = [f"trim=duration={source_duration:.6f}", "setpts=PTS-STARTPTS"]
    if lead > 0:
        filters.append(f"tpad=start_mode=clone:start_duration={lead:.6f}")
    # Keep a small cloned-frame reserve before output trimming. This prevents
    # low-FPS inputs and keyframe seeking from ending one frame too early.
    filters.append(f"tpad=stop_mode=clone:stop_duration={tail + 0.25:.6f}")
    filters.append("fps=25")
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    temporary.unlink(missing_ok=True)
    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{source_start:.6f}",
        "-i",
        str(row["full_video_path"]),
        "-vf",
        ",".join(filters),
        "-t",
        f"{float(row['duration']):.6f}",
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
        "-threads",
        str(ffmpeg_threads),
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or f"ffmpeg exit {result.returncode}")
    temporary.replace(target)
    return "created"


def process_video_group(
    rows: list[dict[str, Any]], mode: str, ffmpeg_threads: int, force: bool
) -> dict[str, Any]:
    counts = {"audio_created": 0, "audio_skipped": 0, "video_created": 0, "video_skipped": 0}
    failures: list[dict[str, str]] = []
    audio = None
    sample_rate = None
    if mode in {"both", "audio"}:
        try:
            samples, sample_rate = sf.read(
                str(rows[0]["full_audio_path"]), always_2d=True, dtype="float32"
            )
            audio = np.mean(samples, axis=1)
        except Exception as exc:  # noqa: BLE001
            for row in rows:
                failures.append(
                    {"sample_id": str(row["sample_id"]), "stage": "audio_source", "error": repr(exc)}
                )

    for row in rows:
        if mode in {"both", "audio"} and audio is not None and sample_rate is not None:
            try:
                status = write_audio(row, audio, sample_rate, force)
                counts[f"audio_{status}"] += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {"sample_id": str(row["sample_id"]), "stage": "audio", "error": repr(exc)}
                )
        if mode in {"both", "video"}:
            try:
                status = write_video(row, ffmpeg_threads, force)
                counts[f"video_{status}"] += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {"sample_id": str(row["sample_id"]), "stage": "video", "error": repr(exc)}
                )
    return {"samples": len(rows), "counts": counts, "failures": failures}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not Path(FFMPEG).is_file():
        raise RuntimeError(f"ffmpeg not found: {FFMPEG}")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    if args.sample_ids:
        wanted = set(args.sample_ids)
        rows = [row for row in rows if str(row["sample_id"]) in wanted]
        found = {str(row["sample_id"]) for row in rows}
        if found != wanted:
            raise ValueError(f"sample IDs not found: {sorted(wanted - found)}")
    if args.limit is not None:
        rows = rows[: args.limit]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["video_id"])].append(row)

    output_root = args.manifest.resolve().parents[1]
    report_path = output_root / "reports/preparation_status.json"
    errors_path = output_root / "reports/preparation_errors.jsonl"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    totals = {"audio_created": 0, "audio_skipped": 0, "video_created": 0, "video_skipped": 0}
    failures: list[dict[str, str]] = []
    completed_samples = 0
    next_progress = args.progress_every

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_video_group, group, args.mode, args.ffmpeg_threads, args.force)
            for group in groups.values()
        ]
        for future in as_completed(futures):
            result = future.result()
            completed_samples += int(result["samples"])
            for key, value in result["counts"].items():
                totals[key] += int(value)
            failures.extend(result["failures"])
            if completed_samples >= next_progress or completed_samples == len(rows):
                status = {
                    "status": "running" if completed_samples < len(rows) else "complete",
                    "manifest": str(args.manifest.resolve()),
                    "manifest_sha256": sha256_file(args.manifest),
                    "mode": args.mode,
                    "workers": args.workers,
                    "ffmpeg_threads": args.ffmpeg_threads,
                    "started_at": started_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "total_samples": len(rows),
                    "completed_samples": completed_samples,
                    "counts": totals,
                    "failures": len(failures),
                    "elapsed_seconds": time.perf_counter() - started,
                    "pid": os.getpid(),
                }
                atomic_json(report_path, status)
                print(
                    f"prepare_progress={completed_samples}/{len(rows)} failures={len(failures)}",
                    flush=True,
                )
                next_progress = completed_samples + args.progress_every

    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
