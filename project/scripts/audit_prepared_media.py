#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import av
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/wy/sjq/kd/dataset/cmu_mosei")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prepared MOSEI audio and silent video")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATASET_ROOT / "manifests/benchmark500_prepared.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/benchmark500_media_audit.json",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def video_info(path: Path) -> dict[str, object]:
    with av.open(str(path)) as container:
        video_streams = len(container.streams.video)
        audio_streams = len(container.streams.audio)
        duration = float(container.duration / av.time_base) if container.duration is not None else None
        frames = int(container.streams.video[0].frames) if video_streams else 0
    return {
        "video_streams": video_streams,
        "audio_streams": audio_streams,
        "duration": duration,
        "frames": frames,
    }


def main() -> int:
    args = parse_args()
    payload = args.manifest.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    if args.limit is not None:
        rows = rows[: args.limit]
    failures: list[dict[str, str]] = []
    records: list[dict[str, object]] = []

    for row in rows:
        sample_id = str(row["sample_id"])
        video_path = Path(row["silent_video_path"])
        audio_path = Path(row["audio_segment_path"])
        try:
            video = video_info(video_path)
            audio = sf.info(str(audio_path))
            expected_duration = float(row["duration"])
            audio_duration = audio.frames / audio.samplerate
            checks = {
                "one_video_stream": video["video_streams"] == 1,
                "zero_video_audio_streams": video["audio_streams"] == 0,
                "audio_mono": audio.channels == 1,
                "audio_16khz": audio.samplerate == 16000,
                "audio_nonempty": audio.frames > 0,
                "audio_duration_match": abs(audio_duration - expected_duration) <= 0.05,
                "video_duration_match": video["duration"] is not None
                and abs(float(video["duration"]) - expected_duration) <= 0.10,
            }
            failed_checks = [name for name, passed in checks.items() if not passed]
            if failed_checks:
                failures.append({"sample_id": sample_id, "error": ",".join(failed_checks)})
            records.append(
                {
                    "sample_id": sample_id,
                    "split": row["split"],
                    "video_bytes": video_path.stat().st_size,
                    "audio_bytes": audio_path.stat().st_size,
                    "expected_duration": expected_duration,
                    "video_duration": video["duration"],
                    "audio_duration": audio_duration,
                    "checks": checks,
                }
            )
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": repr(exc)})

    report = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "samples": len(rows),
        "split": dict(Counter(str(row["split"]) for row in rows)),
        "passed": len(rows) - len(failures),
        "failed": len(failures),
        "total_video_bytes": sum(int(item["video_bytes"]) for item in records),
        "total_audio_bytes": sum(int(item["audio_bytes"]) for item in records),
        "failures": failures,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
