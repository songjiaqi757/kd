#!/usr/bin/env python3
"""Recreate missing MOSEI derived windows without changing locked manifests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/scripts"))
from build_windowed_manifest import write_audio_window, write_video_window

MANIFESTS = ROOT / "dataset/cmu_mosei/manifests"
DERIVED = (ROOT / "dataset/cmu_mosei/derived").resolve()
WINDOWED = ("official_train_valid_windowed.jsonl", "official_test_windowed.jsonl",
            "benchmark500_windowed.jsonl")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    sources = {row["sample_id"]: row for row in rows(MANIFESTS / "all.jsonl")}
    sources.update({row["sample_id"]: row for row in rows(MANIFESTS / "benchmark500_prepared.jsonl")})
    fingerprints = {name: hashlib.sha256((MANIFESTS / name).read_bytes()).hexdigest() for name in WINDOWED}
    tasks = {}
    for name in WINDOWED:
        for row in rows(MANIFESTS / name):
            if row.get("window_count", 1) <= 1:
                continue
            video = Path(row["silent_video_path"])
            audio = Path(row["audio_segment_path"])
            if video.is_file() and audio.is_file():
                continue
            for target in (video, audio):
                if not target.resolve().is_relative_to(DERIVED):
                    raise ValueError(f"unexpected derived target: {target}")
            parent = sources[row["parent_sample_id"]]
            for key in ("silent_video_path", "audio_segment_path"):
                if not Path(parent[key]).is_file():
                    raise FileNotFoundError(parent[key])
            key = str(video)
            task = (row, parent)
            if key in tasks and tasks[key][0] != row:
                raise ValueError(f"conflicting window target: {key}")
            tasks[key] = task
    print(json.dumps({"windows_to_restore": len(tasks)}), flush=True)

    def restore(item: tuple[dict, dict]) -> str:
        row, parent = item
        start, end = float(row["window_start"]), float(row["window_end"])
        video, audio = Path(row["silent_video_path"]), Path(row["audio_segment_path"])
        video.parent.mkdir(parents=True, exist_ok=True)
        audio.parent.mkdir(parents=True, exist_ok=True)
        if not video.is_file():
            write_video_window(Path(parent["silent_video_path"]), video, start, end)
        if not audio.is_file():
            write_audio_window(Path(parent["audio_segment_path"]), audio, start, end)
        return row["sample_id"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(restore, item) for item in tasks.values()]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 10 == 0 or index == len(futures):
                print(json.dumps({"restored": index, "total": len(futures)}), flush=True)
    for name, digest in fingerprints.items():
        if hashlib.sha256((MANIFESTS / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"manifest changed during restore: {name}")
        missing = [row["sample_id"] for row in rows(MANIFESTS / name)
                   if not Path(row["silent_video_path"]).is_file()
                   or not Path(row["audio_segment_path"]).is_file()]
        if missing:
            raise ValueError(f"missing derived media for {name}: {missing[:5]}")
    print(json.dumps({"status": "complete", "manifests_verified": WINDOWED,
                      "restored_windows": len(tasks)}), flush=True)


if __name__ == "__main__":
    main()
