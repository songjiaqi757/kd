#!/usr/bin/env python3
"""Create deterministic video-grouped folds for sample-wise utility cross-fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "project/configs/utility_crossfit_5fold_v1.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def stable_tie(seed: int, video_id: str) -> str:
    return hashlib.sha256(f"{seed}:{video_id}".encode()).hexdigest()


def assign(rows: list[dict], folds: int, seed: int) -> dict:
    if folds < 2:
        raise ValueError("folds must be at least two")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            grouped[str(row["video_id"])].append(row)
    if len(grouped) < folds:
        raise ValueError("fewer train videos than folds")
    fold_videos = [[] for _ in range(folds)]
    fold_windows = [0] * folds
    fold_utterances = [0] * folds
    ordered = sorted(
        grouped,
        key=lambda video: (
            -len(grouped[video]),
            -len({str(row["parent_sample_id"]) for row in grouped[video]}),
            stable_tie(seed, video),
        ),
    )
    for video in ordered:
        fold = min(range(folds), key=lambda index: (fold_windows[index], fold_utterances[index], len(fold_videos[index]), index))
        fold_videos[fold].append(video)
        fold_windows[fold] += len(grouped[video])
        fold_utterances[fold] += len({str(row["parent_sample_id"]) for row in grouped[video]})
    video_to_fold = {video: fold for fold, videos in enumerate(fold_videos) for video in videos}
    return {
        "protocol": "utility-crossfit-video-grouped-v1",
        "seed": seed,
        "folds": folds,
        "video_to_fold": dict(sorted(video_to_fold.items())),
        "fold_summary": [
            {"fold": fold, "videos": len(fold_videos[fold]), "windows": fold_windows[fold], "utterances": fold_utterances[fold]}
            for fold in range(folds)
        ],
        "train_videos": len(grouped),
        "train_windows": sum(fold_windows),
        "train_utterances": sum(fold_utterances),
        "valid_or_test_assigned": False,
        "official_test_evaluated": False,
    }


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    result = assign(rows, args.folds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "fold_summary": result["fold_summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
