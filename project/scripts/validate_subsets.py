#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/wy/sjq/kd/dataset/cmu_mosei")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_omni_utils import process_mm_info  # noqa: E402
from rdid_mosei.subsets import SUBSETS, build_conversation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATASET_ROOT / "manifests/benchmark500_windowed.jsonl",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    sample = rows[args.sample_index]
    report = {"sample_id": sample["sample_id"], "subsets": {}}

    for subset in SUBSETS:
        conversation = build_conversation(sample, subset)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        report["subsets"][subset] = {
            "audio_count": 0 if audios is None else len(audios),
            "image_count": 0 if images is None else len(images),
            "video_count": 0 if videos is None else len(videos),
        }

    expected = {
        "t": (0, 0),
        "a": (1, 0),
        "v": (0, 1),
        "ta": (1, 0),
        "tv": (0, 1),
        "av": (1, 1),
        "tav": (1, 1),
    }
    for subset, (audio_count, video_count) in expected.items():
        actual = report["subsets"][subset]
        if (actual["audio_count"], actual["video_count"]) != (audio_count, video_count):
            raise RuntimeError(f"subset leakage/missing modality: {subset}: {actual}")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
