#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.student_data import load_manifest_row, prepare_student_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen student encoder sequences")
    parser.add_argument("--manifest", type=Path, default=Path("/home/wy/sjq/kd/dataset/cmu_mosei/manifests/benchmark500_windowed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/home/wy/sjq/kd/outputs/student/features/benchmark500"))
    parser.add_argument("--text-model", type=Path, default=Path("/home/wy/sjq/kd/model/Qwen3-0.6B-Base"))
    parser.add_argument("--audio-model", type=Path, default=Path("/home/wy/sjq/kd/model/WavLM-Base-Plus"))
    parser.add_argument("--video-model", type=Path, default=Path("/home/wy/sjq/kd/model/VideoMAE-Base"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-frames", type=int, default=16)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA unavailable")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output / "items"
    feature_dir.mkdir(exist_ok=True)

    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel, AutoTokenizer

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    audio_processor = AutoFeatureExtractor.from_pretrained(args.audio_model, local_files_only=True)
    video_processor = AutoImageProcessor.from_pretrained(args.video_model, local_files_only=True)
    text_model = AutoModel.from_pretrained(args.text_model, local_files_only=True, dtype=dtype).to(device).eval()
    audio_model = AutoModel.from_pretrained(args.audio_model, local_files_only=True, dtype=dtype).to(device).eval()
    video_model = AutoModel.from_pretrained(args.video_model, local_files_only=True, dtype=dtype).to(device).eval()

    index_rows = []
    started = time.time()
    for index, row in enumerate(rows):
        feature_path = feature_dir / f"{index:06d}.pt"
        index_row = {
            "index": index,
            "feature_path": str(feature_path),
            "sample_id": row["sample_id"],
            "parent_sample_id": row.get("parent_sample_id", row["sample_id"]),
            "split": row["split"],
            "sentiment": float(row["sentiment"]),
            "class_7_index": int(row["class_7_index"]),
            "aggregation_weight": float(row.get("aggregation_weight", 1.0)),
        }
        index_rows.append(index_row)
        if feature_path.is_file():
            continue
        batch = prepare_student_sample(
            row, tokenizer, audio_processor, video_processor, video_frames=args.video_frames
        )
        inputs = {}
        for key in ("input_ids", "text_attention_mask", "input_values", "audio_attention_mask", "pixel_values"):
            value = batch[key]
            if value is not None:
                value = value.to(device)
                if value.is_floating_point():
                    value = value.to(dtype)
            inputs[key] = value
        with torch.inference_mode():
            text = text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["text_attention_mask"], return_dict=True
            ).last_hidden_state[0]
            audio_output = audio_model(
                input_values=inputs["input_values"], attention_mask=inputs["audio_attention_mask"], return_dict=True
            ).last_hidden_state[0]
            video = video_model(pixel_values=inputs["pixel_values"], return_dict=True).last_hidden_state[0]
        atomic_torch_save(
            {"t": text.cpu().to(torch.bfloat16), "a": audio_output.cpu().to(torch.bfloat16), "v": video.cpu().to(torch.bfloat16)},
            feature_path,
        )
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            elapsed = time.time() - started
            print(json.dumps({"completed": index + 1, "total": len(rows), "elapsed_seconds": round(elapsed, 1)}), flush=True)

    (args.output / "index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in index_rows), encoding="utf-8"
    )
    shapes = torch.load(index_rows[0]["feature_path"], map_location="cpu", weights_only=True)
    config = {
        "status": "complete" if len(rows) == 521 else "limited",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "count": len(rows),
        "dtype": "bfloat16",
        "video_frames": args.video_frames,
        "hidden_sizes": {key: int(value.shape[-1]) for key, value in shapes.items()},
        "models": {"text": str(args.text_model), "audio": str(args.audio_model), "video": str(args.video_model)},
    }
    (args.output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
