#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.student import RDIDStudent, SUBSETS, student_task_loss
from rdid_mosei.student_data import load_manifest_row, prepare_student_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One real CMU-MOSEI student forward/backward smoke step")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/wy/sjq/kd/dataset/cmu_mosei/manifests/benchmark500_windowed.jsonl"),
    )
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--text-model", type=Path, default=Path("/home/wy/sjq/kd/model/Qwen3-0.6B-Base"))
    parser.add_argument("--audio-model", type=Path, default=Path("/home/wy/sjq/kd/model/WavLM-Base-Plus"))
    parser.add_argument("--video-model", type=Path, default=Path("/home/wy/sjq/kd/model/VideoMAE-Base"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-frames", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer

    row = load_manifest_row(args.manifest, args.row)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    audio_processor = AutoFeatureExtractor.from_pretrained(args.audio_model, local_files_only=True)
    video_processor = AutoImageProcessor.from_pretrained(args.video_model, local_files_only=True)
    batch = prepare_student_sample(
        row,
        tokenizer,
        audio_processor,
        video_processor,
        video_frames=args.video_frames,
    )

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = RDIDStudent.from_local_pretrained(
        args.text_model,
        args.audio_model,
        args.video_model,
        dtype=dtype,
        freeze_encoders=True,
    ).to(device=device, dtype=dtype)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1.0e-4, weight_decay=0.01)

    model_inputs = {}
    for key in ("input_ids", "text_attention_mask", "input_values", "audio_attention_mask", "pixel_values"):
        value = batch[key]
        if value is not None:
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
        model_inputs[key] = value
    sentiment = batch["sentiment"].to(device)
    class_index = batch["class_7_index"].to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**model_inputs)
    losses = student_task_loss(outputs["tav"], sentiment, class_index)
    losses["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()

    report = {
        "status": "ok",
        "sample_id": row["sample_id"],
        "device": str(device),
        "dtype": str(dtype),
        "subsets": list(outputs),
        "encoded_once_for_seven_subsets": tuple(outputs) == SUBSETS,
        "predictions": {name: float(value["regression"].detach().float().cpu()[0]) for name, value in outputs.items()},
        "loss": float(losses["loss"].detach().cpu()),
        "regression_loss": float(losses["regression_loss"].detach().cpu()),
        "classification_loss": float(losses["classification_loss"].detach().cpu()),
        "gradient_norm_before_clip": float(torch.as_tensor(grad_norm).detach().float().cpu()),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "input_shapes": {key: list(value.shape) if value is not None else None for key, value in model_inputs.items()},
        "peak_gpu_memory_gib": (
            round(torch.cuda.max_memory_allocated(device) / 1024**3, 3) if device.type == "cuda" else None
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
