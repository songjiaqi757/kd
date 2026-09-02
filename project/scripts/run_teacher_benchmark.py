#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/wy/sjq/kd/dataset/cmu_mosei")
OUTPUT_ROOT = Path("/home/wy/sjq/kd/outputs/teacher_benchmark")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.benchmark import append_jsonl, load_completed_keys, select_jobs  # noqa: E402
from rdid_mosei.subsets import PROMPT_VERSIONS, SUBSETS, build_conversation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable Qwen3-Omni seven-subset benchmark")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/wy/sjq/kd/model/Qwen3-Omni-30B-A3B-Instruct"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=DATASET_ROOT / "manifests/benchmark500_windowed.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_ROOT / "teacher_benchmark500_windowed.jsonl"
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=OUTPUT_ROOT / "teacher_benchmark500_windowed.errors.jsonl",
    )
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default="v1")
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def tensor_metadata(inputs: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            result[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return result


def gpu_memory() -> list[dict[str, Any]]:
    return [
        {
            "device": index,
            "name": torch.cuda.get_device_name(index),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(index),
        }
        for index in range(torch.cuda.device_count())
    ]


def main() -> int:
    args = parse_args()
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    torch.manual_seed(args.seed)
    rows_payload = args.manifest.read_bytes()
    rows = [json.loads(line) for line in rows_payload.decode("utf-8").splitlines() if line]
    if args.limit is not None:
        rows = rows[: args.limit]
    completed = load_completed_keys(args.output)
    jobs = select_jobs(rows, args.subsets, completed)
    manifest_sha256 = hashlib.sha256(rows_payload).hexdigest()

    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "manifest_sha256": manifest_sha256,
                "rows": len(rows),
                "subsets": args.subsets,
                "completed": len(completed),
                "pending_jobs": len(jobs),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("teacher benchmark requires two visible GPUs")

    load_started = time.perf_counter()
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, local_files_only=True)
    model_config = Qwen3OmniMoeConfig.from_pretrained(args.model, local_files_only=True)
    model_config.enable_audio_output = False
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model,
        config=model_config,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    error_count = 0
    for job_index, (sample, subset) in enumerate(jobs, start=1):
        base = {
            "schema_version": "teacher-benchmark-v2",
            "sample_id": sample["sample_id"],
            "parent_sample_id": sample.get("parent_sample_id", sample["sample_id"]),
            "video_id": sample["video_id"],
            "split": sample["split"],
            "subset": subset,
            "duration": sample["duration"],
            "utterance_duration": sample.get("utterance_duration", sample["duration"]),
            "window_index": sample.get("window_index", 0),
            "window_count": sample.get("window_count", 1),
            "window_start": sample.get("window_start", 0.0),
            "window_end": sample.get("window_end", sample["duration"]),
            "aggregation_weight": sample.get("aggregation_weight", sample["duration"]),
            "manifest_sha256": manifest_sha256,
            "prompt_version": args.prompt_version,
            "video_fps": args.video_fps,
            "model_path": str(args.model.resolve()),
            "attention_backend": "pytorch_sdpa",
            "device_map": args.device_map,
            "dtype": "bfloat16",
            "seed": args.seed,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "model_load_seconds": load_seconds,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            for device_index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device_index)
            preprocess_started = time.perf_counter()
            conversation = build_conversation(
                sample,
                subset,
                prompt_version=args.prompt_version,
                video_fps=args.video_fps,
            )
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(
                text=prompt,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            shapes = tensor_metadata(inputs)
            preprocess_seconds = time.perf_counter() - preprocess_started
            inputs = inputs.to(model.device).to(model.dtype)

            inference_started = time.perf_counter()
            with torch.inference_mode():
                text_ids = model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            for device_index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device_index)
            inference_seconds = time.perf_counter() - inference_started
            generated = text_ids[:, inputs["input_ids"].shape[1] :]
            output_text = processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            append_jsonl(
                args.output,
                {
                    **base,
                    "status": "ok",
                    "input_tensors": shapes,
                    "preprocess_seconds": preprocess_seconds,
                    "inference_seconds": inference_seconds,
                    "output_text": output_text,
                    "gpu_memory": gpu_memory(),
                },
            )
        except Exception as exc:
            error_count += 1
            append_jsonl(
                args.errors,
                {
                    **base,
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        finally:
            if "inputs" in locals():
                del inputs
            torch.cuda.empty_cache()
        print(f"benchmark_progress={job_index}/{len(jobs)}", flush=True)
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
