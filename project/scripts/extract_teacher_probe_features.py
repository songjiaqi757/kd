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

import numpy as np
import torch
import transformers
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/wy/sjq/kd/dataset/cmu_mosei")
OUTPUT_ROOT = Path("/home/wy/sjq/kd/outputs/probe/features/benchmark500")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.benchmark import append_jsonl  # noqa: E402
from rdid_mosei.probe import extract_thinker_last_input_state  # noqa: E402
from rdid_mosei.subsets import PROMPT_VERSIONS, SUBSETS, build_conversation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable frozen-Thinker feature extraction for the teacher Probe")
    parser.add_argument(
        "--model", type=Path, default=Path("/home/wy/sjq/kd/model/Qwen3-Omni-30B-A3B-Instruct")
    )
    parser.add_argument(
        "--manifest", type=Path, default=DATASET_ROOT / "manifests/benchmark500_windowed.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default="v1")
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_index(path: Path, jobs: list[tuple[dict, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for index, (sample, subset) in enumerate(jobs):
            row = {
                "job_index": index,
                "sample_id": sample["sample_id"],
                "parent_sample_id": sample.get("parent_sample_id", sample["sample_id"]),
                "video_id": sample["video_id"],
                "split": sample["split"],
                "subset": subset,
                "target_sentiment": sample["sentiment"],
                "class_7_index": sample["class_7_index"],
                "window_index": sample.get("window_index", 0),
                "window_count": sample.get("window_count", 1),
                "aggregation_weight": sample.get("aggregation_weight", sample["duration"]),
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    torch.manual_seed(args.seed)
    manifest_payload = args.manifest.read_bytes()
    samples = [json.loads(line) for line in manifest_payload.decode("utf-8").splitlines() if line]
    if args.limit is not None:
        samples = samples[: args.limit]
    jobs = [(sample, subset) for sample in samples for subset in args.subsets]

    config = Qwen3OmniMoeConfig.from_pretrained(args.model, local_files_only=True)
    feature_dimension = int(config.thinker_config.text_config.hidden_size)
    run_config = {
        "schema_version": "teacher-probe-features-v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "model": str(args.model.resolve()),
        "subsets": args.subsets,
        "sample_rows": len(samples),
        "jobs": len(jobs),
        "feature_dimension": feature_dimension,
        "feature_dtype": "float16",
        "pooling": "last_valid_input_token",
        "prompt_version": args.prompt_version,
        "video_fps": args.video_fps,
        "seed": args.seed,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("teacher feature extraction requires two visible GPUs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    index_path = args.output_dir / "index.jsonl"
    features_path = args.output_dir / "features.npy"
    completed_path = args.output_dir / "completed.npy"
    errors_path = args.output_dir / "errors.jsonl"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "manifest_sha256", "model", "subsets", "sample_rows", "jobs", "feature_dimension",
            "pooling", "prompt_version", "video_fps", "seed",
        )
        if any(existing.get(key) != run_config.get(key) for key in comparable_keys):
            raise ValueError(f"existing extraction configuration does not match requested run: {config_path}")
    else:
        config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_index(index_path, jobs)
        features = np.lib.format.open_memmap(
            features_path, mode="w+", dtype=np.float16, shape=(len(jobs), feature_dimension)
        )
        features[:] = np.nan
        features.flush()
        completed = np.lib.format.open_memmap(completed_path, mode="w+", dtype=np.bool_, shape=(len(jobs),))
        completed[:] = False
        completed.flush()

    features = np.lib.format.open_memmap(features_path, mode="r+")
    completed = np.lib.format.open_memmap(completed_path, mode="r+")
    if features.shape != (len(jobs), feature_dimension) or completed.shape != (len(jobs),):
        raise ValueError("existing feature arrays have incompatible shapes")
    pending = [index for index in range(len(jobs)) if not bool(completed[index])]
    print(f"completed={int(completed.sum())} pending={len(pending)}", flush=True)
    if not pending:
        return 0

    load_started = time.perf_counter()
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, local_files_only=True)
    config.enable_audio_output = False
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    error_count = 0
    media_subset = ("a" if any("a" in subset for subset in args.subsets) else "") + (
        "v" if any("v" in subset for subset in args.subsets) else ""
    )
    cached_sample_id: str | None = None
    cached_audios = None
    cached_images = None
    cached_videos = None
    for progress, job_index in enumerate(pending, start=1):
        sample, subset = jobs[job_index]
        started = time.perf_counter()
        try:
            sample_id = str(sample["sample_id"])
            if sample_id != cached_sample_id:
                if media_subset:
                    media_conversation = build_conversation(
                        sample,
                        media_subset,
                        prompt_version=args.prompt_version,
                        video_fps=args.video_fps,
                    )
                    cached_audios, cached_images, cached_videos = process_mm_info(
                        media_conversation, use_audio_in_video=False
                    )
                else:
                    cached_audios = cached_images = cached_videos = None
                cached_sample_id = sample_id
            conversation = build_conversation(
                sample, subset, prompt_version=args.prompt_version, video_fps=args.video_fps
            )
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = cached_audios if "a" in subset else None
            images = cached_images if "v" in subset else None
            videos = cached_videos if "v" in subset else None
            inputs = processor(
                text=prompt,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            inputs = inputs.to(model.device).to(model.dtype)
            with torch.inference_mode():
                hidden = extract_thinker_last_input_state(model, inputs)
            vector = hidden[0].detach().to(device="cpu", dtype=torch.float16).numpy()
            if vector.shape != (feature_dimension,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"invalid feature vector shape/values: {vector.shape}")
            features[job_index] = vector
            features.flush()
            completed[job_index] = True
            completed.flush()
        except Exception as exc:
            error_count += 1
            append_jsonl(
                errors_path,
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "job_index": job_index,
                    "sample_id": sample["sample_id"],
                    "subset": subset,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        finally:
            if "inputs" in locals():
                del inputs
            torch.cuda.empty_cache()
        print(
            f"feature_progress={progress}/{len(pending)} job={job_index} "
            f"seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )

    final_config = {**run_config, "model_load_seconds": load_seconds, "completed_jobs": int(completed.sum())}
    config_path.write_text(json.dumps(final_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
