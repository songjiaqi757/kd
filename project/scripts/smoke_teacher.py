#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/wy/sjq/kd/dataset/cmu_mosei")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.subsets import build_conversation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Qwen3-Omni MOSEI teacher smoke inference")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/wy/sjq/kd/model/Qwen3-Omni-30B-A3B-Instruct"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=DATASET_ROOT / "manifests/benchmark500_windowed.jsonl"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--subset", choices=("t", "a", "v", "ta", "tv", "av", "tav"), default="tav")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This smoke test expects the two visible GPUs on the host node.")

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    sample = rows[args.sample_index]
    conversation = build_conversation(sample, args.subset)

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.disable_talker()

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
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.inference_mode():
        text_ids, _ = model.generate(
            **inputs,
            return_audio=False,
            thinker_max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    generated = text_ids[:, inputs["input_ids"].shape[1] :]
    output = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(json.dumps({"sample_id": sample["sample_id"], "subset": args.subset, "output": output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
