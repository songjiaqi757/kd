#!/usr/bin/env python3
"""Measure fixed-batch training on eight longest real train windows (no selection)."""
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor

import train_tav_distillation as train


def main():
    args = train.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train.seed_everything(args.seed, True)
    rows, _ = train.load_rows(args)
    rows = sorted(rows, key=lambda r: float(r["window_duration"]), reverse=True)[:8]
    device = torch.device(args.device)
    model = train.build_model(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audio = AutoFeatureExtractor.from_pretrained(args.audio_model, local_files_only=True)
    video = AutoImageProcessor.from_pretrained(args.video_model, local_files_only=True, use_fast=False)
    loader = DataLoader(train.VideoDataset(rows, video, mode=args.mode), batch_size=8,
                        num_workers=0, collate_fn=train.Collator(tokenizer, audio))
    batch = next(iter(loader))
    assets = json.loads(args.assets.read_text())
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    losses = []
    started = time.time()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**train.batch_inputs(batch, device), all_subsets=args.method in ("M4", "M5", "M6"))
            loss = train.compute_loss(out, batch, device, args, assets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        gradients = {n: float(p.grad.float().norm()) if p.grad is not None else None
                     for n,p in model.named_parameters() if p.requires_grad and "lora_B" in n}
        if any(v is None or not 0 < v < float("inf") for v in gradients.values()):
            raise RuntimeError("invalid active adapter gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    train.atomic_json({"status": "pass", "method": args.method, "seed": args.seed,
        "batch_size": 8, "steps": 3, "losses": losses,
        "elapsed_seconds": time.time()-started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()/1024**3,
        "sample_ids": [r["sample_id"] for r in rows],
        "durations": [r["window_duration"] for r in rows],
        "adapter_gradients": gradients, "official_test_evaluated": False}, args.output / "report.json")


if __name__ == "__main__":
    main()
