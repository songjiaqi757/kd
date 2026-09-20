#!/usr/bin/env python3
"""Evaluate exactly one RDID-v2 epoch checkpoint on exploratory MOSEI test."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]

from evaluate_rdid_v2_epoch_sweep import (
    METHODS, atomic_json, complete_epoch, evaluate_epoch, run_config,
)
from evaluate_tav_test import OfficialTestCollator, load_test_rows
from train_rdid_v2 import build_model
from train_student_baseline import seed_everything
from train_tav_distillation import VideoDataset, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--train-root", type=Path, default=ROOT / "outputs/experiments/rdid_v2_mosei/students")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/experiments/rdid_v2_mosei/epoch_test_sweep_20260919")
    parser.add_argument("--test-manifest", type=Path, default=ROOT / "dataset/cmu_mosei/manifests/official_test_windowed.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--allow-mosei-test-all-epochs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_mosei_test_all_epochs:
        parser.error("explicit --allow-mosei-test-all-epochs is required")
    if args.epoch <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("epoch and batch size must be positive; workers must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    source = run_config(args.train_root.resolve(), args.method)
    if source is None:
        raise FileNotFoundError(f"training run not configured: {args.method}")
    run, config = source
    checkpoint = run / "checkpoints" / f"epoch_{args.epoch:03d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output_root.resolve() / args.method / checkpoint.stem
    if complete_epoch(output):
        print(json.dumps({"status": "already_complete", "method": args.method,
                          "epoch": args.epoch}), flush=True)
        return
    manifest = args.test_manifest.resolve()
    rows, parents = load_test_rows(manifest)
    if args.dry_run:
        print(json.dumps({"method": args.method, "epoch": args.epoch,
                          "checkpoint": str(checkpoint), "output": str(output),
                          "test_windows": len(rows), "test_parents": len(parents),
                          "batch_size": args.batch_size,
                          "test_use_policy": "exploratory_all_epoch_test_not_blind_confirmation"},
                         indent=2), flush=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if complete_epoch(output):
            return
        try:
            seed_everything(13, True)
            device = torch.device("cuda:0")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            model_args = SimpleNamespace(**{
                key: value for key, value in config.items()
                if key not in {"frozen_assets", "input_sha256"}
            })
            for name in ("text_model", "audio_model", "video_model", "assets"):
                setattr(model_args, name, Path(getattr(model_args, name)))
            model = build_model(model_args, config["frozen_assets"], device)

            from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_args.text_model, local_files_only=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            audio_processor = AutoFeatureExtractor.from_pretrained(model_args.audio_model, local_files_only=True)
            video_processor = AutoImageProcessor.from_pretrained(
                model_args.video_model, local_files_only=True, use_fast=False
            )
            loader = DataLoader(
                VideoDataset(rows, video_processor, mode="video_lora"),
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=OfficialTestCollator(tokenizer, audio_processor),
            )
            evaluate_epoch(
                run=run, config=config, method=args.method, epoch=args.epoch,
                checkpoint_path=checkpoint, output=output, model=model,
                loader=loader, device=device, manifest=manifest,
                manifest_sha256=sha256(manifest), parent_count=len(parents),
            )
        except Exception as error:
            atomic_json({"status": "failed", "method": args.method,
                         "epoch": args.epoch, "error": repr(error)}, output / "status.json")
            raise


if __name__ == "__main__":
    main()
