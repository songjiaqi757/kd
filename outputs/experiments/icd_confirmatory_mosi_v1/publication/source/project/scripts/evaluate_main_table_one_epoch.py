#!/usr/bin/env python3
"""Evaluate one committed epoch while its training run may still be active."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]
from evaluate_main_table_all_epochs import atomic_json, read_json, test_rows
from evaluate_main_table_test import OfficialTestCollator
from train_main_table_kd import build_model, evaluate, load_checkpoint_state
from train_tav_distillation import VideoDataset, sha256


def localize_path(path: str | Path) -> Path:
    """Map paths embedded by an original Qiii run to this repository."""
    value = Path(path)
    if value.exists():
        return value
    qiii_root = Path("/ai/sjq/kd")
    try:
        relative = value.relative_to(qiii_root)
    except ValueError:
        return value
    localized = ROOT / relative
    return localized if localized.exists() else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--test-manifest", type=Path, default=ROOT / "dataset/cmu_mosei/manifests/official_test_windowed.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke-limit-parents", type=int)
    args = parser.parse_args()
    if args.epoch < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("invalid epoch, batch size, or worker count")
    run, output, manifest = args.run.resolve(), args.output.resolve(), args.test_manifest.resolve()
    config = read_json(run / "run_config.json")
    history = read_json(run / "history.json")
    retained_every_epoch = (
        config.get("training_variant") == "retain-every-completed-epoch-v1"
        or config.get("save_every_epoch") is True
    )
    if not retained_every_epoch:
        raise ValueError("source is not an every-epoch-retained run")
    if args.epoch > len(history) or history[args.epoch - 1]["epoch"] != args.epoch:
        raise ValueError("epoch has not been committed to training history")
    checkpoint = run / "checkpoints" / f"epoch_{args.epoch:03d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    rows, parents = test_rows(manifest, args.smoke_limit_parents)
    destination = output / "epochs" / f"epoch_{args.epoch:03d}"
    destination.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(run / "run_config.json")
    manifest_hash = sha256(manifest)
    checkpoint_hash = sha256(checkpoint)
    record = {
        "schema": "uniform-interaction-followup-streaming-test-v1",
        "source_run": str(run),
        "source_config_sha256": source_hash,
        "test_manifest": str(manifest),
        "test_manifest_sha256": manifest_hash,
        "test_utterances": len(parents),
        "test_windows": len(rows),
        "smoke_limit_parents": args.smoke_limit_parents,
        "epoch": args.epoch,
        "checkpoint_sha256": checkpoint_hash,
    }
    record_path = destination / "run_config.json"
    if record_path.exists() and read_json(record_path) != record:
        raise ValueError("existing per-epoch test protocol differs")
    atomic_json(record, record_path)
    report_path = destination / "report.json"
    predictions_path = destination / "predictions.jsonl"
    if report_path.exists() and predictions_path.exists():
        report = read_json(report_path)
        if report.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("existing test report checkpoint differs")
        return
    # Legacy --save-every-epoch checkpoints include optimizer/RNG objects that
    # predate PyTorch's weights-only loader; current retained checkpoints are
    # model-only and keep the safer loader path.
    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=config.get("save_every_epoch") is not True,
    )
    if saved.get("protocol") != config or saved.get("epoch") != args.epoch:
        raise ValueError("checkpoint protocol or epoch differs")
    model_args = SimpleNamespace(**{key: value for key, value in config.items() if key not in {"frozen_assets", "upstream_sources", "input_sha256"}})
    for name in ("text_model", "audio_model", "video_model", "assets"):
        setattr(model_args, name, localize_path(getattr(model_args, name)))
    device = torch.device(args.device)
    model = build_model(model_args, config["frozen_assets"], device)
    load_checkpoint_state(model, saved["model"])
    del saved
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.text_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audio = AutoFeatureExtractor.from_pretrained(model_args.audio_model, local_files_only=True)
    video = AutoImageProcessor.from_pretrained(model_args.video_model, local_files_only=True, use_fast=False)
    loader = DataLoader(
        VideoDataset(rows, video, mode="video_lora"), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
        collate_fn=OfficialTestCollator(tokenizer, audio),
    )
    atomic_json({"status": "evaluating", "epoch": args.epoch}, destination / "status.json")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    metrics, predictions = evaluate(model, loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if len(predictions) != len(parents):
        raise ValueError("test utterance aggregation coverage differs")
    temp = destination / "predictions.jsonl.tmp"
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    os.replace(temp, predictions_path)
    result = {
        "epoch": args.epoch, "test_metrics": metrics,
        "valid_metrics": history[args.epoch - 1]["valid_metrics"],
        "test_utterances": len(parents), "test_windows": len(rows),
        "inference_seconds": elapsed,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
        "checkpoint_sha256": checkpoint_hash,
        "source_config_sha256": source_hash,
        "test_manifest_sha256": manifest_hash,
    }
    atomic_json(result, report_path)
    atomic_json({"status": "complete", "epoch": args.epoch}, destination / "status.json")
    print(json.dumps({"epoch": args.epoch, "test_mae": metrics["mae"]}), flush=True)


if __name__ == "__main__":
    main()
