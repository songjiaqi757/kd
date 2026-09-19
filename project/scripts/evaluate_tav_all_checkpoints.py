#!/usr/bin/env python3
"""Diagnostic MOSEI test evaluation of every retained checkpoint in one TAV run.

These measurements must not be used to choose an epoch: best.pt remains the
valid-MAE-selected checkpoint for the paper's official comparison.
"""
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
from evaluate_tav_test import OfficialTestCollator, atomic_json, evaluate_official, load_test_rows
from train_student_baseline import seed_everything
from train_tav_distillation import VideoDataset, build_model, load_trainable, sha256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--allow-diagnostic-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_diagnostic_test:
        parser.error("diagnostic test requires --allow-diagnostic-test")
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("invalid batch size or worker count")
    return args


def read_json(path):
    return json.loads(Path(path).read_text())


def checkpoint_plan(run, output, manifest):
    required = ("run_config.json", "status.json", "report.json", "history.json",
                "checkpoint_inventory.json", "best.pt", "last.pt")
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete source run {run}: {missing}")
    config = read_json(run / "run_config.json")
    status = read_json(run / "status.json")
    report = read_json(run / "report.json")
    history = read_json(run / "history.json")
    inventory = read_json(run / "checkpoint_inventory.json")
    if status.get("status") != "complete" or config.get("protocol") != "tav-m0-m6-v1":
        raise ValueError(f"source run is not a completed TAV run: {run}")
    if report.get("method") != config.get("method") or config.get("checkpoint_selection") != "valid_mae":
        raise ValueError(f"source method or selection policy differs: {run}")
    history_epochs = [int(row["epoch"]) for row in history]
    inventory_epochs = [int(row["epoch"]) for row in inventory["epochs"]]
    if history_epochs != inventory_epochs or history_epochs != list(range(1, len(history) + 1)):
        raise ValueError(f"epoch checkpoint inventory coverage differs: {run}")
    entries = []
    for row in inventory["epochs"]:
        epoch = int(row["epoch"])
        path = (run / "checkpoints" / f"epoch_{epoch:03d}.pt").resolve()
        if path != Path(row["path"]).resolve() or not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"epoch checkpoint path or size differs: {path}")
        entries.append({"name": f"epoch_{epoch:03d}", "role": "completed_epoch",
                        "epoch": epoch, "path": str(path), "sha256": row["sha256"]})
    for name, epoch in (("best", int(report["best_epoch"])), ("last", history_epochs[-1])):
        path = (run / f"{name}.pt").resolve()
        entries.append({"name": name, "role": f"{name}_checkpoint", "epoch": epoch,
                        "path": str(path), "sha256": sha256(path)})
    return {
        "schema": "rdid-msa-tav-all-checkpoints-diagnostic-test-plan-v1",
        "source_run": str(run), "source_method": config["method"], "source_seed": config["seed"],
        "source_run_config_sha256": sha256(run / "run_config.json"),
        "source_valid_selected_epoch": int(report["best_epoch"]),
        "test_manifest": str(manifest), "test_manifest_sha256": sha256(manifest),
        "output": str(output), "checkpoints": entries,
        "test_use_policy": "diagnostic_only_never_select_epoch_or_hyperparameters_on_test",
        "note": "best.pt and last.pt are evaluated separately even when their model weights duplicate an epoch snapshot",
    }


def freeze_plan(plan, output):
    path = output / "evaluation_plan.json"
    if path.is_file():
        if read_json(path) != plan:
            raise ValueError(f"checkpoint evaluation plan changed: {path}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(plan, path)


def evaluate_entry(entry, plan, config, model, loader, device, parents, output):
    destination = output / "checkpoints" / entry["name"]
    destination.mkdir(parents=True, exist_ok=True)
    record = {"schema": "rdid-msa-tav-checkpoint-diagnostic-test-v1",
              "source_run": plan["source_run"], "source_method": plan["source_method"],
              "source_seed": plan["source_seed"], "checkpoint": entry,
              "source_run_config_sha256": plan["source_run_config_sha256"],
              "test_manifest": plan["test_manifest"],
              "test_manifest_sha256": plan["test_manifest_sha256"],
              "test_use_policy": plan["test_use_policy"]}
    record_path = destination / "run_config.json"
    if record_path.is_file() and read_json(record_path) != record:
        raise ValueError(f"existing checkpoint test config differs: {destination}")
    if (destination / "status.json").is_file() and read_json(destination / "status.json").get("status") == "complete":
        if not (destination / "report.json").is_file() or not (destination / "predictions.jsonl").is_file():
            raise ValueError(f"completed checkpoint test lacks artifacts: {destination}")
        return read_json(destination / "report.json")
    atomic_json(record, record_path)
    atomic_json({"status": "loading", "checkpoint": entry["name"]}, destination / "status.json")
    try:
        checkpoint_path = Path(entry["path"])
        if sha256(checkpoint_path) != entry["sha256"]:
            raise ValueError(f"checkpoint changed: {checkpoint_path}")
        # last.pt is a trusted local trainer artifact and also contains numpy
        # RNG/optimizer state, which PyTorch's weights-only loader rejects.
        checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=entry["name"] != "last")
        if checkpoint.get("protocol") != config or int(checkpoint.get("epoch", -1)) != entry["epoch"]:
            raise ValueError(f"checkpoint protocol or epoch differs: {checkpoint_path}")
        load_trainable(model, checkpoint["model"])
        del checkpoint
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        metrics, predictions = evaluate_official(model, loader, device, destination / "status.json")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if len(predictions) != len(parents):
            raise ValueError(f"utterance aggregation coverage differs: {entry['name']}")
        predictions_path = destination / "predictions.jsonl"
        temporary = predictions_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
        os.replace(temporary, predictions_path)
        result = {"method": config["method"], "seed": config["seed"],
                  "checkpoint": entry["name"], "checkpoint_role": entry["role"],
                  "epoch": entry["epoch"], "is_valid_selected_epoch": entry["epoch"] == plan["source_valid_selected_epoch"],
                  "test_metrics": metrics, "test_windows": len(loader.dataset),
                  "test_utterances": len(predictions), "inference_seconds": elapsed,
                  "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
                  "test_use_policy": plan["test_use_policy"]}
        atomic_json(result, destination / "report.json")
        atomic_json({"status": "complete"}, destination / "status.json")
        return result
    except Exception as exc:
        atomic_json({"status": "failed", "error": repr(exc)}, destination / "status.json")
        raise


def main():
    args = parse_args()
    run, output, manifest = args.run.resolve(), args.output.resolve(), args.test_manifest.resolve()
    plan = checkpoint_plan(run, output, manifest)
    rows, parents = load_test_rows(manifest)
    if args.dry_run:
        print(json.dumps({"method": plan["source_method"], "checkpoints": len(plan["checkpoints"]),
                          "test_windows": len(rows), "test_utterances": len(parents),
                          "output": str(output)}, ensure_ascii=False))
        return
    freeze_plan(plan, output)
    config = read_json(run / "run_config.json")
    seed_everything(int(config["seed"]), True)
    model_args = SimpleNamespace(**{key: value for key, value in config.items()
                                    if key not in {"frozen_assets", "input_sha256"}})
    for name in ("text_model", "audio_model", "video_model", "assets"):
        setattr(model_args, name, Path(getattr(model_args, name)))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model = build_model(model_args, device)
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.text_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoFeatureExtractor.from_pretrained(model_args.audio_model, local_files_only=True)
    video_processor = AutoImageProcessor.from_pretrained(model_args.video_model, local_files_only=True, use_fast=False)
    loader = DataLoader(VideoDataset(rows, video_processor, mode=model_args.mode),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", collate_fn=OfficialTestCollator(tokenizer, audio_processor))
    results = []
    for entry in plan["checkpoints"]:
        result = evaluate_entry(entry, plan, config, model, loader, device, parents, output)
        results.append(result)
        atomic_json({"status": "running", "method": config["method"], "completed": len(results),
                     "total": len(plan["checkpoints"]), "last_checkpoint": entry["name"]}, output / "status.json")
        print(json.dumps({"method": config["method"], "checkpoint": entry["name"],
                          "epoch": entry["epoch"], "test_mae": result["test_metrics"]["mae"],
                          "completed": len(results), "total": len(plan["checkpoints"])}, ensure_ascii=False), flush=True)
    atomic_json({"schema": "rdid-msa-tav-run-all-checkpoints-diagnostic-summary-v1",
                 "method": config["method"], "seed": config["seed"],
                 "valid_selected_epoch": plan["source_valid_selected_epoch"],
                 "checkpoints": results, "test_use_policy": plan["test_use_policy"]}, output / "summary.json")
    atomic_json({"status": "complete", "method": config["method"],
                 "completed": len(results), "total": len(results)}, output / "status.json")


if __name__ == "__main__":
    main()
