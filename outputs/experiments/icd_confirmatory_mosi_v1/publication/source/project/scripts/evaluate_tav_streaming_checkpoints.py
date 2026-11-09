#!/usr/bin/env python3
"""Test each retained epoch once its train/valid history row is committed.

The completed source run is still required before testing best.pt/last.pt and
finalizing the diagnostic summary. Test metrics never select an epoch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]
from evaluate_tav_all_checkpoints import checkpoint_plan, evaluate_entry, freeze_plan, read_json
from evaluate_tav_test import OfficialTestCollator, atomic_json, load_test_rows
from train_student_baseline import seed_everything
from train_tav_distillation import VideoDataset, build_model, sha256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--allow-diagnostic-test", action="store_true")
    args = parser.parse_args()
    if not args.allow_diagnostic_test:
        parser.error("diagnostic test requires --allow-diagnostic-test")
    if args.batch_size <= 0 or args.num_workers < 0 or args.poll_seconds <= 0:
        parser.error("invalid batch size, worker count, or polling period")
    return args


def committed_history(run):
    path = run / "history.json"
    rows = read_json(path) if path.is_file() else []
    if [int(row["epoch"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"non-contiguous committed epoch history: {run}")
    for row in rows:
        path = run / "checkpoints" / f"epoch_{int(row['epoch']):03d}.pt"
        if not path.is_file():
            raise ValueError(f"history row lacks retained epoch checkpoint: {path}")
    return rows


def stream_plan(run, output, manifest, config):
    return {
        "schema": "rdid-msa-tav-streaming-checkpoint-diagnostic-plan-v1",
        "source_run": str(run), "source_method": config["method"], "source_seed": config["seed"],
        "source_run_config_sha256": sha256(run / "run_config.json"),
        "source_valid_selected_epoch": None,
        "test_manifest": str(manifest), "test_manifest_sha256": sha256(manifest),
        "output": str(output),
        "test_use_policy": "diagnostic_only_never_select_epoch_or_hyperparameters_on_test",
        "readiness": "epoch_file_exists_and_history_row_committed_after_last_checkpoint_save",
    }


def build_inference(args, config, rows):
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
    return model, loader, device


def main():
    args = parse_args()
    run, output, manifest = args.run.resolve(), args.output.resolve(), args.test_manifest.resolve()
    config = read_json(run / "run_config.json")
    if config.get("protocol") != "tav-m0-m6-v1" or config.get("checkpoint_selection") != "valid_mae":
        raise ValueError(f"source is not a valid-MAE-selected TAV run: {run}")
    rows, parents = load_test_rows(manifest)
    plan = stream_plan(run, output, manifest, config)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "streaming_plan.json"
    if plan_path.is_file() and read_json(plan_path) != plan:
        raise ValueError(f"streaming plan changed: {plan_path}")
    if not plan_path.is_file():
        atomic_json(plan, plan_path)
    model, loader, device = build_inference(args, config, rows)
    completed = []
    next_epoch = 1
    while True:
        if sha256(run / "run_config.json") != plan["source_run_config_sha256"]:
            raise ValueError(f"source run config changed: {run}")
        source_status = read_json(run / "status.json").get("status")
        if source_status == "failed":
            raise RuntimeError(f"source training failed: {run}")
        history = committed_history(run)
        while next_epoch <= len(history):
            path = (run / "checkpoints" / f"epoch_{next_epoch:03d}.pt").resolve()
            entry = {"name": f"epoch_{next_epoch:03d}", "role": "completed_epoch",
                     "epoch": next_epoch, "path": str(path), "sha256": sha256(path)}
            result = evaluate_entry(entry, plan, config, model, loader, device, parents, output)
            completed.append(result)
            atomic_json({"status": "running", "method": config["method"],
                         "completed": len(completed), "source_completed_epochs": len(history),
                         "last_checkpoint": entry["name"]}, output / "status.json")
            print(json.dumps({"method": config["method"], "checkpoint": entry["name"],
                              "test_mae": result["test_metrics"]["mae"],
                              "completed": len(completed)}, ensure_ascii=False), flush=True)
            next_epoch += 1
        if source_status == "complete" and (run / "checkpoint_inventory.json").is_file():
            final_plan = checkpoint_plan(run, output, manifest)
            if final_plan["source_run_config_sha256"] != plan["source_run_config_sha256"]:
                raise ValueError("final source protocol changed after streaming began")
            freeze_plan(final_plan, output)
            results = []
            for entry in final_plan["checkpoints"]:
                result = evaluate_entry(entry, final_plan, config, model, loader, device, parents, output)
                selected = entry["epoch"] == final_plan["source_valid_selected_epoch"]
                if result["is_valid_selected_epoch"] != selected:
                    result["is_valid_selected_epoch"] = selected
                    atomic_json(result, output / "checkpoints" / entry["name"] / "report.json")
                results.append(result)
                atomic_json({"status": "running", "method": config["method"],
                             "completed": len(results), "total": len(final_plan["checkpoints"]),
                             "last_checkpoint": entry["name"]}, output / "status.json")
            atomic_json({"schema": "rdid-msa-tav-run-all-checkpoints-diagnostic-summary-v1",
                         "method": config["method"], "seed": config["seed"],
                         "valid_selected_epoch": final_plan["source_valid_selected_epoch"],
                         "checkpoints": results, "test_use_policy": final_plan["test_use_policy"]},
                        output / "summary.json")
            atomic_json({"status": "complete", "method": config["method"],
                         "completed": len(results), "total": len(results)}, output / "status.json")
            return
        atomic_json({"status": "waiting_for_next_epoch", "method": config["method"],
                     "completed": len(completed), "source_completed_epochs": len(history),
                     "next_epoch": next_epoch}, output / "status.json")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
