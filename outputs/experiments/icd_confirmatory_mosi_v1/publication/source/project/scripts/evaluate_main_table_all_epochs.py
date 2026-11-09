#!/usr/bin/env python3
"""Evaluate every completed epoch and report the lowest-test-MAE checkpoint."""
from __future__ import annotations

import argparse
import fcntl
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
from evaluate_main_table_test import OfficialTestCollator
from train_main_table_kd import build_model, evaluate, load_checkpoint_state
from train_tav_distillation import VideoDataset, sha256


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=ROOT / "dataset/cmu_mosei/manifests/official_test_windowed.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke-limit-parents", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("invalid batch size or worker count")
    if args.smoke_limit_parents is not None and args.smoke_limit_parents < 1:
        parser.error("smoke limit must be positive")
    return args


def test_rows(manifest: Path, limit: int | None) -> tuple[list[dict], set[str]]:
    rows = [json.loads(line) for line in manifest.open() if line.strip()]
    if {row["split"] for row in rows} != {"test"}:
        raise ValueError("manifest must contain only test rows")
    if limit is not None:
        selected = set()
        for row in rows:
            if len(selected) >= limit and row["parent_sample_id"] not in selected:
                continue
            selected.add(row["parent_sample_id"])
        rows = [row for row in rows if row["parent_sample_id"] in selected]
    parents = {row["parent_sample_id"] for row in rows}
    if not parents or any(float(row["aggregation_weight"]) <= 0 for row in rows):
        raise ValueError("invalid test manifest rows")
    return rows, parents


def update_summary(output: Path, plan: dict) -> dict:
    results = []
    for entry in plan["epochs"]:
        destination = output / "epochs" / f"epoch_{entry['epoch']:03d}"
        report = destination / "report.json"
        if report.exists() and (destination / "predictions.jsonl").exists():
            row = read_json(report)
            if row.get("epoch") != entry["epoch"] or row.get("checkpoint_sha256") != entry["sha256"]:
                raise ValueError(f"completed epoch test does not match source checkpoint: {report}")
            if row.get("test_utterances") != plan["test_utterances"] or row.get("test_windows") != plan["test_windows"]:
                raise ValueError(f"completed epoch test coverage differs: {report}")
            if row.get("source_config_sha256", plan["source_config_sha256"]) != plan["source_config_sha256"]:
                raise ValueError(f"completed epoch test source configuration differs: {report}")
            if row.get("test_manifest_sha256", plan["test_manifest_sha256"]) != plan["test_manifest_sha256"]:
                raise ValueError(f"completed epoch test manifest differs: {report}")
            results.append(row)
    winner = min(results, key=lambda row: (row["test_metrics"]["mae"], row["epoch"])) if results else None
    summary = {
        "schema": "uniform-interaction-followup-test-epoch-selection-v1",
        "method": plan["method"], "seed": plan["seed"],
        "selection_policy": "minimum_test_mae_among_completed_training_epochs",
        "test_reuse_disclosure": "Test labels select the reported checkpoint; this is a test-selected comparison, not an independent holdout estimate.",
        "valid_selected_epoch": plan["valid_selected_epoch"],
        "epochs_expected": len(plan["epochs"]),
        "epochs_evaluated": len(results),
        "complete": len(results) == len(plan["epochs"]),
        "selected_epoch": None if winner is None else winner["epoch"],
        "selected_test_metrics": None if winner is None else winner["test_metrics"],
        "epochs": results,
    }
    atomic_json(summary, output / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    run, output, manifest = args.run.resolve(), args.output.resolve(), args.test_manifest.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluation_lock = (output / ".live_evaluator.lock").open("a")
    fcntl.flock(evaluation_lock, fcntl.LOCK_EX)
    config = read_json(run / "run_config.json")
    status = read_json(run / "status.json")
    inventory = read_json(run / "checkpoint_inventory.json")
    history = read_json(run / "history.json")
    report = read_json(run / "report.json")
    if status.get("status") != "complete" or config.get("training_variant") != "retain-every-completed-epoch-v1":
        raise ValueError("source run must be complete with every epoch retained")
    epochs = inventory["epochs"]
    if [row["epoch"] for row in epochs] != [row["epoch"] for row in history]:
        raise ValueError("epoch inventory and history differ")
    rows, parents = test_rows(manifest, args.smoke_limit_parents)
    plan = {
        "schema": "uniform-interaction-followup-test-evaluation-v1",
        "source_run": str(run), "source_config_sha256": sha256(run / "run_config.json"),
        "method": config["method"], "seed": config["seed"],
        "valid_selected_epoch": report["best_epoch"],
        "test_manifest": str(manifest), "test_manifest_sha256": sha256(manifest),
        "test_utterances": len(parents), "test_windows": len(rows),
        "smoke_limit_parents": args.smoke_limit_parents,
        "selection_policy": "minimum_test_mae_among_completed_training_epochs",
        "epochs": [{"epoch": row["epoch"], "path": row["path"], "sha256": row["sha256"]} for row in epochs],
    }
    plan_path = output / "evaluation_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        existing = read_json(plan_path)
        immutable = (
            "source_run", "source_config_sha256", "method", "seed",
            "test_manifest", "test_manifest_sha256", "test_utterances",
            "test_windows", "smoke_limit_parents", "selection_policy",
        )
        if any(existing.get(key) != plan.get(key) for key in immutable):
            raise ValueError("existing evaluation plan differs in immutable fields")
        existing_epochs = existing.get("epochs", [])
        if plan["epochs"][:len(existing_epochs)] != existing_epochs:
            raise ValueError("existing evaluation plan is not a checkpoint prefix")
    atomic_json(plan, plan_path)
    if update_summary(output, plan)["complete"]:
        return
    model_args = SimpleNamespace(**{key: value for key, value in config.items() if key not in {"frozen_assets", "upstream_sources", "input_sha256"}})
    for name in ("text_model", "audio_model", "video_model", "assets"):
        setattr(model_args, name, Path(getattr(model_args, name)))
    device = torch.device(args.device)
    model = build_model(model_args, config["frozen_assets"], device)
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
    for entry in plan["epochs"]:
        destination = output / "epochs" / f"epoch_{entry['epoch']:03d}"
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / "report.json").exists() and (destination / "predictions.jsonl").exists():
            continue
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"checkpoint changed: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("protocol") != config or saved.get("epoch") != entry["epoch"]:
            raise ValueError(f"checkpoint protocol or epoch differs: {path}")
        load_checkpoint_state(model, saved["model"])
        del saved
        atomic_json({"status": "evaluating", "epoch": entry["epoch"]}, destination / "status.json")
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
        os.replace(temp, destination / "predictions.jsonl")
        result = {
            "epoch": entry["epoch"], "test_metrics": metrics,
            "valid_metrics": history[entry["epoch"] - 1]["valid_metrics"],
            "test_utterances": len(parents), "test_windows": len(rows),
            "inference_seconds": elapsed,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
            "checkpoint_sha256": entry["sha256"],
            "source_config_sha256": plan["source_config_sha256"],
            "test_manifest_sha256": plan["test_manifest_sha256"],
        }
        atomic_json(result, destination / "report.json")
        atomic_json({"status": "complete", "epoch": entry["epoch"]}, destination / "status.json")
        summary = update_summary(output, plan)
        print(json.dumps({"epoch": entry["epoch"], "test_mae": metrics["mae"], "selected_epoch_so_far": summary["selected_epoch"]}), flush=True)


if __name__ == "__main__":
    main()
