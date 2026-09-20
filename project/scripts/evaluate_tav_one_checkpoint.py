#!/usr/bin/env python3
"""Evaluate one committed TAV checkpoint on the MOSEI diagnostic test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]
from evaluate_tav_all_checkpoints import checkpoint_plan, evaluate_entry, freeze_plan, read_json
from evaluate_tav_streaming_checkpoints import build_inference, committed_history, stream_plan
from evaluate_tav_test import atomic_json, load_test_rows
from train_tav_distillation import sha256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--allow-diagnostic-test", action="store_true")
    args = parser.parse_args()
    if not args.allow_diagnostic_test:
        parser.error("diagnostic test requires --allow-diagnostic-test")
    if not re.fullmatch(r"epoch_\d{3}|best|last", args.checkpoint_name):
        parser.error("checkpoint name must be epoch_NNN, best, or last")
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("invalid batch size or worker count")
    return args


def resolved_entry(run, output, manifest, config, name):
    source_status = read_json(run / "status.json").get("status")
    if source_status == "failed":
        raise RuntimeError(f"source training failed: {run}")
    final = source_status == "complete" and (run / "checkpoint_inventory.json").is_file()
    if name in {"best", "last"} and not final:
        raise ValueError(f"{name}.pt cannot be tested before final checkpoint audit: {run}")
    if final:
        plan = checkpoint_plan(run, output, manifest)
        freeze_plan(plan, output)
        return plan, next(entry for entry in plan["checkpoints"] if entry["name"] == name)
    if not name.startswith("epoch_"):
        raise ValueError(f"non-epoch checkpoint unavailable during training: {name}")
    epoch = int(name.split("_")[1])
    history = committed_history(run)
    if epoch < 1 or epoch > len(history):
        raise ValueError(f"epoch {epoch} not committed in source history: {run}")
    path = (run / "checkpoints" / f"epoch_{epoch:03d}.pt").resolve()
    plan = stream_plan(run, output, manifest, config)
    plan_path = output / "streaming_plan.json"
    if plan_path.is_file() and read_json(plan_path) != plan:
        raise ValueError(f"streaming plan changed: {plan_path}")
    if not plan_path.is_file():
        atomic_json(plan, plan_path)
    entry = {"name": name, "role": "completed_epoch", "epoch": epoch,
             "path": str(path), "sha256": sha256(path)}
    return plan, entry


def main():
    args = parse_args()
    run, output, manifest = args.run.resolve(), args.output.resolve(), args.test_manifest.resolve()
    config = read_json(run / "run_config.json")
    if config.get("protocol") != "tav-m0-m6-v1" or config.get("checkpoint_selection") != "valid_mae":
        raise ValueError(f"source is not a valid-MAE-selected TAV run: {run}")
    output.mkdir(parents=True, exist_ok=True)
    plan, entry = resolved_entry(run, output, manifest, config, args.checkpoint_name)
    rows, parents = load_test_rows(manifest)
    destination = output / "checkpoints" / entry["name"]
    if (destination / "status.json").is_file() and read_json(destination / "status.json").get("status") == "complete":
        print(json.dumps({"status": "already_complete", "checkpoint": entry["name"]}), flush=True)
        return
    model, loader, device = build_inference(args, config, rows)
    result = evaluate_entry(entry, plan, config, model, loader, device, parents, output)
    print(json.dumps({"status": "complete", "method": config["method"],
                      "checkpoint": entry["name"], "epoch": entry["epoch"],
                      "test_mae": result["test_metrics"]["mae"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
