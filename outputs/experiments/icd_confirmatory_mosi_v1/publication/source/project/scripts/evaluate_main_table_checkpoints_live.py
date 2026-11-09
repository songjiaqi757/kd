#!/usr/bin/env python3
"""Evaluate retained checkpoints as soon as they are atomically published."""
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


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--minimum-free-gib", type=float, default=0.0)
    parser.add_argument("--admission-lock", type=Path)
    parser.add_argument("--smoke-limit-parents", type=int)
    parser.add_argument("--stop-after-evaluations", type=int)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    if (
        args.batch_size < 1 or args.num_workers < 0 or args.poll_seconds <= 0
        or args.minimum_free_gib < 0
    ):
        parser.error("invalid batch size, worker count, or poll interval")
    if args.smoke_limit_parents is not None and args.smoke_limit_parents < 1:
        parser.error("smoke limit must be positive")
    if args.stop_after_evaluations is not None and args.stop_after_evaluations < 1:
        parser.error("stop-after-evaluations must be positive")
    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        parser.error("worker index must be in [0, worker count)")
    return args


def test_rows(manifest: Path, limit: int | None) -> tuple[list[dict], set[str]]:
    rows = [json.loads(line) for line in manifest.open() if line.strip()]
    if {row["split"] for row in rows} != {"test"}:
        raise ValueError("manifest must contain only test rows")
    if limit is not None:
        selected: set[str] = set()
        for row in rows:
            if len(selected) >= limit and row["parent_sample_id"] not in selected:
                continue
            selected.add(row["parent_sample_id"])
        rows = [row for row in rows if row["parent_sample_id"] in selected]
    parents = {row["parent_sample_id"] for row in rows}
    if not parents or any(float(row["aggregation_weight"]) <= 0 for row in rows):
        raise ValueError("invalid test manifest rows")
    return rows, parents


def checkpoint_rows(run: Path, config: dict) -> list[dict]:
    result = []
    for path in sorted((run / "checkpoints").glob("epoch_*.pt")):
        epoch = int(path.stem.rsplit("_", 1)[1])
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("epoch") != epoch or saved.get("protocol") != config:
            raise ValueError(f"checkpoint protocol or epoch differs: {path}")
        valid_metrics = saved.get("valid_metrics")
        if valid_metrics is None:
            history = saved.get("history", [])
            if not history or int(history[-1]["epoch"]) != epoch:
                raise ValueError(f"checkpoint lacks matching validation metrics: {path}")
            valid_metrics = history[-1]["valid_metrics"]
        result.append({
            "epoch": epoch,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "valid_metrics": valid_metrics,
        })
        del saved
    return result


def finished_results(output: Path, plan: dict) -> list[dict]:
    results = []
    for entry in plan["epochs"]:
        destination = output / "epochs" / f"epoch_{entry['epoch']:03d}"
        report = destination / "report.json"
        predictions = destination / "predictions.jsonl"
        if not report.exists() or not predictions.exists():
            continue
        row = read_json(report)
        checks = (
            row.get("epoch") == entry["epoch"],
            row.get("checkpoint_sha256") == entry["sha256"],
            row.get("test_utterances") == plan["test_utterances"],
            row.get("test_windows") == plan["test_windows"],
            row.get("source_config_sha256") == plan["source_config_sha256"],
            row.get("test_manifest_sha256") == plan["test_manifest_sha256"],
        )
        if not all(checks):
            raise ValueError(f"completed epoch test differs from live plan: {report}")
        results.append(row)
    return results


def update_state(output: Path, plan: dict, source_status: dict) -> dict:
    with (output / ".summary_update.lock").open("a") as summary_lock:
        fcntl.flock(summary_lock, fcntl.LOCK_EX)
        results = finished_results(output, plan)
        winner = min(
            results, key=lambda row: (row["test_metrics"]["mae"], row["epoch"])
        ) if results else None
        source_complete = source_status.get("status") == "complete"
        summary = {
            "schema": "uniform-interaction-followup-test-epoch-selection-v1",
            "method": plan["method"],
            "seed": plan["seed"],
            "selection_policy": "minimum_test_mae_among_completed_training_epochs",
            "test_reuse_disclosure": (
                "Test labels select the reported checkpoint; this is a test-selected "
                "comparison, not an independent holdout estimate."
            ),
            "valid_selected_epoch": plan["valid_selected_epoch"],
            "epochs_expected": len(plan["epochs"]) if source_complete else None,
            "epochs_available": len(plan["epochs"]),
            "epochs_evaluated": len(results),
            "complete": source_complete and len(results) == len(plan["epochs"]),
            "selected_epoch": None if winner is None else winner["epoch"],
            "selected_test_metrics": None if winner is None else winner["test_metrics"],
            "epochs": results,
        }
        atomic_json(plan, output / "evaluation_plan.json")
        atomic_json(summary, output / "summary.json")
        atomic_json({
            "status": "complete" if summary["complete"] else "watching",
            "source_status": source_status.get("status"),
            "epochs_available": len(plan["epochs"]),
            "epochs_evaluated": len(results),
            "updated_at_unix": time.time(),
        }, output / "live_status.json")
        return summary


def current_best_epoch(run: Path, checkpoints: list[dict], status: dict) -> int | None:
    report = run / "report.json"
    if report.exists():
        return int(read_json(report)["best_epoch"])
    if status.get("best_epoch") is not None:
        return int(status["best_epoch"])
    if not checkpoints:
        return None
    return min(
        checkpoints,
        key=lambda row: (float(row["valid_metrics"]["mae"]), row["epoch"]),
    )["epoch"]


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    manifest = args.test_manifest.resolve()
    output.mkdir(parents=True, exist_ok=True)
    worker_lock_path = output / (
        f".live_evaluator.worker_{args.worker_index:02d}_of_{args.worker_count:02d}.lock"
    )
    with worker_lock_path.open("a") as worker_lock, (output / ".live_evaluator.lock").open("a") as shared_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(shared_lock, fcntl.LOCK_SH)
        while not (run / "run_config.json").exists():
            time.sleep(args.poll_seconds)
        config = read_json(run / "run_config.json")
        if config.get("training_variant") != "retain-every-completed-epoch-v1":
            raise ValueError("source run must retain every completed epoch")
        rows, parents = test_rows(manifest, args.smoke_limit_parents)
        config_sha = sha256(run / "run_config.json")
        manifest_sha = sha256(manifest)
        model_args = SimpleNamespace(**{
            key: value for key, value in config.items()
            if key not in {"frozen_assets", "upstream_sources", "input_sha256"}
        })
        for name in ("text_model", "audio_model", "video_model", "assets"):
            setattr(model_args, name, Path(getattr(model_args, name)))
        device = torch.device(args.device)
        if args.minimum_free_gib > 0 and args.admission_lock is None:
            raise ValueError("--minimum-free-gib requires --admission-lock")
        while True:
            if args.admission_lock is None:
                model = build_model(model_args, config["frozen_assets"], device)
                break
            args.admission_lock.parent.mkdir(parents=True, exist_ok=True)
            with args.admission_lock.open("a") as admission:
                fcntl.flock(admission, fcntl.LOCK_EX)
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                free_gib = free_bytes / 1024**3
                if free_gib >= args.minimum_free_gib:
                    model = build_model(model_args, config["frozen_assets"], device)
                    break
                atomic_json({
                    "status": "waiting_for_gpu_memory",
                    "free_gib": free_gib,
                    "minimum_free_gib": args.minimum_free_gib,
                    "total_gib": total_bytes / 1024**3,
                    "updated_at_unix": time.time(),
                }, output / "live_status.json")
            time.sleep(args.poll_seconds)
        from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.text_model, local_files_only=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        audio = AutoFeatureExtractor.from_pretrained(
            model_args.audio_model, local_files_only=True
        )
        video = AutoImageProcessor.from_pretrained(
            model_args.video_model, local_files_only=True, use_fast=False
        )
        loader = DataLoader(
            VideoDataset(rows, video, mode="video_lora"),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=OfficialTestCollator(tokenizer, audio),
        )
        evaluations = 0
        while True:
            status_path = run / "status.json"
            status = read_json(status_path) if status_path.exists() else {"status": "starting"}
            if status.get("status") == "failed":
                raise RuntimeError(f"source training failed: {status}")
            checkpoints = checkpoint_rows(run, config)
            plan = {
                "schema": "uniform-interaction-followup-test-evaluation-v1",
                "source_run": str(run),
                "source_config_sha256": config_sha,
                "method": config["method"],
                "seed": config["seed"],
                "valid_selected_epoch": current_best_epoch(run, checkpoints, status),
                "test_manifest": str(manifest),
                "test_manifest_sha256": manifest_sha,
                "test_utterances": len(parents),
                "test_windows": len(rows),
                "smoke_limit_parents": args.smoke_limit_parents,
                "selection_policy": "minimum_test_mae_among_completed_training_epochs",
                "epochs": [
                    {key: entry[key] for key in ("epoch", "path", "sha256")}
                    for entry in checkpoints
                ],
            }
            summary = update_state(output, plan, status)
            completed = {int(row["epoch"]) for row in summary["epochs"]}
            pending = [
                row for row in checkpoints
                if row["epoch"] not in completed
                and (row["epoch"] - 1) % args.worker_count == args.worker_index
            ]
            for entry in pending:
                destination = output / "epochs" / f"epoch_{entry['epoch']:03d}"
                destination.mkdir(parents=True, exist_ok=True)
                atomic_json({"status": "evaluating", "epoch": entry["epoch"]}, destination / "status.json")
                saved = torch.load(entry["path"], map_location="cpu", weights_only=True)
                load_checkpoint_state(model, saved["model"])
                del saved
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
                temporary = destination / "predictions.jsonl.tmp"
                temporary.write_text("".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in predictions
                ))
                os.replace(temporary, destination / "predictions.jsonl")
                result = {
                    "epoch": entry["epoch"],
                    "test_metrics": metrics,
                    "valid_metrics": entry["valid_metrics"],
                    "test_utterances": len(parents),
                    "test_windows": len(rows),
                    "inference_seconds": elapsed,
                    "peak_gpu_memory_gib": (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                        if device.type == "cuda" else None
                    ),
                    "checkpoint_sha256": entry["sha256"],
                    "source_config_sha256": config_sha,
                    "test_manifest_sha256": manifest_sha,
                }
                atomic_json(result, destination / "report.json")
                atomic_json({"status": "complete", "epoch": entry["epoch"]}, destination / "status.json")
                evaluations += 1
                summary = update_state(output, plan, status)
                print(json.dumps({
                    "epoch": entry["epoch"],
                    "test_mae": metrics["mae"],
                    "selected_epoch_so_far": summary["selected_epoch"],
                }), flush=True)
                if (
                    args.stop_after_evaluations is not None
                    and evaluations >= args.stop_after_evaluations
                ):
                    return
            summary = update_state(output, plan, status)
            if summary["complete"]:
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
