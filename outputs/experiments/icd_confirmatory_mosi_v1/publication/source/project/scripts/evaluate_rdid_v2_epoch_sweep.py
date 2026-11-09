#!/usr/bin/env python3
"""Exploratory MOSEI-test sweep of every RDID-v2 epoch checkpoint.

This intentionally uses test data at the epoch level. Its results are not a
blind confirmation and must not be confused with valid-selected development
reports. One process handles one GPU and follows its assigned training runs.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]

from evaluate_tav_test import OfficialTestCollator, evaluate_official, load_test_rows
from train_main_table_kd import load_checkpoint_state
from train_rdid_v2 import build_model
from train_student_baseline import seed_everything
from train_tav_distillation import VideoDataset, sha256

METHODS = (
    "soft_ru_a025", "soft_ru_a050", "soft_ru_a075",
    "r_only", "u_only", "amplitude_u",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--worker-id", help="Unique ID when several evaluators share one GPU")
    parser.add_argument("--train-root", type=Path, default=ROOT / "outputs/experiments/rdid_v2_mosei/students")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/experiments/rdid_v2_mosei/epoch_test_sweep_20260919")
    parser.add_argument("--test-manifest", type=Path, default=ROOT / "dataset/cmu_mosei/manifests/official_test_windowed.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--allow-mosei-test-all-epochs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_mosei_test_all_epochs:
        parser.error("explicit --allow-mosei-test-all-epochs is required")
    if args.batch_size <= 0 or args.num_workers < 0 or args.poll_seconds <= 0:
        parser.error("batch size and poll interval must be positive; workers must be nonnegative")
    if len(args.methods) != len(set(args.methods)):
        parser.error("duplicate method")
    if args.worker_id is not None and not re.fullmatch(r"[a-zA-Z0-9_]+", args.worker_id):
        parser.error("worker-id must contain only letters, digits, and underscores")
    return args


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def model_signature(config: dict) -> dict:
    names = (
        "text_model", "audio_model", "video_model", "seed", "video_layers",
        "video_rank", "video_alpha", "video_dropout", "video_checkpointing",
    )
    return {name: config[name] for name in names} | {"frozen_assets": config["frozen_assets"]}


def run_config(train_root: Path, method: str) -> tuple[Path, dict] | None:
    run = train_root / f"{method}_seed13"
    config = read_json(run / "run_config.json")
    if config is None:
        return None
    if (
        config.get("schema") != "rdid-v2-mosei-development-train-v1"
        or config.get("method") != method
        or config.get("seed") != 13
        or config.get("checkpoint_selection") != "valid_mae"
    ):
        raise ValueError(f"unexpected training protocol: {run}")
    return run, config


def complete_epoch(output: Path) -> bool:
    status = read_json(output / "status.json")
    return bool(status and status.get("status") == "complete" and (output / "report.json").is_file())


def summary(output_root: Path, methods: list[str], train_root: Path, gpu: int) -> dict:
    rows = []
    for method in methods:
        run = train_root / f"{method}_seed13"
        train_status = read_json(run / "status.json") or {"status": "not_started"}
        checkpoints = sorted((run / "checkpoints").glob("epoch_*.pt"))
        done = []
        for path in sorted((output_root / method).glob("epoch_*/report.json")):
            report = json.loads(path.read_text())
            done.append({
                "epoch": report["epoch"],
                "valid_mae": report["valid_metrics"]["mae"],
                "test_mae": report["test_metrics"]["mae"],
                "report": str(path.resolve()),
            })
        rows.append({
            "method": method,
            "train_status": train_status.get("status"),
            "available_checkpoints": len(checkpoints),
            "evaluated_checkpoints": len(done),
            "epochs": done,
        })
    complete = all(
        row["train_status"] == "complete"
        and row["available_checkpoints"] == row["evaluated_checkpoints"]
        for row in rows
    )
    return {
        "schema": "rdid-v2-mosei-exploratory-epoch-test-summary-v1",
        "status": "complete" if complete else "running",
        "gpu": gpu,
        "test_use_policy": "exploratory_all_epoch_test_not_blind_confirmation",
        "mosei_is_blind_confirmation": False,
        "methods": rows,
    }


def evaluate_epoch(
    *, run: Path, config: dict, method: str, epoch: int, checkpoint_path: Path,
    output: Path, model: torch.nn.Module, loader: DataLoader, device: torch.device,
    manifest: Path, manifest_sha256: str, parent_count: int,
) -> None:
    if complete_epoch(output):
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("epoch") != epoch
        or checkpoint.get("run_config") != config
        or checkpoint.get("method_config", {}).get("method") != method
    ):
        raise ValueError(f"checkpoint does not match training run: {checkpoint_path}")
    history = checkpoint["training_history_so_far"]
    if not history or history[-1]["epoch"] != epoch:
        raise ValueError(f"checkpoint history does not reach epoch {epoch}")
    valid_metrics = history[-1]["valid_metrics"]
    checkpoint_digest = sha256(checkpoint_path)
    load_checkpoint_state(model, checkpoint["model_state_dict"])
    del checkpoint
    output.mkdir(parents=True, exist_ok=True)
    atomic_json({
        "schema": "rdid-v2-mosei-exploratory-epoch-test-config-v1",
        "source_run": str(run.resolve()),
        "method": method,
        "seed": 13,
        "epoch": epoch,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "source_run_config_sha256": sha256(run / "run_config.json"),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "valid_metrics": valid_metrics,
        "test_use_policy": "exploratory_all_epoch_test_not_blind_confirmation",
        "mosei_is_blind_confirmation": False,
    }, output / "run_config.json")
    atomic_json({"status": "evaluating", "method": method, "epoch": epoch}, output / "status.json")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    metrics, predictions = evaluate_official(model, loader, device, output / "status.json")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if len(predictions) != parent_count:
        raise ValueError(f"test utterance coverage differs: {method} epoch {epoch}")
    temporary = output / "predictions.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    os.replace(temporary, output / "predictions.jsonl")
    report = {
        "schema": "rdid-v2-mosei-exploratory-epoch-test-report-v1",
        "method": method,
        "seed": 13,
        "epoch": epoch,
        "valid_metrics": valid_metrics,
        "test_metrics": metrics,
        "generalization_gap_mae": metrics["mae"] - valid_metrics["mae"],
        "test_use_policy": "exploratory_all_epoch_test_not_blind_confirmation",
        "mosei_is_blind_confirmation": False,
        "end_to_end_inference_seconds": time.perf_counter() - started,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
        ),
    }
    atomic_json(report, output / "report.json")
    atomic_json({"status": "complete", "method": method, "epoch": epoch}, output / "status.json")
    print(json.dumps({"status": "complete", "method": method, "epoch": epoch,
                      "valid_mae": valid_metrics["mae"], "test_mae": metrics["mae"]}), flush=True)


def main() -> None:
    args = parse_args()
    train_root = args.train_root.resolve()
    output_root = args.output_root.resolve()
    manifest = args.test_manifest.resolve()
    rows, parents = load_test_rows(manifest)
    manifest_digest = sha256(manifest)
    if args.dry_run:
        print(json.dumps({
            "gpu": args.gpu, "methods": args.methods, "test_windows": len(rows),
            "test_parents": len(parents), "output_root": str(output_root),
            "test_use_policy": "exploratory_all_epoch_test_not_blind_confirmation",
        }, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    identity = f"worker_{args.worker_id}" if args.worker_id else f"gpu{args.gpu}"
    summary_path = output_root / f"{identity}_summary.json"
    with (output_root / f".{identity}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seed_everything(13, True)
        device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        model = None
        original_signature = None
        loader = None
        for method in args.methods:
            while True:
                source = run_config(train_root, method)
                if source is None:
                    print(json.dumps({"status": "waiting_for_run", "method": method}), flush=True)
                    time.sleep(args.poll_seconds)
                    continue
                run, config = source
                if model is None:
                    model_args = SimpleNamespace(**{
                        key: value for key, value in config.items()
                        if key not in {"frozen_assets", "input_sha256"}
                    })
                    for name in ("text_model", "audio_model", "video_model", "assets"):
                        setattr(model_args, name, Path(getattr(model_args, name)))
                    original_signature = model_signature(config)
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
                        pin_memory=True, persistent_workers=args.num_workers > 0,
                        collate_fn=OfficialTestCollator(tokenizer, audio_processor),
                    )
                elif model_signature(config) != original_signature:
                    raise ValueError(f"model architecture/assets differ for {method}")

                checkpoints = sorted((run / "checkpoints").glob("epoch_*.pt"))
                pending = [path for path in checkpoints
                           if not complete_epoch(output_root / method / path.stem)]
                if pending:
                    path = pending[0]
                    epoch = int(path.stem.rsplit("_", 1)[1])
                    output = output_root / method / path.stem
                    print(json.dumps({"status": "starting", "method": method, "epoch": epoch,
                                      "gpu": args.gpu}), flush=True)
                    try:
                        evaluate_epoch(
                            run=run, config=config, method=method, epoch=epoch,
                            checkpoint_path=path, output=output, model=model, loader=loader,
                            device=device, manifest=manifest, manifest_sha256=manifest_digest,
                            parent_count=len(parents),
                        )
                    except Exception as error:
                        atomic_json({"status": "failed", "method": method, "epoch": epoch,
                                     "error": repr(error)}, output / "status.json")
                        raise
                    atomic_json(summary(output_root, args.methods, train_root, args.gpu),
                                summary_path)
                    continue

                train_status = read_json(run / "status.json") or {}
                if train_status.get("status") == "complete":
                    break
                if train_status.get("status") == "failed":
                    raise RuntimeError(f"training run failed before sweep completion: {run}")
                atomic_json(summary(output_root, args.methods, train_root, args.gpu),
                            summary_path)
                time.sleep(args.poll_seconds)
        final = summary(output_root, args.methods, train_root, args.gpu)
        atomic_json(final, summary_path)
        print(json.dumps({"status": final["status"], "gpu": args.gpu,
                          "methods": args.methods}), flush=True)


if __name__ == "__main__":
    main()
