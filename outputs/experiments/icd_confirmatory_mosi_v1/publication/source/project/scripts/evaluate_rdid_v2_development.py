#!/usr/bin/env python3
"""Evaluate a valid-selected RDID-v2 checkpoint on MOSEI development test."""
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

from evaluate_tav_test import OfficialTestCollator, evaluate_official, load_test_rows
from train_main_table_kd import load_checkpoint_state
from train_rdid_v2 import build_model
from train_student_baseline import seed_everything
from train_tav_distillation import VideoDataset, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--test-manifest", type=Path,
        default=ROOT / "dataset/cmu_mosei/manifests/official_test_windowed.jsonl",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--allow-mosei-development-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_mosei_development_test:
        parser.error("explicit --allow-mosei-development-test is required")
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("invalid loader settings")
    return args


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    manifest = args.test_manifest.resolve()
    required = [run / name for name in (
        "run_config.json", "status.json", "report.json", "best.pt", "checkpoint_inventory.json"
    )]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete source run: {missing}")
    status = json.loads((run / "status.json").read_text())
    config = json.loads((run / "run_config.json").read_text())
    report = json.loads((run / "report.json").read_text())
    inventory = json.loads((run / "checkpoint_inventory.json").read_text())
    if status.get("status") != "complete" or status.get("checkpoint_audit") != "pass":
        raise ValueError("source run must be complete with a passing checkpoint audit")
    if config.get("schema") != "rdid-v2-mosei-development-train-v1":
        raise ValueError("source run is not RDID-v2")
    if config.get("checkpoint_selection") != "valid_mae" or report.get("official_test_evaluated"):
        raise ValueError("source run did not preserve valid-only checkpoint selection")
    if int(config["seed"]) != 13:
        raise ValueError("RDID-v2 development source must use seed13")
    if not inventory.get("coverage_complete"):
        raise ValueError("source checkpoint inventory is incomplete")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["run_config"] != config or checkpoint["epoch"] != status["best_epoch"]:
        raise ValueError("best checkpoint does not match the source protocol/status")
    rows, parents = load_test_rows(manifest)
    development_protocol = {
        "schema": "rdid-v2-mosei-method-development-evaluation-v1",
        "source_run": str(run),
        "source_method": config["method"],
        "source_seed": config["seed"],
        "source_best_epoch": status["best_epoch"],
        "source_valid_mae": report["valid_metrics"]["mae"],
        "source_run_config_sha256": sha256(run / "run_config.json"),
        "checkpoint": str((run / "best.pt").resolve()),
        "checkpoint_sha256": sha256(run / "best.pt"),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "checkpoint_selection": "valid_mae",
        "test_parents": len(parents),
        "test_windows": len(rows),
        "test_use_policy": "method_level_development_only_never_epoch_selection",
        "mosei_is_blind_confirmation": False,
    }
    if args.dry_run:
        print(json.dumps(development_protocol, ensure_ascii=False, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"development evaluation output must be new: {output}")
    output.mkdir(parents=True)
    atomic_json(development_protocol, output / "run_config.json")
    atomic_json({"status": "evaluating"}, output / "status.json")
    try:
        seed_everything(13, True)
        model_args = SimpleNamespace(**{
            key: value for key, value in config.items()
            if key not in {"frozen_assets", "input_sha256"}
        })
        for name in ("text_model", "audio_model", "video_model", "assets"):
            setattr(model_args, name, Path(getattr(model_args, name)))
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        model = build_model(model_args, config["frozen_assets"], device)
        load_checkpoint_state(model, checkpoint["model_state_dict"])

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
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=OfficialTestCollator(tokenizer, audio_processor),
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        metrics, predictions = evaluate_official(model, loader, device, output / "status.json")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if len(predictions) != len(parents):
            raise ValueError("test utterance coverage differs")
        temporary = output / "predictions.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
        os.replace(temporary, output / "predictions.jsonl")
        valid_mae = float(report["valid_metrics"]["mae"])
        test_mae = float(metrics["mae"])
        result = {
            "schema": "rdid-v2-mosei-method-development-report-v1",
            "method": config["method"],
            "seed": 13,
            "best_epoch": status["best_epoch"],
            "valid_metrics": report["valid_metrics"],
            "test_metrics": metrics,
            "generalization_gap_mae": test_mae - valid_mae,
            "development_score_max_mae": max(valid_mae, test_mae),
            "checkpoint_selection": "valid_mae",
            "test_use_policy": development_protocol["test_use_policy"],
            "mosei_is_blind_confirmation": False,
            "end_to_end_inference_seconds": elapsed,
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
            ),
        }
        atomic_json(result, output / "report.json")
        atomic_json({"status": "complete"}, output / "status.json")
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        atomic_json({"status": "failed", "error": repr(error)}, output / "status.json")
        raise


if __name__ == "__main__":
    main()
