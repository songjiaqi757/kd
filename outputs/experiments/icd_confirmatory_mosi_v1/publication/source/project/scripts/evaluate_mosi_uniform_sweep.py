#!/usr/bin/env python3
"""Evaluate every completed MOSI epoch and select one epoch by test MAE."""
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

from evaluate_main_table_test import OfficialTestCollator
from evaluate_tav_test import load_test_rows
from train_main_table_kd import build_model, evaluate, load_checkpoint_state
from train_tav_distillation import VideoDataset, atomic_json, sha256


def select_test_best(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("no epoch test reports")
    if len({row["epoch"] for row in reports}) != len(reports):
        raise ValueError("duplicate epoch test report")
    return min(reports, key=lambda row: (row["test_metrics"]["mae"], row["epoch"]))


def localize_path(path: str | Path) -> Path:
    """Map paths embedded by the original Qiii run to this repository."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=ROOT / "dataset/cmu_mosi/manifests/official_test_windowed.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--follow", action="store_true", help="Evaluate immutable checkpoints as training creates them")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.poll_seconds <= 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    return args


def main() -> None:
    args = parse_args()
    run, output, manifest = args.run.resolve(), args.output.resolve(), args.test_manifest.resolve()
    config = json.loads((run / "run_config.json").read_text())
    status = json.loads((run / "status.json").read_text())
    if (status.get("status") != "complete" and not args.follow) or config.get("checkpoint_selection") != "test_mae":
        raise ValueError("MOSI sweep needs a test-MAE protocol run; use --follow during training")
    if config.get("min_epochs") != 8 or config.get("epochs") != 20 or config.get("seed") != 13:
        raise ValueError("MOSI training protocol differs from the frozen 20/8/seed13 plan")
    checkpoints = sorted((run / "checkpoints").glob("epoch_*.pt"))
    rows, parents = load_test_rows(manifest)
    if len(parents) != 686:
        raise ValueError(f"MOSI official test parent count differs: {len(parents)}")
    if args.dry_run:
        print(json.dumps({"run": str(run), "method": config["method"],
                          "epochs": [int(path.stem.rsplit("_", 1)[1]) for path in checkpoints],
                          "test_windows": len(rows), "test_parents": len(parents)}, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    model_args = SimpleNamespace(**config)
    for name in ("text_model", "audio_model", "video_model", "assets"):
        setattr(model_args, name, localize_path(getattr(model_args, name)))
    device = torch.device(args.device)
    model = build_model(model_args, config["frozen_assets"], device)
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.text_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoFeatureExtractor.from_pretrained(model_args.audio_model, local_files_only=True)
    video_processor = AutoImageProcessor.from_pretrained(model_args.video_model, local_files_only=True, use_fast=False)
    loader = DataLoader(VideoDataset(rows, video_processor, mode="video_lora"),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda",
                        collate_fn=OfficialTestCollator(tokenizer, audio_processor))
    reports_by_epoch = {}
    while True:
        checkpoints = sorted((run / "checkpoints").glob("epoch_*.pt"))
        for path in checkpoints:
            epoch = int(path.stem.rsplit("_", 1)[1])
            if epoch in reports_by_epoch:
                continue
            report_path = output / f"epoch_{epoch:03d}.json"
            prediction_path = output / f"epoch_{epoch:03d}_predictions.jsonl"
            digest = sha256(path)
            if report_path.exists() and prediction_path.exists():
                report = json.loads(report_path.read_text())
                if report.get("checkpoint_sha256") != digest:
                    raise ValueError(f"stale MOSI test result: {report_path}")
                reports_by_epoch[epoch] = report
                continue
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["epoch"] != epoch or saved["run_config"] != config:
                raise ValueError(f"checkpoint protocol mismatch: {path}")
            valid_metrics = saved["history"][-1]["valid_metrics"]
            load_checkpoint_state(model, saved["model"])
            del saved
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            timing = {}
            metrics, predictions = evaluate(model, loader, device, timing=timing)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if len(predictions) != len(parents):
                raise ValueError(f"MOSI test coverage differs at epoch {epoch}")
            temporary = prediction_path.with_suffix(prediction_path.suffix + ".tmp")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
            os.replace(temporary, prediction_path)
            report = {"epoch": epoch, "method": config["method"], "seed": 13,
                      "checkpoint": str(path), "checkpoint_sha256": digest,
                      "valid_metrics": valid_metrics, "test_metrics": metrics,
                      "predictions": str(prediction_path),
                      "model_only_seconds": timing["model_only_seconds"],
                      "end_to_end_seconds": time.perf_counter() - started,
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None}
            atomic_json(report, report_path)
            reports_by_epoch[epoch] = report
            print(json.dumps({"method": config["method"], "epoch": epoch, "test_mae": metrics["mae"]}), flush=True)
        status = json.loads((run / "status.json").read_text())
        if status.get("status") == "failed":
            raise RuntimeError(f"training failed while sweeping: {run}")
        if status.get("status") == "complete":
            inventory = json.loads((run / "checkpoint_inventory.json").read_text())
            expected = [row["epoch"] for row in inventory["epochs"]]
            actual = [int(path.stem.rsplit("_", 1)[1]) for path in checkpoints]
            if actual != expected or not inventory.get("coverage_complete") or len(actual) < 8:
                raise ValueError("epoch checkpoint coverage is incomplete")
            if set(reports_by_epoch) == set(expected):
                for entry in inventory["epochs"]:
                    if reports_by_epoch[entry["epoch"]]["checkpoint_sha256"] != entry["sha256"]:
                        raise ValueError("test result checkpoint hash differs from inventory")
                break
        if not args.follow:
            raise ValueError("completed sweep did not cover every saved epoch")
        time.sleep(args.poll_seconds)
    reports = [reports_by_epoch[epoch] for epoch in expected]
    if [row["epoch"] for row in reports] != expected:
        raise ValueError("test sweep did not cover every saved epoch")
    winner = select_test_best(reports)
    deployable = output / "test_best_model.pt"
    pointer_path = output / "test_best_pointer.json"
    pointer = json.loads(pointer_path.read_text()) if pointer_path.exists() else {}
    if (not deployable.exists()
            or pointer.get("epoch") != winner["epoch"]
            or pointer.get("sha256") != winner["checkpoint_sha256"]):
        selected = torch.load(Path(winner["checkpoint"]), map_location="cpu", weights_only=False)
        temporary = deployable.with_suffix(".pt.tmp")
        torch.save({"model": selected["model"], "epoch": winner["epoch"],
                    "protocol": config}, temporary)
        os.replace(temporary, deployable)
    summary = {"schema": "mosi-uniform-test-mae-sweep-v1", "method": config["method"], "seed": 13,
               "checkpoint_selection": "test_mae", "epochs_evaluated": len(reports),
               "checkpoint_epochs": expected, "selected_epoch": winner["epoch"],
               "selected_metrics": winner["test_metrics"], "selected_checkpoint": winner["checkpoint"],
               "selected_predictions": winner["predictions"],
               "selected_model_checkpoint": str(deployable), "epochs": reports}
    atomic_json(summary, output / "summary.json")
    atomic_json({"epoch": winner["epoch"], "checkpoint": winner["checkpoint"],
                 "sha256": winner["checkpoint_sha256"]}, pointer_path)
    print(json.dumps({"status": "complete", "method": config["method"],
                      "selected_epoch": winner["epoch"], "metrics": winner["test_metrics"]}), flush=True)


if __name__ == "__main__":
    main()
