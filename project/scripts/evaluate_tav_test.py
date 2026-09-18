#!/usr/bin/env python3
"""One-shot official-test evaluation for a completed TAV M0--M6 run."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]
from train_student_baseline import seed_everything
from train_tav_distillation import (
    VideoDataset,
    batch_inputs,
    build_model,
    load_trainable,
    sha256,
)
from train_student_baseline import aggregate_windows
from rdid_mosei.metrics import sentiment_metrics


class OfficialTestCollator:
    """Build inference inputs without training-only teacher and loss fields."""

    def __init__(self, tokenizer, audio_processor):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor

    def __call__(self, items):
        rows = [item["row"] for item in items]
        text = self.tokenizer(
            [row["text"] for row in rows], padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        )
        audio = self.audio_processor(
            [item["wave"] for item in items], sampling_rate=16000, padding=True,
            return_attention_mask=True, return_tensors="pt",
        )
        return {
            "inputs": {
                "input_ids": text["input_ids"],
                "text_attention_mask": text["attention_mask"],
                "input_values": audio["input_values"],
                "audio_attention_mask": audio.get("attention_mask"),
                "pixel_values": torch.stack([item["pixels"] for item in items]),
            },
            "rows": rows,
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed train/valid run directory")
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--allow-official-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_official_test:
        parser.error("official test requires --allow-official-test after freezing the evaluated run list")
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("invalid batch size or worker count")
    return args


def atomic_json(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def load_test_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or {row["split"] for row in rows} != {"test"}:
        raise ValueError("official-test manifest must contain only test rows")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate official-test sample IDs")
    parents = {}
    for row in rows:
        signature = (
            float(row["sentiment"]), int(row["class_7_index"]), row["video_id"]
        )
        parent = row["parent_sample_id"]
        if parents.setdefault(parent, signature) != signature:
            raise ValueError(f"conflicting parent metadata: {parent}")
        if not math.isfinite(float(row["sentiment"])) or float(row["aggregation_weight"]) <= 0:
            raise ValueError(f"invalid target or aggregation weight: {row['sample_id']}")
    return rows, parents


@torch.inference_mode()
def evaluate_official(model, loader, device, status_path):
    model.eval()
    records = []
    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            outputs = model(**batch_inputs(batch, device))
        for row, prediction, logits in zip(
            batch["rows"], outputs["regression"].float().cpu().tolist(),
            outputs["classification_logits"].float().cpu().tolist(),
        ):
            records.append({
                "parent_sample_id": row["parent_sample_id"],
                "split": row["split"],
                "target_sentiment": row["sentiment"],
                "class_7_index": row["class_7_index"],
                "prediction": prediction,
                "classification_logits": logits,
                "aggregation_weight": row["aggregation_weight"],
            })
        if step == 1 or step % 50 == 0 or step == len(loader):
            atomic_json({
                "status": "evaluating", "step": step, "steps": len(loader),
                "windows_done": len(records),
                "elapsed_seconds": time.perf_counter() - started,
            }, status_path)
    utterances = aggregate_windows(records)
    video_ids = {row["parent_sample_id"]: row["video_id"] for row in loader.dataset.rows}
    for row in utterances:
        row["video_id"] = video_ids[row["parent_sample_id"]]
    metrics = sentiment_metrics(
        [row["target_sentiment"] for row in utterances],
        [row["prediction"] for row in utterances],
    )
    return metrics, utterances


def main():
    args = parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    manifest = args.test_manifest.resolve()
    if output.exists():
        raise FileExistsError(f"official-test output must be new: {output}")
    required = [run / name for name in ("run_config.json", "status.json", "report.json", "best.pt")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete source run: {missing}")
    status = json.loads((run / "status.json").read_text())
    config = json.loads((run / "run_config.json").read_text())
    report = json.loads((run / "report.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("only a completed train/valid run may enter official test")
    if config.get("protocol") != "tav-m0-m6-v1":
        raise ValueError("source run is not a TAV M0--M6 run")
    if config.get("checkpoint_selection") != "valid_mae" or config.get("official_test_evaluated"):
        raise ValueError("source run was not locked under the valid-only protocol")
    if report.get("official_test_evaluated") or report.get("method") != config.get("method"):
        raise ValueError("source report protocol differs")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("protocol") != config:
        raise ValueError("best checkpoint protocol differs from source run config")
    rows, parents = load_test_rows(manifest)
    protocol = {
        "schema": "rdid-msa-tav-official-test-v1",
        "source_run": str(run),
        "source_method": config["method"],
        "source_seed": config["seed"],
        "source_best_epoch": status["best_epoch"],
        "source_valid_mae": report["valid_metrics"]["mae"],
        "source_run_config_sha256": sha256(run / "run_config.json"),
        "checkpoint": str(run / "best.pt"),
        "checkpoint_sha256": sha256(run / "best.pt"),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "checkpoint_selection": "valid_mae",
        "official_test_evaluated": True,
        "test_parents": len(parents),
        "test_windows": len(rows),
        "test_use_policy": "report_only_never_for_configuration_or_checkpoint_selection",
    }
    if args.dry_run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return

    output.mkdir(parents=True)
    atomic_json(protocol, output / "run_config.json")
    atomic_json({"status": "evaluating"}, output / "status.json")
    try:
        seed_everything(int(config["seed"]), True)
        model_args = SimpleNamespace(**{
            key: value for key, value in config.items()
            if key not in {"frozen_assets", "input_sha256"}
        })
        for name in ("text_model", "audio_model", "video_model", "assets"):
            setattr(model_args, name, Path(getattr(model_args, name)))
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        model = build_model(model_args, device)
        load_trainable(model, checkpoint["model"])

        from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_args.text_model, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        audio_processor = AutoFeatureExtractor.from_pretrained(model_args.audio_model, local_files_only=True)
        video_processor = AutoImageProcessor.from_pretrained(
            model_args.video_model, local_files_only=True, use_fast=False
        )
        loader = DataLoader(
            VideoDataset(rows, video_processor, mode=model_args.mode),
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
            raise ValueError("official-test utterance aggregation coverage differs")
        temporary = output / "predictions.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
        os.replace(temporary, output / "predictions.jsonl")
        result = {
            "method": config["method"],
            "seed": config["seed"],
            "best_epoch": status["best_epoch"],
            "valid_metrics": report["valid_metrics"],
            "test_metrics": metrics,
            "checkpoint_selection": "valid_mae",
            "official_test_evaluated": True,
            "test_windows": len(rows),
            "test_utterances": len(predictions),
            "end_to_end_inference_seconds": elapsed,
            "end_to_end_seconds_per_utterance": elapsed / len(predictions),
            "latency_boundary": "manifest media decode, preprocessing, student forward, and window aggregation",
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
            ),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "teacher_required_at_inference": False,
            "seven_subset_ensemble_at_inference": False,
            "test_use_policy": protocol["test_use_policy"],
        }
        atomic_json(result, output / "report.json")
        atomic_json({"status": "complete"}, output / "status.json")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        atomic_json({"status": "failed", "error": repr(exc)}, output / "status.json")
        raise


if __name__ == "__main__":
    main()
