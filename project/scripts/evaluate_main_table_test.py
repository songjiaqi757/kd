#!/usr/bin/env python3
"""One-shot official-test evaluation for a locked fixed-student checkpoint."""
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
from train_tav_distillation import VideoDataset, sha256
from train_main_table_kd import build_model, evaluate, load_checkpoint_state
from rdid_mosei.main_table_kd import teacher_supervision_metadata


class OfficialTestCollator:
    def __init__(self, tokenizer, audio_processor):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor

    def __call__(self, items):
        rows = [item["row"] for item in items]
        text = self.tokenizer([row["text"] for row in rows], padding=True, truncation=True, max_length=256, return_tensors="pt")
        audio = self.audio_processor([item["wave"] for item in items], sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
        return {
            "inputs": {
                "input_ids": text["input_ids"], "text_attention_mask": text["attention_mask"],
                "input_values": audio["input_values"], "audio_attention_mask": audio.get("attention_mask"),
                "pixel_values": torch.stack([item["pixels"] for item in items]),
            },
            "rows": rows,
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed train/valid run directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--allow-official-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_official_test:
        parser.error("official test requires --allow-official-test after configuration lock")
    return args


def main():
    args = parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"official-test output must be new: {output}")
    status = json.loads((run / "status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("only a completed train/valid run may enter final test")
    config = json.loads((run / "run_config.json").read_text())
    if config.get("official_test_evaluated") or config.get("checkpoint_selection") != "valid_mae":
        raise ValueError("run was not locked under the valid-only protocol")
    rows = [json.loads(line) for line in args.test_manifest.read_text().splitlines() if line.strip()]
    if {row["split"] for row in rows} != {"test"}:
        raise ValueError("official-test manifest must contain only test")
    parents = {row["parent_sample_id"] for row in rows}
    if not parents or any(float(row["aggregation_weight"]) <= 0 for row in rows):
        raise ValueError("invalid official-test window manifest")

    protocol = {
        "schema": "rdid-msa-official-test-v1",
        "source_run": str(run),
        "source_run_config_sha256": sha256(run / "run_config.json"),
        "checkpoint": str(run / "best.pt"),
        "checkpoint_sha256": sha256(run / "best.pt"),
        "manifest": str(args.test_manifest.resolve()),
        "manifest_sha256": sha256(args.test_manifest),
        "checkpoint_selection": "valid_mae",
        "official_test_evaluated": True,
        "test_parents": len(parents),
        "test_windows": len(rows),
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    output.mkdir(parents=True)
    (output / "run_config.json").write_text(json.dumps(protocol, indent=2) + "\n")

    model_args = SimpleNamespace(**{key: value for key, value in config.items() if key not in {"frozen_assets", "upstream_sources", "input_sha256"}})
    for name in ("text_model", "audio_model", "video_model", "assets"):
        setattr(model_args, name, Path(getattr(model_args, name)))
    device = torch.device(args.device)
    assets = config["frozen_assets"]
    model = build_model(model_args, assets, device)
    load_checkpoint_state(model, torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.text_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoFeatureExtractor.from_pretrained(model_args.audio_model, local_files_only=True)
    video_processor = AutoImageProcessor.from_pretrained(model_args.video_model, local_files_only=True, use_fast=False)
    loader = DataLoader(
        VideoDataset(rows, video_processor, mode="video_lora"), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
        collate_fn=OfficialTestCollator(tokenizer, audio_processor),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    timing = {}
    metrics, predictions = evaluate(model, loader, device, timing=timing)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if len(predictions) != len(parents):
        raise ValueError("official-test utterance aggregation coverage differs")
    temporary = output / "predictions.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    os.replace(temporary, output / "predictions.jsonl")
    teacher_metadata = teacher_supervision_metadata(model_args.method)
    training_only_projector_parameters = 0 if model.feature_loss is None else sum(
        parameter.numel() for parameter in model.feature_loss.parameters()
    )
    report = {
        "method": model_args.method, "seed": model_args.seed, "metrics": metrics,
        "checkpoint_selection": "valid_mae", "official_test_evaluated": True,
        "model_only_inference_seconds": timing["model_only_seconds"],
        "model_only_seconds_per_window": timing["model_only_seconds"] / timing["model_windows"],
        "model_only_seconds_per_utterance_equivalent": timing["model_only_seconds"] / len(parents),
        "model_only_latency_boundary": "preprocessed host tensors through student forward; includes host-to-device transfer; excludes media decode, tokenization, processor transforms, DataLoader wait, and utterance aggregation",
        "end_to_end_inference_seconds": elapsed,
        "end_to_end_seconds_per_utterance": elapsed / len(parents),
        "end_to_end_latency_boundary": "official manifest DataLoader iteration including media decode, processor transforms, student forward, and window-to-utterance aggregation",
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
        "deployed_parameters": sum(parameter.numel() for parameter in model.parameters()) - training_only_projector_parameters,
        "training_only_projector_parameters": training_only_projector_parameters,
        "teacher_required_at_inference": False,
        "seven_subset_ensemble_at_inference": False,
        **teacher_metadata,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "status.json").write_text(json.dumps({"status": "complete"}, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
