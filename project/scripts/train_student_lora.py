#!/usr/bin/env python3
"""Train Stage-D LoRA(T+A) students while reusing frozen VideoMAE features."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.metrics import sentiment_metrics
from rdid_mosei.student import LoRATextAudioCachedVideoStudent
from train_student_baseline import (
    MULTI_SUBSET_METHODS,
    add_window_weights,
    aggregate_windows,
    attach_teacher_targets,
    combined_loss,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage D attribution methods with LoRA on text and audio")
    parser.add_argument("--features", type=Path, default=Path("outputs/student/features/official_train_valid"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, default=Path("model/Qwen3-0.6B-Base"))
    parser.add_argument("--audio-model", type=Path, default=Path("model/WavLM-Base-Plus"))
    parser.add_argument("--teacher-targets", type=Path, default=Path("outputs/probe/official_train_valid_seed2026/predictions.jsonl"))
    parser.add_argument("--teacher-targets-ensemble", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--method",
        choices=("student", "full_kd", "ensemble_pair", "reliability_utility_pair"),
        default="full_kd",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=1, help="Physical batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha-ce", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lambda-full", type=float, default=1.0)
    parser.add_argument("--lambda-subset", type=float, default=1.0)
    parser.add_argument("--lambda-coordinate", type=float, default=1.0)
    # Compatibility with the polarity branch's shared loss. Stage D does not
    # train a binary head, so both terms remain exactly zero.
    parser.add_argument("--lambda-binary-full", type=float, default=0.0)
    parser.add_argument("--lambda-binary-subset", type=float, default=0.0)
    parser.add_argument("--lambda-kd-regression", type=float, default=1.0)
    parser.add_argument("--lambda-kd-classification", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-calibration-temperature", type=float, default=0.9259549975395203)
    parser.add_argument("--reliability-epsilon", type=float, default=1e-4)
    parser.add_argument("--reliability-w-min", type=float, default=0.25)
    parser.add_argument("--reliability-w-max", type=float, default=4.0)
    parser.add_argument("--selective-keep-fraction", type=float, default=0.5)
    parser.add_argument("--limit-per-split", type=int, help="Smoke/throughput runs only")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--freeze-text-audio",
        action="store_true",
        help="Keep the zero-initialized T/A adapters frozen for the online-path control",
    )
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class OnlineDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, sample_rate = sf.read(row["audio_segment_path"], dtype="float32", always_2d=False)
        if waveform.ndim != 1 or sample_rate != 16_000:
            raise ValueError(f"invalid audio for {row['sample_id']}: shape={waveform.shape}, rate={sample_rate}")
        cached = torch.load(row["feature_path"], map_location="cpu", weights_only=True)
        return {"row": row, "waveform": waveform, "video": cached["v"]}


class OnlineCollator:
    def __init__(self, tokenizer: Any, audio_processor: Any, max_text_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.max_text_tokens = max_text_tokens

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [item["row"] for item in items]
        text = self.tokenizer(
            [row["text"] for row in rows], padding=True, truncation=True,
            max_length=self.max_text_tokens, return_tensors="pt",
        )
        audio = self.audio_processor(
            [item["waveform"] for item in items], sampling_rate=16_000, padding=True,
            return_attention_mask=True, return_tensors="pt",
        )
        video_lengths = torch.tensor([item["video"].shape[0] for item in items])
        video = pad_sequence([item["video"] for item in items], batch_first=True)
        video_mask = torch.arange(video.shape[1]).unsqueeze(0) < video_lengths.unsqueeze(1)
        return {
            "input_ids": text["input_ids"], "text_attention_mask": text["attention_mask"],
            "input_values": audio["input_values"], "audio_attention_mask": audio.get("attention_mask"),
            "video_hidden_states": video, "video_attention_mask": video_mask,
            "sentiment": torch.tensor([row["sentiment"] for row in rows], dtype=torch.float32),
            "classes": torch.tensor([row["class_7_index"] for row in rows], dtype=torch.long),
            "weights": torch.tensor([row["sample_weight"] for row in rows], dtype=torch.float32),
            "teacher_scores": torch.tensor([row["teacher_score"] for row in rows], dtype=torch.float32),
            "teacher_logits": torch.tensor([row["teacher_logits"] for row in rows], dtype=torch.float32),
            "teacher_subset_scores": torch.tensor([row["teacher_subset_scores"] for row in rows], dtype=torch.float32),
            "teacher_interaction_mean": torch.tensor([row.get("teacher_interaction_mean", [math.nan] * 7) for row in rows]),
            "teacher_interaction_var": torch.tensor([row.get("teacher_interaction_var", [math.nan] * 7) for row in rows]),
            "rows": rows,
        }


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cached_rows = [json.loads(line) for line in (args.features / "index.jsonl").read_text().splitlines() if line]
    manifest_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    manifest = {str(row["sample_id"]): row for row in manifest_rows}
    if len(cached_rows) != len(manifest_rows):
        raise RuntimeError(f"cache/manifest count mismatch: {len(cached_rows)} != {len(manifest_rows)}")
    rows = []
    for cached in cached_rows:
        raw = manifest.get(str(cached["sample_id"]))
        if raw is None or raw["split"] != cached["split"]:
            raise RuntimeError(f"cache/manifest alignment failure at {cached['sample_id']}")
        rows.append({**raw, **cached})
    add_window_weights(rows)
    attach_teacher_targets(rows, args.teacher_targets, args.teacher_targets_ensemble)
    train = [row for row in rows if row["split"] == "train"]
    valid = [row for row in rows if row["split"] == "valid"]
    if args.limit_per_split:
        train, valid = train[: args.limit_per_split], valid[: args.limit_per_split]
    if not train or not valid:
        raise RuntimeError("official train and valid rows are both required")
    return train, valid


def configure_method(args: argparse.Namespace, train_rows: list[dict[str, Any]]) -> None:
    args.coordinate_center = np.mean([row["teacher_subset_scores"] for row in train_rows], axis=0).tolist()
    args.coordinate_scale = [1.0] * 7
    args.selective_threshold = None
    args.interaction_utility = None
    args.interaction_utility_normalized = [1.0, 1.0, 1.0]
    if args.method in ("ensemble_pair", "reliability_utility_pair"):
        if not args.teacher_targets_ensemble or len(args.teacher_targets_ensemble) < 2:
            raise ValueError(f"{args.method} requires at least two ensemble teacher target files")
    if args.method == "reliability_utility_pair":
        parents = {}
        for row in train_rows:
            parents.setdefault(str(row["parent_sample_id"]), row)
        targets = np.asarray([row["sentiment"] for row in parents.values()])
        interactions = np.asarray([row["teacher_interaction_mean"][3:6] for row in parents.values()])
        utility = np.asarray([abs(np.corrcoef(interactions[:, i], targets)[0, 1]) for i in range(3)])
        if not np.isfinite(utility).all() or utility.mean() <= 0:
            raise RuntimeError(f"invalid train-only utilities: {utility.tolist()}")
        args.interaction_utility = utility.tolist()
        args.interaction_utility_normalized = (utility / utility.mean()).tolist()


def move_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor | None]:
    keys = ("input_ids", "text_attention_mask", "input_values", "audio_attention_mask", "video_hidden_states", "video_attention_mask")
    result = {}
    for key in keys:
        value = batch[key]
        result[key] = None if value is None else value.to(device, non_blocking=True)
    return result


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace):
    model.eval()
    records, loss_sum, weight_sum = [], 0.0, 0.0
    requested = ("t", "a", "v", "ta", "tv", "av", "tav") if args.method in MULTI_SUBSET_METHODS else ("tav",)
    for batch in loader:
        inputs = move_inputs(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(**inputs, subsets=requested)
            loss, _ = combined_loss(outputs, batch, device, args)
        weights = batch["weights"].to(device)
        loss_sum += float(loss) * float(weights.sum())
        weight_sum += float(weights.sum())
        prediction = outputs["tav"]["regression"].float().cpu().tolist()
        logits = outputs["tav"]["classification_logits"].float().cpu().tolist()
        for row, score, class_logits in zip(batch["rows"], prediction, logits):
            records.append({
                "sample_id": row["sample_id"], "parent_sample_id": row["parent_sample_id"], "split": row["split"],
                "target_sentiment": float(row["sentiment"]), "class_7_index": int(row["class_7_index"]),
                "aggregation_weight": float(row["aggregation_weight"]), "prediction": float(score),
                "classification_logits": class_logits,
            })
    utterances = aggregate_windows(records)
    metrics = sentiment_metrics([x["target_sentiment"] for x in utterances], [x["prediction"] for x in utterances])
    metrics["classification_accuracy"] = float(np.mean([np.argmax(x["classification_logits"]) == x["class_7_index"] for x in utterances]))
    return metrics, loss_sum / weight_sum, records


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in names}


def main() -> int:
    args = parse_args()
    seed_everything(args.seed, args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    config = json.loads((args.features / "run_config.json").read_text())
    if config.get("status") != "complete":
        raise RuntimeError("frozen feature cache is incomplete")
    train_rows, valid_rows = load_rows(args)
    configure_method(args, train_rows)

    from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoFeatureExtractor.from_pretrained(args.audio_model, local_files_only=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    text_encoder = AutoModel.from_pretrained(args.text_model, local_files_only=True, dtype=dtype)
    audio_encoder = AutoModel.from_pretrained(args.audio_model, local_files_only=True, dtype=dtype)
    if args.gradient_checkpointing and not args.freeze_text_audio:
        for encoder in (text_encoder, audio_encoder):
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(encoder.config, "use_cache"):
                encoder.config.use_cache = False
    model = LoRATextAudioCachedVideoStudent(
        text_encoder, audio_encoder, video_hidden_size=int(config["hidden_sizes"]["v"]),
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    ).to(device)
    if args.freeze_text_audio:
        model.text_encoder.requires_grad_(False)
        model.audio_encoder.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    collator = OnlineCollator(tokenizer, audio_processor, args.max_text_tokens)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collator, pin_memory=device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(OnlineDataset(train_rows), shuffle=True, generator=generator, **loader_kwargs)
    valid_loader = DataLoader(OnlineDataset(valid_rows), shuffle=False, **loader_kwargs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(json.dumps({**vars(args), "features": str(args.features), "manifest": str(args.manifest), "output": str(args.output), "text_model": str(args.text_model), "audio_model": str(args.audio_model), "teacher_targets": str(args.teacher_targets), "teacher_targets_ensemble": [str(x) for x in args.teacher_targets_ensemble] if args.teacher_targets_ensemble else None, "total_parameters": total, "trainable_parameters": trainable, "text_lora_modules": model.text_lora_modules, "audio_lora_modules": model.audio_lora_modules}, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"status": "initialized", "method": args.method, "seed": args.seed, "train_windows": len(train_rows), "valid_windows": len(valid_rows), "total_parameters": total, "trainable_parameters": trainable}), flush=True)

    best_mae, best_epoch, stale, history = math.inf, 0, 0, []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    requested = ("t", "a", "v", "ta", "tv", "av", "tav") if args.method in MULTI_SUBSET_METHODS else ("tav",)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = weight_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            inputs = move_inputs(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(**inputs, subsets=requested)
                loss, components = combined_loss(outputs, batch, device, args)
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                gradient_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_weight = float(batch["weights"].sum())
            loss_sum += float(loss.detach()) * batch_weight
            weight_sum += batch_weight
        valid_metrics, valid_loss, _ = evaluate(model, valid_loader, device, args)
        item = {"epoch": epoch, "train_loss": loss_sum / weight_sum, "valid_loss": valid_loss, "valid_mae": valid_metrics["mae"], "valid_pearson": valid_metrics["pearson"], "valid_acc2_nonzero": valid_metrics["acc2_nonzero"], "gradient_norm_last": float(torch.as_tensor(gradient_norm)), "task_loss_last": components["task"], "full_kd_loss_last": components["full_kd"], "coordinate_loss_last": components["coordinate"], "elapsed_seconds": round(time.time() - started, 1)}
        history.append(item)
        print(json.dumps(item), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if valid_metrics["mae"] < best_mae - 1e-5:
            best_mae, best_epoch, stale = float(valid_metrics["mae"]), epoch, 0
            torch.save({"epoch": epoch, "model": trainable_state_dict(model), "valid_metrics": valid_metrics}, args.output / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break

    checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    train_metrics, train_loss, train_records = evaluate(model, DataLoader(OnlineDataset(train_rows), shuffle=False, **loader_kwargs), device, args)
    valid_metrics, valid_loss, valid_records = evaluate(model, valid_loader, device, args)
    with (args.output / "predictions.jsonl").open("w") as handle:
        for item in aggregate_windows(train_records + valid_records):
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    experiment_names = {
        "student": "C0_task_only_lora_ta",
        "full_kd": "D1_full_kd_lora_ta",
        "ensemble_pair": "C2_uniform_ensemble_pair_lora_ta",
        "reliability_utility_pair": "D2_ru_lora_ta",
    }
    experiment = (
        "C_minus1_online_frozen_full_kd"
        if args.freeze_text_audio and args.method == "full_kd"
        else experiment_names[args.method]
    )
    report = {"experiment": experiment, "method": args.method, "freeze_text_audio": args.freeze_text_audio, "seed": args.seed, "best_epoch": best_epoch, "epochs_run": len(history), "train_metrics": train_metrics, "valid_metrics": valid_metrics, "train_loss": train_loss, "valid_loss": valid_loss, "train_windows": len(train_rows), "valid_windows": len(valid_rows), "base_parameters": total - trainable, "trainable_parameters": trainable, "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "trainable": not args.freeze_text_audio, "text_targets": ["q_proj", "k_proj", "v_proj", "o_proj"], "audio_targets": ["q_proj", "k_proj", "v_proj", "out_proj"], "video": "frozen_cached"}, "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3) if device.type == "cuda" else None, "elapsed_seconds": round(time.time() - started, 1), "official_test_evaluated": False}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
