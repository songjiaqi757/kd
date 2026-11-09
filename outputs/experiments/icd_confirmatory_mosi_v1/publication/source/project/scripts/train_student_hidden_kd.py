#!/usr/bin/env python3
"""Stage D3: Full KD plus 2048-d teacher representation KD on cached encoders."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.metrics import sentiment_metrics
from rdid_mosei.student import CachedStudentCore
from train_student_baseline import (
    FeatureDataset,
    add_window_weights,
    aggregate_windows,
    attach_teacher_targets,
    collate_features,
    combined_loss,
    seed_everything,
    to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D3 Full KD + cosine hidden KD")
    parser.add_argument("--features", type=Path, default=Path("outputs/student/features/official_train_valid"))
    parser.add_argument("--teacher-features", type=Path, default=Path("outputs/probe/features/official_train_valid"))
    parser.add_argument("--teacher-targets", type=Path, default=Path("outputs/probe/official_train_valid_seed2026/predictions.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--alpha-ce", type=float, default=0.5)
    parser.add_argument("--lambda-full", type=float, default=1.0)
    parser.add_argument("--lambda-hidden", type=float, default=1.0)
    parser.add_argument("--lambda-kd-regression", type=float, default=1.0)
    parser.add_argument("--lambda-kd-classification", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-calibration-temperature", type=float, default=0.9259549975395203)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class HiddenFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], hidden_path: Path) -> None:
        self.rows = rows
        self.hidden_path = hidden_path
        self.hidden: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.hidden is None:
            self.hidden = np.load(self.hidden_path, mmap_mode="r")
        row = self.rows[index]
        features = torch.load(row["feature_path"], map_location="cpu", weights_only=True)
        teacher_hidden = torch.from_numpy(
            np.array(self.hidden[int(row["teacher_hidden_index"])], copy=True)
        )
        return {"row": row, "features": features, "teacher_hidden": teacher_hidden}


def collate_hidden(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_features(items)
    batch["teacher_hidden"] = torch.stack([item["teacher_hidden"] for item in items])
    return batch


def attach_hidden_indices(rows: list[dict[str, Any]], teacher_dir: Path) -> None:
    config = json.loads((teacher_dir / "run_config.json").read_text())
    completed = np.load(teacher_dir / "completed.npy", mmap_mode="r")
    hidden = np.load(teacher_dir / "features.npy", mmap_mode="r")
    if config.get("completed_jobs") != len(completed) or not bool(np.asarray(completed).all()):
        raise RuntimeError("teacher hidden cache is incomplete")
    if hidden.shape != (len(completed), 2048):
        raise RuntimeError(f"unexpected teacher hidden shape: {hidden.shape}")
    tav: dict[str, tuple[int, str]] = {}
    for line in (teacher_dir / "index.jsonl").read_text().splitlines():
        item = json.loads(line)
        if item["subset"] == "tav":
            key = str(item["sample_id"])
            if key in tav:
                raise RuntimeError(f"duplicate TAV teacher hidden row: {key}")
            tav[key] = (int(item["job_index"]), str(item["split"]))
    if len(tav) != len(rows):
        raise RuntimeError(f"TAV teacher/student count mismatch: {len(tav)} != {len(rows)}")
    for row in rows:
        value = tav.get(str(row["sample_id"]))
        if value is None or value[1] != row["split"]:
            raise RuntimeError(f"teacher/student hidden alignment failure: {row['sample_id']}")
        row["teacher_hidden_index"] = value[0]


def hidden_loss(
    projection: torch.nn.Module,
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    predicted = projection(output["fused"]).float()
    target = batch["teacher_hidden"].to(device, non_blocking=True).float()
    per_sample = 1.0 - F.cosine_similarity(predicted, target, dim=-1)
    weights = batch["weights"].to(device, non_blocking=True)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


@torch.inference_mode()
def evaluate(model: CachedStudentCore, loader: DataLoader, device: torch.device, args: argparse.Namespace):
    model.eval()
    records, loss_sum, weight_sum = [], 0.0, 0.0
    for batch in loader:
        hidden, masks = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(hidden, masks, subsets=("tav",))
            base_loss, _ = combined_loss(outputs, batch, device, args)
            representation_loss = hidden_loss(model.hidden_projection, outputs["tav"], batch, device)
            loss = base_loss + args.lambda_hidden * representation_loss
        weights = batch["weights"].to(device)
        loss_sum += float(loss) * float(weights.sum())
        weight_sum += float(weights.sum())
        predictions = outputs["tav"]["regression"].float().cpu().tolist()
        logits = outputs["tav"]["classification_logits"].float().cpu().tolist()
        for row, prediction, class_logits in zip(batch["rows"], predictions, logits):
            records.append({
                "sample_id": row["sample_id"], "parent_sample_id": row["parent_sample_id"],
                "split": row["split"], "target_sentiment": float(row["sentiment"]),
                "class_7_index": int(row["class_7_index"]),
                "aggregation_weight": float(row["aggregation_weight"]),
                "prediction": float(prediction), "classification_logits": class_logits,
            })
    utterances = aggregate_windows(records)
    metrics = sentiment_metrics(
        [item["target_sentiment"] for item in utterances],
        [item["prediction"] for item in utterances],
    )
    metrics["classification_accuracy"] = float(np.mean([
        int(np.argmax(item["classification_logits"])) == item["class_7_index"] for item in utterances
    ]))
    return metrics, loss_sum / weight_sum, records


def save_checkpoint(model: torch.nn.Module, epoch: int, metrics: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"epoch": epoch, "model": model.state_dict(), "valid_metrics": metrics}, temporary)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    args.method = "full_kd"
    args.lambda_subset = 0.0
    args.lambda_coordinate = 0.0
    args.lambda_binary_full = 0.0
    args.lambda_binary_subset = 0.0
    args.reliability_epsilon = 1e-4
    args.reliability_w_min = 0.25
    args.reliability_w_max = 4.0
    args.selective_keep_fraction = 0.5
    args.coordinate_center = [0.0] * 7
    args.coordinate_scale = [1.0] * 7
    args.interaction_utility_normalized = [1.0] * 3
    args.selective_threshold = None
    seed_everything(args.seed, args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    student_config = json.loads((args.features / "run_config.json").read_text())
    teacher_config = json.loads((args.teacher_features / "run_config.json").read_text())
    if student_config.get("status") != "complete":
        raise RuntimeError("student feature cache is incomplete")
    if student_config["manifest_sha256"] != teacher_config["manifest_sha256"]:
        raise RuntimeError("teacher/student manifest hashes differ")
    rows = [json.loads(line) for line in (args.features / "index.jsonl").read_text().splitlines() if line]
    add_window_weights(rows)
    attach_teacher_targets(rows, args.teacher_targets)
    attach_hidden_indices(rows, args.teacher_features)
    train_rows = [row for row in rows if row["split"] == "train"]
    valid_rows = [row for row in rows if row["split"] == "valid"]
    if args.limit_per_split:
        train_rows, valid_rows = train_rows[:args.limit_per_split], valid_rows[:args.limit_per_split]

    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_hidden, pin_memory=device.type == "cuda",
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        HiddenFeatureDataset(train_rows, args.teacher_features / "features.npy"),
        shuffle=True, generator=generator, **loader_kwargs,
    )
    valid_loader = DataLoader(
        HiddenFeatureDataset(valid_rows, args.teacher_features / "features.npy"),
        shuffle=False, **loader_kwargs,
    )
    model = CachedStudentCore(
        text_hidden_size=int(student_config["hidden_sizes"]["t"]),
        audio_hidden_size=int(student_config["hidden_sizes"]["a"]),
        video_hidden_size=int(student_config["hidden_sizes"]["v"]),
    )
    model.hidden_projection = torch.nn.Linear(512, 2048)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args), "features": str(args.features), "teacher_features": str(args.teacher_features),
        "teacher_targets": str(args.teacher_targets), "output": str(args.output),
        "hidden_target_subset": "tav", "hidden_target_dimension": 2048,
        "student_fused_dimension": 512, "hidden_loss": "cosine",
        "official_test_evaluated": False,
    }
    (args.output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"status": "initialized", "experiment": "D3_full_kd_hidden_kd", "seed": args.seed, "train_windows": len(train_rows), "valid_windows": len(valid_rows)}), flush=True)

    best_mae, best_epoch, stale, history = math.inf, 0, 0, []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = weight_sum = 0.0
        for batch in train_loader:
            hidden, masks = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(hidden, masks, subsets=("tav",))
                base_loss, components = combined_loss(outputs, batch, device, args)
                representation_loss = hidden_loss(model.hidden_projection, outputs["tav"], batch, device)
                loss = base_loss + args.lambda_hidden * representation_loss
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_weight = float(batch["weights"].sum())
            loss_sum += float(loss.detach()) * batch_weight
            weight_sum += batch_weight
        valid_metrics, valid_loss, _ = evaluate(model, valid_loader, device, args)
        item = {
            "epoch": epoch, "train_loss": loss_sum / weight_sum, "valid_loss": valid_loss,
            "valid_mae": valid_metrics["mae"], "valid_pearson": valid_metrics["pearson"],
            "valid_acc2_nonzero": valid_metrics["acc2_nonzero"],
            "gradient_norm_last": float(torch.as_tensor(gradient_norm)),
            "task_loss_last": components["task"], "full_kd_loss_last": components["full_kd"],
            "hidden_kd_loss_last": float(representation_loss.detach()),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        history.append(item)
        print(json.dumps(item), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if valid_metrics["mae"] < best_mae - 1e-5:
            best_mae, best_epoch, stale = float(valid_metrics["mae"]), epoch, 0
            save_checkpoint(model, epoch, valid_metrics, args.output / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_train_loader = DataLoader(
        HiddenFeatureDataset(train_rows, args.teacher_features / "features.npy"),
        shuffle=False, **loader_kwargs,
    )
    train_metrics, train_loss, train_records = evaluate(model, final_train_loader, device, args)
    valid_metrics, valid_loss, valid_records = evaluate(model, valid_loader, device, args)
    with (args.output / "predictions.jsonl").open("w") as handle:
        for item in aggregate_windows(train_records + valid_records):
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "experiment": "D3_full_kd_hidden_kd", "seed": args.seed,
        "best_epoch": best_epoch, "epochs_run": len(history),
        "train_metrics": train_metrics, "valid_metrics": valid_metrics,
        "train_loss": train_loss, "valid_loss": valid_loss,
        "train_windows": len(train_rows), "valid_windows": len(valid_rows),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "hidden_kd": {"teacher_subset": "tav", "student_dimension": 512, "teacher_dimension": 2048, "projection": "linear", "loss": "cosine", "lambda": args.lambda_hidden},
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3) if device.type == "cuda" else None,
        "elapsed_seconds": round(time.time() - started, 1), "official_test_evaluated": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
