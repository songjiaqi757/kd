#!/usr/bin/env python3
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
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.metrics import sentiment_metrics
from rdid_mosei.interaction import mobius_transform
from rdid_mosei.student import CachedStudentCore

MULTI_SUBSET_METHODS = ("subset_value", "subset_value_4", "mobius_full", "high_order_interaction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the B0 task-only student on cached encoder features")
    parser.add_argument("--features", type=Path, default=Path("/home/wy/sjq/kd/outputs/student/features/benchmark500"))
    parser.add_argument("--output", type=Path, default=Path("/home/wy/sjq/kd/outputs/student/baseline_benchmark500_seed2026"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha-ce", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--method",
        choices=("student", "full_kd", "subset_value", "subset_value_4", "mobius_full", "high_order_interaction"),
        default="student",
    )
    parser.add_argument("--teacher-targets", type=Path, default=Path("/home/wy/sjq/kd/outputs/probe/benchmark500/predictions.jsonl"))
    parser.add_argument("--lambda-full", type=float, default=1.0)
    parser.add_argument("--lambda-subset", type=float, default=1.0)
    parser.add_argument("--lambda-coordinate", type=float, default=1.0)
    parser.add_argument("--lambda-kd-regression", type=float, default=1.0)
    parser.add_argument("--lambda-kd-classification", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-calibration-temperature", type=float, default=0.9259549975395203)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        features = torch.load(row["feature_path"], map_location="cpu", weights_only=True)
        return {"row": row, "features": features}


def collate_features(items: list[dict[str, Any]]) -> dict[str, Any]:
    hidden: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for modality in "tav":
        sequences = [item["features"][modality] for item in items]
        lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
        hidden[modality] = pad_sequence(sequences, batch_first=True)
        positions = torch.arange(hidden[modality].shape[1]).unsqueeze(0)
        masks[modality] = positions < lengths.unsqueeze(1)
    rows = [item["row"] for item in items]
    return {
        "hidden": hidden,
        "masks": masks,
        "sentiment": torch.tensor([row["sentiment"] for row in rows], dtype=torch.float32),
        "classes": torch.tensor([row["class_7_index"] for row in rows], dtype=torch.long),
        "weights": torch.tensor([row["sample_weight"] for row in rows], dtype=torch.float32),
        "teacher_scores": torch.tensor([row.get("teacher_score", math.nan) for row in rows], dtype=torch.float32),
        "teacher_logits": torch.tensor(
            [row.get("teacher_logits", [math.nan] * 7) for row in rows], dtype=torch.float32
        ),
        "teacher_subset_scores": torch.tensor(
            [row.get("teacher_subset_scores", [math.nan] * 7) for row in rows], dtype=torch.float32
        ),
        "rows": rows,
    }


def add_window_weights(rows: list[dict[str, Any]]) -> None:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[str(row["parent_sample_id"])] += float(row["aggregation_weight"])
    for row in rows:
        row["sample_weight"] = float(row["aggregation_weight"]) / totals[str(row["parent_sample_id"])]


def weighted_task_loss(
    output: dict[str, torch.Tensor],
    sentiment: torch.Tensor,
    classes: torch.Tensor,
    weights: torch.Tensor,
    alpha_ce: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression = F.smooth_l1_loss(output["regression"].float(), sentiment, reduction="none")
    classification = F.cross_entropy(output["classification_logits"].float(), classes, reduction="none")
    denominator = weights.sum().clamp_min(1e-8)
    regression_loss = (regression * weights).sum() / denominator
    classification_loss = (classification * weights).sum() / denominator
    return regression_loss + alpha_ce * classification_loss, regression_loss, classification_loss


def weighted_full_kd_loss(
    output: dict[str, torch.Tensor],
    teacher_scores: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
    calibration_temperature: float,
    regression_weight: float,
    classification_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression = F.smooth_l1_loss(output["regression"].float(), teacher_scores, reduction="none")
    teacher_probabilities = torch.softmax(
        teacher_logits / (calibration_temperature * temperature), dim=-1
    )
    student_log_probabilities = torch.log_softmax(
        output["classification_logits"].float() / temperature, dim=-1
    )
    classification = F.kl_div(
        student_log_probabilities, teacher_probabilities, reduction="none"
    ).sum(dim=-1) * temperature**2
    denominator = weights.sum().clamp_min(1e-8)
    regression_loss = (regression * weights).sum() / denominator
    classification_loss = (classification * weights).sum() / denominator
    return (
        regression_weight * regression_loss + classification_weight * classification_loss,
        regression_loss,
        classification_loss,
    )


def combined_loss(
    outputs: dict[str, dict[str, torch.Tensor]],
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = outputs["tav"]
    sentiment = batch["sentiment"].to(device, non_blocking=True)
    classes = batch["classes"].to(device, non_blocking=True)
    weights = batch["weights"].to(device, non_blocking=True)
    task, task_regression, task_classification = weighted_task_loss(
        output, sentiment, classes, weights, args.alpha_ce
    )
    kd = task.new_zeros(())
    kd_regression = task.new_zeros(())
    kd_classification = task.new_zeros(())
    if args.method != "student":
        teacher_scores = batch["teacher_scores"].to(device, non_blocking=True)
        teacher_logits = batch["teacher_logits"].to(device, non_blocking=True)
        if not torch.isfinite(teacher_scores).all() or not torch.isfinite(teacher_logits).all():
            raise ValueError("full_kd batch has missing teacher targets")
        kd, kd_regression, kd_classification = weighted_full_kd_loss(
            output,
            teacher_scores,
            teacher_logits,
            weights,
            args.kd_temperature,
            args.teacher_calibration_temperature,
            args.lambda_kd_regression,
            args.lambda_kd_classification,
        )
    subset_value = task.new_zeros(())
    if args.method in ("subset_value", "subset_value_4"):
        teacher_subset_scores = batch["teacher_subset_scores"].to(device, non_blocking=True)
        if not torch.isfinite(teacher_subset_scores).all():
            raise ValueError("subset_value batch has missing teacher targets")
        per_subset = []
        selected = (
            tuple(enumerate(("t", "a", "v", "ta", "tv", "av", "tav")))
            if args.method == "subset_value"
            else tuple(enumerate(("t", "a", "v", "ta", "tv", "av", "tav")))[3:]
        )
        for subset_index, subset in selected:
            per_sample = F.smooth_l1_loss(
                outputs[subset]["regression"].float(), teacher_subset_scores[:, subset_index], reduction="none"
            )
            per_subset.append((per_sample * weights).sum() / weights.sum().clamp_min(1e-8))
        subset_value = torch.stack(per_subset).mean()
    coordinate = task.new_zeros(())
    if args.method in ("mobius_full", "high_order_interaction"):
        teacher_subset_scores = batch["teacher_subset_scores"].to(device, non_blocking=True)
        if not torch.isfinite(teacher_subset_scores).all():
            raise ValueError("coordinate batch has missing teacher targets")
        student_subset_scores = torch.stack(
            [outputs[subset]["regression"].float() for subset in ("t", "a", "v", "ta", "tv", "av", "tav")],
            dim=-1,
        )
        student_coordinates = mobius_transform(student_subset_scores, 0.0)
        teacher_coordinates = mobius_transform(teacher_subset_scores, 0.0)
        coordinate_indices = slice(None) if args.method == "mobius_full" else slice(3, 7)
        per_sample_coordinate = F.smooth_l1_loss(
            student_coordinates[:, coordinate_indices],
            teacher_coordinates[:, coordinate_indices],
            reduction="none",
        ).mean(dim=-1)
        coordinate = (per_sample_coordinate * weights).sum() / weights.sum().clamp_min(1e-8)
    total = (
        task
        + args.lambda_full * kd
        + args.lambda_subset * subset_value
        + args.lambda_coordinate * coordinate
    )
    components = {
        "task": float(task.detach()),
        "task_regression": float(task_regression.detach()),
        "task_classification": float(task_classification.detach()),
        "full_kd": float(kd.detach()),
        "kd_regression": float(kd_regression.detach()),
        "kd_classification": float(kd_classification.detach()),
        "subset_value": float(subset_value.detach()),
        "coordinate": float(coordinate.detach()),
    }
    return total, components


def to_device(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    hidden = {key: value.to(device, non_blocking=True) for key, value in batch["hidden"].items()}
    masks = {key: value.to(device, non_blocking=True) for key, value in batch["masks"].items()}
    return hidden, masks


def aggregate_windows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["parent_sample_id"])].append(record)
    aggregated = []
    for parent_id, items in sorted(groups.items()):
        denominator = sum(float(item["aggregation_weight"]) for item in items)
        prediction = sum(float(item["prediction"]) * float(item["aggregation_weight"]) for item in items) / denominator
        logits = np.average(
            np.asarray([item["classification_logits"] for item in items], dtype=np.float64),
            axis=0,
            weights=np.asarray([item["aggregation_weight"] for item in items]),
        )
        aggregated.append(
            {
                "parent_sample_id": parent_id,
                "split": items[0]["split"],
                "target_sentiment": float(items[0]["target_sentiment"]),
                "class_7_index": int(items[0]["class_7_index"]),
                "prediction": prediction,
                "classification_logits": logits.tolist(),
                "window_count": len(items),
            }
        )
    return aggregated


@torch.inference_mode()
def evaluate(
    model: CachedStudentCore,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int], float, list[dict[str, Any]]]:
    model.eval()
    records = []
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    for batch in loader:
        hidden, masks = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            requested_subsets = ("t", "a", "v", "ta", "tv", "av", "tav") if args.method in MULTI_SUBSET_METHODS else ("tav",)
            outputs = model(hidden, masks, subsets=requested_subsets)
            output = outputs["tav"]
        weights = batch["weights"].to(device)
        loss, _ = combined_loss(outputs, batch, device, args)
        batch_weight = float(weights.sum())
        weighted_loss_sum += float(loss) * batch_weight
        weight_sum += batch_weight
        predictions = output["regression"].float().cpu().tolist()
        logits = output["classification_logits"].float().cpu().tolist()
        for row, prediction, class_logits in zip(batch["rows"], predictions, logits):
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "parent_sample_id": row["parent_sample_id"],
                    "split": row["split"],
                    "target_sentiment": float(row["sentiment"]),
                    "class_7_index": int(row["class_7_index"]),
                    "aggregation_weight": float(row["aggregation_weight"]),
                    "prediction": float(prediction),
                    "classification_logits": class_logits,
                }
            )
    utterances = aggregate_windows(records)
    metrics = sentiment_metrics(
        [item["target_sentiment"] for item in utterances],
        [item["prediction"] for item in utterances],
    )
    metrics["classification_accuracy"] = float(
        np.mean([int(np.argmax(item["classification_logits"])) == item["class_7_index"] for item in utterances])
    )
    return metrics, weighted_loss_sum / weight_sum, records


def attach_teacher_targets(rows: list[dict[str, Any]], path: Path) -> None:
    targets: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        targets[str(item["parent_sample_id"])][str(item["subset"])] = item
    missing = []
    for row in rows:
        items = targets.get(str(row["parent_sample_id"]))
        if items is None or set(items) != {"t", "a", "v", "ta", "tv", "av", "tav"}:
            missing.append(str(row["parent_sample_id"]))
            continue
        item = items["tav"]
        if item["split"] != row["split"]:
            raise ValueError(f"teacher/student split mismatch for {row['parent_sample_id']}")
        row["teacher_score"] = float(item["probe_score"])
        row["teacher_logits"] = [float(value) for value in item["classification_logits"]]
        row["teacher_subset_scores"] = [
            float(items[subset]["probe_score"]) for subset in ("t", "a", "v", "ta", "tv", "av", "tav")
        ]
    if missing:
        raise RuntimeError(f"missing complete teacher subset targets for {len(set(missing))} utterances")


def atomic_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    config = json.loads((args.features / "run_config.json").read_text(encoding="utf-8"))
    if config.get("status") != "complete":
        raise RuntimeError(f"feature cache is not complete: {config.get('status')}")
    rows = [json.loads(line) for line in (args.features / "index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    missing = [row["feature_path"] for row in rows if not Path(row["feature_path"]).is_file()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} cached feature files")
    add_window_weights(rows)
    if args.method != "student":
        attach_teacher_targets(rows, args.teacher_targets)
    train_rows = [row for row in rows if row["split"] == "train"]
    valid_rows = [row for row in rows if row["split"] == "valid"]
    if not train_rows or not valid_rows:
        raise RuntimeError("official train and validation splits are both required")

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_features,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(FeatureDataset(train_rows), shuffle=True, generator=generator, **loader_kwargs)
    valid_loader = DataLoader(FeatureDataset(valid_rows), shuffle=False, **loader_kwargs)
    model = CachedStudentCore(
        text_hidden_size=int(config["hidden_sizes"]["t"]),
        audio_hidden_size=int(config["hidden_sizes"]["a"]),
        video_hidden_size=int(config["hidden_sizes"]["v"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    best_mae = math.inf
    best_epoch = 0
    no_improvement = 0
    history = []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for batch in train_loader:
            hidden, masks = to_device(batch, device)
            weights = batch["weights"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                requested_subsets = ("t", "a", "v", "ta", "tv", "av", "tav") if args.method in MULTI_SUBSET_METHODS else ("tav",)
                outputs = model(hidden, masks, subsets=requested_subsets)
                loss, components = combined_loss(outputs, batch, device, args)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_weight = float(weights.sum())
            train_loss_sum += float(loss.detach()) * batch_weight
            train_weight_sum += batch_weight

        valid_metrics, valid_loss, _ = evaluate(model, valid_loader, device, args)
        item = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_weight_sum,
            "valid_loss": valid_loss,
            "valid_mae": valid_metrics["mae"],
            "valid_pearson": valid_metrics["pearson"],
            "valid_acc2_nonzero": valid_metrics["acc2_nonzero"],
            "gradient_norm_last_batch": float(torch.as_tensor(gradient_norm).detach().cpu()),
            "task_loss_last_batch": components["task"],
            "full_kd_loss_last_batch": components["full_kd"],
            "subset_value_loss_last_batch": components["subset_value"],
            "coordinate_loss_last_batch": components["coordinate"],
            "elapsed_seconds": round(time.time() - started, 1),
        }
        history.append(item)
        print(json.dumps(item), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        atomic_save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)},
            args.output / "last.pt",
        )
        if float(valid_metrics["mae"]) < best_mae - 1e-5:
            best_mae = float(valid_metrics["mae"])
            best_epoch = epoch
            no_improvement = 0
            atomic_save({"epoch": epoch, "model": model.state_dict(), "valid_metrics": valid_metrics}, args.output / "best.pt")
        else:
            no_improvement += 1
        if no_improvement >= args.patience:
            break

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    train_metrics, train_loss, train_records = evaluate(model, DataLoader(FeatureDataset(train_rows), shuffle=False, **loader_kwargs), device, args)
    valid_metrics, valid_loss, valid_records = evaluate(model, valid_loader, device, args)
    all_utterances = aggregate_windows(train_records + valid_records)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in all_utterances:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "experiment": {
            "student": "B0_student_task_only",
            "full_kd": "B1_full_kd",
            "subset_value": "B2_subset_value_kd",
            "subset_value_4": "C_subset_4_equal_dimension",
            "mobius_full": "B3_full_mobius_7",
            "high_order_interaction": "B4_high_order_interaction",
        }[args.method],
        "seed": args.seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "train_windows": len(train_rows),
        "valid_windows": len(valid_rows),
        "train_utterances": int(train_metrics["count"]),
        "valid_utterances": int(valid_metrics["count"]),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3) if device.type == "cuda" else None,
        "elapsed_seconds": round(time.time() - started, 1),
        "distillation": {
            "enabled": args.method != "student",
            "lambda_full": args.lambda_full,
            "lambda_regression": args.lambda_kd_regression,
            "lambda_classification": args.lambda_kd_classification,
            "temperature": args.kd_temperature,
            "teacher_calibration_temperature": args.teacher_calibration_temperature,
            "teacher_targets": str(args.teacher_targets) if args.method != "student" else None,
            "lambda_subset": args.lambda_subset if args.method in ("subset_value", "subset_value_4") else 0.0,
            "subset_dimensions": 7 if args.method == "subset_value" else (4 if args.method == "subset_value_4" else 0),
            "lambda_coordinate": args.lambda_coordinate if args.method in ("mobius_full", "high_order_interaction") else 0.0,
            "coordinate_dimensions": 7 if args.method == "mobius_full" else (4 if args.method == "high_order_interaction" else 0),
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
