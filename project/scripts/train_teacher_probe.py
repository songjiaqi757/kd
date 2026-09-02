#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = Path("/home/wy/sjq/kd/outputs/probe/features/benchmark500")
OUTPUT_ROOT = Path("/home/wy/sjq/kd/outputs/probe/benchmark500")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.metrics import sentiment_metrics  # noqa: E402
from rdid_mosei.probe import TeacherProbe  # noqa: E402
from rdid_mosei.subsets import SUBSETS  # noqa: E402

SUBSET_PROBABILITY = {
    "t": 0.10,
    "a": 0.10,
    "v": 0.10,
    "ta": 0.10,
    "tv": 0.10,
    "av": 0.10,
    "tav": 0.40,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and calibrate the shared frozen-Thinker teacher Probe")
    parser.add_argument("--features", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha-ce", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    parent_totals: dict[tuple[str, str], float] = defaultdict(float)
    for row in rows:
        parent_totals[(str(row["parent_sample_id"]), str(row["subset"]))] += float(
            row["aggregation_weight"]
        )
    values = []
    for row in rows:
        key = (str(row["parent_sample_id"]), str(row["subset"]))
        window_weight = float(row["aggregation_weight"]) / parent_totals[key]
        values.append(SUBSET_PROBABILITY[str(row["subset"])] * window_weight)
    return np.asarray(values, dtype=np.float32)


def weighted_losses(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    classes: torch.Tensor,
    weights: torch.Tensor,
    alpha_ce: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression = F.smooth_l1_loss(outputs["regression"], targets, reduction="none")
    classification = F.cross_entropy(outputs["classification_logits"], classes, reduction="none")
    denominator = weights.sum().clamp_min(1e-8)
    regression_loss = (regression * weights).sum() / denominator
    classification_loss = (classification * weights).sum() / denominator
    return regression_loss + alpha_ce * classification_loss, regression_loss, classification_loss


@torch.inference_mode()
def predict(model: TeacherProbe, features: torch.Tensor, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    regression_parts = []
    logit_parts = []
    for start in range(0, len(features), batch_size):
        output = model(features[start : start + batch_size].to(device))
        regression_parts.append(output["regression"].cpu())
        logit_parts.append(output["classification_logits"].cpu())
    return torch.cat(regression_parts).numpy(), torch.cat(logit_parts).numpy()


def aggregate_predictions(
    rows: list[dict[str, Any]], regression: np.ndarray, logits: np.ndarray
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[tuple[float, float, np.ndarray, dict[str, Any]]]] = defaultdict(list)
    for row, score, class_logits in zip(rows, regression, logits):
        key = (str(row["parent_sample_id"]), str(row["subset"]))
        groups[key].append((float(score), float(row["aggregation_weight"]), class_logits, row))
    aggregated = []
    for (parent_id, subset), items in sorted(groups.items()):
        total_weight = sum(item[1] for item in items)
        first = items[0][3]
        aggregated.append(
            {
                "parent_sample_id": parent_id,
                "split": first["split"],
                "subset": subset,
                "target_sentiment": float(first["target_sentiment"]),
                "class_7_index": int(first["class_7_index"]),
                "probe_score": sum(item[0] * item[1] for item in items) / total_weight,
                "classification_logits": (
                    sum(item[2] * item[1] for item in items) / total_weight
                ).tolist(),
                "window_count": len(items),
            }
        )
    return aggregated


def classification_diagnostics(logits: np.ndarray, classes: np.ndarray, temperature: float) -> dict[str, float]:
    scaled = torch.from_numpy(logits).float() / temperature
    labels = torch.from_numpy(classes).long()
    probabilities = torch.softmax(scaled, dim=-1)
    nll = float(F.cross_entropy(scaled, labels))
    one_hot = F.one_hot(labels, num_classes=7).float()
    brier = float(torch.mean(torch.sum((probabilities - one_hot) ** 2, dim=-1)))
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction == labels
    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, 16)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.float().mean() * (correct[mask].float().mean() - confidence[mask].mean()).abs())
    return {"nll": nll, "brier": brier, "ece_15bin": ece, "temperature": temperature}


def calibrate_temperature(logits: np.ndarray, classes: np.ndarray) -> float:
    logits_tensor = torch.from_numpy(logits).float()
    labels = torch.from_numpy(classes).long()
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits_tensor / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = json.loads((args.features / "run_config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.features / "index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    completed = np.load(args.features / "completed.npy", mmap_mode="r")
    if len(rows) != len(completed) or not bool(np.all(completed)):
        raise RuntimeError(f"feature extraction incomplete: {int(completed.sum())}/{len(completed)}")
    features_array = np.load(args.features / "features.npy", mmap_mode="r")
    if not np.all(np.isfinite(features_array)):
        raise RuntimeError("feature array contains non-finite values")
    features = torch.from_numpy(np.asarray(features_array, dtype=np.float32))
    targets = torch.tensor([float(row["target_sentiment"]) for row in rows], dtype=torch.float32)
    classes = torch.tensor([int(row["class_7_index"]) for row in rows], dtype=torch.long)
    weights = torch.from_numpy(row_weights(rows))
    train_indices = torch.tensor([index for index, row in enumerate(rows) if row["split"] == "train"])
    valid_indices = torch.tensor([index for index, row in enumerate(rows) if row["split"] == "valid"])
    if not len(train_indices) or not len(valid_indices):
        raise RuntimeError("both official train and validation rows are required")

    model = TeacherProbe(
        input_size=int(config["feature_dimension"]),
        hidden_size=args.hidden_size,
        classes=7,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset = TensorDataset(train_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    best_state = None
    best_valid_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for (indices,) in train_loader:
            indices = indices.long()
            output = model(features[indices].to(device))
            loss, _, _ = weighted_losses(
                output,
                targets[indices].to(device),
                classes[indices].to(device),
                weights[indices].to(device),
                args.alpha_ce,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach()) * len(indices)

        model.eval()
        with torch.inference_mode():
            valid_output = model(features[valid_indices].to(device))
            valid_loss, valid_regression, valid_classification = weighted_losses(
                valid_output,
                targets[valid_indices].to(device),
                classes[valid_indices].to(device),
                weights[valid_indices].to(device),
                args.alpha_ce,
            )
        item = {
            "epoch": epoch,
            "train_loss": train_total / len(train_indices),
            "valid_loss": float(valid_loss),
            "valid_regression_loss": float(valid_regression),
            "valid_classification_loss": float(valid_classification),
        }
        history.append(item)
        if float(valid_loss) < best_valid_loss - 1e-5:
            best_valid_loss = float(valid_loss)
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(item), flush=True)
        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    regression, logits = predict(model, features, args.batch_size, device)
    aggregated = aggregate_predictions(rows, regression, logits)
    valid_aggregated = [row for row in aggregated if row["split"] == "valid"]
    valid_logits = np.asarray([row["classification_logits"] for row in valid_aggregated], dtype=np.float32)
    valid_classes = np.asarray([row["class_7_index"] for row in valid_aggregated], dtype=np.int64)
    temperature = calibrate_temperature(valid_logits, valid_classes)

    metrics_by_split: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        metrics_by_split[split] = {}
        for subset in SUBSETS:
            selected = [row for row in aggregated if row["split"] == split and row["subset"] == subset]
            metrics_by_split[split][subset] = sentiment_metrics(
                [row["target_sentiment"] for row in selected],
                [row["probe_score"] for row in selected],
            )

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "input_size": int(config["feature_dimension"]),
        "hidden_size": args.hidden_size,
        "classes": 7,
        "dropout": args.dropout,
        "temperature": temperature,
        "seed": args.seed,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, args.output / "teacher_probe.pt")
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in aggregated:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "teacher-probe-training-v1",
        "features": str(args.features.resolve()),
        "device": str(device),
        "train_parent_samples": len({row["parent_sample_id"] for row in rows if row["split"] == "train"}),
        "valid_parent_samples": len({row["parent_sample_id"] for row in rows if row["split"] == "valid"}),
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "epochs_ran": len(history),
        "subset_sampling_probability": SUBSET_PROBABILITY,
        "metrics": metrics_by_split,
        "calibration": {
            "before": classification_diagnostics(valid_logits, valid_classes, 1.0),
            "after": classification_diagnostics(valid_logits, valid_classes, temperature),
        },
        "history": history,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("best_epoch", "best_valid_loss", "epochs_ran", "calibration")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
