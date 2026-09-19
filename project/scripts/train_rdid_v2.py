#!/usr/bin/env python3
"""Train one seed-13 RDID-v2 MOSEI development candidate.

The script is intentionally independent of the frozen M0--M6 entry point.  It
stores a complete resumable checkpoint after every completed epoch and never
reads the MOSEI official-test split.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "project/src"), str(ROOT / "project/scripts")]

from rdid_mosei.main_table_kd import MainTableKDStudent
from rdid_mosei.rdid_v2 import RDID_V2_METHODS, interaction_loss
from train_main_table_kd import (
    MainTableCollator,
    checkpoint_state,
    evaluate,
    load_checkpoint_state,
    load_rows,
)
from train_student_baseline import seed_everything, weighted_full_kd_loss, weighted_task_loss
from train_tav_distillation import (
    VideoDataset,
    atomic_json,
    atomic_save,
    batch_inputs,
    restore_rng,
    rng_state,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=RDID_V2_METHODS, required=True)
    parser.add_argument(
        "--assets", type=Path,
        default=ROOT / "outputs/experiments/main_table_v1/mosei/assets/protocol.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, default=ROOT / "model/Qwen3-0.6B-Base")
    parser.add_argument("--audio-model", type=Path, default=ROOT / "model/WavLM-Base-Plus")
    parser.add_argument("--video-model", type=Path, default=ROOT / "model/VideoMAE-Base")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--video-layers", type=int, default=12)
    parser.add_argument("--video-rank", type=int, default=8)
    parser.add_argument("--video-alpha", type=float, default=16.0)
    parser.add_argument("--video-dropout", type=float, default=0.05)
    parser.add_argument("--video-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--soft-alpha", type=float, help="Required only for soft_ru_gate")
    parser.add_argument("--limit-per-split", type=int, help="Smoke-only parent limit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.assets = args.assets.resolve()
    assets = json.loads(args.assets.read_text())
    args.manifest = Path(assets["manifest"])
    args.teacher_targets = Path(assets["teacher_targets"])
    args.teacher_probe_report = Path(assets["teacher_probe_report"])
    args.teacher_subset = "tav"
    args.teacher_calibration_temperature = float(assets["teacher_temperature"])
    args.mode = "video_lora"
    if args.seed != 13:
        parser.error("RDID-v2 development runs are restricted to seed=13")
    if args.method == "soft_ru_gate" and (args.soft_alpha is None or not 0 <= args.soft_alpha <= 1):
        parser.error("soft_ru_gate requires --soft-alpha in [0, 1]")
    if args.method != "soft_ru_gate" and args.soft_alpha is not None:
        parser.error("--soft-alpha is only valid for soft_ru_gate")
    if min(args.batch_size, args.epochs, args.patience, args.video_layers, args.video_rank, args.progress_every) <= 0:
        parser.error("counts must be positive")
    if args.num_workers < 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("invalid worker count or optimizer settings")
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        parser.error("limit-per-split must be positive")
    return args


def train_interaction_strength_median(rows: list[dict]) -> float:
    parents = {}
    for row in rows:
        parents.setdefault(row["parent_sample_id"], row["interaction_mean"])
    strengths = np.asarray([np.abs(value[3:]).mean() for value in parents.values()], dtype=np.float64)
    median = float(np.median(strengths))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("train high-order interaction median must be positive")
    return median


def build_model(args: argparse.Namespace, assets: dict, device: torch.device) -> MainTableKDStudent:
    from transformers import AutoModel

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    text = AutoModel.from_pretrained(args.text_model, local_files_only=True, dtype=dtype)
    audio = AutoModel.from_pretrained(args.audio_model, local_files_only=True, dtype=dtype)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed + 100000)
        video = AutoModel.from_pretrained(
            args.video_model, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
        )
    text.config.use_cache = False
    # uniform_interaction selects the common M3-capacity model and enables all
    # seven subset forwards in training. RDID-v2 controls only the loss below.
    return MainTableKDStudent(
        text,
        audio,
        video,
        comparison_method="uniform_interaction",
        teacher_feature_dim=assets["teacher_feature_dimension"],
        cmad_tau=assets["cmad_cafd"]["tau"],
        video_layers=args.video_layers,
        video_rank=args.video_rank,
        video_alpha=args.video_alpha,
        video_dropout=args.video_dropout,
        video_seed=args.seed + 100000,
        video_checkpointing=args.video_checkpointing,
    ).to(device)


def loss_components(model, outputs, batch, device, args, assets, gate_median):
    del model  # Kept in the signature to make future train-only modules explicit.
    output = outputs["tav"]
    sample_weights = batch["weights"].to(device)
    task, task_regression, task_classification = weighted_task_loss(
        output,
        batch["sentiment"].to(device),
        batch["classes"].to(device),
        sample_weights,
        assets["alpha_ce"],
    )
    full, full_regression, full_kl = weighted_full_kd_loss(
        output,
        batch["teacher_scores"].to(device),
        batch["teacher_logits"].to(device),
        sample_weights,
        assets["kd_temperature"],
        args.teacher_calibration_temperature,
        1.0,
        1.0,
    )
    interaction, coordinate_weight, gate = interaction_loss(
        outputs,
        batch["interaction_mean"].to(device),
        batch["interaction_variance"].to(device),
        torch.as_tensor(assets["utility_normalized"], device=device),
        sample_weights,
        assets["student_empty_baseline"],
        args.method,
        soft_alpha=args.soft_alpha,
        gate_train_median=gate_median if args.method == "soft_ru_gate" else None,
    )
    total = task + assets["lambda_full"] * full + assets["lambda_interaction"] * interaction
    components = {
        "total_loss": total,
        "task_loss": task,
        "task_regression_loss": task_regression,
        "task_classification_loss": task_classification,
        "full_kd_loss": full,
        "full_kd_regression_loss": full_regression,
        "full_kd_kl_loss": full_kl,
        "interaction_loss": interaction,
    }
    return total, components, coordinate_weight, gate


def update_weight_accumulator(accumulator: dict, weights: torch.Tensor, gate: torch.Tensor,
                              sample_weights: torch.Tensor) -> None:
    values = weights.detach().double().cpu()
    sample_weights = sample_weights.detach().double().cpu()
    probabilities = values / values.sum(-1, keepdim=True).clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    effective = probabilities.square().sum(-1).reciprocal()
    coordinate_weights = sample_weights[:, None].expand_as(values)
    accumulator["coordinate_mass"] += float(coordinate_weights.sum())
    accumulator["weight_sum"] += float((values * coordinate_weights).sum())
    accumulator["weight_square_sum"] += float((values.square() * coordinate_weights).sum())
    accumulator["weight_min"] = min(accumulator["weight_min"], float(values.min()))
    accumulator["weight_max"] = max(accumulator["weight_max"], float(values.max()))
    accumulator["sample_mass"] += float(sample_weights.sum())
    accumulator["entropy_sum"] += float((entropy * sample_weights).sum())
    accumulator["effective_sum"] += float((effective * sample_weights).sum())
    gate = gate.detach().double().cpu()
    accumulator["gate_sum"] += float((gate * sample_weights).sum())
    accumulator["gate_active_sum"] += float(((gate > 0).double() * sample_weights).sum())


def finalize_weight_accumulator(accumulator: dict) -> dict[str, float]:
    coordinate_mass = accumulator["coordinate_mass"]
    sample_mass = accumulator["sample_mass"]
    if coordinate_mass <= 0 or sample_mass <= 0:
        raise ValueError("empty interaction-weight accumulator")
    mean = accumulator["weight_sum"] / coordinate_mass
    variance = max(accumulator["weight_square_sum"] / coordinate_mass - mean * mean, 0.0)
    return {
        "train_interaction_weight_mean": mean,
        "train_interaction_weight_std": math.sqrt(variance),
        "train_interaction_weight_min": accumulator["weight_min"],
        "train_interaction_weight_max": accumulator["weight_max"],
        "train_interaction_weight_entropy": accumulator["entropy_sum"] / sample_mass,
        "train_effective_coordinate_count": accumulator["effective_sum"] / sample_mass,
        "train_interaction_gate_mean": accumulator["gate_sum"] / sample_mass,
        "train_interaction_gate_active_fraction": accumulator["gate_active_sum"] / sample_mass,
    }


def protocol(args: argparse.Namespace, assets: dict, gate_median: float) -> dict:
    excluded = {"resume", "stop_after_epoch", "dry_run", "progress_every"}
    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in excluded
    }
    config.update(
        schema="rdid-v2-mosei-development-train-v1",
        dataset="cmu_mosei",
        development_seed_policy="seed13_only",
        checkpoint_selection="valid_mae",
        epoch_checkpoint_pattern="checkpoints/epoch_{epoch:03d}.pt",
        epoch_checkpoints_fully_resumable=True,
        official_test_evaluated=False,
        mosei_test_role="method_level_development_only_never_epoch_selection",
        model_comparison_method="uniform_interaction",
        interaction_strength_train_median=gate_median,
        frozen_assets=assets,
    )
    sources = [
        args.assets,
        Path(__file__),
        ROOT / "project/src/rdid_mosei/rdid_v2.py",
        ROOT / "project/src/rdid_mosei/main_table_kd.py",
        ROOT / "project/scripts/train_main_table_kd.py",
        ROOT / "project/scripts/train_tav_distillation.py",
    ]
    config["input_sha256"] = {str(path.resolve()): sha256(path) for path in sources}
    return config


@contextmanager
def run_lock(output: Path, resume: bool):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "run_config.json").exists() and not resume:
            raise FileExistsError(f"existing run: use --resume or a new output: {output}")
        yield


def atomic_alias(target: Path, alias: Path) -> None:
    """Atomically point a convenience alias at an immutable epoch checkpoint."""
    temporary = alias.with_name(f".{alias.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    os.symlink(os.path.relpath(target, alias.parent), temporary)
    os.replace(temporary, alias)


def checkpoint_inventory(output: Path, history: list[dict], best_epoch: int) -> dict:
    files = sorted((output / "checkpoints").glob("epoch_*.pt"))
    expected = [int(row["epoch"]) for row in history]
    actual = [int(path.stem.rsplit("_", 1)[1]) for path in files]
    if actual != expected:
        raise ValueError(f"epoch checkpoint coverage differs: expected={expected}, actual={actual}")
    rows = []
    previous_global_step = -1
    run_config = json.loads((output / "run_config.json").read_text())
    for path, epoch in zip(files, actual):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "model_state_dict", "optimizer_state_dict", "lr_scheduler_state_dict",
            "grad_scaler_state_dict", "epoch", "global_step", "best_valid_mae_so_far",
            "best_epoch_so_far", "stale_epochs", "training_history_so_far", "run_config",
            "method_config", "random_seed", "rng_state",
        }
        if (
            not required.issubset(saved)
            or saved["epoch"] != epoch
            or saved["run_config"] != run_config
            or int(saved["training_history_so_far"][-1]["epoch"]) != epoch
            or int(saved["global_step"]) <= previous_global_step
        ):
            raise ValueError(f"incomplete epoch checkpoint: {path}")
        previous_global_step = int(saved["global_step"])
        rows.append({
            "epoch": epoch,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "valid_mae": float(saved["training_history_so_far"][-1]["valid_metrics"]["mae"]),
            "is_best_epoch": epoch == best_epoch,
            "global_step": previous_global_step,
        })
    if files:
        if not (output / "best.pt").is_symlink() or not (output / "last.pt").is_symlink():
            raise ValueError("best.pt and last.pt must be immutable epoch aliases")
        if (output / "best.pt").resolve() != files[actual.index(best_epoch)].resolve():
            raise ValueError("best.pt does not point to the valid-best epoch")
        if (output / "last.pt").resolve() != files[-1].resolve():
            raise ValueError("last.pt does not point to the last completed epoch")
    return {
        "schema": "rdid-v2-complete-epoch-checkpoint-inventory-v1",
        "history_epoch_count": len(history),
        "checkpoint_epoch_count": len(files),
        "coverage_complete": len(history) == len(files),
        "epochs": rows,
    }


def train(args: argparse.Namespace, output: Path) -> None:
    seed_everything(args.seed, True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    assets = json.loads(args.assets.read_text())
    train_rows, valid_rows = load_rows(args)
    gate_median = train_interaction_strength_median(train_rows)
    config = protocol(args, assets, gate_median)
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("resume protocol/input fingerprint differs")
    if args.resume and not config_path.exists():
        raise FileNotFoundError("--resume requires an existing configured run")
    if args.resume and (output / "status.json").is_file():
        if json.loads((output / "status.json").read_text()).get("status") == "complete":
            print(json.dumps({"status": "already_complete", "output": str(output)}), flush=True)
            return
    atomic_json(config, config_path)

    model = build_model(args, assets, device)
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoFeatureExtractor.from_pretrained(args.audio_model, local_files_only=True)
    video_processor = AutoImageProcessor.from_pretrained(
        args.video_model, local_files_only=True, use_fast=False
    )
    collator = MainTableCollator(tokenizer, audio_processor, None)
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(
        VideoDataset(train_rows, video_processor, mode="video_lora"),
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    valid_loader = DataLoader(
        VideoDataset(valid_rows, video_processor, mode="video_lora"),
        shuffle=False,
        **loader_args,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    best_mae, best_epoch, stale = math.inf, 0, 0
    history, start_epoch, global_step, elapsed_before = [], 1, 0, 0.0
    if args.resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if saved["run_config"] != config:
            raise ValueError("last checkpoint protocol differs")
        load_checkpoint_state(model, saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        best_mae = saved["best_valid_mae_so_far"]
        best_epoch = saved["best_epoch_so_far"]
        stale = saved["stale_epochs"]
        history = saved["training_history_so_far"]
        start_epoch = saved["epoch"] + 1
        global_step = saved["global_step"]
        elapsed_before = saved["elapsed_seconds"]
        restore_rng(saved["rng_state"], generator)

    started = time.time()
    atomic_json({"status": "training", "start_epoch": start_epoch}, output / "status.json")
    (output / "checkpoints").mkdir(exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= args.patience:
            break
        epoch_started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        totals: dict[str, float] = {}
        total_weight = 0.0
        weight_accumulator = {
            "coordinate_mass": 0.0,
            "weight_sum": 0.0,
            "weight_square_sum": 0.0,
            "weight_min": math.inf,
            "weight_max": -math.inf,
            "sample_mass": 0.0,
            "entropy_sum": 0.0,
            "effective_sum": 0.0,
            "gate_sum": 0.0,
            "gate_active_sum": 0.0,
        }
        last_norm = math.nan
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                outputs = model(**batch_inputs(batch, device))
                loss, components, coordinate_weight, gate = loss_components(
                    model, outputs, batch, device, args, assets, gate_median
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            last_norm = float(norm)
            batch_weight = float(batch["weights"].sum())
            total_weight += batch_weight
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_weight
            update_weight_accumulator(
                weight_accumulator, coordinate_weight, gate, batch["weights"]
            )
            if step == 1 or step % args.progress_every == 0:
                progress = {
                    "status": "training",
                    "method": args.method,
                    "epoch": epoch,
                    "step": step,
                    "steps": len(train_loader),
                    "global_step": global_step,
                    "loss": float(loss.detach()),
                    "elapsed_seconds": elapsed_before + time.time() - started,
                }
                atomic_json(progress, output / "status.json")
                print(json.dumps(progress), flush=True)

        valid_metrics, _ = evaluate(model, valid_loader, device)
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / total_weight for name, value in totals.items()},
            **finalize_weight_accumulator(weight_accumulator),
            "valid_metrics": valid_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm_last": last_norm,
            "epoch_wall_time_seconds": time.time() - epoch_started,
            "elapsed_seconds": elapsed_before + time.time() - started,
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
            ),
        }
        history.append(row)
        improved = valid_metrics["mae"] < best_mae - 1e-5
        if improved:
            best_mae, best_epoch, stale = valid_metrics["mae"], epoch, 0
        else:
            stale += 1

        # History is durable before its matching immutable checkpoint.
        atomic_json(history, output / "history.json")
        epoch_checkpoint = output / "checkpoints" / f"epoch_{epoch:03d}.pt"
        payload = {
            "model_state_dict": checkpoint_state(model),
            "model_state_scope": "trainable_parameters_and_trainable_module_buffers",
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": None,
            "grad_scaler_state_dict": None,
            "epoch": epoch,
            "global_step": global_step,
            "best_valid_mae_so_far": best_mae,
            "best_epoch_so_far": best_epoch,
            "stale_epochs": stale,
            "training_history_so_far": history,
            "run_config": config,
            "method_config": {
                "method": args.method,
                "soft_alpha": args.soft_alpha,
                "lambda_interaction": assets["lambda_interaction"],
                "gate_train_median": gate_median if args.method == "soft_ru_gate" else None,
            },
            "random_seed": args.seed,
            "rng_state": rng_state(generator),
            "elapsed_seconds": row["elapsed_seconds"],
        }
        atomic_save(payload, epoch_checkpoint)
        if improved:
            atomic_alias(epoch_checkpoint, output / "best.pt")
        atomic_alias(epoch_checkpoint, output / "last.pt")
        atomic_json(
            {
                "status": "training",
                "completed_epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_valid_mae": best_mae,
                "stale_epochs": stale,
            },
            output / "status.json",
        )
        print(json.dumps(row), flush=True)
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            atomic_json(
                checkpoint_inventory(output, history, best_epoch),
                output / "checkpoint_inventory.json",
            )
            atomic_json({"status": "paused", "epoch": epoch}, output / "status.json")
            return

    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    load_checkpoint_state(model, best["model_state_dict"])
    valid_metrics, predictions = evaluate(model, valid_loader, device)
    temporary = output / "predictions.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    os.replace(temporary, output / "predictions.jsonl")
    inventory = checkpoint_inventory(output, history, best_epoch)
    atomic_json(inventory, output / "checkpoint_inventory.json")
    if not inventory["coverage_complete"]:
        raise RuntimeError("checkpoint audit did not cover every completed epoch")
    report = {
        "schema": "rdid-v2-mosei-development-train-report-v1",
        "method": args.method,
        "soft_alpha": args.soft_alpha,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "valid_metrics": valid_metrics,
        "train_windows": len(train_rows),
        "valid_windows": len(valid_rows),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed_before + time.time() - started,
        "checkpoint_inventory": str((output / "checkpoint_inventory.json").resolve()),
        "checkpoint_selection": "valid_mae",
        "official_test_evaluated": False,
    }
    atomic_json(report, output / "report.json")
    atomic_json(
        {"status": "complete", "best_epoch": best_epoch, "checkpoint_audit": "pass"},
        output / "status.json",
    )
    print(json.dumps(report), flush=True)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        train_rows, _ = load_rows(args)
        gate_median = train_interaction_strength_median(train_rows)
        assets = json.loads(args.assets.read_text())
        print(json.dumps(protocol(args, assets, gate_median), ensure_ascii=False, indent=2))
        return
    output = args.output.resolve()
    with run_lock(output, args.resume):
        try:
            train(args, output)
        except Exception as error:
            atomic_json({"status": "failed", "error": repr(error)}, output / "status.json")
            raise


if __name__ == "__main__":
    main()
