#!/usr/bin/env python3
"""Run the frozen TAV trainer while retaining a model checkpoint per epoch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/scripts"))
import train_tav_distillation as trainer

ORIGINAL_ATOMIC_SAVE = trainer.atomic_save
ORIGINAL_PROTOCOL = trainer.protocol


def wrapper_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    return args


def atomic_json(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def protocol_with_epoch_retention(args):
    config = ORIGINAL_PROTOCOL(args)
    wrapper = Path(__file__).resolve()
    config["training_variant"] = "retain-model-checkpoint-after-every-completed-epoch-v1"
    config["epoch_checkpoint_pattern"] = "checkpoints/epoch_{epoch:03d}.pt"
    config["epoch_checkpoint_contents"] = "trainable_model_state_protocol_epoch_and_valid_selection_metadata"
    config["epoch_checkpoint_excludes"] = ["optimizer_state", "rng_state"]
    config["input_sha256"][str(wrapper)] = trainer.sha256(wrapper)
    return config


def atomic_save_with_epoch_retention(payload, path):
    path = Path(path)
    if path.name != "last.pt":
        ORIGINAL_ATOMIC_SAVE(payload, path)
        return
    epoch = int(payload["epoch"])
    checkpoints = path.parent / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    history_row = payload["history"][-1]
    if int(history_row["epoch"]) != epoch:
        raise ValueError("last checkpoint history does not end at its epoch")
    epoch_payload = {
        "model": payload["model"],
        "epoch": epoch,
        "protocol": payload["protocol"],
        "valid_metrics": history_row["valid_metrics"],
        "best_mae_after_epoch": payload["best_mae"],
        "best_epoch_after_epoch": payload["best_epoch"],
        "stale_after_epoch": payload["stale"],
        "checkpoint_role": "completed_epoch_model_for_analysis_not_test_selection",
    }
    # The epoch snapshot is written before the resumable checkpoint. If a
    # process stops between the two atomic writes, resuming from the preceding
    # last.pt safely reruns and atomically replaces this epoch snapshot.
    ORIGINAL_ATOMIC_SAVE(epoch_payload, checkpoints / f"epoch_{epoch:03d}.pt")
    ORIGINAL_ATOMIC_SAVE(payload, path)


def audit_completed_output(output):
    status_path = output / "status.json"
    if not status_path.is_file() or json.loads(status_path.read_text()).get("status") != "complete":
        return
    history = json.loads((output / "history.json").read_text())
    expected_epochs = [int(row["epoch"]) for row in history]
    files = sorted((output / "checkpoints").glob("epoch_*.pt"))
    actual_epochs = [int(path.stem.split("_")[-1]) for path in files]
    if actual_epochs != expected_epochs:
        raise ValueError(f"epoch checkpoint coverage differs: expected={expected_epochs}, actual={actual_epochs}")
    run_config = json.loads((output / "run_config.json").read_text())
    inventory = []
    for path, epoch in zip(files, actual_epochs):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("epoch") != epoch or saved.get("protocol") != run_config:
            raise ValueError(f"epoch checkpoint protocol mismatch: {path}")
        inventory.append({
            "epoch": epoch,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": trainer.sha256(path),
            "valid_metrics": saved["valid_metrics"],
            "is_best_epoch": epoch == json.loads((output / "report.json").read_text())["best_epoch"],
        })
    atomic_json({
        "schema": "rdid-msa-tav-epoch-checkpoint-inventory-v1",
        "selection_policy": "best.pt remains selected only by valid_mae",
        "epoch_checkpoints_are_for_analysis_not_official_test_selection": True,
        "epochs": inventory,
    }, output / "checkpoint_inventory.json")


def main():
    output = wrapper_args().output.resolve()
    trainer.protocol = protocol_with_epoch_retention
    trainer.atomic_save = atomic_save_with_epoch_retention
    trainer.main()
    try:
        audit_completed_output(output)
    except Exception as exc:
        atomic_json({"status": "failed", "stage": "epoch_checkpoint_audit", "error": repr(exc)}, output / "status.json")
        raise


if __name__ == "__main__":
    main()
