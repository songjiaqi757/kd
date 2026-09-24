#!/usr/bin/env python3
"""Train a main-table method while retaining every completed epoch model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/scripts"))
import train_main_table_kd as trainer

ORIGINAL_ATOMIC_SAVE = trainer.atomic_save
ORIGINAL_RUN_PROTOCOL = trainer.run_protocol


def output_arg() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    return args.output.resolve()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def protocol_with_epoch_retention(args, assets):
    config = ORIGINAL_RUN_PROTOCOL(args, assets)
    config["training_variant"] = "retain-every-completed-epoch-v1"
    config["epoch_checkpoint_pattern"] = "checkpoints/epoch_{epoch:03d}.pt"
    config["epoch_checkpoint_selection"] = "validation_mae_minimum"
    config["input_sha256"][str(Path(__file__).resolve())] = trainer.sha256(__file__)
    return config


def atomic_save_with_epoch_retention(payload, path):
    path = Path(path)
    if path.name != "last.pt":
        ORIGINAL_ATOMIC_SAVE(payload, path)
        return
    epoch = int(payload["epoch"])
    row = payload["history"][-1]
    if int(row["epoch"]) != epoch:
        raise ValueError("checkpoint history does not end at its epoch")
    checkpoints = path.parent / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    epoch_payload = {
        "model": payload["model"],
        "epoch": epoch,
        "protocol": payload["protocol"],
        "valid_metrics": row["valid_metrics"],
        "valid_best_epoch_after_epoch": payload["best_epoch"],
    }
    # Write the epoch model first; if interrupted before last.pt, resume
    # repeats this epoch and replaces its model atomically.
    ORIGINAL_ATOMIC_SAVE(epoch_payload, checkpoints / f"epoch_{epoch:03d}.pt")
    ORIGINAL_ATOMIC_SAVE(payload, path)


def audit(output: Path) -> None:
    status = output / "status.json"
    if not status.exists() or json.loads(status.read_text()).get("status") != "complete":
        return
    history = json.loads((output / "history.json").read_text())
    expected = [int(row["epoch"]) for row in history]
    paths = sorted((output / "checkpoints").glob("epoch_*.pt"))
    actual = [int(path.stem.split("_")[-1]) for path in paths]
    if expected != actual:
        raise ValueError(f"checkpoint coverage differs: expected={expected}, actual={actual}")
    config = json.loads((output / "run_config.json").read_text())
    epochs = []
    for path, epoch in zip(paths, actual):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("epoch") != epoch or saved.get("protocol") != config:
            raise ValueError(f"checkpoint protocol mismatch: {path}")
        epochs.append({
            "epoch": epoch,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": trainer.sha256(path),
            "valid_metrics": saved["valid_metrics"],
        })
    atomic_json({
        "schema": "uniform-interaction-followup-epoch-inventory-v1",
        "epoch_checkpoint_selection": "validation_mae_minimum",
        "epochs": epochs,
    }, output / "checkpoint_inventory.json")


def main() -> None:
    output = output_arg()
    trainer.run_protocol = protocol_with_epoch_retention
    trainer.atomic_save = atomic_save_with_epoch_retention
    trainer.main()
    audit(output)


if __name__ == "__main__":
    main()
