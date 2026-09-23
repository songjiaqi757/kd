#!/usr/bin/env python3
"""Finalize validation-selected MOSEI interaction evidence after epoch 20."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/interaction_evidence_v1/mosei"
REPORT = ROOT / "project/reports/interaction_evidence_v1/mosei_interaction_evidence_valid.json"
MOSI_REPORT = ROOT / "project/reports/interaction_evidence_v1/mosi_interaction_evidence_valid.json"
STATUS = ROOT / "project/reports/interaction_evidence_v1/mosei_validation_finalizer_status.json"
ANALYZER = ROOT / "project/scripts/analyze_interaction_reconstruction.py"
PLOTTER = ROOT / "project/scripts/plot_interaction_evidence.py"
OUTPUT = ROOT / "project/reports/paper_figures"
RUNS = {
    "full_kd": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/students/M3_seed13",
    "subset7": ROOT / "outputs/experiments/uniform_followup_v1/mosei/students/subset7_seed13",
    "first_order_interaction": ROOT / "outputs/experiments/uniform_followup_v1/mosei/students/first_order_interaction_seed13",
    "uniform_interaction": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/students/M4_seed13",
    "first_second_order_interaction": BASE / "students/first_second_order_interaction_seed13",
    "random_orthogonal": BASE / "students/random_orthogonal_seed13",
}


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def wait_for_complete_sweep() -> None:
    training = RUNS["first_second_order_interaction"] / "status.json"
    sweep = BASE / "test_epoch_sweeps/first_second_order_interaction_seed13/summary.json"
    while True:
        training_status = read_json(training) if training.exists() else {}
        sweep_status = read_json(sweep) if sweep.exists() else {}
        atomic_json({
            "schema": "mosei-validation-interaction-finalizer-v1",
            "phase": "waiting_for_epoch20_and_test_sweep",
            "training_status": training_status.get("status"),
            "training_epoch": training_status.get("epoch"),
            "training_step": training_status.get("step"),
            "epochs_available": sweep_status.get("epochs_available"),
            "epochs_evaluated": sweep_status.get("epochs_evaluated"),
            "sweep_complete": sweep_status.get("complete", False),
            "updated_at_unix": time.time(),
        }, STATUS)
        if training_status.get("status") == "complete" and sweep_status.get("complete"):
            return
        time.sleep(30)


def best_validation_checkpoint(run: Path) -> Path:
    history = read_json(run / "history.json")
    rows = history if isinstance(history, list) else history.get("epochs", history.get("history", []))
    if not rows:
        raise ValueError(f"empty history: {run}")
    best = min(rows, key=lambda row: (float(row["valid_metrics"]["mae"]), int(row["epoch"])))
    checkpoint = run / "checkpoints" / f"epoch_{int(best['epoch']):03d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def main() -> None:
    try:
        wait_for_complete_sweep()
        checkpoints = {method: best_validation_checkpoint(run) for method, run in RUNS.items()}
        atomic_json({
            "schema": "mosei-validation-interaction-finalizer-v1",
            "phase": "reconstruction_analysis",
            "checkpoints": {method: str(path) for method, path in checkpoints.items()},
            "updated_at_unix": time.time(),
        }, STATUS)
        checkpoint_args = [
            item
            for method, path in checkpoints.items()
            for item in ("--checkpoint", f"{method}={path}")
        ]
        subprocess.run([
            sys.executable, "-u", str(ANALYZER),
            "--dataset", "mosei", "--device", "cuda:0", "--batch-size", "8",
            "--checkpoint-policy", "validation_mae_minimum_among_completed_epochs",
            "--reuse-from", str(REPORT), "--output", str(REPORT), *checkpoint_args,
        ], cwd=ROOT, check=True)
        atomic_json({
            "schema": "mosei-validation-interaction-finalizer-v1",
            "phase": "plotting",
            "updated_at_unix": time.time(),
        }, STATUS)
        subprocess.run([
            sys.executable, "-u", str(PLOTTER), "--input", str(REPORT),
            "--compare-input", str(MOSI_REPORT), "--output-dir", str(OUTPUT),
        ], cwd=ROOT, check=True)
        atomic_json({
            "schema": "mosei-validation-interaction-finalizer-v1",
            "phase": "complete",
            "report": str(REPORT),
            "aggregate_figure": str(OUTPUT / "fig_aggregate_interaction_reconstruction_both.pdf"),
            "completed_at_unix": time.time(),
        }, STATUS)
    except Exception as error:
        atomic_json({
            "schema": "mosei-validation-interaction-finalizer-v1",
            "phase": "failed",
            "error": repr(error),
            "updated_at_unix": time.time(),
        }, STATUS)
        raise


if __name__ == "__main__":
    main()
