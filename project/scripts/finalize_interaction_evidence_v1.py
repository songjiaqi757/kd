#!/usr/bin/env python3
"""Wait for the two missing sweeps, then rebuild final interaction evidence."""
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
STATUS = ROOT / "project/reports/interaction_evidence_v1/finalizer_status.json"
ANALYZER = ROOT / "project/scripts/analyze_interaction_reconstruction.py"
PLOTTER = ROOT / "project/scripts/plot_interaction_evidence.py"
METHODS = ("first_second_order_interaction", "random_orthogonal")


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def selected_epoch(summary: dict) -> int:
    for key in ("selected_epoch", "best_epoch_by_test_mae", "best_epoch"):
        if key in summary:
            return int(summary[key])
    rows = summary.get("epochs") or summary.get("checkpoints") or []
    rows = [row for row in rows if "test_metrics" in row]
    if not rows:
        raise ValueError("test sweep has no evaluated checkpoints")
    return int(min(rows, key=lambda row: float(row["test_metrics"]["mae"]))["epoch"])


def wait_for_supervisor() -> None:
    while True:
        if (BASE / "status.json").is_file():
            status = json.loads((BASE / "status.json").read_text())
            phase = status.get("phase")
            atomic_json({
                "schema": "interaction-evidence-v1-finalizer",
                "phase": "waiting_for_sweeps",
                "supervisor_phase": phase,
                "updated_at_unix": time.time(),
            }, STATUS)
            if phase == "complete":
                return
            if phase == "failed":
                raise RuntimeError(f"experiment supervisor failed: {status}")
        time.sleep(60)


def checkpoint_arguments() -> list[str]:
    arguments: list[str] = []
    for method in METHODS:
        summary_path = BASE / "test_epoch_sweeps" / f"{method}_seed13" / "summary.json"
        summary = json.loads(summary_path.read_text())
        epoch = selected_epoch(summary)
        checkpoint = (
            BASE / "students" / f"{method}_seed13" / "checkpoints" /
            f"epoch_{epoch:03d}.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        arguments.extend(["--checkpoint", f"{method}={checkpoint}"])
    return arguments


def main() -> None:
    try:
        wait_for_supervisor()
        atomic_json({
            "schema": "interaction-evidence-v1-finalizer",
            "phase": "reconstruction_analysis",
            "updated_at_unix": time.time(),
        }, STATUS)
        subprocess.run([
            sys.executable, "-u", str(ANALYZER),
            "--device", "cuda:0", "--batch-size", "8",
            "--output", str(REPORT), *checkpoint_arguments(),
        ], cwd=ROOT, check=True)
        atomic_json({
            "schema": "interaction-evidence-v1-finalizer",
            "phase": "plotting",
            "updated_at_unix": time.time(),
        }, STATUS)
        subprocess.run([
            sys.executable, "-u", str(PLOTTER), "--input", str(REPORT)
        ], cwd=ROOT, check=True)
        atomic_json({
            "schema": "interaction-evidence-v1-finalizer",
            "phase": "complete",
            "report": str(REPORT),
            "completed_at_unix": time.time(),
        }, STATUS)
    except Exception as error:
        atomic_json({
            "schema": "interaction-evidence-v1-finalizer",
            "phase": "failed",
            "error": repr(error),
            "updated_at_unix": time.time(),
        }, STATUS)
        raise


if __name__ == "__main__":
    main()
