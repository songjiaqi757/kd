#!/usr/bin/env python3
"""Finalize six-method MOSI interaction evidence after live test sweeps."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/interaction_evidence_v1/mosi"
REPORT = ROOT / "project/reports/interaction_evidence_v1/mosi_interaction_evidence_valid.json"
STATUS = ROOT / "project/reports/interaction_evidence_v1/mosi_finalizer_status.json"
ANALYZER = ROOT / "project/scripts/analyze_interaction_reconstruction.py"
PLOTTER = ROOT / "project/scripts/plot_interaction_evidence.py"
METHODS = ("first_second_order_interaction", "random_orthogonal")


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def summaries() -> dict[str, dict]:
    result = {}
    for method in METHODS:
        path = BASE / "test_epoch_sweeps" / f"{method}_seed13" / "summary.json"
        if path.exists():
            result[method] = json.loads(path.read_text())
    return result


def wait_for_sweeps() -> dict[str, dict]:
    while True:
        current = summaries()
        atomic_json({
            "schema": "mosi-interaction-evidence-finalizer-v1",
            "phase": "waiting_for_live_test_sweeps",
            "methods": {
                method: {
                    "present": method in current,
                    "epochs_available": current.get(method, {}).get("epochs_available"),
                    "epochs_evaluated": current.get(method, {}).get("epochs_evaluated"),
                    "complete": current.get(method, {}).get("complete", False),
                }
                for method in METHODS
            },
            "updated_at_unix": time.time(),
        }, STATUS)
        if len(current) == len(METHODS) and all(
            current[method].get("complete") for method in METHODS
        ):
            return current
        time.sleep(30)


def checkpoint_arguments(current: dict[str, dict]) -> list[str]:
    result = []
    for method in METHODS:
        epoch = int(current[method]["selected_epoch"])
        checkpoint = (
            BASE / "students" / f"{method}_seed13" / "checkpoints" /
            f"epoch_{epoch:03d}.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result.extend(["--checkpoint", f"{method}={checkpoint}"])
    return result


def main() -> None:
    try:
        current = wait_for_sweeps()
        atomic_json({
            "schema": "mosi-interaction-evidence-finalizer-v1",
            "phase": "reconstruction_analysis",
            "updated_at_unix": time.time(),
        }, STATUS)
        subprocess.run([
            sys.executable, "-u", str(ANALYZER),
            "--dataset", "mosi", "--device", "cuda:0", "--batch-size", "8",
            "--output", str(REPORT), *checkpoint_arguments(current),
        ], cwd=ROOT, check=True)
        atomic_json({
            "schema": "mosi-interaction-evidence-finalizer-v1",
            "phase": "plotting",
            "updated_at_unix": time.time(),
        }, STATUS)
        subprocess.run([
            sys.executable, "-u", str(PLOTTER), "--input", str(REPORT)
        ], cwd=ROOT, check=True)
        atomic_json({
            "schema": "mosi-interaction-evidence-finalizer-v1",
            "phase": "complete",
            "report": str(REPORT),
            "aggregate_figure": str(
                ROOT / "project/reports/paper_figures/"
                "fig_aggregate_interaction_reconstruction_mosi.pdf"
            ),
            "completed_at_unix": time.time(),
        }, STATUS)
    except Exception as error:
        atomic_json({
            "schema": "mosi-interaction-evidence-finalizer-v1",
            "phase": "failed",
            "error": repr(error),
            "updated_at_unix": time.time(),
        }, STATUS)
        raise


if __name__ == "__main__":
    main()
