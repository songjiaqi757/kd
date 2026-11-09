#!/usr/bin/env python3
"""Wait for persistent local storage, then run and summarize MOSI controls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "outputs/experiments/multiseed_controls_v1/mosi"
STATUS = ROOT / "outputs/experiments/multiseed_controls_v1/local_mosi_launcher_status.json"


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parent = ROOT
    while not parent.is_dir() or not os.access(parent, os.W_OK):
        atomic_json({
            "status": "waiting_for_storage_permission",
            "required_directory": str(parent),
            "required_free_gib": 80,
            "updated_at_unix": time.time(),
        }, STATUS)
        time.sleep(60)
    free_gib = shutil.disk_usage(parent).free / 1024**3
    if free_gib < 80:
        raise OSError(f"only {free_gib:.1f} GiB free at {parent}; need at least 80 GiB")
    atomic_json({
        "status": "launching", "destination": str(DESTINATION),
        "free_gib": free_gib, "updated_at_unix": time.time(),
    }, STATUS)
    subprocess.run([
        sys.executable, "-u", str(ROOT / "project/scripts/run_multiseed_controls.py"),
        "--dataset", "mosi", "--base", str(DESTINATION),
        "--methods", "full_kd", "subset7", "first_second_order_interaction",
        "random_orthogonal", "uniform_interaction",
        "--seeds", "42", "2026",
        "--job-budget-gib", "32",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, str(ROOT / "project/scripts/summarize_multiseed_controls.py"),
        "--base", str(DESTINATION),
    ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    atomic_json({
        "status": "complete", "destination": str(DESTINATION),
        "summary": str(DESTINATION / "summary.json"),
        "completed_at_unix": time.time(),
    }, STATUS)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json({
            "status": "failed", "error": repr(error),
            "updated_at_unix": time.time(),
        }, STATUS)
        raise
