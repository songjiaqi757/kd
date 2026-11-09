#!/usr/bin/env python3
"""Wait for all live MOSI sweeps, then build final tables and statistics."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/uniform_main_v1/mosi"
METHODS = ("adapted_student", "full_kd", "subset7", "ensemble_full",
           "first_order_interaction", "uniform_interaction", "projector", "ea_kd", "cmad_cafd")


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    while True:
        done = []
        for method in METHODS:
            run = BASE / "students" / f"{method}_seed13"
            sweep = BASE / "test_sweep" / f"{method}_seed13"
            status_path = run / "status.json"
            if status_path.is_file():
                status = json.loads(status_path.read_text())
                if status.get("status") == "failed":
                    raise RuntimeError(f"training failed: {method}: {status.get('error')}")
            if (sweep / "summary.json").is_file():
                summary = json.loads((sweep / "summary.json").read_text())
                inventory = json.loads((run / "checkpoint_inventory.json").read_text())
                if summary["epochs_evaluated"] != inventory["checkpoint_epoch_count"]:
                    raise ValueError(f"checkpoint sweep coverage differs: {method}")
                done.append(method)
        atomic_json({"status": "parallel_training_and_live_sweep", "completed_methods": len(done),
                     "pending_methods": [method for method in METHODS if method not in done]},
                    BASE / "queue_status.json")
        if len(done) == len(METHODS):
            break
        time.sleep(30)
    atomic_json({"status": "summarizing", "completed_methods": len(done)}, BASE / "queue_status.json")
    log = BASE / "logs" / "summary.log"
    with log.open("a") as handle:
        subprocess.run([sys.executable, str(ROOT / "project/scripts/summarize_mosi_uniform.py"),
                        "--base", str(BASE), "--resamples", "10000"],
                       check=True, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    atomic_json({"status": "complete", "completed_methods": len(done),
                 "summary": str(BASE / "summary")}, BASE / "queue_status.json")
    print(json.dumps({"status": "complete", "methods": len(done)}), flush=True)


if __name__ == "__main__":
    main()
