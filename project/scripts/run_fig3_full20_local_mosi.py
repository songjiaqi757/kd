#!/usr/bin/env python3
"""Continue MOSI Subset-7 to epoch 20 and evaluate the two new checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/uniform_main_v1/mosi"
RUN = BASE / "students/subset7_seed13"
OUTPUT = BASE / "test_sweep/subset7_seed13"
STATUS = ROOT / "outputs/experiments/fig3_full20_v1/local_mosi_status.json"


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    started = time.time()
    atomic_json({"schema": "fig3-full20-local-mosi-v1", "status": "training",
                 "started_at_unix": started}, STATUS)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
                   "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                   "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"}
    log_root = STATUS.parent / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        with (log_root / "mosi_subset7_train.log").open("a") as log:
            subprocess.run([
                sys.executable, "-u", str(ROOT / "project/scripts/continue_early_stopped_run.py"),
                "--kind", "main_table", "--run", str(RUN),
                "--assets", str(ROOT / "outputs/experiments/main_table_v1/mosi/assets/protocol.json"),
                "--target-epoch", "20",
            ], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        atomic_json({"schema": "fig3-full20-local-mosi-v1", "status": "testing",
                     "updated_at_unix": time.time()}, STATUS)
        with (log_root / "mosi_subset7_test.log").open("a") as log:
            subprocess.run([
                sys.executable, "-u", str(ROOT / "project/scripts/evaluate_mosi_uniform_sweep.py"),
                "--run", str(RUN), "--output", str(OUTPUT),
                "--test-manifest", str(ROOT / "dataset/cmu_mosi/manifests/official_test_windowed.jsonl"),
                "--device", "cuda:0", "--batch-size", "8", "--num-workers", "2",
            ], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((OUTPUT / "summary.json").read_text())
        if summary.get("epochs_evaluated") != 20:
            raise ValueError("MOSI Subset-7 sweep did not reach 20 epochs")
        atomic_json({"schema": "fig3-full20-local-mosi-v1", "status": "complete",
                     "completed_at_unix": time.time()}, STATUS)
    except Exception as error:
        atomic_json({"schema": "fig3-full20-local-mosi-v1", "status": "failed",
                     "error": repr(error), "updated_at_unix": time.time()}, STATUS)
        raise


if __name__ == "__main__":
    main()
