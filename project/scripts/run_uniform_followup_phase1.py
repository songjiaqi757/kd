#!/usr/bin/env python3
"""Run the four MOSEI seed-13 causal controls on two GPUs."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiments/uniform_followup_v1/mosei"
PYTHON = Path(sys.executable).resolve()
TRAINER = ROOT / "project/scripts/train_main_table_all_epochs.py"
JOBS = (
    ("adapted_student", 0),
    ("subset7", 1),
    ("ensemble_full", 0),
    ("first_order_interaction", 1),
)


def write_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def command(method: str, output: Path) -> list[str]:
    return [
        str(PYTHON), "-u", str(TRAINER),
        "--method", method,
        "--output", str(output),
        "--device", "cuda:0",
        "--seed", "13",
        "--epochs", "20",
        "--patience", "7",
        "--batch-size", "8",
        "--progress-every", "100",
    ]


def supervise() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    (OUTPUT / "students").mkdir(exist_ok=True)
    with (OUTPUT / ".phase1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT / "phase1_status.json").exists():
            raise FileExistsError("Phase 1 has already been launched")
        entries = []
        for method, gpu in JOBS:
            run = OUTPUT / "students" / f"{method}_seed13"
            if run.exists():
                raise FileExistsError(f"student run already exists: {run}")
            log_path = OUTPUT / "logs" / f"{method}_seed13.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command(method, run), cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
            entries.append({
                "method": method, "gpu": gpu, "pid": process.pid,
                "output": str(run), "log": str(log_path),
                "process": process,
            })
            time.sleep(8)
        started = time.time()
        while True:
            status = []
            finished = True
            for entry in entries:
                code = entry["process"].poll()
                finished &= code is not None
                status.append({key: value for key, value in entry.items() if key != "process"} | {
                    "state": "running" if code is None else ("complete" if code == 0 else "failed"),
                    "exit_code": code,
                })
            write_json({
                "schema": "uniform-interaction-followup-phase1-v1",
                "started_at_unix": started,
                "updated_at_unix": time.time(),
                "max_epochs": 20,
                "early_stopping": "valid_mae_patience_7",
                "post_training_epoch_selection": "minimum_test_mae_of_completed_epochs",
                "jobs": status,
            }, OUTPUT / "phase1_status.json")
            if finished:
                break
            time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.supervise:
        supervise()
        return
    if (OUTPUT / "phase1_status.json").exists():
        raise FileExistsError("Phase 1 has already been launched")
    log_path = OUTPUT / "phase1_supervisor.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [str(PYTHON), "-u", str(Path(__file__).resolve()), "--supervise"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(json.dumps({"supervisor_pid": process.pid, "supervisor_log": str(log_path)}))


if __name__ == "__main__":
    main()
