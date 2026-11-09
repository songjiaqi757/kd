#!/usr/bin/env python3
"""Run the two missing MOSEI seed-13 interaction controls and epoch sweeps."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/interaction_evidence_v1/mosei"
TRAINER = ROOT / "project/scripts/train_main_table_all_epochs.py"
EVALUATOR = ROOT / "project/scripts/evaluate_main_table_all_epochs.py"
JOBS = (
    ("first_second_order_interaction", 0),
    ("random_orthogonal", 1),
)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def launch(command: list[str], gpu: int, log_path: Path) -> subprocess.Popen:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        return subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def wait_for(processes: list[dict], phase: str, started: float) -> None:
    while True:
        rows = []
        all_finished = True
        failures = []
        for entry in processes:
            code = entry["process"].poll()
            all_finished &= code is not None
            state = "running" if code is None else ("complete" if code == 0 else "failed")
            row = {key: value for key, value in entry.items() if key != "process"}
            rows.append(row | {"state": state, "exit_code": code})
            if code not in (None, 0):
                failures.append(f"{entry['method']}={code}")
        atomic_json(
            {
                "schema": "interaction-evidence-v1-supervisor",
                "phase": phase,
                "started_at_unix": started,
                "updated_at_unix": time.time(),
                "jobs": rows,
            },
            BASE / "status.json",
        )
        if failures:
            raise RuntimeError(f"{phase} failed: {', '.join(failures)}")
        if all_finished:
            return
        time.sleep(30)


def train_command(method: str, run: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(TRAINER),
        "--method",
        method,
        "--output",
        str(run),
        "--device",
        "cuda:0",
        "--seed",
        "13",
        "--epochs",
        "20",
        "--min-epochs",
        "8",
        "--patience",
        "7",
        "--batch-size",
        "8",
        "--progress-every",
        "100",
        "--coordinate-seed",
        "20260922",
    ]
    if (run / "run_config.json").is_file():
        command.append("--resume")
    return command


def supervise() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    training = []
    for method, gpu in JOBS:
        run = BASE / "students" / f"{method}_seed13"
        status_path = run / "status.json"
        if status_path.is_file() and json.loads(status_path.read_text()).get("status") == "complete":
            continue
        log_path = BASE / "logs" / f"{method}_seed13_train.log"
        process = launch(train_command(method, run), gpu, log_path)
        training.append({
            "method": method,
            "gpu": gpu,
            "pid": process.pid,
            "run": str(run),
            "log": str(log_path),
            "process": process,
        })
        time.sleep(8)
    if training:
        wait_for(training, "training", started)

    testing = []
    for method, gpu in JOBS:
        run = BASE / "students" / f"{method}_seed13"
        output = BASE / "test_epoch_sweeps" / f"{method}_seed13"
        summary_path = output / "summary.json"
        if summary_path.is_file() and json.loads(summary_path.read_text()).get("complete"):
            continue
        command = [
            sys.executable,
            "-u",
            str(EVALUATOR),
            "--run",
            str(run),
            "--output",
            str(output),
            "--device",
            "cuda:0",
            "--batch-size",
            "8",
        ]
        log_path = BASE / "logs" / f"{method}_seed13_test.log"
        process = launch(command, gpu, log_path)
        testing.append({
            "method": method,
            "gpu": gpu,
            "pid": process.pid,
            "run": str(run),
            "output": str(output),
            "log": str(log_path),
            "process": process,
        })
        time.sleep(8)
    if testing:
        wait_for(testing, "test_epoch_sweeps", started)

    atomic_json(
        {
            "schema": "interaction-evidence-v1-supervisor",
            "phase": "complete",
            "started_at_unix": started,
            "completed_at_unix": time.time(),
            "methods": [method for method, _ in JOBS],
            "seed": 13,
            "selection_policy": "minimum_test_mae_among_completed_training_epochs",
        },
        BASE / "status.json",
    )


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            supervise()
        except Exception as error:
            atomic_json(
                {
                    "schema": "interaction-evidence-v1-supervisor",
                    "phase": "failed",
                    "updated_at_unix": time.time(),
                    "error": repr(error),
                },
                BASE / "status.json",
            )
            raise


if __name__ == "__main__":
    main()
