#!/usr/bin/env python3
"""Recover the Qiii MOSEI queues without duplicating active training jobs."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/ai/sjq/kd/outputs/experiments/multiseed_controls_v1/mosei")
FOLLOWUP = BASE / "followup_order_controls"
CONTROLLER = ROOT / "project/scripts/run_multiseed_controls.py"
JOB_RUNNER = ROOT / "project/scripts/run_multiseed_control_job.py"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def job_complete(base: Path, name: str) -> bool:
    path = base / "jobs" / f"{name}.json"
    if not path.is_file():
        return False
    try:
        return read_json(path).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def process_alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.is_file():
        return False
    try:
        return stat.read_text().split()[2] != "Z"
    except (OSError, IndexError):
        return False


def controller_command(base: Path, methods: list[str]) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(CONTROLLER),
        "--dataset",
        "mosei",
        "--base",
        str(base),
        "--methods",
        *methods,
        "--seeds",
        "42",
        "2026",
        "--max-jobs",
        "4",
        "--slots-per-gpu",
        "2",
        "--job-budget-gib",
        "30",
        "--headroom-gib",
        "8",
    ]


def archive_unresumable_runs(base: Path, names: list[str]) -> None:
    archive = base / "archive" / f"precheckpoint_failure_{int(time.time())}"
    for name in names:
        run = base / "students" / name
        if not run.exists() or (run / "last.pt").is_file() or job_complete(base, name):
            continue
        archive.mkdir(parents=True, exist_ok=True)
        target = archive / name
        if target.exists():
            raise FileExistsError(target)
        shutil.move(run, target)
        print(f"archived unresumable partial run: {run} -> {target}", flush=True)


def main() -> None:
    lock_path = BASE / ".recovery.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        failed_name = "subset7_seed2026"
        if not job_complete(BASE, failed_name):
            print(f"retrying official test for {failed_name}", flush=True)
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "PYTHONUNBUFFERED": "1",
            }
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(JOB_RUNNER),
                    "--dataset",
                    "mosei",
                    "--method",
                    "subset7",
                    "--seed",
                    "2026",
                    "--base",
                    str(BASE),
                    "--device",
                    "cuda:0",
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )

        archive_unresumable_runs(
            FOLLOWUP,
            [
                "first_second_order_interaction_seed42",
                "first_order_interaction_seed42",
                "first_second_order_interaction_seed2026",
                "first_order_interaction_seed2026",
            ],
        )
        followup_log_path = FOLLOWUP / "logs/controller_recovery.log"
        followup_log_path.parent.mkdir(parents=True, exist_ok=True)
        followup_log = followup_log_path.open("a")
        followup = subprocess.Popen(
            controller_command(
                FOLLOWUP,
                ["first_second_order_interaction", "first_order_interaction"],
            ),
            cwd=ROOT,
            stdout=followup_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"started follow-up controller pid={followup.pid}", flush=True)

        old_status = read_json(BASE / "queue_status.json")
        old_pids = [int(item["pid"]) for item in old_status.get("running", {}).values()]
        while any(process_alive(pid) for pid in old_pids):
            time.sleep(30)

        print("original main jobs exited; reconciling main queue", flush=True)
        subprocess.run(
            controller_command(BASE, ["subset7", "random_orthogonal"]),
            cwd=ROOT,
            check=True,
        )
        followup_code = followup.wait()
        followup_log.close()
        if followup_code:
            raise subprocess.CalledProcessError(followup_code, followup.args)
        print("main and follow-up queues complete", flush=True)


if __name__ == "__main__":
    main()
