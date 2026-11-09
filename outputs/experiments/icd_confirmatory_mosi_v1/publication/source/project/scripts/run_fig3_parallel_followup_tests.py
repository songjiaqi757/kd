#!/usr/bin/env python3
"""Evaluate the two manually parallelized Qiii continuations when ready."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/fig3_full20_v1"
TEST = ROOT / "dataset/cmu_mosei/manifests/official_test_windowed.jsonl"
JOBS = (
    {
        "name": "mosei_adapted_student", "gpu": 0, "kind": "main_table",
        "run": ROOT / "outputs/experiments/uniform_followup_v1/mosei/students/adapted_student_seed13",
        "output": ROOT / "outputs/experiments/uniform_followup_v1/mosei/test_epoch_sweeps/adapted_student_seed13",
    },
    {
        "name": "mosei_subset7", "gpu": 0, "kind": "main_table",
        "run": ROOT / "outputs/experiments/uniform_followup_v1/mosei/students/subset7_seed13",
        "output": ROOT / "outputs/experiments/uniform_followup_v1/mosei/test_epoch_sweeps/subset7_seed13",
    },
    {
        "name": "mosei_full_kd", "gpu": 1, "kind": "tav",
        "run": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/students/M3_seed13",
        "output": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/checkpoint_diagnostic_test_gpu1_20260919/M3_seed13",
    },
    {
        "name": "mosei_uniform", "gpu": 1, "kind": "tav",
        "run": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/students/M4_seed13",
        "output": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/checkpoint_diagnostic_test_gpu1_20260919/M4_seed13",
    },
)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def archive(path: Path) -> None:
    if not path.exists():
        return
    destination = path.with_name(path.name + ".before_full20")
    if destination.exists():
        destination = path.with_name(path.name + f".before_full20_retry_{int(time.time())}")
    os.replace(path, destination)


def worker(job: dict) -> None:
    continuation = Path(job["run"]) / "full_epoch_continuation.json"
    status_path = BASE / f"{job['name']}_parallel_test_status.json"
    while True:
        status = json.loads(continuation.read_text()) if continuation.is_file() else {}
        state = status.get("status")
        atomic_json({"status": "waiting", "continuation": state,
                     "updated_at_unix": time.time()}, status_path)
        if state == "failed":
            raise RuntimeError(f"continuation failed: {status}")
        if state == "complete":
            break
        time.sleep(30)
    output = Path(job["output"])
    archive(output / "evaluation_plan.json")
    if job["kind"] == "tav":
        archive(output / "checkpoints" / "last")
        archive(output / "checkpoints" / "best")
        command = [
            sys.executable, "-u", str(ROOT / "project/scripts/evaluate_tav_all_checkpoints.py"),
            "--run", str(job["run"]), "--output", str(output),
            "--test-manifest", str(TEST), "--device", "cuda:0", "--batch-size", "8",
            "--num-workers", "2", "--allow-diagnostic-test",
        ]
    else:
        command = [
            sys.executable, "-u", str(ROOT / "project/scripts/evaluate_main_table_all_epochs.py"),
            "--run", str(job["run"]), "--output", str(output),
            "--test-manifest", str(TEST), "--device", "cuda:0", "--batch-size", "8",
            "--num-workers", "2",
        ]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
                   "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                   "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    atomic_json({"status": "testing", "gpu": job["gpu"],
                 "updated_at_unix": time.time()}, status_path)
    with (BASE / "logs" / f"{job['name']}_parallel_test.log").open("a") as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    atomic_json({"status": "complete", "gpu": job["gpu"],
                 "completed_at_unix": time.time()}, status_path)


def main() -> None:
    remote_status = BASE / "remote_status.json"
    atomic_json({"schema": "fig3-full20-remote-v1", "status": "running",
                 "mode": "four_training_and_test_jobs_parallel",
                 "updated_at_unix": time.time()}, remote_status)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, job) for job in JOBS]
            for future in futures:
                future.result()
        atomic_json({"status": "complete", "completed_at_unix": time.time()},
                    BASE / "parallel_test_status.json")
        atomic_json({"schema": "fig3-full20-remote-v1", "status": "complete",
                     "mode": "four_training_and_test_jobs_parallel",
                     "completed_at_unix": time.time()}, remote_status)
    except Exception as error:
        atomic_json({"status": "failed", "error": repr(error),
                     "updated_at_unix": time.time()}, BASE / "parallel_test_status.json")
        atomic_json({"schema": "fig3-full20-remote-v1", "status": "failed",
                     "error": repr(error), "updated_at_unix": time.time()}, remote_status)
        raise


if __name__ == "__main__":
    main()
