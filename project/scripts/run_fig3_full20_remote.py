#!/usr/bin/env python3
"""Run the four MOSEI full-20 continuations and missing test sweeps on Qiii."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
CONTINUE = ROOT / "project/scripts/continue_early_stopped_run.py"
MAIN_EVAL = ROOT / "project/scripts/evaluate_main_table_all_epochs.py"
TAV_EVAL = ROOT / "project/scripts/evaluate_tav_all_checkpoints.py"
TEST_MANIFEST = ROOT / "dataset/cmu_mosei/manifests/official_test_windowed.jsonl"
STATUS = ROOT / "outputs/experiments/fig3_full20_v1/remote_status.json"

JOBS = {
    0: (
        {
            "name": "mosei_adapted_student", "kind": "main_table",
            "run": ROOT / "outputs/experiments/uniform_followup_v1/mosei/students/adapted_student_seed13",
            "output": ROOT / "outputs/experiments/uniform_followup_v1/mosei/test_epoch_sweeps/adapted_student_seed13",
            "assets": ROOT / "outputs/experiments/main_table_v1/mosei/assets/protocol.json",
        },
        {
            "name": "mosei_subset7", "kind": "main_table",
            "run": ROOT / "outputs/experiments/uniform_followup_v1/mosei/students/subset7_seed13",
            "output": ROOT / "outputs/experiments/uniform_followup_v1/mosei/test_epoch_sweeps/subset7_seed13",
            "assets": ROOT / "outputs/experiments/main_table_v1/mosei/assets/protocol.json",
        },
    ),
    1: (
        {
            "name": "mosei_full_kd", "kind": "tav",
            "run": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/students/M3_seed13",
            "output": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/checkpoint_diagnostic_test_gpu1_20260919/M3_seed13",
            "assets": ROOT / "outputs/experiments/tav_main_v1/assets/protocol.json",
        },
        {
            "name": "mosei_uniform", "kind": "tav",
            "run": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/students/M4_seed13",
            "output": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/checkpoint_diagnostic_test_gpu1_20260919/M4_seed13",
            "assets": ROOT / "outputs/experiments/tav_main_v1/assets/protocol.json",
        },
    ),
}


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
        os.replace(path, path.with_name(path.name + f".before_full20_retry_{int(time.time())}"))
    else:
        os.replace(path, destination)


def run_command(command: list[str], gpu: int, log_path: Path) -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def run_job(job: dict, gpu: int) -> None:
    log_root = STATUS.parent / "logs"
    run_command([
        str(PYTHON), "-u", str(CONTINUE), "--kind", job["kind"],
        "--run", str(job["run"]), "--assets", str(job["assets"]),
        "--target-epoch", "20",
    ], gpu, log_root / f"{job['name']}_train.log")
    output = Path(job["output"])
    archive(output / "evaluation_plan.json")
    if job["kind"] == "tav":
        archive(output / "checkpoints" / "last")
        archive(output / "checkpoints" / "best")
        command = [
            str(PYTHON), "-u", str(TAV_EVAL), "--run", str(job["run"]),
            "--output", str(output), "--test-manifest", str(TEST_MANIFEST),
            "--device", "cuda:0", "--batch-size", "8", "--num-workers", "2",
            "--allow-diagnostic-test",
        ]
    else:
        command = [
            str(PYTHON), "-u", str(MAIN_EVAL), "--run", str(job["run"]),
            "--output", str(output), "--test-manifest", str(TEST_MANIFEST),
            "--device", "cuda:0", "--batch-size", "8", "--num-workers", "2",
        ]
    run_command(command, gpu, log_root / f"{job['name']}_test.log")


def gpu_queue(gpu: int, jobs: tuple[dict, ...]) -> None:
    for job in jobs:
        atomic_json({
            "schema": "fig3-full20-remote-v1", "status": "running",
            "gpu": gpu, "job": job["name"], "stage": "train_then_test",
            "updated_at_unix": time.time(),
        }, STATUS.parent / f"gpu{gpu}_status.json")
        run_job(job, gpu)
    atomic_json({
        "schema": "fig3-full20-remote-v1", "status": "complete",
        "gpu": gpu, "jobs": [job["name"] for job in jobs],
        "completed_at_unix": time.time(),
    }, STATUS.parent / f"gpu{gpu}_status.json")


def main() -> None:
    atomic_json({
        "schema": "fig3-full20-remote-v1", "status": "running",
        "started_at_unix": time.time(),
    }, STATUS)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(gpu_queue, gpu, jobs) for gpu, jobs in JOBS.items()]
            for future in futures:
                future.result()
        atomic_json({
            "schema": "fig3-full20-remote-v1", "status": "complete",
            "completed_at_unix": time.time(),
        }, STATUS)
    except Exception as error:
        atomic_json({
            "schema": "fig3-full20-remote-v1", "status": "failed",
            "error": repr(error), "updated_at_unix": time.time(),
        }, STATUS)
        raise


if __name__ == "__main__":
    main()
