#!/usr/bin/env python3
"""Promote Video Adaptation seeds 42/2026 in parallel after the seed-13 gate."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/scripts"))

from run_video_adaptation_queue import MODES, summarize
from train_video_adaptation import atomic_json, sha256

OUTPUT = ROOT / "outputs/student/video_adaptation_v1"
ORIGINAL_UNIT = "rdid-video-adaptation-v1.service"
GPU0_UNITS = (
    "rdid-cv2-cminus1-tempfix-gpu0.service",
    "rdid-cv2-c1-remaining-gpu0.service",
)
SEED_GPU = {42: 0, 2026: 1}


def unit_active(unit: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() in {"active", "activating", "reloading", "deactivating"}


def gpu_state(gpu: int) -> tuple[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0:
        memory, utilization = (int(value.strip()) for value in result.stdout.strip().split(","))
        return memory, utilization
    # Preserve conditional scheduling across a transient NVML userspace/kernel
    # mismatch. Query CUDA memory in a short-lived process so this watcher does
    # not retain a CUDA context while waiting.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    fallback = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; free,total=torch.cuda.mem_get_info(0); print((total-free)//1048576)",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=env,
    )
    return int(fallback.stdout.strip()), 0


def wait_for_gate(status_path: Path) -> bool:
    gate_path = OUTPUT / "pilot_gate.json"
    while not gate_path.is_file():
        atomic_json({"status": "waiting_for_seed13_gate", "checked_at": time.time()}, status_path)
        time.sleep(10)
    gate = json.loads(gate_path.read_text())
    return bool(gate["passed"])


def acquire_queue_lock() -> object:
    lock = (OUTPUT / ".queue.lock").open("a")
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock
        except BlockingIOError:
            time.sleep(2)


def wait_for_resources(status_path: Path) -> None:
    idle = {0: 0, 1: 0}
    while min(idle.values()) < 2:
        active = [unit for unit in GPU0_UNITS if unit_active(unit)]
        states = {}
        for gpu in (0, 1):
            memory, utilization = gpu_state(gpu)
            states[str(gpu)] = {"memory_mib": memory, "utilization": utilization}
            ready = memory < 2000 and utilization < 10 and (gpu != 0 or not active)
            idle[gpu] = idle[gpu] + 1 if ready else 0
        atomic_json(
            {
                "status": "waiting_for_two_idle_gpus",
                "checked_at": time.time(),
                "gpu": states,
                "active_gpu0_units": active,
                "consecutive_idle_checks": idle,
            },
            status_path,
        )
        if min(idle.values()) < 2:
            time.sleep(30)


def verify_frozen_sources(plan: dict) -> None:
    for name, expected in plan["source_sha256"].items():
        path = Path(name)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"experiment source changed after registration: {path}")


def complete_report(directory: Path) -> bool:
    report = directory / "report.json"
    status = directory / "status.json"
    return (
        report.is_file()
        and status.is_file()
        and json.loads(status.read_text()).get("status") == "complete"
    )


def run_seed(seed: int, gpu: int, plan: dict) -> None:
    trainer = ROOT / "project/scripts/train_video_adaptation.py"
    log_dir = ROOT / "outputs/logs/video_adaptation_parallel_promotion"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for mode in MODES:
        verify_frozen_sources(plan)
        directory = OUTPUT / f"{mode}_seed{seed}"
        if complete_report(directory):
            continue
        command = [
            sys.executable,
            str(trainer),
            "--mode",
            mode,
            "--seed",
            str(seed),
            "--output",
            str(directory),
            "--device",
            "cuda:0",
            "--batch-size",
            "8",
        ]
        if (directory / "run_config.json").exists():
            if not (directory / "last.pt").exists():
                raise RuntimeError(f"configured run has no resumable checkpoint: {directory}")
            command.append("--resume")
        with (log_dir / f"seed{seed}_{mode}.log").open("a") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan_path = OUTPUT / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = json.loads(plan_path.read_text())
    parallel_plan = {
        "protocol": "video-adaptation-v1-parallel-promotion",
        "gate": str((OUTPUT / "pilot_gate.json").resolve()),
        "seed_gpu": {str(seed): gpu for seed, gpu in SEED_GPU.items()},
        "within_seed_order": list(MODES),
        "between_seed_execution": "parallel",
        "efficiency_metrics_valid": False,
        "reason": "Concurrent video decoding can affect wall-clock throughput; validation metrics remain in scope.",
        "official_test_evaluated": False,
    }
    if args.dry_run:
        print(json.dumps(parallel_plan, indent=2))
        return

    status_path = OUTPUT / "parallel_promotion_status.json"
    atomic_json(parallel_plan, OUTPUT / "parallel_promotion_plan.json")
    try:
        if not wait_for_gate(status_path):
            atomic_json({"status": "not_promoted", "reason": "seed13_gate_failed"}, status_path)
            return

        atomic_json({"status": "gate_passed_stopping_serial_queue"}, status_path)
        subprocess.run(["systemctl", "--user", "stop", ORIGINAL_UNIT], check=True, timeout=30)
        while unit_active(ORIGINAL_UNIT):
            time.sleep(2)

        queue_lock = acquire_queue_lock()
        try:
            wait_for_resources(status_path)
            atomic_json(
                {"status": "running", "seed_gpu": parallel_plan["seed_gpu"], "started_at": time.time()},
                status_path,
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_seed, seed, gpu, plan) for seed, gpu in SEED_GPU.items()]
                for future in futures:
                    future.result()
            summary = summarize(OUTPUT, [13, 42, 2026])
            atomic_json(
                {
                    "status": "complete",
                    "completed_seeds": [13, 42, 2026],
                    "seed_gpu": parallel_plan["seed_gpu"],
                    "mean_valid_mae": summary["mean_valid_mae"],
                    "official_test_evaluated": False,
                },
                status_path,
            )
            atomic_json(
                {
                    "status": "complete",
                    "completed_seeds": [13, 42, 2026],
                    "pilot_gate_passed": True,
                    "promotion_execution": "parallel",
                },
                OUTPUT / "queue_status.json",
            )
        finally:
            fcntl.flock(queue_lock, fcntl.LOCK_UN)
            queue_lock.close()
    except Exception as exc:
        atomic_json({"status": "failed", "error": repr(exc), "failed_at": time.time()}, status_path)
        raise


if __name__ == "__main__":
    main()
