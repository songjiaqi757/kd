#!/usr/bin/env python3
"""GPU-memory-gated queue for exploratory RDID-v2 per-epoch MOSEI tests."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/scripts"))

from evaluate_rdid_v2_epoch_sweep import METHODS, atomic_json, complete_epoch, read_json
from train_tav_distillation import sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=ROOT / "outputs/experiments/rdid_v2_mosei/students")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/experiments/rdid_v2_mosei/epoch_test_sweep_20260919")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-evaluators-per-gpu", type=int, default=3)
    parser.add_argument("--required-free-mib", type=int, default=18 * 1024)
    parser.add_argument("--settle-seconds", type=int, default=90)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--min-disk-free-gib", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-mosei-test-all-epochs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_mosei_test_all_epochs:
        parser.error("explicit --allow-mosei-test-all-epochs is required")
    if (
        not args.gpus or len(args.gpus) != len(set(args.gpus))
        or min(args.gpus) < 0 or args.max_evaluators_per_gpu <= 0
        or min(args.required_free_mib, args.settle_seconds, args.poll_seconds,
               args.min_disk_free_gib, args.batch_size) <= 0
    ):
        parser.error("invalid resource limits")
    return args


def gpu_free_mib() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    values = {}
    for line in result.stdout.splitlines():
        gpu, free = (int(part.strip()) for part in line.split(","))
        values[gpu] = free
    return values


def available_jobs(train_root: Path, output_root: Path) -> tuple[list[tuple[str, int]], dict]:
    pending = []
    methods = {}
    for method in METHODS:
        run = train_root / f"{method}_seed13"
        train_status = (read_json(run / "status.json") or {}).get("status", "not_started")
        checkpoints = sorted((run / "checkpoints").glob("epoch_*.pt"))
        done = 0
        for path in checkpoints:
            epoch = int(path.stem.rsplit("_", 1)[1])
            if complete_epoch(output_root / method / path.stem):
                done += 1
            else:
                pending.append((method, epoch))
        methods[method] = {
            "train_status": train_status,
            "checkpoint_count": len(checkpoints),
            "evaluated_count": done,
        }
    order = {method: index for index, method in enumerate(METHODS)}
    pending.sort(key=lambda item: (item[1], order[item[0]]))
    return pending, methods


def stop_children(running: dict) -> None:
    for job in running.values():
        process = job["process"]
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for job in running.values():
        try:
            job["process"].wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(job["process"].pid, signal.SIGKILL)
        job["log"].close()


def launch(
    *, method: str, epoch: int, gpu: int, output_root: Path,
    train_root: Path, batch_size: int,
) -> dict:
    script = ROOT / "project/scripts/evaluate_rdid_v2_one_epoch.py"
    log_path = output_root / "logs/jobs" / f"{method}_epoch_{epoch:03d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    command = [
        sys.executable, str(script), "--method", method, "--epoch", str(epoch),
        "--train-root", str(train_root), "--output-root", str(output_root),
        "--batch-size", str(batch_size), "--allow-mosei-test-all-epochs",
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    print(json.dumps({"event": "launched", "method": method, "epoch": epoch,
                      "gpu": gpu, "pid": process.pid}), flush=True)
    return {"process": process, "gpu": gpu, "log": log,
            "log_path": str(log_path), "started_at": time.time()}


def main() -> None:
    args = parse_args()
    train_root = args.train_root.resolve()
    output_root = args.output_root.resolve()
    source = ROOT / "project/scripts/evaluate_rdid_v2_one_epoch.py"
    plan = {
        "schema": "rdid-v2-mosei-exploratory-epoch-test-queue-v1",
        "methods": list(METHODS),
        "gpus": args.gpus,
        "max_evaluators_per_gpu": args.max_evaluators_per_gpu,
        "required_free_mib_before_launch": args.required_free_mib,
        "settle_seconds_between_launches_per_gpu": args.settle_seconds,
        "poll_seconds": args.poll_seconds,
        "min_disk_free_gib": args.min_disk_free_gib,
        "batch_size": args.batch_size,
        "train_root": str(train_root),
        "output_root": str(output_root),
        "evaluator_sha256": sha256(source),
        "test_use_policy": "exploratory_all_epoch_test_not_blind_confirmation",
        "mosei_is_blind_confirmation": False,
    }
    if args.dry_run:
        pending, methods = available_jobs(train_root, output_root)
        print(json.dumps({**plan, "pending_jobs": len(pending),
                          "pending_preview": pending[:12], "method_status": methods,
                          "gpu_free_mib": gpu_free_mib()}, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".dynamic_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = output_root / "dynamic_queue_plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing queue plan differs; use the same resource policy")
        atomic_json(plan, plan_path)
        running: dict[tuple[str, int], dict] = {}
        attempts: dict[tuple[str, int], int] = {}
        capacity = {gpu: args.max_evaluators_per_gpu for gpu in args.gpus}
        last_launch = {gpu: 0.0 for gpu in args.gpus}
        interrupted = False

        def interrupt(signum, frame):
            nonlocal interrupted
            interrupted = True

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        try:
            while not interrupted:
                for identity, job in list(running.items()):
                    result = job["process"].poll()
                    if result is None:
                        continue
                    job["log"].close()
                    del running[identity]
                    if result != 0:
                        error = (read_json(
                            output_root / identity[0] / f"epoch_{identity[1]:03d}" / "status.json"
                        ) or {}).get("error", "")
                        print(json.dumps({"event": "failed", "method": identity[0],
                                          "epoch": identity[1], "gpu": job["gpu"],
                                          "exit_code": result, "error": error,
                                          "attempts": attempts[identity]}), flush=True)
                        if "OutOfMemoryError" in error or "CUDA out of memory" in error:
                            capacity[job["gpu"]] = max(1, capacity[job["gpu"]] - 1)
                        if attempts[identity] >= 3:
                            raise RuntimeError(f"repeated evaluation failure: {identity}; see {job['log_path']}")
                    elif not complete_epoch(
                        output_root / identity[0] / f"epoch_{identity[1]:03d}"
                    ):
                        raise RuntimeError(f"evaluator exited without complete output: {identity}")
                    else:
                        print(json.dumps({"event": "completed", "method": identity[0],
                                          "epoch": identity[1], "gpu": job["gpu"]}), flush=True)

                pending, methods = available_jobs(train_root, output_root)
                pending = [identity for identity in pending if identity not in running]
                all_train_finished = all(
                    item["train_status"] in {"complete", "failed"} for item in methods.values()
                )
                if not pending and not running and all_train_finished:
                    final_status = "complete" if all(
                        item["train_status"] == "complete" for item in methods.values()
                    ) else "partial_training_failure"
                    atomic_json({"status": final_status, "methods": methods,
                                 "test_use_policy": plan["test_use_policy"]},
                                output_root / "dynamic_queue_status.json")
                    print(json.dumps({"event": "queue_finished", "status": final_status}), flush=True)
                    return

                free_mib = gpu_free_mib()
                disk_free_gib = shutil.disk_usage(output_root).free / 1024**3
                for gpu in args.gpus:
                    if not pending or disk_free_gib < args.min_disk_free_gib:
                        break
                    if gpu not in free_mib:
                        raise ValueError(f"GPU {gpu} unavailable")
                    count = sum(job["gpu"] == gpu for job in running.values())
                    if (
                        count >= capacity[gpu]
                        or free_mib[gpu] < args.required_free_mib
                        or time.time() - last_launch[gpu] < args.settle_seconds
                    ):
                        continue
                    identity = pending.pop(0)
                    attempts[identity] = attempts.get(identity, 0) + 1
                    running[identity] = launch(
                        method=identity[0], epoch=identity[1], gpu=gpu,
                        output_root=output_root, train_root=train_root,
                        batch_size=args.batch_size,
                    )
                    last_launch[gpu] = time.time()

                atomic_json({
                    "status": "paused_low_disk" if disk_free_gib < args.min_disk_free_gib else "running",
                    "updated_at_unix": time.time(),
                    "disk_free_gib": disk_free_gib,
                    "gpu_free_mib": {str(key): value for key, value in free_mib.items()},
                    "gpu_evaluator_capacity": {str(key): value for key, value in capacity.items()},
                    "pending_jobs": len(pending),
                    "running_jobs": [
                        {"method": method, "epoch": epoch, "gpu": job["gpu"],
                         "pid": job["process"].pid, "started_at_unix": job["started_at"]}
                        for (method, epoch), job in running.items()
                    ],
                    "methods": methods,
                    "test_use_policy": plan["test_use_policy"],
                }, output_root / "dynamic_queue_status.json")
                time.sleep(args.poll_seconds)
        finally:
            stop_children(running)
            if interrupted:
                atomic_json({"status": "interrupted", "methods": methods},
                            output_root / "dynamic_queue_status.json")


if __name__ == "__main__":
    main()
