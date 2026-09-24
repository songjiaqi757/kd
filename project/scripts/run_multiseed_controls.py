#!/usr/bin/env python3
"""Keep a four-slot, two-GPU queue full for formal multi-seed controls."""
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
JOB_RUNNER = ROOT / "project/scripts/run_multiseed_control_job.py"
METHOD_CHOICES = (
    "full_kd",
    "ensemble_full",
    "subset7",
    "first_order_interaction",
    "first_second_order_interaction",
    "random_orthogonal",
    "uniform_interaction",
)
LAUNCH_SETTLE_SECONDS = 120.0


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def complete(base: Path, method: str, seed: int) -> bool:
    path = base / "jobs" / f"{method}_seed{seed}.json"
    return path.is_file() and read_json(path).get("status") == "complete"


def gpu_state(gpu: int) -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True, timeout=20,
    )
    total, used, free, utilization = [int(value.strip()) for value in result.stdout.split(",")]
    return {"total_mib": total, "used_mib": used, "free_mib": free, "utilization": utilization}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mosei", "mosi"), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(13, 42, 2026), required=True)
    parser.add_argument("--max-jobs", type=int, default=4)
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--job-budget-gib", type=int, default=30)
    parser.add_argument("--headroom-gib", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("methods and seeds must be unique")
    if not 1 <= args.max_jobs <= 4 or not 1 <= args.slots_per_gpu <= 2:
        parser.error("this protocol supports at most four jobs and two per GPU")
    if args.job_budget_gib < 25 or args.headroom_gib < 8 or args.poll_seconds <= 0:
        parser.error("unsafe memory budget, headroom, or polling interval")
    return args


def can_launch(state: dict[str, int], occupants: int, args: argparse.Namespace) -> bool:
    if occupants >= args.slots_per_gpu:
        return False
    budget = args.job_budget_gib * 1024
    margin = args.headroom_gib * 1024
    return (
        (occupants + 1) * budget + margin <= state["total_mib"]
        and state["used_mib"] + budget + margin <= state["total_mib"]
    )


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    jobs = [(method, seed) for seed in args.seeds for method in args.methods]
    plan = {
        "schema": "rdid-msa-multiseed-controls-v1",
        "dataset": args.dataset,
        "methods": args.methods,
        "seeds": args.seeds,
        "checkpoint_retention": "every_completed_epoch_model_plus_resumable_last",
        "checkpoint_selection": "validation_mae_minimum",
        "official_test_policy": "evaluate_valid_best_once_after_training",
        "coordinate_seed": 20260922,
        "max_jobs": args.max_jobs,
        "slots_per_gpu": args.slots_per_gpu,
        "job_budget_gib": args.job_budget_gib,
        "headroom_gib": args.headroom_gib,
    }
    if args.dry_run:
        print(json.dumps(plan | {"jobs": [f"{m}_seed{s}" for m, s in jobs]}, indent=2))
        return
    base.mkdir(parents=True, exist_ok=True)
    plan_path = base / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError(f"existing plan differs: {plan_path}")
    atomic_json(plan, plan_path)
    active: dict[str, dict] = {}
    handles: dict[str, object] = {}
    pending = [(method, seed) for method, seed in jobs if not complete(base, method, seed)]
    with (base / ".queue.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while pending or active:
            failures = []
            for name, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                handles.pop(name).close()
                del active[name]
                if code:
                    failures.append({"run": name, "returncode": code})
                else:
                    method, seed = name.rsplit("_seed", 1)
                    if not complete(base, method, int(seed)):
                        failures.append({"run": name, "returncode": code, "reason": "missing complete status"})
            if failures:
                atomic_json({
                    "status": "failed", "failures": failures,
                    "running": {name: {k: v for k, v in item.items() if k != "process"} for name, item in active.items()},
                    "pending": [f"{m}_seed{s}" for m, s in pending],
                    "updated_at_unix": time.time(),
                }, base / "queue_status.json")
                raise RuntimeError(f"jobs failed: {failures}")

            states = {gpu: gpu_state(gpu) for gpu in (0, 1)}
            # Launch at most one new process per GPU in each polling pass.  CUDA
            # allocation lags process creation, so re-read actual memory on the
            # next pass before admitting a second colocated job.
            for gpu in sorted((0, 1), key=lambda value: sum(x["gpu"] == value for x in active.values())):
                if not pending or len(active) >= args.max_jobs:
                    break
                occupants = sum(item["gpu"] == gpu for item in active.values())
                # Model construction can take longer than one polling interval,
                # during which nvidia-smi has not yet accounted for the new job.
                # Do not admit another colocated process until the last launch
                # has had enough time to establish its real CUDA footprint.
                if any(
                    item["gpu"] == gpu
                    and time.time() - item["started_at_unix"] < LAUNCH_SETTLE_SECONDS
                    for item in active.values()
                ):
                    continue
                states[gpu] = gpu_state(gpu)
                if not can_launch(states[gpu], occupants, args):
                    continue
                method, seed = pending.pop(0)
                name = f"{method}_seed{seed}"
                command = [
                    sys.executable, "-u", str(JOB_RUNNER),
                    "--dataset", args.dataset,
                    "--method", method,
                    "--seed", str(seed),
                    "--base", str(base),
                    "--device", "cuda:0",
                ]
                env = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "HF_HUB_OFFLINE": "1",
                    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                    "PYTHONUNBUFFERED": "1",
                }
                log_path = base / "logs" / f"{name}.job.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("a")
                process = subprocess.Popen(
                    command, cwd=ROOT, env=env,
                    stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[name] = {
                    "process": process, "pid": process.pid, "gpu": gpu,
                    "started_at_unix": time.time(), "log": str(log_path),
                }
                handles[name] = handle
            atomic_json({
                "status": "running" if active else "waiting_for_resources",
                "dataset": args.dataset,
                "running": {name: {k: v for k, v in item.items() if k != "process"} for name, item in active.items()},
                "pending": [f"{m}_seed{s}" for m, s in pending],
                "completed": [f"{m}_seed{s}" for m, s in jobs if complete(base, m, s)],
                "gpu": {str(gpu): state for gpu, state in states.items()},
                "updated_at_unix": time.time(),
            }, base / "queue_status.json")
            if pending or active:
                time.sleep(args.poll_seconds)
        atomic_json({
            "status": "complete", "dataset": args.dataset,
            "completed": [f"{m}_seed{s}" for m, s in jobs],
            "completed_at_unix": time.time(),
        }, base / "queue_status.json")


if __name__ == "__main__":
    main()
