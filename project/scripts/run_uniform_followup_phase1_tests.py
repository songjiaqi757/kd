#!/usr/bin/env python3
"""Test committed MOSEI checkpoints as GPU memory becomes available."""
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
ONE = ROOT / "project/scripts/evaluate_main_table_one_epoch.py"
FINAL = ROOT / "project/scripts/evaluate_main_table_all_epochs.py"
METHODS = (("adapted_student", 0), ("subset7", 1), ("ensemble_full", 0), ("first_order_interaction", 1))


def read(path: Path):
    return json.loads(path.read_text())


def write(payload: dict, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def free_mib() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    return {int(i.strip()): int(m.strip()) for i, m in (line.split(",") for line in result.stdout.splitlines())}


def epochs(job: dict) -> list[int]:
    path = Path(job["run"]) / "history.json"
    if not path.is_file():
        return []
    result = [int(row["epoch"]) for row in read(path)]
    if result != list(range(1, len(result) + 1)):
        raise ValueError(f"nonsequential committed history: {path}")
    return result


def tested(job: dict) -> set[int]:
    result = set()
    for epoch in epochs(job):
        path = Path(job["output"]) / "epochs" / f"epoch_{epoch:03d}"
        if (path / "report.json").is_file() and (path / "predictions.jsonl").is_file():
            result.add(epoch)
    return result


def launch(job: dict, epoch: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    command = [str(PYTHON), "-u", str(ONE), "--run", job["run"], "--output", job["output"],
               "--epoch", str(epoch), "--device", "cuda:0", "--batch-size", "8", "--num-workers", "2"]
    log = OUTPUT / "logs" / f"{job['method']}_seed13_epoch_{epoch:03d}_test.log"
    with log.open("a") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    job["active_epoch"] = epoch
    job["active_pid"] = process.pid
    job["active_log"] = str(log)
    job["state"] = "evaluating"
    return process


def finalize(job: dict):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    command = [str(PYTHON), "-u", str(FINAL), "--run", job["run"], "--output", job["output"],
               "--device", "cuda:0", "--batch-size", "8", "--num-workers", "2"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    summary = read(Path(job["output"]) / "summary.json")
    if not summary["complete"]:
        raise ValueError("final summary is incomplete")
    job["selected_epoch"] = summary["selected_epoch"]
    job["selected_test_mae"] = summary["selected_test_metrics"]["mae"]
    job["state"] = "complete"


def supervise(min_free_mib: int):
    with (OUTPUT / ".phase1_tests.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT / "phase1_test_status.json").exists():
            raise FileExistsError("test queue is already running")
        (OUTPUT / "logs").mkdir(exist_ok=True)
        jobs = [{"method": method, "gpu": gpu,
                 "run": str(OUTPUT / "students" / f"{method}_seed13"),
                 "output": str(OUTPUT / "test_epoch_sweeps" / f"{method}_seed13"),
                 "state": "waiting_for_checkpoint", "active_epoch": None, "active_pid": None,
                 "committed_epochs": 0, "tested_epochs": 0} for method, gpu in METHODS]
        active = {}
        failures = {}
        last_method = {0: None, 1: None}
        started = time.time()
        while True:
            source = OUTPUT / "phase1_status.json"
            training = read(source) if source.exists() else {"jobs": []}
            training_states = {row["method"]: row["state"] for row in training["jobs"]}
            for gpu, (process, job, epoch) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[gpu]
                job["active_epoch"] = job["active_pid"] = None
                if code == 0:
                    job["state"] = "waiting_for_checkpoint"
                else:
                    key = (job["method"], epoch)
                    failures[key] = failures.get(key, 0) + 1
                    job["last_failed_epoch"] = epoch
                    job["last_exit_code"] = code
                    job["state"] = "failed" if failures[key] >= 2 else "retry_pending"
            try:
                free, gpu_error = free_mib(), None
            except Exception as error:
                free, gpu_error = {}, repr(error)
            for job in jobs:
                if job["state"] in {"complete", "failed", "blocked_training_failed"}:
                    continue
                committed, finished = epochs(job), tested(job)
                job["committed_epochs"] = len(committed)
                job["tested_epochs"] = len(finished)
                if training_states.get(job["method"]) == "failed":
                    job["state"] = "blocked_training_failed"
                elif job["active_pid"] is None and training_states.get(job["method"]) == "complete":
                    if (Path(job["run"]) / "checkpoint_inventory.json").is_file() and len(finished) == len(committed):
                        try:
                            finalize(job)
                        except Exception as error:
                            job["state"] = "failed"
                            job["finalization_error"] = repr(error)
            for gpu in (0, 1):
                if gpu in active or free.get(gpu, 0) < min_free_mib:
                    continue
                candidates = []
                for job in jobs:
                    if job["gpu"] != gpu or job["state"] in {"complete", "failed", "blocked_training_failed"}:
                        continue
                    pending = [epoch for epoch in epochs(job) if epoch not in tested(job)]
                    if pending:
                        candidates.append((job, pending[0]))
                candidates.sort(key=lambda pair: (pair[0]["method"] == last_method[gpu], pair[1]))
                if not candidates:
                    continue
                job, epoch = candidates[0]
                if not (Path(job["run"]) / "checkpoints" / f"epoch_{epoch:03d}.pt").is_file():
                    continue
                active[gpu] = (launch(job, epoch), job, epoch)
                last_method[gpu] = job["method"]
            write({"schema": "uniform-interaction-followup-streaming-test-queue-v1",
                   "started_at_unix": started, "updated_at_unix": time.time(),
                   "min_free_mib_before_launch": min_free_mib,
                   "gpu_free_mib": free, "gpu_query_error": gpu_error,
                   "max_concurrent_tests_per_gpu": 1, "jobs": jobs}, OUTPUT / "phase1_test_status.json")
            if all(job["state"] in {"complete", "failed", "blocked_training_failed"} for job in jobs):
                return
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=16384)
    args = parser.parse_args()
    if args.min_free_mib < 8192:
        parser.error("minimum free GPU memory must be at least 8192 MiB")
    if args.supervise:
        supervise(args.min_free_mib)
        return
    if (OUTPUT / "phase1_test_status.json").exists():
        raise FileExistsError("test queue is already running")
    log = OUTPUT / "phase1_test_supervisor.log"
    with log.open("w") as stream:
        process = subprocess.Popen([str(PYTHON), "-u", str(Path(__file__).resolve()),
                                    "--supervise", "--min-free-mib", str(args.min_free_mib)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"supervisor_pid": process.pid, "supervisor_log": str(log)}))


if __name__ == "__main__":
    main()
