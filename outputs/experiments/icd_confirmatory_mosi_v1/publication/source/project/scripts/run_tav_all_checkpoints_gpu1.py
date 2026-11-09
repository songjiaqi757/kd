#!/usr/bin/env python3
"""Evaluate every retained TAV checkpoint on the diagnostic test, using GPU1 only."""
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
METHODS = ("M0", "M1", "M3", "M4", "M6")
SEED = 13
DEFAULT_SOURCE = ROOT / "outputs/experiments/tav_epoch_checkpoints_v1"
DEFAULT_OUTPUT = DEFAULT_SOURCE / "checkpoint_diagnostic_test_gpu1_20260919"
DEFAULT_MANIFEST = ROOT / "dataset/cmu_mosei/manifests/official_test_windowed.jsonl"
sys.path.insert(0, str(ROOT / "project/scripts"))
from train_tav_distillation import sha256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--launch-free-mib", type=int, default=20 * 1024)
    parser.add_argument("--max-concurrent-with-training", type=int, default=5)
    parser.add_argument("--max-concurrent-after-training", type=int, default=5)
    parser.add_argument("--settle-seconds", type=float, default=45)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    if min(args.launch_free_mib, args.max_concurrent_with_training,
           args.max_concurrent_after_training, args.batch_size) <= 0:
        parser.error("resource limits and batch size must be positive")
    if args.num_workers < 0 or min(args.poll_seconds, args.settle_seconds) < 0:
        parser.error("worker count and timing must be non-negative")
    return args


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def status(path):
    path = Path(path)
    return read_json(path).get("status") if path.is_file() else None


def gpu1_free_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    memory = {int(index.strip()): int(free.strip())
              for index, free in (line.split(",") for line in result.stdout.splitlines())}
    return memory[1]


def make_plan(args):
    jobs = []
    for method in METHODS:
        run = (args.source / "students" / f"{method}_seed{SEED}").resolve()
        config = run / "run_config.json"
        if not config.is_file():
            raise FileNotFoundError(config)
        jobs.append({"method": method, "seed": SEED, "source_run": str(run),
                     "source_run_config_sha256": sha256(config),
                     "output": str((args.output / f"{method}_seed{SEED}").resolve())})
    return {"schema": "rdid-msa-tav-all-checkpoints-gpu1-scheduler-plan-v1",
            "jobs": jobs, "test_manifest": str(args.test_manifest.resolve()),
            "test_manifest_sha256": sha256(args.test_manifest),
            "gpu": 1, "batch_size": args.batch_size, "num_workers": args.num_workers,
            "launch_free_mib": args.launch_free_mib,
            "max_concurrent_with_training": args.max_concurrent_with_training,
            "max_concurrent_after_training": args.max_concurrent_after_training,
            "streaming_epoch_admission": "history_row_committed_after_atomic_epoch_and_last_checkpoint_save",
            "test_use_policy": "diagnostic_only_never_select_epoch_or_hyperparameters_on_test"}


def freeze_plan(plan, output):
    path = output / "scheduler_plan.json"
    if path.is_file():
        if read_json(path) != plan:
            raise ValueError(f"existing scheduler plan differs: {path}")
    else:
        atomic_json(plan, path)


def source_ready(job):
    run = Path(job["source_run"])
    if sha256(run / "run_config.json") != job["source_run_config_sha256"]:
        raise ValueError(f"source run config changed: {run}")
    current = status(run / "status.json")
    if current == "failed":
        raise RuntimeError(f"source training failed: {run}")
    if current != "complete":
        history_path = run / "history.json"
        if current != "training" or not history_path.is_file():
            return False
        history = read_json(history_path)
        if not history:
            return False
        last_epoch = int(history[-1]["epoch"])
        return (run / "checkpoints" / f"epoch_{last_epoch:03d}.pt").is_file()
    required = ("report.json", "history.json", "best.pt", "last.pt", "checkpoint_inventory.json")
    # status=complete precedes the wrapper's final inventory write.
    return all((run / name).is_file() for name in required)


def output_complete(job):
    output = Path(job["output"])
    if status(output / "status.json") != "complete":
        return False
    summary = output / "summary.json"
    if not summary.is_file():
        raise ValueError(f"completed evaluation lacks summary: {output}")
    report = read_json(summary)
    if len(report["checkpoints"]) < 3:
        raise ValueError(f"completed evaluation has too few checkpoints: {output}")
    return True


def launch(job, args):
    log_path = args.output / "logs" / f"{job['method']}_seed{SEED}.log"
    # Physical GPU1 is the sole visible CUDA device for this child, so cuda:0
    # inside the evaluator maps to GPU1 and cannot allocate on physical GPU0.
    streaming = status(Path(job["source_run"]) / "status.json") != "complete"
    script = "evaluate_tav_streaming_checkpoints.py" if streaming else "evaluate_tav_all_checkpoints.py"
    command = [sys.executable, str(ROOT / "project/scripts" / script),
               "--run", job["source_run"], "--output", job["output"],
               "--test-manifest", str(args.test_manifest), "--device", "cuda:0",
               "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
               "--allow-diagnostic-test"]
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                        "HF_HUB_OFFLINE": "1"})
    log = log_path.open("a")
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    return {"process": process, "log_handle": log, "log": str(log_path),
            "pid": process.pid, "launched_at": time.time(), "gpu": 1,
            "streaming": streaming}


def snapshot(args, jobs, running, free_mib, state="running", error=None):
    atomic_json({"status": state, "gpu": 1, "gpu1_free_mib": free_mib,
                 "running": {method: {key: value for key, value in item.items()
                                     if key not in {"process", "log_handle"}}
                             for method, item in running.items()},
                 "completed": [job["method"] for job in jobs if output_complete(job)],
                 "training_in_progress": [job["method"] for job in jobs
                                          if status(Path(job["source_run"]) / "status.json") == "training"],
                 "waiting_for_training": [job["method"] for job in jobs
                                          if not output_complete(job) and not source_ready(job)],
                 "error": error}, args.output / "scheduler_status.json")


def main():
    args = parse_args()
    args.source, args.output, args.test_manifest = (args.source.resolve(), args.output.resolve(),
                                                     args.test_manifest.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    with (args.output / ".scheduler.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("checkpoint diagnostic scheduler already running") from exc
        plan = make_plan(args)
        freeze_plan(plan, args.output)
        jobs = plan["jobs"]
        running = {}
        try:
            while True:
                for method, item in list(running.items()):
                    returncode = item["process"].poll()
                    if returncode is None:
                        continue
                    item["log_handle"].close()
                    del running[method]
                    if returncode != 0 or not output_complete(next(job for job in jobs if job["method"] == method)):
                        raise RuntimeError(f"checkpoint evaluation failed: {method}; see {item['log']}")
                if all(output_complete(job) for job in jobs):
                    summaries = [read_json(Path(job["output"]) / "summary.json") for job in jobs]
                    atomic_json({"schema": "rdid-msa-tav-five-run-all-checkpoints-diagnostic-summary-v1",
                                 "gpu": 1, "methods": list(METHODS), "runs": summaries,
                                 "test_use_policy": plan["test_use_policy"]}, args.output / "summary.json")
                    snapshot(args, jobs, running, gpu1_free_mib(), state="complete")
                    return
                free_mib = gpu1_free_mib()
                training_incomplete = any(not source_ready(job) for job in jobs)
                capacity = (args.max_concurrent_with_training if training_incomplete
                            else args.max_concurrent_after_training)
                pending = [job for job in jobs if job["method"] not in running
                           and not output_complete(job) and source_ready(job)]
                if pending and len(running) < capacity and free_mib >= args.launch_free_mib:
                    job = pending[0]
                    running[job["method"]] = launch(job, args)
                    snapshot(args, jobs, running, free_mib)
                    time.sleep(args.settle_seconds)
                else:
                    snapshot(args, jobs, running, free_mib)
                    time.sleep(args.poll_seconds)
        except Exception as exc:
            for item in running.values():
                if item["process"].poll() is None:
                    item["process"].terminate()
                item["log_handle"].close()
            atomic_json({"status": "failed", "gpu": 1, "error": repr(exc)},
                        args.output / "scheduler_status.json")
            raise


if __name__ == "__main__":
    main()
