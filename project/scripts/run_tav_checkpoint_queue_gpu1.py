#!/usr/bin/env python3
"""GPU1-only, VRAM-aware queue of individual retained TAV checkpoints."""
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
from evaluate_tav_all_checkpoints import checkpoint_plan, freeze_plan, read_json
from evaluate_tav_streaming_checkpoints import committed_history
from evaluate_tav_test import atomic_json
from train_tav_distillation import sha256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--launch-free-mib", type=int, default=20 * 1024)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--settle-seconds", type=float, default=45)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    if min(args.launch_free_mib, args.max_concurrent, args.batch_size) <= 0:
        parser.error("memory gate, concurrency, and batch size must be positive")
    if args.num_workers < 0 or min(args.poll_seconds, args.settle_seconds) < 0:
        parser.error("worker count and timing must be non-negative")
    return args


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
    return {"schema": "rdid-msa-tav-per-checkpoint-gpu1-queue-plan-v1",
            "jobs": jobs, "test_manifest": str(args.test_manifest),
            "test_manifest_sha256": sha256(args.test_manifest),
            "gpu": 1, "launch_free_mib": args.launch_free_mib,
            "max_concurrent": args.max_concurrent, "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "epoch_admission": "committed_history_row_after_atomic_epoch_and_last_checkpoint_save",
            "final_checkpoints_admission": "completed_source_and_audited_checkpoint_inventory",
            "test_use_policy": "diagnostic_only_never_select_epoch_or_hyperparameters_on_test"}


def freeze_queue_plan(plan, output):
    path = output / "checkpoint_queue_plan.json"
    if path.is_file():
        if read_json(path) != plan:
            raise ValueError(f"existing per-checkpoint queue plan changed: {path}")
    else:
        atomic_json(plan, path)


def available_entries(job, manifest, final_plans):
    run, output = Path(job["source_run"]), Path(job["output"])
    if sha256(run / "run_config.json") != job["source_run_config_sha256"]:
        raise ValueError(f"source run config changed: {run}")
    state = read_json(run / "status.json").get("status")
    if state == "failed":
        raise RuntimeError(f"source training failed: {run}")
    history = committed_history(run)
    if state == "complete" and (run / "checkpoint_inventory.json").is_file():
        if job["method"] not in final_plans:
            final = checkpoint_plan(run, output, manifest)
            freeze_plan(final, output)
            final_plans[job["method"]] = final
        return [entry["name"] for entry in final_plans[job["method"]]["checkpoints"]]
    return [f"epoch_{int(row['epoch']):03d}" for row in history]


def checkpoint_complete(job, name):
    destination = Path(job["output"]) / "checkpoints" / name
    state = destination / "status.json"
    if not state.is_file() or read_json(state).get("status") != "complete":
        return False
    required = ("run_config.json", "report.json", "predictions.jsonl")
    if not all((destination / item).is_file() for item in required):
        raise ValueError(f"completed checkpoint evaluation lacks artifacts: {destination}")
    return True


def launch(job, name, args):
    identity = f"{job['method']}_{name}"
    log_path = args.output / "logs" / f"{identity}.log"
    command = [sys.executable, str(ROOT / "project/scripts/evaluate_tav_one_checkpoint.py"),
               "--run", job["source_run"], "--output", job["output"],
               "--test-manifest", str(args.test_manifest), "--checkpoint-name", name,
               # Physical GPU1 is remapped to cuda:0 for this child only.
               "--device", "cuda:0", "--batch-size", str(args.batch_size),
               "--num-workers", str(args.num_workers), "--allow-diagnostic-test"]
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                        "HF_HUB_OFFLINE": "1"})
    log = log_path.open("a")
    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                               stdout=log, stderr=subprocess.STDOUT)
    return {"job": job, "name": name, "process": process, "log_handle": log,
            "pid": process.pid, "log": str(log_path), "launched_at": time.time(), "gpu": 1}


def finish_method(job, final_plan):
    output = Path(job["output"])
    results = []
    for entry in final_plan["checkpoints"]:
        name = entry["name"]
        if not checkpoint_complete(job, name):
            return False
        record = read_json(output / "checkpoints" / name / "run_config.json")
        if (record.get("checkpoint") != entry
                or record.get("source_run_config_sha256") != final_plan["source_run_config_sha256"]
                or record.get("test_manifest_sha256") != final_plan["test_manifest_sha256"]):
            raise ValueError(f"checkpoint evaluation provenance differs: {output / 'checkpoints' / name}")
        path = output / "checkpoints" / name / "report.json"
        result = read_json(path)
        if result.get("checkpoint") != name or int(result.get("epoch", -1)) != entry["epoch"]:
            raise ValueError(f"checkpoint evaluation result differs: {path}")
        selected = entry["epoch"] == final_plan["source_valid_selected_epoch"]
        if result["is_valid_selected_epoch"] != selected:
            result["is_valid_selected_epoch"] = selected
            atomic_json(result, path)
        results.append(result)
    atomic_json({"schema": "rdid-msa-tav-run-all-checkpoints-diagnostic-summary-v1",
                 "method": job["method"], "seed": job["seed"],
                 "valid_selected_epoch": final_plan["source_valid_selected_epoch"],
                 "checkpoints": results, "test_use_policy": final_plan["test_use_policy"]},
                output / "summary.json")
    atomic_json({"status": "complete", "method": job["method"],
                 "completed": len(results), "total": len(results)}, output / "status.json")
    return True


def snapshot(args, jobs, names, running, final_plans, free_mib, state="running"):
    completed = {job["method"]: sum(checkpoint_complete(job, name)
                                      for name in names[job["method"]]) for job in jobs}
    waiting = {job["method"]: sum(not checkpoint_complete(job, name)
                                  for name in names[job["method"]]) for job in jobs}
    atomic_json({"status": state, "gpu": 1, "gpu1_free_mib": free_mib,
                 "max_concurrent": args.max_concurrent,
                 "running": {identity: {key: value for key, value in item.items()
                                       if key not in {"job", "process", "log_handle"}}
                             for identity, item in running.items()},
                 "source_completed_epochs": {job["method"]: len(committed_history(Path(job["source_run"])))
                                             for job in jobs},
                 "completed_checkpoints": completed, "waiting_checkpoints": waiting,
                 "training_in_progress": [job["method"] for job in jobs
                                          if read_json(Path(job["source_run"]) / "status.json").get("status") == "training"],
                 "finalized_methods": sorted(final_plans), "error": None},
                args.output / "scheduler_status.json")


def main():
    args = parse_args()
    args.source, args.output, args.test_manifest = (args.source.resolve(), args.output.resolve(),
                                                     args.test_manifest.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    with (args.output / ".checkpoint_queue.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("per-checkpoint diagnostic queue already running") from exc
        plan = make_plan(args)
        freeze_queue_plan(plan, args.output)
        jobs = plan["jobs"]
        running = {}
        final_plans = {}
        try:
            while True:
                for identity, item in list(running.items()):
                    returncode = item["process"].poll()
                    if returncode is None:
                        continue
                    item["log_handle"].close()
                    del running[identity]
                    if returncode != 0 or not checkpoint_complete(item["job"], item["name"]):
                        raise RuntimeError(f"checkpoint evaluation failed: {identity}; see {item['log']}")
                names = {job["method"]: available_entries(job, args.test_manifest, final_plans)
                         for job in jobs}
                finished = [job for job in jobs if job["method"] in final_plans
                            and finish_method(job, final_plans[job["method"]])]
                if len(finished) == len(jobs) and not running:
                    summaries = [read_json(Path(job["output"]) / "summary.json") for job in jobs]
                    atomic_json({"schema": "rdid-msa-tav-five-run-all-checkpoints-diagnostic-summary-v1",
                                 "gpu": 1, "methods": list(METHODS), "runs": summaries,
                                 "test_use_policy": plan["test_use_policy"]}, args.output / "summary.json")
                    snapshot(args, jobs, names, running, final_plans, gpu1_free_mib(), state="complete")
                    return
                free_mib = gpu1_free_mib()
                running_by_method = {method: sum(item["job"]["method"] == method for item in running.values())
                                     for method in METHODS}
                pending = [(job, name) for job in jobs for name in names[job["method"]]
                           if f"{job['method']}_{name}" not in running and not checkpoint_complete(job, name)]
                pending.sort(key=lambda item: (running_by_method[item[0]["method"]],
                                               METHODS.index(item[0]["method"]),
                                               0 if item[1].startswith("epoch_") else 1,
                                               item[1]))
                if pending and len(running) < args.max_concurrent and free_mib >= args.launch_free_mib:
                    job, name = pending[0]
                    identity = f"{job['method']}_{name}"
                    running[identity] = launch(job, name, args)
                    snapshot(args, jobs, names, running, final_plans, free_mib)
                    time.sleep(args.settle_seconds)
                else:
                    snapshot(args, jobs, names, running, final_plans, free_mib)
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
