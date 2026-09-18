#!/usr/bin/env python3
"""Run one seed-13 retrain per TAV method and retain every epoch model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("M0", "M1", "M3", "M4", "M6")
SEED = 13
DEFAULT_OUTPUT = ROOT / "outputs/experiments/tav_epoch_checkpoints_v1"
DEFAULT_ASSETS = ROOT / "outputs/experiments/tav_main_v1/assets/protocol.json"
DEFAULT_TEST_ROOT = ROOT / "outputs/experiments/tav_main_v1/official_test_20260917"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--reserve-mib", type=int, default=34 * 1024)
    parser.add_argument("--margin-mib", type=int, default=8 * 1024)
    parser.add_argument("--settle-seconds", type=float, default=45.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def read_status(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("status")


def gpu_states():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    states = {}
    for line in result.stdout.splitlines():
        index, total, used, free, utilization = [int(value.strip()) for value in line.split(",")]
        states[index] = {"total_mib": total, "used_mib": used, "free_mib": free, "utilization": utilization}
    return states


def freeze_plan(args):
    wrapper = ROOT / "project/scripts/train_tav_distillation_all_epochs.py"
    trainer = ROOT / "project/scripts/train_tav_distillation.py"
    jobs = [{
        "method": method, "seed": SEED,
        "output": str((args.output / "students" / f"{method}_seed{SEED}").resolve()),
    } for method in METHODS]
    payload = {
        "schema": "rdid-msa-tav-epoch-checkpoint-retrain-plan-v1",
        "created_at_unix": time.time(),
        "purpose": "one_additional_seed13_run_per_method_with_every_completed_epoch_model_retained",
        "methods": list(METHODS), "seed": SEED, "jobs": jobs,
        "training_protocol": {
            "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
            "checkpoint_selection": "valid_mae", "official_test_evaluated": False,
            "epoch_checkpoint_use": "analysis_only_not_official_test_selection",
        },
        "resource_policy": {
            "hard_gate": "wait_until_15_best_checkpoint_official_tests_complete",
            "reserve_mib_per_training": args.reserve_mib,
            "free_memory_margin_mib": args.margin_mib,
            "admission": "launch_only_when_free_mib_at_least_reserve_plus_margin_then_recheck_after_settle",
        },
        "assets": str(args.assets.resolve()), "assets_sha256": sha256(args.assets),
        "sources_sha256": {str(path.resolve()): sha256(path) for path in (wrapper, trainer)},
    }
    path = args.output / "training_plan.json"
    if path.exists():
        existing = json.loads(path.read_text())
        comparable = {key: value for key, value in existing.items() if key != "created_at_unix"}
        expected = {key: value for key, value in payload.items() if key != "created_at_unix"}
        if comparable != expected:
            raise ValueError("existing epoch-checkpoint retrain plan differs")
        return existing
    atomic_json(payload, path)
    return payload


def test_gate(args):
    scheduler = args.test_root / "dynamic_scheduler_status.json"
    summary = args.test_root / "all_five_methods_summary.json"
    return read_status(scheduler) == "complete" and summary.is_file()


def job_complete(output):
    output = Path(output)
    if read_status(output / "status.json") != "complete":
        return False
    required = ("run_config.json", "report.json", "history.json", "best.pt", "last.pt", "checkpoint_inventory.json")
    if not all((output / name).is_file() for name in required):
        raise RuntimeError(f"completed retrain lacks required artifacts: {output}")
    history = json.loads((output / "history.json").read_text())
    inventory = json.loads((output / "checkpoint_inventory.json").read_text())
    if [row["epoch"] for row in inventory["epochs"]] != [row["epoch"] for row in history]:
        raise ValueError(f"epoch checkpoint inventory coverage differs: {output}")
    return True


def command(job, args):
    output = Path(job["output"])
    cmd = [
        sys.executable, str(ROOT / "project/scripts/train_tav_distillation_all_epochs.py"),
        "--method", job["method"], "--seed", str(job["seed"]),
        "--output", str(output), "--assets", str(args.assets),
        "--device", "cuda:0", "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers), "--epochs", str(args.epochs),
        "--patience", str(args.patience), "--progress-every", "100", "--no-diagnostics",
    ]
    if (output / "run_config.json").exists():
        if not (output / "last.pt").is_file():
            raise RuntimeError(f"configured retrain lacks resumable last.pt: {output}")
        cmd.append("--resume")
    return cmd


def launch(job, gpu, args):
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "logs" / f"{job['method']}_seed{job['seed']}.log"
    environment = dict(os.environ)
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_OFFLINE": "1",
    })
    log = log_path.open("a")
    process = subprocess.Popen(command(job, args), cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    return {"job": job, "gpu": gpu, "process": process, "log_handle": log, "log": str(log_path), "started": time.time()}


def summarize(args, jobs):
    rows = []
    for job in jobs:
        output = Path(job["output"])
        report = json.loads((output / "report.json").read_text())
        inventory = json.loads((output / "checkpoint_inventory.json").read_text())
        rows.append({
            "method": job["method"], "seed": job["seed"], "best_epoch": report["best_epoch"],
            "epochs_run": report["epochs_run"], "epoch_checkpoints": len(inventory["epochs"]),
            "valid_metrics": report["valid_metrics"], "output": str(output),
        })
    result = {
        "schema": "rdid-msa-tav-epoch-checkpoint-retrain-summary-v1",
        "official_test_evaluated": False,
        "checkpoint_selection": "valid_mae",
        "epoch_checkpoints_are_for_analysis_not_test_selection": True,
        "runs": rows,
    }
    atomic_json(result, args.output / "summary.json")
    return result


def main():
    args = parse_args()
    args.output = args.output.resolve()
    args.assets = args.assets.resolve()
    args.test_root = args.test_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    plan = freeze_plan(args)
    jobs = plan["jobs"]
    running = {}
    try:
        while not test_gate(args):
            atomic_json({
                "status": "waiting_for_official_tests", "official_test_scheduler_status": read_status(args.test_root / "dynamic_scheduler_status.json"),
                "official_test_summary_ready": (args.test_root / "all_five_methods_summary.json").is_file(),
            }, args.output / "queue_status.json")
            time.sleep(args.poll_seconds)

        pending = [job for job in jobs if not job_complete(job["output"])]
        while pending or running:
            for identity, item in list(running.items()):
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                item["log_handle"].close()
                del running[identity]
                if returncode != 0:
                    raise RuntimeError(f"retrain failed: {identity}; see {item['log']}")
                if not job_complete(item["job"]["output"]):
                    raise RuntimeError(f"retrain exited without complete audited output: {identity}")

            pending = [job for job in pending if not job_complete(job["output"])]
            states = gpu_states()
            launched = False
            if pending and shutil.disk_usage(args.output).free < 40 * 1024**3:
                raise RuntimeError("less than 40 GiB free; stop before risking incomplete epoch checkpoints")
            for gpu in sorted(args.gpus, key=lambda index: states[index]["free_mib"], reverse=True):
                if not pending or states[gpu]["free_mib"] < args.reserve_mib + args.margin_mib:
                    continue
                job = pending.pop(0)
                identity = f"{job['method']}_seed{job['seed']}"
                running[identity] = launch(job, gpu, args)
                launched = True
                atomic_json({
                    "status": "running", "resource_policy": plan["resource_policy"],
                    "gpu": states, "pending": [f"{item['method']}_seed{item['seed']}" for item in pending],
                    "running": {name: {"gpu": item["gpu"], "pid": item["process"].pid, "started": item["started"]} for name, item in running.items()},
                    "completed": [f"{item['method']}_seed{item['seed']}" for item in jobs if job_complete(item["output"])],
                    "official_test_evaluated": False,
                }, args.output / "queue_status.json")
                time.sleep(args.settle_seconds)
                break
            if not launched:
                atomic_json({
                    "status": "running" if running else "waiting_for_memory",
                    "resource_policy": plan["resource_policy"], "gpu": states,
                    "pending": [f"{item['method']}_seed{item['seed']}" for item in pending],
                    "running": {name: {"gpu": item["gpu"], "pid": item["process"].pid, "started": item["started"]} for name, item in running.items()},
                    "completed": [f"{item['method']}_seed{item['seed']}" for item in jobs if job_complete(item["output"])],
                    "official_test_evaluated": False,
                }, args.output / "queue_status.json")
                time.sleep(args.poll_seconds)
        summary = summarize(args, jobs)
        atomic_json({"status": "complete", "completed_runs": 5, "official_test_evaluated": False}, args.output / "queue_status.json")
        print(json.dumps(summary, ensure_ascii=False))
    except Exception as exc:
        for item in running.values():
            if item["process"].poll() is None:
                item["process"].terminate()
            item["log_handle"].close()
        atomic_json({"status": "failed", "error": repr(exc), "official_test_evaluated": False}, args.output / "queue_status.json")
        raise


if __name__ == "__main__":
    main()
