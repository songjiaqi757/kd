#!/usr/bin/env python3
"""VRAM-aware scheduler for all five TAV methods and all three checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/experiments/tav_main_v1/official_test_20260917"
DEFAULT_MANIFEST = ROOT / "dataset/cmu_mosei/manifests/official_test_windowed.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--launch-free-mib", type=int, default=22000)
    parser.add_argument("--settle-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def status(path):
    path = Path(path)
    if not path.is_file():
        return None
    return read_json(path).get("status")


def load_jobs(output):
    primary = read_json(output / "evaluation_plan.json")
    extension = read_json(output / "m4_extension_plan.json")
    jobs = list(primary["jobs"]) + list(extension["jobs"])
    identities = [(job["method"], int(job["seed"])) for job in jobs]
    expected = [(method, seed) for method in ("M0", "M1", "M3", "M4", "M6") for seed in (13, 42, 2026)]
    if sorted(identities) != sorted(expected) or len(set(identities)) != 15:
        raise ValueError(f"expected exactly five methods x three seeds, got {identities}")
    return jobs


def free_memory_mib(devices):
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    available = {}
    for line in completed.stdout.splitlines():
        index, free = [value.strip() for value in line.split(",")]
        available[f"cuda:{int(index)}"] = int(free)
    return {device: available[device] for device in devices}


def eligible(job):
    source = Path(job["source_run"])
    source_status = status(source / "status.json")
    if source_status == "failed":
        raise RuntimeError(f"source training failed: {source}")
    if source_status != "complete":
        return False
    required = (source / "run_config.json", source / "report.json", source / "best.pt")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"completed source lacks required artifacts: {source}")
    expected_hash = job.get("source_run_config_sha256_precommitted") or job.get("source_run_config_sha256")
    if sha256(source / "run_config.json") != expected_hash:
        raise ValueError(f"source run config changed after precommit: {source}")
    return True


def job_state(job):
    output = Path(job["output"])
    current = status(output / "status.json")
    if current in {"evaluating", "complete"}:
        return current
    if output.exists():
        raise FileExistsError(f"failed/nonstandard output requires audit: {output}")
    return "pending"


def launch(job, device, args, serial):
    output = Path(job["output"])
    log_path = args.output / "logs" / f"dynamic_{job['method']}_seed{job['seed']}.log"
    command = [
        sys.executable, str(ROOT / "project/scripts/evaluate_tav_test.py"),
        "--run", job["source_run"], "--output", str(output),
        "--test-manifest", str(args.test_manifest), "--device", device,
        "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
        "--allow-official-test",
    ]
    environment = dict(os.environ)
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    log = log_path.open("w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    return {
        "identity": f"{job['method']}_seed{job['seed']}", "device": device,
        "pid": process.pid, "process": process, "log_handle": log,
        "log": str(log_path), "serial": serial, "launched_at": time.time(),
    }


def close_finished(running):
    failures = []
    for key, item in list(running.items()):
        returncode = item["process"].poll()
        if returncode is None:
            continue
        item["log_handle"].close()
        if returncode != 0:
            failures.append((key, returncode, item["log"]))
        del running[key]
    if failures:
        raise RuntimeError(f"official-test subprocess failures: {failures}")


def summarize(jobs, output):
    per_run = []
    for job in jobs:
        report = read_json(Path(job["output"]) / "report.json")
        per_run.append({
            "method": report["method"], "seed": report["seed"],
            "best_epoch": report["best_epoch"], "valid_metrics": report["valid_metrics"],
            "test_metrics": report["test_metrics"],
            "end_to_end_inference_seconds": report["end_to_end_inference_seconds"],
            "peak_gpu_memory_gib": report["peak_gpu_memory_gib"],
        })
    by_method = {}
    for method in ("M0", "M1", "M3", "M4", "M6"):
        rows = sorted((row for row in per_run if row["method"] == method), key=lambda row: row["seed"])
        aggregate = {"seeds": [row["seed"] for row in rows], "runs": len(rows)}
        for metric in ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7"):
            values = np.asarray([row["test_metrics"][metric] for row in rows], dtype=np.float64)
            aggregate[metric] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}
        by_method[method] = aggregate
    result = {
        "schema": "rdid-msa-tav-five-method-official-test-summary-v1",
        "official_test_evaluated": True,
        "test_use_policy": "report_only_never_for_configuration_or_checkpoint_selection",
        "runs": 15, "per_run": per_run, "by_method": by_method,
    }
    atomic_json(result, output / "all_five_methods_summary.json")
    atomic_json(result, output / "summary.json")
    m4 = {"schema": "rdid-msa-tav-m4-official-test-summary-v1", "per_run": [row for row in per_run if row["method"] == "M4"], "M4": by_method["M4"], "official_test_evaluated": True, "test_use_policy": result["test_use_policy"]}
    atomic_json(m4, output / "m4_three_seed_summary.json")
    return result


def snapshot(args, jobs, running, memory, state="running"):
    states = {f"{job['method']}_seed{job['seed']}": job_state(job) for job in jobs}
    atomic_json({
        "status": state, "policy": "vram_aware", "launch_free_mib": args.launch_free_mib,
        "free_memory_mib": memory,
        "running_owned": [{key: value for key, value in item.items() if key not in {"process", "log_handle"}} for item in running.values()],
        "counts": {name: sum(value == name for value in states.values()) for name in ("pending", "evaluating", "complete")},
        "jobs": states,
    }, args.output / "dynamic_scheduler_status.json")


def main():
    args = parse_args()
    args.output = args.output.resolve()
    args.test_manifest = args.test_manifest.resolve()
    jobs = load_jobs(args.output)
    atomic_json({
        "schema": "rdid-msa-tav-official-test-checkpoint-scope-v1",
        "methods": ["M0", "M1", "M3", "M4", "M6"],
        "seeds": [13, 42, 2026],
        "runs": 15,
        "checkpoint_per_run": "best.pt selected by valid_mae",
        "excluded_checkpoints": ["last.pt", "per_epoch_checkpoints"],
        "m4_seed42_seed2026_gate": "source training status must be complete",
        "test_use_policy": "report_only_never_for_configuration_or_checkpoint_selection",
    }, args.output / "checkpoint_scope.json")
    running = {}
    serial = 0
    try:
        while True:
            close_finished(running)
            states = {f"{job['method']}_seed{job['seed']}": job_state(job) for job in jobs}
            if all(value == "complete" for value in states.values()):
                summary = summarize(jobs, args.output)
                memory = free_memory_mib(args.devices)
                snapshot(args, jobs, running, memory, state="complete")
                atomic_json({"status": "complete", "jobs": 15, "scheduler": "vram_aware"}, args.output / "batch_status.json")
                print(json.dumps(summary, ensure_ascii=False))
                return

            memory = free_memory_mib(args.devices)
            pending = [
                job for job in jobs
                if states[f"{job['method']}_seed{job['seed']}"] == "pending" and eligible(job)
            ]
            # Prefer adapted runs because they have the larger measured footprint;
            # launching those first makes the threshold conservative for later jobs.
            pending.sort(key=lambda job: (job["method"] not in {"M3", "M4", "M6"}, -int(job["seed"])))
            launched = False
            for device in sorted(args.devices, key=lambda name: memory[name], reverse=True):
                if not pending or memory[device] < args.launch_free_mib:
                    continue
                job = pending.pop(0)
                identity = f"{job['method']}_seed{job['seed']}"
                serial += 1
                running[identity] = launch(job, device, args, serial)
                launched = True
                # Do not speculate about post-launch memory. Wait for allocation,
                # then re-query nvidia-smi before admitting another process.
                snapshot(args, jobs, running, memory)
                time.sleep(args.settle_seconds)
                break
            if not launched:
                snapshot(args, jobs, running, memory)
                time.sleep(args.poll_seconds)
    except Exception as exc:
        for item in running.values():
            if item["process"].poll() is None:
                item["process"].terminate()
            item["log_handle"].close()
        atomic_json({"status": "failed", "error": repr(exc)}, args.output / "dynamic_scheduler_status.json")
        raise


if __name__ == "__main__":
    main()
