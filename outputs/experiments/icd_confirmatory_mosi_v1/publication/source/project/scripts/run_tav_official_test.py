#!/usr/bin/env python3
"""Run the frozen set of completed TAV checkpoints on MOSEI official test."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = ROOT / "outputs/experiments/tav_main_v1/students"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/tav_main_v1/official_test_20260917"
DEFAULT_MANIFEST = ROOT / "dataset/cmu_mosei/manifests/official_test_windowed.jsonl"
FROZEN_RUNS = (
    ("M0", 13), ("M0", 42), ("M0", 2026),
    ("M1", 13), ("M1", 42), ("M1", 2026),
    ("M3", 13), ("M3", 42), ("M3", 2026),
    ("M4", 13),
    ("M6", 13), ("M6", 42), ("M6", 2026),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
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


def result_directory(output, method, seed):
    # The first M3 run retained an unsuccessful pre-forward smoke attempt.
    if method == "M3" and seed == 13:
        return output / "M3_seed13_retry1"
    return output / f"{method}_seed{seed}"


def freeze_plan(args):
    jobs = []
    for method, seed in FROZEN_RUNS:
        source = (args.runs / f"{method}_seed{seed}").resolve()
        status = json.loads((source / "status.json").read_text())
        config = json.loads((source / "run_config.json").read_text())
        if status.get("status") != "complete" or config.get("method") != method or config.get("seed") != seed:
            raise ValueError(f"run was not complete at freeze time: {source}")
        jobs.append({
            "method": method,
            "seed": seed,
            "source_run": str(source),
            "source_run_config_sha256": sha256(source / "run_config.json"),
            "checkpoint_sha256": sha256(source / "best.pt"),
            "output": str(result_directory(args.output, method, seed).resolve()),
        })
    payload = {
        "schema": "rdid-msa-tav-official-test-plan-v1",
        "frozen_at_unix": time.time(),
        "selection": "all_and_only_completed_runs_at_user_request_time",
        "excluded_incomplete_runs": ["M4_seed42", "M4_seed2026"],
        "test_use_policy": "report_only_never_for_configuration_or_checkpoint_selection",
        "manifest": str(args.test_manifest.resolve()),
        "manifest_sha256": sha256(args.test_manifest),
        "jobs": jobs,
    }
    plan_path = args.output / "evaluation_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        comparable = {key: value for key, value in existing.items() if key != "frozen_at_unix"}
        expected = {key: value for key, value in payload.items() if key != "frozen_at_unix"}
        if comparable != expected:
            raise ValueError("existing official-test plan differs from frozen run set")
        return existing
    atomic_json(payload, plan_path)
    return payload


def output_status(path):
    status_path = path / "status.json"
    if not status_path.is_file():
        return None
    return json.loads(status_path.read_text()).get("status")


def wait_for_external_job(path, poll_seconds):
    while output_status(path) == "evaluating":
        time.sleep(poll_seconds)
    return output_status(path)


def run_job(job, device, args):
    output = Path(job["output"])
    status = output_status(output)
    if status == "evaluating":
        status = wait_for_external_job(output, args.poll_seconds)
    if status == "complete":
        return {"method": job["method"], "seed": job["seed"], "status": "complete", "reused": True}
    if output.exists():
        raise FileExistsError(f"non-complete official-test output requires audit: {output}")
    log_path = args.output / "logs" / f"{job['method']}_seed{job['seed']}.log"
    command = [
        sys.executable, str(ROOT / "project/scripts/evaluate_tav_test.py"),
        "--run", job["source_run"], "--output", str(output),
        "--test-manifest", str(args.test_manifest.resolve()),
        "--device", device, "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers), "--allow-official-test",
    ]
    environment = dict(os.environ)
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    if completed.returncode != 0 or output_status(output) != "complete":
        raise RuntimeError(f"official-test job failed: {job['method']}_seed{job['seed']}; see {log_path}")
    return {"method": job["method"], "seed": job["seed"], "status": "complete", "reused": False}


def worker(queue, device, args):
    completed = []
    for job in queue:
        completed.append(run_job(job, device, args))
        atomic_json({
            "status": "running", "device": device, "completed": completed,
            "remaining": len(queue) - len(completed),
        }, args.output / f"worker_{device.replace(':', '_')}.json")
    return completed


def summarize(plan, output):
    per_run = []
    for job in plan["jobs"]:
        report = json.loads((Path(job["output"]) / "report.json").read_text())
        per_run.append({
            "method": report["method"], "seed": report["seed"],
            "best_epoch": report["best_epoch"], "valid_metrics": report["valid_metrics"],
            "test_metrics": report["test_metrics"],
            "end_to_end_inference_seconds": report["end_to_end_inference_seconds"],
            "peak_gpu_memory_gib": report["peak_gpu_memory_gib"],
        })
    by_method = {}
    for method in dict.fromkeys(row["method"] for row in per_run):
        rows = [row for row in per_run if row["method"] == method]
        aggregate = {"seeds": [row["seed"] for row in rows], "runs": len(rows)}
        for metric in ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7"):
            values = np.asarray([row["test_metrics"][metric] for row in rows], dtype=np.float64)
            aggregate[metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
            }
        by_method[method] = aggregate
    summary = {
        "schema": "rdid-msa-tav-official-test-summary-v1",
        "official_test_evaluated": True,
        "test_use_policy": plan["test_use_policy"],
        "per_run": per_run,
        "by_method": by_method,
    }
    atomic_json(summary, output / "summary.json")
    return summary


def main():
    args = parse_args()
    args.runs = args.runs.resolve()
    args.output = args.output.resolve()
    args.test_manifest = args.test_manifest.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    plan = freeze_plan(args)
    # Two seed-13 jobs may have been launched manually as the measured smoke
    # pair. Never add another inference process while either one is active.
    while any(output_status(Path(job["output"])) == "evaluating" for job in plan["jobs"]):
        atomic_json({
            "status": "waiting_for_initial_pair", "jobs": len(plan["jobs"]),
            "devices": args.devices,
        }, args.output / "batch_status.json")
        time.sleep(args.poll_seconds)
    queues = [[] for _ in args.devices]
    for index, job in enumerate(plan["jobs"]):
        queues[index % len(args.devices)].append(job)
    atomic_json({"status": "running", "jobs": len(plan["jobs"]), "devices": args.devices}, args.output / "batch_status.json")
    try:
        with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
            futures = [executor.submit(worker, queue, device, args) for queue, device in zip(queues, args.devices)]
            completed = [future.result() for future in futures]
        summary = summarize(plan, args.output)
        atomic_json({"status": "complete", "jobs": len(plan["jobs"])}, args.output / "batch_status.json")
        print(json.dumps(summary, ensure_ascii=False))
    except Exception as exc:
        atomic_json({"status": "failed", "error": repr(exc)}, args.output / "batch_status.json")
        raise


if __name__ == "__main__":
    main()
