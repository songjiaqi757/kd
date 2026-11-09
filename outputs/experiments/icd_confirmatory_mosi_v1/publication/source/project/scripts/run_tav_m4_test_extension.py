#!/usr/bin/env python3
"""Precommitted watcher for M4 seed42/2026 official-test evaluation."""
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
EXTENSION_SEEDS = (42, 2026)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
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


def read_status(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("status")


def freeze_extension(args):
    jobs = []
    for seed, device in zip(EXTENSION_SEEDS, args.devices):
        source = (args.runs / f"M4_seed{seed}").resolve()
        config_path = source / "run_config.json"
        config = json.loads(config_path.read_text())
        if config.get("method") != "M4" or config.get("seed") != seed:
            raise ValueError(f"unexpected M4 source config: {source}")
        if config.get("checkpoint_selection") != "valid_mae" or config.get("official_test_evaluated"):
            raise ValueError(f"source is not valid-only locked: {source}")
        jobs.append({
            "method": "M4", "seed": seed, "device": device,
            "source_run": str(source),
            "source_run_config_sha256_precommitted": sha256(config_path),
            "output": str((args.output / f"M4_seed{seed}").resolve()),
        })
    payload = {
        "schema": "rdid-msa-tav-official-test-extension-plan-v1",
        "precommitted_at_unix": time.time(),
        "selection_rule": "evaluate_both_remaining_M4_seeds_unconditionally_after_best_valid_training_completes",
        "resource_rule": "wait_for_primary_13_run_test_batch_to_finish_then_run_one_M4_seed_per_GPU",
        "checkpoint_selection": "valid_mae",
        "test_use_policy": "report_only_never_for_configuration_or_checkpoint_selection",
        "manifest": str(args.test_manifest.resolve()),
        "manifest_sha256": sha256(args.test_manifest),
        "jobs": jobs,
    }
    path = args.output / "m4_extension_plan.json"
    if path.exists():
        existing = json.loads(path.read_text())
        comparable = {key: value for key, value in existing.items() if key != "precommitted_at_unix"}
        expected = {key: value for key, value in payload.items() if key != "precommitted_at_unix"}
        if comparable != expected:
            raise ValueError("existing M4 extension plan differs")
        return existing
    atomic_json(payload, path)
    return payload


def wait_until_ready(args, plan):
    while True:
        primary = read_status(args.output / "batch_status.json")
        training = {
            f"M4_seed{job['seed']}": read_status(Path(job["source_run"]) / "status.json")
            for job in plan["jobs"]
        }
        if primary == "failed" or any(status == "failed" for status in training.values()):
            raise RuntimeError(f"prerequisite failed: primary={primary}, training={training}")
        ready = primary == "complete" and all(status == "complete" for status in training.values())
        atomic_json({
            "status": "ready" if ready else "waiting",
            "primary_batch_status": primary,
            "training_status": training,
        }, args.output / "m4_extension_status.json")
        if ready:
            return
        time.sleep(args.poll_seconds)


def run_job(job, args):
    source = Path(job["source_run"])
    output = Path(job["output"])
    if sha256(source / "run_config.json") != job["source_run_config_sha256_precommitted"]:
        raise ValueError(f"M4 source config changed after precommit: {source}")
    if read_status(source / "status.json") != "complete":
        raise ValueError(f"M4 source is not complete: {source}")
    if read_status(output / "status.json") == "complete":
        return
    if output.exists():
        raise FileExistsError(f"non-complete official-test output requires audit: {output}")
    log_path = args.output / "logs" / f"M4_seed{job['seed']}.log"
    command = [
        sys.executable, str(ROOT / "project/scripts/evaluate_tav_test.py"),
        "--run", str(source), "--output", str(output),
        "--test-manifest", str(args.test_manifest.resolve()),
        "--device", job["device"], "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers), "--allow-official-test",
    ]
    environment = dict(os.environ)
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    if completed.returncode != 0 or read_status(output / "status.json") != "complete":
        raise RuntimeError(f"M4 official-test extension failed for seed {job['seed']}; see {log_path}")


def summarize(args, plan):
    rows = []
    for seed in (13, 42, 2026):
        run_output = args.output / f"M4_seed{seed}"
        report = json.loads((run_output / "report.json").read_text())
        test_protocol = json.loads((run_output / "run_config.json").read_text())
        rows.append({
            "seed": seed, "best_epoch": report["best_epoch"],
            "valid_metrics": report["valid_metrics"], "test_metrics": report["test_metrics"],
            "checkpoint_sha256": test_protocol["checkpoint_sha256"],
        })
    aggregate = {"seeds": [13, 42, 2026], "runs": 3}
    for metric in ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7"):
        values = np.asarray([row["test_metrics"][metric] for row in rows], dtype=np.float64)
        aggregate[metric] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}
    result = {
        "schema": "rdid-msa-tav-m4-official-test-summary-v1",
        "official_test_evaluated": True,
        "test_use_policy": plan["test_use_policy"],
        "per_run": rows, "M4": aggregate,
    }
    atomic_json(result, args.output / "m4_three_seed_summary.json")
    return result


def main():
    args = parse_args()
    args.runs = args.runs.resolve()
    args.output = args.output.resolve()
    args.test_manifest = args.test_manifest.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    plan = freeze_extension(args)
    try:
        wait_until_ready(args, plan)
        atomic_json({"status": "evaluating"}, args.output / "m4_extension_status.json")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_job, job, args) for job in plan["jobs"]]
            for future in futures:
                future.result()
        summary = summarize(args, plan)
        atomic_json({
            "status": "complete", "summary": str(args.output / "m4_three_seed_summary.json")
        }, args.output / "m4_extension_status.json")
        print(json.dumps(summary, ensure_ascii=False))
    except Exception as exc:
        atomic_json({"status": "failed", "error": repr(exc)}, args.output / "m4_extension_status.json")
        raise


if __name__ == "__main__":
    main()
