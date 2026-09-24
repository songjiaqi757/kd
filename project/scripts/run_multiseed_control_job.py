#!/usr/bin/env python3
"""Run one retained-checkpoint control and evaluate only its valid-best model."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "project/scripts"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return read_json(path).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mosei", "mosi"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, choices=(13, 42, 2026), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--coordinate-seed", type=int, default=20260922)
    return parser.parse_args()


def run_command(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    name = f"{args.method}_seed{args.seed}"
    run = base / "students" / name
    test = base / "official_test" / name
    status_path = base / "jobs" / f"{name}.json"
    assets = ROOT / f"outputs/experiments/main_table_v1/{args.dataset}/assets/protocol.json"
    manifest = ROOT / f"dataset/cmu_{args.dataset}/manifests/official_test_windowed.jsonl"
    if not assets.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"missing assets or test manifest: {assets}, {manifest}")
    lock_path = base / "locks" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json({
            "status": "training",
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "pid": os.getpid(),
            "updated_at_unix": time.time(),
        }, status_path)
        if not complete(run / "status.json"):
            command = [
                sys.executable, "-u", str(SCRIPTS / "train_main_table_all_epochs.py"),
                "--method", args.method,
                "--assets", str(assets),
                "--output", str(run),
                "--seed", str(args.seed),
                "--device", args.device,
                "--epochs", str(args.epochs),
                "--min-epochs", str(args.min_epochs),
                "--patience", str(args.patience),
                "--batch-size", str(args.batch_size),
                "--coordinate-seed", str(args.coordinate_seed),
                "--checkpoint-selection", "valid_mae",
            ]
            if (run / "run_config.json").is_file():
                command.append("--resume")
            run_command(command, base / "logs" / f"{name}.train.log")
        if not complete(run / "status.json"):
            raise RuntimeError(f"training did not complete: {run}")

        atomic_json({
            "status": "official_test",
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "pid": os.getpid(),
            "updated_at_unix": time.time(),
        }, status_path)
        if not complete(test / "status.json"):
            if test.exists():
                archived = test.with_name(f"{test.name}.incomplete_{int(time.time())}")
                shutil.move(test, archived)
            command = [
                sys.executable, "-u", str(SCRIPTS / "evaluate_main_table_test.py"),
                "--run", str(run),
                "--output", str(test),
                "--test-manifest", str(manifest),
                "--device", args.device,
                "--batch-size", str(args.batch_size),
                "--allow-official-test",
            ]
            run_command(command, base / "logs" / f"{name}.test.log")
        if not complete(test / "status.json"):
            raise RuntimeError(f"official test did not complete: {test}")
        atomic_json({
            "status": "complete",
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "run": str(run),
            "official_test": str(test),
            "completed_at_unix": time.time(),
        }, status_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # The scheduler also records the nonzero exit. Keep the traceback in the
        # per-job log while making the last error easy to inspect.
        print(repr(error), file=sys.stderr, flush=True)
        raise
