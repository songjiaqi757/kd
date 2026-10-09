#!/usr/bin/env python3
"""Resource-aware serial screening queue. Never runs official-test evaluation."""
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


def atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mosi", "mosei"), default="mosi")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42])
    parser.add_argument("--min-gpu-free-mib", type=int, default=32000)
    parser.add_argument("--min-disk-free-gib", type=float, default=12)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {"dataset": args.dataset, "device": args.device, "seeds": args.seeds,
                "modes": ["none", "full", "subset", "interaction"],
                "epochs": 20, "decision_temperature": 1.0, "lambda_decision": 1.0,
                "official_test_evaluated": False, "selection_policies": ["valid_mae", "valid_acc2_nonzero"],
                "screening_gate": "primary valid_acc2_nonzero: both seeds positive, mean Acc2 gain >=0.005, mean MAE degradation <=0.005",
                "next_stage": "freeze candidate on validation; add seed2026 and replicate on other dataset; no automatic test sweep"}
        plan_path = output / "screen_protocol.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing screening protocol differs")
        atomic_json(plan, plan_path)
        logs = output / ".logs"
        logs.mkdir(exist_ok=True)
        for seed in args.seeds:
            for mode in plan["modes"]:
                run = output / f"icd_decision_{mode}_seed{seed}"
                name = run.name
                if (run / "status.json").exists() and json.loads((run / "status.json").read_text()).get("status") == "complete":
                    continue
                while True:
                    if (output / "STOP_AFTER_CURRENT").exists():
                        atomic_json({"status": "stopped_by_marker", "next": name}, output / "queue_status.json")
                        return
                    free = int(subprocess.check_output(["nvidia-smi", "-i", str(args.device), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                    disk = shutil.disk_usage(output).free / 1024**3
                    if free >= args.min_gpu_free_mib and disk >= args.min_disk_free_gib:
                        break
                    atomic_json({"status": "waiting_resources", "next": name, "gpu_free_mib": free, "disk_free_gib": disk}, output / "queue_status.json")
                    time.sleep(30)
                command = [sys.executable, str(ROOT / "project/scripts/train_decision_icd.py"),
                           "--method", "uniform_interaction", "--decision-mode", mode,
                           "--assets", str(ROOT / f"outputs/experiments/main_table_v1/{args.dataset}/assets/protocol.json"),
                           "--output", str(run), "--device", f"cuda:{args.device}", "--seed", str(seed),
                           "--epochs", "20", "--min-epochs", "20", "--patience", "20",
                           "--batch-size", "8", "--learning-rate", "0.0001", "--weight-decay", "0.01",
                           "--checkpoint-selection", "valid_mae", "--retain-best-mae-and-acc2",
                           "--decision-temperature", "1.0", "--lambda-decision", "1.0",
                           "--gradient-diagnostic-every", "100"]
                if (run / "run_config.json").exists():
                    if not (run / "last.pt").exists():
                        raise RuntimeError(f"partial run without resumable checkpoint; inspect: {run}")
                    command.append("--resume")
                with (logs / f"{name}.log").open("a") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    atomic_json({"status": "training", "current": name, "child_pid": process.pid,
                                 "command": command}, output / "queue_status.json")
                    code = process.wait()
                if code:
                    atomic_json({"status": "failed", "current": name, "returncode": code}, output / "queue_status.json")
                    raise SystemExit(code)
                subprocess.run([sys.executable, str(ROOT / "project/scripts/summarize_decision_icd.py"),
                                "--input-root", str(output), "--output", str(output / "summary.json"),
                                "--seeds", *map(str, args.seeds)], check=True, cwd=ROOT)
        atomic_json({"status": "complete", "official_test_evaluated": False,
                     "summary": str(output / "summary.json")}, output / "queue_status.json")


if __name__ == "__main__":
    main()
