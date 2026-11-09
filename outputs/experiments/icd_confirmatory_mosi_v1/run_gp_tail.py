#!/usr/bin/env python3
"""Finish the GP-ICD phase outside the campaign supervisor.

The old supervisor waited for 46 GiB free and stalled while an external job held
~52 GiB of GPU1. This watcher starts the remaining GP run as soon as one worker
fits, resumes it after an external OOM, and then restarts run_campaign.py, which
skips every completed run and proceeds to analysis, official-test and
publication.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/sjq/kd")
BASE = ROOT / "outputs/experiments/icd_confirmatory_mosi_v1"
GP = BASE / "gradient_projected"
PYTHON = "/home/wy/sjq/miniconda3/envs/kd/bin/python"
TRAINER = ROOT / "project/scripts/train_icd_confirmatory.py"
JOBS = {"GP_icd_ce": 0.5, "GP_icd_no_ce": 0.0}
MIN_FREE_MIB = 30000
MAX_ATTEMPTS = 3


def read(path: Path):
    return json.loads(path.read_text())


def status(run: str):
    path = GP / run / "status.json"
    return read(path).get("status") if path.is_file() else None


def free_mib() -> int:
    text = subprocess.check_output(
        ["nvidia-smi", "--id=1", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], text=True)
    return int(text.strip().splitlines()[0])


def lock_held(run: str) -> bool:
    path = GP / run / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def launch(run: str, resume: bool):
    out = GP / run
    command = [PYTHON, "-u", str(TRAINER),
               "--method", "uniform_interaction_gradient_projected",
               "--experiment-id", run,
               "--assets", str(BASE / "assets/protocol.json"),
               "--output", str(out), "--seed", "13", "--device", "cuda:0",
               "--batch-size", "8", "--epochs", "40", "--patience", "8",
               "--retain-best-mae-and-acc2", "--checkpoint-selection", "valid_mae",
               "--learning-rate", "0.0001", "--weight-decay", "0.01",
               "--num-workers", "2", "--progress-every", "20",
               "--diagnostic-every", "100", "--task-ce", str(JOBS[run]),
               "--lambda-interaction", "0.25", "--lr-schedule", "constant"]
    if resume:
        command.append("--resume")
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                       PYTHONUNBUFFERED="1")
    log = (BASE / "logs" / f"{run}.tail.log").open("a")
    log.write(json.dumps(dict(launched_unix=time.time(), resume=resume,
                              command=command)) + "\n")
    log.flush()
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return process, log


def main() -> None:
    active: dict[str, tuple] = {}
    attempts = {run: 0 for run in JOBS}
    while True:
        for run, (process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[run]
            print(json.dumps(dict(event="exit", run=run, code=code)), flush=True)
        complete = [run for run in JOBS if status(run) == "complete"]
        if len(complete) == len(JOBS):
            break
        for run in JOBS:
            if run in active or status(run) == "complete":
                continue
            if lock_held(run):
                continue
            if attempts[run] >= MAX_ATTEMPTS:
                continue
            if free_mib() < MIN_FREE_MIB:
                continue
            out = GP / run
            if status(run) == "failed" and not (out / "last.pt").exists():
                shutil.rmtree(out, ignore_errors=True)
            if (out / "run_config.json").exists() and not (out / "last.pt").exists():
                shutil.rmtree(out, ignore_errors=True)
            resume = (out / "last.pt").exists()
            attempts[run] += 1
            process, log = launch(run, resume)
            active[run] = (process, log)
            print(json.dumps(dict(event="launch", run=run, resume=resume,
                                  attempt=attempts[run],
                                  free_mib=free_mib())), flush=True)
            break
        time.sleep(60)
    print(json.dumps(dict(event="gp_complete",
                          statuses={r: status(r) for r in JOBS})), flush=True)
    subprocess.run([PYTHON, str(BASE / "run_campaign.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
