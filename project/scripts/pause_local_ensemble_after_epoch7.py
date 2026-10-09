#!/usr/bin/env python3
"""Pause both local MOSEI Ensemble Full KD runs after durable epoch 7."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import time


BASE = Path("/home/wy/sjq/kd/outputs/experiments/multiseed_controls_v1/mosei_ensemble_controls")
TARGET_EPOCH = 7
RUNS = ("ensemble_full_seed42", "ensemble_full_seed2026")


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def process_alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (OSError, IndexError):
        return False


def find_process(required: tuple[str, ...]) -> int | None:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if all(token in command for token in required):
            return int(proc.name)
    return None


def find_job_runner(name: str) -> int | None:
    method, seed = name.rsplit("_seed", 1)
    return find_process((
        "run_multiseed_control_job.py",
        f"--method\x00{method}",
        f"--seed\x00{seed}",
        f"--base\x00{BASE}",
    ))


def controller_alive() -> bool:
    return find_process(("run_multiseed_controls.py", f"--base\x00{BASE}")) is not None


def durable_row(name: str) -> dict | None:
    run = BASE / "students" / name
    history_path = run / "history.json"
    if not history_path.is_file():
        return None
    history = read_json(history_path)
    row = next((item for item in history if int(item["epoch"]) == TARGET_EPOCH), None)
    if row is None:
        return None
    if not (run / "checkpoints" / f"epoch_{TARGET_EPOCH:03d}.pt").is_file():
        return None
    if not (run / "last.pt").is_file():
        return None
    return row


def stop_group(pid: int) -> None:
    group = os.getsid(pid)
    os.killpg(group, signal.SIGTERM)
    deadline = time.time() + 30
    while process_alive(pid) and time.time() < deadline:
        time.sleep(0.5)
    if process_alive(pid):
        os.killpg(group, signal.SIGKILL)
    deadline = time.time() + 10
    while process_alive(pid) and time.time() < deadline:
        time.sleep(0.5)
    if process_alive(pid):
        raise RuntimeError(f"could not stop process group {group}")


def write_paused(name: str, row: dict) -> None:
    run = BASE / "students" / name
    seed = int(name.rsplit("_seed", 1)[1])
    payload = {
        "status": "paused",
        "reason": "user_requested_after_epoch7",
        "epoch": TARGET_EPOCH,
        "valid_metrics": row["valid_metrics"],
        "resumable_last": str((run / "last.pt").resolve()),
        "paused_at_unix": time.time(),
    }
    atomic_json(payload, run / "status.json")
    atomic_json(
        payload | {
            "dataset": "mosei",
            "method": "ensemble_full",
            "seed": seed,
            "run": str(run),
        },
        BASE / "jobs" / f"{name}.json",
    )


def main() -> None:
    with (BASE / ".pause_after_epoch7.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        waiting = set(RUNS)
        rows: dict[str, dict] = {}
        while waiting:
            for name in tuple(waiting):
                row = durable_row(name)
                if row is None:
                    continue
                pid = find_job_runner(name)
                if pid is not None:
                    stop_group(pid)
                rows[name] = row
                write_paused(name, row)
                waiting.remove(name)
                print(f"paused {name} after durable epoch {TARGET_EPOCH}", flush=True)
            if waiting:
                time.sleep(1)

        deadline = time.time() + 60
        while controller_alive() and time.time() < deadline:
            time.sleep(1)
        # The controller records killed workers as failures; write the intended
        # terminal state only after it has exited so it cannot overwrite this.
        for name in RUNS:
            write_paused(name, rows[name])
        atomic_json(
            {
                "status": "paused_user_requested",
                "dataset": "mosei",
                "paused": list(RUNS),
                "epoch": TARGET_EPOCH,
                "resume_from": {
                    name: str((BASE / "students" / name / "last.pt").resolve())
                    for name in RUNS
                },
                "updated_at_unix": time.time(),
            },
            BASE / "queue_status.json",
        )
        print("both local Ensemble Full KD runs are paused", flush=True)


if __name__ == "__main__":
    main()
