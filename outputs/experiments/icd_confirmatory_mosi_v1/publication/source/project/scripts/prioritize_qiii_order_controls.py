#!/usr/bin/env python3
"""Pause MOSEI Random at epoch 15, then resume each run in the next safe slot."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("/ai/sjq/kd/outputs/experiments/multiseed_controls_v1/mosei")
FOLLOWUP = MAIN / "followup_order_controls"
CONTROLLER = ROOT / "project/scripts/run_multiseed_controls.py"
JOB_RUNNER = ROOT / "project/scripts/run_multiseed_control_job.py"
TARGET_EPOCH = 15
RANDOM_RUNS = ("random_orthogonal_seed42", "random_orthogonal_seed2026")
JOB_BUDGET_MIB = 30 * 1024
HEADROOM_MIB = 8 * 1024
SLOTS_PER_GPU = 2
LAUNCH_SETTLE_SECONDS = 120.0


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def complete(path: Path) -> bool:
    try:
        return path.is_file() and read_json(path).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def process_alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.is_file():
        return False
    try:
        return stat.read_text().split()[2] != "Z"
    except (OSError, IndexError):
        return False


def find_job_runner(name: str) -> int | None:
    method, seed = name.rsplit("_seed", 1)
    required = (
        "run_multiseed_control_job.py",
        f"--method\x00{method}",
        f"--seed\x00{seed}",
        f"--base\x00{MAIN}",
    )
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


def process_gpu(pid: int) -> int | None:
    try:
        environment = (Path(f"/proc/{pid}") / "environ").read_bytes().split(b"\0")
    except OSError:
        return None
    prefix = b"CUDA_VISIBLE_DEVICES="
    for item in environment:
        if item.startswith(prefix):
            value = item[len(prefix):].decode(errors="replace")
            if value in {"0", "1"}:
                return int(value)
    return None


def gpu_occupants() -> dict[int, int]:
    """Count live job-runner process groups, including both experiment bases."""
    occupants = {0: 0, 1: 0}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if "run_multiseed_control_job.py" not in command:
            continue
        gpu = process_gpu(int(proc.name))
        if gpu in occupants and process_alive(int(proc.name)):
            occupants[gpu] += 1
    return occupants


def gpu_state(gpu: int) -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True, timeout=20,
    )
    total, used, free, utilization = [int(value.strip()) for value in result.stdout.split(",")]
    return {"total_mib": total, "used_mib": used, "free_mib": free, "utilization": utilization}


def can_launch(gpu: int, occupants: dict[int, int], states: dict[int, dict[str, int]]) -> bool:
    state = states[gpu]
    return (
        occupants[gpu] < SLOTS_PER_GPU
        and state["free_mib"] >= JOB_BUDGET_MIB + HEADROOM_MIB
        and (occupants[gpu] + 1) * JOB_BUDGET_MIB + HEADROOM_MIB <= state["total_mib"]
    )


def launch_random(name: str, gpu: int) -> tuple[subprocess.Popen, object]:
    method, seed = name.rsplit("_seed", 1)
    command = [
        sys.executable, "-u", str(JOB_RUNNER),
        "--dataset", "mosei",
        "--method", method,
        "--seed", seed,
        "--base", str(MAIN),
        "--device", "cuda:0",
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    log_path = MAIN / "logs" / f"{name}.job.log"
    handle = log_path.open("a")
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment,
        stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"resumed {name} on physical GPU {gpu}: pid={process.pid}", flush=True)
    return process, handle


def epoch_is_durable(name: str) -> bool:
    run = MAIN / "students" / name
    history_path = run / "history.json"
    if not history_path.is_file():
        return False
    history = read_json(history_path)
    if not history or int(history[-1]["epoch"]) < TARGET_EPOCH:
        return False
    return (
        (run / "last.pt").is_file()
        and (run / "checkpoints" / f"epoch_{TARGET_EPOCH:03d}.pt").is_file()
    )


def stop_process_group(pid: int) -> None:
    group = os.getsid(pid)
    os.killpg(group, signal.SIGTERM)
    deadline = time.time() + 30
    while process_alive(pid) and time.time() < deadline:
        time.sleep(1)
    if process_alive(pid):
        os.killpg(group, signal.SIGKILL)
        deadline = time.time() + 10
        while process_alive(pid) and time.time() < deadline:
            time.sleep(1)
    if process_alive(pid):
        raise RuntimeError(f"process group did not stop: pid={pid} pgid={group}")


def pause_random_run(name: str) -> None:
    run = MAIN / "students" / name
    pid = find_job_runner(name)
    if pid is not None:
        stop_process_group(pid)
    history = read_json(run / "history.json")
    row = next(item for item in history if int(item["epoch"]) == TARGET_EPOCH)
    payload = {
        "status": "paused",
        "reason": "priority_handoff_to_order_controls",
        "epoch": TARGET_EPOCH,
        "valid_metrics": row["valid_metrics"],
        "resumable_last": str((run / "last.pt").resolve()),
        "paused_at_unix": time.time(),
    }
    atomic_json(payload, run / "status.json")
    atomic_json(
        {
            **payload,
            "dataset": "mosei",
            "method": "random_orthogonal",
            "seed": int(name.rsplit("_seed", 1)[1]),
            "run": str(run),
        },
        MAIN / "jobs" / f"{name}.json",
    )
    print(f"paused {name} after durable epoch {TARGET_EPOCH}", flush=True)


def main() -> None:
    lock_path = MAIN / ".priority_handoff.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        waiting = {
            name for name in RANDOM_RUNS
            if not complete(MAIN / "jobs" / f"{name}.json")
            and read_json(MAIN / "students" / name / "status.json").get("status") != "paused"
        }
        while waiting:
            for name in tuple(waiting):
                if epoch_is_durable(name):
                    pause_random_run(name)
                    waiting.remove(name)
            if waiting:
                # Keep the boundary race below one training step in normal
                # operation; only a fully durable epoch can trigger stopping.
                time.sleep(1)

        atomic_json(
            {
                "status": "paused_for_priority_handoff",
                "paused": list(RANDOM_RUNS),
                "running": ["subset7_seed42"],
                "completed": ["subset7_seed2026"],
                "resume_policy": "resume each Random run in the next safe free GPU slot",
                "updated_at_unix": time.time(),
            },
            MAIN / "queue_status.json",
        )

        # Give the remaining order-control job first claim on the next slot.
        # Once its pending list is empty, Random jobs may independently fill
        # later vacancies while the other order controls continue running.
        while True:
            followup_status = read_json(FOLLOWUP / "queue_status.json")
            state = followup_status.get("status")
            if state == "failed":
                raise RuntimeError(f"follow-up queue failed: {followup_status}")
            if not followup_status.get("pending"):
                break
            time.sleep(10)

        pending = [
            name for name in RANDOM_RUNS
            if not complete(MAIN / "jobs" / f"{name}.json")
            and find_job_runner(name) is None
        ]
        active: dict[str, dict] = {}
        handles: dict[str, object] = {}
        recent_launch: dict[int, float] = {}
        while pending or active:
            for name, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                handles.pop(name).close()
                del active[name]
                if code or not complete(MAIN / "jobs" / f"{name}.json"):
                    raise RuntimeError(f"resumed Random job failed: {name}, returncode={code}")
                print(f"completed resumed job: {name}", flush=True)

            followup_status = read_json(FOLLOWUP / "queue_status.json")
            if followup_status.get("status") == "failed":
                raise RuntimeError(f"follow-up queue failed: {followup_status}")
            occupants = gpu_occupants()
            states = {gpu: gpu_state(gpu) for gpu in (0, 1)}
            now = time.time()
            for gpu in sorted((0, 1), key=lambda value: states[value]["free_mib"], reverse=True):
                if not pending:
                    break
                if now - recent_launch.get(gpu, 0) < LAUNCH_SETTLE_SECONDS:
                    continue
                if not can_launch(gpu, occupants, states):
                    continue
                name = pending.pop(0)
                process, handle = launch_random(name, gpu)
                active[name] = {
                    "process": process,
                    "pid": process.pid,
                    "gpu": gpu,
                    "started_at_unix": time.time(),
                }
                handles[name] = handle
                recent_launch[gpu] = time.time()
                occupants[gpu] += 1

            completed = [
                name for name in ("subset7_seed42", "subset7_seed2026", *RANDOM_RUNS)
                if complete(MAIN / "jobs" / f"{name}.json")
            ]
            atomic_json(
                {
                    "status": "running_resumed_random" if active else "waiting_for_safe_random_slot",
                    "running": {
                        name: {key: value for key, value in item.items() if key != "process"}
                        for name, item in active.items()
                    },
                    "pending": pending,
                    "completed": completed,
                    "gpu_occupants": {str(key): value for key, value in occupants.items()},
                    "gpu": {str(key): value for key, value in states.items()},
                    "updated_at_unix": time.time(),
                },
                MAIN / "queue_status.json",
            )
            if pending or active:
                time.sleep(10)

        atomic_json(
            {
                "status": "complete",
                "dataset": "mosei",
                "completed": ["subset7_seed42", "subset7_seed2026", *RANDOM_RUNS],
                "completed_at_unix": time.time(),
            },
            MAIN / "queue_status.json",
        )
        print("main queue complete after independently resumed Random jobs", flush=True)


if __name__ == "__main__":
    main()
