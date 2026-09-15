#!/usr/bin/env python3
"""Adopt live training PIDs, admit by memory budget, and refill slots without restarting training."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from run_tav_distillation import ROOT, command, complete, verify, gpu_state
from audit_tav_seed13_chain import CORE_METHODS, audit
from train_video_adaptation_v2 import atomic_json

VERSION = "p0-live-five-jobs-v5"


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            return None
        return {"pid": int(pid), "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except FileNotFoundError:
        return None


def alive(item):
    return identity(item["pid"]) == item["identity"]


def validate_process(name, item, base):
    if not alive(item):
        return False
    args = Path(f"/proc/{item['pid']}/cmdline").read_bytes().decode().split("\0")
    method, seed = name.split("_seed")
    expected = {"--method":method, "--seed":seed, "--output":str(base / "students" / name)}
    if str(ROOT / "project/scripts/train_tav_distillation.py") not in args:
        raise ValueError(f"live PID is not this trainer: {name}")
    if any(k not in args or args[args.index(k)+1] != v for k,v in expected.items()):
        raise ValueError(f"live PID arguments differ: {name}")
    env = Path(f"/proc/{item['pid']}/environ").read_bytes().split(b"\0")
    if f"CUDA_VISIBLE_DEVICES={item['gpu']}".encode() not in env:
        raise ValueError(f"live PID GPU binding differs: {name}")
    return True


def budget_mib(name, limits):
    method = name.split("_seed")[0]
    return int(limits["method_budget_gib"]["frozen" if method in ("M0", "M1") else "adapted"] * 1024)


def admit(state, gpu, active, name, limits):
    if state is None or len(active) >= limits["max_jobs"]:
        return False
    occupants = [n for n,v in active.items() if v["gpu"] == gpu]
    if len(occupants) >= limits["slots_per_gpu"][str(gpu)]:
        return False
    if not occupants and (state["used_mib"] > 2000 or state["utilization"] > 10):
        return False
    margin = limits["gpu_headroom_gib"] * 1024
    needed = budget_mib(name, limits)
    planned = sum(budget_mib(n, limits) for n in occupants)
    return (planned + needed + margin <= state["total_mib"]
            and state["used_mib"] + needed + margin <= state["total_mib"])


def load_limits(path):
    limits = json.loads(path.read_text())
    if (not 1 <= limits["max_jobs"] <= 5 or set(limits["slots_per_gpu"]) != {"0", "1"}
            or any(not 1 <= n <= 3 for n in limits["slots_per_gpu"].values())
            or limits["gpu_headroom_gib"] < 8
            or limits["method_budget_gib"]["frozen"] < 16
            or limits["method_budget_gib"]["adapted"] < 34):
        raise ValueError("invalid resource limits")
    return limits


def all_jobs():
    return [(m,13) for m in CORE_METHODS] + [(m,s) for m in CORE_METHODS for s in (42,2026)]


def candidate(pending, audited, gpu, active, state, limits):
    for method, seed in pending:
        name = f"{method}_seed{seed}"
        if (seed == 13 or method in audited) and admit(state, gpu, active, name, limits):
            return method, seed
    return None


def run(base):
    assets = json.loads((base / "assets/protocol.json").read_text())
    plan = json.loads((base / "plan.json").read_text())
    if plan.get("schedule_version") != VERSION:
        raise ValueError("live scheduler plan has not been registered")
    verify(plan, assets, verify_media=True)
    registry = base / "live_processes.json"
    saved = json.loads(registry.read_text()) if registry.exists() else {"running":{}}
    active, children, handles = {}, {}, {}
    for name, item in saved["running"].items():
        if validate_process(name, item, base):
            active[name] = item
            print(json.dumps({"event":"adopted_live_training", "run":name, "pid":item["pid"]}), flush=True)
    # Keep a record even if controller startup fails after adoption.
    def persist():
        atomic_json({"running":active, "updated_at":time.time()}, registry)
    persist()
    pending = [(m,s) for m,s in all_jobs() if f"{m}_seed{s}" not in active
               and not complete(base / "students" / f"{m}_seed{s}")]
    audited, full_audit_done = set(), False
    error = None
    state_path = base / "status.json"

    def stop(signum, frame):
        # The service uses KillMode=process. Stopping/replacing this controller
        # deliberately leaves every registered training PID alive for adoption.
        raise KeyboardInterrupt(f"controller signal {signum}; training left running")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            for name, item in list(active.items()):
                child = children.get(name)
                running = child.poll() is None if child is not None else alive(item)
                if running:
                    continue
                del active[name]
                if name in handles:
                    handles.pop(name).close()
                if not complete(base / "students" / name):
                    raise RuntimeError(f"training exited without completed artifacts: {name}; other runs kept alive")
                print(json.dumps({"event":"completed", "run":name}), flush=True)
                persist()
            for method in CORE_METHODS:
                if method not in audited and complete(base / "students" / f"{method}_seed13"):
                    verify(plan, assets)
                    audit(base, assets, methods=(method,))
                    audited.add(method)
            if len(audited) == len(CORE_METHODS) and not full_audit_done:
                audit(base, assets)
                full_audit_done = True
            # Read on each pass: future limit changes require no process restart.
            limits = load_limits(base / "live_limits.json")
            gpu_states = {}
            for gpu in sorted((0,1), key=lambda g: sum(v["gpu"]==g for v in active.values())):
                state = gpu_state(gpu)
                gpu_states[str(gpu)] = state
                job = candidate(pending, audited, gpu, active, state, limits)
                if job is None:
                    continue
                available = int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))
                if available < 16*1024**2 or shutil.disk_usage(base).free < 15*1024**3:
                    continue
                verify(plan, assets)
                method, seed = job
                name = f"{method}_seed{seed}"
                # command() is only called after the live registry has excluded
                # adopted runs, so no active output can be archived or resumed twice.
                cmd = command(base, method, seed)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES":str(gpu), "OMP_NUM_THREADS":"2",
                       "MKL_NUM_THREADS":"2", "TOKENIZERS_PARALLELISM":"false",
                       "HF_HUB_DISABLE_PROGRESS_BARS":"1", "HF_HUB_OFFLINE":"1"}
                log = (base / "logs" / f"{name}.log").open("a")
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                token = identity(child.pid)
                if token is None:
                    log.close()
                    raise RuntimeError(f"new training exited immediately: {name}")
                active[name] = {"gpu":gpu, "pid":child.pid, "identity":token, "started":time.time()}
                children[name], handles[name] = child, log
                persist()
                pending.remove(job)
                print(json.dumps({"event":"launched", "run":name, "gpu":gpu, "pid":child.pid}), flush=True)
            atomic_json({"status":"running" if active else "waiting_for_resources", "stage":"rolling_replication",
                "schedule_version":VERSION, "expected_runs":15, "running":active, "gpu":gpu_states,
                "limits":limits, "audited_seed13_methods":sorted(audited),
                "pending":[f"{m}_seed{s}" for m,s in pending], "updated_at":time.time(),
                "blocked_on_own_seed13":[f"{m}_seed{s}" for m,s in pending if s!=13 and m not in audited],
                "official_test_evaluated":False}, state_path)
            if pending or active:
                time.sleep(10)
        if not full_audit_done:
            audit(base, assets)
        subprocess.run([sys.executable, str(ROOT / "project/scripts/summarize_tav_distillation.py"),
                        "--output", str(base)], check=True)
        atomic_json({"status":"complete", "stage":"P0_complete", "completed_runs":15,
                     "expected_runs":15, "running":{}, "official_test_evaluated":False}, state_path)
    except KeyboardInterrupt as exc:
        error = {"status":"controller_paused", "reason":str(exc)}
    except Exception as exc:
        error = {"status":"controller_failed", "error":repr(exc)}
        raise
    finally:
        persist()
        if error:
            atomic_json({**error, "running":active, "training_processes_left_running":True,
                         "expected_runs":15, "official_test_evaluated":False}, state_path)
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/tav_main_v1")
    args=p.parse_args()
    with (args.output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.output.resolve())
