#!/usr/bin/env python3
"""Fill two GPUs with P0 runs; each method replicates after its own seed13 audit."""
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

from train_video_adaptation_v2 import ROOT, atomic_json, sha256
from audit_tav_seed13_chain import CORE_METHODS, audit

SEEDS = (13, 42, 2026)
SCHEDULE_VERSION = "p0-rolling-replication-v4"


def schedule():
    return [[(method, 13) for method in CORE_METHODS] +
        [(method, seed) for method in CORE_METHODS for seed in (42, 2026)]]


def eligible_job(pending, audited_methods):
    return next(((m,s) for m,s in pending if s == 13 or m in audited_methods), None)


def gpu_state(gpu):
    result = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    if result.returncode:
        return None
    used, total, utilization = map(int, result.stdout.strip().split(","))
    return {"used_mib": used, "total_mib": total, "utilization": utilization}


def admission(state, active_count, slots):
    if state is None or active_count >= slots:
        return False
    # An empty device must actually be idle; unknown occupancy never means free.
    if active_count == 0 and (state["used_mib"] > 2000 or state["utilization"] > 10):
        return False
    reserve, margin = 34*1024, 8*1024
    return ((active_count+1)*reserve + margin <= state["total_mib"]
            and state["used_mib"] + reserve + margin <= state["total_mib"])


def complete(output):
    status = output / "status.json"
    if not status.exists() or json.loads(status.read_text()).get("status") != "complete":
        return False
    for name in ("report.json", "predictions.jsonl", "best.pt", "last.pt", "run_config.json"):
        if not (output / name).is_file():
            raise RuntimeError(f"completed run missing {name}: {output}")
    return True


def command(base, method, seed):
    output = base / "students" / f"{method}_seed{seed}"
    result = [sys.executable, str(ROOT / "project/scripts/train_tav_distillation.py"),
              "--method", method, "--seed", str(seed), "--output", str(output),
              "--assets", str(base / "assets/protocol.json"), "--batch-size", "8",
              "--num-workers", "2", "--epochs", "30", "--patience", "7", "--progress-every", "100"]
    # All utilization diagnostics follow the complete P0 chain.
    result.append("--no-diagnostics")
    if (output / "run_config.json").exists():
        if (output / "last.pt").exists():
            result.append("--resume")
        else:
            archive = output.with_name(output.name + f"_interrupted_before_epoch1_{time.time_ns()}")
            output.rename(archive)
    return result


def verify(plan, assets, verify_media=False):
    for path, digest in {**assets["input_sha256"], **plan["source_sha256"]}.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen source/asset checksum changed: {path}")
    if verify_media:
        for line in Path(assets["student_input_manifest"]).open():
            row = json.loads(line)
            stat = Path(row["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
                raise RuntimeError(f"media identity changed: {row['path']}")


def run(base, slots):
    base.mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(exist_ok=True)
    assets = json.loads((base / "assets/protocol.json").read_text())
    plan_path = base / "plan.json"
    sources = sorted((ROOT / "project/scripts").glob("*.py")) + sorted((ROOT / "project/src/rdid_mosei").glob("*.py"))
    sources += [base / "assets/protocol.json"]
    if not plan_path.exists():
        plan = {"protocol": "tav-m0-m6-v1", "schedule_version":SCHEDULE_VERSION,
                "stages":schedule(), "core_methods":CORE_METHODS, "expected_runs":15,
                "seeds": SEEDS, "gpus": [0,1],
                "slots_per_gpu": slots, "reserve_per_job_gib": 34, "gpu_headroom_gib": 8,
                "batch_size": 8, "gradient_accumulation": 1, "epochs": 30, "patience": 7,
                "source_sha256": {str(p): sha256(p) for p in sources},
                "created_at": time.time(), "official_test_evaluated": False}
        atomic_json(plan, plan_path)
        snapshot = base / "source_snapshot"
        for source in sources:
            if source.is_relative_to(ROOT / "project"):
                target = snapshot / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    plan = json.loads(plan_path.read_text())
    if plan.get("schedule_version") != SCHEDULE_VERSION or plan.get("stages") != json.loads(json.dumps(schedule())):
        raise ValueError("schedule revision must be explicitly recorded before restarting")
    if plan["slots_per_gpu"] != slots:
        raise ValueError("scheduler concurrency differs from frozen plan")
    verify(plan, assets, verify_media=True)
    active = {}
    state_path = base / "status.json"

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def shutdown():
        for item in active.values():
            if item["process"].poll() is None:
                item["process"].terminate()
        for item in active.values():
            try:
                item["process"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                item["process"].kill()
                item["process"].wait()
            item["log"].close()

    try:
        pending = [(m,s) for m,s in schedule()[0]
                   if not complete(base / "students" / f"{m}_seed{s}")]
        audited_methods = set()
        full_audit_done = False
        while pending or active:
            for name, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                del active[name]
                if code != 0 or not complete(base / "students" / name):
                    raise RuntimeError(f"{name} failed, exit={code}; inspect logs/{name}.log")
            for method in CORE_METHODS:
                if method not in audited_methods and complete(base / "students" / f"{method}_seed13"):
                    verify(plan, assets)
                    audit(base, assets, methods=(method,))
                    audited_methods.add(method)
                    print(json.dumps({"event":"method_ready_for_replication", "method":method}), flush=True)
            if len(audited_methods) == len(CORE_METHODS) and not full_audit_done:
                verify(plan, assets, verify_media=True)
                audit(base, assets)
                full_audit_done = True
            gpu_states = {}
            for gpu in sorted((0,1), key=lambda g: sum(x["gpu"]==g for x in active.values())):
                current = gpu_state(gpu)
                gpu_states[str(gpu)] = current
                count = sum(x["gpu"]==gpu for x in active.values())
                job = eligible_job(pending, audited_methods)
                if job is None or not admission(current, count, slots):
                    continue
                available_kib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))
                if available_kib < 12*1024**2:
                    continue
                if shutil.disk_usage(base).free < 15*1024**3:
                    raise RuntimeError("less than 15 GiB disk free; preserve checkpoints and stop queue")
                verify(plan, assets)
                method, seed = job
                pending.remove(job)
                name = f"{method}_seed{seed}"
                cmd = command(base, method, seed)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES":str(gpu), "OMP_NUM_THREADS":"2",
                       "MKL_NUM_THREADS":"2", "TOKENIZERS_PARALLELISM":"false",
                       "HF_HUB_DISABLE_PROGRESS_BARS":"1", "HF_HUB_OFFLINE":"1"}
                log = (base / "logs" / f"{name}.log").open("a")
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[name] = {"process":process, "log":log, "gpu":gpu, "started":time.time()}
                print(json.dumps({"event":"launched", "run":name, "gpu":gpu, "pid":process.pid}), flush=True)
            atomic_json({"status":"running" if active else "waiting_for_resources",
                "schedule_version":SCHEDULE_VERSION, "expected_runs":15, "stage":"rolling_replication",
                "audited_seed13_methods":sorted(audited_methods),
                "methods":list(CORE_METHODS), "pending":[f"{m}_seed{s}" for m,s in pending], "gpu":gpu_states,
                "blocked_on_own_seed13":[f"{m}_seed{s}" for m,s in pending if s!=13 and m not in audited_methods],
                "running":{name:{"gpu":item["gpu"],"pid":item["process"].pid,"started":item["started"]}
                           for name,item in active.items()}, "updated_at":time.time(),
                "official_test_evaluated":False}, state_path)
            if pending or active:
                time.sleep(15)
        # Also revalidate on a restart after all jobs have already completed.
        if not full_audit_done:
            audit(base, assets)
        subprocess.run([sys.executable, str(ROOT / "project/scripts/summarize_tav_distillation.py"),
                        "--output", str(base)], cwd=ROOT, check=True)
        atomic_json({"status":"complete", "stage":"P0_complete", "completed_runs":15, "expected_runs":15,
                     "official_test_evaluated":False,
                     "next_stage":"P1_R_U_RplusU_RtimesU_ablation_then_P2_diagnostics"}, state_path)
    except KeyboardInterrupt as exc:
        shutdown()
        atomic_json({"status":"paused", "reason":str(exc), "resumption":"restart_service"}, state_path)
    except Exception as exc:
        shutdown()
        atomic_json({"status":"failed", "error":repr(exc), "automatic_hyperparameter_retry":False}, state_path)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/tav_main_v1")
    p.add_argument("--slots-per-gpu", type=int, choices=(1,2), default=2)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.output.resolve(), args.slots_per_gpu)
