#!/usr/bin/env python3
"""A-series MOSI CE x ICD-strength sweep on physical GPU1, two workers.

The batch reuses the frozen confirmatory protocol (same trainer, archived source,
assets, seed, data split, optimizer and checkpoint policy) and only varies the
Task CE weight c and the ICD weight lambda. Official-test is never evaluated.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/sjq/kd")
BASE = ROOT / "outputs/experiments/icd_lambda_ce_sweep_mosi_v1"
PYTHON = "/home/wy/sjq/miniconda3/envs/kd/bin/python"
TRAINER = ROOT / "project/scripts/train_icd_confirmatory.py"
CHECKPOINT_BYTES = 512551000
GPU_SAFETY_MIB = 8192
DISK_SAFETY_BYTES = 20 * 1024**3
CHILDREN = []
A = [
    ("A1_no_ce_lambda005", "uniform_interaction", 0.0, 0.05),
    ("A2_no_ce_lambda010", "uniform_interaction", 0.0, 0.10),
    ("A3_no_ce_lambda015", "uniform_interaction", 0.0, 0.15),
    ("A4_ce010_lambda010", "uniform_interaction", 0.10, 0.10),
    ("A5_ce025_lambda010", "uniform_interaction", 0.25, 0.10),
    ("A6_ce050_lambda010", "uniform_interaction", 0.50, 0.10),
]


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def stamp(status, **kwargs):
    record = dict(status=status, updated_at_unix=time.time(), gpu=1,
                  official_test_evaluated=False, **kwargs)
    save(BASE / "campaign_status.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def job(name, method, ce, lam, group="runs", schedule="constant", epochs=40):
    return dict(id=name, method=method, ce=ce, lam=lam, group=group,
                schedule=schedule, base_epochs=epochs, budget=epochs,
                output=str(BASE / group / name))


def command(item, resume=False):
    result = [PYTHON, "-u", str(TRAINER),
              "--method", item["method"], "--experiment-id", item["id"],
              "--assets", str(BASE / "assets/protocol.json"), "--output", item["output"],
              "--seed", "13", "--device", "cuda:0", "--batch-size", "8",
              "--epochs", str(item["base_epochs"]), "--patience", "8",
              "--retain-best-mae-and-acc2", "--checkpoint-selection", "valid_mae",
              "--learning-rate", "0.0001", "--weight-decay", "0.01",
              "--num-workers", "2", "--progress-every", "20",
              "--diagnostic-every", "100", "--task-ce", str(item["ce"]),
              "--lambda-interaction", str(item["lam"]),
              "--lr-schedule", item["schedule"]]
    if item["budget"] != item["base_epochs"]:
        result += ["--budget-epochs", str(item["budget"])]
    if resume:
        result += ["--resume"]
    return result


def complete(item):
    out = Path(item["output"])
    if not (out / "status.json").exists() or not (out / "report.json").exists():
        return False
    if read(out / "status.json").get("status") != "complete":
        return False
    report = read(out / "report.json")
    if report.get("official_test_evaluated") is not False:
        raise RuntimeError("Official-test policy violation: " + item["id"])
    if report.get("task_ce") != item["ce"] or report.get("lambda_interaction") != item["lam"]:
        raise RuntimeError("Completed run does not match requested weights: " + item["id"])
    return True


def gpu_free():
    text = subprocess.check_output(
        ["nvidia-smi", "--id=1", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True)
    return int(text.strip().splitlines()[0])


def diagnostics_complete(out):
    return all((out / name).exists() for name in
               ["report.json", "history.json", "predictions.jsonl", "run_config.json",
                "initialization.json", "validation_diagnostics.json",
                "validation_interactions.jsonl"])


def disk_remaining(items, future_runs=0, future_epochs=40):
    pending = sum(not complete(item) for item in items)
    return (pending + future_runs) * 3 * CHECKPOINT_BYTES + DISK_SAFETY_BYTES


def verify_initializations(items):
    reference_path = BASE / "preflight/campaign_initialization_reference.json"
    reference = read(reference_path)
    for item in items:
        path = Path(item["output"]) / "initialization.json"
        if not path.exists():
            continue
        record = read(path)
        key = {name: record.get(name) for name in
               ["initialization_sha256", "total_parameters", "trainable_parameters"]}
        if key["initialization_sha256"] != reference["initialization_sha256"] or \
                key["total_parameters"] != reference["total_parameters"] or \
                key["trainable_parameters"] != reference["trainable_parameters"]:
            raise RuntimeError("Frozen seed13 initialization differs: " + item["id"])


def execute(items, phase, preflight, future_runs=0, future_epochs=40):
    active = {}
    finished = {x["id"] for x in items if complete(x)}
    reservation = int(preflight.get("memory_reservation_mib", 0))
    if not reservation:
        raise RuntimeError("Preflight must record memory_reservation_mib")
    max_workers = min(2, int(preflight.get("max_parallel", 2)))
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    while len(finished) < len(items):
        verify_initializations(items)
        for name, record in list(active.items()):
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            del active[name]
            if code:
                stamp("failed", phase=phase, failed_run=name, exit_code=code,
                      active_pids={k: r["process"].pid for k, r in active.items()})
                for other in active.values():
                    other["process"].terminate()
                for other in active.values():
                    try:
                        other["process"].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        other["process"].kill()
                    other["log"].close()
                raise RuntimeError(name + " exited " + str(code))
            if not diagnostics_complete(Path(record["item"]["output"])):
                raise RuntimeError("Incomplete run artifacts: " + name)
            finished.add(name)
            print(json.dumps({"completed": name, "remaining": len(items) - len(finished)},
                             ensure_ascii=False), flush=True)
        pending = [x for x in items if x["id"] not in finished and x["id"] not in active]
        required = disk_remaining([x for x in items if x["id"] not in finished],
                                  future_runs, future_epochs)
        free_disk = shutil.disk_usage(ROOT).free
        if required > free_disk:
            stamp("waiting_for_storage", phase=phase, required_free_bytes=required,
                  available_free_bytes=free_disk,
                  active_pids={k: r["process"].pid for k, r in active.items()},
                  pending=[x["id"] for x in pending])
            if not active:
                raise RuntimeError("Insufficient storage for retained-checkpoint budget")
        else:
            for item in pending:
                if len(active) >= max_workers:
                    break
                free = gpu_free()
                loading_reserve = sum(reservation for r in active.values()
                                      if time.time() - r["started"] < 90)
                if free - loading_reserve < reservation + GPU_SAFETY_MIB:
                    break
                out = Path(item["output"])
                out.mkdir(parents=True, exist_ok=True)
                config_exists = (out / "run_config.json").exists()
                if config_exists and not (out / "last.pt").exists():
                    raise RuntimeError("Configured run lacks resumable checkpoint: " + item["id"])
                cmd = command(item, config_exists)
                log_path = BASE / "logs" / (item["id"] + ".log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("a")
                log.write("\n" + json.dumps({"launch_unix": time.time(), "command": cmd,
                                             "CUDA_VISIBLE_DEVICES": "1",
                                             "budget": item["budget"]}) + "\n")
                log.flush()
                process = subprocess.Popen(cmd, cwd=ROOT, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                CHILDREN.append(process)
                active[item["id"]] = dict(process=process, log=log, item=item,
                                          started=time.time())
                save(out / "launch.json", dict(command=cmd, pid=process.pid,
                                               launched_unix=time.time(), gpu=1,
                                               budget=item["budget"]))
                break
        stamp("running" if active else "waiting_for_gpu", phase=phase,
              active_pids={k: r["process"].pid for k, r in active.items()},
              completed=sorted(finished),
              pending=[x["id"] for x in items if x["id"] not in active and x["id"] not in finished],
              memory_reservation_mib=reservation)
        if len(finished) < len(items):
            time.sleep(30)
    stamp("phase_complete", phase=phase, completed=sorted(finished))


def select(items):
    return min(items, key=lambda x: read(Path(x["output"]) / "report.json")["valid_metrics"]["mae"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [job(*entry) for entry in A]
        save(BASE / "campaign_jobs.json", dict(phase="a_series", gpu=1, max_parallel=2,
                                               runs=jobs, official_test_evaluated=False))
        if args.plan_only:
            print(json.dumps({"jobs": jobs, "commands": [command(x) for x in jobs]}, indent=2))
            return
        while not (BASE / "preflight/phase0_complete.json").exists():
            stamp("waiting_for_phase0", phase="preflight")
            time.sleep(30)
        preflight = read(BASE / "preflight/phase0_complete.json")
        if preflight.get("passed") is not True:
            raise RuntimeError("Phase0 preflight must explicitly set passed=true")
        execute(jobs, "a_series", preflight, future_runs=0,
                future_epochs=max(x["budget"] for x in jobs))
        best = select(jobs)
        best_no_ce = select([x for x in jobs if x["ce"] == 0.0])
        best_lambda010 = select([x for x in jobs if x["lam"] == 0.10])
        best_mae = read(Path(best["output"]) / "report.json")["valid_metrics"]["mae"]
        equivalent = [x["id"] for x in jobs
                      if read(Path(x["output"]) / "report.json")["valid_metrics"]["mae"] <= best_mae + 0.005]
        save(BASE / "selection.json", dict(
            best_overall=best, best_no_ce=best_no_ce, best_ce_sweep_at_lambda010=best_lambda010,
            equivalence_band_0p005=sorted(equivalent),
            criterion="minimum validation MAE; no test-set or auxiliary-metric selection",
            budget=max(x["budget"] for x in jobs), official_test_evaluated=False))
        subprocess.run([PYTHON, str(BASE / "analyze_sweep.py")], cwd=ROOT, check=True)
        # Reporting-only official-test evaluation of both retained checkpoints.
        subprocess.run([PYTHON, str(BASE / "run_official_test_stage.py"),
                        "--no-wait", "--max-parallel", "2"], cwd=ROOT, check=True)
        audit = read(BASE / "completion_audit.json")
        official_path = ROOT / "outputs/experiments/official_test_a_series_v1/summary.json"
        official = read(official_path) if official_path.exists() else {"runs": []}
        status = "complete" if not audit.get("errors") else "complete_with_audit_errors"
        stamp(status, phase="finalized",
              final_report=str(BASE / "analysis/report.md"),
              completion_audit=str(BASE / "completion_audit.json"),
              audit_errors=audit.get("errors", []),
              official_test_evaluated=True,
              official_test_used_for_selection=False,
              official_test_summary=str(official_path),
              official_test_runs=len(official.get("runs", [])))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        for child in CHILDREN:
            if child.poll() is None:
                child.terminate()
        for child in CHILDREN:
            if child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
        stamp("failed", error=repr(error))
        raise
