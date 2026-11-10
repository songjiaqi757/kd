#!/usr/bin/env python3
"""Official-test evaluation for the A-series after the validation campaign ends.

For every completed A run the two retained validation-selected checkpoints are
evaluated independently: ``best_valid_mae.pt`` and
``best_valid_acc2_nonzero.pt``. Test numbers are reporting-only and are never
used for training, early stopping or checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/sjq/kd")
BASE = ROOT / "outputs/experiments/icd_lambda_ce_sweep_mosi_v1"
OUT = ROOT / "outputs/experiments/official_test_a_series_v1"
EVAL = ROOT / "project/scripts/evaluate_confirmatory_test.py"
TEST_MANIFEST = ROOT / "dataset/cmu_mosi/manifests/official_test_windowed.jsonl"
PYTHON = "/home/wy/sjq/miniconda3/envs/kd/bin/python"
POLICIES = ("valid_mae", "valid_acc2_nonzero")
TERMINAL_STATUS = {"complete", "complete_with_audit_errors", "failed"}


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover() -> tuple[list[dict], list[dict]]:
    jobs, skipped = [], []
    runs_dir = BASE / "runs"
    if not runs_dir.is_dir():
        return jobs, skipped
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue
        status_path = run / "status.json"
        status = read(status_path).get("status") if status_path.is_file() else None
        missing = [policy for policy in POLICIES
                   if not (run / f"best_{policy}.pt").is_file()]
        if status != "complete" or missing:
            skipped.append(dict(run=run.name, status=status,
                                missing_checkpoints=missing))
            continue
        for policy in POLICIES:
            jobs.append(dict(run=run.name, policy=policy, run_dir=str(run),
                             output=str(OUT / run.name / policy)))
    return jobs, skipped


def gpu_free_mib() -> int:
    text = subprocess.check_output(
        ["nvidia-smi", "--id=1", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], text=True)
    return int(text.strip().splitlines()[0])


def finished(job: dict) -> bool:
    status = Path(job["output"]) / "status.json"
    return status.is_file() and read(status).get("status") == "complete"


def launch(job: dict) -> dict:
    output = Path(job["output"])
    if output.exists():
        shutil.rmtree(output)
    log_path = OUT / "logs" / f"{job['run']}_{job['policy']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    log.write(json.dumps(dict(launched_unix=time.time(), job=job)) + "\n")
    log.flush()
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    command = [PYTHON, "-u", str(EVAL), "--run", job["run_dir"],
               "--output", str(output), "--test-manifest", str(TEST_MANIFEST),
               "--checkpoint-selection", job["policy"], "--device", "cuda:0",
               "--batch-size", "8", "--num-workers", "2", "--allow-official-test"]
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return dict(process=process, log=log, job=job, started=time.time())


def wait_for_campaign(wait: bool, status_path: Path) -> str:
    if not wait:
        return "no-wait"
    while True:
        status = None
        if status_path.is_file():
            status = read(status_path).get("status")
            if status in TERMINAL_STATUS:
                return status
        write(OUT / "stage_status.json",
              dict(campaign_status=status, waiting=True, updated_unix=time.time()))
        time.sleep(60)


def summarize(jobs: list[dict], skipped: list[dict], campaign_status: str) -> dict:
    rows = []
    for job in jobs:
        report_path = Path(job["output"]) / "report.json"
        if not report_path.is_file():
            continue
        report = read(report_path)
        config = read(Path(job["run_dir"]) / "run_config.json")
        test = report["metrics"]
        valid = report["selected_valid_metrics"]
        rows.append(dict(
            run=job["run"], policy=job["policy"],
            method=report.get("method"), task_ce=config.get("task_ce"),
            lambda_interaction=config.get("lambda_interaction"),
            lr_schedule=config.get("lr_schedule"),
            selected_epoch=report.get("selected_epoch"),
            valid_mae=valid.get("mae"), valid_acc2_nonzero=valid.get("acc2_nonzero"),
            test_count=test.get("count"), test_mae=test.get("mae"),
            test_pearson=test.get("pearson"),
            test_acc2_nonzero=test.get("acc2_nonzero"),
            test_f1_weighted_nonzero=test.get("f1_weighted_nonzero"),
            test_acc7=test.get("acc7")))
    rows.sort(key=lambda row: (row["run"], row["policy"]))
    summary = dict(schema="icd-asweep-official-test-v1", created_unix=time.time(),
                   campaign=str(BASE), campaign_status=campaign_status,
                   test_manifest=str(TEST_MANIFEST),
                   test_manifest_sha256=sha(TEST_MANIFEST),
                   test_windows=688, test_utterances=686,
                   checkpoint_policies=list(POLICIES),
                   selection_uses_test=False,
                   note=("Both retained checkpoints are reported; the primary "
                         "number remains the valid_mae checkpoint and the acc2 "
                         "checkpoint is auxiliary, never selected post hoc."),
                   runs=rows, skipped=skipped)
    write(OUT / "summary.json", summary)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# MOSI A 系列 official-test（仅报告）", "",
             f"测试清单：`{TEST_MANIFEST}`（688 windows / 686 utterances）。",
             "测试指标在全部训练与 Validation 选择冻结之后计算，未参与任何选择。", "",
             "|Run|checkpoint|epoch|valid MAE|test MAE|test Pearson|test Acc-2|test F1|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {run} | {policy} | {selected_epoch} | {valid_mae:.4f} | "
                     "{test_mae:.4f} | {test_pearson:.4f} | {test_acc2_nonzero:.4f} | "
                     "{test_f1_weighted_nonzero:.4f} |".format(**row))
    if skipped:
        lines += ["", "## Skipped", ""]
        lines += [f"- {entry['run']}: status={entry['status']} "
                  f"missing={entry['missing_checkpoints']}" for entry in skipped]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    analysis = BASE / "analysis"
    analysis.mkdir(exist_ok=True)
    for name in ("summary.json", "summary.csv", "summary.md"):
        shutil.copy2(OUT / name, analysis / f"official_test_{name}")
    return summary


def update_campaign_status(summary: dict) -> None:
    path = BASE / "campaign_status.json"
    record = read(path) if path.is_file() else {}
    record.update(official_test_evaluated=True,
                  official_test_used_for_selection=False,
                  official_test_stage="complete",
                  official_test_summary=str(OUT / "summary.json"),
                  official_test_runs=len(summary["runs"]),
                  updated_at_unix=time.time())
    write(path, record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    campaign_status = wait_for_campaign(not args.no_wait, BASE / "campaign_status.json")
    if args.dry_run:
        jobs, skipped = discover()
        print(json.dumps(dict(campaign_status=campaign_status, jobs=jobs, skipped=skipped),
                         indent=2, ensure_ascii=False))
        return
    stamp = dict(campaign_status=campaign_status, completed=[], failed=[], active=[],
                 updated_unix=time.time())
    write(OUT / "stage_status.json", stamp)
    active: dict[str, dict] = {}
    finished_jobs: list[dict] = []
    failed: list[dict] = []
    attempts: dict[str, int] = {}
    while True:
        for key, record in list(active.items()):
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            del active[key]
            if code == 0 and finished(record["job"]):
                finished_jobs.append(record["job"])
            else:
                failed.append(dict(job=record["job"], exit_code=code))
        jobs, skipped = discover()
        pending = [job for job in jobs if not finished(job)
                   and attempts.get(job["output"], 0) < args.max_attempts
                   and job["output"] not in {record["job"]["output"] for record in active.values()}]
        if not pending and not active:
            break
        for job in pending:
            if len(active) >= max(1, args.max_parallel):
                break
            if gpu_free_mib() < args.min_free_mib:
                break
            attempts[job["output"]] = attempts.get(job["output"], 0) + 1
            active[job["output"]] = launch(job)
        stamp.update(completed=[job["output"] for job in finished_jobs],
                     failed=failed, active=list(active), updated_unix=time.time())
        write(OUT / "stage_status.json", stamp)
        time.sleep(10)
    jobs, skipped = discover()
    summary = summarize(jobs, skipped, campaign_status)
    update_campaign_status(summary)
    stamp.update(completed=[job["output"] for job in jobs if finished(job)],
                 failed=failed, active=[], summary_runs=len(summary["runs"]),
                 updated_unix=time.time(), finished=True)
    write(OUT / "stage_status.json", stamp)
    subprocess.run([PYTHON, str(BASE / "analyze_sweep.py")], cwd=ROOT, check=True)
    print(json.dumps({key: stamp[key] for key in
                      ("campaign_status", "summary_runs", "failed")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
