#!/usr/bin/env python3
"""Evaluate official-test for every completed run of both MOSI rounds.

Runs after the confirmatory campaign finishes (or on demand). For each run the
two retained validation-selected checkpoints are evaluated independently:
``best_valid_mae.pt`` and ``best_valid_acc2_nonzero.pt``. Test numbers are never
used for selection; they are reported side by side for both selections.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/sjq/kd")
BASE = ROOT / "outputs/experiments/icd_confirmatory_mosi_v1"
ROUND1 = ROOT / "outputs/experiments/icd_improvement_mosi_20261107"
OUT = ROOT / "outputs/experiments/official_test_two_rounds_v1"
EVAL = ROOT / "project/scripts/evaluate_confirmatory_test.py"
TEST_MANIFEST = ROOT / "dataset/cmu_mosi/manifests/official_test_windowed.jsonl"
PYTHON = "/home/wy/sjq/miniconda3/envs/kd/bin/python"
POLICIES = ("valid_mae", "valid_acc2_nonzero")
GROUPS = (("round1", ROUND1 / "runs"),
          ("round2", BASE / "runs"),
          ("round2", BASE / "scheduler"),
          ("round2", BASE / "gradient_projected"))


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def discover(rounds: tuple[str, ...] = ("round1", "round2")) -> tuple[list[dict], list[dict]]:
    jobs, skipped = [], []
    for round_name, directory in GROUPS:
        if round_name not in rounds or not directory.is_dir():
            continue
        for run in sorted(directory.iterdir()):
            if not run.is_dir():
                continue
            status_path = run / "status.json"
            status = read(status_path).get("status") if status_path.is_file() else None
            missing = [p for p in POLICIES if not (run / f"best_{p}.pt").is_file()]
            if status != "complete" or missing:
                skipped.append(dict(round=round_name, run=run.name, status=status,
                                    missing_checkpoints=missing))
                continue
            for policy in POLICIES:
                jobs.append(dict(round=round_name, run=run.name, policy=policy,
                                 run_dir=str(run),
                                 output=str(OUT / round_name / run.name / policy)))
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
    log_path = OUT / "logs" / f"{job['round']}_{job['run']}_{job['policy']}.log"
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
            if status in {"complete", "failed"}:
                return status
        write(OUT / "stage_status.json",
              dict(campaign_status=status, waiting=True, updated_unix=time.time()))
        time.sleep(60)


def summarize(jobs: list[dict], skipped: list[dict]) -> dict:
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
            round=job["round"], run=job["run"], policy=job["policy"],
            method=report.get("method"), task_ce=config.get("task_ce"),
            lambda_interaction=config.get("lambda_interaction"),
            lr_schedule=config.get("lr_schedule"),
            selected_epoch=report.get("selected_epoch"),
            valid_mae=valid.get("mae"), valid_acc2_nonzero=valid.get("acc2_nonzero"),
            test_count=test.get("count"), test_mae=test.get("mae"),
            test_pearson=test.get("pearson"),
            test_acc2_nonzero=test.get("acc2_nonzero"),
            test_f1_weighted_nonzero=test.get("f1_weighted_nonzero"),
            test_acc7=test.get("acc7"),
        ))
    rows.sort(key=lambda r: (r["round"], r["run"], r["policy"]))
    summary = dict(schema="icd-official-test-two-rounds-v1",
                   created_unix=time.time(), test_manifest=str(TEST_MANIFEST),
                   test_manifest_sha256=hashlib_sha(TEST_MANIFEST),
                   checkpoint_policies=list(POLICIES),
                   selection_uses_test=False,
                   note=("Both retained checkpoints are reported; the primary "
                         "number remains the valid_mae checkpoint, the acc2 "
                         "checkpoint is auxiliary and never selected post hoc."),
                   runs=rows, skipped=skipped)
    write(OUT / "summary.json", summary)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["round"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# MOSI official-test: first + second round", "",
             f"Manifest: `{TEST_MANIFEST}` (686 windows / 686 utterances)", "",
             "| round | run | checkpoint | epoch | valid MAE | test MAE | test Pearson | test Acc-2 | test F1 | test Acc-7 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {round} | {run} | {policy} | {selected_epoch} | {valid_mae:.4f} | "
                     "{test_mae:.4f} | {test_pearson:.4f} | {test_acc2_nonzero:.4f} | "
                     "{test_f1_weighted_nonzero:.4f} | {test_acc7:.4f} |".format(**row))
    if skipped:
        lines += ["", "## Skipped", ""]
        lines += [f"- {s['round']}/{s['run']}: status={s['status']} missing={s['missing_checkpoints']}"
                  for s in skipped]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    analysis = BASE / "analysis"
    analysis.mkdir(exist_ok=True)
    for name in ("summary.json", "summary.csv", "summary.md"):
        shutil.copy2(OUT / name, analysis / f"official_test_{name}")
    return summary


def hashlib_sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wait", action="store_true",
                        help="do not wait for campaign_status.json")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=12000,
                        help="launch another evaluation only while GPU1 has at "
                             "least this much free memory")
    parser.add_argument("--rounds", default="round1,round2",
                        help="comma separated subset of round1,round2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-publication", action="store_true")
    args = parser.parse_args()
    rounds = tuple(x.strip() for x in args.rounds.split(",") if x.strip())
    unknown = [x for x in rounds if x not in {"round1", "round2"}]
    if unknown:
        parser.error(f"unknown rounds: {unknown}")
    OUT.mkdir(parents=True, exist_ok=True)
    campaign_status = BASE / "campaign_status.json"
    status = wait_for_campaign(not args.no_wait, campaign_status)
    total = len(discover(rounds)[0])
    stamp = dict(campaign_status=status, total_jobs=total, completed=[], failed=[],
                 active=[], updated_unix=time.time())
    write(OUT / "stage_status.json", stamp)
    if args.dry_run:
        jobs, skipped = discover(rounds)
        print(json.dumps(dict(campaign_status=status, jobs=jobs, skipped=skipped),
                         indent=2, ensure_ascii=False))
        return
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
        jobs, skipped = discover(rounds)
        pending = [j for j in jobs if not finished(j)
                   and attempts.get(j["output"], 0) < args.max_attempts
                   and j["output"] not in {r["job"]["output"] for r in active.values()}]
        if not pending and not active:
            break
        for job in pending:
            if len(active) >= max(1, args.max_parallel):
                break
            if gpu_free_mib() < args.min_free_mib:
                break
            attempts[job["output"]] = attempts.get(job["output"], 0) + 1
            active[job["output"]] = launch(job)
        stamp.update(completed=[j["output"] for j in finished_jobs],
                     failed=failed, active=list(active), updated_unix=time.time())
        write(OUT / "stage_status.json", stamp)
        time.sleep(10)
    jobs, skipped = discover(rounds)
    summary = summarize(jobs, skipped)
    stamp.update(completed=[j["output"] for j in jobs if finished(j)],
                 failed=failed, active=[], summary_runs=len(summary["runs"]),
                 rounds=list(rounds),
                 updated_unix=time.time(), finished=True)
    write(OUT / "stage_status.json", stamp)
    if not args.skip_publication and (BASE / "prepare_publication.py").is_file():
        subprocess.run([PYTHON, str(BASE / "prepare_publication.py")], cwd=ROOT,
                       check=False)
    print(json.dumps({k: stamp[k] for k in ("campaign_status", "total_jobs",
                                           "summary_runs", "failed")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
