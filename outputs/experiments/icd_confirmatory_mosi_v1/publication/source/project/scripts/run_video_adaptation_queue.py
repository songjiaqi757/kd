#!/usr/bin/env python3
"""Wait for existing jobs, run seed13 A/B/C, promote only through a fixed gate."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/scripts"))
from train_video_adaptation import atomic_json, sha256

MODES = ("frozen_video", "video_lora", "ta_only")
OLD_UNITS = ("rdid-stage-d-gpu1-queue.service", "rdid-stage-d-d1-seed2026.service",
             "rdid-stage-d-d2-seed2026-queue.service")


def gpu_ready(gpu):
    old = []
    for unit in OLD_UNITS:
        result = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, timeout=10)
        status = result.stdout.strip()
        if status not in {"active", "activating", "reloading", "deactivating", "inactive", "failed", "unknown"}:
            return False, {"systemd_error": result.stderr.strip() or status, "unit": unit}
        if status in {"active", "activating", "reloading", "deactivating"}:
            old.append(unit)
    result = subprocess.run(["nvidia-smi", f"--id={gpu}",
                             "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        return False, {"gpu_error": result.stderr.strip() or result.stdout.strip(), "old_units": old}
    memory, utilization = [int(x.strip()) for x in result.stdout.strip().split(",")]
    return not old and memory < 2000 and utilization < 10, {"memory_mib": memory, "utilization": utilization, "old_units": old}


def compare(base, candidate, repetitions=10000):
    def read(path):
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if any(r["split"] != "valid" for r in rows):
            raise ValueError("comparison requires validation-only predictions")
        result = {r["parent_sample_id"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate prediction IDs")
        return result
    left, right = read(base), read(candidate)
    if set(left) != set(right) or not left:
        raise ValueError("prediction IDs differ/empty")
    ids = sorted(left)
    for i in ids:
        if left[i]["target_sentiment"] != right[i]["target_sentiment"] or left[i]["video_id"] != right[i]["video_id"]:
            raise ValueError("prediction labels/video IDs differ")
    y = np.asarray([left[i]["target_sentiment"] for i in ids])
    lp = np.asarray([left[i]["prediction"] for i in ids])
    rp = np.asarray([right[i]["prediction"] for i in ids])
    if not np.isfinite([y,lp,rp]).all():
        raise ValueError("non-finite predictions")
    delta = abs(rp-y)-abs(lp-y)
    videos = sorted({left[i]["video_id"] for i in ids})
    lookup = {v:i for i,v in enumerate(videos)}
    groups = np.asarray([lookup[left[i]["video_id"]] for i in ids])
    sums, counts = np.bincount(groups,weights=delta), np.bincount(groups)
    rng = np.random.default_rng(2026)
    weights = rng.multinomial(len(videos), np.full(len(videos),1/len(videos)), size=repetitions)
    bootstrap = (weights@sums)/(weights@counts)
    return {"delta_mae": float(delta.mean()), "cluster_ci95": np.quantile(bootstrap,[.025,.975]).tolist(),
            "video_clusters":len(videos), "utterances":len(ids), "repetitions":repetitions}


def summarize(output, seeds):
    result = {"seeds":seeds, "runs":{}, "comparisons":{}, "official_test_evaluated":False,
              "inference_scope":"validation, conditional on selected checkpoints; seed13 gate is exploratory"}
    for seed in seeds:
        for mode in MODES:
            directory = output / f"{mode}_seed{seed}"
            if json.loads((directory / "status.json").read_text())["status"] != "complete":
                raise ValueError("cannot summarize incomplete run")
            result["runs"][f"{mode}_seed{seed}"] = json.loads((directory / "report.json").read_text())
        candidate = output / f"video_lora_seed{seed}/predictions.jsonl"
        for mode in ("frozen_video", "ta_only"):
            result["comparisons"][f"video_lora_minus_{mode}_seed{seed}"] = compare(
                output / f"{mode}_seed{seed}/predictions.jsonl", candidate)
    result["mean_valid_mae"] = {mode:statistics.mean(result["runs"][f"{mode}_seed{s}"]["valid_metrics"]["mae"] for s in seeds) for mode in MODES}
    atomic_json(result, output / "summary.json")
    lines=["**Video adaptation results**", "",
           "Official valid only. All numbers are conditional on validation-selected checkpoints.", "",
           "| Mode | Seed | MAE | Pearson | Acc-2 |", "|---|---:|---:|---:|---:|"]
    for seed in seeds:
        for mode in MODES:
            m=result["runs"][f"{mode}_seed{seed}"]["valid_metrics"]
            lines.append(f"| {mode} | {seed} | {m['mae']:.6f} | {m['pearson']:.6f} | {m['acc2_nonzero']:.6f} |")
    lines.extend(["", "| Comparison | ΔMAE | Video-cluster 95% CI |", "|---|---:|---|"])
    for name, r in result["comparisons"].items():
        lo,hi=r["cluster_ci95"]
        lines.append(f"| {name} | {r['delta_mae']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(["", "TA-only uses TA teacher targets. A/B use identical TAV teacher targets.",
                  "Single-seed promotion is an exploratory gate, not a significance claim.",
                  f"Machine-readable results: {output / 'summary.json'}", ""])
    path=ROOT / "docs" / f"{output.name}_results.md"
    temp=path.with_suffix(".md.tmp")
    temp.write_text("\n".join(lines)); os.replace(temp,path)
    return result


def pilot_gate(summary, minimum_improvement):
    mae = summary["mean_valid_mae"]
    return mae["frozen_video"]-mae["video_lora"] >= minimum_improvement and mae["video_lora"] < mae["ta_only"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/student/video_adaptation_v1")
    parser.add_argument("--promote", action="store_true", help="If seed13 gate passes, run the fixed seeds42/2026 matrix")
    parser.add_argument("--minimum-improvement", type=float, default=.005)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        trainer = ROOT / "project/scripts/train_video_adaptation.py"
        sources = [trainer,Path(__file__),ROOT / "project/src/rdid_mosei/video_adaptation.py",
                   ROOT / "project/src/rdid_mosei/student.py", ROOT / "project/scripts/train_student_baseline.py",
                   ROOT / "project/src/rdid_mosei/metrics.py"]
        hashes = {str(p):sha256(p) for p in sources}
        plan = {"protocol":"video-adaptation-v1", "gpu":args.gpu,"pilot_seed":13,
                "promote":args.promote,"promoted_seeds":[42,2026],"modes":list(MODES),
                "minimum_mae_improvement_vs_frozen_video":args.minimum_improvement,
                "must_beat_ta_only":True,"source_sha256":hashes,
                "training":{"batch_size":8,"epochs":30,"patience":7},
                "resource_policy":"wait for old units and two consecutive idle GPU checks", "official_test_evaluated":False}
        plan_path=args.output / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing queue plan differs; use a new output/version")
        atomic_json(plan,plan_path)
        env={**os.environ,"CUDA_VISIBLE_DEVICES":str(args.gpu),"HF_HUB_DISABLE_PROGRESS_BARS":"1",
             "OMP_NUM_THREADS":"4","TOKENIZERS_PARALLELISM":"false"}
        completed_seeds=[]
        try:
            for seed in (13,42,2026):
                if seed!=13:
                    if not args.promote or not pilot_gate(pilot_summary,args.minimum_improvement):
                        break
                for mode in MODES:
                    directory=args.output / f"{mode}_seed{seed}"
                    idle=0
                    while idle<2:
                        ready, info=gpu_ready(args.gpu)
                        idle=idle+1 if ready else 0
                        atomic_json({"status":"waiting_for_gpu","next_mode":mode,"next_seed":seed,
                                     "gpu":args.gpu,"checked_at":time.time(),**info},args.output / "queue_status.json")
                        if idle<2:time.sleep(60)
                    if any(sha256(p)!=h for p,h in hashes.items()):
                        raise RuntimeError("experiment code changed after queue registration")
                    command=[sys.executable,str(trainer),"--mode",mode,"--seed",str(seed),
                             "--output",str(directory),"--device","cuda:0","--batch-size","8"]
                    if (directory / "run_config.json").exists():command.append("--resume")
                    atomic_json({"status":"running","mode":mode,"seed":seed,"command":command,
                                 "started_at":time.time()},args.output / "queue_status.json")
                    with (args.output / f"{mode}_seed{seed}.log").open("a") as log:
                        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                completed_seeds.append(seed)
                summary=summarize(args.output,completed_seeds)
                if seed==13:
                    pilot_summary=summary
                    atomic_json({"passed":pilot_gate(summary,args.minimum_improvement),
                                 "minimum_improvement":args.minimum_improvement,
                                 "mean_valid_mae":summary["mean_valid_mae"],
                                 "note":"Single-seed promotion gate, not a significance claim."},args.output / "pilot_gate.json")
            atomic_json({"status":"complete","completed_seeds":completed_seeds,
                         "pilot_gate_passed":pilot_gate(pilot_summary,args.minimum_improvement)},args.output / "queue_status.json")
        except Exception as exc:
            atomic_json({"status":"failed","error":repr(exc)},args.output / "queue_status.json")
            raise


if __name__=="__main__":
    main()
