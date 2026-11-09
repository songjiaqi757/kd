#!/usr/bin/env python3
"""GPU1-only confirmatory MOSI campaign; never evaluates official-test."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path("/data/sjq/kd")
BASE = ROOT / "outputs/experiments/icd_confirmatory_mosi_v1"
PYTHON = "/home/wy/sjq/miniconda3/envs/kd/bin/python"
TRAINER = ROOT / "project/scripts/train_icd_confirmatory.py"
CHECKPOINT_BYTES = 512551000
GPU_SAFETY_MIB = 8192
DISK_SAFETY_BYTES = 20 * 1024**3
CHILDREN = []
F = [
 ("F0_full_kd_ce", "full_kd", .5, 0.),
 ("F1_full_kd_no_ce", "full_kd", 0., 0.),
 ("F2_icd_ce_lambda1", "uniform_interaction", .5, 1.),
 ("F3_icd_no_ce_lambda1", "uniform_interaction", 0., 1.),
 ("F4_icd_ce_lambda05", "uniform_interaction", .5, .5),
 ("F5_icd_no_ce_lambda05", "uniform_interaction", 0., .5),
 ("F6_icd_ce_lambda025", "uniform_interaction", .5, .25),
 ("F7_icd_no_ce_lambda025", "uniform_interaction", 0., .25),
]
def read(path):
    return json.loads(path.read_text())
def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
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
        "--lambda-interaction", str(item["lam"]), "--lr-schedule", item["schedule"]]
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
    if item["budget"] <= item["base_epochs"]:
        return True
    policy = BASE / "budget_extension.json"
    entries = read(policy).get("runs", {}) if policy.exists() else {}
    decision = entries.get(item["id"], {})
    if decision.get("retain_early_stop"):
        return True
    return report.get("effective_max_epochs", 0) >= item["budget"] or report.get("max_epochs", 0) >= item["budget"]
def gpu_free():
    text = subprocess.check_output(["nvidia-smi", "--id=1",
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return int(text.strip().splitlines()[0])
def diagnostics_complete(out):
    return all((out / name).exists() for name in
        ["report.json","history.json","predictions.jsonl","run_config.json",
         "initialization.json","validation_diagnostics.json","validation_interactions.jsonl"])
def disk_remaining(items, future_runs=0, future_epochs=40):
    # User override: only MAE-best, Acc2-best and resumable last.pt are retained.
    # Reserve three full payload sizes per unfinished run, conservatively including atomic writes.
    pending = sum(not complete(item) for item in items)
    return (pending + future_runs) * 3 * CHECKPOINT_BYTES + DISK_SAFETY_BYTES
def verify_initializations(items):
    reference_path=BASE/"preflight/campaign_initialization_reference.json"
    reference=read(reference_path) if reference_path.exists() else None
    for item in items:
        path=Path(item["output"])/"initialization.json"
        if not path.exists():
            continue
        record=read(path)
        key={name:record.get(name) for name in ["initialization_sha256","total_parameters","trainable_parameters"]}
        if not key["initialization_sha256"] or key["total_parameters"]!=819309176 or key["trainable_parameters"]!=42650120:
            raise RuntimeError("Invalid frozen initialization: "+item["id"])
        if reference is None:
            reference=key
            save(reference_path,reference)
        elif key!=reference:
            raise RuntimeError("Seed13 initialization differs across workers: "+item["id"])
def execute(items, phase, preflight, future_runs=0, future_epochs=40):
    active = {}
    finished = {x["id"] for x in items if complete(x)}
    reservation = int(preflight.get("memory_reservation_mib", 0))
    if not reservation:
        peak = preflight.get("peak_gpu_memory_gib", 38)
        if isinstance(peak, dict):
            peak = max(peak.values())
        reservation = max(16000, int(float(peak) * 1024 * 1.15 + 1024))
    max_workers = min(2, int(preflight.get("max_parallel", 2)))
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
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
                      active_pids={k:r["process"].pid for k,r in active.items()})
                # Gracefully preserve resumable epoch checkpoints in other workers.
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
        pending = [x for x in items if x["id"] not in finished and x["id"] not in active]
        required = disk_remaining([x for x in items if x["id"] not in finished],
                                  future_runs, future_epochs)
        free_disk = shutil.disk_usage(ROOT).free
        if required > free_disk:
            stamp("waiting_for_storage", phase=phase, required_free_bytes=required,
                  available_free_bytes=free_disk, active_pids={k:r["process"].pid for k,r in active.items()},
                  pending=[x["id"] for x in pending])
            if not active:
                raise RuntimeError("Insufficient storage for retained-checkpoint campaign budget")
        else:
            for item in pending:
                if len(active) >= max_workers:
                    break
                free = gpu_free()
                # Reserve not-yet-allocated GPU memory for workers still loading.
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
                log.write("\n" + json.dumps({"launch_unix":time.time(),"command":cmd,
                    "CUDA_VISIBLE_DEVICES":"1","budget":item["budget"]}) + "\n")
                log.flush()
                process = subprocess.Popen(cmd, cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                CHILDREN.append(process)
                active[item["id"]] = dict(process=process, log=log, item=item, started=time.time())
                save(out / "launch.json", dict(command=cmd,pid=process.pid,
                    launched_unix=time.time(),gpu=1,budget=item["budget"]))
                # Permit the next scheduling loop to measure allocation before another launch.
                break
        stamp("running" if active else "waiting_for_gpu", phase=phase,
              active_pids={k:r["process"].pid for k,r in active.items()},
              completed=sorted(finished), pending=[x["id"] for x in items
                  if x["id"] not in active and x["id"] not in finished],
              memory_reservation_mib=reservation)
        if len(finished) < len(items):
            time.sleep(30)
    stamp("phase_complete", phase=phase, completed=sorted(finished))
def select(items, positive_only=False):
    candidates = [x for x in items if not positive_only or x["lam"] > 0]
    return min(candidates, key=lambda x:read(Path(x["output"])/"report.json")["valid_metrics"]["mae"])
def extend(items):
    decisions = {}
    for item in items:
        out = Path(item["output"])
        report = read(out / "report.json")
        history = read(out / "history.json")
        # Patience stopping would be unchanged under a longer uniform maximum budget.
        stopped_early = len(history) < item["base_epochs"]
        decisions[item["id"]] = dict(retain_early_stop=stopped_early,
            epochs_run=len(history), original_stop_reason=report.get("stop_reason"),
            rationale="identical patience=8 remains binding" if stopped_early else "resume to common 50-epoch cap")
        item["budget"] = 50
    save(BASE / "budget_extension.json", dict(trigger="at least one factorial best epoch >=36",
        original_budget=40,uniform_budget=50,runs=decisions))
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan-only",action="store_true")
    args=parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/".campaign.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        factorial=[job(*entry) for entry in F]
        save(BASE/"campaign_jobs.json",dict(phase1=factorial,gpu=1,
            dependent_phases=["scheduler","gradient_projected","analysis"],
            official_test_evaluated=False,excluded=["MOSEI","official-test","additional_seeds"]))
        if args.plan_only:
            print(json.dumps({"jobs":factorial,"commands":[command(x) for x in factorial]},indent=2))
            return
        while not (BASE/"preflight/phase0_complete.json").exists():
            stamp("waiting_for_phase0",phase="preflight")
            time.sleep(30)
        preflight=read(BASE/"preflight/phase0_complete.json")
        if preflight.get("passed") is not True:
            raise RuntimeError("Phase0 preflight must explicitly set passed=true")
        if (BASE/"budget_extension.json").exists():
            for item in factorial:
                item["budget"]=50
        execute(factorial,"factorial",preflight,future_runs=3,
                future_epochs=max(x["budget"] for x in factorial))
        budget=max(x["budget"] for x in factorial)
        if budget==40 and any(read(Path(x["output"])/"report.json")["best_epoch"]>=36 for x in factorial):
            extend(factorial)
            budget=50
            execute(factorial,"factorial_uniform_extension",preflight,future_runs=3,future_epochs=50)
        global_best=select(factorial)
        icd_best=select(factorial,positive_only=True)
        save(BASE/"selection.json",dict(phase1_best=global_best,best_positive_icd=icd_best,
            criterion="minimum validation MAE; engineering interpretation occurs in final report",
            budget=budget,official_test_evaluated=False))
        scheduler_job=job("S1_warmup_cosine","uniform_interaction" if global_best["lam"]>0 else "full_kd",
            global_best["ce"],global_best["lam"],group="scheduler",schedule="warmup_cosine",epochs=budget)
        save(BASE/"scheduler/S0_reference.json",dict(source=global_best,
            rationale="exact fixed-LR Phase1 run is reused without retraining",budget=budget))
        execute([scheduler_job],"scheduler",preflight,future_runs=2,future_epochs=budget)
        scheduler_jobs=[scheduler_job]
        if budget==40 and read(Path(scheduler_job["output"])/"report.json")["best_epoch"]>=36:
            # A fresh schedule is necessary: cosine 40->50 cannot be resumed without changing its trajectory.
            fixed50=job("S0_fixed_lr_budget50",global_best["method"],global_best["ce"],
                global_best["lam"],group="scheduler",epochs=50)
            cosine50=job("S1_warmup_cosine_budget50",scheduler_job["method"],global_best["ce"],
                global_best["lam"],group="scheduler",schedule="warmup_cosine",epochs=50)
            save(BASE/"scheduler/extension_policy.json",dict(trigger="S1 best epoch>=36",
                policy="rerun both S0 and S1 from identical seed13 initialization with 50-step-budget schedules",
                excluded_from_final_pair="S1_warmup_cosine",fixed=fixed50,cosine=cosine50))
            execute([fixed50,cosine50],"scheduler_uniform50_rerun",preflight,future_runs=2,future_epochs=budget)
            scheduler_jobs=[fixed50,cosine50]
        # GP modifies gradient routing only, using the constant-LR matched factorial objective.
        gp_jobs=[]
        for ce in (.5,0.):
            name="GP_icd_ce" if ce else "GP_icd_no_ce"
            gp_jobs.append(job(name,"uniform_interaction_gradient_projected",ce,
                icd_best["lam"],group="gradient_projected",epochs=budget))
        save(BASE/"gradient_projected/comparison_protocol.json",dict(reference_icd=icd_best,
            reason="best positive-lambda ICD selected separately if global winner is Full KD",
            jobs=gp_jobs,only_changed_factor="gradient routing",scheduler="constant"))
        execute(gp_jobs,"gradient_projected",preflight)
        cmd=[PYTHON,str(BASE/"analyze_campaign.py")]
        subprocess.run(cmd,cwd=ROOT,check=True)
        # Validation-only selection stays frozen; official-test is reported
        # afterwards for both retained checkpoints of every completed run.
        subprocess.run([PYTHON,str(BASE/"run_official_test_stage.py"),"--no-wait",
                        "--max-parallel","4"],
                       cwd=ROOT,check=True)
        subprocess.run([PYTHON,str(BASE/"prepare_publication.py")],cwd=ROOT,check=True)
        stamp("complete",phase="finalized",final_report=str(BASE/"analysis/final_report.md"),
            completion_audit=str(BASE/"completion_audit.json"))
if __name__=="__main__":
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
        stamp("failed",error=repr(error))
        raise
