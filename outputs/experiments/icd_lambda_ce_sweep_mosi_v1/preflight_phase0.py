#!/usr/bin/env python3
"""Phase 0 preflight for the A-series MOSI CE x ICD-strength sweep.

Verifies that this batch runs the exact frozen assets/source/trainer used by
icd_confirmatory_mosi_v1, checks data splits, GPU/disk headroom, then runs two
1-epoch smoke jobs (smallest lambda, and a CE>0 configuration).
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/sjq/kd")
BASE = ROOT / "outputs/experiments/icd_lambda_ce_sweep_mosi_v1"
REF = ROOT / "outputs/experiments/icd_confirmatory_mosi_v1"
PYTHON = "/home/wy/sjq/miniconda3/envs/kd/bin/python"
TRAINER = ROOT / "project/scripts/train_icd_confirmatory.py"
ASSETS = BASE / "assets/protocol.json"
RESERVATION_MIB = 38000
GPU_SAFETY_MIB = 8192
DISK_SAFETY_BYTES = 20 * 1024**3
CHECKPOINT_BYTES = 512551000
SMOKE_JOBS = [
    ("A1_no_ce_lambda005", 0.0, 0.05),
    ("A4_ce010_lambda010", 0.10, 0.10),
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    force = "--force" in sys.argv
    marker = BASE / "preflight/phase0_complete.json"
    if marker.exists() and not force:
        record = read(marker)
        if record.get("passed") is True:
            print("Phase0 already passed:", marker)
            return
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    reference_phase0 = read(REF / "preflight/phase0_complete.json")
    reference_init = read(REF / "preflight/campaign_initialization_reference.json")

    # 1. Frozen assets must be byte-identical to the confirmatory campaign.
    checks["assets_protocol_sha256_match"] = sha(ASSETS) == sha(REF / "assets/protocol.json")
    details["assets_protocol_sha256"] = sha(ASSETS)
    for name in ["official_train_valid_windowed.jsonl", "teacher_ensemble_confirmatory.jsonl",
                 "interaction_targets.jsonl", "teacher_manifest.json",
                 "student_models_manifest.json"]:
        checks[f"asset_present::{name}"] = (BASE / "assets" / name).exists()

    # 2. Trainer and archived source must be the frozen ones.
    trainer_sha = sha(TRAINER)
    details["trainer_sha256"] = trainer_sha
    checks["trainer_sha256_match"] = trainer_sha == reference_phase0["trainer_sha256"]
    fingerprints_path = REF / "source_fingerprints.json"
    checks["source_fingerprints_sha256_match"] = (
        sha(fingerprints_path) == reference_phase0["frozen_artifact_sha256"]["source_fingerprints.json"])
    fingerprints = read(fingerprints_path)
    mismatched = [path for path, expected in fingerprints.items()
                  if not Path(path).exists() or sha(Path(path)) != expected]
    details["source_file_count"] = len(fingerprints)
    details["source_file_mismatches"] = mismatched[:10]
    checks["all_frozen_source_files_match"] = not mismatched

    # 3. Data splits must match the frozen protocol.
    manifest_rows = [json.loads(line) for line in
                     (BASE / "assets/official_train_valid_windowed.jsonl").read_text().splitlines() if line.strip()]
    train_windows = [row for row in manifest_rows if row["split"] == "train"]
    valid_windows = [row for row in manifest_rows if row["split"] == "valid"]
    train_utterances = {row["parent_sample_id"] for row in train_windows}
    valid_utterances = {row["parent_sample_id"] for row in valid_windows}
    train_videos = {row["video_id"] for row in train_windows}
    valid_videos = {row["video_id"] for row in valid_windows}
    details.update(train_windows=len(train_windows), valid_windows=len(valid_windows),
                   train_utterances=len(train_utterances), valid_utterances=len(valid_utterances),
                   valid_source_videos=len(valid_videos))
    checks["split_counts_match"] = (len(train_windows), len(valid_windows),
                                    len(train_utterances), len(valid_utterances)) == (1286, 229, 1284, 229)
    checks["no_video_leakage"] = not (train_videos & valid_videos)

    # 4. Resource headroom.
    gpu_free = int(subprocess.check_output(
        ["nvidia-smi", "--id=1", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).strip().splitlines()[0])
    details["gpu1_free_mib"] = gpu_free
    checks["gpu1_has_room_for_one_worker"] = gpu_free >= RESERVATION_MIB + GPU_SAFETY_MIB
    required_disk = 6 * 3 * CHECKPOINT_BYTES + DISK_SAFETY_BYTES
    free_disk = shutil.disk_usage(ROOT).free
    details.update(disk_required_bytes=required_disk, disk_free_bytes=free_disk)
    checks["disk_headroom"] = free_disk >= required_disk

    # 5. Dry-run every requested configuration (argument/loss validation only).
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    jobs = read(BASE / "plan.json")["runs"]
    dry_runs = []
    for job in jobs:
        command = [PYTHON, "-u", str(TRAINER), "--method", job["method"],
                   "--experiment-id", job["id"], "--assets", str(ASSETS),
                   "--output", str(BASE / "preflight/dryrun" / job["id"]),
                   "--task-ce", str(job["ce"]), "--lambda-interaction", str(job["lam"]),
                   "--retain-best-mae-and-acc2", "--checkpoint-selection", "valid_mae",
                   "--dry-run"]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        dry_runs.append(dict(id=job["id"], returncode=result.returncode,
                             stderr=result.stderr.strip()[-500:]))
    checks["all_configurations_dry_run"] = all(row["returncode"] == 0 for row in dry_runs)
    details["dry_runs"] = dry_runs

    # 6. Two 1-epoch smoke jobs on 8 windows per split; verify finite losses,
    #    retained checkpoints and the frozen seed13 initialization.
    smoke_dir = BASE / "preflight"
    processes = []
    for name, ce, lam in SMOKE_JOBS:
        output = smoke_dir / f"smoke_{name}"
        if output.exists():
            shutil.rmtree(output)
        command = [PYTHON, "-u", str(TRAINER), "--method", "uniform_interaction",
                   "--experiment-id", "smoke_" + name, "--assets", str(ASSETS),
                   "--output", str(output), "--seed", "13", "--device", "cuda:0",
                   "--batch-size", "8", "--epochs", "1", "--patience", "8",
                   "--retain-best-mae-and-acc2", "--checkpoint-selection", "valid_mae",
                   "--learning-rate", "0.0001", "--weight-decay", "0.01",
                   "--num-workers", "0", "--progress-every", "1",
                   "--task-ce", str(ce), "--lambda-interaction", str(lam),
                   "--lr-schedule", "constant", "--limit-per-split", "8"]
        log = (smoke_dir / f"smoke_{name}.log").open("w")
        processes.append((name, ce, lam, output, log,
                          subprocess.Popen(command, cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)))
    smoke_results = []
    for name, ce, lam, output, log, process in processes:
        code = process.wait()
        log.close()
        if code:
            raise RuntimeError(f"smoke {name} failed with exit code {code}")
        report = read(output / "report.json")
        initialization = read(output / "initialization.json")
        history = read(output / "history.json")
        retained = sorted(path.name for path in output.glob("*.pt"))
        finite = all(abs(value) < 1e9 for value in
                     [history[0]["train_loss"], report["valid_metrics"]["mae"]])
        smoke_results.append(dict(
            id=name, task_ce=report["task_ce"], lambda_interaction=report["lambda_interaction"],
            initialization=initialization, retained=retained, finite=finite,
            peak_gpu_memory_gib=report["peak_gpu_memory_gib"],
            valid_metrics=report["valid_metrics"]))
        if retained != ["best_valid_acc2_nonzero.pt", "best_valid_mae.pt", "last.pt"]:
            raise RuntimeError(f"smoke {name} retention differs: {retained}")
        if initialization["initialization_sha256"] != reference_init["initialization_sha256"]:
            raise RuntimeError(f"smoke {name} initialization differs from confirmatory campaign")
        if not finite:
            raise RuntimeError(f"smoke {name} produced non-finite losses")
        if report["task_ce"] != ce or report["lambda_interaction"] != lam:
            raise RuntimeError(f"smoke {name} weight echo differs")
        shutil.rmtree(output)
    saves = {"schema": "icd-asweep-smoke-v1", "created_unix": time.time(),
             "passed": True, "runs": smoke_results}
    save(smoke_dir / "smoke_complete.json", saves)
    checks["smoke_runs_passed"] = True

    passed = all(checks.values())
    save(BASE / "preflight/campaign_initialization_reference.json", reference_init)
    record = dict(schema="icd-asweep-phase0-v1", passed=passed,
                  created_unix=time.time(), gpu=1, max_parallel=2,
                  memory_reservation_mib=RESERVATION_MIB,
                  reference_campaign=str(REF),
                  reference_initialization=reference_init,
                  checkpoint_retention=["best_valid_mae.pt", "best_valid_acc2_nonzero.pt", "last.pt"],
                  checks=checks, details=details,
                  official_test_evaluated=False)
    save(marker, record)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit("Phase0 preflight failed")


if __name__ == "__main__":
    main()
