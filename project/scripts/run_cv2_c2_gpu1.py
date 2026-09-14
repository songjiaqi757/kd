#!/usr/bin/env python3
"""Run the unfinished C-v2 C2 seed42 on physical GPU1 and summarize seeds13/42."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/student"
OUTPUT = OUTPUT_ROOT / "stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean_seed42"
STATUS = OUTPUT_ROOT / "cv2_c2_gpu1_status.json"
PLAN_PATH = OUTPUT_ROOT / "cv2_c2_tempfix_emptymean_plan.json"
LOG = ROOT / "outputs/logs/cv2_c2_tempfix_emptymean/seed42_gpu1.log"
MINIMUM_FREE_MIB = 32_000


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_inputs(plan: dict) -> None:
    for raw_path, expected in plan["source_sha256"].items():
        path = Path(raw_path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen C-v2 C2 input changed: {path}: {actual} != {expected}")


def wait_for_gpu1_capacity() -> None:
    while True:
        result = subprocess.run(
            [
                "nvidia-smi", "--id=1", "--query-gpu=memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            free, used, utilization = (int(value.strip()) for value in result.stdout.strip().split(","))
            if free >= MINIMUM_FREE_MIB:
                return
            detail = {"free_mib": free, "used_mib": used, "utilization": utilization}
        else:
            detail = {"query_error": result.stderr.strip()}
        atomic_json(
            {
                "status": "waiting_for_gpu1_capacity", "minimum_free_mib": MINIMUM_FREE_MIB,
                "gpu1": detail, "checked_at": time.time(), "official_test_evaluated": False,
            },
            STATUS,
        )
        time.sleep(30)


def archive_incomplete_output() -> str | None:
    if not OUTPUT.exists() or (OUTPUT / "report.json").is_file():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive = OUTPUT.with_name(f"{OUTPUT.name}_stopped_before_gpu1_restart_{stamp}")
    OUTPUT.rename(archive)
    return str(archive)


def validate_report(path: Path, seed: int) -> dict:
    report = json.loads(path.read_text())
    if report.get("seed") != seed or report.get("method") != "ensemble_pair":
        raise RuntimeError(f"unexpected C2 report identity: {path}")
    if report.get("official_test_evaluated") is not False:
        raise RuntimeError(f"official-test seal missing: {path}")
    return report


def main() -> int:
    plan = json.loads(PLAN_PATH.read_text())
    verify_frozen_inputs(plan)
    validate_report(
        OUTPUT_ROOT / "stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean_seed13/report.json", 13,
    )
    wait_for_gpu1_capacity()
    archived = archive_incomplete_output()

    trainer = ROOT / "project/scripts/train_student_lora.py"
    ensemble = [ROOT / f"outputs/probe/cv2_emptymean_seed{seed}/predictions.jsonl" for seed in (2026, 2027, 2028)]
    command = [
        sys.executable, str(trainer), "--output", str(OUTPUT), "--device", "cuda:0",
        "--method", "ensemble_pair", "--teacher-targets",
        str(ROOT / "outputs/probe/official_train_valid_seed2026/predictions.jsonl"),
        "--teacher-targets-ensemble", *(str(path) for path in ensemble),
        "--teacher-calibration-temperature", "1.014997959136963", "--seed", "42",
        "--batch-size", "8", "--gradient-accumulation", "1", "--epochs", "30",
        "--patience", "7", "--num-workers", "2",
    ]
    atomic_json(
        {
            "status": "running", "seed": 42, "physical_gpu": 1, "command": command,
            "started_at": time.time(), "restart_mode": "from_epoch_1",
            "reason": "The preserved checkpoint lacks optimizer and RNG state.",
            "archived_incomplete_output": archived, "official_test_evaluated": False,
        },
        STATUS,
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    with LOG.open("a") as handle:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)

    reports = {
        seed: validate_report(
            OUTPUT_ROOT / f"stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean_seed{seed}/report.json", seed,
        )
        for seed in (13, 42)
    }
    maes = [float(reports[seed]["valid_metrics"]["mae"]) for seed in (13, 42)]
    summary = {
        "status": "complete", "runs": [
            {"seed": seed, "mae": float(reports[seed]["valid_metrics"]["mae"])} for seed in (13, 42)
        ],
        "mean_mae": statistics.mean(maes), "sample_std_mae": statistics.stdev(maes),
        "physical_gpu": 1, "completed_at": time.time(), "official_test_evaluated": False,
    }
    atomic_json(summary, OUTPUT_ROOT / "cv2_c2_tempfix_emptymean_summary.json")
    atomic_json(summary, STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
