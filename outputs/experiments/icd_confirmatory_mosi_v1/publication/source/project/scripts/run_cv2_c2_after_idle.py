#!/usr/bin/env python3
"""Wait for audited C-v2 assets and an idle GPU0, then run C2 seeds 13/42."""
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
STATUS = OUTPUT_ROOT / "cv2_c2_tempfix_emptymean_queue_status.json"
ASSET_ROOT = ROOT / "outputs/audits/cv2_assets"
GPU0_UNITS = ("rdid-cv2-cminus1-tempfix-gpu0.service", "rdid-cv2-c1-remaining-gpu0.service")
SEEDS = (13, 42)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active(unit: str) -> bool:
    result = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, timeout=10)
    return result.stdout.strip() in {"active", "activating", "reloading", "deactivating"}


def gpu_state() -> tuple[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        return tuple(int(value.strip()) for value in result.stdout.strip().split(","))
    # A userspace NVML update can temporarily mismatch the still-loaded kernel
    # module. CUDA remains usable for existing and new processes in that case.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    fallback = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; free,total=torch.cuda.mem_get_info(0); print((total-free)//1048576)",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=env,
    )
    return int(fallback.stdout.strip()), 0


def wait_ready() -> None:
    idle = 0
    while idle < 2:
        identity = ASSET_ROOT / "report.json"
        quick = ASSET_ROOT / "quick_report.json"
        assets_ready = all(path.is_file() and json.loads(path.read_text())["status"] == "pass" for path in (identity, quick))
        units = [unit for unit in GPU0_UNITS if active(unit)]
        memory, utilization = gpu_state()
        ready = assets_ready and not units and memory < 2000 and utilization < 10
        idle = idle + 1 if ready else 0
        atomic_json(
            {
                "status": "waiting",
                "assets_ready": assets_ready,
                "active_gpu0_units": units,
                "gpu0_memory_mib": memory,
                "gpu0_utilization": utilization,
                "consecutive_idle_checks": idle,
                "checked_at": time.time(),
            }, STATUS,
        )
        if idle < 2:
            time.sleep(30)


def main() -> int:
    wait_ready()
    trainer = ROOT / "project/scripts/train_student_lora.py"
    ensemble = [ROOT / f"outputs/probe/cv2_emptymean_seed{seed}/predictions.jsonl" for seed in (2026, 2027, 2028)]
    frozen_paths = [
        trainer,
        ROOT / "project/scripts/train_student_baseline.py",
        ROOT / "project/src/rdid_mosei/student.py",
        ROOT / "project/src/rdid_mosei/metrics.py",
        ROOT / "outputs/probe/official_train_valid_seed2026/predictions.jsonl",
        *ensemble,
    ]
    hashes = {str(path): sha256(path) for path in frozen_paths}
    plan = {
        "protocol": "C-v2-C2-uniform-ensemble-pair-tempfix-emptymean",
        "seeds": list(SEEDS),
        "gpu": 0,
        "teacher_calibration_temperature": 1.014997959136963,
        "empty_baseline": "per-probe official-train TAV utterance mean",
        "source_sha256": hashes,
        "official_test_evaluated": False,
    }
    atomic_json(plan, OUTPUT_ROOT / "cv2_c2_tempfix_emptymean_plan.json")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
    log_dir = ROOT / "outputs/logs/cv2_c2_tempfix_emptymean"
    log_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in SEEDS:
        output = OUTPUT_ROOT / f"stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean_seed{seed}"
        report_path = output / "report.json"
        if not report_path.is_file():
            for path, expected in hashes.items():
                if sha256(Path(path)) != expected:
                    raise RuntimeError(f"frozen C-v2 C2 input changed: {path}")
            command = [
                sys.executable, str(trainer), "--output", str(output), "--device", "cuda:0",
                "--method", "ensemble_pair", "--teacher-targets",
                str(ROOT / "outputs/probe/official_train_valid_seed2026/predictions.jsonl"),
                "--teacher-targets-ensemble", *(str(path) for path in ensemble),
                "--teacher-calibration-temperature", "1.014997959136963", "--seed", str(seed),
                "--batch-size", "8", "--gradient-accumulation", "1", "--epochs", "30",
                "--patience", "7", "--num-workers", "2",
            ]
            atomic_json({"status": "running", "seed": seed, "command": command, "started_at": time.time()}, STATUS)
            with (log_dir / f"seed{seed}.log").open("a") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(report_path.read_text())
        if report.get("official_test_evaluated") is not False:
            raise RuntimeError("C2 report lacks sealed official-test assertion")
        runs.append({"seed": seed, "mae": float(report["valid_metrics"]["mae"]), "report": str(report_path)})
    summary = {
        "status": "complete",
        "runs": runs,
        "mean_mae": statistics.mean(run["mae"] for run in runs),
        "sample_std_mae": statistics.stdev(run["mae"] for run in runs),
        "official_test_evaluated": False,
    }
    atomic_json(summary, OUTPUT_ROOT / "cv2_c2_tempfix_emptymean_summary.json")
    atomic_json(summary, STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
