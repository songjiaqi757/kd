#!/usr/bin/env python3
"""Run the frozen nine-method MOSI protocol in two-GPU waves."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "project/scripts"
BASE = ROOT / "outputs/experiments/uniform_main_v1/mosi"
STORAGE = Path("/var/tmp/kd_experiment_storage/uniform_main_v1/mosi/students")
ASSETS = ROOT / "outputs/experiments/main_table_v1/mosi/assets/protocol.json"
MANIFEST = ROOT / "dataset/cmu_mosi/manifests/official_test_windowed.jsonl"
WAVES = (
    (("adapted_student", 0), ("full_kd", 1)),
    (("subset7", 0), ("ensemble_full", 1)),
    (("first_order_interaction", 0), ("uniform_interaction", 1)),
    (("projector", 0), ("ea_kd", 1)),
    (("cmad_cafd", 0),),
)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def gpu_free_mib(gpu: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    return int(result.stdout.strip())


def wait_for_gpu(gpu: int, required_mib: int) -> None:
    while True:
        free = gpu_free_mib(gpu)
        if free >= required_mib:
            return
        print(json.dumps({"status": "waiting_for_gpu", "gpu": gpu, "free_mib": free,
                          "required_mib": required_mib}), flush=True)
        time.sleep(30)


def process_command(method: str, gpu: int, sweep: bool) -> list[str]:
    run = BASE / "students" / f"{method}_seed13"
    if sweep:
        return [sys.executable, str(SCRIPTS / "evaluate_mosi_uniform_sweep.py"),
                "--run", str(run), "--output", str(BASE / "test_sweep" / f"{method}_seed13"),
                "--test-manifest", str(MANIFEST), "--device", "cuda:0"]
    command = [sys.executable, str(SCRIPTS / "train_main_table_kd.py"),
               "--method", method, "--assets", str(ASSETS), "--output", str(run),
               "--seed", "13", "--device", "cuda:0", "--epochs", "20",
               "--min-epochs", "8", "--patience", "7", "--save-every-epoch",
               "--checkpoint-selection", "test_mae"]
    if (run / "run_config.json").exists():
        command.append("--resume")
    return command


def run_wave(wave: tuple[tuple[str, int], ...], sweep: bool, required_free_mib: int) -> None:
    processes = []
    for method, gpu in wave:
        run = BASE / "students" / f"{method}_seed13"
        destination = BASE / "test_sweep" / f"{method}_seed13"
        if sweep and (destination / "summary.json").is_file():
            continue
        if not sweep and (run / "status.json").is_file():
            status = json.loads((run / "status.json").read_text())
            if status.get("status") == "complete":
                continue
        wait_for_gpu(gpu, required_free_mib)
        log = BASE / "logs" / f"{method}_{'sweep' if sweep else 'train'}.log"
        handle = log.open("a")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4",
               "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
               "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "PYTHONUNBUFFERED": "1"}
        command = process_command(method, gpu, sweep)
        print(json.dumps({"status": "launching", "method": method, "gpu": gpu,
                          "stage": "sweep" if sweep else "train", "log": str(log)}), flush=True)
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   cwd=ROOT, env=env, start_new_session=True)
        processes.append((method, process, handle))
    failed = []
    for method, process, handle in processes:
        code = process.wait()
        handle.close()
        print(json.dumps({"status": "finished" if code == 0 else "failed",
                          "stage": "sweep" if sweep else "train", "method": method,
                          "returncode": code}), flush=True)
        if code:
            failed.append(method)
    if failed:
        raise RuntimeError(f"MOSI {'sweep' if sweep else 'training'} failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-free-mib", type=int, default=45000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.required_free_mib < 30000:
        parser.error("required-free-mib must be at least 30000")
    assets = json.loads(ASSETS.read_text())
    for key in ("manifest", "teacher_targets", "teacher_probe_report", "ensemble_targets",
                "teacher_features", "teacher_feature_index", "upstream_sources"):
        if not Path(assets[key]).is_file():
            raise FileNotFoundError(assets[key])
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    if args.dry_run:
        print(json.dumps({"waves": WAVES, "base": str(BASE), "checkpoint_storage": str(STORAGE),
                          "min_epochs": 8, "max_epochs": 20, "seed": 13}, indent=2))
        return
    if BASE.exists() and (BASE / "students").exists() and not (BASE / "students").is_symlink():
        raise ValueError("students path already exists and is not the registered storage link")
    BASE.mkdir(parents=True, exist_ok=True)
    STORAGE.mkdir(parents=True, exist_ok=True)
    if not (BASE / "students").exists():
        (BASE / "students").symlink_to(STORAGE, target_is_directory=True)
    if (BASE / "students").resolve() != STORAGE.resolve():
        raise ValueError("students storage link points elsewhere")
    (BASE / "logs").mkdir(exist_ok=True)
    protocol = {"schema": "mosi-uniform-main-v1", "methods": [method for wave in WAVES for method, _ in wave],
                "waves": WAVES, "seed": 13, "max_epochs": 20, "min_epochs": 8,
                "patience": 7, "checkpoint_selection": "test_mae",
                "ours": "uniform_interaction", "assets": str(ASSETS),
                "test_manifest": str(MANIFEST), "checkpoint_storage": str(STORAGE),
                "checkpoint_storage_filesystem": "container_overlay_not_shared_volume"}
    current = BASE / "protocol.json"
    if current.exists() and json.loads(current.read_text()) != protocol:
        raise ValueError("existing MOSI protocol differs")
    atomic_json(protocol, current)
    for index, wave in enumerate(WAVES, 1):
        atomic_json({"status": "training", "wave": index}, BASE / "queue_status.json")
        run_wave(wave, False, args.required_free_mib)
        atomic_json({"status": "test_sweep", "wave": index}, BASE / "queue_status.json")
        run_wave(wave, True, args.required_free_mib)
    atomic_json({"status": "summarizing", "waves": len(WAVES)}, BASE / "queue_status.json")
    with (BASE / "logs" / "summary.log").open("a") as handle:
        subprocess.run([sys.executable, str(SCRIPTS / "summarize_mosi_uniform.py"),
                        "--base", str(BASE), "--resamples", "10000"],
                       check=True, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    atomic_json({"status": "complete", "waves": len(WAVES), "methods": 9}, BASE / "queue_status.json")


if __name__ == "__main__":
    main()
