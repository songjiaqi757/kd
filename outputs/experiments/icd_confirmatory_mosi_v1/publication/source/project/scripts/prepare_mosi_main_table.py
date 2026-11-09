#!/usr/bin/env python3
"""Prepare the MOSI teacher and fixed-student main-table assets end to end.

The pipeline is intentionally resumable.  It waits until both GPUs have the
registered free-memory capacity, extracts the seven teacher subsets once,
trains/calibrates the three Probes,
freezes the MOSI-only interaction statistics, and then builds the main-table
asset bundle and execution plan.  Official-test media and labels are never
loaded by this pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "project/scripts"
MANIFEST = ROOT / "dataset/cmu_mosi/manifests/official_train_valid_windowed.jsonl"
MEDIA_AUDIT = ROOT / "dataset/cmu_mosi/reports/full_media_audit.json"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/main_table_v1/mosi"
SUBSETS = ("t", "a", "v", "ta", "tv", "av", "tav")
PROBE_SEEDS = (2026, 2027, 2028)
EXPECTED_PARENTS = {"train": 1284, "valid": 229}
ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONUNBUFFERED": "1",
    "FORCE_QWENVL_VIDEO_READER": "decord",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit_manifest() -> dict:
    rows = read_rows(MANIFEST)
    if len(rows) != 1515 or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("MOSI train/valid window manifest count or sample IDs differ")
    if {row["split"] for row in rows} != {"train", "valid"}:
        raise ValueError("MOSI preparation must contain train/valid only")
    parents: dict[str, dict] = {}
    for row in rows:
        identity = {
            key: row[key]
            for key in ("parent_sample_id", "split", "video_id", "sentiment", "class_7_index")
        }
        if parents.setdefault(row["parent_sample_id"], identity) != identity:
            raise ValueError(f"conflicting MOSI parent identity: {row['parent_sample_id']}")
        for key in ("audio_segment_path", "silent_video_path"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
    counts = Counter(item["split"] for item in parents.values())
    if dict(counts) != EXPECTED_PARENTS:
        raise ValueError(f"MOSI official parent counts differ: {counts}")
    split_videos = {
        split: {item["video_id"] for item in parents.values() if item["split"] == split}
        for split in EXPECTED_PARENTS
    }
    if split_videos["train"] & split_videos["valid"]:
        raise ValueError("MOSI train/valid source-video overlap")
    media = json.loads(MEDIA_AUDIT.read_text(encoding="utf-8"))
    if media.get("samples") != 2199 or media.get("passed") != 2199 or media.get("failed") != 0:
        raise ValueError("MOSI full-media audit is not clean")
    return {
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "windows": len(rows),
        "parents": dict(counts),
        "source_videos": {key: len(value) for key, value in split_videos.items()},
        "teacher_jobs": len(rows) * len(SUBSETS),
        "official_test_evaluated": False,
    }


def prepare_plan(base: Path) -> dict:
    audit = audit_manifest()
    source_paths = [
        Path(__file__),
        SCRIPTS / "extract_teacher_probe_features.py",
        SCRIPTS / "train_teacher_probe_v2.py",
        SCRIPTS / "prepare_msa_protocol.py",
        SCRIPTS / "prepare_main_table_assets.py",
        SCRIPTS / "build_main_table_plan.py",
    ]
    model = ROOT / "model/Qwen3-Omni-30B-A3B-Instruct"
    required_model_files = sorted(model.glob("*.json")) + sorted(model.glob("*.safetensors"))
    if not required_model_files:
        raise FileNotFoundError(f"local teacher model is incomplete: {model}")
    plan = {
        "schema": "rdid-msa-mosi-main-table-preparation-v1",
        "dataset_audit": audit,
        "teacher_model": str(model),
        "teacher_model_files": {
            str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in required_model_files
        },
        "subsets": list(SUBSETS),
        "probe_seeds": list(PROBE_SEEDS),
        "resource_policy": {
            "teacher": "two-GPU model parallel; require at least 45000 MiB free on each GPU",
            "probes": "three concurrent processes; GPU0 seeds 2026/2027 and GPU1 seed 2028",
            "probe_minimum_free_mib_per_used_gpu": 4096,
        },
        "video_timing_policy": "sampled_fps_v2",
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "official_test_evaluated": False,
    }
    path = base / "preparation_plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("existing MOSI preparation plan differs; do not silently resume with changed inputs")
    atomic_json(plan, path)
    return plan


def verify_plan(plan: dict) -> None:
    if sha256(MANIFEST) != plan["dataset_audit"]["manifest_sha256"]:
        raise ValueError("MOSI manifest changed after preparation was registered")
    for path, digest in plan["source_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"MOSI preparation source changed: {path}")
    for path, identity in plan["teacher_model_files"].items():
        stat = Path(path).stat()
        if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != identity:
            raise ValueError(f"teacher model file changed: {path}")


def gpu_state(gpu: int, minimum_free_mib: int) -> dict:
    result = subprocess.run(
        [
            "nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode:
        return {"ready": False, "error": result.stderr.strip() or result.stdout.strip()}
    memory, total, utilization = [int(value.strip()) for value in result.stdout.strip().split(",")]
    free = total - memory
    return {
        "ready": free >= minimum_free_mib,
        "memory_mib": memory,
        "total_mib": total,
        "free_mib": free,
        "minimum_free_mib": minimum_free_mib,
        "utilization": utilization,
    }


def write_status(base: Path, **values) -> None:
    payload = {"updated_at": time.time(), "official_test_evaluated": False, **values}
    atomic_json(payload, base / "preparation_status.json")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def wait_for_gpus(base: Path, gpus: tuple[int, ...], stage: str, minimum_free_mib: int) -> None:
    consecutive = 0
    while consecutive < 2:
        states = {str(gpu): gpu_state(gpu, minimum_free_mib) for gpu in gpus}
        consecutive = consecutive + 1 if all(item["ready"] for item in states.values()) else 0
        write_status(
            base,
            status="waiting_for_gpu",
            stage=stage,
            gpus=states,
            consecutive_idle_checks=consecutive,
        )
        if consecutive < 2:
            time.sleep(30)


def validate_features(features: Path, plan: dict) -> None:
    config = json.loads((features / "run_config.json").read_text(encoding="utf-8"))
    expected_jobs = plan["dataset_audit"]["teacher_jobs"]
    if (
        config.get("manifest_sha256") != plan["dataset_audit"]["manifest_sha256"]
        or config.get("video_timing_policy") != "sampled_fps_v2"
        or config.get("subsets") != list(SUBSETS)
        or config.get("jobs") != expected_jobs
    ):
        raise ValueError("MOSI teacher feature configuration differs")
    completed = np.load(features / "completed.npy", mmap_mode="r")
    values = np.load(features / "features.npy", mmap_mode="r")
    if completed.shape != (expected_jobs,) or not bool(completed.all()):
        raise ValueError(f"MOSI teacher features incomplete: {int(completed.sum())}/{expected_jobs}")
    if values.shape != (expected_jobs, 2048):
        raise ValueError(f"unexpected MOSI teacher feature shape: {values.shape}")
    for start in range(0, expected_jobs, 2048):
        if not np.isfinite(values[start : start + 2048]).all():
            raise ValueError("MOSI teacher features contain non-finite values")
    index = read_rows(features / "index.jsonl")
    if len(index) != expected_jobs or len({(row["sample_id"], row["subset"]) for row in index}) != expected_jobs:
        raise ValueError("MOSI teacher feature index coverage differs")


def validate_probe(output: Path, features: Path, seed: int) -> None:
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    temperature = float(report["calibration"]["after"]["temperature"])
    if (
        report.get("seed") != seed
        or Path(report.get("features", "")).resolve() != features.resolve()
        or report.get("official_test_evaluated") is not False
        or not math.isfinite(temperature)
        or not 0.05 <= temperature <= 20.0
        or report.get("train_parent_samples") != EXPECTED_PARENTS["train"]
        or report.get("valid_parent_samples") != EXPECTED_PARENTS["valid"]
    ):
        raise ValueError(f"invalid MOSI Probe report: seed{seed}")
    predictions = read_rows(output / "predictions.jsonl")
    expected = sum(EXPECTED_PARENTS.values()) * len(SUBSETS)
    keys = {(row["parent_sample_id"], row["subset"]) for row in predictions}
    if len(predictions) != expected or len(keys) != expected:
        raise ValueError(f"MOSI Probe coverage differs: seed{seed}")


def run_gpu_step(
    base: Path,
    plan: dict,
    stage: str,
    command: list[str],
    gpus: tuple[int, ...],
    progress_path: Path,
    validate,
) -> None:
    marker = base / f"{stage}.done.json"
    if marker.exists():
        validate()
        return
    verify_plan(plan)
    # The 66 GiB teacher uses both devices via a balanced model map.  Capacity
    # admission permits co-resident jobs while retaining roughly 9 GiB after
    # the teacher's conservative 45 GiB/device budget.
    wait_for_gpus(base, gpus, stage, minimum_free_mib=45000)
    log_path = base / "logs" / f"{stage}.log"
    env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpus)}
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        started = time.time()
        while process.poll() is None:
            progress = {}
            if (progress_path / "completed.npy").is_file():
                completed = np.load(progress_path / "completed.npy", mmap_mode="r")
                progress = {"completed": int(completed.sum()), "total": len(completed)}
            elif (progress_path / "history.json").is_file():
                history = json.loads((progress_path / "history.json").read_text(encoding="utf-8"))
                progress = history[-1] if history else {}
            write_status(
                base,
                status="running",
                stage=stage,
                pid=process.pid,
                elapsed_seconds=round(time.time() - started),
                progress=progress,
                log=str(log_path),
            )
            time.sleep(30)
        if process.returncode:
            raise RuntimeError(f"{stage} exited {process.returncode}; inspect {log_path}")
    validate()
    atomic_json({"stage": stage, "completed_at": time.time()}, marker)


def run_cpu_step(base: Path, plan: dict, stage: str, command: list[str], validate) -> None:
    marker = base / f"{stage}.done.json"
    if marker.exists():
        validate()
        return
    # Recover the narrow crash window after a protocol was atomically frozen
    # but before this orchestrator wrote its own completion marker.
    try:
        validate()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    else:
        atomic_json({"stage": stage, "completed_at": time.time(), "recovered": True}, marker)
        return
    verify_plan(plan)
    log_path = base / "logs" / f"{stage}.log"
    write_status(base, status="running", stage=stage, log=str(log_path))
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
    validate()
    atomic_json({"stage": stage, "completed_at": time.time()}, marker)


def run_probes_parallel(base: Path, plan: dict, features: Path) -> None:
    """Run all unfinished Probes concurrently; two tiny Probes share GPU0."""
    assignments = {2026: 0, 2027: 0, 2028: 1}
    pending = []
    for seed, gpu in assignments.items():
        output = base / "probes" / f"seed{seed}"
        marker = base / f"probe_seed{seed}.done.json"
        if marker.exists():
            validate_probe(output, features, seed)
        else:
            try:
                validate_probe(output, features, seed)
            except (FileNotFoundError, json.JSONDecodeError):
                pending.append((seed, gpu, output, marker))
            else:
                atomic_json(
                    {"stage": f"probe_seed{seed}", "completed_at": time.time(), "recovered": True},
                    marker,
                )
    if not pending:
        return
    verify_plan(plan)
    wait_for_gpus(
        base,
        tuple(sorted({item[1] for item in pending})),
        "probes_parallel",
        minimum_free_mib=4096,
    )
    processes: dict[int, subprocess.Popen] = {}
    handles = {}
    started = time.time()
    try:
        for seed, gpu, output, _ in pending:
            output.mkdir(parents=True, exist_ok=True)
            log_path = base / "logs" / f"probe_seed{seed}.log"
            handle = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(SCRIPTS / "train_teacher_probe_v2.py"),
                "--features", str(features),
                "--output", str(output),
                "--seed", str(seed),
                "--device", "cuda:0",
            ]
            env = {**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)}
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes[seed] = process
            handles[seed] = handle
        while any(process.poll() is None for process in processes.values()):
            progress = {}
            for seed, process in processes.items():
                history_path = base / "probes" / f"seed{seed}" / "history.json"
                history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
                progress[str(seed)] = {
                    "gpu": assignments[seed],
                    "pid": process.pid,
                    "returncode": process.poll(),
                    "latest": history[-1] if history else {},
                }
            write_status(
                base,
                status="running",
                stage="probes_parallel",
                elapsed_seconds=round(time.time() - started),
                progress=progress,
            )
            time.sleep(15)
        failures = {seed: process.returncode for seed, process in processes.items() if process.returncode}
        if failures:
            raise RuntimeError(f"parallel MOSI Probe training failed: {failures}")
        for seed, _, output, marker in pending:
            validate_probe(output, features, seed)
            atomic_json({"stage": f"probe_seed{seed}", "completed_at": time.time()}, marker)
    finally:
        for handle in handles.values():
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(exist_ok=True)
    with (base / ".preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            plan = prepare_plan(base)
            write_status(base, status="prepared", stage="preflight", plan=str(base / "preparation_plan.json"))
            if args.prepare_only:
                return 0
            if shutil.disk_usage(base).free < 10 * 1024**3:
                raise RuntimeError("at least 10 GiB free disk is required for MOSI preparation")

            features = base / "teacher_features"
            run_gpu_step(
                base,
                plan,
                "teacher_features",
                [
                    sys.executable,
                    str(SCRIPTS / "extract_teacher_probe_features.py"),
                    "--manifest", str(MANIFEST),
                    "--output-dir", str(features),
                    "--video-timing-policy", "sampled_fps_v2",
                ],
                (0, 1),
                features,
                lambda: validate_features(features, plan),
            )
            run_probes_parallel(base, plan, features)

            base_assets = base / "base_assets"
            run_cpu_step(
                base,
                plan,
                "base_assets",
                [
                    sys.executable,
                    str(SCRIPTS / "prepare_msa_protocol.py"),
                    "--dataset", "mosi",
                    "--manifest", str(MANIFEST),
                    "--features", str(features),
                    "--probes-root", str(base / "probes"),
                    "--output", str(base_assets),
                    "--expected-train", str(EXPECTED_PARENTS["train"]),
                    "--expected-valid", str(EXPECTED_PARENTS["valid"]),
                ],
                lambda: _validate_protocol(base_assets / "protocol.json", "rdid-msa-three-probe-protocol-v1"),
            )
            assets = base / "assets"
            run_cpu_step(
                base,
                plan,
                "main_table_assets",
                [
                    sys.executable,
                    str(SCRIPTS / "prepare_main_table_assets.py"),
                    "--base", str(base_assets / "protocol.json"),
                    "--output", str(assets),
                ],
                lambda: _validate_protocol(assets / "protocol.json", "rdid-msa-main-table-assets-v1"),
            )
            plan_path = ROOT / "project/configs/main_table_v1/mosi_plan.jsonl"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "build_main_table_plan.py"),
                    "--dataset", "mosi",
                    "--output", str(plan_path),
                ],
                cwd=ROOT,
                env=ENV,
                capture_output=True,
                text=True,
                check=True,
            )
            summary = json.loads(result.stdout.strip().splitlines()[-1])
            # Eight of the 47 student runs are post-selection templates and
            # remain intentionally blocked; six native-system runs have their
            # own external-feature prerequisites.
            if (
                summary["planned_training_runs"] != 53
                or summary["ready_runs"] != 39
                or summary["blocked_runs"] != 14
                or summary["pending_selection_gates"] != 4
            ):
                raise ValueError(f"unexpected MOSI main-table plan summary: {summary}")
            write_status(
                base,
                status="complete",
                stage="complete",
                assets=str(assets / "protocol.json"),
                execution_plan=str(plan_path),
                plan_summary=summary,
            )
            return 0
        except BaseException as error:
            write_status(
                base,
                status="failed",
                stage="stopped",
                error=repr(error),
                resume=f"{sys.executable} {Path(__file__).resolve()}",
            )
            raise


def _validate_protocol(path: Path, schema: str) -> None:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema") != schema or protocol.get("official_test_evaluated") is not False:
        raise ValueError(f"invalid frozen protocol: {path}")
    if schema == "rdid-msa-three-probe-protocol-v1" and protocol.get("dataset") != "mosi":
        raise ValueError("base protocol is not MOSI-specific")


if __name__ == "__main__":
    raise SystemExit(main())
