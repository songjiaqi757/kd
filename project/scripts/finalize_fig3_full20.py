#!/usr/bin/env python3
"""Collect Qiii results, verify 20-epoch coverage, and redraw Fig. 3."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/fig3_full20_v1"
STATUS = BASE / "finalizer_status.json"
LOCAL_STATUS = BASE / "local_mosi_status.json"
REMOTE_ROOT = "/ai/sjq/kd"
REMOTE_STATUS = "outputs/experiments/fig3_full20_v1/remote_status.json"
SSH = [
    "ssh", "-o", "BatchMode=yes", "-i", "/home/wy/sjq/private_key_sjq.pem",
    "-p", "34989", "root@172.23.166.144",
]
REMOTE_PATHS = (
    "outputs/experiments/uniform_followup_v1/mosei/students/adapted_student_seed13",
    "outputs/experiments/uniform_followup_v1/mosei/students/subset7_seed13",
    "outputs/experiments/uniform_followup_v1/mosei/test_epoch_sweeps/adapted_student_seed13",
    "outputs/experiments/uniform_followup_v1/mosei/test_epoch_sweeps/subset7_seed13",
    "outputs/experiments/tav_epoch_checkpoints_v1/students/M3_seed13",
    "outputs/experiments/tav_epoch_checkpoints_v1/students/M4_seed13",
    "outputs/experiments/tav_epoch_checkpoints_v1/checkpoint_diagnostic_test_gpu1_20260919/M3_seed13",
    "outputs/experiments/tav_epoch_checkpoints_v1/checkpoint_diagnostic_test_gpu1_20260919/M4_seed13",
)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def remote_json(path: str) -> dict | None:
    result = subprocess.run(
        [*SSH, f"cd {REMOTE_ROOT} && test -f {path} && cat {path}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def wait_for_jobs() -> None:
    while True:
        local = json.loads(LOCAL_STATUS.read_text()) if LOCAL_STATUS.is_file() else None
        remote = remote_json(REMOTE_STATUS)
        local_state = None if local is None else local.get("status")
        remote_state = None if remote is None else remote.get("status")
        atomic_json({
            "schema": "fig3-full20-finalizer-v1", "status": "waiting",
            "local_mosi": local_state, "remote_mosei": remote_state,
            "updated_at_unix": time.time(),
        }, STATUS)
        if local_state == "failed":
            raise RuntimeError(f"local MOSI continuation failed: {local}")
        if remote_state == "failed":
            raise RuntimeError(f"remote MOSEI continuation failed: {remote}")
        if local_state == remote_state == "complete":
            return
        time.sleep(60)


def collect_remote() -> None:
    command = f"cd {REMOTE_ROOT} && tar -cf - " + " ".join(REMOTE_PATHS)
    remote = subprocess.Popen([*SSH, command], stdout=subprocess.PIPE)
    if remote.stdout is None:
        raise RuntimeError("failed to open remote tar stream")
    local = subprocess.run(["tar", "-xf", "-"], cwd=ROOT, stdin=remote.stdout)
    remote.stdout.close()
    remote_code = remote.wait()
    if local.returncode or remote_code:
        raise RuntimeError(f"result collection failed: remote={remote_code}, local={local.returncode}")
    subprocess.run([
        sys.executable, str(ROOT / "project/scripts/rewrite_json_path_prefix.py"),
        *sum((["--root", str(ROOT / path)] for path in REMOTE_PATHS), []),
        "--old", "/ai/sjq/kd", "--new", str(ROOT),
    ], cwd=ROOT, check=True)


def verify() -> list[dict]:
    checks = []
    main = {
        "MOSEI Adapted Student": ROOT / "outputs/experiments/uniform_followup_v1/mosei/test_epoch_sweeps/adapted_student_seed13/summary.json",
        "MOSEI Subset-7": ROOT / "outputs/experiments/uniform_followup_v1/mosei/test_epoch_sweeps/subset7_seed13/summary.json",
        "MOSI Subset-7": ROOT / "outputs/experiments/uniform_main_v1/mosi/test_sweep/subset7_seed13/summary.json",
    }
    for name, path in main.items():
        payload = json.loads(path.read_text())
        rows = payload["epochs"]
        epochs = [int(row["epoch"]) for row in rows]
        if epochs != list(range(1, 21)):
            raise ValueError(f"{name} coverage differs: {epochs}")
        checks.append({"curve": name, "epochs": len(epochs), "source": str(path)})
    tav = {
        "MOSEI Full KD": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/checkpoint_diagnostic_test_gpu1_20260919/M3_seed13/summary.json",
        "MOSEI Uniform": ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/checkpoint_diagnostic_test_gpu1_20260919/M4_seed13/summary.json",
    }
    for name, path in tav.items():
        payload = json.loads(path.read_text())
        epochs = sorted({
            int(row["epoch"]) for row in payload["checkpoints"]
            if row.get("checkpoint_role") == "completed_epoch"
        })
        if epochs != list(range(1, 21)):
            raise ValueError(f"{name} coverage differs: {epochs}")
        checks.append({"curve": name, "epochs": len(epochs), "source": str(path)})
    return checks


def main() -> None:
    try:
        wait_for_jobs()
        atomic_json({"schema": "fig3-full20-finalizer-v1", "status": "collecting",
                     "updated_at_unix": time.time()}, STATUS)
        collect_remote()
        checks = verify()
        subprocess.run([
            sys.executable, str(ROOT / "project/scripts/plot_fig3_test_mae_vs_epoch.py")
        ], cwd=ROOT, check=True)
        subprocess.run([
            sys.executable,
            str(ROOT / "project/scripts/plot_fig3_validation_mae_vs_epoch.py"),
        ], cwd=ROOT, check=True)
        atomic_json({
            "schema": "fig3-full20-finalizer-v1", "status": "complete",
            "coverage": checks,
            "main_figure": str(
                ROOT / "project/reports/paper_figures/fig3_validation_mae_vs_epoch.pdf"
            ),
            "supplementary_test_figure": str(
                ROOT / "project/reports/paper_figures/supp_test_mae_vs_epoch.pdf"
            ),
            "completed_at_unix": time.time(),
        }, STATUS)
    except Exception as error:
        atomic_json({"schema": "fig3-full20-finalizer-v1", "status": "failed",
                     "error": repr(error), "updated_at_unix": time.time()}, STATUS)
        raise


if __name__ == "__main__":
    main()
