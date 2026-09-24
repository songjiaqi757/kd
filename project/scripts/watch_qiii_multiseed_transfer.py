#!/usr/bin/env python3
"""Wait for Qiii completion and local storage, then pull and summarize results."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "outputs/experiments/multiseed_controls_v1/mosei"
STATUS = ROOT / "outputs/experiments/multiseed_controls_v1/qiii_transfer_status.json"
SSH = [
    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
    "-p", "34989", "-i", "/home/wy/sjq/private_key_sjq.pem",
    "root@172.23.166.144",
]
REMOTE_STATUS = "/ai/sjq/kd/outputs/experiments/multiseed_controls_v1/mosei/queue_status.json"
REMOTE_FOLLOWUP_STATUS = "/ai/sjq/kd/outputs/experiments/multiseed_controls_v1/mosei/followup_order_controls/queue_status.json"
REMOTE_DIR = "/ai/sjq/kd/outputs/experiments/multiseed_controls_v1/mosei"
RESERVE_BYTES = 20 * 1024**3


def atomic_json(payload: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, STATUS)


def main() -> None:
    while True:
        writable = ROOT.is_dir() and os.access(ROOT, os.W_OK)
        free_bytes = shutil.disk_usage(ROOT).free if writable else 0
        remote_state = "unreachable"
        followup_state = "unreachable"
        remote_bytes = None
        try:
            result = subprocess.run(
                [*SSH, (
                    f"cat {REMOTE_STATUS}; printf '\\n---FOLLOWUP---\\n'; "
                    f"cat {REMOTE_FOLLOWUP_STATUS}; printf '\\n---SIZE---\\n'; "
                    f"du -sb {REMOTE_DIR} | cut -f1"
                )],
                check=True, capture_output=True, text=True,
            )
            main_text, remainder = result.stdout.split("\n---FOLLOWUP---\n", 1)
            followup_text, size_text = remainder.split("\n---SIZE---\n", 1)
            remote_state = json.loads(main_text).get("status", "unknown")
            followup_state = json.loads(followup_text).get("status", "unknown")
            remote_bytes = int(size_text.strip())
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError, ValueError):
            pass
        required_bytes = None if remote_bytes is None else remote_bytes + RESERVE_BYTES
        storage_ready = writable and required_bytes is not None and free_bytes >= required_bytes
        atomic_json({
            "status": "waiting",
            "remote_queue_status": remote_state,
            "remote_followup_queue_status": followup_state,
            "remote_result_gib": None if remote_bytes is None else remote_bytes / 1024**3,
            "local_free_gib": free_bytes / 1024**3,
            "required_local_free_gib": None if required_bytes is None else required_bytes / 1024**3,
            "local_storage_ready": storage_ready,
            "local_destination": str(LOCAL),
            "updated_at_unix": time.time(),
        })
        if storage_ready and remote_state == "complete" and followup_state == "complete":
            break
        time.sleep(60)
    subprocess.run([
        sys.executable, str(ROOT / "project/scripts/transfer_multiseed_results.py")
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, str(ROOT / "project/scripts/summarize_multiseed_controls.py"),
        "--base", str(LOCAL),
    ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([
        sys.executable, str(ROOT / "project/scripts/summarize_multiseed_controls.py"),
        "--base", str(LOCAL / "followup_order_controls"),
    ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    atomic_json({
        "status": "complete", "local_destination": str(LOCAL),
        "summary": str(LOCAL / "summary.json"),
        "followup_summary": str(LOCAL / "followup_order_controls/summary.json"),
        "completed_at_unix": time.time(),
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json({
            "status": "failed", "error": repr(error),
            "updated_at_unix": time.time(),
        })
        raise
