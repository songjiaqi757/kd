#!/usr/bin/env python3
"""Wait for the additional MOSI controls and build the seven-method summary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/multiseed_controls_v1/mosi_additional_controls"
STATUS = BASE / "seven_method_summary_status.json"


def atomic_json(payload: object) -> None:
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, STATUS)


def main() -> None:
    queue = BASE / "queue_status.json"
    while True:
        state = json.loads(queue.read_text()).get("status") if queue.is_file() else "missing"
        atomic_json({"status": "waiting", "queue_status": state, "updated_at_unix": time.time()})
        if state == "complete":
            break
        if state == "failed":
            raise RuntimeError("additional MOSI queue failed")
        time.sleep(60)
    subprocess.run([
        sys.executable, str(ROOT / "project/scripts/summarize_multiseed_controls.py"),
        "--base", str(BASE),
    ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([
        sys.executable, str(ROOT / "project/scripts/summarize_mosi_seven_controls.py"),
    ], cwd=ROOT, check=True)
    atomic_json({
        "status": "complete",
        "summary": str(ROOT / "outputs/experiments/multiseed_controls_v1/mosi_seven_controls_valid_mae/summary.json"),
        "completed_at_unix": time.time(),
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json({"status": "failed", "error": repr(error), "updated_at_unix": time.time()})
        raise
