#!/usr/bin/env python3
"""Retire the paused fixed scheduler after its two in-flight tests finish."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiments/tav_main_v1/official_test_20260917"
WATCH = (OUTPUT / "M0_seed2026/status.json", OUTPUT / "M1_seed13/status.json")
STATUS = OUTPUT / "fixed_scheduler_retirement.json"
UNIT = "rdid-tav-official-test.service"


def state(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("status")


def write(payload):
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)


def main():
    while True:
        states = {path.parent.name: state(path) for path in WATCH}
        write({"status": "waiting", "watched_jobs": states, "retire_unit": UNIT})
        if all(value == "complete" for value in states.values()):
            break
        if any(value == "failed" for value in states.values()):
            raise RuntimeError(f"in-flight fixed-scheduler test failed: {states}")
        time.sleep(15)
    subprocess.run(["systemctl", "--user", "stop", UNIT], check=True)
    write({"status": "complete", "watched_jobs": states, "retired_unit": UNIT})


if __name__ == "__main__":
    main()
