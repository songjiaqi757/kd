#!/usr/bin/env python3
"""Keep at most two live interaction evaluators active on GPU1."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/interaction_evidence_v1"
STATUS = BASE / "live_evaluation_queue_status.json"
MAX_ACTIVE = 8

BASE_UNITS = (
    "rdid-liveeval-mosei-first-second.service",
    "rdid-liveeval-mosei-random.service",
)
QUEUED = (
    (
        "rdid-liveeval-mosi-first-second.service",
        BASE / "mosi/test_epoch_sweeps/first_second_order_interaction_seed13/summary.json",
    ),
    (
        "rdid-liveeval-mosi-random.service",
        BASE / "mosi/test_epoch_sweeps/random_orthogonal_seed13/summary.json",
    ),
)
SHARD_UNITS = (
    "rdid-liveeval-mosei-first-second-worker1.service",
    "rdid-liveeval-mosei-random-worker1.service",
    "rdid-liveeval-mosei-first-second-worker2.service",
    "rdid-liveeval-mosei-random-worker2.service",
    "rdid-liveeval-mosei-first-second-worker3.service",
    "rdid-liveeval-mosei-random-worker3.service",
    "rdid-liveeval-mosi-first-second-worker1.service",
    "rdid-liveeval-mosi-random-worker1.service",
)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def is_active(unit: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit], check=False
    )
    return result.returncode == 0


def is_complete(summary: Path) -> bool:
    if not summary.exists():
        return False
    return bool(json.loads(summary.read_text()).get("complete"))


def main() -> None:
    units = BASE_UNITS + tuple(unit for unit, _ in QUEUED) + SHARD_UNITS
    while True:
        completed = {unit: is_complete(summary) for unit, summary in QUEUED}
        active = [unit for unit in units if is_active(unit)]
        for unit, _ in QUEUED:
            if len(active) >= MAX_ACTIVE:
                break
            if completed[unit] or unit in active:
                continue
            subprocess.run(["systemctl", "--user", "start", unit], check=True)
            active.append(unit)
        payload = {
            "schema": "interaction-live-evaluation-queue-v1",
            "max_active_evaluators": MAX_ACTIVE,
            "active_units": active,
            "queued_units": [
                unit for unit, _ in QUEUED
                if unit not in active and not completed[unit]
            ],
            "completed": completed,
            "updated_at_unix": time.time(),
        }
        atomic_json(payload, STATUS)
        if all(completed.values()):
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
