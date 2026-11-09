#!/usr/bin/env python3
"""Show live training, checkpoint, and test-sweep coverage for MOSI."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/uniform_main_v1/mosi"
METHODS = ("adapted_student", "full_kd", "subset7", "ensemble_full",
           "first_order_interaction", "uniform_interaction", "projector", "ea_kd", "cmad_cafd")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def main() -> None:
    print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"{'method':27} {'train':10} {'epoch':>5} {'ckpt':>5} {'test':>5} {'best':>5} {'MAE':>8}")
    print("-" * 72)
    for method in METHODS:
        run = BASE / "students" / f"{method}_seed13"
        sweep = BASE / "test_sweep" / f"{method}_seed13"
        status = read_json(run / "status.json")
        summary = read_json(sweep / "summary.json")
        checkpoints = list((run / "checkpoints").glob("epoch_*.pt"))
        evaluated = list(sweep.glob("epoch_*.json"))
        history = []
        try:
            history = json.loads((run / "history.json").read_text())
        except FileNotFoundError:
            pass
        epoch = history[-1]["epoch"] if history else status.get("epoch", "-")
        best = summary.get("selected_epoch", "-")
        mae = summary.get("selected_metrics", {}).get("mae")
        if mae is None and evaluated:
            tested = [read_json(path) for path in evaluated]
            tested = [row for row in tested if "test_metrics" in row]
            if tested:
                interim = min(tested, key=lambda row: (row["test_metrics"]["mae"], row["epoch"]))
                best, mae = f"{interim['epoch']}*", interim["test_metrics"]["mae"]
        mae_text = f"{mae:.4f}" if mae is not None else "-"
        print(f"{method:27} {status.get('status', '-'):10} {str(epoch):>5} "
              f"{len(checkpoints):>5} {len(evaluated):>5} {str(best):>5} {mae_text:>8}")
    print("* provisional Test MAE best among epochs evaluated so far")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True,
                                text=True, check=True, timeout=5)
        print("GPU free MiB, utilization %:")
        print(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    print(f"results: {BASE / 'summary'}")


if __name__ == "__main__":
    main()
