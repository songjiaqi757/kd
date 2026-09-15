#!/usr/bin/env python3
"""Read queue and epoch progress without initializing CUDA or loading models."""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/tav_main_v1"


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def main():
    queue = read(BASE / "status.json")
    print(f"TAV P0 | {queue.get('status','not_started')} | {queue.get('stage','—')} | phase={queue.get('phase','—')}")
    completed = 0
    for method in ("M0", "M1", "M3", "M4", "M6"):
        for seed in (13,42,2026):
            name=f"{method}_seed{seed}"
            directory=BASE/'students'/name
            status=read(directory/'status.json')
            history=read(directory/'history.json') or []
            running=queue.get('running',{}).get(name,{})
            if not status and not running:
                continue
            completed += status.get('status')=='complete'
            best=min((r['valid_metrics']['mae'] for r in history),default=None)
            best_text='—' if best is None else f'{best:.5f}'
            print(f"{name:13} GPU={str(running.get('gpu','—')):1} {status.get('status','initializing'):12} "
                  f"epoch={status.get('epoch',len(history))} step={status.get('step','—')}/{status.get('steps','—')} "
                  f"best_valid_MAE={best_text}")
    print(f"Completed: {completed}/15 | 各模型 seed13 技术核查后即可补 42/2026 | official test 未使用")
    if queue.get('error'):
        print(f"ERROR: {queue['error']}")
    print(f"Logs: {BASE/'logs'}")


if __name__=='__main__':
    main()
