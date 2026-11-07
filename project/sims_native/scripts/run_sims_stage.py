#!/usr/bin/env python3
"""Run one resumable preparation stage and validate its artifacts before release."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np


def teacher_timeout_seconds(heartbeat, pid, started_at, now):
    """Allow cold model loading, but bound a single media/processor/forward operation."""
    if heartbeat is None or heartbeat.get('pid') != pid:
        return max(0, 1200 - (now - started_at))
    limit = 1200 if heartbeat['phase'] == 'loading_model' else 180
    return max(0, limit - (now - heartbeat['updated_at']))


def supervised_command(spec):
    if spec['kind'] != 'teacher':
        return subprocess.run(spec['command'], cwd=spec['cwd'])
    process = subprocess.Popen(spec['command'], cwd=spec['cwd'])
    started_at = time.time()
    heartbeat_path = Path(spec['run']) / 'progress.json'
    while process.poll() is None:
        time.sleep(5)
        heartbeat = None
        if heartbeat_path.exists():
            heartbeat = json.loads(heartbeat_path.read_text())
        if teacher_timeout_seconds(heartbeat, process.pid, started_at, time.time()) <= 0:
            print(json.dumps({'event': 'teacher_watchdog_timeout', 'pid': process.pid,
                              'heartbeat': heartbeat}, ensure_ascii=False), flush=True)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            return subprocess.CompletedProcess(spec['command'], 124)
    return subprocess.CompletedProcess(spec['command'], process.returncode)


def atomic(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def validate(spec):
    folder = Path(spec['run'])
    if spec['kind'] == 'teacher':
        config = json.loads((folder / 'run_config.json').read_text())
        complete = np.load(folder / 'completed.npy', mmap_mode='r')
        values = np.load(folder / 'features.npy', mmap_mode='r')
        index = [json.loads(line) for line in (folder / 'index.jsonl').read_text().splitlines()]
        assert complete.all() and complete.shape == (len(index),)
        assert config['subsets'] == ['t', 'a', 'v', 'ta', 'tv', 'av', 'tav']
        assert config['classification_classes'] == 5 and config['native_label_range'] == [-1.0, 1.0]
        assert len(index) == config['jobs'] and values.shape == (len(index), 2048)
        assert {row['split'] for row in index} == {'train', 'valid'}
        for start in range(0, len(index), 2048):
            assert np.isfinite(values[start:start + 2048]).all()
    elif spec['kind'] == 'probe':
        report = json.loads((folder / 'report.json').read_text())
        assert report['seed'] == spec['seed'] and report['official_test_evaluated'] is False
        assert report['train_parent_samples'] == spec['expected_train']
        assert report['valid_parent_samples'] == spec['expected_valid']
        assert 0 < report['calibration']['after']['temperature'] <= 20
        for line in (folder / 'predictions.jsonl').open():
            row = json.loads(line)
            assert len(row['classification_logits']) == 5 and -1.000001 <= row['probe_score'] <= 1.000001
    elif spec['kind'] == 'freeze':
        protocol = json.loads((folder / 'assets/protocol.json').read_text())
        assert protocol['dataset'] == spec['dataset']
        assert protocol['sentiment_scale'] == 1.0 and protocol['official_test_evaluated'] is False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    marker = Path(spec['completion_marker'])
    if marker.is_file() and json.loads(marker.read_text()).get('status') == 'complete':
        validate(spec)
        return
    atomic({'status': 'running', 'pid': os.getpid(), 'started_at': time.time()}, marker)
    try:
        result = None
        for attempt in range(1, (3 if spec['kind'] == 'teacher' else 1) + 1):
            result = supervised_command(spec)
            if result.returncode == 0:
                break
            if spec['kind'] == 'teacher' and attempt < 3:
                time.sleep(10)
        result.check_returncode()
        validate(spec)
        atomic({'status': 'complete', 'finished_at': time.time(), 'kind': spec['kind']}, marker)
    except BaseException as exc:
        atomic({'status': 'failed', 'error': repr(exc), 'finished_at': time.time()}, marker)
        raise


if __name__ == '__main__':
    main()
