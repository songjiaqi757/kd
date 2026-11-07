import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from teacher_test_common import ROOT, GPU, atomic, apply_probe, sha256, verify_inputs


def supervised(command, features, environment):
    process = subprocess.Popen(command, cwd=ROOT, env=environment)
    started = time.time()
    while process.poll() is None:
        time.sleep(5)
        path = features / 'progress.json'
        heartbeat = json.loads(path.read_text()) if path.exists() else None
        if heartbeat is None or heartbeat.get('pid') != process.pid:
            remaining = 1200 - (time.time() - started)
        else:
            limit = 1200 if heartbeat['phase'] == 'loading_model' else 180
            remaining = limit - (time.time() - heartbeat['updated_at'])
        if remaining <= 0:
            print(json.dumps({'event': 'teacher_test_watchdog_timeout', 'pid': process.pid, 'heartbeat': heartbeat}), flush=True)
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            return 124
    return process.returncode


def run(spec):
    verify_inputs(spec)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    # This condition is independently checked even if invoked outside the queue.
    for path in spec['after_completion_markers']:
        marker = Path(path)
        assert marker.exists() and json.loads(marker.read_text()).get('status') == 'complete', 'Student campaign is not complete'
    output = Path(spec['output']);output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'complete':
        report = json.loads((output / 'report.json').read_text())
        assert report['input_sha256'] == spec['input_sha256'] and report['test_samples'] == spec['expected_test_parents']
        return
    atomic({'status': 'running', 'pid': os.getpid(), 'started_at': time.time(), 'phase': 'teacher_features'}, status_path)
    try:
        features = output / 'teacher_features'
        training_config = json.loads(Path(spec['training_feature_config']).read_text())
        command = [sys.executable, '-u', str(Path(__file__).with_name('extract_teacher_test_features.py')),
            '--manifest', spec['test_manifest'], '--output-dir', str(features), '--subsets', 'tav',
            '--label-space', spec['label_space'], '--device-map', 'cuda:0', '--model', spec['model'],
            '--prompt-version', training_config['prompt_version'], '--seed', '2026',
            '--video-timing-policy', training_config['video_timing_policy'], '--allow-official-test']
        if training_config.get('video_fps') is not None:
            command += ['--video-fps', str(training_config['video_fps'])]
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': GPU, 'OMP_NUM_THREADS': '4',
            'OPENBLAS_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        for attempt in range(1, 4):
            result = supervised(command, features, environment)
            if result == 0: break
            if attempt < 3: time.sleep(10)
        assert result == 0, f'Teacher test extraction failed after retries: {result}'
        atomic({'status': 'running', 'pid': os.getpid(), 'phase': 'frozen_probe_test'}, status_path)
        predictions, metrics, checkpoint = apply_probe(features, spec['probe_checkpoint'], spec['label_space'], 'test')
        assert len(predictions) == spec['expected_test_parents'] and metrics['count'] == len(predictions)
        assert {row['split'] for row in predictions} == {'test'}
        report = {'dataset': spec['dataset'], 'model': spec['model'], 'model_policy': 'frozen_30B_plus_pretrained_probe',
            'probe_seed': 2026, 'probe_best_epoch': checkpoint['best_epoch'],
            'checkpoint_selection': 'existing_probe_validation_loss_minimum',
            'probe_checkpoint': spec['probe_checkpoint'], 'probe_temperature': checkpoint['temperature'],
            'test_manifest': spec['test_manifest'], 'test_samples': len(predictions),
            'test_windows': spec['expected_test_windows'], 'subsets': ['tav'],
            'official_test_evaluated': True, 'test_labels_used_for_fitting': False,
            'native_label_range': [-1, 1] if spec['label_space'] == 'sims' else [-3, 3],
            'input_sha256': spec['input_sha256'], 'metrics': metrics, 'finished_at': time.time()}
        temporary = output / 'predictions.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in predictions))
        os.replace(temporary, output / 'predictions.jsonl')
        atomic(report, output / 'report.json')
        atomic({'status': 'complete', 'dataset': spec['dataset'], 'finished_at': time.time()}, status_path)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except BaseException as exc:
        atomic({'status': 'failed', 'error': repr(exc), 'finished_at': time.time()}, status_path)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args();run(json.loads(args.spec.read_text()))
