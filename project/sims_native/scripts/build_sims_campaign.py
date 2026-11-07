#!/usr/bin/env python3
"""Lock both native SIMS experiment matrices and the physical-GPU1 DAG."""
import hashlib
import json
from pathlib import Path
import sys
from run_sims_gpu_queue import validate_plan
from sims_native_common import METRIC_PROTOCOL

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'project/sims_native/scripts'
CAMPAIGN = ROOT / 'outputs/experiments/sims_native_v1'
GPU_UUID = 'GPU-ac40f187-8891-00f2-3999-959bbdcf952f'
MAIN = ('adapted_student', 'full_kd', 'projector', 'ea_kd', 'cmad_cafd', 'uniform_interaction')
ABLATION = ('full_kd', 'subset7', 'first_order_interaction', 'first_second_order_interaction',
            'random_orthogonal', 'uniform_interaction')
SEEDS = (13, 42, 2026)
POLICIES = ('valid_mae', 'valid_acc2')
COUNTS = {'sims': (1368, 456, 457), 'sims_v2': (2722, 647, 1034)}


def atomic(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    queue = CAMPAIGN / 'queue'
    if (queue / 'plan.json').exists():
        raise FileExistsError('This campaign is already frozen; do not overwrite its plan')
    plan = {'dataset': 'sims_and_sims_v2', 'device': 1, 'safety_mib': 6000,
            'launch_settle_seconds': 10, 'allow_official_test': True,
            'external_reservations': [], 'jobs': [], 'environment': {
                'CUDA_VISIBLE_DEVICES': GPU_UUID, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
                'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1',
                'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'PYTHONUNBUFFERED': '1',
                'FORCE_QWENVL_VIDEO_READER': 'decord'},
            'native_metric_protocol': METRIC_PROTOCOL,
            'epochs': 20, 'early_stopping': False,
            'checkpoint_retention': ['best_valid_mae.pt', 'best_valid_acc2.pt', 'last.pt'],
            'same_result_reuse': 'Full KD/Uniform seed13 shared between main and ablation tables',
            'memory_policy': 'reserve initial peaks, then measured completed-run peaks by dataset and forward family',
            'official_test_policy': 'each validation-selected checkpoint evaluated separately after its 20-epoch training'}
    registry = {'campaign': str(CAMPAIGN), 'metric_protocol': METRIC_PROTOCOL,
                'main_methods': MAIN, 'ablation_methods': ABLATION, 'seeds': SEEDS, 'runs': []}

    def stage(dataset, name, kind, command, dependencies, run, memory, priority, **metadata):
        identifier = dataset + '_' + name
        marker = CAMPAIGN / dataset / '.stages' / (name + '.json')
        spec = {'dataset': dataset, 'kind': kind, 'run': str(run), 'command': command,
                'completion_marker': str(marker), 'cwd': str(ROOT), **metadata}
        spec_path = queue / 'specs' / (identifier + '.json')
        atomic(spec, spec_path)
        plan['jobs'].append({'id': identifier, 'kind': kind, 'run': str(run),
                             'memory_mib': memory, 'disk_mib': 2000, 'priority': priority,
                             'dependencies': dependencies, 'completion_marker': str(marker),
                             'command': [sys.executable, '-u', str(SCRIPTS / 'run_sims_stage.py'), '--spec', str(spec_path)]})
        return identifier

    for dataset in COUNTS:
        data = ROOT / 'dataset' / ('ch_sims' if dataset == 'sims' else 'ch_sims_v2')
        stage(dataset, 'prepare', 'prepare',
              [sys.executable, '-u', str(SCRIPTS / 'prepare_sims_dataset.py'), '--dataset', dataset, '--workers', '6'],
              [], data, 1, 0)
    for dataset, counts in COUNTS.items():
        base = CAMPAIGN / dataset
        data = ROOT / 'dataset' / ('ch_sims' if dataset == 'sims' else 'ch_sims_v2')
        manifest = data / 'manifests/official_train_valid_windowed.jsonl'
        teacher = stage(dataset, 'teacher', 'teacher',
                        [sys.executable, '-u', str(SCRIPTS / 'extract_sims_teacher_features.py'),
                         '--manifest', str(manifest), '--output-dir', str(base / 'teacher_features'),
                         '--device-map', 'cuda:0', '--video-timing-policy', 'sampled_fps_v2'],
                        [dataset + '_prepare'], base / 'teacher_features', 75000, 5)
        probes = []
        for seed in (2026, 2027, 2028):
            probes.append(stage(dataset, f'probe{seed}', 'probe',
                                [sys.executable, '-u', str(SCRIPTS / 'train_teacher_probe_v2.py'),
                                 '--features', str(base / 'teacher_features'), '--output', str(base / 'probes' / f'seed{seed}'),
                                 '--seed', str(seed), '--device', 'cuda:0'],
                                [teacher], base / 'probes' / f'seed{seed}', 1000, 1,
                                seed=seed, expected_train=counts[0], expected_valid=counts[1]))
        stage(dataset, 'freeze', 'freeze',
              [sys.executable, '-u', str(SCRIPTS / 'freeze_sims_assets.py'), '--dataset', dataset, '--base', str(base)],
              probes, base, 1, 1)

    order = [(m, 13) for m in MAIN]
    for seed in SEEDS:
        order.extend((m, seed) for m in ABLATION if (m, seed) not in order)
    assert len(order) == 22
    for method, seed in order:
        for dataset in COUNTS:
            base = CAMPAIGN / dataset
            name = f'{method}_seed{seed}'
            identifier = dataset + '_' + name
            run = base / 'students' / name
            training = [sys.executable, '-u', str(SCRIPTS / 'train_main_table_kd.py'),
                        '--method', method, '--assets', str(base / 'assets/protocol.json'), '--output', str(run),
                        '--device', 'cuda:0', '--seed', str(seed), '--epochs', '20', '--min-epochs', '20',
                        '--patience', '20', '--batch-size', '8', '--learning-rate', '0.0001',
                        '--weight-decay', '0.01', '--coordinate-seed', '20260922',
                        '--checkpoint-selection', 'valid_mae', '--retain-best-mae-and-acc2']
            forward = 'seven_subset' if method in ABLATION and method != 'full_kd' else 'single_subset'
            plan['jobs'].append({'id': identifier, 'kind': 'train', 'run': str(run),
                                 'resource_family': dataset + '_' + forward,
                                 'memory_mib': 34000, 'initial_memory_mib': 34000,
                                 'disk_mib': 2000, 'priority': 10, 'dependencies': [dataset + '_freeze'],
                                 'completion_marker': str(run / 'status.json'), 'command': training})
            entry = {'dataset': dataset, 'method': method, 'seed': seed, 'run': str(run),
                     'main_table': seed == 13 and method in MAIN, 'ablation_table': method in ABLATION, 'tests': {}}
            for policy in POLICIES:
                test = base / 'official_test' / name / policy
                entry['tests'][policy] = str(test)
                manifest = ROOT / 'dataset' / ('ch_sims' if dataset == 'sims' else 'ch_sims_v2') / 'manifests/official_test_windowed.jsonl'
                plan['jobs'].append({'id': identifier + '_test_' + policy, 'kind': 'evaluate',
                                     'run': str(run), 'memory_mib': 8000, 'disk_mib': 20, 'priority': 0,
                                     'dependencies': [identifier], 'completion_marker': str(test / 'status.json'),
                                     'command': [sys.executable, '-u', str(SCRIPTS / 'evaluate_main_table_test.py'),
                                                 '--run', str(run), '--output', str(test), '--test-manifest', str(manifest),
                                                 '--device', 'cuda:0', '--batch-size', '8', '--checkpoint-selection', policy,
                                                 '--allow-official-test']})
            registry['runs'].append(entry)
    stage('sims', 'summary', 'summary',
          [sys.executable, str(SCRIPTS / 'summarize_sims_campaign.py'), '--registry', str(queue / 'registry.json')],
          [j['id'] for j in plan['jobs']], CAMPAIGN, 1, 0)
    assert len(registry['runs']) == 44
    validate_plan(plan)
    native = ROOT / 'project/sims_native'
    sources = sorted(native.rglob('*.py'))
    plan['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    atomic(registry, queue / 'registry.json')
    atomic(plan, queue / 'plan.json')
    print(json.dumps({'plan': str(queue / 'plan.json'), 'training_runs': 44, 'test_evaluations': 88,
                      'jobs': len(plan['jobs']), 'gpu': 1, 'native_label_scale': 1.0}))


if __name__ == '__main__':
    main()
