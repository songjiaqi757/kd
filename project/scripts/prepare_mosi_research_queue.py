#!/usr/bin/env python3
"""Freeze the GPU0 MOSI matrix; reuse two fingerprint-checked ICD reference runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'project/scripts'))
from run_gpu_resource_queue import validate_plan

FIXED = ('uniform_interaction', 'uniform_interaction_shared_dropout',
         'uniform_interaction_gradient_projected', 'uniform_interaction_coordinate_compatible',
         'explicit_full_kd', 'explicit_uniform_interaction')
RESEARCH = ('order_routed', 'scrambled_routed', 'fusion_only', 'dgl_icd',
            'ensemble_source', 'target_uniform', 'target_variance', 'late_decay')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_reference(run, seed):
    config = json.loads((run/'run_config.json').read_text())
    expected = dict(method='uniform_interaction', decision_mode='none', seed=seed, epochs=20,
                    min_epochs=20, patience=20, batch_size=8, learning_rate=1e-4, weight_decay=.01,
                    retain_best_mae_and_acc2=True, save_every_epoch=False,
                    assets=str(ROOT/'outputs/experiments/main_table_v1/mosi/assets/protocol.json'))
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Baseline mismatch {run}: {key}')
    if json.loads((run/'status.json').read_text())['status'] != 'complete':
        raise ValueError(f'Incomplete reference: {run}')
    for name, digest in config['input_sha256'].items():
        if sha256(name) != digest:
            raise ValueError(f'Baseline source/asset changed: {name}')
    for name in ('best_valid_mae.pt', 'best_valid_acc2_nonzero.pt', 'last.pt'):
        if not (run/name).is_file():
            raise FileNotFoundError(run/name)
    return {'source_config_sha256': sha256(run/'run_config.json'),
            'verified_input_sha256': config['input_sha256'],
            'note': 'decision_mode=none adds a zero decision loss and read-only gradient diagnostics'}


def main():
    base = ROOT/'outputs/experiments/mosi_research_v1'
    directory = base/'resource_queue'
    if (directory/'plan.json').exists() or (directory/'registry.json').exists():
        raise FileExistsError('Frozen plan already exists; do not overwrite an active queue')
    directory.mkdir(parents=True, exist_ok=True)
    registry_path = directory/'registry.json'
    registry = {'dataset': 'mosi', 'device': 0, 'seeds': [13, 42, 2026], 'runs': [],
                'research_selection': 'Validation only; no automatic test for the eight research variants'}
    plan = {'dataset': 'mosi', 'device': 0, 'safety_mib': 6000, 'launch_settle_seconds': 30,
            'allow_official_test': True, 'external_reservations': [], 'jobs': [],
            'official_test_scope': 'Only the six previously fixed equal-modality methods',
            'checkpoint_retention': ['best_valid_mae.pt', 'best_valid_acc2_nonzero.pt', 'last.pt'],
            'max_parallel_train': 2, 'max_parallel_evaluate': 2}
    # Start the missing reference and main literature hypothesis early.
    order = [('uniform_interaction', 2026), ('order_routed', 13)]
    for seed in (13, 42, 2026):
        for method in FIXED + RESEARCH:
            if (method, seed) not in order:
                order.append((method, seed))
    for method, seed in order:
        name = f'{method}_seed{seed}'
        reused = method == 'uniform_interaction' and seed in (13, 42)
        run = (ROOT/f'outputs/experiments/decision_icd_v1/mosi/icd_decision_none_seed{seed}'
               if reused else base/name)
        row = {'method': method, 'seed': seed, 'run': str(run), 'reused': reused,
               'group': 'fixed_six' if method in FIXED else 'research',
               'official_test_planned': method in FIXED}
        if reused:
            row['reuse_audit'] = check_reference(run, seed)
        else:
            if run.exists():
                raise FileExistsError(run)
            if method in RESEARCH:
                entry = 'train_research_icd.py'
                extra = ['--research-variant', method]
                model_method = 'uniform_interaction'
            elif method == 'uniform_interaction':
                entry = 'train_decision_icd.py'
                extra = ['--decision-mode', 'none', '--gradient-diagnostic-every', '100']
                model_method = method
            else:
                entry, extra, model_method = 'train_main_table_kd.py', [], method
            command = [sys.executable, str(ROOT/'project/scripts'/entry), '--method', model_method,
                       '--assets', str(ROOT/'outputs/experiments/main_table_v1/mosi/assets/protocol.json'),
                       '--output', str(run), '--device', 'cuda:0', '--seed', str(seed),
                       '--epochs', '20', '--min-epochs', '20', '--patience', '20', '--batch-size', '8',
                       '--learning-rate', '0.0001', '--weight-decay', '0.01',
                       '--checkpoint-selection', 'valid_mae', '--retain-best-mae-and-acc2', *extra]
            plan['jobs'].append({'id': name, 'kind': 'train', 'run': str(run), 'memory_mib': 34000,
                'disk_mib': 2000, 'dependencies': [], 'completion_marker': str(run/'status.json'), 'command': command})
        registry['runs'].append(row)
        dependency = [] if reused else [name]
        plan['jobs'].append({'id': name+'_validation', 'kind': 'evaluate', 'run': str(run),
            'memory_mib': 8000, 'disk_mib': 50, 'dependencies': dependency,
            'completion_marker': str(run/'validation_postrun/status.json'),
            'command': [sys.executable, str(ROOT/'project/scripts/evaluate_research_validation.py'),
                        '--run', str(run), '--device', 'cuda:0', '--registry', str(registry_path)]})
        if method in FIXED:
            for policy in ('valid_mae', 'valid_acc2_nonzero'):
                output = run/'test'/policy
                plan['jobs'].append({'id': name+'_test_'+policy, 'kind': 'evaluate', 'run': str(run),
                    'memory_mib': 8000, 'disk_mib': 50, 'dependencies': dependency,
                    'completion_marker': str(output/'status.json'),
                    'command': [sys.executable, str(ROOT/'project/scripts/evaluate_main_table_test.py'),
                        '--run', str(run), '--output', str(output), '--test-manifest',
                        str(ROOT/'dataset/cmu_mosi/manifests/official_test_windowed.jsonl'),
                        '--device', 'cuda:0', '--batch-size', '8', '--checkpoint-selection', policy,
                        '--allow-official-test']})
    plan['jobs'].append({'id': 'final_summary', 'kind': 'evaluate', 'run': str(base),
        'memory_mib': 1, 'disk_mib': 1, 'dependencies': [job['id'] for job in plan['jobs']],
        'completion_marker': str(directory/'summary_complete.json'),
        'command': [sys.executable, str(ROOT/'project/scripts/summarize_mosi_research.py'),
                    '--registry', str(registry_path), '--final']})
    validate_plan(plan)
    sources = ['project/scripts/'+name for name in ('train_research_icd.py', 'train_decision_icd.py',
               'train_main_table_kd.py', 'evaluate_research_validation.py', 'evaluate_main_table_test.py',
               'run_gpu_resource_queue.py', 'summarize_mosi_research.py', 'prepare_mosi_research_queue.py')]
    sources += ['project/src/rdid_mosei/'+name for name in ('research_icd.py', 'main_table_kd.py',
                                                         'student.py', 'tav_distillation.py')]
    plan['source_sha256'] = {str(ROOT/rel): sha256(ROOT/rel) for rel in sources}
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2)+'\n')
    (directory/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'methods': len(FIXED+RESEARCH), 'runs': len(registry['runs']),
                      'reused': sum(row['reused'] for row in registry['runs']),
                      'new_training': sum(job['kind']=='train' for job in plan['jobs']),
                      'jobs': len(plan['jobs']), 'plan': str(directory/'plan.json')}))


if __name__ == '__main__':
    main()
