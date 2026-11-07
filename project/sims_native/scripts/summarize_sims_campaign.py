#!/usr/bin/env python3
"""Report both selection policies, single-seed main tables and 3-seed ablations."""
import argparse
import csv
import json
from pathlib import Path
import statistics

METRICS = ('acc2', 'f1_weighted', 'mae', 'pearson', 'acc3', 'acc5', 'acc2_nonzero', 'f1_weighted_nonzero')
LABELS = {'adapted_student': 'Adapted Student', 'full_kd': 'Full KD', 'projector': 'Projector Feature KD',
          'ea_kd': 'EA-KD', 'cmad_cafd': 'CMAD-style CAFD', 'uniform_interaction': 'ICD (Uniform)',
          'subset7': 'Subset-7 KD', 'first_order_interaction': 'First-order',
          'first_second_order_interaction': 'First+Second-order', 'random_orthogonal': 'Random Orthogonal'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    results, missing = [], []
    for run in registry['runs']:
        trained = json.loads((Path(run['run']) / 'report.json').read_text())
        assert trained['epochs_run'] == 20, 'Every student must complete exactly 20 epochs'
        for policy, folder in run['tests'].items():
            folder = Path(folder)
            if not (folder / 'report.json').exists():
                missing.append(str(folder))
                continue
            report = json.loads((folder / 'report.json').read_text())
            assert report['checkpoint_selection'] == policy and report['official_test_evaluated']
            results.append({**{k: run[k] for k in ('dataset', 'method', 'seed', 'main_table', 'ablation_table')},
                            'checkpoint_selection': policy, 'selected_epoch': report['selected_epoch'],
                            **{k: report['metrics'][k] for k in METRICS}, 'report': str(folder / 'report.json')})
    assert not missing and len(results) == 88, 'Missing official-test reports: ' + repr(missing)
    destination = Path(registry['campaign']) / 'reports'
    destination.mkdir(exist_ok=True)
    with (destination / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    for policy in ('valid_mae', 'valid_acc2'):
        for table in ('main', 'ablation'):
            lines = [f'# SIMS / SIMS v2: {table}; {policy}', '',
                     'Native [-1,1] scores; no rescaling. Official Acc2 includes zeros in the non-positive class.',
                     'Each checkpoint selected on validation only; then evaluated once on official test.', '',
                     '| Dataset | Method | Acc2 | F1 | MAE | Pearson | Acc3 | Acc5 |',
                     '|---|---|---:|---:|---:|---:|---:|---:|']
            for dataset in ('sims', 'sims_v2'):
                for method in registry[table + '_methods']:
                    rows = [r for r in results if r['dataset'] == dataset and r['method'] == method
                            and r['checkpoint_selection'] == policy and r[table + '_table']]
                    assert len(rows) == (1 if table == 'main' else 3)
                    cells = []
                    for metric in METRICS[:6]:
                        values = [r[metric] for r in rows]
                        cells.append(f'{values[0]:.4f}' if table == 'main' else
                                     f'{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}')
                    lines.append('| ' + ' | '.join([dataset, LABELS[method], *cells]) + ' |')
            (destination / f'{table}_{policy}.md').write_text('\n'.join(lines) + '\n')
    (destination / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'training_runs': 44, 'test_evaluations': 88, 'reports': str(destination)}))


if __name__ == '__main__':
    main()
