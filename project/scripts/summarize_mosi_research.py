#!/usr/bin/env python3
"""Summarize a frozen registry without mixing validation and test or selection policies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics


def summarize(registry_path, final=False):
    registry = json.loads(Path(registry_path).read_text())
    records, aggregates, missing = [], [], []
    for row in registry['runs']:
        run = Path(row['run'])
        splits = ['validation'] + (['test'] if row['official_test_planned'] else [])
        for split in splits:
            for policy in ('valid_mae', 'valid_acc2_nonzero'):
                path = run/('validation_postrun' if split == 'validation' else 'test')/policy/'report.json'
                if not path.is_file():
                    missing.append(dict(method=row['method'], seed=row['seed'], split=split, policy=policy))
                    continue
                data = json.loads(path.read_text())
                records.append(dict(method=row['method'], seed=row['seed'], split=split, policy=policy,
                                    epoch=data['selected_epoch'], metrics=data['metrics'], source=str(path)))
    for method in dict.fromkeys(row['method'] for row in registry['runs']):
        for split in ('validation', 'test'):
            for policy in ('valid_mae', 'valid_acc2_nonzero'):
                rows = [r for r in records if (r['method'], r['split'], r['policy']) == (method, split, policy)]
                if not rows:
                    continue
                metrics = {key: {'mean': statistics.mean(r['metrics'][key] for r in rows),
                    'sample_sd': statistics.stdev(r['metrics'][key] for r in rows) if len(rows)>1 else None}
                    for key in ('mae', 'acc2_nonzero', 'pearson', 'f1_weighted_nonzero')}
                aggregates.append(dict(method=method, split=split, policy=policy, seeds=[r['seed'] for r in rows], metrics=metrics))
    directory = Path(registry_path).parent
    payload = {'records': records, 'aggregates': aggregates, 'missing': missing,
               'note': 'Partial rows are not three-seed conclusions. Validation is development evidence; test does not select research variants.'}
    temporary = directory/f'summary.{os.getpid()}.json.tmp'
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(directory/'summary.json')
    lines = ['# MOSI GPU0 实验汇总', '', '验证与测试分开，两个选模策略分开。不足3个seed的行仅为进度。', '',
             '| Method | Split | Selection | Seeds | MAE | Acc2 NZ (%) | Pearson | F1 NZ (%) |',
             '|---|---|---|---|---:|---:|---:|---:|']
    for row in aggregates:
        m = row['metrics']
        lines.append(f"| {row['method']} | {row['split']} | {row['policy']} | {row['seeds']} | "
                     f"{m['mae']['mean']:.6f} | {100*m['acc2_nonzero']['mean']:.3f} | "
                     f"{m['pearson']['mean']:.6f} | {100*m['f1_weighted_nonzero']['mean']:.3f} |")
    lines += ['', f'Pending report entries: {len(missing)}. Sample standard deviations and individual runs are in summary.json.']
    temporary = directory/f'summary.{os.getpid()}.md.tmp'
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(directory/'summary.md')
    if final:
        if missing:
            raise ValueError(f'Final summary has {len(missing)} missing reports')
        (directory/'summary_complete.json').write_text('{"status": "complete"}\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    result = summarize(args.registry, args.final)
    print(json.dumps({'reports': len(result['records']), 'pending': len(result['missing'])}))
