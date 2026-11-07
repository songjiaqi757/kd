import argparse
import json
from pathlib import Path
import time
from teacher_test_common import atomic


def summarize(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    rows = []
    for dataset in plan['datasets']:
        directory = Path(plan['datasets'][dataset]['output'])
        assert json.loads((directory / 'status.json').read_text())['status'] == 'complete'
        report = json.loads((directory / 'report.json').read_text())
        assert report['probe_seed'] == 2026 and report['official_test_evaluated']
        metrics = report['metrics']
        native = dataset in ('sims', 'sims_v2')
        rows.append({'dataset': dataset, 'probe_seed': 2026, 'test_samples': report['test_samples'],
            'acc2': metrics['acc2'] if native else metrics['acc2_nonzero'],
            'f1': metrics['f1_weighted'] if native else metrics['f1_weighted_nonzero'],
            'mae': metrics['mae'], 'pearson': metrics['pearson'],
            'acc5': metrics.get('acc5'), 'acc7': metrics.get('acc7'),
            'acc2_policy': 'include_zero_nonpositive' if native else 'exclude_zero',
            'report': str(directory / 'report.json')})
    output = Path(plan['output']) / 'reports'; output.mkdir(parents=True, exist_ok=True)
    atomic(rows, output / 'results.json')
    lines = ['# 30B 教师正式测试结果', '',
        '冻结 Qwen3-Omni-30B-A3B-Instruct + 各数据集已训练的 seed2026 Probe；T+A+V。',
        'Probe 在训练/验证集上已冻结，本次只进行测试，不重新训练或用测试集选择参数。',
        'MOSI/MOSEI 的 Acc2/F1 排除零标签，MAE 标度为 [-3,3]；SIMS/SIMSv2 的 Acc2 包含零标签，MAE 标度为 [-1,1]。', '',
        '| Dataset | Test clips | Acc2 ↑ | F1 ↑ | MAE ↓ | Pearson ↑ | Acc7 ↑ | Acc5 ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        cells = [row['dataset'], str(row['test_samples'])] + [f'{row[k]:.4f}' if row[k] is not None else '—' for k in ('acc2', 'f1', 'mae', 'pearson', 'acc7', 'acc5')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    (output / 'results.md').write_text('\n'.join(lines) + '\n')
    atomic({'status': 'complete', 'test_datasets': len(rows), 'finished_at': time.time()}, Path(plan['output']) / 'summary_status.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True)
    summarize(parser.parse_args().plan)
