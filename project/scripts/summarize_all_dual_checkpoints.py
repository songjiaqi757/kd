#!/usr/bin/env python3
"""Read-only reconstruction of validation-selected historical experiment results.

Never loads checkpoints or launches inference. Test records must match a training
run AND the validation-selected epoch. Missing metrics remain missing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

METRICS = ('mae', 'acc2_nonzero', 'pearson', 'f1_weighted_nonzero',
           'acc2_has_zero', 'f1_weighted_has_zero', 'acc7')
POLICIES = ('valid_mae', 'valid_acc2_nonzero')


def metrics(row):
    if isinstance(row.get('valid_metrics'), dict):
        return dict(row['valid_metrics'])
    return {k: row['valid_' + k] for k in METRICS if 'valid_' + k in row}


def select(history, policy):
    key = 'mae' if policy == 'valid_mae' else 'acc2_nonzero'
    sign = 1 if policy == 'valid_mae' else -1
    valid = [r for r in history if isinstance(metrics(r).get(key), (int, float))
             and math.isfinite(metrics(r)[key])]
    return min(valid, key=lambda r: (sign * metrics(r)[key], int(r['epoch']))) if valid else None


def consistent(a, b, tolerance=1e-7):
    keys = set(a) & set(b) & set(METRICS)
    return bool(keys) and all(abs(a[k] - b[k]) <= tolerance for k in keys)


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def choose_test(choices):
    """Prefer explicitly selected evaluations over older diagnostic sweeps.

    Never rank by model performance. Disagreement within the preferred provenance
    class remains unresolved; disagreement with lower-priority sweeps is audited.
    """
    if not choices:
        return None, False, False
    preferred = [x for x in choices if not x['exploratory']] or choices
    if all(x['exploratory'] for x in preferred):
        # Fixed epoch filenames are stronger provenance than mutable best aliases.
        explicit = [x for x in preferred if re.search(r'/epoch_\d+(?:/report)?\.json$', x['source'])]
        preferred = explicit or preferred
    chosen = sorted(preferred, key=lambda x: x['source'])[0]
    unresolved = any(not consistent(chosen['metrics'], x['metrics'], 1e-6) for x in preferred)
    variation = any(not consistent(chosen['metrics'], x['metrics'], 1e-6) for x in choices)
    return None if unresolved else chosen, variation, unresolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache, parse_errors = {}, []

    def read(path):
        if path not in cache:
            try:
                cache[path] = json.loads(path.read_text()) if path.is_file() else {}
            except (ValueError, OSError) as exc:
                parse_errors.append({'path': str(path), 'error': str(exc)})
                cache[path] = {}
        return cache[path]

    def relocate(value):
        value = str(value)
        if '/outputs/' in value:
            return root / 'outputs' / value.split('/outputs/', 1)[1]
        prefix = '/var/tmp/kd_experiment_storage/'
        if value.startswith(prefix):
            return root / 'outputs/experiments' / value[len(prefix):]
        return Path(value) if value.startswith('/') else root / value

    runs, excluded, hashes, by_name = {}, [], defaultdict(set), defaultdict(set)
    for hp in sorted((root / 'outputs').rglob('history.json')):
        history = read(hp)
        if not isinstance(history, list) or not history or not metrics(history[0]):
            excluded.append({'path': str(hp), 'reason': 'no per-epoch sentiment metrics (e.g. teacher probe)'})
            continue
        run = hp.parent
        config, report, status = (read(run / f) for f in ('run_config.json', 'report.json', 'status.json'))
        seed_match = re.search(r'(?:^|_)seed(\d+)(?:_|$)', run.name)
        seed = config.get('seed', report.get('seed', int(seed_match[1]) if seed_match else None))
        method = re.sub(r'_seed\d+(?=_|$)', '', run.name)
        relative = str(run.relative_to(root))
        protocol = config or report.get('protocol', {})
        count = metrics(history[-1]).get('count', report.get('valid_metrics', {}).get('count'))
        dataset = 'mosi' if 'mosi' in str(protocol.get('manifest', '')) or '/mosi/' in relative else 'mosei'
        if count == 229:
            dataset = 'mosi'
        scope = ('smoke/preflight' if any(t in relative for t in ('smoke', 'preflight')) or (count is not None and count < 20)
                 else 'pilot' if count is not None and count < 200 else 'full')
        state = status.get('status', 'complete' if report else 'partial_no_report')
        campaign = str(run.parent.relative_to(root / 'outputs')).removesuffix('/students')
        rid = relative
        runs[rid] = dict(run=rid, path=run, method=method, seed=seed, dataset=dataset,
                         scope=scope, state=state, campaign=campaign, history=history,
                         report=report, config=config, valid_count=count,
                         epochs=len(history), epoch_index={int(r['epoch']): r for r in history})
        by_name[run.name].add(rid)
        cp = run / 'run_config.json'
        if cp.exists():
            hashes[hashlib.sha256(cp.read_bytes()).hexdigest()].add(rid)

    def run_from_path(value):
        p = relocate(value)
        for candidate in [p, *p.parents]:
            try:
                rid = str(candidate.relative_to(root))
            except ValueError:
                continue
            if rid in runs:
                return rid
        return None

    tests, unmatched, conflicts = defaultdict(list), [], []
    test_files = sorted(set((root / 'outputs').rglob('report.json')) |
                        set((root / 'outputs').rglob('epoch_*.json')))
    for p in test_files:
        data = read(p)
        if not isinstance(data, dict):
            continue
        tm = data.get('test_metrics')
        if tm is None and data.get('official_test_evaluated') is True:
            tm = data.get('metrics')
        if not isinstance(tm, dict) or 'mae' not in tm:
            continue
        strong = set()
        metadata = [data]
        for parent in [p.parent, *list(p.parents)[1:5]]:
            for name in ('run_config.json', 'evaluation_plan.json'):
                meta = read(parent / name)
                if isinstance(meta, dict):
                    metadata.append(meta)
        for meta in metadata:
            for key in ('source_run', 'checkpoint'):
                if meta.get(key):
                    match = run_from_path(meta[key])
                    if match:
                        strong.add(match)
            for key in ('source_config_sha256', 'source_run_config_sha256'):
                strong.update(hashes.get(meta.get(key), set()))
        candidates = strong.copy()
        if not candidates:
            for part in p.parts:
                candidates.update(by_name.get(part, set()))
            if not candidates and data.get('method') and data.get('seed') is not None:
                candidates.update(by_name.get(f"{data['method']}_seed{data['seed']}", set()))
        ep = data.get('selected_epoch', data.get('epoch', data.get('best_epoch')))
        vm = data.get('selected_valid_metrics', data.get('valid_metrics', {}))
        qualified = []
        for rid in candidates:
            r = runs[rid]
            epoch = ep
            if epoch is None and data.get('checkpoint_selection') == 'valid_mae':
                epoch = r['report'].get('best_epoch')
            if epoch is None or int(epoch) not in r['epoch_index']:
                continue
            if tm.get('count') != (686 if r['dataset'] == 'mosi' else 4659):
                continue
            if vm and not consistent(vm, metrics(r['epoch_index'][int(epoch)])):
                continue
            qualified.append((rid, int(epoch)))
        if len(qualified) > 1:
            same_family = [(rid, ep) for rid, ep in qualified
                           if runs[rid]['campaign'].split('/')[:2] == list(p.relative_to(root/'outputs').parts[:2])]
            if len(same_family) == 1:
                qualified = same_family
        if len(qualified) != 1:
            unmatched.append({'source': str(p), 'candidates': qualified, 'reason': 'ambiguous or missing training identity/epoch'})
            continue
        rid, epoch = qualified[0]
        exploratory = any(x in str(p) for x in ('sweep', 'diagnostic')) or 'exploratory' in str(data.get('test_use_policy', ''))
        tests[rid, epoch].append(dict(metrics=tm, source=str(p), exploratory=exploratory,
                                     identity='source/config hash' if strong else 'run name + epoch + validation/coverage'))

    selected = []
    for rid, run in runs.items():
        for policy in POLICIES:
            row = select(run['history'], policy)
            if row is None:
                continue
            epoch, vm = int(row['epoch']), metrics(row)
            report = run['report']
            if report.get('best_epoch') == epoch and consistent(report.get('valid_metrics', {}), vm):
                vm = {**report['valid_metrics'], **vm}
            choices = tests.get((rid, epoch), [])
            chosen, variation, unresolved = choose_test(choices)
            if variation:
                conflicts.append({'run': rid, 'epoch': epoch, 'sources': choices,
                                  'resolution': 'unresolved' if unresolved else 'prefer selected evaluation, then explicit epoch over mutable best alias'})
            cp = run['path'] / f'best_{policy}.pt'
            retained = report.get('retained_best_checkpoints', {}).get(policy, {})
            # Presence does not imply its epoch matches the reconstructed rule.
            available = cp.exists() and retained.get('epoch') == epoch
            cp_paths = [str(cp)] if available else []
            for candidate in (run['path']/f'checkpoints/epoch_{epoch:03d}.pt', run['path']/f'checkpoints/epoch_{epoch:03d}.pt.zst'):
                if candidate.exists():
                    cp_paths.append(str(candidate))
            if report.get('best_epoch') == epoch and (run['path']/'best.pt').exists():
                cp_paths.append(str(run['path']/'best.pt'))
            if epoch == max(run['epoch_index']) and (run['path']/'last.pt').exists():
                cp_paths.append(str(run['path']/'last.pt'))
            selected.append({k: run[k] for k in ('run', 'method', 'seed', 'dataset', 'scope', 'state', 'campaign', 'epochs')} |
                            dict(policy=policy, epoch=epoch, valid_metrics=vm,
                                 test_metrics=chosen['metrics'] if chosen else {},
                                 test_source=chosen['source'] if chosen else None,
                                 test_exploratory_origin=chosen['exploratory'] if chosen else None,
                                 test_source_count=len(choices), checkpoint_paths=cp_paths,
                                 test_metric_variation=variation,
                                 test_status='available' if chosen else 'conflicting_records' if choices else 'no_matching_epoch_test',
                                 validation_source=str(run['path']/'history.json'),
                                 retained_epoch=retained.get('epoch')))

    groups = []

    def aggregate(view, rows):
        grouped = defaultdict(list)
        for r in rows:
            if r['state'] == 'complete':
                grouped[r['dataset'], r['scope'], r['campaign'] if view == 'all_campaigns' else view, r['method'], r['policy']].append(r)
        for key, batch in sorted(grouped.items()):
            if len({r['seed'] for r in batch}) != len(batch):
                raise ValueError(f'duplicate seeds in aggregation: {key}')
            for split, mk in (('validation', 'valid_metrics'), ('test', 'test_metrics')):
                available = [r for r in batch if r[mk]]
                metric_stats = {}
                for metric in METRICS:
                    usable = [r for r in available if isinstance(r[mk].get(metric), (int, float)) and math.isfinite(r[mk][metric])]
                    values = [r[mk][metric] for r in usable]
                    metric_stats[metric] = dict(mean=statistics.mean(values) if values else None,
                                               sd=statistics.stdev(values) if len(values)>1 else None,
                                               n=len(values), seeds=[r['seed'] for r in usable])
                groups.append(dict(view=view, dataset=key[0], scope=key[1], campaign=key[2], method=key[3], policy=key[4], split=split,
                                   runs=len(batch), available=len(available), seeds=[r['seed'] for r in available],
                                   missing_seeds=[r['seed'] for r in batch if not r[mk]], metrics=metric_stats))

    aggregate('all_campaigns', selected)
    lookup = {(r['run'], r['policy']): r for r in selected}
    views = {}
    old = read(root/'outputs/experiments/multiseed_controls_v1/mosi_seven_controls_valid_mae/summary.json')
    views['historical_mosi_seven'] = [(run_from_path(x['source_run']), x['method']) for x in old.get('rows', [])]
    mo = []
    exp = root/'outputs/experiments'
    for seed in (13, 42, 2026):
        for method in ('full_kd','uniform_interaction','subset7','random_orthogonal','ensemble_full','first_order_interaction','first_second_order_interaction'):
            if method in ('full_kd','uniform_interaction'):
                p = exp / ('tav_epoch_checkpoints_v1' if seed == 13 else 'tav_main_v1') / 'students' / f'{"M3" if method=="full_kd" else "M4"}_seed{seed}'
            elif seed == 13:
                p = exp / ('interaction_evidence_v1' if method in ('random_orthogonal','first_second_order_interaction') else 'uniform_followup_v1') / 'mosei/students' / f'{method}_seed{seed}'
            else:
                branch = 'mosei_ensemble_controls' if method == 'ensemble_full' else 'mosei/followup_order_controls' if method in ('first_order_interaction','first_second_order_interaction') else 'mosei'
                p = exp/'multiseed_controls_v1'/branch/'students'/f'{method}_seed{seed}'
            mo.append((run_from_path(p), method))
    views['historical_mosei_seven'] = mo
    reg = read(exp/'mosi_research_v1/resource_queue/registry.json')
    unplanned = {run_from_path(x['run']) for x in reg.get('runs', []) if not x['official_test_planned']}
    for row in selected:
        if row['run'] in unplanned and not row['test_metrics']:
            row['test_status'] = 'not_planned_this_round'
    views['mosi_research_14'] = [(run_from_path(x['run']),x['method']) for x in reg.get('runs', [])]
    views['mosei_extensions_v3'] = [(rid, r['method']) for rid,r in runs.items() if 'equal_modality_extensions_v3' in rid]
    view_rows = []
    for view, items in views.items():
        batch = []
        for rid, method in items:
            if not rid:
                raise ValueError(f'missing curated run: {view} {method}')
            for policy in POLICIES:
                row = {**lookup[rid, policy], 'method': method, 'view': view}
                batch.append(row)
                view_rows.append(row)
        aggregate(view, batch)

    # Preserve the older F1/MAE tie-break protocol as its own view, never silently
    # mix it with today's earliest-epoch tie-break.
    historic_acc = read(exp/'multiseed_controls_v1/mosi/official_test_valid_acc2_nonzero/summary.json')
    (output/'historical_acc2_original_tiebreak.json').write_text(json.dumps(historic_acc, ensure_ascii=False, indent=2)+'\n')
    inventory = [{k:v for k,v in r.items() if k not in ('path','history','epoch_index','report','config')} for r in runs.values()]
    no_history = []
    for p in sorted((root/'outputs').rglob('run_config.json')):
        config=read(p)
        if not (p.parent/'history.json').exists() and 'epochs' in config and 'seed' in config and not any(t in str(p) for t in ('source_snapshot','schedule_revisions')):
            no_history.append(dict(run=str(p.parent.relative_to(root)), method=config.get('method',config.get('mode')), seed=config['seed'],
                                   recorded_status=read(p.parent/'status.json').get('status'), reason='no completed epoch history; archived status is not live process status'))
    ancillary = [dict(path=str(p.relative_to(root)), reason='teacher probe / efficiency / teacher diagnostic; dual student checkpoint selection not applicable')
                 for p in sorted((root/'outputs').rglob('report.json'))
                 if ('probe' in p.parts or 'probes' in p.parts or 'efficiency_benchmark_v1' in p.parts)]
    snapshot = datetime.now().astimezone().isoformat()
    payload = dict(snapshot=snapshot, selection='min validation MAE / max validation Acc2_nonzero; ties earliest epoch',
                   runs=inventory, selections=selected, aggregates=groups, curated_view_rows=view_rows,
                   excluded_histories=excluded, unmatched_test_records=unmatched, conflicting_test_records=conflicts,
                   no_history_runs=no_history, ancillary_reports=ancillary, parse_errors=parse_errors)
    (output/'summary.json').write_text(json.dumps(finite_json(payload), ensure_ascii=False, indent=2, allow_nan=False)+'\n')

    def write_csv(name, rows):
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: json.dumps(finite_json(v), ensure_ascii=False) if isinstance(v,(list,dict)) else finite_json(v) for k,v in row.items()} for row in rows)

    flat = []
    for r in selected:
        row = {k:v for k,v in r.items() if k not in ('valid_metrics','test_metrics')}
        for split, mk in (('validation','valid_metrics'),('test','test_metrics')):
            row.update({f'{split}_{k}':r[mk].get(k) for k in METRICS})
        flat.append(row)
    write_csv('per_run.csv',flat)
    write_csv('inventory.csv',inventory)
    write_csv('no_history_runs.csv',no_history)
    write_csv('ancillary_reports.csv',ancillary)
    write_csv('missing_tests.csv',[r for r in flat if r['test_status']!='available'])
    write_csv('aggregates.csv',[{k:v for k,v in g.items() if k!='metrics'} | {f'{metric}_{stat}':v for metric, stats in g['metrics'].items() for stat,v in stats.items()} for g in groups])

    def value(stats, metric):
        v=stats[metric]
        if v['mean'] is None:return '—'
        scale=100 if 'acc' in metric or 'f1' in metric else 1
        decimals=2 if scale==100 else 4
        result=f'{v["mean"]*scale:.{decimals}f}'
        if v['sd'] is not None:result+=f' ± {v["sd"]*scale:.{decimals}f}'
        return result

    def table(rows):
        lines=['| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |',
               '|---|---|---|---|---|---|---:|---:|---:|---:|']
        for g in rows:
            seedtext=','.join(map(str,g['seeds'])) or '—'
            lines.append(f'| {g["method"]} | {g["dataset"]} | {g["scope"]} | {g["policy"]} | {g["split"]} | {seedtext} ({g["available"]}/{g["runs"]}) | '+ ' | '.join(value(g['metrics'],m) for m in METRICS[:4])+' |')
        return lines

    counts=Counter(r['state'] for r in inventory)
    available=Counter(r['policy'] for r in selected if r['test_status']=='available')
    lines=['# 全部历史实验：两种验证 checkpoint 选择汇总','',f'快照时间：{snapshot}','',
           f'- 清点 {len(inventory)} 个有逐轮情感指标的学生训练目录，状态分布：{dict(counts)}。另有 {len(no_history)} 个无已完成 epoch 的配置目录，见 no_history_runs.csv；{len(excluded)} 个 history 不含逐轮情感指标（教师 Probe），独立列入 JSON 排除清单。',
           f'- 共 {len(selected)} 个 run × 选模记录。已有同轮测试：MAE 策略 {available["valid_mae"]} 个，Acc2 策略 {available["valid_acc2_nonzero"]} 个；缺失记录详见 missing_tests.csv。',
           '- 主口径统一为验证 MAE 最小 / 验证 Acc2_nonzero 最大；并列取最早 epoch。从各自 history 选择同一轮的全部指标，绝不拼接不同轮次的最优值。',
           '- 历史有一份 Acc2 汇总采用“Acc2 并列后 F1 更大、再 MAE 更小”的规则，已原样保存在 historical_acc2_original_tiebreak.json；与本表统一规则不同，不能混用其结果。',
           '- 若有测试 sweep，只取由验证指标确定的那个 epoch；不取 test-best。历史 sweep 已经看过测试集的事实不会因重新选模而消失，故不称为新的盲测。每个测试来源和探索属性均保留。',
           '- 同轮存在多次测评且数值略异时，按来源优先采用正式选模测评；仅有旧 sweep 时优先固定 epoch 路径而非可变 best 别名。不按成绩挑选，相同优先级仍不一致则标为冲突。',
           '- 早期 history 只记录 MAE、Acc2、Pearson：未保存的 F1 等指标保持缺失，不能从另一个 checkpoint 补齐。CSV/JSON 对每项聚合保留实际 n 与 seeds。',
           '- 同一方法名在不同轮次、预算、架构、数据子集下分组报告，不直接混成三 seed。当前未完成训练只进入逐 run 明细，不进入完成实验均值。',
           '- ± 为 seed 样本标准差；单 seed 不显示标准差；不足三 seed 不能称为三 seed 结论。测试覆盖不齐时均值仅描述表中已有 seeds，不与完整三 seed直接排名。',
           '- 路径和 epoch 能追溯的已有权重位置写入 checkpoint_paths；这不是加载权重验证。没有权重不代表没有历史指标；没有同轮测试也不代表一定没有权重。',
           '- 本次只读取并汇总记录，没有重训、补测或更改 GPU 队列。教师 probe、效率、交互重建诊断不按学生两种情感 checkpoint 重新排名。','',
           '## 文件','', '- [逐 run 双策略明细](per_run.csv)；[各组均值和标准差](aggregates.csv)；[缺失测试](missing_tests.csv)；[完整来源与审计](summary.json)。',
           '- 按策略单独阅读：[best_valid_mae 汇总](best_valid_mae.md)；[best_valid_acc2_nonzero 汇总](best_valid_acc2_nonzero.md)。',
           '- [全部实验线分组表](all_campaigns.md)；[全部 run 的轮次与结果](all_runs.md)。','']
    for view in views:
        lines += ['## '+view,'']+table([g for g in groups if g['view']==view])+['']
    lines+=['## 历史原 Acc2 并列规则（独立附表）','',
            '以下保留当时已发布的 5 方法 × 3 seed 测试结果，按 Acc2、F1、MAE、最早轮次依次选模。它与上文统一规则不同，不能互相填补缺失。','',
            '| 方法 | seeds | Test MAE ↓ | Test Acc2 NZ ↑ (%) | Test Pearson ↑ | Test F1 NZ ↑ (%) |',
            '|---|---|---:|---:|---:|---:|']
    older=defaultdict(list)
    for r in historic_acc.get('rows',[]):
        older[r['method']].append(r)
    for method, batch in sorted(older.items()):
        stats={k:{'mean':statistics.mean(r['test_metrics'][k] for r in batch), 'sd':statistics.stdev(r['test_metrics'][k] for r in batch) if len(batch)>1 else None} for k in METRICS}
        lines.append(f'| {method} | {",".join(str(r["seed"]) for r in batch)} | '+' | '.join(value(stats,m) for m in METRICS[:4])+' |')
    lines+=['','## 尚未完成的运行','', '| Run | 状态 | 已完成 epoch |','|---|---|---:|']
    lines += [f'| {r["run"]} | {r["state"]} | {r["epochs"]} |' for r in inventory if r['state']!='complete']
    lines+=['','## 匹配审计','',f'- JSON 解析问题：{len(parse_errors)}；未唯一匹配的测试记录：{len(unmatched)}；同 run 同 epoch 测评数值差异：{len(conflicts)}，其中未解决 {sum(x["resolution"]=="unresolved" for x in conflicts)}。详情见 summary.json。','']
    (output/'README.md').write_text('\n'.join(lines))
    detail=['# 全部实验线（不跨协议合并）','']
    for campaign in sorted({g['campaign'] for g in groups if g['view']=='all_campaigns'}):
        detail+=['## '+campaign,'']+table([g for g in groups if g['view']=='all_campaigns' and g['campaign']==campaign])+['']
    (output/'all_campaigns.md').write_text('\n'.join(detail))
    for policy in POLICIES:
        lines=[f'# best_{policy}：全部实验汇总','',
               '[口径、缺失项说明及来源审计](README.md)。所有表均采用同一选中 epoch 的完整指标，± 为样本标准差。','']
        for view in views:
            lines+=['## '+view,'']+table([g for g in groups if g['view']==view and g['policy']==policy])+['']
        lines+=['## 全部历史实验线（包括 pilot 与 smoke）','']
        for campaign in sorted({g['campaign'] for g in groups if g['view']=='all_campaigns'}):
            lines+=['### '+campaign,'']+table([g for g in groups if g['view']=='all_campaigns' and g['campaign']==campaign and g['policy']==policy])+['']
        (output/f'best_{policy}.md').write_text('\n'.join(lines))
    lines=['# 全部学生 run：双 checkpoint 明细','', '| Run | 状态 | 策略 | epoch | Valid MAE | Valid Acc2 (%) | Test MAE | Test Acc2 (%) | 测试记录 |', '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in selected:
        def scalar(block,key,scale=1):
            return f'{block[key]*scale:.6f}' if isinstance(block.get(key), (int,float)) and math.isfinite(block[key]) else '—'
        lines.append(f'| {r["run"]} | {r["state"]} | {r["policy"]} | {r["epoch"]} | {scalar(r["valid_metrics"],"mae")} | {scalar(r["valid_metrics"],"acc2_nonzero",100)} | {scalar(r["test_metrics"],"mae")} | {scalar(r["test_metrics"],"acc2_nonzero",100)} | {r["test_status"]} |')
    (output/'all_runs.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(runs=len(runs),selected=len(selected),test_available=available,excluded=len(excluded),unmatched=len(unmatched),conflicts=len(conflicts),parse_errors=len(parse_errors),output=str(output))))


if __name__ == '__main__':
    main()
