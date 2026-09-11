#!/usr/bin/env python3
"""Read-only analysis of completed experiments; writes a dated docs snapshot."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "outputs/student"
V = S / "video_adaptation_v1"
HASHES = {}


def read(path):
    path = Path(path)
    payload = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in payload.splitlines() if line.strip()]
    return json.loads(payload)


def main():
    import torch

    HASHES[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = read(ROOT / "dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl")
    assert {r['split'] for r in manifest} == {'train', 'valid'}
    metadata = {}
    for r in manifest:
        if r['split'] == 'valid':
            metadata[r['parent_sample_id']] = r
    ids = sorted(metadata)
    y = np.array([metadata[i]['sentiment'] for i in ids])
    videos = np.array([metadata[i]['video_id'] for i in ids])
    durations = np.array([metadata[i]['utterance_duration'] for i in ids])
    groups = np.unique(videos, return_inverse=True)[1]
    count = int(groups.max()) + 1
    weights = np.random.default_rng(20260911).multinomial(
        count, np.full(count, 1 / count), size=10000)
    denominator = weights @ np.bincount(groups)

    def ci(delta):
        means = (weights @ np.bincount(groups, weights=delta)) / denominator
        return np.quantile(means, [.025, .975]).tolist()

    def predictions(path, subset=None):
        rows = [r for r in read(path) if r['split'] == 'valid' and
                (subset is None or r.get('subset') == subset)]
        mapping = {r['parent_sample_id']: r for r in rows}
        assert len(rows) == len(mapping) and set(mapping) == set(ids), path
        assert np.allclose([mapping[i]['target_sentiment'] for i in ids], y, atol=1e-7, rtol=0), path
        assert all(mapping[i].get('video_id', videos[n]) == videos[n] for n, i in enumerate(ids)), path
        p = np.array([mapping[i].get('prediction', mapping[i].get('probe_score')) for i in ids])
        assert np.isfinite(p).all(), path
        return p

    def compare(a, b):
        delta = abs(b-y) - abs(a-y)
        shift = abs(b-a)
        return {'delta_mae_candidate_minus_base': float(delta.mean()), 'cluster_ci95': ci(delta),
                'mean_absolute_prediction_change': float(shift.mean()),
                'p50_p90_p99_absolute_prediction_change': np.quantile(shift, [.5, .9, .99]).tolist(),
                'fraction_prediction_change_over_0_1': float(np.mean(shift > .1)),
                'fraction_error_improved': float(np.mean(delta < -1e-8)),
                'fraction_error_worsened': float(np.mean(delta > 1e-8)),
                'gain_mass': float(np.maximum(-delta, 0).mean()),
                'harm_mass': float(np.maximum(delta, 0).mean()),
                'fraction_polarity_changed_nonzero_labels': float(np.mean((a[y != 0] >= 0) != (b[y != 0] >= 0)))}

    result = {'created_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'Completed reports and existing train/valid artifacts only. No model inference or official-test access.',
              'bootstrap': {'seed': 20260911, 'repetitions': 10000, 'video_clusters': count,
                            'utterances': len(ids), 'scope': 'Conditional on selected checkpoints and observed seeds; exploratory, no multiplicity adjustment.'},
              'video': {}, 'families': {}, 'paired_core': {}}
    p = {}
    for mode in ['frozen_video', 'video_lora', 'ta_only']:
        d = V / f'{mode}_seed13'
        assert read(d/'status.json')['status'] == 'complete'
        report = read(d/'report.json')
        p[mode] = predictions(d/'predictions.jsonl')
        assert abs(np.mean(abs(p[mode]-y)) - report['valid_metrics']['mae']) < 1e-10
        row = {'report': report, 'perturbation_analysis': {}}
        for name in report['diagnostics']:
            q = predictions(d/f'diagnostic_{name}.json')
            row['perturbation_analysis'][name] = compare(p[mode], q)
            assert abs(np.mean(abs(q-y)) - report['diagnostics'][name]['metrics']['mae']) < 1e-10
        config = read(d/'run_config.json')
        row['teacher_subset'] = config['teacher_subset']
        row['teacher_calibration_temperature'] = config['teacher_calibration_temperature']
        result['video'][mode] = row
    result['video_comparisons'] = {f'{b}_minus_{a}':compare(p[a], p[b]) for a,b in
                                  [('frozen_video','video_lora'),('ta_only','video_lora'),('ta_only','frozen_video')]}
    configs = [read(V/f'{mode}_seed13/run_config.json') for mode in p]
    common = set.intersection(*(set(c['input_sha256']) for c in configs))
    mismatches = [k for k in sorted(common) if len({c['input_sha256'][k] for c in configs}) > 1]
    current_source_mismatches = sorted({k for c in configs for k,h in c['input_sha256'].items()
        if '/project/' in k and hashlib.sha256(Path(k).read_bytes()).hexdigest() != h})
    checkpoint_path = V/'video_lora_seed13/best.pt'
    HASHES[str(checkpoint_path.relative_to(ROOT))] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    visual_b_norms = {k:float(t.float().norm()) for k,t in checkpoint['model'].items()
                     if k.startswith('video_encoder.') and 'lora_B' in k}
    assert len(visual_b_norms) == 8 and all(np.isfinite(v) and v > 0 for v in visual_b_norms.values())
    first_grads = read(V/'video_lora_seed13/video_gradient_audit.json')
    result['implementation_audit'] = {'common_input_fingerprint_mismatches':mismatches,
        'current_source_fingerprint_mismatches':current_source_mismatches,
        'best_epoch':checkpoint['epoch'],'best_visual_lora_B_norms':visual_b_norms,
        'first_visual_lora_B_gradients_all_nonzero':all(v is not None and np.isfinite(v) and v > 0
            for k,v in first_grads.items() if 'lora_B' in k)}
    assert not mismatches and not current_source_mismatches
    del checkpoint
    result['duration_groups_exploratory'] = {}
    for name, mask in [('up_to_7_5s',durations<=7.5),('7_5_to_15s',(durations>7.5)&(durations<=15)),('over_15s',durations>15)]:
        result['duration_groups_exploratory'][name] = {'n':int(mask.sum()),
            'mae':{mode:float(np.mean(abs(pred[mask]-y[mask]))) for mode,pred in p.items()}}
    teacher_ta = predictions(ROOT/'outputs/probe/official_train_valid_seed2026/predictions.jsonl', 'ta')
    teacher_tav = predictions(ROOT/'outputs/probe/official_train_valid_seed2026/predictions.jsonl', 'tav')
    teacher_gain = abs(teacher_ta-y)-abs(teacher_tav-y)
    student_gain = abs(p['ta_only']-y)-abs(p['video_lora']-y)
    result['teacher_student_exploratory'] = {'teacher_tav_vs_ta':compare(teacher_ta,teacher_tav),
        'gain_pearson':float(np.corrcoef(teacher_gain,student_gain)[0,1]),
        'caveat':'Student TA and TAV are separately trained and use different teacher subsets; gain correlation is not causal visual transfer.',
        'label_defined_groups':{}}
    for name, mask in [('teacher_video_helps',teacher_gain>0),('teacher_video_hurts_or_ties',teacher_gain<=0)]:
        result['teacher_student_exploratory']['label_defined_groups'][name] = {
            'n':int(mask.sum()),'teacher_mean_gain':float(teacher_gain[mask].mean()),
            'student_mean_gain':float(student_gain[mask].mean())}
    families = {'legacy_D1':'stage_d_d1_full_kd_lora', 'legacy_D2_RU':'stage_d_d2_ru_lora',
                'C0_task_only':'stage_d_c0_task_only_lora',
                'cv2_frozen':'stage_d_cv2_cminus1_online_frozen_full_kd_tempfix',
                'cv2_C1':'stage_d_cv2_c1_full_kd_lora_tempfix',
                'cv2_C2':'stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean'}
    family_predictions = {}
    for name, prefix in families.items():
        rows = []; family_predictions[name] = {}
        for seed in [13,42,2026]:
            d = S/f'{prefix}_seed{seed}'
            if not (d/'report.json').exists():
                continue
            report = read(d/'report.json')
            q = predictions(d/'predictions.jsonl')
            assert abs(np.mean(abs(q-y)) - report['valid_metrics']['mae']) < 1e-10
            family_predictions[name][seed] = q
            rows.append({'seed':seed,'best_epoch':report['best_epoch'], 'valid_metrics':report['valid_metrics']})
        maes = [r['valid_metrics']['mae'] for r in rows]
        result['families'][name] = {'runs':rows,'n_seeds':len(rows),
            'mean_mae':float(np.mean(maes)) if maes else None,
            'sample_sd_mae':float(np.std(maes,ddof=1)) if len(maes)>1 else None}
    for a,b in [('legacy_D1','legacy_D2_RU'),('cv2_frozen','cv2_C1'),('C0_task_only','cv2_C1'),('cv2_C1','cv2_C2')]:
        seeds = sorted(set(family_predictions[a]) & set(family_predictions[b]))
        diffs = [abs(family_predictions[b][s]-y)-abs(family_predictions[a][s]-y) for s in seeds]
        mean_delta = np.mean(diffs,axis=0)
        result['paired_core'][f'{b}_minus_{a}'] = {
            'seeds':seeds,'per_seed':{s:compare(family_predictions[a][s],family_predictions[b][s]) for s in seeds},
            'mean_delta_mae':float(mean_delta.mean()), 'cluster_ci95_mean_over_fixed_seeds':ci(mean_delta)}
    result['input_sha256'] = HASHES
    path = ROOT/'docs/video_followup_20260911.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(path)
    print(json.dumps({'video_comparisons':result['video_comparisons'],
                      'perturbations':{k:v['perturbation_analysis'] for k,v in result['video'].items()},
                      'duration_groups':result['duration_groups_exploratory'],
                      'teacher_student':result['teacher_student_exploratory'],
                      'paired_core':result['paired_core']},indent=2))


if __name__ == '__main__':
    main()
