"""Audit caches and freeze the MOSI-first plan before training or test."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = Path('/ai/sjq/kd')
PREVIOUS = ROOT/'outputs/experiments/ea_n2_transfer_20261111_v1'

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def lines(p): return [json.loads(line) for line in p.open() if line.strip()]

def main():
    assert not (BASE/'plan.json').exists(), 'Never overwrite frozen campaign'
    for name in ['assets','analysis','preflight','runs','tests','logs','workers']:
        (BASE/name).mkdir(exist_ok=True)
    shutil.copytree(PREVIOUS/'source', BASE/'source', ignore=shutil.ignore_patterns('__pycache__'))
    sys.path[:0] = [str(BASE/'source/project/src'),str(BASE/'source/project/scripts')]
    from rdid_mosei.interaction import mobius_matrix, mobius_transform, inverse_mobius
    from mechanisms import transform, transform_metadata
    from types import SimpleNamespace
    assets=json.loads((PREVIOUS/'assets/mosi/protocol.json').read_text())
    original_hashes=assets.pop('input_sha256')
    assets['upstream_input_provenance']=original_hashes
    assets['seeds']=[13]
    required=[Path(assets[k]) for k in ['manifest','teacher_targets','teacher_probe_report','ensemble_targets','interaction_targets','teacher_feature_index','upstream_sources']]
    for model in ['Qwen3-0.6B-Base','WavLM-Base-Plus','VideoMAE-Base']:
        directory=ROOT/'model'/model
        required += list(directory.glob('*.safetensors'))+list(directory.glob('*.bin'))+[directory/'config.json']
        assert list(directory.glob('*.safetensors')) or list(directory.glob('*.bin')), directory
    for p in required: assert p.is_file(),p
    assets['input_sha256']={str(p):digest(p) for p in required}
    save(BASE/'assets/protocol.json',assets)
    ensemble=lines(Path(assets['ensemble_targets']))
    interactions={r['parent_sample_id']:r for r in lines(Path(assets['interaction_targets']))}
    manifest=lines(Path(assets['manifest']))
    assert {r['split'] for r in manifest}=={'train','valid'}
    index={r['parent_sample_id']:r for r in ensemble}
    assert len(index)==len(ensemble)
    for row in manifest:
        target=index[row['parent_sample_id']]
        assert target['split']==row['split'] and target['video_id']==row['video_id']
        for field in ['audio_segment_path','silent_video_path']: assert Path(row[field]).is_file(),row[field]
    for split in ['train','valid']:
        assert {r['parent_sample_id'] for r in manifest if r['split']==split}=={r['parent_sample_id'] for r in ensemble if r['split']==split}
    order=['t','a','v','ta','tv','av','tav']
    for r in ensemble:
        assert r['subsets']==order and r['probe_seeds']==[2026,2027,2028]
        assert abs(sum(r['tav_kd_probability_mean'])-1)<1e-10
    m=mobius_matrix().numpy(); gram=m.T@m
    train=[r for r in ensemble if r['split']=='train']
    raw=np.array([r['subset_score_mean'] for r in train])
    baseline=assets['student_empty_baseline']
    coords=(raw-baseline)@m.T
    reference=np.array([interactions[r['parent_sample_id']]['mean'] for r in train])
    assert np.max(np.abs(coords-reference))<1e-6
    std=dict(subset_std=raw.std(0,ddof=1).tolist(),coordinate_std=coords.std(0,ddof=1).tolist(),
             epsilon=0.05,ddof=1,split='train',unit='unique utterance',count=len(train),source=str(assets['ensemble_targets']),source_sha256=digest(Path(assets['ensemble_targets'])))
    save(BASE/'assets/normalization.json',std)
    # Validate ensemble caches against each of the three original prediction files.
    probes=[]; missing=[]
    for seed in [2026,2027,2028]:
        directory=ROOT/f'outputs/experiments/main_table_v1/mosi/probes/seed{seed}'
        pred=directory/'predictions.jsonl'
        pindex={(r['parent_sample_id'],r['subset']):r for r in lines(pred)}
        score=np.array([[pindex[(r['parent_sample_id'],s)]['probe_score'] for s in order] for r in ensemble])
        probes.append(score)
        checkpoint=directory/'teacher_probe.pt'
        if not checkpoint.is_file():missing.append(str(checkpoint))
    assert np.max(np.abs(np.mean(probes,0)-np.array([r['subset_score_mean'] for r in ensemble])))<1e-6
    q=[]
    for seed in [2026,2027,2028]:
        directory=ROOT/f'outputs/experiments/main_table_v1/mosi/probes/seed{seed}'
        pindex={(r['parent_sample_id'],r['subset']):r for r in lines(directory/'predictions.jsonl')}
        logits=np.array([pindex[(r['parent_sample_id'],'tav')]['classification_logits'] for r in ensemble])
        logits=logits/(assets['probe_temperatures'][str(seed)]*assets['kd_temperature'])
        prob=np.exp(logits-logits.max(1,keepdims=True));q.append(prob/prob.sum(1,keepdims=True))
    assert np.max(np.abs(np.mean(q,0)-np.array([r['tav_kd_probability_mean'] for r in ensemble])))<1e-6
    torch.manual_seed(13)
    x=torch.randn(16,7,dtype=torch.float64)
    assert torch.max(torch.abs(inverse_mobius(mobius_transform(x,baseline),baseline)-x))<1e-12
    singular=np.linalg.svd(m,compute_uv=False)
    matrices={}
    for seed in [20261111,20261112,20261113]:
        a=SimpleNamespace(mechanism='matched',matrix_seed=seed,loss_kind='huber')
        matrix=transform(a).numpy()
        assert np.max(np.abs(np.linalg.svd(matrix,compute_uv=False)-singular))<1e-12
        matrices[str(seed)]=transform_metadata(a)
    save(BASE/'analysis/matched_matrices.json',matrices)
    stats=dict(matrix=m.tolist(),rank=int(np.linalg.matrix_rank(m)),singular_values=singular.tolist(),
               condition_number=float(np.linalg.cond(m)),gram=gram.tolist(),gram_eigenvalues=np.linalg.eigvalsh(gram).tolist(),
               inverse_reconstruction_max_error=float(torch.max(torch.abs(inverse_mobius(mobius_transform(x,baseline),baseline)-x))),
               coordinate_mean=coords.mean(0).tolist(),coordinate_std=coords.std(0,ddof=1).tolist(),
               coordinate_quantiles=np.quantile(coords,[0,.05,.25,.5,.75,.95,1],axis=0).tolist(),
               coordinate_covariance=np.cov(coords,rowvar=False).tolist(),coordinate_correlation=np.corrcoef(coords,rowvar=False).tolist(),
               probe_variance_mean=np.mean([interactions[r['parent_sample_id']]['variance'] for r in train],0).tolist())
    save(BASE/'analysis/geometry.json',stats)
    jobs=[]
    def add(name,mechanism='icd',ce=0,weight=.30,kind='huber',matrix=20261111,seed=13,stage='phase2'):
        jobs.append(dict(id=f'{name}_seed{seed}',name=name,mechanism=mechanism,task_ce=ce,lambda_interaction=weight,
                         loss_kind=kind,matrix_seed=matrix,seed=seed,stage=stage))
    add('A_full_no_ce','none',weight=0)
    add('B_N2_icd030')
    add('C_full_ce','none',ce=.5,weight=0)
    add('D_icd030_ce',ce=.5)
    add('subset7','subset')
    for i in range(3):add(f'matched{i}_huber','matched',matrix=20261111+i)
    add('diagonal_mse','diagonal',kind='mse')
    add('weighted_mse','weighted',kind='mse')
    add('weighted_huber','weighted')
    add('normalized_icd','normalized_icd')
    add('normalized_subset','normalized_subset')
    add('orthogonal_huber','orthogonal')
    add('icd_mse',kind='mse')
    add('matched0_mse','matched',kind='mse')
    add('F7_icd025',weight=.25)
    add('E4_icd050_ce',ce=.5,weight=.5)
    plan=dict(schema='mosi-first-20261111-v1',dataset='mosi',jobs=jobs,epochs=30,patience=7,batch_size=8,
              learning_rate=.0001,weight_decay=.01,student_seeds=[13],teacher_seeds=[2026,2027,2028],
              checkpoint_retention=['best_valid_mae.pt','best_valid_acc2_nonzero.pt','last.pt'],
              test_checkpoints=['valid_mae','valid_acc2_nonzero','last'],official_test_selects_nothing=True,
              phase3_gate='disabled by user: seed13 only',
              max_processes_per_gpu=2,minimum_free_mib=38000,skipped_missing_weights=missing,
              deferred=['all additional student seeds per user instruction','checkpoint diagnostics requiring absent checkpoints','optional normalized multiseed pending validation evidence','adaptive residual methods pending controls','MOSEI and other datasets'])
    save(BASE/'plan.json',plan)
    save(BASE/'teacher_manifest.json',dict(assets=assets,missing_original_probe_checkpoints=missing,
          use_policy='Existing verified prediction caches suffice for student training. Do not retrain missing Probe weights or run experiments requiring their loading.'))
    save(BASE/'preflight/asset_audit.json',dict(passed=True,train_utterances=len(train),valid_utterances=sum(r['split']=='valid' for r in ensemble),
          train_valid_windows=len(manifest),missing_weights=missing,mathematical_identity_max_error=float(np.max(np.abs(coords-reference))),
          ensemble_scores_and_probabilities_verified_against_all_three_probes=True))
    report=['# MOSI offline geometry report','',f'Train utterances: {len(train)}; rank: {stats["rank"]}; condition number: {stats["condition_number"]:.8f}.',
            'Subset order: T, A, V, TA, TV, AV, TAV. Common anchor equals the mean of the three train-TAV probe means.',
            'Three-probe raw scores and temperature-softmax probabilities were reconstructed and verified against the cached ensemble.',
            'Normalization uses unique training utterances only; sample standard deviation; floor 0.05.',
            'Exact matrices, covariance, correlations, quantiles and reliability statistics: geometry.json and matched_matrices.json.',
            'Original Probe weights are missing; checkpoint-dependent work is skipped. No missing weights are downloaded or retrained.',
            'Fixed-batch base/auxiliary and per-coordinate gradient diagnostics are recorded before every training epoch.',
            'Historical MOSI test comparisons and this mechanism screen are exploratory. All methods and representative matrices are locked before new test evaluation.']
    (BASE/'analysis/mosi_offline_geometry_report.md').write_text('\n\n'.join(report)+'\n')
    print(json.dumps(dict(passed=True,jobs=len(jobs),phase2=18,phase3=0,missing_probe_weights=len(missing))))

if __name__=='__main__':main()
