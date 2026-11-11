"""Freeze immutable train/validation audit inputs; never train or read test predictions."""
import itertools
import random
import shutil
import subprocess
from collections import Counter,defaultdict
from common import *

def main():
    assert not (OUT/'protocol/plan.json').exists(), 'Use frozen plan, do not silently replace it'
    for name in ['protocol','tables','figures','decisions','predictions','gradients','logs','status']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    assets_paths={d:ROOT/f'outputs/experiments/main_table_v1/{d}/assets/protocol.json' for d in ['mosi','mosei']}
    assets_paths.update({d:ROOT/f'outputs/experiments/sims_native_v1/{d}/assets/protocol.json' for d in ['sims','sims_v2']})
    runs=[]
    for d,method,seed in itertools.product(['mosi','mosei','sims','sims_v2'],['full_kd','subset7','uniform_interaction'],[13,42,2026]):
        if d in ['sims','sims_v2']:run=ROOT/f'outputs/experiments/sims_native_v1/{d}/students/{method}_seed{seed}'
        elif d=='mosi':
            run=ROOT/(f'outputs/experiments/uniform_main_v1/mosi/students/{method}_seed13' if seed==13 else f'outputs/experiments/multiseed_controls_v1/mosi/students/{method}_seed{seed}')
        elif method=='subset7':
            run=ROOT/(f'outputs/experiments/uniform_followup_v1/mosei/students/subset7_seed13' if seed==13 else f'outputs/experiments/multiseed_controls_v1/mosei/students/subset7_seed{seed}')
        else:run=ROOT/f'outputs/experiments/tav_{"epoch_checkpoints" if seed==13 else "main"}_v1/students/{"M3" if method=="full_kd" else "M4"}_seed{seed}'
        runs.append(dict(id=f'{d}_{method}_seed{seed}',dataset=d,method=method,seed=seed,run=str(run)))
    protocols={};issues=[];fingerprints={};manifest={};audits=[]
    for d,path in assets_paths.items():
        original=read(path);a=mapped(original)
        # Only the audit copy changes paths. Original bytes and original hashes stay recorded.
        save(OUT/f'protocol/{d}_assets_original.json',original)
        manifest_rows=mapped(rows(a['manifest']))
        for r in manifest_rows:
            if 'class_7_index' not in r:r['class_7_index']=r['class_5_index']
        assert {r['split'] for r in manifest_rows}=={'train','valid'}
        assert len({r['sample_id'] for r in manifest_rows})==len(manifest_rows)
        parents={};videos=defaultdict(set)
        for r in manifest_rows:
            key=r['parent_sample_id'];signature=(r['split'],r['sentiment'],r['video_id'],r['class_7_index'])
            assert parents.setdefault(key,signature)==signature
            videos[r['split']].add(r['video_id'])
            assert r['aggregation_weight']>0
            for k in ['audio_segment_path','silent_video_path']:
                if not Path(r[k]).is_file():issues.append(dict(dataset=d,type='missing_media',path=r[k]))
        overlaps={'train_valid':len(videos['train']&videos['valid'])}
        # Test split metadata is inspected for overlap only; never predictions or test task metrics.
        test_path=Path(a['manifest']).parent/'official_test_windowed.jsonl'
        if test_path.exists():
            test=rows(test_path);tv={r['video_id'] for r in test}
            overlaps.update(train_test=len(videos['train']&tv),valid_test=len(videos['valid']&tv))
            fingerprints[str(test_path)]=sha(test_path)
        data_path=OUT/f'protocol/{d}_train_valid_mapped.jsonl';jsonl(data_path,manifest_rows)
        original_manifest=a['manifest'];a['manifest']=str(data_path)
        probes_root=Path(a['teacher_targets']).parents[1]
        probes=[];probe_ids=None
        for seed in a['probe_seeds']:
            directory=probes_root/f'seed{seed}'
            pred=rows(directory/'predictions.jsonl');keys={(r['parent_sample_id'],r['subset']) for r in pred}
            assert len(keys)==len(pred)
            assert keys=={(p,s) for p in parents for s in NAMES}
            for r in pred:
                meta=parents[r['parent_sample_id']]
                assert r['split']==meta[0] and abs(r['target_sentiment']-meta[1])<1e-6
                assert r.get('video_id',meta[2])==meta[2]
            probes.append(dict(seed=seed,directory=str(directory),prediction_sha256=sha(directory/'predictions.jsonl'),checkpoint_sha256=sha(directory/'teacher_probe.pt'),report=read(directory/'report.json')))
        ensemble=rows(a['ensemble_targets']);interaction=rows(a['interaction_targets'])
        assert {r['parent_sample_id'] for r in ensemble}==set(parents)
        assert {r['parent_sample_id'] for r in interaction}==set(parents)
        for p in [path,Path(original_manifest),Path(a['teacher_targets']),Path(a['ensemble_targets']),Path(a['interaction_targets']),Path(a['teacher_feature_index'])]:
            fingerprints[str(p)]=sha(p)
        mismatches=[];path_only_repairs=[]
        for p,expected in original.get('input_sha256',{}).items():
            pp=Path(mapped(p))
            if pp in [Path(a['ensemble_targets']),Path(a['interaction_targets']),Path(original_manifest),Path(a['teacher_targets'])]:
                actual=sha(pp)
                if actual!=expected:
                    raw=pp.read_bytes()
                    recovered=hashlib.sha256(raw.replace(b'/data/sjq/kd',b'/home/wy/sjq/kd')).hexdigest()
                    if str(pp)==original_manifest and recovered==expected:
                        path_only_repairs.append(dict(path=str(pp),expected=expected,actual=actual,proof='replacing /data/sjq/kd with /home/wy/sjq/kd recovers exact frozen SHA256'))
                    else:mismatches.append(dict(path=str(pp),expected=expected,actual=actual))
        assert not mismatches, f'{d} frozen data/targets changed: {mismatches}'
        protocol_path=OUT/f'protocol/{d}_assets_mapped.json';save(protocol_path,a)
        rng=random.Random(130013)
        candidates=[r for r in manifest_rows if r['split']=='train'];rng.shuffle(candidates)
        selected=[];seen=set()
        # Six deterministic train windows, stratified by duration/sign; no label-dependent adaptation.
        for low,high in [(0,5),(5,15),(15,float('inf'))]:
            for sign in [-1,1]:
                eligible=[r for r in candidates if low<=r.get('window_duration',r.get('duration',0))<high and r['sentiment']*sign>0 and r['video_id'] not in seen]
                if eligible:selected.append(eligible[0]);seen.add(eligible[0]['video_id'])
        if len(selected)<6:
            for r in candidates:
                if len(selected)>=6:break
                if r['video_id'] not in seen:selected.append(r);seen.add(r['video_id'])
        assert len(selected)==6
        tolerance=.5 if d in ['mosi','mosei'] else .5/3
        protocol=dict(dataset=d,assets=str(protocol_path),original_assets=str(path),original_manifest=original_manifest,path_only_repairs=path_only_repairs,
            probe_root=str(probes_root),probes=probes,window_counts=dict(Counter(r['split'] for r in manifest_rows)),
            parent_counts=dict(Counter(v[0] for v in parents.values())),source_video_overlap=overlaps,
            classification_classes=5 if d.startswith('sims') else 7,label_range=[-1,1] if d.startswith('sims') else [-3,3],
            acc2='y>0 and prediction>0; zeros included' if d.startswith('sims') else 'y!=0; y>0 and prediction>0',
            quadrant_tolerance=tolerance,quadrant_rule='absolute regression error <= tolerance; fixed at 1/12 of native range width before prediction analysis',
            gradient_samples=[{k:r[k] for k in ['sample_id','parent_sample_id','video_id','split','sentiment','window_duration']} for r in selected],
            probe_independence='three independently fitted heads share the same teacher hidden cache; not independent teachers',
            feature_config=mapped(read(Path(a['teacher_features']).parent/'run_config.json')))
        protocols[d]=protocol
        audits.append(dict(dataset=d,scope='train_valid',status='passed' if not [x for x in issues if x['dataset']==d] else 'failed',
            train_parents=protocol['parent_counts']['train'],valid_parents=protocol['parent_counts']['valid'],
            native_label_range=str(protocol['label_range']),classes=protocol['classification_classes'],acc2=protocol['acc2'],
            source_overlap_train_valid=overlaps['train_valid'],source_overlap_train_test=overlaps.get('train_test'),source_overlap_valid_test=overlaps.get('valid_test'),
            window_policy='max30_overlap5; duration/overlap corrected weighted utterance aggregation',
            teacher_base_probe_count=1,icd_target_probe_count=3,teacher_temperature=a['teacher_temperature'],quadrant_tolerance=tolerance))
    for case in runs:
        run=Path(case['run']);config=read(run/'run_config.json');history=read(run/'history.json')
        if isinstance(history,dict):history=history.get('epochs',history.get('history'))
        chosen=min(history,key=lambda r:(r['valid_metrics']['mae'],int(r['epoch'])))
        desired=int(chosen['epoch']);candidates=[run/f'checkpoints/epoch_{desired:03d}.pt',run/'best_valid_mae.pt',run/'best.pt']
        available=[p for p in candidates if p.is_file()]
        if not available:issues.append(dict(id=case['id'],type='checkpoint_missing',expected_epoch=desired));continue
        # Actual serialized epoch is checked in GPU loader; best.pt is never trusted by filename alone.
        checkpoint=available[0];stat=checkpoint.stat()
        case.update(checkpoint=str(checkpoint),epoch=desired,checkpoint_sha256=sha(checkpoint),checkpoint_bytes=stat.st_size,checkpoint_mtime_ns=stat.st_mtime_ns,
            config_sha256=sha(run/'run_config.json'),config=mapped(config),valid_metrics=chosen['valid_metrics'],
            original_checkpoint_selection=config.get('checkpoint_selection','valid_mae'),selection='min validation MAE, earliest epoch tie break')
        save(OUT/f'protocol/{case["id"]}_config_original.json',config)
    manifest=dict(schema='phase0-transferability-audit-v1',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status_before=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        data_fingerprints=fingerprints,protocols=protocols,cases=runs,issues=issues,
        policy=dict(phase=0,training=False,optimizer_steps=0,test_predictions_used=False,test_labels_used_for_design=False,
                    checkpoint_selection='validation MAE minimum; earliest epoch tie-break',gradient_seed=130013,gradient_mode='eval with autograd; deterministic dropout disabled; no parameter/buffer updates',
                    gradient_batch_size=1,gradient_parameter_groups=['all','fusion','text_lora','audio_lora','video_lora','prediction_head','other'],
                    tolerance='1/12 native range width, fixed before reading predictions',scheduled_phases=[0]),
        source_sha256={str(p):sha(p) for directory in [ROOT/'project/src/rdid_mosei',ROOT/'project/sims_native/src/rdid_sims'] for p in directory.glob('*.py')})
    save(OUT/'protocol/asset_manifest.json',manifest)
    save(OUT/'protocol/split_fingerprints.json',fingerprints)
    table('protocol_audit.csv',audits)
    save(OUT/'protocol/plan.json',manifest)
    save(OUT/'status/prepare.json',dict(status='complete' if not issues else 'failed',issues=issues,cases=len(runs)))
    print(json.dumps(dict(cases=len(runs),protocols={d:{k:v for k,v in p.items() if k in ['parent_counts','source_video_overlap']} for d,p in protocols.items()},issues=issues)),flush=True)
    assert not issues,issues

if __name__=='__main__':main()
