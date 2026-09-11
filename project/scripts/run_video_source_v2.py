#!/usr/bin/env python3
"""Persistent, resumable teacher-timing and equal-coverage T/A/V experiment."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'outputs/experiments/video_source_v2'
MANIFEST = ROOT/'dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl'
SCRIPTS = ROOT/'project/scripts'
sys.path.insert(0,str(SCRIPTS))
from train_video_adaptation import atomic_json, sha256
from run_video_adaptation_queue import compare

SEEDS = (13,42,2026)
MODES = ('frozen_video','video_lora','ta_only')
PROBE_SEEDS = (2026,2027,2028)
ENV = {**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
       'HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false',
       'PYTHONUNBUFFERED':'1','FORCE_QWENVL_VIDEO_READER':'decord'}


def select_diagnostic(rows):
    candidates=[r for r in rows if r['split']=='train' and r.get('window_count',1)==1]
    rng=np.random.default_rng(20260911)
    candidates=[candidates[int(i)] for i in rng.permutation(len(candidates))]
    used=set(); selected=[]
    for sign,count in [(-1,22),(0,20),(1,22)]:
        group=[]
        for r in candidates:
            if np.sign(r['sentiment'])==sign and r['video_id'] not in used:
                group.append(r);used.add(r['video_id'])
                if len(group)==count:break
        if len(group)!=count:raise ValueError('insufficient diagnostic source videos')
        selected.extend(group)
    return sorted(selected,key=lambda r:r['sample_id'])


def gpu_idle(gpu):
    try:
        r=subprocess.run(['nvidia-smi',f'--id={gpu}',
            '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=10)
        if r.returncode: return False,{'error':r.stderr.strip()}
        memory,util=[int(x.strip()) for x in r.stdout.strip().split(',')]
        return memory<2000 and util<10,{'memory_mib':memory,'utilization':util}
    except (subprocess.TimeoutExpired,ValueError,FileNotFoundError) as exc:
        return False,{'error':repr(exc)}


def status(base, **kwargs):
    value={'updated_at':time.time(),'official_test_evaluated':False,**kwargs}
    atomic_json(value,base/'status.json')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def progress(output,kind):
    try:
        if kind=='features' and (output/'completed.npy').exists():
            flags=np.load(output/'completed.npy',mmap_mode='r')
            return {'completed':int(flags.sum()),'total':len(flags)}
        if kind=='diagnostic':
            return {'completed':len(list(output.glob('[0-9][0-9][0-9]_*.json'))),'total':192}
        if kind=='probe' and (output/'history.json').exists():
            h=json.loads((output/'history.json').read_text())
            return h[-1] if h else {}
        if kind=='student' and (output/'status.json').exists():
            return json.loads((output/'status.json').read_text())
    except (OSError,ValueError):
        pass
    return {}


def snapshot_sources():
    names=['run_video_source_v2.py','diagnose_teacher_video_pair.py','extract_teacher_probe_features.py',
        'train_teacher_probe_v2.py','train_video_adaptation_v2.py','train_video_adaptation.py',
        'run_video_adaptation_queue.py','train_student_baseline.py']
    paths=[SCRIPTS/n for n in names]+list((ROOT/'project/src/rdid_mosei').glob('*.py'))
    return {str(p):sha256(p) for p in sorted(set(paths))}


def verify_sources(plan):
    for path,digest in plan['source_sha256'].items():
        if sha256(Path(path))!=digest:
            raise ValueError(f'Frozen experiment source changed: {path}')
    if sha256(MANIFEST)!=plan['manifest_sha256']:
        raise ValueError('full manifest changed')
    for path,identity in plan['model_files'].items():
        st=Path(path).stat()
        if {'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}!=identity:
            raise ValueError(f'local model file identity changed: {path}')


def prepare(base):
    base.mkdir(parents=True,exist_ok=True)
    (base/'logs').mkdir(exist_ok=True)
    rows=[json.loads(line) for line in MANIFEST.read_text().splitlines() if line]
    if {r['split'] for r in rows}!={'train','valid'} or len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('invalid full manifest')
    if {r['video_id'] for r in rows if r['split']=='train'} & {r['video_id'] for r in rows if r['split']=='valid'}:
        raise ValueError('source video split overlap')
    selected=select_diagnostic(rows)
    payload=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in selected)
    dp=base/'diagnostic64.jsonl'
    if dp.exists() and dp.read_text()!=payload:raise ValueError('diagnostic manifest changed')
    dp.write_text(payload)
    models={}
    for name in ['Qwen3-Omni-30B-A3B-Instruct','Qwen3-0.6B-Base','WavLM-Base-Plus','VideoMAE-Base']:
        for p in sorted((ROOT/'model'/name).iterdir()):
            if p.suffix in {'.json','.safetensors','.bin'}:
                st=p.stat();models[str(p)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    plan={'protocol':'video-source-v2','manifest_sha256':sha256(MANIFEST),
        'diagnostic_manifest_sha256':sha256(dp),'diagnostic':{'train_samples':64,'source_videos':64,
        'negative':22,'neutral':20,'positive':22,'subsets':['v','ta','tav'],'both_timing_policies':True},
        'teacher_features':{'subsets':['t','a','v','ta','tv','av','tav'],'windows':len(rows),
            'jobs':len(rows)*7,'policy':'sampled_fps_v2','recompute_all_subsets':True},
        'probe_seeds':list(PROBE_SEEDS),'student_teacher_seed':2026,'student_seeds':list(SEEDS),
        'student_modes':list(MODES),'student_teacher_subset':'tav',
        'student':{'video_layers':12,'rank':8,'alpha':16,'dropout':.05,'batch_size':8,
            'gradient_accumulation':1,'epochs':30,'patience':7,'learning_rate':1e-4,
            'diagnostic_video_swaps':5,'seeds_are_not_gated_on_pilot':True},
        'constraints':{'final_model_keeps_TAV':True,'ta_only_is_diagnostic':True,'c2_rerun':False,
            'official_test_evaluated':False,'hold_audio_and_video_spatial_preprocessing_fixed':True},
        'source_sha256':snapshot_sources(),'model_files':models,
        'model_identity_note':'Stat guards here; paired teacher uses one weight load; student runner additionally hashes its model weights.',
        'resource_policy':'Teacher alone on both GPUs; probes sequential; at most one student per GPU.'}
    pp=base/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan:raise ValueError('existing experiment plan differs')
    atomic_json(plan,pp)
    return plan


def wait_gpus(base,gpus,stage):
    consecutive=0
    while consecutive<2:
        checks={g:gpu_idle(g) for g in gpus}
        consecutive=consecutive+1 if all(x[0] for x in checks.values()) else 0
        status(base,status='waiting_for_gpu',stage=stage,gpus={g:x[1] for g,x in checks.items()},idle_checks=consecutive)
        if consecutive<2:time.sleep(15)


def run_step(base,plan,name,command,output,kind,gpus,validate):
    marker=base/f'{name}.done.json'
    if marker.exists():
        validate();return
    verify_sources(plan)
    wait_gpus(base,gpus,name)
    log_path=base/'logs'/f'{name}.log'
    with log_path.open('a') as log:
        env={**ENV,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))}
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        started=time.time()
        try:
            while child.poll() is None:
                status(base,status='running',stage=name,pid=child.pid,current_log=str(log_path),
                    elapsed_seconds=round(time.time()-started),progress=progress(output,kind))
                time.sleep(15)
            if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}; log: {log_path}')
        finally:
            if child.poll() is None:
                child.terminate();child.wait(timeout=60)
    validate()
    atomic_json({'stage':name,'completed_at':time.time()},marker)


def validate_features(output,plan):
    cfg=json.loads((output/'run_config.json').read_text())
    assert cfg['video_timing_policy']=='sampled_fps_v2'
    assert cfg['manifest_sha256']==plan['manifest_sha256']
    flags=np.load(output/'completed.npy',mmap_mode='r')
    features=np.load(output/'features.npy',mmap_mode='r')
    assert len(flags)==plan['teacher_features']['jobs'] and bool(flags.all())
    assert features.shape==(len(flags),2048)
    for start in range(0,len(flags),4096):assert np.isfinite(features[start:start+4096]).all()
    rows=[json.loads(x) for x in (output/'index.jsonl').read_text().splitlines()]
    assert len(rows)==len(flags) and {r['split'] for r in rows}=={'train','valid'}
    assert len({(r['sample_id'],r['subset']) for r in rows})==len(rows)


def validate_probe(output,features,seed):
    report=json.loads((output/'report.json').read_text())
    assert report['seed']==seed and report['official_test_evaluated'] is False
    assert Path(report['features']).resolve()==features.resolve()
    temperature=report['calibration']['after']['temperature']
    assert math.isfinite(temperature) and .05<=temperature<=20
    rows=[json.loads(x) for x in (output/'predictions.jsonl').read_text().splitlines()]
    assert {r['split'] for r in rows}=={'train','valid'}
    assert len(rows)==18197*7 and len({(r['parent_sample_id'],r['subset']) for r in rows})==len(rows)
    assert all(np.isfinite([r['probe_score'],*r['classification_logits']]).all() for r in rows)
    assert all(v['count']==1871 for v in report['metrics']['valid'].values())
    return report


def validate_student(output,mode,smoke=False):
    assert json.loads((output/'status.json').read_text())['status']=='complete'
    report=json.loads((output/'report.json').read_text())
    assert report['mode']==mode and report['official_test_evaluated'] is False
    assert report['video_trainable_parameters']==(589824 if mode=='video_lora' else 0)
    if not smoke:
        assert report['valid_metrics']['count']==1871
        assert len(report['diagnostics'])==(0 if mode=='ta_only' else 7)
    if mode=='video_lora':
        grads=json.loads((output/'video_gradient_audit.json').read_text())
        values=[v for k,v in grads.items() if 'lora_B' in k]
        assert len(values)==48 and all(v is not None and math.isfinite(v) and v>0 for v in values)
    return report


def student_command(base,mode,seed,output):
    probe=base/'probes/seed2026'
    command=[sys.executable,str(SCRIPTS/'train_video_adaptation_v2.py'),'--mode',mode,
        '--seed',str(seed),'--output',str(output),'--teacher-targets',str(probe/'predictions.jsonl'),
        '--teacher-probe-report',str(probe/'report.json'),'--teacher-subset','tav',
        '--video-layers','12','--batch-size','8','--device','cuda:0','--manifest',str(MANIFEST)]
    if (output/'run_config.json').exists():
        if (output/'last.pt').exists():
            command.append('--resume')
        else:
            # Initialization/first-epoch interruption has no optimizer checkpoint.
            archive=output.with_name(output.name+f'_interrupted_before_epoch1_{time.time_ns()}')
            output.rename(archive)
    return command


def run_students(base,plan):
    pending=[]
    for seed in SEEDS:
        for mode in MODES:
            output=base/'students'/f'{mode}_seed{seed}'
            if (output/'status.json').exists() and json.loads((output/'status.json').read_text()).get('status')=='complete':
                validate_student(output,mode)
            else:pending.append((mode,seed,output))
    active={}; idle={0:0,1:0}
    try:
        while pending or active:
            for gpu,job in list(active.items()):
                code=job['child'].poll()
                if code is not None:
                    job['log'].close()
                    if code:raise RuntimeError(f"student {job['name']} exited {code}: {job['log_path']}")
                    validate_student(job['output'],job['mode'])
                    del active[gpu]
            for gpu in [0,1]:
                if gpu in active or not pending:continue
                ready,_=gpu_idle(gpu)
                idle[gpu]=idle[gpu]+1 if ready else 0
                if idle[gpu]<2:continue
                verify_sources(plan)
                mode,seed,output=pending.pop(0)
                output.parent.mkdir(parents=True,exist_ok=True)
                name=f'{mode}_seed{seed}'
                log_path=base/'logs'/f'{name}.log';log=log_path.open('a')
                command=student_command(base,mode,seed,output)
                child=subprocess.Popen(command,cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':str(gpu)},
                    stdout=log,stderr=subprocess.STDOUT)
                active[gpu]={'child':child,'log':log,'output':output,'mode':mode,'name':name,'log_path':log_path}
                idle[gpu]=0
            status(base,status='running',stage='students',pending_runs=len(pending),
                active=[{'gpu':gpu,'run':j['name'],'pid':j['child'].pid,'log':str(j['log_path']),
                         'progress':progress(j['output'],'student')} for gpu,j in active.items()])
            if active or pending:time.sleep(15)
    finally:
        for job in active.values():
            if job['child'].poll() is None:job['child'].terminate()
        for job in active.values():
            try:job['child'].wait(timeout=60)
            finally:job['log'].close()


def summarize(base):
    runs={};comparisons={};perturbations={}
    for seed in SEEDS:
        for mode in MODES:
            directory=base/'students'/f'{mode}_seed{seed}'
            runs[f'{mode}_seed{seed}']=validate_student(directory,mode)
            for diagnostic in runs[f'{mode}_seed{seed}']['diagnostics']:
                # Existing comparator reads JSONL; create derived files in the analysis directory.
                analysis=base/'analysis';analysis.mkdir(exist_ok=True)
                path=analysis/f'{mode}_seed{seed}_{diagnostic}.jsonl'
                rows=json.loads((directory/f'diagnostic_{diagnostic}.json').read_text())
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                perturbations[f'{mode}_seed{seed}_{diagnostic}']=compare(directory/'predictions.jsonl',path)
        candidate=base/'students'/f'video_lora_seed{seed}'/'predictions.jsonl'
        for mode in ['frozen_video','ta_only']:
            comparisons[f'video_lora_minus_{mode}_seed{seed}']=compare(
                base/'students'/f'{mode}_seed{seed}'/'predictions.jsonl',candidate)
    result={'runs':runs,'comparisons':comparisons,'perturbation_comparisons':perturbations,
        'mean_mae':{m:float(np.mean([runs[f'{m}_seed{s}']['valid_metrics']['mae'] for s in SEEDS])) for m in MODES},
        'sample_sd_mae':{m:float(np.std([runs[f'{m}_seed{s}']['valid_metrics']['mae'] for s in SEEDS],ddof=1)) for m in MODES},
        'official_test_evaluated':False,'ta_only_is_diagnostic':True,'scope':'Validation-selected checkpoints; seed-wise cluster CIs, not a guarantee for unseen seeds.'}
    atomic_json(result,base/'summary.json')
    lines=['**Video source v2 — completed results**','',
        'All student modes use the same corrected TAV Probe seed2026 targets and its recorded temperature.',
        'TA-only is a diagnostic control; the final model requirement remains TAV. Official test was not evaluated.','',
        '| Mode | Seed | MAE | Pearson | Acc-2 |','|---|---:|---:|---:|---:|']
    for name,r in runs.items():
        m=r['valid_metrics'];lines.append(f"| {r['mode']} | {r['seed']} | {m['mae']:.6f} | {m['pearson']:.6f} | {m['acc2_nonzero']:.6f} |")
    lines+=['','| Comparison | Delta MAE | Video-cluster 95% CI |','|---|---:|---|']
    for name,c in comparisons.items():
        lo,hi=c['cluster_ci95'];lines.append(f"| {name} | {c['delta_mae']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['',f'Machine-readable results: {base / "summary.json"}',
        f'Teacher paired diagnostic: {base / "teacher_pair/summary.json"}',
        'Wall-clock efficiency is not an exclusive-device benchmark; two student jobs may decode media concurrently.','']
    path=ROOT/'docs/video_source_v2_results.md';tmp=path.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(lines));os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();base=args.output.resolve();base.mkdir(parents=True,exist_ok=True)
    with (base/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=prepare(base)
            if args.prepare_only:
                status(base,status='prepared',stage='preflight');return
            if shutil.disk_usage(base).free < 20*1024**3:
                raise RuntimeError('at least 20 GiB free disk required before starting')
            pair=base/'teacher_pair'
            def pair_valid():
                summary=json.loads((pair/'summary.json').read_text());assert summary['jobs']==192
                assert summary['hidden']['ta']['max_abs_change']==0
            run_step(base,plan,'teacher_pair',[sys.executable,str(SCRIPTS/'diagnose_teacher_video_pair.py'),
                '--manifest',str(base/'diagnostic64.jsonl'),'--output',str(pair)],pair,'diagnostic',[0,1],pair_valid)
            features=base/'teacher_features'
            if features.exists() and not all((features/n).exists() for n in ['run_config.json','index.jsonl','features.npy','completed.npy']):
                features.rename(features.with_name(f'teacher_features_interrupted_init_{time.time_ns()}'))
            run_step(base,plan,'teacher_features',[sys.executable,str(SCRIPTS/'extract_teacher_probe_features.py'),
                '--manifest',str(MANIFEST),'--output-dir',str(features),'--video-timing-policy','sampled_fps_v2'],
                features,'features',[0,1],lambda:validate_features(features,plan))
            for seed in PROBE_SEEDS:
                out=base/'probes'/f'seed{seed}'
                run_step(base,plan,f'probe_seed{seed}',[sys.executable,str(SCRIPTS/'train_teacher_probe_v2.py'),
                    '--features',str(features),'--output',str(out),'--seed',str(seed),'--device','cuda:0'],
                    out,'probe',[0],lambda out=out,seed=seed:validate_probe(out,features,seed))
            smoke=base/'student_smoke'
            command=student_command(base,'video_lora',13,smoke)+['--limit-per-split','8','--epochs','1','--no-diagnostics','--num-workers','0']
            run_step(base,plan,'student_smoke',command,smoke,'student',[1],lambda:validate_student(smoke,'video_lora',True))
            run_students(base,plan)
            status(base,status='running',stage='summarizing')
            summarize(base)
            status(base,status='complete',stage='complete',summary=str(base/'summary.json'),
                report=str(ROOT/'docs/video_source_v2_results.md'),c2_rerun=False)
        except BaseException as exc:
            status(base,status='failed',stage='stopped',error=repr(exc),resume='systemctl --user restart rdid-video-source-v2.service')
            raise


if __name__=='__main__':
    main()
