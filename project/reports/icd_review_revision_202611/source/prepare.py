"""Freeze Phase A inputs and jobs. No model training; no test predictions."""
import collections
import shutil
import subprocess
from common import *

def main():
    assert not (OUT/'protocol/plan.json').exists(), 'Frozen plan already exists'
    for x in ['protocol','status','logs','gradients','source']:(OUT/x).mkdir(parents=True,exist_ok=True)
    plan=read(OLD/'protocol/plan.json')
    inventory=read(OLD/'protocol/retained_checkpoint_manifest.json')
    plan.update(schema='phaseA-v1',phaseA_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),phaseA_git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),prior_plan_sha256=sha(OLD/'protocol/plan.json'))
    provenance=[]
    for c in plan['cases']:
        assert sha(c['checkpoint'])==c['checkpoint_sha256'],c['id']
        for split in ['train','valid']:
            p=OLD/f'predictions/{c["id"]}_{split}.jsonl';rep=read(p.with_suffix('.report.json'))
            assert sha(p)==rep['sha256'] and rep['checkpoint_sha256']==c['checkpoint_sha256']
            provenance.append(dict(path=str(p),sha256=rep['sha256'],case=c['id'],split=split,checkpoint_sha256=c['checkpoint_sha256']))
    for d,p in plan['protocols'].items():
        for probe in p['probes']:assert sha(Path(probe['directory'])/'predictions.jsonl')==probe['prediction_sha256']
    n2run=ROOT/'outputs/experiments/mosi_method_search_20261110_v1/runs/N2_no_ce_l030'
    cfg=read(n2run/'run_config.json');assets=read(cfg['assets'])
    assert cfg['task_ce']==0 and cfg['lambda_interaction']==.3 and cfg['kd_kl']==1 and cfg['variant']=='plain'
    save(OUT/'protocol/n2_run_config.json',cfg);save(OUT/'protocol/n2_assets.json',assets)
    n2ck=n2run/'best_valid_mae.pt'
    plan['n2']=dict(run=str(n2run),checkpoint=str(n2ck),checkpoint_sha256=sha(n2ck),config_sha256=sha(n2run/'run_config.json'),assets_sha256=sha(cfg['assets']),definition='SmoothL1 task regression + ensemble3 TAV SmoothL1 KD + ensemble3 calibrated KL + 0.30 mean SmoothL1 Mobius errors; task CE=0; seed13; historical epoch23; not a 20-epoch result')
    stage=[];jobs=[]
    for c in plan['cases']:
        if c['seed']!=13:continue
        retained={}
        for v in inventory:
            if v['case']==c['id']:retained.setdefault(v['epoch'],v)
        for tag,low,high in [('early',1,6),('middle',7,13),('late',14,1000)]:
            available=[v for e,v in retained.items() if low<=e<=high]
            stage.append(dict(case=c['id'],dataset=c['dataset'],stage=tag,available_epochs=','.join(str(v['epoch']) for v in available),status='available' if available else 'missing_weights; no retraining permitted'))
        for epoch,v in sorted(retained.items()):
            assert sha(v['path'])==v['sha256']
            item=c.copy();st=Path(v['path']).stat();item.update(id=c['id']+f'_epoch{epoch:03d}',checkpoint=v['path'],epoch=epoch,checkpoint_sha256=v['sha256'],checkpoint_bytes=st.st_size,checkpoint_mtime_ns=st.st_mtime_ns,stage='early' if epoch<=6 else 'middle' if epoch<=13 else 'late')
            jobs.append(item)
    n2case=next(c for c in plan['cases'] if c['id']=='mosi_uniform_interaction_seed13').copy()
    stat=n2ck.stat();n2case.update(id='mosi_historical_N2_seed13_epoch023',method='historical_N2',checkpoint=str(n2ck),epoch=23,checkpoint_sha256=plan['n2']['checkpoint_sha256'],checkpoint_bytes=stat.st_size,checkpoint_mtime_ns=stat.st_mtime_ns,stage='late',config=mapped(cfg))
    jobs.append(n2case)
    plan['gradient_cases']=jobs
    plan['analysis_protocol']=dict(test_predictions_used=False,optimizer_steps=0,offline_analyzer='standardized ridge, fixed capacity <=7, intercept unpenalized',outer_group_folds=5,inner_group_folds=3,group_key='video_id; all windows aggregated to parent first',alphas=[.01,.1,1.,10.,100.],hyperparameter_metric=dict(response='MSE',task='MAE'),validation='external only; also report subset with source videos absent from training',utility_empty='train-fold target mean',tav_response_caveat='TAV response as input yields near-trivial reconstruction; main exclusion-of-TAV sensitivity also reported',gradient_microbatches=6,gradient_batch_size=1,gradient_mode='eval, dropout disabled, autograd only',stage_bins='early1-6; middle7-13; late14+; missing weights explicit')
    save(OUT/'protocol/plan.json',plan);save(OUT/'protocol/prediction_manifest.json',provenance)
    table('gradient_compatibility/checkpoint_stage_coverage.csv',stage)
    shards={'Q':['mosi'],'Qiii':['mosei','sims_v2'],'A100':['sims']}
    save(OUT/'protocol/schedule.json',dict(shards=shards,parallelism_per_gpu=1,a100_host_limit_gib=32,gpu_cap_mib=12288,data_loader_workers=0,automatic_finalizer=True,parameter_updates=0))
    # Copy protocol audit evidence, retaining the original files and provenance.
    dest=OUT/'phase0_5/protocol_audit';dest.mkdir(parents=True,exist_ok=True)
    for name in ['teacher_manifest.json','student_manifest.json','split_fingerprints.json','sims_cross_version_overlap.json','retained_checkpoint_manifest.json','feature_and_base_model_fingerprints.json']:
        shutil.copy2(OLD/'protocol'/name,dest/name)
    table('protocol_audit/checkpoint_hashes.csv',[dict(case=c['id'],checkpoint=c['checkpoint'],sha256=c['checkpoint_sha256'],epoch=c['epoch'],seed=c['seed'],dataset=c['dataset']) for c in plan['cases']]+[dict(case='historical_N2',checkpoint=str(n2ck),sha256=plan['n2']['checkpoint_sha256'],epoch=23,seed=13,dataset='mosi')])
    table('protocol_audit/student_run_manifest.csv',[dict(case=c['id'],dataset=c['dataset'],method=c['method'],seed=c['seed'],checkpoint=c['checkpoint'],epoch=c['epoch'],config_sha256=c['config_sha256'],config_json=json.dumps(c['config'],ensure_ascii=False),checkpoint_selection=c['selection'],git_commit=c['config'].get('git_commit','not recorded in historical run config'),audit_git_commit=plan['phaseA_git_commit']) for c in plan['cases']])
    save(OUT/'status/prepare.json',dict(status='complete',prediction_files=len(provenance),gradient_checkpoints=len(jobs),missing_stages=[v for v in stage if not v['available_epochs']],test_predictions_used=False))
    print(json.dumps(dict(cases=len(plan['cases']),gradient_checkpoints=len(jobs),shards=shards)),flush=True)

if __name__=='__main__':main()
