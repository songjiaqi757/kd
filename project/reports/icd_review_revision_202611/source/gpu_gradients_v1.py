"""Read-only seven-subset inference and train-only autograd; no optimizer exists."""
import argparse
import gc
import math
import os
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import *

def parameter_group(name):
    if name.startswith(('fusion.regressor.','fusion.classifier.')):return 'prediction_head'
    if 'lora_' in name:
        for modality in ['text','audio','video']:
            if f'{modality}_encoder' in name:return modality+'_lora'
    if name.startswith('fusion.') or name.startswith('pools.'):return 'fusion'
    return 'other'

def vectors(loss,named):
    grads=torch.autograd.grad(loss,[p for _,p in named],retain_graph=True,allow_unused=True)
    groups={k:[] for k in ['fusion','text_lora','audio_lora','video_lora','prediction_head','other']}
    for (n,p),g in zip(named,grads):
        groups[parameter_group(n)].append(torch.zeros(p.numel()) if g is None else g.detach().reshape(-1).float().cpu())
    return {k:torch.cat(v).double() if v else torch.zeros(0,dtype=torch.float64) for k,v in groups.items()}

def geom(left,right):
    result={};sums=np.zeros(3)
    for k,a in left.items():
        b=right[k];aa=float(a@a);bb=float(b@b);ab=float(a@b);sums+=[aa,bb,ab]
        result[k]=dict(base_norm=math.sqrt(aa),aux_norm=math.sqrt(bb),cosine=ab/math.sqrt(aa*bb) if aa*bb>1e-24 else None,norm_ratio=math.sqrt(bb/aa) if aa>1e-24 else None)
    aa,bb,ab=sums;result['all']=dict(base_norm=math.sqrt(aa),aux_norm=math.sqrt(bb),cosine=float(ab/math.sqrt(aa*bb)) if aa*bb>1e-24 else None,norm_ratio=math.sqrt(bb/aa) if aa>1e-24 else None)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--mode',choices=['gradients','export','both'],default='both');parser.add_argument('--seeds',nargs='+',type=int,default=[13,42,2026]);parser.add_argument('--cap-mib',type=int,default=16384);parser.add_argument('--batch-size',type=int,default=4);parser.add_argument('--worker',default='a100');parser.add_argument('--export-cap-mib',type=int);arg=parser.parse_args()
    if arg.mode=='export':
        arg.batch_size=32 if arg.dataset=='mosei' else 16
        arg.cap_mib=arg.export_cap_mib or (32768 if arg.dataset=='mosei' else 16384)
    torch.set_num_threads(2)
    plan=read(OUT/'protocol/plan.json');p=plan['protocols'][arg.dataset];a=read(p['assets'])
    assert not plan['issues']
    native=arg.dataset.startswith('sims');package='rdid_sims' if native else 'rdid_mosei'
    sys.path[:0]=[str(ROOT/('project/sims_native/src' if native else 'project/src')),str(ROOT/('project/sims_native/scripts' if native else 'project/scripts'))]
    import train_main_table_kd as train
    import train_student_baseline as base
    from transformers import AutoTokenizer,AutoFeatureExtractor,AutoImageProcessor
    from importlib import import_module
    tav=import_module(package+'.tav_distillation').TAVDistillationStudent
    interaction=import_module(package+'.interaction')
    device=torch.device('cuda:0');props=torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(arg.cap_mib/(props.total_memory/1024**2),device)
    train.seed_everything(13,True)
    candidates=[c for c in plan['gradient_cases'] if c['dataset']==arg.dataset and c['seed'] in arg.seeds]
    config=candidates[0]['config'].copy();config['method']='uniform_interaction';config.setdefault('cmad_tau',.2)
    args=SimpleNamespace(**config)
    for key in ['text_model','audio_model','video_model']:setattr(args,key,Path(getattr(args,key)))
    model=train.build_model(args,a,device)
    tokenizer=AutoTokenizer.from_pretrained(args.text_model,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    audio=AutoFeatureExtractor.from_pretrained(args.audio_model,local_files_only=True)
    video=AutoImageProcessor.from_pretrained(args.video_model,local_files_only=True,use_fast=False)
    collator=train.MainTableCollator(tokenizer,audio,None)
    n2ens={r['parent_sample_id']:r for r in rows(read(OUT/'protocol/n2_assets.json')['ensemble_targets'])} if arg.dataset=='mosi' else {}
    data=rows(a['manifest']);single={r['parent_sample_id']:r for r in rows(a['teacher_targets']) if r['subset']=='tav'}
    ens={r['parent_sample_id']:r for r in rows(a['ensemble_targets'])};it={r['parent_sample_id']:r for r in rows(a['interaction_targets'])}
    for r in data:
        key=r['parent_sample_id'];target=single[key]
        r.update(teacher_score=target['probe_score'],teacher_logits=target['classification_logits'],teacher_subset_scores=ens[key]['subset_score_mean'],teacher_ensemble_probabilities=ens[key]['tav_calibrated_probability_mean'],interaction_mean=it[key]['mean'],interaction_variance=it[key]['variance'],weight_interaction_mean=it[key]['mean'],weight_interaction_variance=it[key]['variance'])
    base.add_window_weights(data)
    index={r['sample_id']:r for r in data};selected=[index[r['sample_id']] for r in p['gradient_samples']]
    diagnostic_dataset=train.VideoDataset(selected,video,mode='video_lora')
    batches=[collator([diagnostic_dataset[i]]) for i in range(len(selected))]
    input_hashes=[]
    for batch in batches:
        h=hashlib.sha256()
        for key,value in sorted(batch['inputs'].items()):
            if value is not None:h.update(key.encode()+str((tuple(value.shape),str(value.dtype))).encode()+value.contiguous().view(torch.uint8).numpy().tobytes())
        input_hashes.append(dict(sample_id=batch['rows'][0]['sample_id'],sha256=h.hexdigest()))
    input_path=OUT/f'protocol/{arg.dataset}_gradient_inputs.json'
    if input_path.exists():assert read(input_path)==input_hashes,'Fixed diagnostic tensor inputs changed across workers'
    else:save(input_path,input_hashes)
    # Deterministic duration buckets avoid padding every WavLM batch to 30 seconds.
    # All methods see identical windows/weights; eval outputs are restored by parent ID.
    inference_rows={s:sorted([r for r in data if r['split']==s],key=lambda r:(-float(r['window_duration']),r['sample_id'])) for s in ['train','valid']}
    loaders={s:DataLoader(train.VideoDataset(inference_rows[s],video,mode='video_lora'),batch_size=arg.batch_size,shuffle=False,num_workers=2,prefetch_factor=1,multiprocessing_context='spawn',collate_fn=collator,pin_memory=True) for s in ['train','valid']}
    named=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    M=interaction.mobius_matrix(dtype=torch.float32).to(device)
    for case in candidates:
        checkpoint=Path(case['checkpoint']);stat=checkpoint.stat()
        assert (stat.st_size,stat.st_mtime_ns)==(case['checkpoint_bytes'],case['checkpoint_mtime_ns'])
        assert sha(checkpoint)==case['checkpoint_sha256']
        state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        assert state['epoch']==case['epoch'], f'Wrong validation-selected epoch: {case["id"]}, {state["epoch"]} != {case["epoch"]}'
        train.load_checkpoint_state(model,state['model']);del state;gc.collect()
        model.eval();model.zero_grad(set_to_none=True)
        versions={n:v._version for n,v in model.named_parameters()};torch.cuda.reset_peak_memory_stats(device);started=time.time()
        provenance=dict(case=case['id'],dataset=arg.dataset,method=case['method'],seed=case['seed'],epoch=case['epoch'],stage=case['stage'],checkpoint_sha256=case['checkpoint_sha256'],device=props.name,worker=arg.worker,teacher_target='original FullKD single probe2026; subset/ICD ensemble3; N2 objective uses exact historical ensemble3 TAV score and q',inference_batching='descending window duration; sample_id tie break; common across methods',audit_script_sha256=sha(__file__),parameter_counts={g:sum(v.numel() for n,v in named if parameter_group(n)==g) for g in ['fusion','text_lora','audio_lora','video_lora','prediction_head','other']},trainable_parameters=sum(v.numel() for n,v in named))
        gradpath=OUT/f'gradients/{case["id"]}.json'
        if arg.mode in ['both','gradients'] and not gradpath.exists():
            results=[]
            for bi,batch in enumerate(batches):
                train.seed_everything(130013+bi,True)
                save(OUT/f'status/{arg.worker}.json',dict(status='running',phase='0F',case=case['id'],sample=bi+1,pid=os.getpid()))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    outputs=tav.forward(model,**train.batch_inputs(batch,device),all_subsets=True)
                    o=outputs['tav'];weights=batch['weights'].to(device)
                    reg=train.weighted_mean(F.smooth_l1_loss(o['regression'].float(),batch['sentiment'].to(device),reduction='none'),weights)
                    ce=train.weighted_mean(F.cross_entropy(o['classification_logits'].float(),batch['classes'].to(device),reduction='none'),weights)
                    kdreg=train.regression_kd(o,batch['teacher_scores'].to(device),weights)
                    prob=torch.softmax(batch['teacher_logits'].to(device).float()/(a['teacher_temperature']*a['kd_temperature']),-1)
                    kdkl=train.weighted_mean(train.probability_kd_per_sample(o['classification_logits'],prob,a['kd_temperature']),weights)
                    task=reg+a['alpha_ce']*ce;kd=kdreg+kdkl;baseloss=task+kd
                    values=torch.stack([outputs[s]['regression'].float() for s in NAMES],-1)
                with torch.autocast('cuda',enabled=False):
                    coord=interaction.mobius_transform(values.float(),a['student_empty_baseline'])-batch['interaction_mean'].to(device).float()
                    losses=(F.smooth_l1_loss(coord,torch.zeros_like(coord),reduction='none')*weights[:,None]).sum(0)/weights.sum()
                    icd=losses.mean()
                subset=train.weighted_mean(F.smooth_l1_loss(values,batch['teacher_subset_scores'].to(device),reduction='none').mean(-1),weights)
                objectives={'task':task,'full_kd_aux':kd,'subset7_aux':subset,'icd_aux':icd,'full_kd_total':baseloss,'subset7_total':baseloss+subset,'icd_total':baseloss+icd}
                if arg.dataset=='mosi':
                    target=n2ens[batch['rows'][0]['parent_sample_id']]
                    q=torch.tensor([target['tav_kd_probability_mean']],device=device,dtype=torch.float32)
                    n2reg=train.regression_kd(o,torch.tensor([target['subset_score_mean'][-1]],device=device),weights)
                    n2kl=train.weighted_mean(train.probability_kd_per_sample(o['classification_logits'],q,a['kd_temperature']),weights)
                    objectives.update(task_no_ce=reg,n2_aux=n2reg+n2kl+.3*icd,n2_total=reg+n2reg+n2kl+.3*icd)
                allvectors={label:vectors(term,named) for label,term in objectives.items()}
                comparisons={}
                refs={k:allvectors[k] for k in ['task','full_kd_aux']}
                if arg.dataset=='mosi':refs['task_no_ce']=allvectors['task_no_ce']
                for label,g in allvectors.items():
                    comparisons[label]={key:geom(ref,g) for key,ref in refs.items()}
                results.append(dict(sample_id=batch['rows'][0]['sample_id'],losses=dict(task=float(task.detach()),kd=float(kd.detach()),base=float(baseloss.detach()),icd=float(icd.detach()),coordinates=losses.detach().cpu().tolist()),geometry=comparisons))
                del refs,allvectors,objectives,outputs,o,values,coord,losses,icd,subset,task,kd,baseloss,reg,ce,kdreg,kdkl
                if arg.dataset=='mosi':del n2reg,n2kl
                gc.collect();torch.cuda.empty_cache()
                print(json.dumps(dict(case=case['id'],phase='gradient',sample=bi+1)),flush=True)
            assert versions=={n:v._version for n,v in model.named_parameters()}
            assert all(v.grad is None for n,v in named)
            save(gradpath,dict(status='complete',**provenance,samples=results,mode='eval/autograd; dropout disabled; FP32 Mobius; microbatch=1',base_definition='SmoothL1 task + alpha_ce CE + seed2026 TAV SmoothL1 KD + calibrated KL; unit KD coefficients',icd_definition='mean of seven SmoothL1 anchored interaction errors; raw unweighted; lambda is not applied',parameter_updates=0,parameter_versions_unchanged=True,peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,elapsed_seconds=time.time()-started))
        if arg.mode in ['both','export']:
            for split in ['valid','train']:
                dest=OUT/f'predictions/{case["id"]}_{split}.jsonl'
                if dest.exists():continue
                records=[]
                with torch.inference_mode():
                    for bi,batch in enumerate(loaders[split]):
                        with torch.autocast('cuda',dtype=torch.bfloat16):outputs=tav.forward(model,**train.batch_inputs(batch,device),all_subsets=True)
                        preds={s:outputs[s]['regression'].float().cpu().tolist() for s in NAMES}
                        logits=outputs['tav']['classification_logits'].float().cpu().tolist()
                        for i,r in enumerate(batch['rows']):records.append(dict(parent_sample_id=r['parent_sample_id'],split=r['split'],target_sentiment=r['sentiment'],class_7_index=r['class_7_index'],class_5_index=r.get('class_5_index',r['class_7_index']),prediction=preds['tav'][i],classification_logits=logits[i],aggregation_weight=r['aggregation_weight'],subset_predictions={s:preds[s][i] for s in NAMES}))
                        if bi%25==0:
                            save(OUT/f'status/{arg.worker}.json',dict(status='running',phase='0D_export',case=case['id'],split=split,batches=bi+1,total_batches=len(loaders[split]),pid=os.getpid()))
                            print(json.dumps(dict(case=case['id'],phase='export',split=split,batch=bi+1,total=len(loaders[split]))),flush=True)
                utterances=base.aggregate_windows(records)
                assert all(np.isfinite(list(r['subset_predictions'].values())).all() for r in utterances)
                replay={}
                if split=='valid':
                    metric_fn=import_module(package+'.metrics').sentiment_metrics
                    measured=metric_fn([r['target_sentiment'] for r in utterances],[r['prediction'] for r in utterances])
                    delta=abs(measured['mae']-case['valid_metrics']['mae'])
                    replay=dict(validation_metrics=measured,historical_validation_metrics=case['valid_metrics'],validation_mae_abs_delta=delta,validation_mae_tolerance=.02)
                    assert delta<=.02, f'Validation replay mismatch: {case["id"]}: {delta}'
                jsonl(dest,utterances)
                save(dest.with_suffix('.report.json'),dict(status='complete',**provenance,**replay,split=split,parents=len(utterances),sha256=sha(dest),batch_size=arg.batch_size,data_loader_workers=2,prefetch_factor=1,parameter_updates=0,parameter_versions_unchanged=versions=={n:v._version for n,v in model.named_parameters()},peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,elapsed_seconds=time.time()-started))
                print(json.dumps(dict(case=case['id'],phase='export_complete',split=split,parents=len(utterances))),flush=True)
        assert sha(checkpoint)==case['checkpoint_sha256']
        after=checkpoint.stat();assert (after.st_size,after.st_mtime_ns)==(case['checkpoint_bytes'],case['checkpoint_mtime_ns'])
    save(OUT/f'status/{arg.worker}.json',dict(status='complete',dataset=arg.dataset,mode=arg.mode,cases=len(candidates),parameter_updates=0))

if __name__=='__main__':main()
