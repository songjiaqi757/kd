"""Small fixed-Train-window checkpoint trajectories; no temporal causality claims."""
import collections
import numpy as np
from common import *
from legacy_cpu import pearson

def run(plan):
    checkpoints=[];records=[];deltas=[];associations=[];paired=collections.defaultdict(list)
    for c in plan['gradient_cases']:
        p=OUT/f'trajectories/{c["id"]}.json'
        if not p.exists():continue
        r=read(p);assert len(r['samples'])==6 and r['parameter_updates']==0 and r['parameter_versions_unchanged']
        assert r['checkpoint_sha256']==c['checkpoint_sha256']
        scale=np.array(read(OLD/'protocol/teacher_statistics.json')[c['dataset']]['train']['coordinate_std']);scale=np.maximum(scale,1e-12)
        run=Path(plan['n2']['run']) if c['method']=='historical_N2' else Path(c['run'])
        h=read(run/'history.json');h=h.get('epochs',h.get('history')) if isinstance(h,dict) else h
        validmae=next((v['valid_metrics']['mae'] for v in h if v['epoch']==c['epoch']),None)
        for sample in r['samples']:
            err=np.array(sample['student_coordinates'])-sample['teacher_coordinates'];task_error=abs(sample['student_subsets'][-1]-sample['label'])
            for q,idx in [(1,[0,1,2]),(2,[3,4,5]),(3,[6])]:
                entry=dict(dataset=c['dataset'],method=c['method'],seed=c['seed'],checkpoint=c['id'],epoch=c['epoch'],sample_id=sample['sample_id'],source_video_id=sample['source_video_id'],order=q,coordinate_mse=float(np.mean(err[idx]**2)),normalized_coordinate_mse=float(np.mean((err[idx]/scale[idx])**2)),task_absolute_error=task_error,historical_valid_mae=validmae,scope='six fixed Train windows; retained checkpoint selection bias')
                records.append(entry);paired[(c['dataset'],c['method'],q,sample['sample_id'])].append(entry)
        checkpoints.append(c['id'])
    for (d,m,q,sample),entries in paired.items():
        entries=sorted(entries,key=lambda r:r['epoch'])
        for before,after in zip(entries[:-1],entries[1:]):deltas.append(dict(dataset=d,method=m,order=q,sample_id=sample,epoch_before=before['epoch'],epoch_after=after['epoch'],fidelity_improvement=before['normalized_coordinate_mse']-after['normalized_coordinate_mse'],task_improvement=before['task_absolute_error']-after['task_absolute_error'],historical_valid_mae_improvement=None if before['historical_valid_mae'] is None or after['historical_valid_mae'] is None else before['historical_valid_mae']-after['historical_valid_mae']))
    buckets=collections.defaultdict(list)
    for r in deltas:buckets[(r['dataset'],r['method'],r['order'],r['epoch_before'],r['epoch_after'])].append(r)
    for key,items in buckets.items():
        x=[r['fidelity_improvement'] for r in items];y=[r['task_improvement'] for r in items]
        associations.append(dict(zip(['dataset','method','order','epoch_before','epoch_after'],key),n_fixed_train_windows=len(items),pearson_fidelity_task_improvement=pearson(x,y),mean_fidelity_improvement=float(np.mean(x)),mean_task_improvement=float(np.mean(y)),historical_valid_mae_improvement=items[0]['historical_valid_mae_improvement'],scope='descriptive six-window paired trajectory; nonuniform retained epochs; not causal/confirmatory'))
    table('interaction_order_utility/checkpoint_order_trajectory.csv',records);table('interaction_order_utility/checkpoint_order_improvement.csv',deltas);table('interaction_order_utility/checkpoint_fidelity_task_association.csv',associations)
    save(OUT/'status/trajectory_analysis.json',dict(status='complete' if len(checkpoints)==len(plan['gradient_cases']) else 'partial',checkpoints=len(checkpoints),expected=len(plan['gradient_cases']),sample_scope='six pre-frozen Train windows per checkpoint; no full-split temporal inference',parameter_updates=0))
    return len(checkpoints),associations
