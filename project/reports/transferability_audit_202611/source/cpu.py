"""0B/0C on existing predictions only, train/valid separated, no fitting."""
import itertools
import sys
import numpy as np
from common import *
sys.path[:0]=[str(ROOT/'project/src'),str(ROOT/'project/sims_native/src')]
from rdid_mosei.metrics import sentiment_metrics as english_metrics
from rdid_sims.metrics import sentiment_metrics as chinese_metrics

def pearson(x,y):
    x=np.asarray(x);y=np.asarray(y)
    return float(np.corrcoef(x,y)[0,1]) if x.std()>1e-12 and y.std()>1e-12 else None

def metrics(d,y,p):
    m=(chinese_metrics if d.startswith('sims') else english_metrics)(y,p)
    return {k:float(v) if isinstance(v,(float,np.floating)) and np.isfinite(v) else (None if isinstance(v,(float,np.floating)) else v) for k,v in m.items()}

def matrix():
    return np.array([[((-1)**(len(t)-len(s))) if set(s)<=set(t) else 0 for s in NAMES] for t in NAMES],dtype=np.float64)

def icc(a):
    # ICC(A,1), two-way random, absolute agreement, single measurement.
    n,k=a.shape;grand=a.mean();row=a.mean(1);col=a.mean(0)
    msr=k*((row-grand)**2).sum()/(n-1)
    msc=n*((col-grand)**2).sum()/(k-1)
    mse=((a-row[:,None]-col[None,:]+grand)**2).sum()/((n-1)*(k-1))
    denom=msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/denom) if abs(denom)>1e-12 else None

def teacher_arrays(d,p):
    a=read(p['assets']);ensemble=rows(a['ensemble_targets']);ids=[r['parent_sample_id'] for r in ensemble]
    labels=np.array([r['sentiment'] for r in ensemble]);splits=np.array([r['split'] for r in ensemble]);train=splits=='train'
    raw=[];anchors=[]
    for probe in p['probes']:
        index={(r['parent_sample_id'],r['subset']):r for r in rows(Path(probe['directory'])/'predictions.jsonl')}
        values=np.array([[index[(key,s)]['probe_score'] for s in NAMES] for key in ids])
        raw.append(values);anchors.append(float(values[train,-1].mean()))
    raw=np.asarray(raw);anchors=np.asarray(anchors);M=matrix()
    coords=(raw-anchors[:,None,None])@M.T
    assert np.isfinite(raw).all() and np.isfinite(coords).all()
    np.testing.assert_allclose(raw.mean(0),np.array([r['subset_score_mean'] for r in ensemble]),atol=1e-6,rtol=0)
    it={r['parent_sample_id']:r for r in rows(a['interaction_targets'])}
    np.testing.assert_allclose(coords.mean(0),np.array([it[key]['mean'] for key in ids]),atol=2e-6,rtol=0)
    np.testing.assert_allclose(coords.var(0,ddof=1),np.array([it[key]['variance'] for key in ids]),atol=2e-6,rtol=0)
    np.testing.assert_allclose(anchors.mean(),a['student_empty_baseline'],atol=1e-7)
    return dict(ids=ids,labels=labels,splits=splits,raw=raw,coords=coords,anchors=anchors,ensemble=ensemble)

def main():
    plan=read(OUT/'protocol/plan.json');quality=[];agreement=[];reliability=[];probescores=[];details={};variance_error=[]
    for d,p in plan['protocols'].items():
        t=teacher_arrays(d,p);raw=t['raw'];coords=t['coords'];details[d]={}
        for split in ['train','valid']:
            mask=t['splits']==split;y=t['labels'][mask];details[d][split]={}
            for pi,probe in enumerate(p['probes']):
                for k,name in enumerate(NAMES):
                    probescores.append(dict(dataset=d,split=split,probe_seed=probe['seed'],target=name.upper(),**metrics(d,y,raw[pi,mask,k])))
            for k,name in enumerate(NAMES):
                variance=raw[:,mask,k].var(0,ddof=1);error=np.abs(raw[:,mask,k].mean(0)-y)
                variance_error.append(dict(dataset=d,split=split,subset=name,n=len(y),probe_variance_vs_absolute_task_error_pearson=pearson(variance,error),interpretation='descriptive agreement/error association; shared hidden cache and label-strength confounding remain; not used as adaptive weights'))
                for target,values in [('seed2026',raw[0]),('ensemble3',raw.mean(0))]:
                    pred=values[mask,k]
                    quality.append(dict(dataset=d,split=split,teacher_target=target,subset=name,n=len(y),mae=float(np.abs(pred-y).mean()),pearson=pearson(pred,y),bias=float((pred-y).mean()),prediction_mean=float(pred.mean()),prediction_variance=float(pred.var(ddof=1)),**{f'task_{key}':val for key,val in metrics(d,y,pred).items()}))
                for kind,array in [('subset',raw),('interaction',coords)]:
                    samples=array[:,mask,k].T
                    pair=[(i,j) for i,j in itertools.combinations(range(len(p['probes'])),2)]
                    corrs=[pearson(samples[:,i],samples[:,j]) for i,j in pair]
                    entry=dict(dataset=d,split=split,space=kind,coordinate=name,n=len(y),probe_count=len(p['probes']),
                        mean_probe_variance=float(samples.var(1,ddof=1).mean()),pairwise_mae=float(np.mean([np.abs(samples[:,i]-samples[:,j]).mean() for i,j in pair])),
                        pairwise_pearson_mean=float(np.mean([c for c in corrs if c is not None])) if any(c is not None for c in corrs) else None,
                        icc_absolute_single=icc(samples),sign_agreement=float(((samples>=0).all(1)|(samples<=0).all(1)).mean()),
                        signal_variance=float(samples.mean(1).var(ddof=1)),shared_teacher_hidden_cache=True)
                    agreement.append(entry)
                    if kind=='subset':reliability.append(entry.copy())
            details[d][split]=dict(n=int(mask.sum()),probe_anchors=t['anchors'].tolist(),ensemble_train_anchor=float(t['anchors'].mean()),
                coordinate_std=coords.mean(0)[mask].std(0,ddof=1).tolist(),coordinate_mae_signal=np.abs(coords.mean(0)[mask]).mean(0).tolist())
        print(json.dumps(dict(dataset=d,status='complete',parents=len(t['ids']))),flush=True)
    table('teacher_subset_quality.csv',quality);table('probe_agreement.csv',agreement)
    table('teacher_reliability_by_subset.csv',reliability)
    table('probe_agreement_by_coordinate.csv',[r for r in agreement if r['space']=='interaction'])
    table('probe_task_quality.csv',probescores)
    table('probe_variance_task_error.csv',variance_error)
    save(OUT/'protocol/teacher_statistics.json',details)
    checkpoint_metrics=[]
    for c in plan['cases']:
        history=read(Path(c['run'])/'history.json')
        if isinstance(history,dict):history=history.get('epochs',history.get('history'))
        for r in history:checkpoint_metrics.append(dict(case=c['id'],dataset=c['dataset'],method=c['method'],seed=c['seed'],epoch=r['epoch'],selected_mae=r['epoch']==c.get('epoch'),**r['valid_metrics']))
    table('checkpoint_validation_metrics.csv',checkpoint_metrics)
    save(OUT/'status/cpu.json',dict(status='complete',quality_rows=len(quality),agreement_rows=len(agreement),official_test_predictions_used=False,teacher_target_reconstruction_passed=True))

if __name__=='__main__':main()
