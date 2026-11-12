"""A1/A2/A3/A4/A6, exclusively frozen train/validation arrays; numpy ridge only."""
import argparse
import collections
import itertools
import math
import time
import numpy as np
from common import *
from legacy_cpu import teacher_arrays, matrix, icc, pearson, metrics

def emit(section,name,data):table(f'{section}/by_dataset/{DATASET}/{name}.csv',data)
def note(section,text):
    p=OUT/f'phase0_5/{section}/by_dataset/{DATASET}/report.md';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text+'\n')
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not math.isfinite(v):return None
    return v
def ranked(a):
    order=np.argsort(a,kind='stable');out=np.empty(len(a),float);vals=a[order]
    edges=np.r_[0,np.flatnonzero(np.diff(vals)!=0)+1,len(a)]
    for l,r in zip(edges[:-1],edges[1:]):out[order[l:r]]=(l+r-1)/2
    return out
def corr(x,y,method='pearson'):return pearson(ranked(x),ranked(y)) if method=='spearman' else pearson(x,y)
def quad(y,t,s,d):
    tc=(t>0)==(y>0);sc=(s>0)==(y>0)
    q=np.where(tc,np.where(sc,'Q1','Q2'),np.where(sc,'Q3','Q4')).astype('U20')
    if not d.startswith('sims'):q[y==0]='Q0_zero_excluded'
    return q
def groupsplits(groups,k):
    unique,counts=np.unique(groups,return_counts=True);k=min(k,len(unique))
    assert k>=2,'Too few source-video groups'
    # Greedy sample-balanced assignment; ties broken by fixed hash, never labels.
    order=sorted(zip(unique,counts),key=lambda z:(-z[1],hashlib.sha256(('20261112:'+str(z[0])).encode()).hexdigest()))
    bins=[set() for _ in range(k)];sizes=[0]*k
    for g,n in order:
        f=min(range(k),key=lambda j:(sizes[j],j));bins[f].add(g);sizes[f]+=n
    splits=[]
    for b in bins:
        te=np.array([g in b for g in groups]);tr=~te
        assert not set(groups[tr])&set(groups[te]);splits.append((np.flatnonzero(tr),np.flatnonzero(te)))
    return splits
def ridge(X,y,V,alpha):
    if X.shape[1]==0:return np.full(len(V),y.mean())
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-12]=1;z=(X-mu)/sd;v=(V-mu)/sd
    return y.mean()+v@np.linalg.solve(z.T@z+alpha*np.eye(X.shape[1]),z.T@(y-y.mean()))
def loss(y,p,kind):return float(np.mean((y-p)**2 if kind=='response' else np.abs(y-p)))
ALPHAS=[.01,.1,1.,10.,100.]
def choose(X,y,g,kind):
    folds=groupsplits(g,3);scores=[]
    for a in ALPHAS:scores.append(np.mean([loss(y[v],ridge(X[t],y[t],X[v],a),kind) for t,v in folds]))
    return ALPHAS[int(np.argmin(scores))]

def utilities(d,t,meta):
    train=t['splits']=='train';valid=t['splits']=='valid';g=np.array([meta[k]['video_id'] for k in t['ids']]);X=t['raw'].mean(0);coords=t['coords'].mean(0)
    gt=g[train];folds=groupsplits(gt,5);novel=~np.isin(g[valid],gt)
    assignments=[]
    trainids=np.array(t['ids'])[train]
    for fi,(tr,va) in enumerate(folds):
        assignments.extend(dict(dataset=d,sample_id=k,source_video_id=group,outer_fold=fi) for k,group in zip(trainids[va],gt[va]))
    emit('incremental_utility','cv_assignments',assignments)
    save(OUT/f'protocol/{d}_grouped_cv.json',dict(train_parents=int(train.sum()),valid_parents=int(valid.sum()),train_groups=len(set(gt)),valid_groups=len(set(g[valid])),shared_train_valid_groups=len(set(gt)&set(g[valid])),novel_valid_parents=int(novel.sum()),outer_folds=len(folds),inner_folds=3,alphas=ALPHAS,standardization='training fold only',group_disjoint_assertions=True))
    records=[];orderrecords=[]
    for space,features,masks in [('subset',X,range(128)),('order',coords,[0,7,63,127])]:
        for kind,target in [('response',X[:,-1]),('task',t['labels'])]:
            yt=target[train];yv=target[valid];xt=features[train];xv=features[valid]
            for mask in masks:
                cols=[k for k in range(7) if mask&(1<<k)];xx=xt[:,cols];vv=xv[:,cols]
                for fi,(tr,va) in enumerate(folds):
                    a=choose(xx[tr],yt[tr],gt[tr],kind) if cols else 0
                    pred=ridge(xx[tr],yt[tr],xx[va],a)
                    entry=dict(dataset=d,space=space,utility=kind,mask=mask,subsets=','.join(NAMES[k] for k in cols),split='grouped_cv',fold=fi,n=len(va),alpha=a,score=-loss(yt[va],pred,kind),includes_tav=bool(mask&64) if space=='subset' else False)
                    (records if space=='subset' else orderrecords).append(entry)
                a=choose(xx,yt,gt,kind) if cols else 0;pred=ridge(xx,yt,vv,a)
                for tag,keep in [('valid_all',np.ones(len(yv),bool)),('valid_novel_source',novel)]:
                    if not keep.any():continue
                    entry=dict(dataset=d,space=space,utility=kind,mask=mask,subsets=','.join(NAMES[k] for k in cols),split=tag,fold=-1,n=int(keep.sum()),alpha=a,score=-loss(yv[keep],pred[keep],kind),includes_tav=bool(mask&64) if space=='subset' else False)
                    (records if space=='subset' else orderrecords).append(entry)
            print(json.dumps(dict(dataset=d,phase='utility',space=space,utility=kind,status='complete')),flush=True)
    for kind,name in [('response','teacher_response_utility'),('task','task_utility')]:emit('incremental_utility',name,[r for r in records if r['utility']==kind])
    emit('interaction_order_utility','order_task_utility',orderrecords)
    marginal=[];shapley=[]
    index=collections.defaultdict(dict)
    for r in records:index[(r['utility'],r['split'],r['fold'])][r['mask']]=r['score']
    for (kind,split,fold),scores in index.items():
        for excluded in [False,True]:
            players=list(range(6 if excluded else 7));n=len(players)
            for k in players:
                phi=0
                for mask in range(1<<n):
                    if mask&(1<<k):continue
                    delta=scores[mask|(1<<k)]-scores[mask];size=mask.bit_count();weight=math.factorial(size)*math.factorial(n-size-1)/math.factorial(n);phi+=weight*delta
                    marginal.append(dict(dataset=d,utility=kind,split=split,fold=fold,exclude_tav=excluded,added_subset=NAMES[k],context_mask=mask,context_size=size,delta_utility=delta))
                shapley.append(dict(dataset=d,utility=kind,split=split,fold=fold,exclude_tav=excluded,subset=NAMES[k],shapley=phi))
            total=sum(r['shapley'] for r in shapley if r['utility']==kind and r['split']==split and r['fold']==fold and r['exclude_tav']==excluded)
            np.testing.assert_allclose(total,scores[(1<<n)-1]-scores[0],atol=1e-9)
    emit('incremental_utility','conditional_marginal_utility',marginal);emit('incremental_utility','shapley_utility',shapley)
    ranking=[]
    for kind,ex,k in itertools.product(['response','task'],[False,True],NAMES):
        items=[r for r in shapley if r['utility']==kind and r['exclude_tav']==ex and r['subset']==k]
        cv=[r['shapley'] for r in items if r['split']=='grouped_cv']
        if not cv:continue
        ranking.append(dict(dataset=d,utility=kind,exclude_tav=ex,subset=k,cv_mean=np.mean(cv),cv_std=np.std(cv,ddof=1),cv_min=min(cv),cv_max=max(cv),cv_positive_fraction=np.mean(np.array(cv)>0),valid_all=next((r['shapley'] for r in items if r['split']=='valid_all'),None),valid_novel_source=next((r['shapley'] for r in items if r['split']=='valid_novel_source'),None)))
    emit('incremental_utility','subset_ranking',clean(ranking))
    note('incremental_utility',f'# {d}: A3\n\nSource-video grouped nested CV: outer {len(folds)} folds, inner 3 folds, alpha grid {ALPHAS}. Standardization/intercept fit on training fold only. Train and valid never mixed. Empty input predicts training-fold mean. Response utility=-MSE; task utility=-MAE. All 127 nonempty combinations plus empty baseline use identical ridge budget. Scores report each outer fold; fold std is descriptive, not a population confidence interval.\n\nTAV input trivially exposes the response target; exclusion-of-TAV six-player sensitivity is reported separately. Shapley describes analyzer predictive utility, not causal effects or Student training gains. Official Validation may share source videos with Train; valid_novel_source excludes these videos. No novel subset means that external independence check is unavailable. Shared pretrained teacher/probe fitting makes this diagnostic, not independent end-to-end generalization evidence. No adaptive weights are applied.\n')
    return ranking,orderrecords

def spectra(d,values,provenance,correlations,spectrum):
    for kind,a in [('prediction',values[0]),('interaction',values[1])]:
        for method in ['pearson','spearman']:
            for i,j in itertools.product(range(7),repeat=2):correlations.append(dict(**provenance,space=kind,correlation=method,row=NAMES[i],column=NAMES[j],value=corr(a[:,i],a[:,j],method)))
        for standardized in [False,True]:
            z=a.copy()
            if standardized:
                sd=z.std(0,ddof=1);sd[sd<1e-12]=1;z=(z-z.mean(0))/sd
            eig=np.maximum(np.linalg.eigvalsh(np.cov(z,rowvar=False)),0)[::-1];total=eig.sum();p=eig/total if total>1e-24 else eig
            effective=np.exp(-np.sum(p[p>0]*np.log(p[p>0]))) if total>1e-24 else 0
            singular=eig[-1]<=max(eig[0]*1e-12,1e-24)
            spectrum.append(dict(**provenance,space=kind,standardized=standardized,eigenvalues=json.dumps(eig.tolist()),effective_rank=effective,stable_rank_centered_data=total/eig[0] if eig[0]>0 else 0,stable_rank_covariance=np.sum(eig**2)/eig[0]**2 if eig[0]>0 else 0,condition_number=None if singular else eig[0]/eig[-1],numerically_singular=bool(singular),rank=int(np.sum(eig>max(eig[0]*1e-12,1e-24)))))

def stability(d,t):
    variance=[];difference=[];orders=[]
    for split in ['train','valid']:
        keep=t['splits']==split
        for space,arr in [('subset',t['raw']),('interaction',t['coords'])]:
            for k,name in enumerate(NAMES):
                a=arr[:,keep,k].T;v=a.var(1,ddof=1)
                variance.append(dict(dataset=d,split=split,space=space,coordinate=name,n=len(a),mean_probe_variance=v.mean(),variance_p50=np.quantile(v,.5),variance_p95=np.quantile(v,.95),icc_absolute_single=icc(a),ensemble_std=a.mean(1).std(ddof=1),mean_within_sample_probe_std=np.sqrt(v).mean(),coordinate_scale_sensitivity=t['coords'][:,keep,k].mean(0).std(ddof=1)/(t['raw'][:,keep,k].mean(0).std(ddof=1)+1e-12)))
                for i,j in itertools.combinations(range(3),2):
                    delta=a[:,i]-a[:,j]
                    difference.append(dict(dataset=d,split=split,space=space,coordinate=name,probe_i=t['probe_seeds'][i],probe_j=t['probe_seeds'][j],difference_mean=delta.mean(),difference_std=delta.std(ddof=1),difference_p05=np.quantile(delta,.05),difference_p50=np.quantile(delta,.5),difference_p95=np.quantile(delta,.95),absolute_p95=np.quantile(np.abs(delta),.95)))
            for q,idx in [(1,[0,1,2]),(2,[3,4,5]),(3,[6])]:
                items=[r for r in variance if r['split']==split and r['space']==space and NAMES.index(r['coordinate']) in idx]
                orders.append(dict(dataset=d,split=split,space=space,order=q,mean_coordinate_icc=np.mean([r['icc_absolute_single'] for r in items]),mean_coordinate_std=np.mean([r['ensemble_std'] for r in items]),mean_probe_variance=np.mean([r['mean_probe_variance'] for r in items])))
    emit('probe_stability','subset_probe_variance',clean([r for r in variance if r['space']=='subset']));emit('probe_stability','interaction_probe_variance',clean([r for r in variance if r['space']=='interaction']));emit('probe_stability','icc_analysis',clean(variance));emit('probe_stability','difference_distribution',clean(difference));emit('probe_stability','order_stability',clean(orders))
    note('probe_stability','# A4\n\nICC(A,1): two-way random absolute agreement, single measurement. Sample variance uses ddof=1 over three Probe heads. Order ICC is mean of coordinate-specific ICC, avoiding pooling unlike coordinate means/scales. Difference distributions retain probe-pair direction. All three heads share teacher hidden cache; variance is Probe-head sensitivity, not independent Teacher uncertainty. Anchors are per-probe Train TAV means, as frozen upstream; no Validation anchor fitting.')

def sample_diagnostics(d,t,meta,plan):
    corrs=[];spec=[];qm=[];qe=[];qi=[];fidelity=[];assoc=[];sources=[];casesamples=[];caseexamples=[];recon=[];gaps=[]
    mean=t['raw'].mean(0);tc=t['coords'].mean(0);anchor=t['anchors'].mean();M=matrix();train=t['splits']=='train';scale=tc[train].std(0,ddof=1);scale=np.maximum(scale,1e-12)
    for split in ['train','valid']:
        mask=t['splits']==split
        for seed,raw,coord in [('ensemble3',mean,tc)]+[(str(seed),t['raw'][i],t['coords'][i]) for i,seed in enumerate(t['probe_seeds'])]:spectra(d,(raw[mask],coord[mask]),dict(dataset=d,split=split,role='teacher',seed=seed,n=int(mask.sum())),corrs,spec)
        for q,idx in [(1,[0,1,2]),(2,list(range(6))),(3,list(range(7)))]:
            c=np.zeros_like(tc[mask]);c[:,idx]=tc[mask][:,idx];pred=anchor+c@np.linalg.inv(M).T
            recon.append(dict(dataset=d,split=split,through_order=q,n=int(mask.sum()),tav_rmse=np.sqrt(np.mean((pred[:,-1]-mean[mask,-1])**2)),all7_rmse=np.sqrt(np.mean((pred-mean[mask])**2))))
    for case in [c for c in plan['cases'] if c['dataset']==d]:
        for split in ['train','valid']:
            predrows=rows(OLD/f'predictions/{case["id"]}_{split}.jsonl');predmap={r['parent_sample_id']:r for r in predrows};mask=t['splits']==split;ids=np.array(t['ids'])[mask];y=t['labels'][mask];T=mean[mask];I=tc[mask]
            assert set(ids)==set(predmap) and len(predmap)==len(predrows)
            for k,yy in zip(ids,y):assert abs(predmap[k]['target_sentiment']-yy)<1e-6 and meta[k]['split']==split
            S=np.array([[predmap[k]['subset_predictions'][s] for s in NAMES] for k in ids]);J=(S-anchor)@M.T;Q=quad(y,T[:,-1],S[:,-1],d);sq=(J-I)**2;taskerr=np.abs(S[:,-1]-y)
            provenance=dict(dataset=d,split=split,case=case['id'],method=case['method'],seed=case['seed'],epoch=case['epoch'],checkpoint_id=case['checkpoint'],checkpoint_sha256=case['checkpoint_sha256'],teacher_target='ensemble3 subset responses and Mobius residuals; historical full KD base target may differ')
            spectra(d,(S,J),dict(dataset=d,split=split,role='student',case=case['id'],seed=case['seed'],n=len(y)),corrs,spec)
            gaps.append(dict(**provenance,n=len(y),teacher_mae=np.abs(T[:,-1]-y).mean(),student_mae=taskerr.mean(),student_minus_teacher_mae=taskerr.mean()-np.abs(T[:,-1]-y).mean(),official_teacher_metrics=json.dumps(metrics(d,y,T[:,-1])),official_student_metrics=json.dumps(metrics(d,y,S[:,-1]))))
            for j,k in enumerate(ids):
                casesamples.append(dict(**provenance,sample_id=str(k),source_video_id=meta[k]['video_id'],label=y[j],label_strength=abs(y[j]),teacher_tav=T[j,-1],student_tav=S[j,-1],teacher_error=abs(T[j,-1]-y[j]),student_error=taskerr[j],teacher_signed_error=T[j,-1]-y[j],student_signed_error=S[j,-1]-y[j],quadrant=Q[j],teacher_subsets=json.dumps(T[j].tolist()),student_subsets=json.dumps(S[j].tolist()),teacher_coordinates=json.dumps(I[j].tolist()),student_coordinates=json.dumps(J[j].tolist())))
            for quadrant in ['ALL','Q1','Q2','Q3','Q4','Q0_zero_excluded']:
                sel=np.ones(len(y),bool) if quadrant=='ALL' else Q==quadrant;n=int(sel.sum())
                if not n:continue
                eligible=np.sum(y!=0) if not d.startswith('sims') else len(y)
                qm.append(dict(**provenance,quadrant=quadrant,n=n,split_n=len(y),proportion_all=n/len(y),proportion_acc2_eligible=n/eligible if quadrant not in ['ALL','Q0_zero_excluded'] else None,teacher_mae=np.abs(T[sel,-1]-y[sel]).mean(),student_mae=taskerr[sel].mean(),teacher_bias=(T[sel,-1]-y[sel]).mean(),student_bias=(S[sel,-1]-y[sel]).mean(),label_abs_mean=np.abs(y[sel]).mean(),label_abs_p25=np.quantile(np.abs(y[sel]),.25),label_abs_p50=np.quantile(np.abs(y[sel]),.5),label_abs_p75=np.quantile(np.abs(y[sel]),.75),label_zero_n=int(np.sum(y[sel]==0))))
                for k,name in enumerate(NAMES):qe.append(dict(**provenance,quadrant=quadrant,subset=name,n=n,teacher_mae=np.abs(T[sel,k]-y[sel]).mean(),student_mae=np.abs(S[sel,k]-y[sel]).mean(),student_teacher_rmse=np.sqrt(np.mean((S[sel,k]-T[sel,k])**2))))
                for order,idx in [(1,[0,1,2]),(2,[3,4,5]),(3,[6])]:
                    err=J[sel][:,idx]-I[sel][:,idx];norm=err/scale[idx]
                    item=dict(**provenance,quadrant=quadrant,order=order,n=n,mae=np.abs(err).mean(),mse=np.mean(err**2),rmse=np.sqrt(np.mean(err**2)),normalized_rmse=np.sqrt(np.mean(norm**2)),normalizer='per-coordinate teacher ensemble Train standard deviation',error_p50=np.quantile(np.abs(err),.5),error_p95=np.quantile(np.abs(err),.95))
                    fidelity.append(item);qi.append(item.copy())
                    if n>2:assoc.append(dict(**provenance,quadrant=quadrant,order=order,n=n,pearson=corr(sq[sel][:,idx].mean(1),taskerr[sel]),spearman=corr(sq[sel][:,idx].mean(1),taskerr[sel],'spearman'),association='coordinate MSE versus absolute TAV task error; descriptive/confounded, no causality'))
                if quadrant not in ['ALL','Q0_zero_excluded']:
                    selected=np.flatnonzero(sel);ordered=sorted(selected,key=lambda j:(taskerr[j],str(ids[j])))
                    for tag,j in [('lowest_student_error',ordered[0]),('median_student_error',ordered[len(ordered)//2]),('highest_student_error',ordered[-1])]:caseexamples.append(dict(**provenance,quadrant=quadrant,selection_rule=tag,sample_id=str(ids[j]),source_video_id=meta[ids[j]]['video_id'],label=y[j],teacher_tav=T[j,-1],student_tav=S[j,-1]))
            for group in sorted({meta[k]['video_id'] for k in ids}):
                selected=np.array([meta[k]['video_id']==group for k in ids]);sources.append(dict(**provenance,source_video_id=group,n=int(selected.sum()),teacher_mae=np.abs(T[selected,-1]-y[selected]).mean(),student_mae=taskerr[selected].mean(),**{q+'_n':int(np.sum(Q[selected]==q)) for q in ['Q1','Q2','Q3','Q4','Q0_zero_excluded']}))
        print(json.dumps(dict(dataset=d,phase='samples',case=case['id'],status='complete')),flush=True)
    emit('sample_quadrants',d+'_quadrants',clean(casesamples));emit('sample_quadrants','quadrant_metrics',clean(qm));emit('sample_quadrants','quadrant_subset_errors',clean(qe));emit('sample_quadrants','quadrant_interaction_errors',clean(qi));emit('sample_quadrants','source_video_aggregation',clean(sources));emit('sample_quadrants','representative_examples',clean(caseexamples));emit('sample_quadrants','teacher_student_gap',clean(gaps))
    emit('interaction_order_utility','order_fidelity',clean(fidelity));emit('interaction_order_utility','order_reconstruction',clean(recon));emit('interaction_order_utility','fidelity_task_association',clean(assoc));emit('interaction_order_utility','dataset_order_comparison',clean([r for r in fidelity if r['quadrant']=='ALL']))
    for role in ['teacher','student']:
        for space,name in [('prediction','prediction'),('interaction','interaction')]:emit('subset_redundancy',role+'_'+name+'_corr',clean([r for r in corrs if r['role']==role and r['space']==space]))
    for name in ['covariance_eigenvalues','effective_rank','stable_rank']:emit('subset_redundancy',name,clean(spec))
    note('sample_quadrants',f'# {d}: A1\n\nQuadrants use Acc-2 correctness (prediction >0 and label >0). MOSI/MOSEI exclude label=0 from Acc-2 and retain those rows as Q0_zero_excluded. SIMS uses native official >0 convention, zero included. Every parent appears once per case/split; regression errors retain raw scores; official SIMS MAE in gap metrics clips predictions to [-1,1]. Cases/seed/checkpoint hashes remain separate; no cross-campaign pooling. Examples select lowest/median/highest Student absolute error in every quadrant deterministically. Q2 does not by itself identify a transfer mechanism. Source-video aggregates retain quadrant counts and mean errors.\n')
    note('subset_redundancy','# A2\n\nPearson/Spearman with average ranks for ties. Constant columns yield empty correlations. Covariance spectra are computed separately on raw scores and per-split standardized scores (descriptive only). Eigenvalues clipped at zero for numerical roundoff. Effective rank=exp(entropy(normalized eigenvalues)); stable rank of centered data=trace(cov)/lambda_max; covariance stable rank=sum(lambda^2)/lambda_max^2. Covariance condition number is empty and numerically_singular=true when min eigenvalue<=max(1e-12*max,1e-24). Invertible transforms add no information; correlation does not establish conditional usefulness.')
    note('interaction_order_utility','# A6\n\nOrder error normalizes each coordinate by Teacher ensemble Train standard deviation. Raw error alone cannot establish learning difficulty. Exact reconstruction through each cumulative order reports both TAV and all-seven RMSE separately. Ridge order utility follows A3 grouped nested CV; it is not Student training ablation. Fidelity/task correlations are sample-level descriptions; retained best checkpoints are insufficient for within-run epoch improvement causality. Dataset scales differ and SIMS cohorts overlap.')
    return gaps

def audit_dataset(d,t,meta,p):
    counts=[];overlap=[]
    for split in ['train','valid']:
        ids=[k for k,s in zip(t['ids'],t['splits']) if s==split];videos={meta[k]['video_id'] for k in ids}
        canonical=[(k,meta[k]['video_id'],meta[k]['split'],float(meta[k]['sentiment'])) for k in sorted(ids)]
        counts.append(dict(dataset=d,split=split,parents=len(ids),source_videos=len(videos),canonical_membership_sha256=hashlib.sha256(json.dumps(canonical,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),manifest_path=p['assets'],manifest_sha256=sha(read(p['assets'])['manifest'])))
    gs={split:{meta[k]['video_id'] for k,s in zip(t['ids'],t['splits']) if s==split} for split in ['train','valid']};both=gs['train']&gs['valid']
    overlap.append(dict(dataset=d,comparison='train_valid',shared_source_videos=len(both),train_video_fraction=len(both)/len(gs['train']),valid_video_fraction=len(both)/len(gs['valid']),valid_parent_fraction_on_shared_sources=np.mean([meta[k]['video_id'] in both for k,s in zip(t['ids'],t['splits']) if s=='valid'])))
    emit('protocol_audit','dataset_fingerprints',counts);emit('protocol_audit','source_video_overlap',overlap)
    jsonl(OUT/f'protocol/{d}_parent_identity.jsonl',[dict(parent_sample_id=k,video_id=meta[k]['video_id'],split=meta[k]['split']) for k in sorted(meta)])

def main():
    global DATASET
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);args=ap.parse_args();DATASET=args.dataset;d=DATASET;started=time.time()
    plan=read(OUT/'protocol/plan.json');p=plan['protocols'][d];t=teacher_arrays(d,p);t['probe_seeds']=[v['seed'] for v in p['probes']]
    manifest=rows(read(p['assets'])['manifest']);meta={}
    for r in manifest:
        k=r['parent_sample_id']
        if k in meta:assert (meta[k]['video_id'],meta[k]['split'],meta[k]['sentiment'])==(r['video_id'],r['split'],r['sentiment'])
        meta[k]=r
    assert set(meta)==set(t['ids']);audit_dataset(d,t,meta,p);stability(d,t);gaps=sample_diagnostics(d,t,meta,plan)
    rankings,orders=utilities(d,t,meta)
    save(OUT/f'status/analysis_{d}.json',dict(status='complete',dataset=d,elapsed_seconds=time.time()-started,samples=len(t['ids']),cases=9,parameter_updates=0,test_predictions_used=False))
    save(OUT/f'protocol/{d}_findings.json',clean(dict(gaps=[r for r in gaps if r['split']=='valid'],rankings=rankings,orders=orders)))
    print(json.dumps(dict(dataset=d,status='complete',elapsed_seconds=time.time()-started)),flush=True)

if __name__=='__main__':main()
