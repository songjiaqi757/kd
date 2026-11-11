"""Rebuild Phase 0 tables, figures, replay audit and evidence-limited decisions."""
from collections import defaultdict
import math
import numpy as np
from common import *
from cpu import teacher_arrays,metrics,pearson,matrix

def avg(values):
    valid=[v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else None

def sd(values):
    valid=[v for v in values if v is not None and np.isfinite(v)]
    return float(np.std(valid,ddof=1)) if len(valid)>1 else None

def fmt(v):return 'NA' if v is None else f'{v:.4f}'

def main():
    plan=read(OUT/'protocol/plan.json');subsets=[];coordinate=[];orders=[];quadrants=[];taskrows=[];replay=[];gradrows=[];coverage=[];parameter_counts=[]
    teacher={d:teacher_arrays(d,p) for d,p in plan['protocols'].items()};M=matrix();gap_samples={};student_errors={};error_distributions={}
    padding_flags={}
    for d,p in plan['protocols'].items():
        manifest=rows(read(p['assets'])['manifest']);keys=[k for k in manifest[0] if 'padding_seconds' in k]
        if keys:
            flags=defaultdict(bool)
            for r in manifest:flags[r['parent_sample_id']] |= any(float(r.get(k,0))>0 for k in keys)
            padding_flags[d]=flags
    for c in plan['cases']:
        d=c['dataset'];t=teacher[d];p=plan['protocols'][d];idx={key:i for i,key in enumerate(t['ids'])}
        trainmask=t['splits']=='train';scale=t['coords'].mean(0)[trainmask].std(0,ddof=1)
        for split in ['train','valid']:
            path=OUT/f'predictions/{c["id"]}_{split}.jsonl';available=path.exists()
            coverage.append(dict(case=c['id'],dataset=d,method=c['method'],seed=c['seed'],split=split,available=available))
            if not available:continue
            data=rows(path);ids=[r['parent_sample_id'] for r in data]
            assert len(ids)==len(set(ids))
            expected={key for key,s in zip(t['ids'],t['splits']) if s==split};assert set(ids)==expected
            sel=[idx[key] for key in ids];y=t['labels'][sel]
            np.testing.assert_allclose(y,[r['target_sentiment'] for r in data],atol=1e-6,rtol=0)
            f=np.array([[r['subset_predictions'][s] for s in NAMES] for r in data]);target=t['raw'].mean(0)[sel];base=t['raw'][0,sel]
            assert np.isfinite(f).all()
            I=(f-float(t['anchors'].mean()))@M.T;IT=t['coords'].mean(0)[sel];delta=I-IT
            if c['method']=='uniform_interaction' and c['seed']==13 and split=='valid':
                error_distributions[d]={order:(np.abs(delta[:,indices])/np.maximum(scale[indices],1e-8)).mean(1) for order,indices in [(1,[0,1,2]),(2,[3,4,5]),(3,[6])]}
            # The common-anchor error must agree with the linear transformation of prediction error.
            np.testing.assert_allclose(delta,(f-target)@M.T,atol=1e-10,rtol=0)
            identity=dict(dataset=d,method=c['method'],seed=c['seed'],split=split,epoch=c['epoch'],checkpoint_sha256=c['checkpoint_sha256'])
            for k,s in enumerate(NAMES):
                subsets.append(dict(**identity,subset=s,n=len(y),teacher_fidelity_mae=float(np.abs(f[:,k]-target[:,k]).mean()),teacher_fidelity_pearson=pearson(f[:,k],target[:,k]),student_task_mae=float(np.abs(f[:,k]-y).mean()),teacher_task_mae=float(np.abs(target[:,k]-y).mean()),student_task_pearson=pearson(f[:,k],y)))
                coordinate.append(dict(**identity,coordinate=s,n=len(y),mae=float(np.abs(delta[:,k]).mean()),rmse=float(np.sqrt(np.square(delta[:,k]).mean())),teacher_train_std=float(scale[k]),normalized_mae=float(np.abs(delta[:,k]).mean()/max(scale[k],1e-8)),normalizer='ensemble teacher coordinate train std, fixed across methods and splits',teacher_fidelity_pearson=pearson(I[:,k],IT[:,k])))
            for order,indices in [(1,[0,1,2]),(2,[3,4,5]),(3,[6])]:
                orders.append(dict(**identity,order=order,n=len(y),mae=float(np.abs(delta[:,indices]).mean()),normalized_mae=float((np.abs(delta[:,indices])/np.maximum(scale[indices],1e-8)).mean())))
            student_metrics=metrics(d,y,f[:,-1]);tm=metrics(d,y,target[:,-1]);bm=metrics(d,y,base[:,-1])
            taskrows.append(dict(**identity,n=len(y),**student_metrics,teacher_ensemble_mae=tm['mae'],teacher_base_seed2026_mae=bm['mae'],student_minus_ensemble_mae=student_metrics['mae']-tm['mae'],student_minus_base_mae=student_metrics['mae']-bm['mae']))
            if split=='valid':
                rr=read(path.with_suffix('.report.json'));replay.append(dict(case=c['id'],mae_abs_delta=rr.get('validation_mae_abs_delta'),tolerance=rr.get('validation_mae_tolerance'),parameter_updates=rr['parameter_updates'],parameters_unchanged=rr['parameter_versions_unchanged']))
                parameter_counts.append(dict(case=c['id'],dataset=d,method=c['method'],seed=c['seed'],initialization_seed=c['config']['seed'],trainable_parameters=rr['trainable_parameters'],**rr['parameter_counts']))
                gap_samples[c['id']]=(ids,np.abs(f[:,-1]-y)-np.abs(target[:,-1]-y),t)
                student_errors[(d,c['method'],c['seed'])]={key:float(err) for key,err in zip(ids,np.abs(f[:,-1]-y))}
            tolerance=p['quadrant_tolerance'];student_ok=np.abs(f[:,-1]-y)<=tolerance
            strength=np.abs(y)/(1 if d.startswith('sims') else 3)
            for target_name,teacher_score in [('ensemble3',target[:,-1]),('seed2026',base[:,-1])]:
                teacher_ok=np.abs(teacher_score-y)<=tolerance
                q=np.where(teacher_ok,np.where(student_ok,'T_correct_S_correct','T_correct_S_wrong'),np.where(student_ok,'T_wrong_S_correct','T_wrong_S_wrong'))
                group_masks=[('all',np.ones(len(y),dtype=bool)),('strength_le_1_3',strength<=1/3),('strength_1_3_to_2_3',(strength>1/3)&(strength<=2/3)),('strength_gt_2_3',strength>2/3)]
                if d in padding_flags:
                    flag=np.array([padding_flags[d][key] for key in ids]);group_masks += [('prepared_media_padding_present',flag),('prepared_media_padding_absent',~flag)]
                for group,mask in group_masks:
                    for label in ['T_correct_S_correct','T_correct_S_wrong','T_wrong_S_correct','T_wrong_S_wrong']:
                        qm=mask&(q==label);count=int(qm.sum())
                        quadrants.append(dict(**identity,teacher_target=target_name,group=group,quadrant=label,n_group=int(mask.sum()),count=count,fraction=float(count/mask.sum()) if mask.sum() else None,tolerance=tolerance,mean_absolute_interaction_teacher=float(np.abs(IT[qm]).mean()) if count else None,mean_interaction_error=float(np.abs(delta[qm]).mean()) if count else None,modality_quality='preparation padding metadata; not a semantic quality score' if d in padding_flags else 'not available; no quality proxy invented',**{f'teacher_abs_{s}':float(np.abs(IT[qm,k]).mean()) if count else None for k,s in enumerate(NAMES)},**{f'error_{s}':float(np.abs(delta[qm,k]).mean()) if count else None for k,s in enumerate(NAMES)}))
    for path in sorted((OUT/'gradients').glob('*.json')):
        result=read(path);samples=result['samples'];identity={k:result[k] for k in ['dataset','method','seed','epoch','checkpoint_sha256']}
        for component in samples[0]['geometry']:
            for reference in ['task','kd','base']:
                for group in samples[0]['geometry'][component][reference]:
                    values=[s['geometry'][component][reference][group] for s in samples]
                    cs=[v['cosine'] for v in values if v['cosine'] is not None]
                    gradrows.append(dict(**identity,component=component,reference=reference,parameter_group=group,n_train_microbatches=len(samples),base_norm_mean=avg([v['base_norm'] for v in values]),aux_norm_mean=avg([v['aux_norm'] for v in values]),norm_ratio_mean=avg([v['norm_ratio'] for v in values]),cosine_mean=avg(cs),cosine_sd=sd(cs),negative_cosine_fraction=float(np.mean(np.array(cs)<0)) if cs else None,diagnostic_mode='eval; dropout disabled; microbatch1; 6 fixed train windows; no optimizer',icd_weight='raw, no lambda applied'))
    table('student_absorbability.csv',subsets,['dataset','method','seed','split','epoch','checkpoint_sha256','subset','n','teacher_fidelity_mae','teacher_fidelity_pearson','student_task_mae','teacher_task_mae','student_task_pearson'])
    table('interaction_coordinate_error.csv',coordinate,['dataset','method','seed','split','epoch','checkpoint_sha256','coordinate','n','mae','rmse','teacher_train_std','normalized_mae','normalizer','teacher_fidelity_pearson'])
    table('interaction_order_error.csv',orders,['dataset','method','seed','split','epoch','checkpoint_sha256','order','n','mae','normalized_mae'])
    table('student_task_quality.csv',taskrows,['dataset','method','seed','split','epoch','checkpoint_sha256','n']+list(dict.fromkeys(k for r in taskrows for k in r if k not in ['dataset','method','seed','split','epoch','checkpoint_sha256','n'])))
    table('sample_quadrants.csv',quadrants,['dataset','method','seed','split','epoch','checkpoint_sha256','teacher_target','group','quadrant','n_group','count','fraction','tolerance','mean_absolute_interaction_teacher','mean_interaction_error','modality_quality']+[f'teacher_abs_{s}' for s in NAMES]+[f'error_{s}' for s in NAMES])
    table('gradient_compatibility.csv',gradrows,['dataset','method','seed','epoch','checkpoint_sha256','component','reference','parameter_group','n_train_microbatches','base_norm_mean','aux_norm_mean','norm_ratio_mean','cosine_mean','cosine_sd','negative_cosine_fraction','diagnostic_mode','icd_weight'])
    table('validation_replay.csv',replay,['case','mae_abs_delta','tolerance','parameter_updates','parameters_unchanged'])
    table('student_parameter_counts.csv',parameter_counts,['case','dataset','method','seed','initialization_seed','trainable_parameters','fusion','text_lora','audio_lora','video_lora','prediction_head','other'])
    table('coverage.csv',coverage)
    aggregate=[]
    for tablename,data,keys,fields in [('task',taskrows,['dataset','method','split'],['mae','pearson','acc2','f1_weighted','acc2_nonzero','f1_weighted_nonzero','student_minus_ensemble_mae']),('order',orders,['dataset','method','split','order'],['mae','normalized_mae'])]:
        groups=defaultdict(list)
        for r in data:groups[tuple(r[k] for k in keys)].append(r)
        for key,records in groups.items():
            identity=dict(zip(keys,key))
            for metric in fields:
                values=[r.get(metric) for r in records]
                aggregate.append(dict(table=tablename,**identity,metric=metric,n_seeds=len(records),mean=avg(values),sd=sd(values)))
    table('seed_aggregates.csv',aggregate,['table','dataset','method','split','order','metric','n_seeds','mean','sd'])
    gaps=[]
    for data,space,key,fields in [(subsets,'subset','subset',['teacher_fidelity_mae','student_task_mae']), (coordinate,'interaction','coordinate',['mae','normalized_mae'])]:
        groups=defaultdict(dict)
        for r in data:groups[(r['dataset'],r['method'],r['seed'],r[key])][r['split']]=r
        for (d,method,seed,name),parts in groups.items():
            if set(parts)!={'train','valid'}:continue
            for field in fields:gaps.append(dict(dataset=d,method=method,seed=seed,space=space,coordinate=name,metric=field,train=parts['train'][field],valid=parts['valid'][field],valid_minus_train=parts['valid'][field]-parts['train'][field]))
    table('train_validation_gap.csv',gaps,['dataset','method','seed','space','coordinate','metric','train','valid','valid_minus_train'])
    bootstrap=[]
    for case,(ids,diff,t) in gap_samples.items():
        d=next(c['dataset'] for c in plan['cases'] if c['id']==case)
        bootstrap.append(dict(case=case,dataset=d,comparison='student_minus_teacher_ensemble_mae',**cluster_bootstrap(ids,diff,t)))
    paired=[]
    for (d,method,seed),errors in student_errors.items():
        if method=='full_kd' or (d,'full_kd',seed) not in student_errors:continue
        reference=student_errors[(d,'full_kd',seed)];ids=sorted(errors);assert set(ids)==set(reference)
        diffs=np.array([errors[key]-reference[key] for key in ids])
        paired.append(dict(dataset=d,method=method,reference='full_kd',seed=seed,**cluster_bootstrap(ids,diffs,teacher[d])))
    table('paired_bootstrap.csv',bootstrap,['case','dataset','comparison','n_utterances','n_source_videos','mean_difference','ci95_low','ci95_high','standardized_paired_effect','replicates','bootstrap_seed','unit'])
    table('paired_method_differences.csv',paired,['dataset','method','reference','seed','n_utterances','n_source_videos','mean_difference','ci95_low','ci95_high','standardized_paired_effect','replicates','bootstrap_seed','unit'])
    pairs=defaultdict(list)
    for r in paired:pairs[(r['dataset'],r['method'])].append(r)
    table('paired_method_summary.csv',[dict(dataset=d,method=m,reference='full_kd',n_seeds=len(rs),paired_mae_difference_mean=avg([r['mean_difference'] for r in rs]),paired_mae_difference_sd=sd([r['mean_difference'] for r in rs])) for (d,m),rs in pairs.items()],['dataset','method','reference','n_seeds','paired_mae_difference_mean','paired_mae_difference_sd'])
    protocols=[]
    for c in plan['cases']:
        cf=c['config'];a=read(plan['protocols'][c['dataset']]['assets'])
        history=read(Path(c['run'])/'history.json')
        if isinstance(history,dict):history=history.get('epochs',history.get('history'))
        protocols.append(dict(case=c['id'],dataset=c['dataset'],method=c['method'],seed=c['seed'],batch_size=cf['batch_size'],max_epochs=cf['epochs'],completed_epochs=len(history),min_epochs=cf.get('min_epochs'),patience=cf['patience'],optimizer='AdamW',learning_rate=cf['learning_rate'],weight_decay=cf['weight_decay'],scheduler=cf.get('lr_schedule','constant'),alpha_ce=a['alpha_ce'],lambda_interaction=cf.get('lambda_interaction',a['lambda_interaction']),selected_epoch=c['epoch'],selection='valid_mae_minimum_earliest_tie',original_selection=c['original_checkpoint_selection']))
    table('training_protocol_differences.csv',protocols)
    save(OUT/'protocol/teacher_manifest.json',{d:plan['protocols'][d] for d in teacher})
    save(OUT/'protocol/student_manifest.json',plan['cases'])
    sims={r['parent_sample_id']:r for r in teacher['sims']['ensemble']};sims2={r['parent_sample_id']:r for r in teacher['sims_v2']['ensemble']};shared=set(sims)&set(sims2)
    save(OUT/'protocol/sims_cross_version_overlap.json',dict(scope='Train/Validation metadata only',sims_parents=len(sims),sims_v2_parents=len(sims2),shared_parent_ids=len(shared),same_split=sum(sims[k]['split']==sims2[k]['split'] for k in shared),same_label=sum(sims[k]['sentiment']==sims2[k]['sentiment'] for k in shared),interpretation='These overlapping official Train/Validation cohorts are not independent benchmark replicates. No test labels were inspected for this check.'))
    complete=sum(r['available'] for r in coverage)==72 and len(list((OUT/'gradients').glob('*.json')))==12
    make_figures(teacher,orders,gradrows,taskrows,coordinate,error_distributions)
    write_report(plan,teacher,taskrows,orders,gradrows,coverage,complete,replay)
    save(OUT/'status/summary.json',dict(status='complete' if complete else 'in_progress',prediction_exports=sum(r['available'] for r in coverage),expected_exports=72,gradient_cases=len(list((OUT/'gradients').glob('*.json'))),expected_gradient_cases=12,optimizer_steps=0,official_test_predictions_used=False))
    print(json.dumps(read(OUT/'status/summary.json')),flush=True)

def cluster_bootstrap(ids,diff,t):
    videos={r['parent_sample_id']:r['video_id'] for r in t['ensemble']};unique=sorted({videos[key] for key in ids});vi={v:i for i,v in enumerate(unique)}
    sums=np.zeros(len(unique));counts=np.zeros(len(unique))
    for key,value in zip(ids,diff):i=vi[videos[key]];sums[i]+=value;counts[i]+=1
    rng=np.random.default_rng(130013);sample=rng.integers(0,len(unique),size=(2000,len(unique)))
    boot=sums[sample].sum(1)/counts[sample].sum(1);lo,hi=np.quantile(boot,[.025,.975]);std=float(np.std(diff,ddof=1))
    return dict(n_utterances=len(ids),n_source_videos=len(unique),mean_difference=float(np.mean(diff)),ci95_low=float(lo),ci95_high=float(hi),standardized_paired_effect=float(np.mean(diff)/std) if std>1e-12 else None,replicates=2000,bootstrap_seed=130013,unit='source_video_cluster; resample videos, retain all utterances')

def make_figures(teacher,orders,gradrows,taskrows,coordinate,error_distributions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets=list(teacher)
    for metric in ['mae','pearson']:
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
        for ax,split in zip(axes,['train','valid']):
            values=[]
            for d in datasets:
                t=teacher[d];mask=t['splits']==split;y=t['labels'][mask];raw=t['raw'].mean(0)[mask]
                values.append([float(np.abs(raw[:,k]-y).mean()) if metric=='mae' else pearson(raw[:,k],y) for k in range(7)])
            arr=np.array(values,dtype=float);im=ax.imshow(arr,cmap='viridis' if metric=='mae' else 'coolwarm',vmin=None if metric=='mae' else -1,vmax=None if metric=='mae' else 1)
            ax.set_xticks(range(7),[s.upper() for s in NAMES]);ax.set_yticks(range(4),datasets);ax.set_title(f'Teacher ensemble {metric.upper()} ({split})')
            for i in range(4):
                for j in range(7):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',color='white' if metric=='mae' and arr[i,j]<np.nanmax(arr)*.5 else 'black',fontsize=9)
            fig.colorbar(im,ax=ax,shrink=.8)
        fig.suptitle('Native scales differ: compare subsets within each dataset')
        for ext in ['png','pdf']:fig.savefig(OUT/f'figures/teacher_subset_{metric}_heatmap.{ext}',dpi=160)
        plt.close(fig)
    if orders:
        fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
        for ax,d in zip(axes,datasets):
            for method in ['full_kd','subset7','uniform_interaction']:
                vals=[avg([r['normalized_mae'] for r in orders if r['dataset']==d and r['method']==method and r['split']=='valid' and r['order']==k]) for k in [1,2,3]]
                if any(v is not None for v in vals):ax.plot([1,2,3],vals,'o-',label=method)
            ax.set_title(d);ax.set_xticks([1,2,3]);ax.set_xlabel('Interaction order');ax.set_ylabel('MAE / teacher train SD');ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7)
        for ext in ['png','pdf']:fig.savefig(OUT/f'figures/interaction_order_error.{ext}',dpi=160)
        plt.close(fig)
    if coordinate:
        fig,axes=plt.subplots(1,4,figsize=(17,4),layout='constrained')
        for ax,d in zip(axes,datasets):
            methods=['full_kd','subset7','uniform_interaction']
            values=np.array([[avg([r['normalized_mae'] for r in coordinate if r['dataset']==d and r['method']==method and r['split']=='valid' and r['coordinate']==s]) for s in NAMES] for method in methods],dtype=float)
            im=ax.imshow(values,aspect='auto',cmap='magma');ax.set_xticks(range(7),[s.upper() for s in NAMES],rotation=45);ax.set_yticks(range(3),['Full KD','Subset-7','ICD']);ax.set_title(d);fig.colorbar(im,ax=ax,shrink=.8)
        fig.suptitle('Validation interaction MAE / ensemble teacher train SD')
        for ext in ['png','pdf']:fig.savefig(OUT/f'figures/interaction_error_heatmap.{ext}',dpi=160)
        plt.close(fig)
    if gradrows:
        fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
        groups=['fusion','text_lora','audio_lora','video_lora','prediction_head']
        for ax,d in zip(axes,datasets):
            records=[r for r in gradrows if r['dataset']==d and r['method']=='uniform_interaction' and r['component']=='icd' and r['reference']=='base']
            values=[avg([r['cosine_mean'] for r in records if r['parameter_group']==g]) for g in groups]
            ax.bar(range(5),[v if v is not None else np.nan for v in values]);ax.set_xticks(range(5),['Fusion','Text','Audio','Video','Head'],rotation=40);ax.set_ylim(-1,1);ax.axhline(0,color='gray',lw=1);ax.set_title(d);ax.set_ylabel('Mean cosine(base, ICD)')
        for ext in ['png','pdf']:fig.savefig(OUT/f'figures/gradient_compatibility.{ext}',dpi=160)
        plt.close(fig)
        fig,axes=plt.subplots(1,4,figsize=(17,4.5),layout='constrained')
        for ax,d in zip(axes,datasets):
            values=np.array([[avg([r['cosine_mean'] for r in gradrows if r['dataset']==d and r['method']=='uniform_interaction' and r['reference']=='base' and r['component']=='coordinate_'+s and r['parameter_group']==g]) for s in NAMES] for g in groups],dtype=float)
            im=ax.imshow(values,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1);ax.set_xticks(range(7),[s.upper() for s in NAMES],rotation=40);ax.set_yticks(range(5),['Fusion','Text LoRA','Audio LoRA','Video LoRA','Head']);ax.set_title(d);fig.colorbar(im,ax=ax,shrink=.8)
        fig.suptitle('ICD seed13: mean cosine(base, coordinate), six fixed train windows')
        for ext in ['png','pdf']:fig.savefig(OUT/f'figures/gradient_coordinate_heatmap.{ext}',dpi=160)
        plt.close(fig)
    if error_distributions:
        fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
        for ax,d in zip(axes,datasets):
            for order,values in error_distributions.get(d,{}).items():
                x=np.sort(values);ax.plot(x,np.arange(1,len(x)+1)/len(x),label=f'Order {order}')
            ax.set_title(d);ax.set_xlabel('Normalized interaction error');ax.set_ylabel('Empirical CDF');ax.grid(alpha=.2)
        axes[-1].legend(fontsize=8)
        for ext in ['png','pdf']:fig.savefig(OUT/f'figures/interaction_error_distribution.{ext}',dpi=160)
        plt.close(fig)
    if taskrows:
        fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
        for ax,d in zip(axes,datasets):
            methods=['full_kd','subset7','uniform_interaction'];vals=[avg([r['mae'] for r in taskrows if r['dataset']==d and r['split']=='valid' and r['method']==m]) for m in methods]
            ax.bar(range(3),[v if v is not None else np.nan for v in vals]);t=teacher[d];mask=t['splits']=='valid';tm=float(np.abs(t['raw'].mean(0)[mask,-1]-t['labels'][mask]).mean());ax.axhline(tm,color='red',ls='--',label='Teacher ensemble')
            ax.set_xticks(range(3),['Full KD','Subset-7','ICD'],rotation=30);ax.set_ylabel('Validation native MAE');ax.set_title(d);ax.legend(fontsize=7)
        for ext in ['png','pdf']:fig.savefig(OUT/f'figures/teacher_student_gap.{ext}',dpi=160)
        plt.close(fig)

def write_report(plan,teacher,tasks,orders,grads,coverage,complete,replay):
    lines=['# 四数据集 Phase 0 Transferability Audit','',f'状态：{"已完成" if complete else "运行中；下表仅包含已落盘的完整导出"}。仅 Phase 0；训练和 optimizer step 均为 0；未使用 Test 预测设计方法。',
        '',f'代码基线：`{plan["git_commit"]}`。资产、checkpoint SHA256、原配置、路径修复证明和 split 指纹见 `protocol/asset_manifest.json`。本次新增脚本单独保存在 `source/`。',
        '', '## 审计范围与协议','',
        '- 四数据集 × Full KD / Subset-7 / ICD × seeds 13、42、2026；按各历史 run 的 validation MAE 最低、同分较早 epoch 选权重。MAE-best 的指标整行报告，不拼接 Acc2-best。',
        '- 梯度诊断覆盖四数据集、三方法的 seed13，共 12 个 checkpoint；每个 6 个固定 Train 窗口、microbatch1。Eval 模式关闭 dropout，保留 autograd；因此这是 checkpoint 局部诊断，不能解释为完整训练轨迹。',
        '- 参数组：Fusion（含模态池化）、Text/Audio/Video LoRA、Prediction head；同时保留全模型结果。ICD 按七坐标和三阶数分解。',
        '- Base = SmoothL1 task + 原协议 alpha_ce × CE + seed2026 TAV SmoothL1 KD + calibrated KL；ICD = 7 个 anchored interaction SmoothL1 均值，报告原始未乘 λ 梯度。',
        '- 教师质量同时给出 seed2026 与三 Probe ensemble。原始 Full KD base 仅使用 seed2026，而 ICD/Subset-7 的额外目标来自三 Probe；这个历史差异不能当成纯几何效应。',
        '- 三 Probe 共享同一教师 hidden cache，只是不同拟合 seed 的预测头；一致性不等于正确性，也不等于独立教师不确定性。',
        '- 每个 Probe 的 anchor 为自身 Train TAV 预测均值；Student 使用三 Probe anchor 的均值。交互坐标为 A(f−anchor)，所有 anchor 均未使用 Validation/Test 估计。',
        '- 历史训练预算并非完全统一，MOSEI 各 campaign 的 epoch 上限/早停等有差异；本报告的配对差值只作描述性审计，不能当成等预算确认实验。差异见 `tables/training_protocol_differences.csv`。MOSI seed13 历史配置曾标记 test_mae；本次从 history 重新按 validation MAE 选择具体 epoch，不沿用 Test-best。',
        '- 四象限容差在查看预测前固定为原生标签范围宽度的 1/12：英语 0.5，SIMS 1/6；按绝对回归误差判定，另按标签强度分组。MOSEI 按实际准备清单的音/视频填充记录分“有填充/无填充”；这只是预处理属性，不等于语义质量。其余数据集无对应可靠字段，明确不提供质量分组。',
        '- SIMS 原生 [-1,1] / 5 类、Acc2 含零；英语 [-3,3] / 7 类、主 Acc2 排除真值零。绝对 MAE 不跨数据集排名。',
        '- Test 仅检查 split ID/source-video 元数据；没有新增 Test 预测导出、权重估计或超参数选择。历史 Test 已多轮接触，后续不得称为完全盲测。','',
        '| 数据集 | Train utterances | Valid utterances | Train/Valid 源视频重叠 | Train/Test 源视频重叠 | Valid/Test 源视频重叠 |','|---|---:|---:|---:|---:|---:|']
    for d,p in plan['protocols'].items():
        o=p['source_video_overlap'];lines.append(f'| {d} | {p["parent_counts"]["train"]} | {p["parent_counts"]["valid"]} | {o["train_valid"]} | {o.get("train_test")} | {o.get("valid_test")} |')
    lines+=['','SIMS/SIMS v2 保留官方 utterance split，但它们并非 source-video 独立划分；不能把共享视频的 utterance 当作独立视频样本。另据本地 Train/Validation 清单，SIMS 的 1824 个 parent IDs 全部也出现在 SIMS v2，标签和 split 一致；这两个版本不能视为独立 benchmark 重复。英语 manifest 的历史指纹差异已证明仅为 `/home/wy/sjq/kd` → `/data/sjq/kd` 路径替换，反向替换精确恢复原 SHA256。历史文件未改写。','',
        '## Teacher 与 Student 验证集指标','',
        '| 数据集 | 方法 | 完成 seeds | Student MAE ± SD | Teacher ensemble MAE | Student − Teacher |','|---|---|---:|---:|---:|']
    for d,t in teacher.items():
        mask=t['splits']=='valid';tm=float(np.abs(t['raw'].mean(0)[mask,-1]-t['labels'][mask]).mean())
        for method in ['full_kd','subset7','uniform_interaction']:
            rs=[r for r in tasks if r['dataset']==d and r['method']==method and r['split']=='valid'];values=[r['mae'] for r in rs]
            lines.append(f'| {d} | {method} | {",".join(str(r["seed"]) for r in rs) or "待完成"} | {fmt(avg(values))} ± {fmt(sd(values))} | {tm:.4f} | {fmt(avg([r["student_minus_ensemble_mae"] for r in rs]))} |')
    lines+=['','所有 seed 的 MAE、Acc2、Pearson、F1、checkpoint hash、Teacher fidelity 与标签任务指标保存在 `tables/student_task_quality.csv`、`student_absorbability.csv`。`checkpoint_validation_metrics.csv` 保留历史全部 epoch 的 validation 指标。',
        '', '交互坐标归一化定义为 MAE(I_S,I_T) / Std_Train(I_T)，按坐标使用固定教师 Train 统计，分母下限 1e-8；阶数指标在该阶的坐标内取均值。标准差很小会放大归一化误差，因此必须同时审阅原始 MAE 和该分母。跨 Probe 方差及跨 seed SD 均采用 ddof=1。',
        '', '## 当前证据与决策门','']
    decisions=[]
    for d in teacher:
        op=[r for r in orders if r['dataset']==d and r['method']=='uniform_interaction' and r['split']=='valid']
        ordervals={k:avg([r['normalized_mae'] for r in op if r['order']==k]) for k in [1,2,3]}
        gs=[r for r in grads if r['dataset']==d and r['method']=='uniform_interaction' and r['component']=='icd' and r['reference']=='base' and r['parameter_group']=='all']
        if op:decisions.append(f'- {d}：ICD 验证集一/二/三阶归一化 MAE 为 {" / ".join(fmt(ordervals[k]) for k in [1,2,3])}。该误差衡量吸收，不直接代表坐标的任务效用。')
        if gs:decisions.append(f'- {d}：ICD seed13 的固定 Train 样本全参数 mean cosine(base, ICD) = {fmt(gs[0]["cosine_mean"])}，负余弦比例 {fmt(gs[0]["negative_cosine_fraction"])}，raw gradient norm ratio = {fmt(gs[0]["norm_ratio_mean"])}；层组与坐标明细应优先用于定位，六个窗口不足以证明普遍训练冲突。')
    lines+=decisions or ['诊断尚在执行，暂不据此选择机制。']
    lines+=['','当前保持 Phase 1–5 关闭。Phase 0 只提供描述性机制证据：可靠性差异需结合任务质量，梯度范数与方向应分别判断，原始/归一化阶数误差应同时审阅。若教师拟合普遍差，先考虑表示/数据/容量；若 fidelity 高但任务差，再分析错误传递。不得根据本次 Test 元数据或历史 Test 排名调权重。',
        '', '## 完成度与验证','',f'- 七子集 Train/Validation 导出：{sum(r["available"] for r in coverage)}/72。梯度 checkpoint：{len(list((OUT/"gradients").glob("*.json")))}/12。',
        f'- 已完成验证集回放 {len(replay)}/36，MAE 最大绝对偏差 {fmt(max((r["mae_abs_delta"] for r in replay if r["mae_abs_delta"] is not None),default=None))}。容许 0.02 用于检测迁移后模型/权重不兼容；bf16 和不同 batch size 可导致小幅数值差异，详见 `validation_replay.csv`。',
        '- 对每个导出断言样本 ID 完全覆盖、无重复、真实标签一致、预测有限；断言 (I_S − I_T) = A(f_S − f_T)，教师 ensemble / anchor / interaction mean 和 variance 可精确重建。',
        '- 四数据集教师 hidden 缓存均逐字节匹配冻结 SHA256；12 个 Probe checkpoint 另在固定 16 Train + 16 Valid parents 的全部七子集、全部窗口上做 CPU 回放，误差须小于 1e-4。',
        '- `checkpoint_inventory_audit.csv` 对审计 run 实际保留的每个权重文件计算当前 SHA256 并关联其 validation 指标；`checkpoint_history_coverage.csv` 区分历史 epoch 是否仍有权重。旧归档 SHA 另列，优化器剥离/重新序列化可能改变文件 SHA，不能将旧 SHA 当成当前验证。',
        '- `paired_bootstrap.csv` 与 `paired_method_differences.csv` 使用 source-video cluster bootstrap（2000 次，固定 seed130013），保留每个源视频所有 utterances。区间仅为历史 checkpoint 的描述性不确定性，不修复跨 split 的视频重叠，也不替代跨训练 seed 确认。',
        '- 每个诊断只用 autograd.grad；断言模型参数 version 未变，.grad 未填充；checkpoint 文件 size/mtime 未变。',
        '- 原生 SIMS 导出初次因遗漏 class_5_index 字段失败，已修复并通过原生加权窗口聚合校验；失败日志与重试记录保留在 `protocol/native_export_schema_failed_attempts.json`，不把失败导出算作完成。',
        '- 局限：只有一条 validation-selected 轨迹/seed；梯度只用 seed13 的少量 Train 窗口；无额外 Probe 重训；无新增 Test 因果比较；没有自动进入大规模训练。','',
        '## 产物','',
        '- `tables/protocol_audit.csv`、`protocol/asset_manifest.json`、`protocol/split_fingerprints.json`',
        '- `tables/teacher_subset_quality.csv`、`teacher_reliability_by_subset.csv`、`probe_agreement.csv`、`probe_agreement_by_coordinate.csv`',
        '- `tables/student_absorbability.csv`、`interaction_coordinate_error.csv`、`interaction_order_error.csv`、`sample_quadrants.csv`',
        '- `tables/gradient_compatibility.csv`、`student_task_quality.csv`、`seed_aggregates.csv`、`validation_replay.csv`、`coverage.csv`',
        '- `figures/`：Teacher MAE/Pearson 热图、交互阶数误差、梯度兼容性、Teacher–Student gap。',
        '- `predictions/`、`gradients/`：逐样本原始诊断；`source/`、`logs/`、`status/`：脚本、运行证据和完成状态。','']
    (OUT/'four_dataset_transferability_audit.md').write_text('\n'.join(lines))
    (OUT/'decisions/phase0_decision.md').write_text('# Phase 0 决策\n\n'+('\n'.join(decisions) or '运行中，尚不选择方法。')+'\n\nPhase 1–5 保持关闭。上述是描述性诊断，不是因果归因或自适应机制确认。\n')
    (OUT/'README.md').write_text('# Transferability Audit 202611\n\n仅 Phase 0；不启动大规模训练。\n\n[报告](four_dataset_transferability_audit.md) · [决策](decisions/phase0_decision.md) · [资产清单](protocol/asset_manifest.json) · [完成度](status/summary.json)\n')

if __name__=='__main__':main()
