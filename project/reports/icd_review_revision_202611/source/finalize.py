"""Dependency-triggered aggregation and acceptance. Stops after terminal job states."""
import argparse
import collections
import datetime
import fcntl
import shutil
import time
import numpy as np
from common import *
from trajectory_analysis import run as trajectory_analysis

SECTIONS=['sample_quadrants','subset_redundancy','incremental_utility','probe_stability','gradient_compatibility','interaction_order_utility','protocol_audit']
DATASETS=['mosi','mosei','sims','sims_v2']
def loadcsv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def val(r,k):
    try:return float(r[k])
    except (KeyError,TypeError,ValueError):return None
def fmt(x):return '缺失' if x is None else (f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x))
def mdtable(fields,data):
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']+['| '+' | '.join(fmt(r.get(k)).replace('|','/') for k in fields)+' |' for r in data])+'\n'
def merge():
    for section in SECTIONS:
        root=OUT/'phase0_5'/section;files=collections.defaultdict(list)
        for p in sorted((root/'by_dataset').glob('*/*.csv')):files[p.name].append(p)
        for name,paths in files.items():
            fields=[]
            for path in paths:
                with path.open() as f:fields.extend(x for x in (next(csv.reader(f),[])) if x not in fields)
            if not fields:continue
            tmp=root/(name+'.tmp')
            with tmp.open('w',newline='') as out:
                w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
                for path in paths:
                    with path.open() as f:w.writerows(csv.DictReader(f))
            os.replace(tmp,root/name)
        subreports=[]
        for p in sorted((root/'by_dataset').glob('*/report.md')):subreports.append(p.read_text())
        if subreports:(root/'report.md').write_text('\n\n'.join(subreports))

def gradients(plan):
    records=[];done=[];errors=[]
    for c in plan['gradient_cases']:
        path=OUT/f'gradients/{c["id"]}.json'
        if not path.exists():continue
        r=read(path)
        assert r['checkpoint_sha256']==c['checkpoint_sha256'] and r['epoch']==c['epoch']
        assert r['parameter_updates']==0 and r['parameter_versions_unchanged'] and len(r['samples'])==6
        done.append(dict(case=c['id'],dataset=c['dataset'],epoch=c['epoch'],stage=c['stage'],checkpoint_sha256=c['checkpoint_sha256'],device=r['device'],worker=r['worker'],peak_allocated_mib=r['peak_allocated_mib'],elapsed_seconds=r['elapsed_seconds']))
        for sample in r['samples']:
            for target,refs in sample['geometry'].items():
                for ref,groups in refs.items():
                    for group,values in groups.items():records.append(dict(case=c['id'],dataset=c['dataset'],method=c['method'],seed=c['seed'],epoch=c['epoch'],stage=c['stage'],sample_id=sample['sample_id'],reference=ref,target=target,parameter_group=group,**values))
    table('gradient_compatibility/gradient_cosines.csv',records);table('gradient_compatibility/gradient_norms.csv',records)
    bins=collections.defaultdict(list)
    for r in records:bins[tuple(r[k] for k in ['dataset','method','seed','epoch','stage','reference','target','parameter_group'])].append(r)
    summary=[]
    for key,items in bins.items():
        cosine=[r['cosine'] for r in items if r['cosine'] is not None];ratio=[r['norm_ratio'] for r in items if r['norm_ratio'] is not None]
        summary.append(dict(zip(['dataset','method','seed','epoch','stage','reference','target','parameter_group'],key),microbatches=len(items),defined_cosines=len(cosine),mean_cosine=float(np.mean(cosine)) if cosine else None,median_cosine=float(np.median(cosine)) if cosine else None,negative_cosine_fraction=float(np.mean(np.array(cosine)<0)) if cosine else None,mean_reference_norm=float(np.mean([r['base_norm'] for r in items])),mean_target_norm=float(np.mean([r['aux_norm'] for r in items])),mean_norm_ratio=float(np.mean(ratio)) if ratio else None))
    for name in ['negative_cosine_fraction','parameter_group_analysis','checkpoint_stage_analysis']:table(f'gradient_compatibility/{name}.csv',summary)
    table('gradient_compatibility/checkpoint_execution.csv',done)
    coverage=loadcsv(OUT/'phase0_5/gradient_compatibility/checkpoint_stage_coverage.csv');missing=[r for r in coverage if not r['available_epochs']]
    p=OUT/'phase0_5/gradient_compatibility/report.md'
    p.write_text(f'# A5 梯度诊断\n\n已产出 {len(done)}/{len(plan["gradient_cases"])} 个 checkpoint，每个六个固定 Train microbatch，batch=1，eval 模式、dropout 关闭。分别报告 Fusion、Text/Audio/Visual LoRA、Prediction Head、other 和全部参数。余弦零分母记缺失，不当作零或负数。原始 Full KD 使用原实验 seed2026 TAV target；Subset/ICD 使用原 ensemble3 target。N2 辅助项及总目标严格使用历史 ensemble3 TAV regression/probability、No-CE、0.30 ICD，并额外对 task_no_ce 计算余弦。原始与 N2 不混为统一 target 的受控训练。\n\n阶段预先按 epoch1–6 / 7–13 / 14+ 定义；{len(missing)} 个 case-stage 缺少保留权重。已使用所有 seed13 保留的不同 epoch，而没有重训补造 Early。完整 Early/Middle/Late 覆盖为受限。六个 batch 的负余弦比例是诊断，不能证明梯度投影必然改善任务。\n\n没有创建 optimizer，没有参数更新；checkpoint 哈希在加载前后核验，参数 version 未变化。\n')
    return done,summary,missing

def audit(plan):
    folder=OUT/'phase0_5/protocol_audit';checks=[]
    # Recheck all originally frozen assets, including metadata file hashes only for Test.
    for path,expected in plan['data_fingerprints'].items():
        p=Path(path);actual=sha(p) if p.is_file() else None
        checks.append(dict(path=path,expected_sha256=expected,current_sha256=actual,status='passed' if actual==expected else ('missing' if actual is None else 'changed'),scope='metadata fingerprint only' if 'test' in p.name else 'train/validation assets'))
    table('protocol_audit/asset_reverification.csv',checks)
    sourcefiles=[]
    n2root=Path(plan['n2']['run']).parents[1]
    for name in ['train_search.py','search_objectives.py','git_revision.txt','source_fingerprints.json']:
        p=n2root/name
        if p.exists():shutil.copy2(p,folder/('n2_'+name));sourcefiles.append(dict(path=str(p),sha256=sha(p)))
    n2cfg=read(OUT/'protocol/n2_run_config.json');n2impl=[]
    for path,expected in n2cfg.get('input_sha256',{}).items():
        if path.endswith('.py'):
            actual=sha(path) if Path(path).exists() else None;n2impl.append(dict(path=path,expected_sha256=expected,current_sha256=actual,match=actual==expected))
    save(folder/'n2_manifest.json',dict(**plan['n2'],student_seed=13,probe_seeds=[2026,2027,2028],historical_best_epoch=23,configured_max_epochs=n2cfg['epochs'],training_loss_implementation=sourcefiles,implementation_hash_checks=n2impl,loss_implementation_scope='historical frozen source copied verbatim; no student retraining'))
    teacher=read(folder/'teacher_manifest.json');save(folder/'teacher_target_manifest.json',dict(original_manifest=teacher,phaseA_protocols={d:dict(assets=p['assets'],assets_sha256=sha(p['assets']),probe_seeds=[q['seed'] for q in p['probes']],anchor=read(p['assets'])['student_empty_baseline'],anchor_definition='arithmetic mean of per-probe official Train TAV utterance means',transform='anchored Mobius; linear coefficient (-1)^(|target|-|source|); inverse verified',feature_config=p['feature_config']) for d,p in plan['protocols'].items()}))
    identities={d:rows(OUT/f'protocol/{d}_parent_identity.jsonl') for d in DATASETS if (OUT/f'protocol/{d}_parent_identity.jsonl').exists()}
    cross=[]
    if 'sims' in identities and 'sims_v2' in identities:
        a={r['parent_sample_id']:r for r in identities['sims']};b={r['parent_sample_id']:r for r in identities['sims_v2']};pa=set(a)&set(b);va={r['video_id'] for r in a.values()};vb={r['video_id'] for r in b.values()};shared=va&vb
        cross=[dict(comparison='SIMS_vs_SIMSv2_train_valid',shared_parent_ids=len(pa),sims_parent_fraction=len(pa)/len(a),sims_v2_parent_fraction=len(pa)/len(b),shared_source_video_ids=len(shared),sims_video_fraction=len(shared)/len(va),sims_v2_video_fraction=len(shared)/len(vb),shared_parent_cross_split=sum(a[k]['split']!=b[k]['split'] for k in pa),identifier_scope='official source-video IDs; parent matches audited upstream, ID overlap not proof all distinct IDs are independent media')]
        table('protocol_audit/cross_dataset_overlap.csv',cross)
    (folder/'config_consistency_report.md').write_text(f'# A7 数据与协议审计\n\n当前代码：`{plan["phaseA_git_commit"]}`；上游 Phase0 plan SHA256：`{plan["prior_plan_sha256"]}`。36 个既有最佳 checkpoint、72 份 Train/Validation 预测已核验一致；运行配置及 checkpoint SHA256 单独保存。历史训练 commit 只在 run_config 实际记录时填写，未记录不由审计 commit 代替。Teacher Probe seed 与 Student seed 分列。原始 Full KD seed2026 与 ICD ensemble3 额外 target 的混杂仍需 Phase B 统一 target 控制。\n\nN2：No-CE、lambda0.30、seed13、历史 best epoch23，配置预算40；不能称20-epoch复现。历史完整 loss 源码及配置已复制，target/probability规则保留。Test 只核对已有 metadata 字节指纹，不读取其预测或计算其任务效用。\n\nSIMS/SIMS v2 的 parent/source-video 重叠见 cross_dataset_overlap.csv，不能视为完全独立的复现。官方 Train/Validation 的 source-video 重叠见 source_video_overlap.csv；grouped CV 按 source-video切分，外部 Validation 另提供 novel-source子集。Teacher/Probe 已有拟合过程不会因为离线CV而变成独立验证，全部离线 utility 是条件诊断。\n\n资产复核异常数：{sum(r["status"]!="passed" for r in checks)}；完整清单见 asset_reverification.csv。\n')
    return checks,cross

def reports(plan,done,summary,missing,checks,cross,terminal):
    findings={d:read(OUT/f'protocol/{d}_findings.json') for d in DATASETS if (OUT/f'protocol/{d}_findings.json').exists()};gaps=[];q2=[];ranks=[];stability=[];order=[];decision=[]
    for d,f in findings.items():
        for r in f['gaps']:
            if r['seed']==13:gaps.append(dict(dataset=d,method=r['method'],epoch=r['epoch'],teacher_MAE=r['teacher_mae'],student_MAE=r['student_mae'],gap=r['student_minus_teacher_mae']))
        for r in f['rankings']:
            if r['utility']=='task' and r['exclude_tav']:
                cv=read(OUT/f'protocol/{d}_grouped_cv.json')
                # Validation of one novel-source sample cannot support a stable ranking.
                supported=r['cv_positive_fraction']==1 and r['valid_novel_source'] is not None and r['valid_novel_source']>0 and cv['novel_valid_parents']>=30 and cv.get('novel_valid_groups',cv['valid_groups']-cv['shared_train_valid_groups'])>=5
                ranks.append(dict(dataset=d,subset=r['subset'],CV_Shapley=r['cv_mean'],fold_std=r['cv_std'],positive_fraction=r['cv_positive_fraction'],valid_all=r['valid_all'],valid_novel=r['valid_novel_source'],stable_positive=supported))
        qpath=OUT/f'phase0_5/sample_quadrants/by_dataset/{d}/quadrant_metrics.csv'
        if qpath.exists():
            for r in loadcsv(qpath):
                if r['split']=='valid' and r['method']=='uniform_interaction' and r['seed']=='13' and r['quadrant']=='Q2':q2.append(dict(dataset=d,n=int(r['n']),fraction=val(r,'proportion_acc2_eligible'),teacher_MAE=val(r,'teacher_mae'),student_MAE=val(r,'student_mae'),label_abs_median=val(r,'label_abs_p50')))
        for r in loadcsv(OUT/f'phase0_5/subset_redundancy/by_dataset/{d}/effective_rank.csv'):
            if r['split']=='valid' and r['role']=='teacher' and r['seed']=='ensemble3' and r['standardized']=='False':decision.append(dict(dataset=d,space=r['space'],effective_rank=val(r,'effective_rank'),stable_rank=val(r,'stable_rank_centered_data'),condition_number=val(r,'condition_number')))
        ss=loadcsv(OUT/f'phase0_5/probe_stability/by_dataset/{d}/icc_analysis.csv')
        for r in ss:
            if r['split']=='valid' and r['coordinate']=='tav':stability.append(dict(dataset=d,space=r['space'],mean_probe_variance=val(r,'mean_probe_variance'),ICC=val(r,'icc_absolute_single'),coordinate_std=val(r,'ensemble_std')))
        for r in f['orders']:
            if r['split']=='valid_novel_source':order.append(dict(dataset=d,utility=r['utility'],through_order={0:0,7:1,63:2,127:3}[r['mask']],score=r['score'],n=r['n']))
    conditional=[];orderincrements=[]
    for d in findings:
        path=OUT/f'phase0_5/incremental_utility/by_dataset/{d}/conditional_marginal_utility.csv'
        rows_=loadcsv(path)
        for ex,n in [('True',6),('False',7)]:
            for k in range(n):
                mask=((1<<n)-1)^(1<<k)
                selected=[r for r in rows_ if r['exclude_tav']==ex and r['utility']=='task' and int(r['context_mask'])==mask and r['added_subset']==NAMES[k]]
                cv=[float(r['delta_utility']) for r in selected if r['split']=='grouped_cv']
                conditional.append(dict(dataset=d,exclude_tav=ex,subset=NAMES[k],cv_mean=float(np.mean(cv)),cv_min=min(cv),cv_max=max(cv),valid_novel=next((float(r['delta_utility']) for r in selected if r['split']=='valid_novel_source'),None)))
        f=findings[d]
        for kind in ['response','task']:
            for before,after in [(7,63),(63,127)]:
                a={r['fold']:r['score'] for r in f['orders'] if r['utility']==kind and r['mask']==before and r['split']=='grouped_cv'}
                b={r['fold']:r['score'] for r in f['orders'] if r['utility']==kind and r['mask']==after and r['split']=='grouped_cv'}
                delta=[b[k]-a[k] for k in a];orderincrements.append(dict(dataset=d,utility=kind,from_order=1 if before==7 else 2,to_order=2 if after==63 else 3,cv_mean_increment=float(np.mean(delta)),cv_min_increment=min(delta),cv_max_increment=max(delta),positive_fold_fraction=float(np.mean(np.array(delta)>0))))
    table('incremental_utility/full_context_marginal_summary.csv',conditional)
    table('interaction_order_utility/order_cv_increment.csv',orderincrements)
    observations=[]
    for d in findings:
        sd={r['space']:r for r in stability if r['dataset']==d}
        task=[r for r in orderincrements if r['dataset']==d and r['utility']=='task' and r['to_order']==3][0]
        observations.append(dict(dataset=d,residual_to_raw_TAV_probe_variance=sd['interaction']['mean_probe_variance']/sd['subset']['mean_probe_variance'],raw_TAV_ICC=sd['subset']['ICC'],residual_TAV_ICC=sd['interaction']['ICC'],third_order_task_cv_increment=task['cv_mean_increment'],third_order_positive_fold_fraction=task['positive_fold_fraction']))
    gradselect=[dict(dataset=r['dataset'],method=r['method'],epoch=r['epoch'],group=r['parameter_group'],mean_cosine=r['mean_cosine'],negative_fraction=r['negative_cosine_fraction'],norm_ratio=r['mean_norm_ratio']) for r in summary if r['reference']=='task' and r['target']=='icd_aux' and r['parameter_group']=='all']
    running=not terminal;fullcpu=len(findings)==4
    items=[dict(task='A1',status='完成' if fullcpu else '运行中/受限',evidence='四数据集逐样本 Acc-2 四象限、子集/阶数误差、source-video 汇总、固定规则案例'),dict(task='A2',status='完成' if fullcpu else '运行中/受限',evidence='Teacher各Probe/ensemble与Student分开；Pearson/Spearman、原始/标准化谱、秩、条件数'),dict(task='A3',status='完成（解释受限）' if fullcpu else '运行中/受限',evidence='127+empty组合、5外层/3内层 grouped CV、两utility、Shapley与去TAV敏感性；受Probe既有拟合及官方split重叠限制'),dict(task='A4',status='完成' if fullcpu else '运行中/受限',evidence='Probe-head方差、ICC、差值分布和尺度；不解释为独立Teacher不确定性'),dict(task='A5',status='运行中' if running and len(done)<len(plan['gradient_cases']) else '受限',evidence=f'{len(done)}/{len(plan["gradient_cases"])} checkpoint完成；{len(missing)}个预定阶段缺少历史权重'),dict(task='A6',status='完成（解释受限）' if fullcpu else '运行中/受限',evidence='归一化fidelity、两种精确reconstruction、grouped CV阶数utility及fidelity-task关联；无训练消融'),dict(task='A7',status='完成（历史记录受限）' if fullcpu else '运行中/受限',evidence='数据指纹、source/parent重叠、Teacher target、Student seed、checkpoint/config与N2审计；历史缺失git信息保留缺失'),dict(task='A8',status='已生成动态报告' if running else '完成',evidence='本报告与phaseB_decision.md；未启动Phase B训练')]
    text=f'''# Phase A 综合机制诊断与研究方向探索

生成时间：{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}。状态：{'实验运行中，当前报告为阶段结果；终态后自动重建' if running else '已汇总所有终态任务；严格完成度见下表'}。

代码 commit `{plan['phaseA_git_commit']}`；只复用既有 Teacher/Student，optimizer steps=0；未使用 Test 选择模型、子集、权重或研究方向。分析口径与冻结执行协议见 protocol/plan.json及后续amendment。

{mdtable(['task','status','evidence'],items)}
## 1. 四数据集 Teacher–Student 差距

以下仅列 seed13 的独立历史 run；不跨 campaign/target/预算拼接均值。Teacher列为 ensemble3 TAV，原始Full KD训练监督未必与该列相同。主表为原始 score MAE；SIMS官方clip口径另存 teacher_student_gap.csv。

{mdtable(['dataset','method','epoch','teacher_MAE','student_MAE','gap'],gaps)}
## 2. 四象限与误差模式

按 Acc-2 正确性划分，英语零标签排除后另记Q0，SIMS零包含。下面为ICD seed13 Validation的Q2；只定位最终预测差距，不认定某一坐标导致失败。全部Train/Validation、36模型、各象限标签强度及子集/阶数误差见 sample_quadrants/。

{mdtable(['dataset','n','fraction','teacher_MAE','student_MAE','label_abs_median'],q2)}
## 3. 七子集冗余与几何

Teacher ensemble Validation原始协方差谱如下。相关性和低有效秩不能单独证明子集无条件价值；Möbius变换可逆，不产生新Teacher信息。标准化谱、Teacher各Probe和全部Student分别保存。

{mdtable(['dataset','space','effective_rank','stable_rank','condition_number'],decision)}
## 4. 条件增量效用与Shapley

Train内嵌套source-video grouped CV：外5折、内3折，Ridge alpha=[.01,.1,1,10,100]，所有组合相同预算；标准化仅由训练折拟合。两种utility严格分开（response=-MSE，task=-MAE）。TAV作为response输入有近乎平凡的泄露目标性质，因此另列六玩家去TAV敏感性。以下task Shapley去TAV表，stable_positive要求所有外折和novel-source Validation增量均为正，且novel Validation至少30个样本/5个source groups；这是候选筛查而非显著性检验。SIMS无novel Validation，SIMS v2仅1条，均不得升级为稳定排名。

{mdtable(['dataset','subset','CV_Shapley','fold_std','positive_fraction','valid_all','valid_novel','stable_positive'],ranks)}
平均Shapley正值不等于给定所有其他子集后仍不可替代；完整上下文条件增量见full_context_marginal_summary.csv。是否删除某个子集必须结合这两类证据及真实训练消融。
## 5. Probe-head与残差稳定性

三Probe共用Teacher hidden cache；ICC(A,1)与方差是Probe-head敏感性，不能解释为独立Teacher epistemic uncertainty。下表比较最终TAV原始响应与三阶残差。

{mdtable(['dataset','space','mean_probe_variance','ICC','coordinate_std'],stability)}
## 6. Gradient compatibility

每checkpoint六个固定Train microbatch，eval/autograd、分参数组、无更新。原始目标保留其single-probe/ensemble差异；N2目标按历史No-CE+ensemble Full KD+0.30 ICD单独报告。以下ICD辅助项相对task全参数余弦；分组/范数比/中位数与负比例见 gradient_compatibility/。负余弦不是投影优化有效性的证据。

{mdtable(['dataset','method','epoch','group','mean_cosine','negative_fraction','norm_ratio'],gradselect)}
在SIMS的ICD保留checkpoint中，epoch18/20全参数task–ICD平均cosine约为-0.280/-0.311，六个microbatch中负余弦比例为4/6和5/6；MOSI对应ICD保留checkpoint的平均cosine仍为正。这支持优先检查SIMS上的loss compatibility及参数组，而不支持“所有数据集普遍梯度冲突”的结论。样本数仍很小，且Early/Middle覆盖不完整，不能据此直接宣称PCGrad会改善性能。
历史权重缺失 {len(missing)} 个阶段，Early/Middle/Late覆盖不能视为完整；未通过重训补造权重。

## 7. 不同阶可吸收性与任务价值

各阶MAE、RMSE及相对Teacher Train坐标std归一化RMSE均保存；数学重建分别报告最终TAV与全部七响应。下面是novel-source Validation累计阶数离线utility；空输入是Train均值，full third-order包括全部七坐标。该分析器的增益不是实际Student删除阶数的消融收益。样本fidelity-task Pearson/Spearman及象限差异见 interaction_order_utility/。另补充所有保留checkpoint上六个固定Train windows的阶数fidelity/task配对改善（checkpoint_order_trajectory.csv、checkpoint_fidelity_task_association.csv），并列历史Validation MAE。该轨迹仅覆盖六个窗口、非均匀保留epoch且有checkpoint选择偏差，不能推断因果或完整训练动力学。

{mdtable(['dataset','utility','through_order','score','n'],order)}
## 8. 数据与协议限制

{mdtable(['comparison','shared_parent_ids','sims_parent_fraction','sims_v2_parent_fraction','shared_source_video_ids','shared_parent_cross_split'],cross)}
官方Train/Validation源视频重叠另列source_video_overlap.csv；SIMS与v2不能描述为独立外部复现。原始single-Probe Full KD与ensemble额外目标混杂仍需统一target训练。历史N2是探索性seed13、epoch23、40epoch上限；不能充当20epoch确认结果。既有Probe可能用过Validation，因此外部Validation utility仍属于诊断；nested CV只保证离线analyzer的训练分组与调参隔离。

资产重核异常：{sum(r['status']!='passed' for r in checks)}。缺失历史checkpoint、git记录或其他资产均保留缺失，不替代、不伪造。

## 9. 审稿意见证据范围

N7yq：A2/A3/A6可提供七子集冗余和条件增量的分析证据；自适应选择、双模态泛化及真实收益仍需新增训练。QhEZ：明确Möbius为可逆监督几何、residual contrast术语及Probe ensemble细节；几何独特性、统一target公平比较仍需训练。2Z9A：梯度/fidelity/数据协议可核验；多Seed确认性收益、配对差值置信区间和Matched-Spectrum几何控制不能由这些离线结果替代。

## 10. 研究决策

以下是直接由已计算数据形成的机制线索：

{mdtable(['dataset','residual_to_raw_TAV_probe_variance','raw_TAV_ICC','residual_TAV_ICC','third_order_task_cv_increment','third_order_positive_fold_fraction'],observations)}
四数据集的原始七响应都呈低有效秩，但去TAV的六个子集task Shapley均在五个外折为正；这支持“存在冗余但条件上下文仍有价值”的检验假设，不支持仅凭相关性删除子集。三阶residual的Probe-head方差高于原始TAV且ICC较低，支持比较尺度归一化与匹配谱控制，尚不证明新几何会改善任务。第三阶的task utility增量应结合逐折符号与外部novel-source检查；完整response重建接近零误差来自Möbius可逆性，不能当作新发现或新信息。

见 phaseB_decision.md。Phase A没有预设新方法，不把离线Shapley当作训练权重收益，不把相关性当作因果贡献。下一阶段以统一target、20epoch和独立Student seeds进行验证；Phase B训练尚未授权启动。
'''
    (OUT/'phaseA_final_report.md').write_text(text)
    supported=[r for r in ranks if r['stable_positive']]
    b=f'''# Phase B 决策与最小可检验矩阵

当前Phase A状态：{'部分梯度仍运行，决策为候选清单' if running else '分析终态；存在资产/解释限制，不能宣称所有诊断完整'}。

## 证据驱动候选

{mdtable(['dataset','residual_to_raw_TAV_probe_variance','raw_TAV_ICC','residual_TAV_ICC','third_order_task_cv_increment','third_order_positive_fold_fraction'],observations)}
具体可证伪假设H1：尺度敏感性导致部分ICD优化困难；若统一target/预算下normalization不优于diagonal与matched-spectrum控制，则不接受其独特机制。H2：高Shapley但完整上下文低增量意味着监督冗余；只有固定Train内选择且真实训练删减不损失任务，才接受subset reduction。H3：阶数fidelity与任务收益并不等价；只有真实阶数loss消融与离线方向一致才接受Order-aware机制。H4：MOSI较大Teacher–Student gap可能与loss balance有关；若FullKD-NoCE与ICD-NoCE改善相近，不能把NoCE收益归于ICD。

去TAV的task Shapley在所有外层fold及novel-source Validation均为正的坐标如下；如为空，则不支持升级Utility-aware主方法。正增量也只支持下一轮训练假设，不说明训练权重应直接等于该数值。

{mdtable(['dataset','subset','CV_Shapley','fold_std','valid_novel'],supported)}
1. Utility-aware：只在各折和novel-source检查稳定时提出候选；候选冻结后，用相同初始化、target、预算与uniform ICD配对训练。若排名不稳定，保留均匀基线。
2. Stable Geometry：联合Probe-head variance、ICC及各坐标scale检验，不因第三阶原始幅度大就判为噪声；比较统一尺度normalization与matched-spectrum/diagonal控制。不能解释为增加Teacher信息。
3. Order-aware/Absorbability-aware：只有归一化拟合难度高且task增量稳定时考虑absorbability；若高阶task价值小且跨折一致，再提出reduced-order。验证为实际loss消融，离线reconstruction不能代替。
4. Gradient-compatible：SIMS ICD在epoch18/20的task–ICD全参数均值均为负（约-0.280/-0.311，负比例4/6和5/6），可优先检验loss balance及参数组冲突；MOSI保留ICD checkpoints的均值为正，不提出普遍冲突。若配对训练的梯度投影/平衡不优于同等target的NoCE或尺度控制，则否定其额外机制。缺少Early权重时结论局限于保留阶段。
5. Fidelity不等于Task：同时报告response与task utility、样本fidelity-task关联及不同象限；若方向不一致，不按Teacher fidelity单独分配权重。

## 最小确认矩阵（待研究决策；未启动）

| 阶段 | 数据/预算 | 方法 | 要回答的问题 |
| --- | --- | --- | --- |
| B0 loss balance | MOSI，统一20epochs，Student13/42/2026，固定共同ensemble3 target/初始化 | Full KD、Subset7、ICD × CE有/无；NoCE ICD lambda0.30与CE ICD lambda0.30同权重配对，另保留历史lambda1参照 | No-CE收益是否独立于ICD；lambda/CE是否混杂；N2能否在20epoch确认 |
| B1 geometry | B0共同target/预算/seed，冻结同一CE状态与权重 | ICD、direct Subset7、Random Orthogonal、Diagonal、Matched-Spectrum | 改善是否由Möbius结构而非一般几何/尺度带来 |
| B2 一个机制候选 | 由Phase A最稳定证据选一个候选，MOSI先验冻结，3seeds | uniform ICD对一个新候选及相应单因素消融 | 新信号是否带来真实Student收益；不并行盲搜多个Adaptive方法 |
| B3 外部迁移/双模态 | MOSEI、SIMS/v2及双模态；相同配对seed和预算 | 只迁移B0/B2确认的最小方法组 | 跨数据收益与双模态适用性；SIMS/v2重叠分层披露 |

所有方法以Validation MAE选checkpoint，记录best epoch和epoch上限。Teacher Probe seeds不冒充Student seeds；不同campaign不拼接均值。方法和主要指标预先冻结后，Test仅作一次描述/确认审计，不反向选方法。B0共7种配置×3seeds=21次训练；B1复用已完成基线，仅新增3种几何控制×3seeds=9次训练。后续候选一次只增加一个机制。配对置信区间按source-video bootstrap，并同时报告每个Student seed的配对差值；不要以样本bootstrap代替训练seed不确定性。训练结束后再考虑方法升级。

## 未解决项

Early/Middle/Late缺失历史权重的阶段无法由Phase A补齐；历史run未记录的commit等字段保持缺失。独立Teacher不确定性、双模态训练、统一target多seed、20epoch N2复现和几何因果作用均需要Phase B新训练。当前不预先把N2升级为主方法。
'''
    (OUT/'phaseB_decision.md').write_text(b)
    save(OUT/'status/acceptance.json',dict(status='running' if running else ('complete_with_limitations' if len(done)==len(plan['gradient_cases']) and fullcpu else 'incomplete'),items=items,gradient_completed=len(done),gradient_expected=len(plan['gradient_cases']),missing_checkpoint_stages=missing,analysis_datasets=list(findings),parameter_updates=0,test_predictions_used=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');arg=ap.parse_args();lock=(OUT/'protocol/finalizer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(OUT/'protocol/plan.json');start=time.time();terminal=False
    if arg.wait:
        while time.time()-start<48*3600:
            paths=[OUT/'status'/name for name in ['cpu_worker.json','worker_Q.json','worker_Qiii.json','worker_A100.json','worker_Q_trajectory.json','worker_Qiii_trajectory.json','worker_A100_trajectory.json']]
            terminal=all(p.exists() and read(p)['status'] in ['complete','failed'] for p in paths)
            if terminal:break
            save(OUT/'status/finalizer.json',dict(status='waiting_for_dependencies',pid=os.getpid(),started_unix=start,elapsed_seconds=time.time()-start))
            time.sleep(30)
    else:
        paths=[OUT/'status'/name for name in ['cpu_worker.json','worker_Q.json','worker_Qiii.json','worker_A100.json','worker_Q_trajectory.json','worker_Qiii_trajectory.json','worker_A100_trajectory.json']]
        terminal=all(p.exists() and read(p)['status'] in ['complete','failed'] for p in paths)
    merge();done,summary,missing=gradients(plan);checks,cross=audit(plan);trajectory_count,trajectory_associations=trajectory_analysis(plan);reports(plan,done,summary,missing,checks,cross,terminal)
    counts={}
    for d in DATASETS:
        p=OUT/f'phase0_5/sample_quadrants/{d}_quadrants.csv'
        if p.exists():
            actual=collections.Counter()
            with p.open() as f:
                for row in csv.DictReader(f):actual[(row['case'],row['split'])]+=1
            for c in [c for c in plan['cases'] if c['dataset']==d]:
                for split in ['train','valid']:assert actual[(c['id'],split)]==plan['protocols'][d]['parent_counts'][split]
            counts[d]=sum(actual.values())
    files={str(p.relative_to(OUT)):sha(p) for directory in ['phase0_5','source','protocol'] for p in (OUT/directory).rglob('*') if p.is_file() and p.suffix in ['.csv','.json','.jsonl','.py','.md']}
    for name in ['phaseA_final_report.md','phaseB_decision.md']:files[name]=sha(OUT/name)
    save(OUT/'status/final_verification.json',dict(status='passed' if terminal and len(done)==len(plan['gradient_cases']) and len(counts)==4 and trajectory_count==len(plan['gradient_cases']) and all(r['status']=='passed' for r in checks) else 'partial',terminal=terminal,sample_rows=counts,gradient_completed=len(done),trajectory_checkpoints=trajectory_count,optimizer_steps=0,test_predictions_used=False,sha256=files))
    save(OUT/'status/finalizer.json',dict(status='complete' if terminal else 'partial_report_generated',finished_unix=time.time(),final_report=str(OUT/'phaseA_final_report.md')))
    print(json.dumps(dict(terminal=terminal,gradient_completed=len(done),sample_rows=counts)),flush=True)

if __name__=='__main__':main()
