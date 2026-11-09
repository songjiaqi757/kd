#!/usr/bin/env python3
"""Independently recompute validation-only confirmatory results and audit artifacts."""
from __future__ import annotations
import csv, hashlib, json, math, os
from pathlib import Path
import numpy as np
ROOT=Path("/data/sjq/kd")
BASE=ROOT/"outputs/experiments/icd_confirmatory_mosi_v1"
def read(path):
    return json.loads(path.read_text())
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda:stream.read(8*1024**2),b""):
            h.update(data)
    return h.hexdigest()
def f1(truth,pred):
    value=0.
    for label in [False,True]:
        tp=np.sum((truth==label)&(pred==label))
        fp=np.sum((truth!=label)&(pred==label))
        fn=np.sum((truth==label)&(pred!=label))
        support=np.sum(truth==label)
        value+=(support/len(truth))*(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return float(value)
def metrics(records):
    y=np.asarray([r["target_sentiment"] for r in records],dtype=float)
    p=np.asarray([r["prediction"] for r in records],dtype=float)
    if y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid predictions")
    keep=y!=0
    pred=p[keep]>=0
    truth=y[keep]>0
    den=np.linalg.norm(y-y.mean())*np.linalg.norm(p-p.mean())
    round7=lambda a:np.clip(np.sign(a)*np.floor(np.abs(a)+.5),-3,3)
    return dict(count=len(y),mae=float(np.abs(y-p).mean()),
        pearson=float(np.dot(y-y.mean(),p-p.mean())/den) if den else None,
        acc2_nonzero=float(np.mean(pred==truth)),
        f1_weighted_nonzero=f1(truth,pred),nonzero_count=int(keep.sum()),
        acc7=float(np.mean(round7(y)==round7(p))),
        prediction_bias=float((p-y).mean()),prediction_mean=float(p.mean()),
        prediction_std=float(p.std()),target_mean=float(y.mean()),target_std=float(y.std()))
def canonical(records):
    lookup={r["parent_sample_id"]:r for r in records}
    if len(lookup)!=len(records):
        raise ValueError("Duplicate utterance predictions")
    if any(r["split"] not in {"valid","validation","dev"} for r in records):
        raise ValueError("Non-validation predictions encountered")
    return lookup
def paired(left,right):
    a,b=left["predictions"],right["predictions"]
    if set(a)!=set(b):
        raise ValueError("Unmatched validation utterances")
    key=sorted(a)
    if any(a[k]["video_id"]!=b[k]["video_id"] or
           a[k]["target_sentiment"]!=b[k]["target_sentiment"] for k in key):
        raise ValueError("Unmatched labels/source videos")
    # Positive means right has lower MAE than left.
    deltas=np.asarray([abs(a[k]["prediction"]-a[k]["target_sentiment"])-
                      abs(b[k]["prediction"]-b[k]["target_sentiment"]) for k in key])
    vids=np.asarray([a[k]["video_id"] for k in key])
    clusters=sorted(set(vids))
    sums=np.asarray([deltas[vids==v].sum() for v in clusters])
    counts=np.asarray([np.sum(vids==v) for v in clusters])
    rng=np.random.default_rng(20261108)
    draws=rng.integers(0,len(clusters),size=(10000,len(clusters)))
    draws_mean=sums[draws].sum(1)/counts[draws].sum(1)
    return dict(left=left["id"],right=right["id"],mae_improvement=float(deltas.mean()),
        source_video_count=len(clusters),
        source_video={str(v):dict(count=int(n),mae_improvement=float(s/n))
            for v,s,n in zip(clusters,sums,counts)},
        clustered_bootstrap=dict(replicates=10000,seed=20261108,
            percentile_95=[float(x) for x in np.quantile(draws_mean,[.025,.975])],
            description="paired source-video clustered resampling, pooled utterance MAE",
            caution="Only 10 validation clusters; interval is exploratory and not a robust significance claim."))
def main():
    analysis=BASE/"analysis"
    analysis.mkdir(parents=True,exist_ok=True)
    allruns=[]
    errors=[]
    identities=None
    expected_init=None
    for group in ["runs","scheduler","gradient_projected"]:
        for out in sorted((BASE/group).glob("*")):
            if not out.is_dir() or not (out/"report.json").exists():
                continue
            report=read(out/"report.json")
            config=read(out/"run_config.json")
            history=read(out/"history.json")
            predictions=canonical(rows(out/"predictions.jsonl"))
            m=metrics(list(predictions.values()))
            for name in ["mae","pearson","acc2_nonzero","f1_weighted_nonzero","acc7","count"]:
                actual=m[name]
                reported=report["valid_metrics"][name]
                if actual is None or not math.isclose(float(actual),float(reported),rel_tol=1e-8,abs_tol=1e-8):
                    errors.append(out.name+": independently recomputed "+name+" mismatch")
            if report.get("official_test_evaluated") is not False:
                errors.append(out.name+": test policy violation")
            if len(predictions)!=229 or len({r["video_id"] for r in predictions.values()})!=10:
                errors.append(out.name+": expected 229 utterances and 10 source videos")
            current={k:(v["target_sentiment"],v["video_id"]) for k,v in predictions.items()}
            if identities is None:
                identities=current
            elif identities!=current:
                errors.append(out.name+": prediction sample identity mismatch")
            epoch=int(report["best_epoch"])
            best_score=float("inf")
            selected=0
            for row in history:
                if row["valid_metrics"]["mae"]<best_score-1e-5:
                    best_score=row["valid_metrics"]["mae"]
                    selected=row["epoch"]
            if epoch!=selected or not math.isclose(m["mae"],best_score,abs_tol=1e-8):
                errors.append(out.name+": checkpoint selection/re-evaluation mismatch")
            checkpoint_files=sorted(out.rglob("*.pt"))
            expected_files={"best_valid_mae.pt","best_valid_acc2_nonzero.pt","last.pt"}
            if {str(p.relative_to(out)) for p in checkpoint_files} != expected_files:
                errors.append(out.name+": retained checkpoint set must contain exactly MAE-best, Acc2-best and last")
            retained=report.get("retained_best_checkpoints") or {}
            expected_acc2=-float("inf")
            selected_acc2=0
            for row in history:
                value=row["valid_metrics"]["acc2_nonzero"]
                if value > expected_acc2+1e-5:
                    expected_acc2=value
                    selected_acc2=row["epoch"]
            if retained.get("valid_mae",{}).get("epoch") != selected:
                errors.append(out.name+": MAE retained checkpoint metadata mismatch")
            if retained.get("valid_acc2_nonzero",{}).get("epoch") != selected_acc2:
                errors.append(out.name+": Acc2 retained checkpoint metadata mismatch")
            init=read(out/"initialization.json") if (out/"initialization.json").exists() else {}
            init_hash=init.get("initialization_sha256",init.get("trainable_state_sha256",init.get("sha256")))
            if init_hash is None:
                errors.append(out.name+": missing initialization SHA256")
            elif expected_init is None:
                expected_init=init_hash
            elif init_hash!=expected_init:
                errors.append(out.name+": nonidentical seed13 initialization")
            if report.get("total_parameters")!=819309176 or report.get("trainable_parameters")!=42650120:
                errors.append(out.name+": parameter counts differ from frozen protocol")
            ce=float(config.get("task_ce",0))
            lam=float(config.get("lambda_interaction",0))
            run=dict(id=out.name,group=group,output=str(out),method=report["method"],
                ce=ce,lambda_icd=lam,scheduler=config.get("lr_schedule","constant"),
                best_epoch=epoch,epochs_run=len(history),seed=report["seed"],
                stop_reason=report.get("stop_reason"),metrics=m,
                initialization_sha256=init_hash,checkpoint_count=len(checkpoint_files),
                report_sha256=sha(out/"report.json"),config_sha256=sha(out/"run_config.json"),
                predictions_sha256=sha(out/"predictions.jsonl"),predictions=predictions,
                history=history,diagnostics=read(out/"validation_diagnostics.json"))
            allruns.append(run)
    factorial=[r for r in allruns if r["group"]=="runs"]
    if len(factorial)!=8:
        errors.append("Expected all eight factorial runs")
    grid={(r["ce"],r["lambda_icd"]):r for r in factorial}
    effects=dict(ce_main_effect=[],icd_main_effect=[],interaction_effect=[])
    comparisons=[]
    for lam in [0.,.25,.5,1.]:
        if (.5,lam) in grid and (0.,lam) in grid:
            pair=paired(grid[(.5,lam)],grid[(0.,lam)])
            effects["ce_main_effect"].append(dict(lambda_icd=lam,
                delta_ce=pair["mae_improvement"],interpretation="positive favors removing task CE",
                paired_analysis=pair))
            comparisons.append(pair)
    for ce in [0.,.5]:
        for lam in [.25,.5,1.]:
            if (ce,0.) in grid and (ce,lam) in grid:
                pair=paired(grid[(ce,0.)],grid[(ce,lam)])
                effects["icd_main_effect"].append(dict(ce=ce,lambda_icd=lam,
                    delta_icd=pair["mae_improvement"],interpretation="positive favors ICD",
                    paired_analysis=pair))
                comparisons.append(pair)
    for lam in [.25,.5,1.]:
        if all(k in grid for k in [(0.,0.),(0.,lam),(.5,0.),(.5,lam)]):
            noce=grid[(0.,0.)]["metrics"]["mae"]-grid[(0.,lam)]["metrics"]["mae"]
            ce=grid[(.5,0.)]["metrics"]["mae"]-grid[(.5,lam)]["metrics"]["mae"]
            effects["interaction_effect"].append(dict(lambda_icd=lam,
                icd_gain_no_ce=noce,icd_gain_ce=ce,
                difference_in_differences=noce-ce,
                interpretation="positive means ICD helps more after removing task CE"))
    for run in allruns:
        if run["group"]=="gradient_projected":
            key=(run["ce"],run["lambda_icd"])
            if key in grid:
                comparisons.append(paired(grid[key],run))
    selection=read(BASE/"selection.json")
    sched=[r for r in allruns if r["group"]=="scheduler"]
    extension=BASE/"scheduler/extension_policy.json"
    excluded=set()
    if extension.exists():
        policy=read(extension)
        excluded.add(policy["excluded_from_final_pair"])
        byname={r["id"]:r for r in sched}
        if policy["fixed"]["id"] in byname and policy["cosine"]["id"] in byname:
            comparisons.append(paired(byname[policy["fixed"]["id"]],byname[policy["cosine"]["id"]]))
    else:
        byname={r["id"]:r for r in allruns}
        if selection["phase1_best"]["id"] in byname and "S1_warmup_cosine" in byname:
            comparisons.append(paired(byname[selection["phase1_best"]["id"]],byname["S1_warmup_cosine"]))
    candidates=[r for r in allruns if r["id"] not in excluded]
    if not candidates:
        raise RuntimeError("No completed runs to analyze")
    raw_best=min(candidates,key=lambda r:r["metrics"]["mae"])
    nearby=[r for r in candidates if r["metrics"]["mae"]<=raw_best["metrics"]["mae"]+.005]
    simplicity=lambda r:(r["lambda_icd"]>0,"gradient_projected" in r["method"],
                        r["scheduler"]!="constant",r["lambda_icd"],r["metrics"]["mae"])
    preferred=min(nearby,key=simplicity)
    method_choice=dict(raw_minimum_validation_mae=raw_best["id"],
        engineering_preference_within_0_005=preferred["id"],
        within_0_005_candidates=[r["id"] for r in nearby],
        interpretation="Single seed13 development decision; engineering thresholds are not statistical significance.",
        minimum_research_advantage_mae=.01,indistinguishability_threshold_mae=.005,
        deferred=["MOSEI cross-dataset validation","multi-seed confirmation"],
        official_test_reporting=("completed after training (reporting only)" if (analysis/"official_test_summary.json").exists() else "pending"))
    independent=[]
    for r in factorial:
        if r["lambda_icd"]<=0:
            continue
        gain=grid[(r["ce"],0.)]["metrics"]["mae"]-r["metrics"]["mae"]
        independent.append(dict(id=r["id"],matched_full_kd=grid[(r["ce"],0.)]["id"],
            mae_improvement=gain,passes_0_01_screen=gain>=.01,
            pearson_change=r["metrics"]["pearson"]-grid[(r["ce"],0.)]["metrics"]["pearson"],
            acc2_change=r["metrics"]["acc2_nonzero"]-grid[(r["ce"],0.)]["metrics"]["acc2_nonzero"],
            f1_change=r["metrics"]["f1_weighted_nonzero"]-grid[(r["ce"],0.)]["metrics"]["f1_weighted_nonzero"]))
    method_choice["matched_icd_screen"]=independent
    save(analysis/"method_selection.json",method_choice)
    save(analysis/"factorial_analysis.json",effects)
    save(analysis/"paired_source_video_analysis.json",dict(comparisons=comparisons,
        caution="Validation-based configuration selection and only 10 clusters limit inferential claims."))
    interactions=[]
    gradients=[]
    gradient_records={}
    for run in allruns:
        out=Path(run["output"])
        if (out/"validation_interactions.jsonl").exists():
            recs=rows(out/"validation_interactions.jsonl")
            diff=np.asarray([np.asarray(r["student_interactions"])-np.asarray(r["teacher_interactions"]) for r in recs])
            coordinate=np.abs(diff).mean(0)
            interactions.append(dict(id=run["id"],coordinate_mae=coordinate.tolist(),
                first_order_mae=float(coordinate[:3].mean()),
                second_order_mae=float(coordinate[3:6].mean()),
                third_order_mae=float(coordinate[6]),
                task_mae=run["metrics"]["mae"],
                label_strata=run["diagnostics"].get("label_strata"),
                interaction_strength_strata=run["diagnostics"].get("interaction_strength_strata"),
                source_video=run["diagnostics"].get("source_video")))
        diag=out/"diagnostics/gradient_geometry.jsonl"
        if diag.exists():
            data=rows(diag)
            gradient_records[run["id"]]=data
            if [x["epoch"] for x in data] != [x["epoch"] for x in run["history"]]:
                errors.append(run["id"]+": fixed diagnostic epoch coverage mismatch")
            summaries={}
            for scope in ["all","fusion","encoder_lora","pools_other"]:
                if not all(scope in x.get("parameter_groups",{}) for x in data):
                    errors.append(run["id"]+": missing gradient parameter scope "+scope)
                    continue
                summaries[scope]=dict(
                    mean_raw_gradient_norms=np.mean([x["parameter_groups"][scope]["gradient_norms"] for x in data],axis=0).tolist(),
                    mean_raw_gradient_cosines=np.mean([x["parameter_groups"][scope]["gradient_cosines"] for x in data],axis=0).tolist())
            gradients.append(dict(id=run["id"],path=str(diag),sha256=sha(diag),
                records=len(data),sample_positions=[{"epoch":x.get("epoch"),"position":x.get("position"),
                    "sample_ids":x.get("sample_ids")} for x in data],
                terms=data[0]["terms"] if data else [],
                parameter_scope_summaries=summaries,
                interpretation="Descriptive raw gradient geometry; not causal proof."))
        else:
            errors.append(run["id"]+": missing fixed all-parameter gradient diagnostics")
    if gradient_records:
        shared_epochs=set.intersection(*[{x["epoch"] for x in rs} for rs in gradient_records.values()])
        for epoch in sorted(shared_epochs):
            aligned=[next(x for x in rs if x["epoch"]==epoch) for rs in gradient_records.values()]
            reference=aligned[0]
            if any(x.get("sample_ids")!=reference.get("sample_ids") or
                   x.get("position")!=reference.get("position") or
                   x.get("diagnostic_seed")!=reference.get("diagnostic_seed") for x in aligned[1:]):
                errors.append("Fixed diagnostic batches/positions differ at epoch "+str(epoch))
    source_video_records=[]
    if identities:
        for video in sorted({v[1] for v in identities.values()}):
            keys=sorted(k for k,v in identities.items() if v[1]==video)
            record=dict(video_id=video,count=len(keys))
            for run in allruns:
                record[run["id"]]=float(np.mean([abs(run["predictions"][k]["prediction"]-
                    run["predictions"][k]["target_sentiment"]) for k in keys]))
            source_video_records.append(record)
        with (analysis/"source_video_error_matrix.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=["video_id","count"]+[r["id"] for r in allruns])
            writer.writeheader()
            writer.writerows(source_video_records)
        save(analysis/"source_video_error_matrix.json",dict(
            common_validation_utterances=len(identities),source_videos=len(source_video_records),
            alignment="identical parent_sample_id, target_sentiment and video_id in every run",
            rows=source_video_records))
    save(analysis/"interaction_analysis.json",dict(runs=interactions,
        note="Interaction coordinate errors are reported separately from task MAE."))
    save(analysis/"gradient_diagnostic_manifest.json",dict(runs=gradients))
    fieldnames=["id","group","method","seed","ce","lambda_icd","scheduler",
        "best_epoch","epochs_run","mae","pearson","acc2_nonzero","f1_weighted_nonzero",
        "acc7","prediction_bias","prediction_std","initialization_sha256","stop_reason"]
    with (analysis/"per_run.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames)
        writer.writeheader()
        for r in allruns:
            values={**r,**r["metrics"]}
            writer.writerow({k:values.get(k) for k in fieldnames})
    save(analysis/"independent_metrics.json",dict(runs=[{k:v for k,v in r.items()
        if k not in ["predictions","history","diagnostics"]} for r in allruns],
        tolerance=1e-8,errors=errors))
    plot_errors=[]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(13,4.5))
        for r in factorial:
            x=[x["epoch"] for x in r["history"]]
            axs[0].plot(x,[x["valid_metrics"]["mae"] for x in r["history"]],label=r["id"].split("_")[0])
            axs[1].plot(x,[x["train_loss"] for x in r["history"]],label=r["id"].split("_")[0])
        axs[0].set(xlabel="Epoch",ylabel="Validation MAE",title="MAE-selected checkpoints")
        axs[1].set(xlabel="Epoch",ylabel="Training objective",title="Different loss weights: objectives have different scales")
        for ax in axs:
            ax.legend(ncol=2)
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(analysis/"factorial_training_curves.png",dpi=180)
        plt.close(fig)
    except Exception as exc:
        plot_errors.append(repr(exc))
    lines=["# MOSI 第二阶段确认实验报告","",
        "全部结果使用 seed13，Validation MAE 选取同一个 checkpoint；Official-test 未参与训练、早停或任何配置选择，仅在全部训练结束后用于报告。所有资产保存在 Qi 服务器。",
        "","## 完整结果","",
        "|Run|CE|ICD λ|Scheduler|Best epoch|MAE|Pearson|Acc-2 nonzero|Weighted F1 nonzero|",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in allruns:
        m=r["metrics"]
        lines.append(f"|{r['id']}|{r['ce']}|{r['lambda_icd']}|{r['scheduler']}|{r['best_epoch']}|{m['mae']:.4f}|{m['pearson']:.4f}|{m['acc2_nonzero']:.4f}|{m['f1_weighted_nonzero']:.4f}|")
    lines+=["","## CE、ICD 与交互效应","","正的 CE 效应表示移除真实标签 CE 改善 MAE；教师 KL 始终保留。"]
    for effect in effects["ce_main_effect"]:
        lines.append(f"- λ={effect['lambda_icd']:g}: ΔCE={effect['delta_ce']:+.4f}。")
    for effect in effects["icd_main_effect"]:
        lines.append(f"- CE={effect['ce']:g}, λ={effect['lambda_icd']:g}: ΔICD={effect['delta_icd']:+.4f}。")
    for effect in effects["interaction_effect"]:
        lines.append(f"- λ={effect['lambda_icd']:g}: No-CE ICD 收益减去 CE ICD 收益={effect['difference_in_differences']:+.4f}。")
    lines+=["","## 开发阶段方法选择","",
        f"原始最低 Validation MAE 为 {raw_best['id']}（{raw_best['metrics']['mae']:.4f}）。",
        f"依据 MAE 差不超过 0.005 时优先简单方案的工程规则，当前建议候选为 {preferred['id']}（{preferred['metrics']['mae']:.4f}）。",
        "上述阈值不是统计显著性阈值；候选集合、匹配 Full KD 的至少 0.01 MAE 改善筛选和辅助指标变化见 method_selection.json。",
        "固定 LR 的 S0 直接引用对应 Phase1 run；若调度预算扩展，则 S0/S1 均从相同初始化重跑，排除原 40 epoch cosine 结果。",
        "GP 与同 CE、同 λ、同训练预算的普通 ICD 比较；仅梯度路由改变。GP 两种 CE 条件均保留，防止把 CE 的变化归因于投影。",
        "","## Official-test 最终评估（仅报告）","",
    ]
    test_path = analysis / "official_test_summary.json"
    if test_path.exists():
        test_summary = read(test_path)
        test_rows = [r for r in test_summary.get("runs", [])
                     if r.get("round") == "round2" and r.get("policy") == "valid_mae"]
        test_rows.sort(key=lambda r: r["test_mae"])
        lines += [
            f"测试清单 `{test_summary.get('test_manifest')}`（686 windows / 686 utterances）。"
            "下表 checkpoint 仍由 Validation MAE 选出；Acc-2 最优 checkpoint 及第一轮结果"
            "见 `official_test_summary.csv`。",
            "",
            "|Run|checkpoint|epoch|valid MAE|test MAE|test Pearson|test Acc-2|",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in test_rows:
            lines.append(
                f"|{row['run']}|{row['policy']}|{row['selected_epoch']}|"
                f"{row['valid_mae']:.4f}|{row['test_mae']:.4f}|"
                f"{row['test_pearson']:.4f}|{row['test_acc2_nonzero']:.4f}|")
        lines += ["", "测试集排名不作为方法选择依据；单 seed、686 条测试样本上的差异仍需多种子确认。"]
    lines += [
        "","## 机制、曲线及不确定性","",
        "七交互坐标、一/二/三阶重建误差与任务 MAE 分别见 interaction_analysis.json。分情感区间、交互强度和 source-video 误差均来自同一 MAE-selected checkpoint。",
        "逐样本预测独立重算全部主要指标，并核对 229 utterances、10 source videos、初始化哈希、参数数量、完整逐轮 history、恰好三个保留 checkpoint 与 1e-5 选择阈值。",
        "配对 source-video clustered bootstrap 使用 10,000 次重采样。只有 10 个 validation clusters，区间仅用于描述不确定性；不能解释成多种子确认。",
        "梯度范数与方向只描述训练现象，不作为因果证明。训练目标因 loss 权重不同，其数值不能直接跨组比较。",
        "","## 后续阶段","",
        "Official-test 最终评估已按协议在全部训练结束后完成，仅用于报告；MOSEI 跨数据集验证与多种子确认尚未执行，本报告不宣称跨数据集普适性。",
        "","## 完成审计","",
        f"审计发现 {len(errors)} 项错误；详情见 ../completion_audit.json。"]
    if errors:
        lines+=[""]+["- "+e for e in errors]
    (analysis/"final_report.md").write_text("\n".join(lines)+"\n")
    paper=["# ICLR 实验文字草稿","",
        "We conducted a validation-only 2 × 4 factorial study on CMU-MOSI at seed 13, varying the ground-truth classification weight c ∈ {0, 0.5} and interaction distillation weight λ ∈ {0, 0.25, 0.5, 1}. All runs share three teacher probes, the student initialization, video-disjoint train/validation split, and validation-MAE checkpoint selection. Removing task CE retains teacher KL supervision.",
        f"The raw minimum validation MAE was {raw_best['metrics']['mae']:.4f} ({raw_best['id']}). This single-seed development result does not establish cross-dataset or multi-seed robustness. Paired source-video bootstrap analysis describes uncertainty across the ten validation videos.",
        "","The main and ablation tables are available as per_run.csv. All auxiliary metrics refer to each run's same MAE-selected checkpoint. Gradient geometry and interaction reconstruction errors provide descriptive mechanism evidence and are not treated as causal proof."]
    (analysis/"paper_experiment_draft.md").write_text("\n".join(paper)+"\n")
    artifacts=[]
    for path in sorted(analysis.glob("*")):
        if path.is_file():
            artifacts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    audit=dict(schema="icd-confirmatory-completion-audit-v1",complete=not errors,
        scope="Phase0-4 MOSI validation-only seed13",official_test_used_for_selection=False,
        expected_factorial_runs=8,actual_factorial_runs=len(factorial),
        checkpoint_retention_policy="user override: MAE-best, Acc2-best, last only",
        completed_run_count=len(allruns),errors=errors,plot_errors=plot_errors,
        independent_metric_tolerance=1e-8,artifacts=artifacts,
        deferred=["MOSEI","additional seeds"],
        official_test_reporting=("completed after training (reporting only)" if (analysis/"official_test_summary.json").exists() else "pending"),
        github_sync_status="pending root-agent synchronization")
    save(BASE/"completion_audit.json",audit)
    print(json.dumps(audit,ensure_ascii=False))
    if errors:
        raise RuntimeError("Completion audit failed: "+"; ".join(errors))
if __name__=="__main__":
    main()
