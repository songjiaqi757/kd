#!/usr/bin/env python3
"""Independent validation-only analysis and audit of the A-series sweep."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path("/data/sjq/kd")
BASE = ROOT / "outputs/experiments/icd_lambda_ce_sweep_mosi_v1"
REF = ROOT / "outputs/experiments/icd_confirmatory_mosi_v1"
TOL = 1e-8
TIE = 1e-5
REFERENCE_RUNS = [
    ("F1_full_kd_no_ce", 0.0, 0.0, "Full KD, No-CE"),
    ("F7_icd_no_ce_lambda025", 0.0, 0.25, "ICD, No-CE"),
    ("F5_icd_no_ce_lambda05", 0.0, 0.50, "ICD, No-CE"),
    ("F3_icd_no_ce_lambda1", 0.0, 1.00, "ICD, No-CE"),
    ("F0_full_kd_ce", 0.50, 0.0, "Full KD, CE"),
    ("F6_icd_ce_lambda025", 0.50, 0.25, "ICD, CE"),
    ("F4_icd_ce_lambda05", 0.50, 0.50, "ICD, CE"),
    ("F2_icd_ce_lambda1", 0.50, 1.00, "ICD, CE"),
]


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f1(truth, pred) -> float:
    value = 0.0
    for label in [False, True]:
        tp = np.sum((truth == label) & (pred == label))
        fp = np.sum((truth != label) & (pred == label))
        fn = np.sum((truth == label) & (pred != label))
        support = np.sum(truth == label)
        value += (support / len(truth)) * (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return float(value)


def metrics(records) -> dict:
    y = np.asarray([record["target_sentiment"] for record in records], dtype=float)
    p = np.asarray([record["prediction"] for record in records], dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid predictions")
    keep = y != 0
    pred = p[keep] >= 0
    truth = y[keep] > 0
    den = np.linalg.norm(y - y.mean()) * np.linalg.norm(p - p.mean())
    round7 = lambda a: np.clip(np.sign(a) * np.floor(np.abs(a) + 0.5), -3, 3)
    return dict(count=len(y), mae=float(np.abs(y - p).mean()),
                pearson=float(np.dot(y - y.mean(), p - p.mean()) / den) if den else None,
                acc2_nonzero=float(np.mean(pred == truth)),
                f1_weighted_nonzero=f1(truth, pred), nonzero_count=int(keep.sum()),
                acc7=float(np.mean(round7(y) == round7(p))))


def load_run(run_id: str, run_dir: Path, expected_ce: float, expected_lam: float,
             errors: list[str], initialization: dict | None) -> dict | None:
    report_path = run_dir / "report.json"
    history_path = run_dir / "history.json"
    config_path = run_dir / "run_config.json"
    if not report_path.exists() or not history_path.exists() or not config_path.exists():
        errors.append(f"{run_id}: missing report/history/run_config")
        return None
    report = read(report_path)
    history = read(history_path)
    config = read(config_path)
    positive = f"{run_id}: "
    if report.get("official_test_evaluated") is not False:
        errors.append(positive + "official-test policy violation")
    if report.get("task_ce") != expected_ce or report.get("lambda_interaction") != expected_lam:
        errors.append(positive + "loss weights differ from request")
    expected_config = {"seed": 13, "batch_size": 8, "epochs": 40, "patience": 8,
                       "learning_rate": 0.0001, "weight_decay": 0.01,
                       "lr_schedule": "constant", "checkpoint_selection": "valid_mae",
                       "task_ce": expected_ce, "lambda_interaction": expected_lam,
                       "method": "uniform_interaction", "device": "cuda:0"}
    for key, value in expected_config.items():
        if config.get(key) != value:
            errors.append(positive + f"run_config {key}={config.get(key)!r} expected {value!r}")
    if config.get("official_test_evaluated") is not False:
        errors.append(positive + "run_config official_test_evaluated is not False")
    recorded_inputs = config.get("input_sha256", {})
    for path_text, recorded in recorded_inputs.items():
        path = Path(path_text)
        if not path.exists() or sha(path) != recorded:
            errors.append(positive + f"input hash mismatch: {path_text}")
    reference_phase0 = read(REF / "preflight/phase0_complete.json")
    if config.get("campaign_fingerprint_sha256") != reference_phase0["frozen_artifact_sha256"]["source_fingerprints.json"]:
        errors.append(positive + "campaign source fingerprint differs")
    if config.get("campaign_fingerprint_sha256") != sha(REF / "source_fingerprints.json"):
        errors.append(positive + "source_fingerprints.json changed on disk")

    retained = sorted(path.name for path in run_dir.glob("*.pt"))
    if retained != ["best_valid_acc2_nonzero.pt", "best_valid_mae.pt", "last.pt"]:
        errors.append(positive + f"checkpoint retention differs: {retained}")
    for name in retained:
        if (run_dir / name).stat().st_size < 100 * 1024**2:
            errors.append(positive + f"suspiciously small checkpoint: {name}")

    if not history:
        errors.append(positive + "empty history")
        return None
    if len(history) != report.get("epochs_run"):
        errors.append(positive + "history length differs from report")
    if len(history) > 40:
        errors.append(positive + "training exceeded the 40-epoch budget")

    predictions = rows(run_dir / "predictions.jsonl")
    if len(predictions) != 229:
        errors.append(positive + f"expected 229 validation predictions, got {len(predictions)}")
    if len({record["parent_sample_id"] for record in predictions}) != len(predictions):
        errors.append(positive + "duplicate validation utterances")
    if any(record["split"] not in {"valid", "validation", "dev"} for record in predictions):
        errors.append(positive + "non-validation predictions present")
    recomputed = metrics(predictions)
    for key in ["mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7", "count"]:
        recorded = report["valid_metrics"].get(key)
        if recorded is None or abs(float(recorded) - float(recomputed[key])) > TOL:
            errors.append(positive + f"independent metric mismatch: {key}")

    best_score, best_epoch, stale = math.inf, 0, 0
    for row in history:
        score = row["valid_metrics"]["mae"]
        if score < best_score - TIE:
            best_score, best_epoch, stale = score, row["epoch"], 0
        else:
            stale += 1
    if best_epoch != report.get("best_epoch"):
        errors.append(positive + f"best epoch mismatch: {best_epoch} != {report.get('best_epoch')}")
    if abs(best_score - report["valid_metrics"]["mae"]) > TOL:
        errors.append(positive + "reported MAE is not the history minimum")
    expected_stop = "early_stopping" if stale >= 8 else "max_epochs"
    if report.get("stop_reason") != expected_stop:
        errors.append(positive + f"stop_reason {report.get('stop_reason')} expected {expected_stop}")
    retained_epochs = report.get("retained_best_checkpoints", {})
    if retained_epochs.get("valid_mae", {}).get("epoch") != best_epoch:
        errors.append(positive + "retained MAE checkpoint epoch differs")

    init = read(run_dir / "initialization.json")
    if init.get("initialization_sha256") != initialization["initialization_sha256"]:
        errors.append(positive + "initialization hash differs from frozen reference")
    if init.get("total_parameters") != initialization["total_parameters"] or \
            init.get("trainable_parameters") != initialization["trainable_parameters"]:
        errors.append(positive + "parameter counts differ from frozen reference")
    return dict(id=run_id, ce=expected_ce, lambda_icd=expected_lam,
                best_epoch=best_epoch, epochs_run=len(history),
                stop_reason=report.get("stop_reason"),
                elapsed_seconds=report.get("elapsed_seconds"),
                peak_gpu_memory_gib=report.get("peak_gpu_memory_gib"),
                metrics=recomputed)


def load_reference(run_id: str, errors: list[str], initialization: dict) -> dict:
    run_dir = REF / "runs" / run_id
    report = read(run_dir / "report.json")
    init = read(run_dir / "initialization.json")
    if init.get("initialization_sha256") != initialization["initialization_sha256"]:
        errors.append(f"reference {run_id}: initialization differs")
    return dict(id=run_id, ce=report["task_ce"], lambda_icd=report["lambda_interaction"],
                best_epoch=report["best_epoch"], metrics=report["valid_metrics"])


def main() -> None:
    errors: list[str] = []
    analysis = BASE / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    initialization = read(BASE / "preflight/campaign_initialization_reference.json")
    jobs = read(BASE / "campaign_jobs.json")["runs"]
    runs = []
    for entry in jobs:
        run = load_run(entry["id"], Path(entry["output"]), entry["ce"], entry["lam"],
                       errors, initialization)
        if run is None:
            errors.append(entry["id"] + ": run unavailable")
            continue
        runs.append(run)
    references = [load_reference(name, errors, initialization)
                  for name, _, _, _ in REFERENCE_RUNS]
    reference_labels = {name: label for name, _, _, label in REFERENCE_RUNS}

    runs.sort(key=lambda run: run["ce"] * 100 + run["lambda_icd"])
    best = min(runs, key=lambda run: run["metrics"]["mae"])
    equivalence = [run["id"] for run in runs
                   if run["metrics"]["mae"] <= best["metrics"]["mae"] + 0.005]

    lambda_sweep = sorted([run for run in runs if run["ce"] == 0.0],
                          key=lambda run: run["lambda_icd"])
    no_ce_reference = sorted([run for run in references if run["ce"] == 0.0],
                             key=lambda run: run["lambda_icd"])
    ce_sweep = sorted([run for run in runs if run["lambda_icd"] == 0.10],
                      key=lambda run: run["ce"])

    summary = dict(schema="icd-asweep-analysis-v1", created_unix=__import__("time").time(),
                   runs=runs, references=references,
                   lambda_sweep_at_zero_ce=lambda_sweep,
                   no_ce_reference_runs=no_ce_reference,
                   ce_sweep_at_lambda_010=ce_sweep,
                   best_overall=best["id"], equivalence_band_0p005=sorted(equivalence),
                   selection_rule="minimum validation MAE; official-test never used",
                   errors=errors)
    save(analysis / "summary.json", summary)

    with (analysis / "per_run.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "task_ce", "lambda_icd", "best_epoch", "epochs_run",
                         "stop_reason", "mae", "pearson", "acc2_nonzero",
                         "f1_weighted_nonzero", "acc7", "elapsed_seconds",
                         "peak_gpu_memory_gib"])
        for run in runs:
            metrics_row = run["metrics"]
            writer.writerow([run["id"], run["ce"], run["lambda_icd"], run["best_epoch"],
                             run["epochs_run"], run["stop_reason"],
                             f"{metrics_row['mae']:.6f}", f"{metrics_row['pearson']:.6f}",
                             f"{metrics_row['acc2_nonzero']:.6f}",
                             f"{metrics_row['f1_weighted_nonzero']:.6f}",
                             f"{metrics_row['acc7']:.6f}",
                             f"{run['elapsed_seconds']:.1f}",
                             f"{run['peak_gpu_memory_gib']:.2f}"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 5))
        outputs = {entry["id"]: Path(entry["output"]) for entry in jobs}
        for run in sorted(runs, key=lambda run: run["lambda_icd"] + run["ce"]):
            history = read(outputs[run["id"]] / "history.json")
            axis.plot([row["epoch"] for row in history],
                      [row["valid_metrics"]["mae"] for row in history],
                      marker="o", markersize=3,
                      label=f"{run['id']} (c={run['ce']}, λ={run['lambda_icd']})")
        axis.set_xlabel("epoch")
        axis.set_ylabel("validation MAE")
        axis.set_title("A-series validation MAE")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(analysis / "training_curves.png", dpi=160)
        plt.close(figure)
    except Exception as error:  # plotting is informative only
        errors.append("plot error: " + repr(error))

    def table(entries, include_reference: bool = False) -> list[str]:
        lines = ["|Run|CE|ICD λ|Best epoch|MAE|Pearson|Acc-2|F1|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for run in entries:
            metrics_row = run["metrics"]
            label = run["id"] if not include_reference or not run["id"].startswith("F") \
                else f"{run['id']} (参考)"
            lines.append(f"|{label}|{run['ce']:.2f}|{run['lambda_icd']:.2f}|{run['best_epoch']}|"
                         f"{metrics_row['mae']:.4f}|{metrics_row['pearson']:.4f}|"
                         f"{metrics_row['acc2_nonzero']:.4f}|{metrics_row['f1_weighted_nonzero']:.4f}|")
        return lines

    lines = ["# MOSI A 系列：CE × ICD 强度细化扫描报告", "",
             "全部结果使用 seed13、与确认实验相同的冻结资产、冻结源码、训练协议和 "
             "Validation MAE checkpoint 选择。Official-test 仅在训练与 Validation 选择"
             "冻结之后用于报告，未参与任何选择（若尚未执行，则本报告只含 Validation）。", "",
             "## 1. 本批结果", ""]
    lines += table(runs)
    lines += ["", "## 2. CE=0 的 λ 扫描（含确认实验参考）", ""]
    merged = sorted(lambda_sweep + no_ce_reference, key=lambda run: run["lambda_icd"])
    lines += table(merged, include_reference=True)
    lines += ["", "F 系列数值来自 `icd_confirmatory_mosi_v1`，仅作为同一冻结协议下的上下文；"
              "它们不属于本批实验，不参与本批选择。", "",
              "## 3. λ=0.10 的 CE 扫描", ""]
    lines += table(ce_sweep)
    lines += ["", "## 4. 工程判断", ""]
    best_metrics = best["metrics"]
    lines.append(f"- 本批最低 Validation MAE：**{best['id']}** = {best_metrics['mae']:.4f}"
                 f"（best epoch {best['best_epoch']}，Pearson {best_metrics['pearson']:.4f}）。")
    lines.append(f"- 与最低 MAE 差不超过 0.005 的配置：{', '.join(sorted(equivalence)) or '无'}；"
                 "单 seed 下视为难以区分。")
    lines.append("- 判定阈值是工程筛选标准，不是统计显著性阈值。")
    deltas = []
    if len(lambda_sweep) > 1:
        base_run = lambda_sweep[0]
        for run in lambda_sweep[1:]:
            deltas.append(f"- 相对 {base_run['id']}（λ={base_run['lambda_icd']:.2f}），"
                          f"{run['id']}（λ={run['lambda_icd']:.2f}）的 MAE 变化 "
                          f"{run['metrics']['mae'] - base_run['metrics']['mae']:+.4f}。")
    if len(ce_sweep) > 1:
        base_run = ce_sweep[0]
        for run in ce_sweep[1:]:
            deltas.append(f"- 相对 {base_run['id']}（c={base_run['ce']:.2f}），"
                          f"{run['id']}（c={run['ce']:.2f}）的 MAE 变化 "
                          f"{run['metrics']['mae'] - base_run['metrics']['mae']:+.4f}。")
    lines += deltas
    lines += ["", "## 5. 独立复核与审计", ""]
    lines.append(f"- 审计错误数：{len(errors)}。")
    lines.append("- 逐样本预测独立重算 MAE、Pearson、Acc-2、Weighted F1，并核对 "
                 "229 条 validation utterance、最佳轮次、三个保留 checkpoint、"
                 "初始化哈希与参数数量。")
    if errors:
        lines.append("- 错误详情见 `summary.json` 与 `../completion_audit.json`。")
    official_path = analysis / "official_test_summary.json"
    official = read(official_path) if official_path.exists() else None
    if official and official.get("runs"):
        lines += ["", "## 6. Official-test（仅报告，未参与选择）", "",
                  f"测试清单：`{official['test_manifest']}`（688 windows / 686 utterances）。"
                  "测试指标在全部训练与 Validation 选择冻结之后计算。", "",
                  "|Run|checkpoint|epoch|valid MAE|test MAE|test Pearson|test Acc-2|test F1|",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in sorted(official["runs"], key=lambda item: (item["run"], item["policy"])):
            lines.append(f"|{row['run']}|{row['policy']}|{row['selected_epoch']}|"
                         f"{row['valid_mae']:.4f}|{row['test_mae']:.4f}|"
                         f"{row['test_pearson']:.4f}|{row['test_acc2_nonzero']:.4f}|"
                         f"{row['test_f1_weighted_nonzero']:.4f}|")
        lines += ["", "`valid_acc2_nonzero` checkpoint 仅作辅助留档；主报告仍以 "
                  "`valid_mae` checkpoint 为准，测试集排名不作为任何配置选择的依据。"]
    else:
        lines += ["", "## 6. Official-test", "",
                  "本批尚未执行或尚未完成 Official-test 评估；测试结果不可用于选择。"]
    lines += ["", "## 7. 范围声明", "",
              "- MOSI Validation 结果来自本批 6 个 seed13 run；Official-test 仅在全部训练"
              "结束后用于报告，未参与训练、早停或 checkpoint 选择。",
              "- 不包含 MOSEI 与额外种子；结论只能在本批与确认实验共享的冻结协议内陈述。", ""]
    (analysis / "report.md").write_text("\n".join(lines))

    artifacts = []
    for path in [analysis / "summary.json", analysis / "per_run.csv",
                 analysis / "report.md", analysis / "training_curves.png",
                 analysis / "official_test_summary.json",
                 analysis / "official_test_summary.csv",
                 analysis / "official_test_summary.md",
                 BASE / "selection.json", BASE / "campaign_jobs.json",
                 BASE / "plan.json", BASE / "plan.md"]:
        if path.exists():
            artifacts.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    audit = dict(schema="icd-asweep-completion-audit-v1",
                 complete=len(runs) == len(jobs) and not errors,
                 scope="A-series MOSI validation-only seed13",
                 official_test_used_for_selection=False,
                 expected_runs=len(jobs), actual_runs=len(runs),
                 checkpoint_retention_policy="MAE-best, Acc2-best, last only",
                 independent_metric_tolerance=TOL,
                 errors=errors, artifacts=artifacts)
    save(BASE / "completion_audit.json", audit)
    print(json.dumps({"complete": audit["complete"], "runs": len(runs),
                      "errors": errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
