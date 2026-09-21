#!/usr/bin/env python3
"""Build an exhaustive, non-ranked catalog of every machine-readable experiment result."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
CORE_METRICS = (
    "count", "mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero",
    "nonzero_count", "acc2_has_zero", "f1_weighted_has_zero", "acc7",
    "classification_accuracy", "binary_acc2", "binary_f1_weighted",
    "binary_f1_macro", "binary_nll", "binary_brier", "binary_ece_15bin",
)
STANDARD_COLUMNS = (
    "count", "mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero",
    "nonzero_count", "acc2_has_zero", "f1_weighted_has_zero", "acc7",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=True)


def clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.absolute().relative_to(ROOT))


def family(path: Path) -> str:
    parts = path.absolute().relative_to(ROOT).parts
    if parts[:2] == ("outputs", "experiments") and len(parts) > 2:
        return parts[2]
    if parts[:2] == ("outputs", "student"):
        return "legacy_student"
    if parts[:2] == ("outputs", "probe"):
        return "legacy_probe"
    if parts[:2] == ("project", "reports"):
        return "project_reports"
    if parts and parts[0] == "docs":
        return "docs_reports"
    return "/".join(parts[:2])


def series(path: Path) -> str:
    name = path.parent.name.lower()
    fam = family(path)
    if fam != "legacy_student":
        return fam
    if "benchmark500" in name or re.match(r"a(?:[3-9]|10)_", name) or name.startswith(("high_order", "mobius", "subset_value")):
        return "stage_a_benchmark500"
    if name.startswith("fullscale_"):
        return "stage_b_c_fullscale"
    if name.startswith("stage_d_"):
        return "stage_d"
    if name.startswith("polarity_"):
        return "polarity"
    if name.startswith(("v2_", "v2d_")):
        return "stage_c_development"
    if name.startswith("video_"):
        return "video_adaptation"
    return "legacy_other"


def dataset(path: Path, payload: dict[str, Any] | None = None) -> str:
    text = relative(path).lower()
    if payload:
        text += " " + stable_json({key: payload.get(key) for key in ("dataset", "manifest", "test_manifest", "features")})[:2000].lower()
    if "mosi" in text and "mosei" not in text:
        return "CMU-MOSI"
    if "benchmark500" in text:
        return "CMU-MOSEI benchmark500"
    return "CMU-MOSEI"


def seed_from(path: Path, *payloads: Any) -> Any:
    for payload in payloads:
        if isinstance(payload, dict) and payload.get("seed") is not None:
            return payload["seed"]
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else ""


def metrics_from(row: dict[str, Any], prefix: str = "valid") -> dict[str, Any]:
    nested = row.get(f"{prefix}_metrics")
    if isinstance(nested, dict):
        return nested
    result = {}
    for key in CORE_METRICS:
        value = row.get(f"{prefix}_{key}")
        if value is not None:
            result[key] = value
    return result


def identity(path: Path, payload: dict[str, Any] | None = None) -> tuple[str, Any]:
    payload = payload or {}
    method = payload.get("method") or payload.get("experiment") or payload.get("mode") or path.parent.name
    return str(method), seed_from(path, payload)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stable_json(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def walk_dict(value: Any, pointer: str = "") -> Iterable[tuple[str, Any]]:
    yield pointer or "/", value
    if isinstance(value, dict):
        for key, item in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from walk_dict(item, f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_dict(item, f"{pointer}/{index}")


def status_value(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return "unknown"
    for key in ("status", "state"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return "recorded"


def display(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else "NaN"
    return str(value)


def build(output: Path, document: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        set(ROOT.glob("outputs/**/*.json"))
        | set(ROOT.glob("project/reports/*.json"))
        | set(ROOT.glob("docs/*.json"))
    )
    sources = [path for path in sources if output.resolve() not in path.resolve().parents]
    parsed: dict[Path, Any] = {}
    source_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for path in sources:
        try:
            payload = read_json(path)
            parsed[path] = payload
            schema = payload.get("schema") or payload.get("schema_version") or "" if isinstance(payload, dict) else ""
            source_rows.append({
                "source_path": relative(path), "family": family(path), "series": series(path),
                "file_name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
                "json_type": type(payload).__name__, "schema": schema,
            })
        except Exception as error:
            parse_errors.append({"source_path": relative(path), "error": repr(error)})

    source_hashes = {row["source_path"]: row["sha256"] for row in source_rows}
    artifact_paths = sorted(
        set(ROOT.glob("outputs/**/*"))
        | set(ROOT.glob("project/reports/*"))
        | set(ROOT.glob("docs/*.json"))
    )
    artifact_rows: list[dict[str, Any]] = []
    for path in artifact_paths:
        if not path.is_file() or output.resolve() in path.resolve().parents:
            continue
        rel = relative(path)
        if path.name.endswith("predictions.jsonl") or path.name.endswith("predictions.json"):
            kind = "predictions"
        elif path.suffix in {".pt", ".bin", ".safetensors"}:
            kind = "checkpoint_or_weights"
        elif path.suffix == ".json":
            kind = "json_metadata"
        elif path.suffix == ".log":
            kind = "log"
        elif path.suffix in {".npy", ".npz"}:
            kind = "array_or_features"
        elif path.suffix in {".md", ".csv", ".svg"}:
            kind = "report_or_table"
        else:
            kind = "other"
        is_link = path.is_symlink()
        artifact_rows.append({
            "artifact_path": rel, "family": family(path), "kind": kind,
            "suffix": path.suffix, "bytes": path.lstat().st_size,
            "is_symlink": is_link, "link_target": str(path.readlink()) if is_link else "",
            "target_bytes": path.stat().st_size,
            "sha256": source_hashes.get(rel, ""),
        })

    history_paths = sorted(path for path in parsed if path.name == "history.json")
    run_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for path in history_paths:
        history = parsed[path]
        if not isinstance(history, list):
            parse_errors.append({"source_path": relative(path), "error": "history is not a list"})
            continue
        report_path = path.parent / "report.json"
        status_path = path.parent / "status.json"
        config_path = path.parent / "run_config.json"
        report = parsed.get(report_path, {})
        status = parsed.get(status_path)
        config = parsed.get(config_path, {})
        method, seed = identity(path, report if isinstance(report, dict) else config)
        if status is not None:
            state = status_value(status)
        elif isinstance(report, dict) and report:
            state = "complete"
        elif history:
            state = "partial_no_report"
        else:
            state = "empty"
        best_epoch = report.get("best_epoch", "") if isinstance(report, dict) else ""
        valid = report.get("valid_metrics", {}) if isinstance(report, dict) else {}
        train = report.get("train_metrics", {}) if isinstance(report, dict) else {}
        last = history[-1] if history and isinstance(history[-1], dict) else {}
        last_valid = metrics_from(last)
        run_row = {
            "family": family(path), "series": series(path), "dataset": dataset(path, config if isinstance(config, dict) else None),
            "run": relative(path.parent), "method": method, "seed": seed, "state": state,
            "epochs_recorded": len(history), "best_epoch_recorded": best_epoch,
            "report_exists": report_path in parsed, "status_exists": status_path in parsed,
            "run_config_exists": config_path in parsed, "last_epoch": last.get("epoch", ""),
            "last_valid_mae": last_valid.get("mae", ""), "last_valid_pearson": last_valid.get("pearson", ""),
            "report_valid_mae": valid.get("mae", "") if isinstance(valid, dict) else "",
            "report_valid_pearson": valid.get("pearson", "") if isinstance(valid, dict) else "",
            "report_valid_acc2_nonzero": valid.get("acc2_nonzero", "") if isinstance(valid, dict) else "",
            "report_valid_f1_weighted_nonzero": valid.get("f1_weighted_nonzero", "") if isinstance(valid, dict) else "",
            "report_valid_acc7": valid.get("acc7", "") if isinstance(valid, dict) else "",
            "report_train_mae": train.get("mae", "") if isinstance(train, dict) else "",
            "elapsed_seconds": report.get("elapsed_seconds", "") if isinstance(report, dict) else "",
            "peak_gpu_memory_gib": report.get("peak_gpu_memory_gib", "") if isinstance(report, dict) else "",
            "official_test_evaluated": report.get("official_test_evaluated", "") if isinstance(report, dict) else "",
            "history_path": relative(path), "report_path": relative(report_path) if report_path.exists() else "",
        }
        run_rows.append(run_row)
        for index, item in enumerate(history, 1):
            if not isinstance(item, dict):
                epoch_rows.append({**{key: run_row[key] for key in ("family", "series", "dataset", "run", "method", "seed")},
                                   "epoch": index, "row_json": item})
                continue
            valid_metrics = metrics_from(item)
            epoch_row = {key: run_row[key] for key in ("family", "series", "dataset", "run", "method", "seed")}
            epoch_row.update({
                "epoch": item.get("epoch", index), "train_loss": item.get("train_loss", ""),
                "valid_loss": item.get("valid_loss", ""), "elapsed_seconds": item.get("elapsed_seconds", ""),
                **{f"valid_{key}": valid_metrics.get(key, "") for key in CORE_METRICS},
                "row_json": item,
            })
            epoch_rows.append(epoch_row)

    metric_rows: list[dict[str, Any]] = []
    embedded_history_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for path, payload in parsed.items():
        if path.name == "status.json":
            status_rows.append({
                "family": family(path), "series": series(path), "dataset": dataset(path, payload if isinstance(payload, dict) else None),
                "source_path": relative(path), "status": status_value(payload), "payload_json": payload,
            })
        if path.name == "summary.json" and isinstance(payload, dict):
            method, seed = identity(path, payload)
            summary_rows.append({
                "family": family(path), "series": series(path), "dataset": dataset(path, payload),
                "source_path": relative(path), "method": method, "seed": seed,
                "complete": payload.get("complete", ""), "selected_epoch": payload.get("selected_epoch", ""),
                "epochs_evaluated": payload.get("epochs_evaluated", len(payload.get("epochs", [])) if isinstance(payload.get("epochs"), list) else ""),
                "selection_policy": payload.get("selection_policy", payload.get("checkpoint_selection", "")),
                "summary_json": payload,
            })
        for pointer, item in walk_dict(payload):
            if pointer.endswith("/history") and isinstance(item, list):
                for index, history_item in enumerate(item, 1):
                    if not isinstance(history_item, dict):
                        embedded_history_rows.append({
                            "family": family(path), "series": series(path),
                            "source_path": relative(path), "json_pointer": f"{pointer}/{index - 1}",
                            "epoch": index, "row_json": history_item,
                        })
                        continue
                    embedded_history_rows.append({
                        "family": family(path), "series": series(path), "dataset": dataset(path, payload if isinstance(payload, dict) else None),
                        "source_path": relative(path), "json_pointer": f"{pointer}/{index - 1}",
                        "seed": seed_from(path, payload), "epoch": history_item.get("epoch", index),
                        "train_loss": history_item.get("train_loss", ""), "valid_loss": history_item.get("valid_loss", ""),
                        "valid_regression_loss": history_item.get("valid_regression_loss", ""),
                        "valid_classification_loss": history_item.get("valid_classification_loss", ""),
                        "row_json": history_item,
                    })
            if not isinstance(item, dict):
                continue
            present = [key for key in CORE_METRICS if isinstance(item.get(key), (int, float))]
            if len(present) >= 2 and any(key in present for key in ("mae", "pearson", "acc2_nonzero", "binary_acc2")):
                method, seed = identity(path, payload if isinstance(payload, dict) else {})
                scope = pointer.rsplit("/", 1)[-1] if pointer != "/" else "root"
                metric_rows.append({
                    "family": family(path), "series": series(path), "dataset": dataset(path, payload if isinstance(payload, dict) else None),
                    "source_path": relative(path), "json_pointer": pointer, "scope": scope,
                    "method": method, "seed": seed,
                    "epoch": payload.get("epoch", payload.get("best_epoch", "")) if isinstance(payload, dict) else "",
                    "checkpoint": payload.get("checkpoint", payload.get("checkpoint_role", "")) if isinstance(payload, dict) else "",
                    **{key: item.get(key, "") for key in CORE_METRICS}, "metrics_json": item,
                })
            keys = set(item)
            has_delta = any("candidate_minus_baseline" in key or key in {"delta", "delta_mae", "delta_ci95", "cluster_ci95", "two_sided_centered_bootstrap_p_value", "exact_two_sided_p"} for key in keys)
            if has_delta:
                ci = item.get("delta_ci95", item.get("cluster_ci95", ""))
                if isinstance(ci, dict):
                    ci_low, ci_high = ci.get("low", ""), ci.get("high", "")
                elif isinstance(ci, list) and len(ci) >= 2:
                    ci_low, ci_high = ci[0], ci[1]
                else:
                    ci_low = ci_high = ""
                comparison_rows.append({
                    "family": family(path), "series": series(path), "source_path": relative(path),
                    "json_pointer": pointer, "baseline": item.get("baseline", ""), "candidate": item.get("candidate", ""),
                    "candidate_minus_baseline": item.get("candidate_minus_baseline", item.get("candidate_minus_baseline_mae", item.get("delta", item.get("delta_mae", "")))),
                    "ci_low": ci_low, "ci_high": ci_high,
                    "p_value": item.get("two_sided_centered_bootstrap_p_value", item.get("exact_two_sided_p", "")),
                    "probability_candidate_better": item.get("bootstrap_probability_candidate_better", ""),
                    "comparison_json": item,
                })

    probe_rows: list[dict[str, Any]] = []
    for path, payload in parsed.items():
        if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
            continue
        splits = payload["metrics"]
        if not any(isinstance(splits.get(key), dict) for key in ("train", "valid", "test")):
            continue
        for split, subsets in splits.items():
            if not isinstance(subsets, dict):
                continue
            for subset, metrics in subsets.items():
                if not isinstance(metrics, dict) or not any(key in metrics for key in CORE_METRICS):
                    continue
                probe_rows.append({
                    "family": family(path), "series": series(path), "dataset": dataset(path, payload),
                    "source_path": relative(path), "seed": seed_from(path, payload), "split": split, "subset": subset,
                    **{key: metrics.get(key, "") for key in CORE_METRICS}, "metrics_json": metrics,
                })

    artifacts = {
        "artifact_inventory.csv": artifact_rows,
        "source_inventory.csv": source_rows,
        "training_runs.csv": run_rows,
        "training_epochs.csv": epoch_rows,
        "embedded_history_rows.csv": embedded_history_rows,
        "metric_records.csv": metric_rows,
        "probe_subset_metrics.csv": probe_rows,
        "summary_records.csv": summary_rows,
        "status_records.csv": status_rows,
        "statistical_comparisons.csv": comparison_rows,
        "parse_errors.csv": parse_errors,
    }
    for name, rows in artifacts.items():
        write_csv(rows, output / name)
    if not parse_errors:
        (output / "parse_errors.csv").write_text("source_path,error\n", encoding="utf-8")

    family_stats: dict[str, Counter] = defaultdict(Counter)
    for row in source_rows:
        family_stats[row["family"]]["json_files"] += 1
    for row in run_rows:
        family_stats[row["family"]]["training_runs"] += 1
        family_stats[row["family"]]["epoch_rows"] += int(row["epochs_recorded"])
        family_stats[row["family"]][f"state:{row['state']}"] += 1
    for row in metric_rows:
        family_stats[row["family"]]["metric_blocks"] += 1
    for row in embedded_history_rows:
        family_stats[row["family"]]["embedded_history_rows"] += 1
    for row in probe_rows:
        family_stats[row["family"]]["probe_rows"] += 1
    for row in comparison_rows:
        family_stats[row["family"]]["comparison_rows"] += 1

    catalog = clean({
        "schema": "rdid-msa-all-experiment-data-catalog-v1",
        "generated_at": "2026-09-21",
        "policy": "Exhaustive inventory; no best-only filtering and no ranking.",
        "source_roots": ["outputs/**/*.json", "project/reports/*.json", "docs/*.json"],
        "counts": {name.removesuffix(".csv"): len(rows) for name, rows in artifacts.items()},
        "family_stats": {key: dict(value) for key, value in sorted(family_stats.items())},
        "artifacts": list(artifacts),
    })
    (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    states = Counter(str(row["state"]) for row in run_rows)
    families = sorted(family_stats)
    lines = [
        "# RDID-MSA 全部实验数据总账（2026-09-21）", "",
        "本总账按文件系统中的机器可读记录生成，覆盖历史实验、正式实验、开发实验、smoke/preflight、失败启动和未完成运行。这里不按指标筛选方法，也不只保留最佳 epoch。每个训练历史中的所有 epoch 均写入 `training_epochs.csv`；所有可识别的 train/valid/test 指标块均写入 `metric_records.csv`。", "",
        "## 覆盖范围", "",
        f"- 实验产物文件：{len(artifact_rows)} 个；其中 JSON 来源文件 {len(source_rows)} 个，解析失败 {len(parse_errors)} 个。",
        f"- 含 `history.json` 的训练运行：{len(run_rows)} 个；累计训练 epoch 记录：{len(epoch_rows)} 条。",
        f"- 报告内嵌历史记录：{len(embedded_history_rows)} 条；与独立 `history.json` 分开保留来源，不自动去重。",
        f"- 标准指标块：{len(metric_rows)} 条；教师 Probe 子集指标：{len(probe_rows)} 条。",
        f"- 状态记录：{len(status_rows)} 条；summary 记录：{len(summary_rows)} 条；统计比较节点：{len(comparison_rows)} 条。",
        "- `predictions.jsonl`、模型权重、媒体与特征文件不复制进总账；全部产物的路径、类别和大小收录在 `artifact_inventory.csv`，逐样本预测仍保留在原目录。JSON 元数据另记录 SHA-256。", "",
        "## 输出文件", "",
        "| 文件 | 内容 |", "|---|---|",
        "| `artifact_inventory.csv` | `outputs/`、`project/reports/` 与 `docs/*.json` 的全部产物路径、类别和大小 |",
        "| `source_inventory.csv` | 所有纳入 JSON 文件的路径、大小、SHA-256、schema |",
        "| `training_runs.csv` | 每个训练运行的身份、状态、epoch 数、报告与最后一轮指标 |",
        "| `training_epochs.csv` | 所有训练运行的每一个 epoch，不做最佳轮筛选 |",
        "| `embedded_history_rows.csv` | Probe 报告等 JSON 中内嵌的逐 epoch/history 数组 |",
        "| `metric_records.csv` | 所有来源中递归识别出的 train/valid/test 标准指标块 |",
        "| `probe_subset_metrics.csv` | 教师 Probe 的每个 seed × split × 模态子集结果 |",
        "| `summary_records.csv` | checkpoint sweep 与批次 summary 原文及选择策略 |",
        "| `status_records.csv` | 完成、运行、失败、阻塞等状态记录原文 |",
        "| `statistical_comparisons.csv` | bootstrap、置信区间、差值与显著性节点 |",
        "| `catalog.json` | 以上数据规模和各实验线统计的机器可读索引 |", "",
        "## 各实验线覆盖", "",
        "| 实验线 | JSON 文件 | 训练运行 | 独立 history epoch | 内嵌 history | 指标块 | Probe 行 | 统计比较节点 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fam in families:
        stat = family_stats[fam]
        lines.append(f"| `{fam}` | {stat['json_files']} | {stat['training_runs']} | {stat['epoch_rows']} | {stat['embedded_history_rows']} | {stat['metric_blocks']} | {stat['probe_rows']} | {stat['comparison_rows']} |")
    lines += ["", "## 训练运行状态", "", "| 状态 | 运行数 |", "|---|---:|"]
    for state, count in sorted(states.items()):
        lines.append(f"| `{state}` | {count} |")
    lines += ["", "状态来自同目录 `status.json`；没有状态文件时，有 `report.json` 记为 `complete`，只有 history 时记为 `partial_no_report`。smoke、preflight 和正式运行均保留，通过 `run` 路径区分。", ""]

    incomplete = [row for row in run_rows if row["state"] not in {"complete", "completed", "pass"}]
    lines += ["## 非完整训练运行", "", "| 实验线 | Run | 状态 | 已记录 epoch | 报告存在 |", "|---|---|---|---:|---|"]
    if incomplete:
        for row in incomplete:
            lines.append(f"| `{row['family']}` | `{row['run']}` | `{row['state']}` | {row['epochs_recorded']} | {row['report_exists']} |")
    else:
        lines.append("| — | — | 无 | 0 | — |")
    lines += ["", "## 全部训练运行及其具体验证数值", "", "下表列出每一个含训练历史的目录，不按成绩排序。`Report Valid` 是原报告所记录 checkpoint 的完整指标，最后一列同时列出 history 最后一轮 MAE；全部逐 epoch 数值在 `training_epochs.csv`。", "", "| 实验线 | Run | 方法/实验 | Seed | 状态 | Epoch 数 | Report Valid MAE | Pearson | Acc-2 | Weighted F1 | Acc-7 | 最后一轮 Valid MAE |", "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(run_rows, key=lambda item: (item["family"], item["run"])):
        lines.append(f"| `{row['family']}` | `{row['run']}` | {row['method']} | {row['seed']} | `{row['state']}` | {row['epochs_recorded']} | {display(row['report_valid_mae'])} | {display(row['report_valid_pearson'])} | {display(row['report_valid_acc2_nonzero'])} | {display(row['report_valid_f1_weighted_nonzero'])} | {display(row['report_valid_acc7'])} | {display(row['last_valid_mae'])} |")

    direct_test_rows = []
    for row in metric_rows:
        source = Path(row["source_path"])
        if not row["json_pointer"].endswith("/test_metrics"):
            continue
        direct_report = source.name == "report.json"
        mosi_epoch = row["family"] == "uniform_main_v1" and re.fullmatch(r"epoch_\d+\.json", source.name) is not None
        if direct_report or mosi_epoch:
            direct_test_rows.append(row)
    lines += ["", "## 所有直接 Test 测评具体数值", "", f"下表列出 {len(direct_test_rows)} 条直接 checkpoint/test 报告，不按 MAE 或其他指标筛选。聚合 summary 中对这些行的重复引用不再次抄入；原始聚合结构仍在 `summary_records.csv`。", "", "| 实验线 | 方法 | Seed | Epoch | MAE | Pearson | Acc-2（非零） | Weighted F1（非零） | Acc-7 | Acc-2（含零） | Weighted F1（含零） | 来源 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in sorted(direct_test_rows, key=lambda item: (item["family"], str(item["method"]), str(item["seed"]), int(item["epoch"]) if str(item["epoch"]).isdigit() else -1, item["source_path"])):
        lines.append(f"| `{row['family']}` | {row['method']} | {row['seed']} | {row['epoch']} | {display(row['mae'])} | {display(row['pearson'])} | {display(row['acc2_nonzero'])} | {display(row['f1_weighted_nonzero'])} | {display(row['acc7'])} | {display(row['acc2_has_zero'])} | {display(row['f1_weighted_has_zero'])} | `{row['source_path']}` |")

    lines += ["", "## 全部教师 Probe 子集具体数值", "", f"下表列出 {len(probe_rows)} 条 seed × split × 模态子集指标，不按结果筛选。", "", "| 实验线 | Seed | Split | 子集 | MAE | Pearson | Acc-2（非零） | Weighted F1（非零） | Acc-7 | 来源 |", "|---|---:|---|---|---:|---:|---:|---:|---:|---|"]
    for row in sorted(probe_rows, key=lambda item: (item["family"], str(item["seed"]), item["split"], item["subset"], item["source_path"])):
        lines.append(f"| `{row['family']}` | {row['seed']} | {row['split']} | `{row['subset']}` | {display(row['mae'])} | {display(row['pearson'])} | {display(row['acc2_nonzero'])} | {display(row['f1_weighted_nonzero'])} | {display(row['acc7'])} | `{row['source_path']}` |")
    lines += ["", "## 统计与显著性文件索引", "", "所有统计 JSON 都纳入 `source_inventory.csv`；递归识别出的每个差值、区间和 p 值节点写入 `statistical_comparisons.csv`。以下列出 `project/reports` 中的 JSON 源文件，不依据显著性挑选。", ""]
    for path in sorted(ROOT.glob("project/reports/*.json")):
        lines.append(f"- `{relative(path)}`")
    lines += ["", "## 口径说明", "", "- `report_valid_*` 是原训练报告记录的选定 checkpoint 指标；`last_valid_*` 是 history 的最后一轮。两者同时保留。", "- `training_epochs.csv` 保存完整轨迹，能恢复任意 epoch 的结果，不需要从最佳轮反推。", "- test 是否参与 checkpoint 选择以各来源的 `selection_policy`、`checkpoint_selection` 和 `test_use_policy` 为准；总账不替来源改写口径。", "- 同一模型的重跑、失败启动、smoke 和正式结果可能同时存在；总账保留它们的真实路径，不自动去重。", ""]
    document.write_text("\n".join(lines), encoding="utf-8")
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiment_catalog_20260921")
    parser.add_argument("--document", type=Path, default=ROOT / "docs/全部实验数据总账_20260921.md")
    args = parser.parse_args()
    catalog = build(args.output.resolve(), args.document.resolve())
    print(json.dumps(catalog["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
