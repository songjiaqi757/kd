#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path("/home/wy/sjq/kd")
STUDENT = ROOT / "outputs/student"
OUTPUT_JSON = STUDENT / "comparisons/stage_a_seed2026.json"
OUTPUT_MD = ROOT / "docs/RDID-MOSEI_StageA_实验结果.md"

RUNS = {
    "A1 subset4": STUDENT / "subset_value_4_benchmark500_seed2026",
    "A2 raw interaction4": STUDENT / "high_order_interaction_benchmark500_seed2026",
    "A3 z-score interaction4": STUDENT / "a3_zscore_interaction4_seed2026",
    "A4 inverse-variance interaction4": STUDENT / "a4_inverse_variance_interaction4_seed2026",
    "A5 SNR interaction4": STUDENT / "a5_snr_interaction4_seed2026",
    "A6 selective top25": STUDENT / "a6_selective25_interaction4_seed2026",
    "A6 selective top50": STUDENT / "a6_selective50_interaction4_seed2026",
    "A6 selective top75": STUDENT / "a6_selective75_interaction4_seed2026",
    "A7 pair raw": STUDENT / "a7_pair_raw_seed2026",
    "A7 pair SNR": STUDENT / "a7_pair_snr_seed2026",
    "A8 triple raw": STUDENT / "a8_triple_raw_seed2026",
    "A9 orthogonal 100": STUDENT / "a9_random_orthogonal_cseed100_seed2026",
    "A9 orthogonal 200": STUDENT / "a9_random_orthogonal_cseed200_seed2026",
    "A9 orthogonal 300": STUDENT / "a9_random_orthogonal_cseed300_seed2026",
    "A10 nonorthogonal 100": STUDENT / "a10_random_nonorthogonal_cseed100_seed2026",
}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def high_interaction_mae(directory: Path, high_ids: set[str]) -> float:
    predictions = [row for row in jsonl(directory / "predictions.jsonl") if row["split"] == "valid"]
    high = [row for row in predictions if row["parent_sample_id"] in high_ids]
    return statistics.mean(abs(float(row["prediction"]) - float(row["target_sentiment"])) for row in high)


def summarize_three_seed(prefix: str, high_ids: set[str]) -> dict:
    per_seed = {}
    for seed in (2026, 2027, 2028):
        directory = STUDENT / f"{prefix}{seed}"
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        per_seed[str(seed)] = {
            **{key: float(report["valid_metrics"][key]) for key in ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7")},
            "high_interaction_mae": high_interaction_mae(directory, high_ids),
        }
    summary = {}
    for metric in ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero", "acc7", "high_interaction_mae"):
        values = [per_seed[str(seed)][metric] for seed in (2026, 2027, 2028)]
        summary[metric] = {"mean": statistics.mean(values), "sample_standard_deviation": statistics.stdev(values)}
    return {"per_seed": per_seed, "summary": summary}


def main() -> int:
    reliability = jsonl(ROOT / "outputs/probe/benchmark500_interaction_reliability.jsonl")
    valid_strength = {}
    for row in reliability:
        if row["split"] == "valid":
            values = [row["interaction_mean"][name] for name in ("ta", "tv", "av", "tav")]
            valid_strength[row["sample_id"]] = float(np.linalg.norm(values))
    threshold = float(np.quantile(list(valid_strength.values()), 0.75))
    high_ids = {sample_id for sample_id, strength in valid_strength.items() if strength >= threshold}
    rows = []
    for name, directory in RUNS.items():
        report_path = directory / "report.json"
        prediction_path = directory / "predictions.jsonl"
        if not report_path.is_file() or not prediction_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        predictions = [row for row in jsonl(prediction_path) if row["split"] == "valid"]
        high = [row for row in predictions if row["parent_sample_id"] in high_ids]
        rows.append({
            "method": name,
            "mae": float(report["valid_metrics"]["mae"]),
            "pearson": float(report["valid_metrics"]["pearson"]),
            "acc2": float(report["valid_metrics"]["acc2_nonzero"]),
            "f1": float(report["valid_metrics"]["f1_weighted_nonzero"]),
            "acc7": float(report["valid_metrics"]["acc7"]),
            "best_epoch": int(report["best_epoch"]),
            "elapsed_seconds": float(report["elapsed_seconds"]),
            "high_interaction_count": len(high),
            "high_interaction_mae": statistics.mean(
                abs(float(row["prediction"]) - float(row["target_sentiment"])) for row in high
            ),
        })
    subset = next(row for row in rows if row["method"] == "A1 subset4")
    candidates = [row for row in rows if not row["method"].startswith("A1")]
    best = min(candidates, key=lambda row: row["mae"])
    best_high = min(candidates, key=lambda row: row["high_interaction_mae"])
    three_seed = {
        "subset4": summarize_three_seed("subset_value_4_benchmark500_seed", high_ids),
        "zscore_interaction4": summarize_three_seed("a3_zscore_interaction4_seed", high_ids),
        "selective_top50": summarize_three_seed("a6_selective50_interaction4_seed", high_ids),
    }
    zscore_wins = sum(
        three_seed["zscore_interaction4"]["per_seed"][str(seed)]["mae"]
        < three_seed["subset4"]["per_seed"][str(seed)]["mae"] for seed in (2026, 2027, 2028)
    )
    selective_high_wins = sum(
        three_seed["selective_top50"]["per_seed"][str(seed)]["high_interaction_mae"]
        < three_seed["subset4"]["per_seed"][str(seed)]["high_interaction_mae"] for seed in (2026, 2027, 2028)
    )
    gate = {
        "condition_1_overall_mae": best["mae"] < subset["mae"],
        "condition_2_high_interaction_mae": best_high["high_interaction_mae"] < subset["high_interaction_mae"],
        "condition_3_interaction_distortion": "not_available_from_saved_tav_predictions",
        "condition_4_seed_variance": (
            three_seed["zscore_interaction4"]["summary"]["mae"]["sample_standard_deviation"]
            < three_seed["subset4"]["summary"]["mae"]["sample_standard_deviation"]
        ),
        "three_seed_condition_1_zscore": (
            three_seed["zscore_interaction4"]["summary"]["mae"]["mean"]
            < three_seed["subset4"]["summary"]["mae"]["mean"] and zscore_wins >= 2
        ),
        "three_seed_condition_2_selective": (
            three_seed["selective_top50"]["summary"]["high_interaction_mae"]["mean"]
            < three_seed["subset4"]["summary"]["high_interaction_mae"]["mean"] and selective_high_wins >= 2
        ),
    }
    gate["go"] = bool(gate["three_seed_condition_1_zscore"] or gate["three_seed_condition_2_selective"] or gate["condition_4_seed_variance"])
    payload = {
        "seed": 2026, "high_interaction_definition": "top quartile L2 norm of teacher mean ta/tv/av/tav interactions",
        "high_interaction_threshold": threshold, "rows": rows, "best_overall_candidate": best["method"],
        "best_high_interaction_candidate": best_high["method"], "three_seed": three_seed, "gate": gate,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# RDID-MOSEI Stage A 实验结果", "", "单 seed（2026）快速筛选；A1/A2 为既有固定基线。", "",
        "| 方法 | MAE | Pearson | Acc-2 | F1 | Acc-7 | 高交互 MAE | Best epoch |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['mae']:.4f} | {row['pearson']:.4f} | {row['acc2']:.4f} | "
            f"{row['f1']:.4f} | {row['acc7']:.4f} | {row['high_interaction_mae']:.4f} | {row['best_epoch']} |"
        )
    lines += ["", "## 三 seed 候选复验", "", "| 方法 | MAE | Pearson | Acc-7 | 高交互 MAE |", "|---|---:|---:|---:|---:|"]
    for key, label in (("subset4", "subset4"), ("zscore_interaction4", "z-score interaction4"), ("selective_top50", "selective top50")):
        summary = three_seed[key]["summary"]
        lines.append(
            f"| {label} | {summary['mae']['mean']:.4f} ± {summary['mae']['sample_standard_deviation']:.4f} | "
            f"{summary['pearson']['mean']:.4f} ± {summary['pearson']['sample_standard_deviation']:.4f} | "
            f"{summary['acc7']['mean']:.4f} ± {summary['acc7']['sample_standard_deviation']:.4f} | "
            f"{summary['high_interaction_mae']['mean']:.4f} ± {summary['high_interaction_mae']['sample_standard_deviation']:.4f} |"
        )
    lines += ["", "## Stage A Gate", "", f"```json\n{json.dumps(gate, ensure_ascii=False, indent=2)}\n```", ""]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD), "rows": len(rows), "gate": gate}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
