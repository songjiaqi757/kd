#!/usr/bin/env python3
"""Relate per-utterance interaction fidelity gains to task-error gains on valid."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
SUBSETS = ("t", "a", "v", "ta", "tv", "av", "tav")
SUBSET_SETS = tuple(frozenset(name) for name in SUBSETS)
CROSS_MODAL_SUBSETS = ("ta", "tv", "av", "tav")
CROSS_MODAL_INDICES = tuple(SUBSETS.index(name) for name in CROSS_MODAL_SUBSETS)
METHODS = ("full_kd", "uniform_interaction")
POLARITY_GROUPS = (
    "full_wrong_uniform_correct",
    "both_correct",
    "both_wrong",
    "full_correct_uniform_wrong",
)
POLARITY_LABELS = {
    "full_wrong_uniform_correct": "Full wrong → Uniform correct",
    "both_correct": "Both correct",
    "both_wrong": "Both wrong",
    "full_correct_uniform_wrong": "Full correct → Uniform wrong",
}

DATASETS = {
    "mosei": {
        "protocol": ROOT / "outputs/experiments/tav_main_v1/assets/protocol.json",
        "predictions": ROOT / "project/reports/interaction_evidence_v1/student_subset_predictions_valid",
        "aggregate": ROOT / "project/reports/interaction_evidence_v1/mosei_interaction_evidence_valid.json",
        "expected_utterances": 1871,
    },
    "mosi": {
        "protocol": ROOT / "outputs/experiments/main_table_v1/mosi/base_assets/protocol.json",
        "predictions": ROOT / "project/reports/interaction_evidence_v1/student_subset_predictions_valid_mosi",
        "aggregate": ROOT / "project/reports/interaction_evidence_v1/mosi_interaction_evidence_valid.json",
        "expected_utterances": 229,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "project/reports/interaction_fidelity_task_gain_valid.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=ROOT / "project/reports/interaction_fidelity_task_gain_valid.md",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "project/reports/paper_figures/fig_interaction_fidelity_task_gain_valid.pdf",
    )
    parser.add_argument(
        "--caption",
        type=Path,
        default=ROOT / "project/reports/paper_figures/fig_interaction_fidelity_task_gain_valid_caption.md",
    )
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    parser.add_argument("--figure-width", type=float, default=8.2)
    parser.add_argument("--figure-height", type=float, default=7.0)
    parser.add_argument(
        "--figure-layout",
        choices=("grid", "row"),
        default="grid",
        help="Use the original 2x2 grid or place all four panels in one row.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mobius(values: np.ndarray, baseline: float) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != len(SUBSETS):
        raise ValueError(f"expected an Nx7 subset array, got {values.shape}")
    interactions = []
    for target in SUBSET_SETS:
        result = np.zeros(values.shape[0], dtype=np.float64)
        for index, source in enumerate(SUBSET_SETS):
            if source.issubset(target):
                result += ((-1) ** (len(target) - len(source))) * values[:, index]
        result += ((-1) ** len(target)) * baseline
        interactions.append(result)
    return np.stack(interactions, axis=1)


def prediction_map(path: Path) -> dict[str, dict]:
    result = {}
    for row in read_jsonl(path):
        if row.get("split") != "valid":
            raise ValueError(f"non-validation prediction in {path}: {row.get('split')}")
        sample_id = str(row["parent_sample_id"])
        if sample_id in result:
            raise ValueError(f"duplicate sample ID in {path}: {sample_id}")
        if set(row["subset_predictions"]) != set(SUBSETS):
            raise ValueError(f"incomplete subset predictions in {path}: {sample_id}")
        result[sample_id] = row
    return result


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    video_ids: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict:
    videos = sorted(set(video_ids.tolist()))
    members = [np.flatnonzero(video_ids == video) for video in videos]
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        chosen = rng.integers(0, len(videos), len(videos))
        indices = np.concatenate([members[index] for index in chosen])
        samples[iteration] = float(stats.spearmanr(x[indices], y[indices]).statistic)
    finite = samples[np.isfinite(samples)]
    if len(finite) != repetitions:
        raise ValueError("non-finite clustered bootstrap Spearman replicate")
    nonpositive = int(np.sum(finite <= 0.0))
    nonnegative = int(np.sum(finite >= 0.0))
    p_value = min(1.0, 2.0 * (min(nonpositive, nonnegative) + 1) / (repetitions + 1))
    return {
        "video_clusters": len(videos),
        "repetitions": repetitions,
        "seed": seed,
        "percentile_ci95": percentile_interval(finite),
        "two_sided_sign_p_value": float(p_value),
        "mean": float(finite.mean()),
        "sample_std": float(finite.std(ddof=1)),
    }


def clustered_mean_interval(
    values: np.ndarray,
    video_ids: np.ndarray,
    repetitions: int,
    seed: int,
) -> list[float]:
    videos = sorted(set(video_ids.tolist()))
    members = [np.flatnonzero(video_ids == video) for video in videos]
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        chosen = rng.integers(0, len(videos), len(videos))
        indices = np.concatenate([members[index] for index in chosen])
        samples[iteration] = float(values[indices].mean())
    return percentile_interval(samples)


def clustered_group_difference(
    values: np.ndarray,
    video_ids: np.ndarray,
    group_labels: np.ndarray,
    left_group: str,
    right_group: str,
    repetitions: int,
    seed: int,
) -> dict:
    videos = sorted(set(video_ids.tolist()))
    members = [np.flatnonzero(video_ids == video) for video in videos]
    left = group_labels == left_group
    right = group_labels == right_group
    if not np.any(left) or not np.any(right):
        raise ValueError(f"cannot compare empty polarity groups: {left_group}, {right_group}")
    estimate = float(values[left].mean() - values[right].mean())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(videos), len(videos))
        indices = np.concatenate([members[index] for index in chosen])
        selected_labels = group_labels[indices]
        selected_left = selected_labels == left_group
        selected_right = selected_labels == right_group
        if np.any(selected_left) and np.any(selected_right):
            samples.append(
                float(values[indices][selected_left].mean() - values[indices][selected_right].mean())
            )
    sample_array = np.asarray(samples, dtype=np.float64)
    if len(sample_array) < int(0.99 * repetitions):
        raise ValueError("too many invalid polarity group bootstrap replicates")
    nonpositive = int(np.sum(sample_array <= 0.0))
    nonnegative = int(np.sum(sample_array >= 0.0))
    return {
        "left_group": left_group,
        "right_group": right_group,
        "left_minus_right": estimate,
        "valid_repetitions": int(len(sample_array)),
        "percentile_ci95": percentile_interval(sample_array),
        "two_sided_sign_p_value": float(
            min(1.0, 2.0 * (min(nonpositive, nonnegative) + 1) / (len(sample_array) + 1))
        ),
    }


def polarity_analysis(
    targets: np.ndarray,
    full_predictions: np.ndarray,
    uniform_predictions: np.ndarray,
    delta_cross_fidelity: np.ndarray,
    delta_overall_fidelity: np.ndarray,
    video_ids: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[dict, np.ndarray]:
    nonzero = targets != 0.0
    truth = targets > 0.0
    full_correct = (full_predictions >= 0.0) == truth
    uniform_correct = (uniform_predictions >= 0.0) == truth
    labels = np.full(len(targets), "excluded_zero_target", dtype=object)
    labels[nonzero & ~full_correct & uniform_correct] = "full_wrong_uniform_correct"
    labels[nonzero & full_correct & uniform_correct] = "both_correct"
    labels[nonzero & ~full_correct & ~uniform_correct] = "both_wrong"
    labels[nonzero & full_correct & ~uniform_correct] = "full_correct_uniform_wrong"
    if int(np.sum(nonzero)) != sum(int(np.sum(labels == group)) for group in POLARITY_GROUPS):
        raise ValueError("polarity outcome groups do not partition non-zero targets")

    groups = []
    for index, group in enumerate(POLARITY_GROUPS):
        mask = labels == group
        if not np.any(mask):
            raise ValueError(f"empty polarity outcome group: {group}")
        values = delta_cross_fidelity[mask]
        groups.append({
            "outcome": group,
            "label": POLARITY_LABELS[group],
            "utterances": int(np.sum(mask)),
            "rate_among_nonzero": float(np.mean(labels[nonzero] == group)),
            "video_clusters": int(len(set(video_ids[mask].tolist()))),
            "mean_delta_cross_fidelity": float(values.mean()),
            "median_delta_cross_fidelity": float(np.median(values)),
            "mean_delta_cross_fidelity_cluster_ci95": clustered_mean_interval(
                values, video_ids[mask], repetitions, seed + index
            ),
            "mean_delta_overall_fidelity": float(delta_overall_fidelity[mask].mean()),
        })

    return {
        "protocol": "Acc-2 non-zero; true positive iff target > 0; predicted positive iff regression prediction >= 0",
        "nonzero_utterances": int(np.sum(nonzero)),
        "zero_target_utterances_excluded": int(np.sum(~nonzero)),
        "full_kd_acc2_nonzero": float(np.mean(full_correct[nonzero])),
        "uniform_acc2_nonzero": float(np.mean(uniform_correct[nonzero])),
        "uniform_minus_full_acc2_nonzero": float(
            np.mean(uniform_correct[nonzero]) - np.mean(full_correct[nonzero])
        ),
        "mcnemar_exact_two_sided_p": float(stats.binomtest(
            int(np.sum(nonzero & ~full_correct & uniform_correct)),
            int(np.sum(nonzero & (full_correct != uniform_correct))),
            p=0.5,
            alternative="two-sided",
        ).pvalue),
        "groups": groups,
        "corrected_minus_harmed": clustered_group_difference(
            delta_cross_fidelity[nonzero],
            video_ids[nonzero],
            labels[nonzero],
            "full_wrong_uniform_correct",
            "full_correct_uniform_wrong",
            repetitions,
            seed + 100,
        ),
    }, labels


def analyze_dataset(name: str, specification: dict, repetitions: int, seed: int) -> dict:
    protocol_path = Path(specification["protocol"])
    aggregate_path = Path(specification["aggregate"])
    prediction_root = Path(specification["predictions"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    teacher_path = Path(protocol["interaction_targets"])
    paths = {
        "protocol": protocol_path,
        "teacher_interactions": teacher_path,
        "aggregate_evidence": aggregate_path,
        "full_kd_predictions": prediction_root / "full_kd.jsonl",
        "uniform_predictions": prediction_root / "uniform_interaction.jsonl",
    }
    teacher = {
        str(row["parent_sample_id"]): row
        for row in read_jsonl(teacher_path)
        if row.get("split") == "valid"
    }
    predictions = {
        "full_kd": prediction_map(paths["full_kd_predictions"]),
        "uniform_interaction": prediction_map(paths["uniform_predictions"]),
    }

    identities = set(teacher)
    if len(identities) != int(specification["expected_utterances"]):
        raise ValueError(f"{name}: unexpected teacher validation coverage: {len(identities)}")
    for method in METHODS:
        if set(predictions[method]) != identities:
            missing = sorted(identities - set(predictions[method]))[:5]
            extra = sorted(set(predictions[method]) - identities)[:5]
            raise ValueError(f"{name}/{method}: identity mismatch; missing={missing}, extra={extra}")

    sample_ids = sorted(identities)
    targets = np.asarray([float(teacher[x]["sentiment"]) for x in sample_ids], dtype=np.float64)
    video_ids = np.asarray([str(teacher[x]["video_id"]) for x in sample_ids])
    teacher_interactions = np.asarray([teacher[x]["mean"] for x in sample_ids], dtype=np.float64)
    if teacher_interactions.shape != (len(sample_ids), 7):
        raise ValueError(f"{name}: invalid teacher interaction shape: {teacher_interactions.shape}")

    per_method = {}
    for method in METHODS:
        rows = predictions[method]
        row_targets = np.asarray([float(rows[x]["target_sentiment"]) for x in sample_ids])
        if not np.allclose(row_targets, targets, rtol=0.0, atol=1e-7):
            raise ValueError(f"{name}/{method}: prediction labels differ from teacher labels")
        subset_values = np.asarray(
            [[float(rows[x]["subset_predictions"][subset]) for subset in SUBSETS] for x in sample_ids],
            dtype=np.float64,
        )
        interactions = mobius(subset_values, float(protocol["student_empty_baseline"]))
        coordinate_errors = np.abs(interactions - teacher_interactions)
        overall_fidelity_error = coordinate_errors.mean(axis=1)
        cross_fidelity_error = coordinate_errors[:, CROSS_MODAL_INDICES].mean(axis=1)
        predictions_tav = np.asarray([float(rows[x]["prediction"]) for x in sample_ids])
        task_error = np.abs(predictions_tav - targets)
        aggregate_error = float(aggregate["reconstruction"][method]["all_coordinates"])
        if not math.isclose(float(overall_fidelity_error.mean()), aggregate_error, abs_tol=1e-10):
            raise ValueError(
                f"{name}/{method}: per-sample reconstruction mean {overall_fidelity_error.mean()} "
                f"does not match aggregate evidence {aggregate_error}"
            )
        expected_cross = float(np.mean([
            aggregate["reconstruction"][method]["by_coordinate"][subset]
            for subset in CROSS_MODAL_SUBSETS
        ]))
        if not math.isclose(float(cross_fidelity_error.mean()), expected_cross, abs_tol=1e-10):
            raise ValueError(
                f"{name}/{method}: cross-modal reconstruction mean {cross_fidelity_error.mean()} "
                f"does not match aggregate evidence {expected_cross}"
            )
        per_method[method] = {
            "interactions": interactions,
            "overall_fidelity_error": overall_fidelity_error,
            "cross_fidelity_error": cross_fidelity_error,
            "task_error": task_error,
            "prediction": predictions_tav,
            "mean_overall_fidelity_error": float(overall_fidelity_error.mean()),
            "mean_cross_fidelity_error": float(cross_fidelity_error.mean()),
            "task_mae": float(task_error.mean()),
        }

    delta_cross_fidelity = (
        per_method["full_kd"]["cross_fidelity_error"]
        - per_method["uniform_interaction"]["cross_fidelity_error"]
    )
    delta_overall_fidelity = (
        per_method["full_kd"]["overall_fidelity_error"]
        - per_method["uniform_interaction"]["overall_fidelity_error"]
    )
    delta_task = per_method["full_kd"]["task_error"] - per_method["uniform_interaction"]["task_error"]
    cross_spearman = stats.spearmanr(delta_cross_fidelity, delta_task)
    cross_pearson = stats.pearsonr(delta_cross_fidelity, delta_task)
    cross_bootstrap = bootstrap_spearman(
        delta_cross_fidelity, delta_task, video_ids, repetitions, seed
    )
    overall_spearman = stats.spearmanr(delta_overall_fidelity, delta_task)
    overall_bootstrap = bootstrap_spearman(
        delta_overall_fidelity, delta_task, video_ids, repetitions, seed + 1_000
    )
    polarity, polarity_labels = polarity_analysis(
        targets,
        per_method["full_kd"]["prediction"],
        per_method["uniform_interaction"]["prediction"],
        delta_cross_fidelity,
        delta_overall_fidelity,
        video_ids,
        repetitions,
        seed + 10_000,
    )
    sample_rows = []
    for index, sample_id in enumerate(sample_ids):
        sample_rows.append({
            "parent_sample_id": sample_id,
            "video_id": str(video_ids[index]),
            "target_sentiment": float(targets[index]),
            "full_kd_cross_fidelity_error": float(per_method["full_kd"]["cross_fidelity_error"][index]),
            "uniform_cross_fidelity_error": float(per_method["uniform_interaction"]["cross_fidelity_error"][index]),
            "delta_cross_fidelity_full_minus_uniform": float(delta_cross_fidelity[index]),
            "full_kd_overall_fidelity_error": float(per_method["full_kd"]["overall_fidelity_error"][index]),
            "uniform_overall_fidelity_error": float(per_method["uniform_interaction"]["overall_fidelity_error"][index]),
            "delta_overall_fidelity_full_minus_uniform": float(delta_overall_fidelity[index]),
            "full_kd_task_error": float(per_method["full_kd"]["task_error"][index]),
            "uniform_task_error": float(per_method["uniform_interaction"]["task_error"][index]),
            "delta_task_error_full_minus_uniform": float(delta_task[index]),
            "full_kd_prediction": float(per_method["full_kd"]["prediction"][index]),
            "uniform_prediction": float(per_method["uniform_interaction"]["prediction"][index]),
            "polarity_outcome": str(polarity_labels[index]),
        })

    return {
        "dataset": name,
        "split": "valid",
        "seed": 13,
        "checkpoint_policy": aggregate["checkpoint_policy"],
        "utterances": len(sample_ids),
        "video_clusters": len(set(video_ids.tolist())),
        "coordinate_order": list(SUBSETS),
        "definitions": {
            "cross_fidelity_error": "mean over TA/TV/AV/TAV of abs(student interaction - teacher interaction)",
            "delta_cross_fidelity": "Full KD cross-modal fidelity error - Uniform cross-modal fidelity error; positive favors Uniform",
            "overall_fidelity_error_auxiliary": "mean over all seven coordinates of abs(student interaction - teacher interaction)",
            "task_error": "abs(sentiment prediction - target sentiment)",
            "delta_task_error": "Full KD task error - Uniform task error; positive favors Uniform",
            "task_prediction_source": "stored TAV output from the same all-seven-subset validation forward used for interaction reconstruction",
        },
        "continuous_diagnostic": {
            "coordinates": list(CROSS_MODAL_SUBSETS),
            "means": {
                "full_kd_cross_fidelity_error": per_method["full_kd"]["mean_cross_fidelity_error"],
                "uniform_cross_fidelity_error": per_method["uniform_interaction"]["mean_cross_fidelity_error"],
                "delta_cross_fidelity": float(delta_cross_fidelity.mean()),
                "full_kd_task_mae": per_method["full_kd"]["task_mae"],
                "uniform_task_mae": per_method["uniform_interaction"]["task_mae"],
                "delta_task_error": float(delta_task.mean()),
            },
            "association": {
                "spearman_rho": float(cross_spearman.statistic),
                "spearman_asymptotic_two_sided_p": float(cross_spearman.pvalue),
                "pearson_r": float(cross_pearson.statistic),
                "pearson_asymptotic_two_sided_p": float(cross_pearson.pvalue),
                "video_cluster_bootstrap_spearman": cross_bootstrap,
            },
        },
        "polarity_outcomes": polarity,
        "overall_coordinate_auxiliary": {
            "coordinates": list(SUBSETS),
            "means": {
                "full_kd_overall_fidelity_error": per_method["full_kd"]["mean_overall_fidelity_error"],
                "uniform_overall_fidelity_error": per_method["uniform_interaction"]["mean_overall_fidelity_error"],
                "delta_overall_fidelity": float(delta_overall_fidelity.mean()),
            },
            "association": {
                "spearman_rho": float(overall_spearman.statistic),
                "spearman_asymptotic_two_sided_p": float(overall_spearman.pvalue),
                "video_cluster_bootstrap_spearman": overall_bootstrap,
            },
        },
        "task_error_means": {
            "full_kd_task_mae": per_method["full_kd"]["task_mae"],
            "uniform_task_mae": per_method["uniform_interaction"]["task_mae"],
            "delta_task_error": float(delta_task.mean()),
        },
        "checkpoints": {
            method: aggregate["checkpoints"][method]
            for method in METHODS
        },
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "samples": sample_rows,
    }


def format_p(value: float) -> str:
    return "<1e-4" if value < 1e-4 else f"{value:.4f}"


def build_markdown(report: dict, figure: Path) -> str:
    lines = [
        "# Cross-modal interaction fidelity and polarity correction (official validation)",
        "",
        "比较 seed13、按 validation MAE 选中的 Full KD 与 Uniform Interaction。正的 "
        "`ΔE_cross = E_cross,Full − E_cross,Uniform` 表示 Uniform 对 TA/TV/AV/TAV 的 "
        "cross-modal interaction reconstruction 更好；正的 `ΔM = M_Full − M_Uniform` 表示 "
        "Uniform 的逐样本情感绝对误差更小。",
        "",
        "## Continuous prediction diagnostic (auxiliary)",
        "",
        "| Dataset | N / video clusters | E_cross Full | E_cross Uniform | Mean ΔE_cross | Mean ΔM | Spearman ρ | Asymptotic p | Video-cluster bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("mosei", "mosi"):
        row = report["datasets"][name]
        continuous = row["continuous_diagnostic"]
        means = continuous["means"]
        association = continuous["association"]
        ci = association["video_cluster_bootstrap_spearman"]["percentile_ci95"]
        lines.append(
            f"| {name.upper()} | {row['utterances']:,} / {row['video_clusters']} | "
            f"{means['full_kd_cross_fidelity_error']:.6f} | "
            f"{means['uniform_cross_fidelity_error']:.6f} | "
            f"{means['delta_cross_fidelity']:+.6f} | {means['delta_task_error']:+.6f} | "
            f"{association['spearman_rho']:+.4f} | "
            f"{format_p(association['spearman_asymptotic_two_sided_p'])} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )

    lines.extend([
        "",
        "Asymptotic p 值把 utterance 当作独立观测，仅作参考；主要不确定性结果是按原视频聚类的 "
        f"{report['bootstrap_repetitions']:,} 次 percentile bootstrap CI。",
        "",
        "## Polarity correction analysis (primary)",
        "",
        "采用 Acc-2 non-zero 口径：排除真实标签为 0 的样本；真实标签 `>0` 为 positive，回归预测 "
        "`>=0` 为 predicted positive。纵轴统计量是组内 mean ΔE_cross。",
    ])
    for name in ("mosei", "mosi"):
        polarity = report["datasets"][name]["polarity_outcomes"]
        lines.extend([
            "",
            f"### {name.upper()} polarity outcomes",
            "",
            f"Full/Uniform Acc-2 = {polarity['full_kd_acc2_nonzero']:.6f} / "
            f"{polarity['uniform_acc2_nonzero']:.6f}，ΔAcc-2 = "
            f"{polarity['uniform_minus_full_acc2_nonzero']:+.6f}，exact McNemar p = "
            f"{polarity['mcnemar_exact_two_sided_p']:.4f}；"
            f"non-zero N={polarity['nonzero_utterances']:,}。",
            "",
            "| Outcome | N | Video clusters | Mean ΔE_cross | Median ΔE_cross | Cluster-bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for group in polarity["groups"]:
            ci = group["mean_delta_cross_fidelity_cluster_ci95"]
            lines.append(
                f"| {group['label']} | {group['utterances']} | {group['video_clusters']} | "
                f"{group['mean_delta_cross_fidelity']:+.6f} | "
                f"{group['median_delta_cross_fidelity']:+.6f} | "
                f"[{ci[0]:+.6f}, {ci[1]:+.6f}] |"
            )
        contrast = polarity["corrected_minus_harmed"]
        ci = contrast["percentile_ci95"]
        lines.extend([
            "",
            "Corrected minus harmed contrast: "
            f"{contrast['left_minus_right']:+.6f}, cluster-bootstrap 95% CI "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}]。",
        ])

    lines.extend([
        "",
        "## Seven-coordinate overall fidelity (auxiliary audit)",
        "",
        "| Dataset | E_all Full | E_all Uniform | Mean ΔE_all | Spearman(ΔE_all, ΔM) | Cluster-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in ("mosei", "mosi"):
        auxiliary = report["datasets"][name]["overall_coordinate_auxiliary"]
        means = auxiliary["means"]
        association = auxiliary["association"]
        ci = association["video_cluster_bootstrap_spearman"]["percentile_ci95"]
        lines.append(
            f"| {name.upper()} | {means['full_kd_overall_fidelity_error']:.6f} | "
            f"{means['uniform_overall_fidelity_error']:.6f} | "
            f"{means['delta_overall_fidelity']:+.6f} | {association['spearman_rho']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
        )

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "这是固定 seed13、固定 validation-selected checkpoint 的样本级相关分析。Validation 同时参与了 "
        "checkpoint 选择，cluster bootstrap 只覆盖原视频抽样不确定性，不覆盖训练随机性或模型选择不确定性。"
        "无论相关系数是否显著，都不能解释为 interaction fidelity 对任务改善的因果效应。",
        "逐样本任务误差使用 interaction evidence 已保存的同一次七子集 forward 中的 TAV 输出，以确保 "
        "interaction 与任务预测逐样本配对；它不是从训练 history 的汇总 MAE 反推得到。",
        "MOSI validation 只有 10 个原视频 cluster，因此其 cluster bootstrap 区间尤其不精确。",
        "",
        f"Figure: `{figure}`",
        f"Caption: `{figure.with_name(figure.stem + '_caption.md')}`",
    ])
    return "\n".join(lines) + "\n"


def build_caption(report: dict) -> str:
    mosei = report["datasets"]["mosei"]
    mosi = report["datasets"]["mosi"]
    mosei_groups = mosei["polarity_outcomes"]["groups"]
    mosi_groups = mosi["polarity_outcomes"]["groups"]
    mosei_values = [f"{group['utterances']:,}" for group in mosei_groups]
    mosi_values = [f"{group['utterances']:,}" for group in mosi_groups]
    mosei_counts = ", ".join(mosei_values[:-1]) + f", and {mosei_values[-1]}"
    mosi_counts = ", ".join(mosi_values[:-1]) + f", and {mosi_values[-1]}"
    return (
        "**Cross-modal interaction fidelity and prediction outcomes on official validation.** "
        "Left: continuous prediction diagnostics for validation-MAE-selected seed13 Full KD and "
        "Uniform Interaction. We define "
        "$E_{\\mathrm{cross}}=(e_{TA}+e_{TV}+e_{AV}+e_{TAV})/4$, "
        "$\\Delta E_{\\mathrm{cross}}=E_{\\mathrm{cross}}^{\\mathrm{Full}}-"
        "E_{\\mathrm{cross}}^{\\mathrm{Uniform}}$, and "
        "$\\Delta M=|y-\\hat y^{\\mathrm{Full}}|-|y-\\hat y^{\\mathrm{Uniform}}|$; "
        "positive values favor Uniform. $\\rho$ is Spearman correlation. Right: non-zero Acc-2 "
        "polarity outcomes. “Corrected” denotes Full KD wrong $\\rightarrow$ Uniform correct, while "
        "“Harmed” denotes Full KD correct $\\rightarrow$ Uniform wrong. Error bars and the annotated "
        "corrected-minus-harmed confidence intervals are 95% percentile cluster-bootstrap intervals "
        "over source videos (10,000 resamples). MOSEI contains "
        f"{mosei['utterances']:,} utterances/{mosei['video_clusters']} source videos; its four polarity "
        "groups (Corrected, Both correct, Both wrong, Harmed) contain "
        f"{mosei_counts} samples. "
        f"MOSI contains {mosi['utterances']:,} utterances/{mosi['video_clusters']} source videos; its "
        f"groups contain {mosi_counts} samples. "
        "The MOSI corrected and harmed groups are small ($n=9$ each), so their same-direction result "
        "should be treated as exploratory. Associations do not establish causality.\n"
    )


def plot(
    report: dict,
    path: Path,
    figure_size: tuple[float, float] = (8.2, 7.0),
    figure_layout: str = "grid",
) -> None:
    colors = {"mosei": "#2B6CB0", "mosi": "#C05621"}
    outcome_colors = ("#2F855A", "#3182CE", "#718096", "#C53030")
    outcome_tick_labels = (
        "Corrected",
        "Both correct",
        "Both wrong",
        "Harmed",
    )
    if figure_layout == "grid":
        rows, columns = 2, 2
    elif figure_layout == "row":
        rows, columns = 1, 4
    else:
        raise ValueError(f"unknown figure layout: {figure_layout}")
    fig, axes = plt.subplots(rows, columns, figsize=figure_size, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for row_index, name in enumerate(("mosei", "mosi")):
        dataset = report["datasets"][name]
        samples = dataset["samples"]
        x = np.asarray([row["delta_cross_fidelity_full_minus_uniform"] for row in samples])
        y = np.asarray([row["delta_task_error_full_minus_uniform"] for row in samples])
        ax = axes[2 * row_index]
        ax.scatter(x, y, s=10 if name == "mosi" else 7, alpha=0.35, linewidths=0,
                   color=colors[name], rasterized=True)
        ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        ax.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        association = dataset["continuous_diagnostic"]["association"]
        ci = association["video_cluster_bootstrap_spearman"]["percentile_ci95"]
        ax.text(
            0.03,
            0.96,
            f"ρ = {association['spearman_rho']:+.3f}\ncluster CI [{ci[0]:+.3f}, {ci[1]:+.3f}]",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.5},
        )
        panel = "a" if name == "mosei" else "c"
        ax.set_title(f"({panel}) CMU-{name.upper()}: Continuous prediction")
        ax.set_xlabel(r"Cross-modal fidelity gain $\Delta E_{\mathrm{cross}}\uparrow$")
        ax.set_ylabel(r"Prediction-error gain $\Delta M\uparrow$")
        ax.grid(alpha=0.18, linewidth=0.5)

        ax = axes[2 * row_index + 1]
        groups = dataset["polarity_outcomes"]["groups"]
        means = np.asarray([row["mean_delta_cross_fidelity"] for row in groups])
        intervals = np.asarray([row["mean_delta_cross_fidelity_cluster_ci95"] for row in groups])
        errors = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
        positions = np.arange(4)
        for index, (position, mean) in enumerate(zip(positions, means)):
            ax.errorbar(
                position,
                mean,
                yerr=errors[:, index:index + 1],
                fmt="o",
                markersize=6,
                color=outcome_colors[index],
                ecolor=outcome_colors[index],
                elinewidth=1.4,
                capsize=3,
            )
            ax.annotate(
                f"n={groups[index]['utterances']}",
                (position, intervals[index, 1]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        lower = float(min(0.0, intervals[:, 0].min()))
        upper = float(max(0.0, intervals[:, 1].max()))
        span = max(upper - lower, 1e-6)
        ax.set_ylim(lower - 0.13 * span, upper + 0.24 * span)
        ax.set_xticks(positions, outcome_tick_labels, fontsize=7.5)
        if figure_layout == "row":
            # Keep the edge-group error bars and sample-count annotations clear
            # of the left/right spines in the compact four-panel layout.
            ax.set_xlim(-0.4, 3.4)
        panel = "b" if name == "mosei" else "d"
        ax.set_title(f"({panel}) CMU-{name.upper()}: Polarity outcomes")
        ax.set_ylabel(r"Mean $\Delta E_{\mathrm{cross}}\uparrow$")
        contrast = dataset["polarity_outcomes"]["corrected_minus_harmed"]
        contrast_ci = contrast["percentile_ci95"]
        ax.text(
            0.03,
            0.04,
            "Corrected − Harmed: "
            f"{contrast['left_minus_right']:+.3f}\n95% CI [{contrast_ci[0]:+.3f}, {contrast_ci[1]:+.3f}]",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 2.5},
        )
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    datasets = {
        name: analyze_dataset(name, specification, args.repetitions, args.bootstrap_seed + offset)
        for offset, (name, specification) in enumerate(DATASETS.items())
    }
    report = {
        "schema": "cross-modal-fidelity-polarity-valid-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "official validation only",
        "comparison": "Uniform Interaction versus Full KD",
        "seed": 13,
        "checkpoint_policy": "validation_mae_minimum_among_completed_epochs",
        "bootstrap_unit": "source video",
        "bootstrap_repetitions": args.repetitions,
        "datasets": datasets,
        "limitations": [
            "sample-level association is not a causal effect",
            "validation was also used for checkpoint selection",
            "single-seed analysis does not include training-seed uncertainty",
            "cluster bootstrap covers source-video sampling uncertainty only",
        ],
    }
    atomic_json(report, args.output.resolve())
    args.markdown.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.markdown.resolve().write_text(
        build_markdown(report, args.figure.resolve()), encoding="utf-8"
    )
    args.caption.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.caption.resolve().write_text(build_caption(report), encoding="utf-8")
    plot(
        report,
        args.figure.resolve(),
        figure_size=(args.figure_width, args.figure_height),
        figure_layout=args.figure_layout,
    )
    print(json.dumps({
        "status": "complete",
        "output": str(args.output.resolve()),
        "markdown": str(args.markdown.resolve()),
        "figure": str(args.figure.resolve()),
        "caption": str(args.caption.resolve()),
        "summary": {
            name: {
                "spearman_rho": row["continuous_diagnostic"]["association"]["spearman_rho"],
                "cluster_ci95": row["continuous_diagnostic"]["association"]["video_cluster_bootstrap_spearman"]["percentile_ci95"],
                "mean_delta_cross_fidelity": row["continuous_diagnostic"]["means"]["delta_cross_fidelity"],
                "mean_delta_task_error": row["continuous_diagnostic"]["means"]["delta_task_error"],
                "corrected_minus_harmed": row["polarity_outcomes"]["corrected_minus_harmed"],
            }
            for name, row in datasets.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
