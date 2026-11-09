#!/usr/bin/env python3
"""Plot paper-ready interaction heatmaps for the selected MOSEI cases."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "project/reports/interaction_evidence_v1"
PREDICTION_ROOT = REPORT_ROOT / "student_subset_predictions_valid"
OUTPUT_STEM = REPORT_ROOT / "interaction_case_heatmaps"

COORDINATES = ("T", "A", "V", "TA", "TV", "AV", "TAV")
CASES = (
    ("125676[7]", "Strong TA interaction", "TA"),
    ("130456[11]", "Cross-modal conflict", "TV"),
)
METHODS = (
    ("full_kd", "Full KD"),
    ("subset7", "Subset-7 KD"),
    ("first_order_interaction", "First-order"),
    ("first_second_order_interaction", "First + Second-order"),
    ("random_orthogonal", "Random Orthogonal"),
    ("uniform_interaction", "Uniform"),
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def student_interactions(subset: dict[str, float], baseline: float) -> np.ndarray:
    t, a, v, ta, tv, av, tav = (
        float(subset[key]) for key in ("t", "a", "v", "ta", "tv", "av", "tav")
    )
    return np.asarray(
        [
            t - baseline,
            a - baseline,
            v - baseline,
            ta - t - a + baseline,
            tv - t - v + baseline,
            av - a - v + baseline,
            tav - ta - tv - av + t + a + v - baseline,
        ],
        dtype=np.float64,
    )


def load_data() -> tuple[dict, list[tuple[str, str]]]:
    protocol_path = (
        ROOT / "outputs/experiments/main_table_v1/mosei/assets/protocol.json"
    )
    protocol = json.loads(protocol_path.read_text())
    baseline = float(protocol["student_empty_baseline"])
    teacher_path = Path(protocol["interaction_targets"])
    teacher = {
        row["parent_sample_id"]: row
        for row in read_jsonl(teacher_path)
        if row["split"] == "valid"
    }

    available = [
        (key, label)
        for key, label in METHODS
        if (PREDICTION_ROOT / f"{key}.jsonl").is_file()
    ]
    predictions = {}
    for key, _ in available:
        predictions[key] = {
            row["parent_sample_id"]: row
            for row in read_jsonl(PREDICTION_ROOT / f"{key}.jsonl")
        }

    result = {}
    for sample_id, _, salient in CASES:
        teacher_values = np.asarray(teacher[sample_id]["mean"], dtype=np.float64)
        rows = [{"method": "Teacher", "interactions": teacher_values.tolist()}]
        for key, label in available:
            record = predictions[key][sample_id]
            values = student_interactions(record["subset_predictions"], baseline)
            rows.append(
                {
                    "method": label,
                    "prediction": float(record["prediction"]),
                    "interaction_mae": float(np.abs(values - teacher_values).mean()),
                    "interactions": values.tolist(),
                }
            )
        result[sample_id] = {
            "salient_coordinate": salient,
            "target_sentiment": float(predictions[available[0][0]][sample_id]["target_sentiment"]),
            "rows": rows,
        }
    return result, available


def plot(data: dict, available: list[tuple[str, str]]) -> None:
    row_count = 1 + len(available)
    height = max(3.7, 0.55 * row_count + 1.45)
    figure, axes = plt.subplots(1, 2, figsize=(10.4, height), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    image = None

    for panel_index, (axis, (sample_id, title, salient)) in enumerate(zip(axes, CASES)):
        rows = data[sample_id]["rows"]
        values = np.asarray([row["interactions"] for row in rows])
        labels = []
        for row in rows:
            if row["method"] == "Teacher":
                labels.append("Teacher")
            else:
                labels.append(f"{row['method']}  ({row['interaction_mae']:.3f})")

        image = axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
        axis.set_xticks(np.arange(len(COORDINATES)), COORDINATES)
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.tick_params(axis="both", length=0, labelsize=9)
        panel = chr(ord("a") + panel_index)
        axis.set_title(f"({panel}) {title}", loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel("Anchored Möbius interaction coordinate", fontsize=9)

        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                color = "white" if abs(value) >= 1.1 else "#222222"
                weight = "bold" if (
                    COORDINATES[column_index] == salient
                    and rows[row_index]["method"] in {"Teacher", "Uniform"}
                ) else "normal"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color=color,
                    fontweight=weight,
                )

        # Separate first-, second-, and third-order blocks.
        axis.axvline(2.5, color="#444444", linewidth=1.2)
        axis.axvline(5.5, color="#444444", linewidth=1.2)
        salient_index = COORDINATES.index(salient)
        axis.add_patch(
            Rectangle(
                (salient_index - 0.5, -0.5),
                1,
                row_count,
                fill=False,
                edgecolor="#F2B134",
                linewidth=2.2,
                clip_on=False,
            )
        )
        axis.get_yticklabels()[0].set_fontweight("bold")
        for tick in axis.get_yticklabels():
            if tick.get_text().startswith("Uniform"):
                tick.set_fontweight("bold")

    colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("Interaction value", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        "Interaction profiles (row labels report interaction MAE ↓ in parentheses)",
        fontsize=11,
        y=1.035,
    )

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_separate_cases(data: dict, available: list[tuple[str, str]]) -> dict[str, dict[str, str]]:
    """Export one heatmap-plus-prediction figure per selected case."""
    outputs: dict[str, dict[str, str]] = {}
    row_count = 1 + len(available)
    method_count = len(available)
    height = max(3.8, 0.58 * row_count + 1.35)
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)

    for sample_id, title, salient in CASES:
        rows = data[sample_id]["rows"]
        values = np.asarray([row["interactions"] for row in rows])
        labels = [
            "Teacher"
            if row["method"] == "Teacher"
            else f"{row['method']}  ({row['interaction_mae']:.3f})"
            for row in rows
        ]

        figure, (heat_axis, prediction_axis) = plt.subplots(
            1,
            2,
            figsize=(10.5, height),
            gridspec_kw={"width_ratios": [3.35, 1.65]},
            constrained_layout=True,
        )
        image = heat_axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
        heat_axis.set_xticks(np.arange(len(COORDINATES)), COORDINATES)
        heat_axis.set_yticks(np.arange(len(labels)), labels)
        heat_axis.tick_params(axis="both", length=0, labelsize=9)
        heat_axis.set_title(
            title, loc="left", fontsize=11, fontweight="bold"
        )
        heat_axis.set_xlabel("Anchored Möbius interaction coordinate", fontsize=9)

        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                color = "white" if abs(value) >= 1.1 else "#222222"
                weight = "bold" if (
                    COORDINATES[column_index] == salient
                    and rows[row_index]["method"] in {"Teacher", "Uniform"}
                ) else "normal"
                heat_axis.text(
                    column_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color=color,
                    fontweight=weight,
                )
        heat_axis.axvline(2.5, color="#444444", linewidth=1.2)
        heat_axis.axvline(5.5, color="#444444", linewidth=1.2)
        salient_index = COORDINATES.index(salient)
        heat_axis.add_patch(
            Rectangle(
                (salient_index - 0.5, -0.5),
                1,
                row_count,
                fill=False,
                edgecolor="#F2B134",
                linewidth=2.2,
                clip_on=False,
            )
        )
        heat_axis.get_yticklabels()[0].set_fontweight("bold")
        for tick in heat_axis.get_yticklabels():
            if tick.get_text().startswith("Uniform"):
                tick.set_fontweight("bold")
        colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.047, pad=0.025)
        colorbar.set_label("Interaction value", fontsize=8.5)
        colorbar.ax.tick_params(labelsize=8)

        target = data[sample_id]["target_sentiment"]
        method_rows = rows[1:]
        y_positions = np.arange(method_count)
        estimates = np.asarray([row["prediction"] for row in method_rows])
        span = max(float(np.ptp(np.r_[estimates, target])), 0.25)
        data_min = min(float(estimates.min()), target)
        data_max = max(float(estimates.max()), target)
        text_x = data_max + 0.07 * span
        prediction_axis.set_xlim(
            data_min - 0.18 * span,
            data_max + 0.68 * span,
        )
        prediction_axis.set_ylim(-0.75, method_count - 0.35)

        for y, row in zip(y_positions, method_rows):
            estimate = float(row["prediction"])
            error = abs(estimate - target)
            proposed = row["method"] == "Uniform"
            color = "#C43C39" if proposed else "#376996"
            prediction_axis.plot(
                [target, estimate], [y, y], color="#8A8A8A", linewidth=1.5, zorder=1
            )
            prediction_axis.scatter(
                [target], [y], marker="D", s=34, color="#E3A018", edgecolor="white",
                linewidth=0.5, zorder=3,
            )
            prediction_axis.scatter(
                [estimate], [y], marker="o", s=50 if proposed else 42, color=color,
                edgecolor="white", linewidth=0.6, zorder=4,
            )
            prediction_axis.text(
                text_x,
                y,
                f"{estimate:+.3f}  |e|={error:.3f}",
                ha="left",
                va="center",
                fontsize=8,
                fontweight="bold" if proposed else "normal",
                color=color,
            )

        prediction_axis.axvline(
            target, color="#E3A018", linewidth=1.1, linestyle="--", zorder=0
        )
        prediction_axis.set_yticks(
            y_positions, [row["method"] for row in method_rows], fontsize=8.5
        )
        prediction_axis.invert_yaxis()
        for tick in prediction_axis.get_yticklabels():
            if tick.get_text() == "Uniform":
                tick.set_fontweight("bold")
        prediction_axis.set_title(
            f"Sentiment prediction\nGround truth = {target:+.3f}",
            fontsize=10,
            fontweight="bold",
        )
        prediction_axis.set_xlabel("Sentiment score", fontsize=9)
        prediction_axis.grid(axis="x", color="#D8D8D8", linewidth=0.7, alpha=0.75)
        prediction_axis.tick_params(axis="x", labelsize=8)
        prediction_axis.tick_params(axis="y", length=0)
        prediction_axis.spines[["top", "right", "left"]].set_visible(False)

        safe_id = sample_id.replace("[", "_").replace("]", "")
        stem = REPORT_ROOT / f"interaction_case_{safe_id}"
        case_outputs = {}
        for suffix in (".pdf", ".svg", ".png"):
            output = stem.with_suffix(suffix)
            save_kwargs = {"bbox_inches": "tight"}
            if suffix == ".png":
                save_kwargs["dpi"] = 300
            figure.savefig(output, **save_kwargs)
            case_outputs[suffix[1:]] = str(output)
        plt.close(figure)
        outputs[sample_id] = case_outputs

    return outputs


def main() -> None:
    data, available = load_data()
    plot(data, available)
    separate_outputs = plot_separate_cases(data, available)
    manifest = {
        "schema": "interaction-case-heatmaps-v1",
        "dataset": "CMU-MOSEI",
        "split": "official valid",
        "coordinate_order": list(COORDINATES),
        "color_limits": [-2.0, 2.0],
        "available_methods": [key for key, _ in available],
        "cases": data,
        "outputs": {
            suffix[1:]: str(OUTPUT_STEM.with_suffix(suffix))
            for suffix in (".pdf", ".svg", ".png")
        },
        "separate_case_outputs": separate_outputs,
    }
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_manifest").with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "combined_heatmaps": manifest["outputs"],
                "separate_cases": separate_outputs,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
