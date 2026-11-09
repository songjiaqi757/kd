#!/usr/bin/env python3
"""Compose full Interaction Case Study figures from frames and model evidence."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from plot_interaction_case_heatmaps import CASES, COORDINATES, REPORT_ROOT, load_data


FRAME_ROOT = REPORT_ROOT / "case_study_frames"
CASE_CONTENT = {
    "125676[7]": {
        "frame_dir": "125676_7",
        "stem": "interaction_case_study_strong_ta",
        "utterance": (
            "I mean it had a plot but it was sort of like what's the point"
        ),
    },
    "130456[11]": {
        "frame_dir": "130456_11",
        "stem": "interaction_case_study_cross_modal_conflict",
        "utterance": (
            "I really like how it's done because, if you watch this movie five times "
            "you will still not understand everything about it"
        ),
    },
}


def draw_heatmap(
    figure: plt.Figure,
    axis: plt.Axes,
    rows: list[dict],
    salient: str,
) -> None:
    values = np.asarray([row["interactions"] for row in rows])
    labels = [
        "Teacher"
        if row["method"] == "Teacher"
        else f"{row['method']}  ({row['interaction_mae']:.3f})"
        for row in rows
    ]
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    image = axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    axis.set_xticks(np.arange(len(COORDINATES)), COORDINATES)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.tick_params(axis="both", length=0, labelsize=8.5)
    axis.set_title(
        "Interaction decomposition (MAE ↓ in parentheses)",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    axis.set_xlabel("Anchored Möbius interaction coordinate", fontsize=8.5)

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
                fontsize=7.25,
                color=color,
                fontweight=weight,
            )

    axis.axvline(2.5, color="#444444", linewidth=1.1)
    axis.axvline(5.5, color="#444444", linewidth=1.1)
    salient_index = COORDINATES.index(salient)
    axis.add_patch(
        Rectangle(
            (salient_index - 0.5, -0.5),
            1,
            len(rows),
            fill=False,
            edgecolor="#F2B134",
            linewidth=2.1,
            clip_on=False,
        )
    )
    axis.get_yticklabels()[0].set_fontweight("bold")
    for tick in axis.get_yticklabels():
        if tick.get_text().startswith("Uniform"):
            tick.set_fontweight("bold")

    colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.025)
    colorbar.set_label("Interaction value", fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5)


def draw_prediction(axis: plt.Axes, rows: list[dict], target: float) -> None:
    method_rows = rows[1:]
    y_positions = np.arange(len(method_rows))
    estimates = np.asarray([row["prediction"] for row in method_rows])
    span = max(float(np.ptp(np.r_[estimates, target])), 0.25)
    data_min = min(float(estimates.min()), target)
    data_max = max(float(estimates.max()), target)
    text_x = data_max + 0.07 * span
    axis.set_xlim(data_min - 0.18 * span, data_max + 0.68 * span)
    axis.set_ylim(-0.75, len(method_rows) - 0.35)

    for y, row in zip(y_positions, method_rows):
        estimate = float(row["prediction"])
        error = abs(estimate - target)
        proposed = row["method"] == "Uniform"
        color = "#C43C39" if proposed else "#376996"
        axis.plot([target, estimate], [y, y], color="#8A8A8A", linewidth=1.5, zorder=1)
        axis.scatter(
            [target],
            [y],
            marker="D",
            s=32,
            color="#E3A018",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        axis.scatter(
            [estimate],
            [y],
            marker="o",
            s=48 if proposed else 40,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        axis.text(
            text_x,
            y,
            f"{estimate:+.3f}  |e|={error:.3f}",
            ha="left",
            va="center",
            fontsize=7.7,
            fontweight="bold" if proposed else "normal",
            color=color,
        )

    axis.axvline(target, color="#E3A018", linewidth=1.1, linestyle="--", zorder=0)
    axis.set_yticks(y_positions, [row["method"] for row in method_rows], fontsize=8)
    axis.invert_yaxis()
    for tick in axis.get_yticklabels():
        if tick.get_text() == "Uniform":
            tick.set_fontweight("bold")
    axis.set_title(
        f"Sentiment prediction\nGround truth = {target:+.3f}",
        fontsize=9.5,
        fontweight="bold",
    )
    axis.set_xlabel("Sentiment score", fontsize=8.5)
    axis.grid(axis="x", color="#D8D8D8", linewidth=0.7, alpha=0.75)
    axis.tick_params(axis="x", labelsize=7.5)
    axis.tick_params(axis="y", length=0)
    axis.spines[["top", "right", "left"]].set_visible(False)


def render_case(sample_id: str, title: str, salient: str, data: dict) -> dict[str, str]:
    content = CASE_CONTENT[sample_id]
    figure = plt.figure(figsize=(10.8, 7.0), constrained_layout=True)
    outer = figure.add_gridspec(3, 1, height_ratios=[1.55, 0.48, 3.5])

    frame_grid = outer[0].subgridspec(1, 5, wspace=0.018)
    frame_paths = sorted((FRAME_ROOT / content["frame_dir"]).glob("frame_*.png"))
    if len(frame_paths) != 5:
        raise ValueError(f"expected five frames for {sample_id}, got {len(frame_paths)}")
    for index, frame_path in enumerate(frame_paths):
        frame_axis = figure.add_subplot(frame_grid[0, index])
        frame_axis.imshow(plt.imread(frame_path))
        frame_axis.set_axis_off()

    text_axis = figure.add_subplot(outer[1])
    text_axis.set_axis_off()
    text_axis.text(
        0.5,
        0.5,
        f'“{content["utterance"]}”',
        transform=text_axis.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        fontstyle="italic",
        wrap=True,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F4F4F4",
            "edgecolor": "#D4D4D4",
            "linewidth": 0.8,
        },
    )

    evidence_grid = outer[2].subgridspec(1, 2, width_ratios=[3.35, 1.65], wspace=0.29)
    heat_axis = figure.add_subplot(evidence_grid[0])
    prediction_axis = figure.add_subplot(evidence_grid[1])
    rows = data[sample_id]["rows"]
    draw_heatmap(figure, heat_axis, rows, salient)
    draw_prediction(prediction_axis, rows, data[sample_id]["target_sentiment"])
    figure.suptitle(title, fontsize=14, fontweight="bold")

    stem = REPORT_ROOT / content["stem"]
    outputs = {}
    for suffix in (".pdf", ".svg", ".png"):
        output = stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 300
        figure.savefig(output, **kwargs)
        outputs[suffix[1:]] = str(output)
    plt.close(figure)
    return outputs


def main() -> None:
    data, available = load_data()
    outputs = {
        sample_id: render_case(sample_id, title, salient, data)
        for sample_id, title, salient in CASES
    }
    manifest = {
        "schema": "full-interaction-case-study-v1",
        "dataset": "CMU-MOSEI",
        "split": "official valid",
        "methods": [key for key, _ in available],
        "outputs": outputs,
    }
    manifest_path = REPORT_ROOT / "interaction_case_studies_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(outputs, ensure_ascii=False))


if __name__ == "__main__":
    main()
