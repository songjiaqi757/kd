#!/usr/bin/env python3
"""Plot the final two-panel qualitative interaction-distillation figure."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from plot_interaction_case_heatmaps import COORDINATES, REPORT_ROOT, load_data


mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "project/reports/paper_figures"
OUTPUT_STEM = OUTPUT_ROOT / "fig5_qualitative_interaction_distillation"
FRAME_ROOT = REPORT_ROOT / "case_study_frames"

PANELS = (
    {
        "label": "a",
        "sample_id": "130456[11]",
        "title": "Conflicting first-order modality effects",
        "frame_dir": "130456_11",
        "highlight": None,
        "utterance": (
            "I really like how it's done because, if you watch this movie five times "
            "you will still not understand everything about it"
        ),
    },
    {
        "label": "b",
        "sample_id": "125676[7]",
        "title": "Strong text–audio interaction",
        "frame_dir": "125676_7",
        "highlight": "TA",
        "utterance": "I mean it had a plot but it was sort of like what's the point",
    },
)

DISPLAY_METHODS = {
    "Teacher": "Teacher",
    "Full KD": "Full KD",
    "Subset-7 KD": "Subset-7",
    "First-order": "First-order",
    "Uniform": "Ours",
}


def selected_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["method"] in DISPLAY_METHODS]


def draw_frames_and_text(
    figure: plt.Figure,
    slot,
    frame_dir: str,
    utterance: str,
) -> None:
    media = slot.subgridspec(1, 2, width_ratios=[2.05, 0.78], wspace=0.025)
    frame_grid = media[0].subgridspec(1, 3, wspace=0.018)
    paths = [
        FRAME_ROOT / frame_dir / f"frame_{index:02d}.png"
        for index in (1, 3, 5)
    ]
    for column, path in enumerate(paths):
        axis = figure.add_subplot(frame_grid[0, column])
        axis.imshow(plt.imread(path))
        axis.set_axis_off()

    text_axis = figure.add_subplot(media[1])
    text_axis.set_axis_off()
    wrapped = textwrap.fill(f'“{utterance}”', width=48)
    text_axis.text(
        0.5,
        0.5,
        wrapped,
        transform=text_axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.2,
        linespacing=1.3,
        fontstyle="italic",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F5F5F5",
            "edgecolor": "#D5D5D5",
            "linewidth": 0.6,
        },
    )


def draw_heatmap(
    figure: plt.Figure,
    axis: plt.Axes,
    colorbar_axis: plt.Axes,
    rows: list[dict],
    highlight: str | None,
) -> None:
    values = np.asarray([row["interactions"] for row in rows])
    labels = []
    for row in rows:
        label = DISPLAY_METHODS[row["method"]]
        if row["method"] != "Teacher":
            label += f"\n[{row['interaction_mae']:.3f}]"
        labels.append(label)

    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    image = axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    axis.set_xticks(np.arange(len(COORDINATES)), COORDINATES)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.tick_params(axis="both", length=0, labelsize=9.3, pad=2)
    axis.set_title(
        "Interaction coordinates (reconstruction MAE ↓)",
        loc="left",
        fontsize=10.2,
        fontweight="bold",
        pad=4,
    )
    axis.set_xlabel("Anchored Möbius coordinate", fontsize=9.0, labelpad=3)

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            color = "white" if abs(value) >= 1.1 else "#222222"
            weight = "bold" if (
                highlight == COORDINATES[column_index]
                and rows[row_index]["method"] in {"Teacher", "Uniform"}
            ) else "normal"
            axis.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=10.0,
                color=color,
                fontweight=weight,
            )

    axis.axvline(2.5, color="#444444", linewidth=0.75)
    axis.axvline(5.5, color="#444444", linewidth=0.75)
    if highlight is not None:
        coordinate_index = COORDINATES.index(highlight)
        axis.add_patch(
            Rectangle(
                (coordinate_index - 0.5, -0.5),
                1,
                len(rows),
                fill=False,
                edgecolor="#E5A114",
                linewidth=1.5,
                clip_on=False,
            )
        )
    axis.get_yticklabels()[0].set_fontweight("bold")
    for tick in axis.get_yticklabels():
        if tick.get_text().startswith("Ours"):
            tick.set_fontweight("bold")

    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Interaction value", fontsize=8.0, labelpad=3)
    colorbar.ax.tick_params(labelsize=7.8, length=2, pad=2)


def draw_prediction(
    axis: plt.Axes,
    rows: list[dict],
    target: float,
) -> None:
    method_rows = rows[1:]
    y_positions = np.arange(len(method_rows))
    axis.set_xlim(-3.0, 3.0)
    axis.set_ylim(-0.65, len(method_rows) - 0.35)
    axis.axvline(target, color="#E3A018", linewidth=1.0, linestyle="--", zorder=0)

    for y, row in zip(y_positions, method_rows):
        estimate = float(row["prediction"])
        error = abs(estimate - target)
        ours = row["method"] == "Uniform"
        color = "#C43C39" if ours else "#376996"
        axis.plot([target, estimate], [y, y], color="#8A8A8A", linewidth=1.0, zorder=1)
        axis.scatter(
            [estimate],
            [y],
            marker="o",
            s=28 if ours else 23,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axis.text(
            -2.92,
            y,
            DISPLAY_METHODS[row["method"]],
            ha="left",
            va="center",
            fontsize=9.0,
            fontweight="bold" if ours else "normal",
            color=color,
        )
        axis.text(
            2.92,
            y,
            f"{estimate:+.3f}\n[{error:.3f}]",
            ha="right",
            va="center",
            fontsize=8.5,
            linespacing=1.0,
            fontweight="bold" if ours else "normal",
            color=color,
        )

    axis.set_yticks([])
    axis.invert_yaxis()
    axis.set_xticks((-3, 0, 3))
    axis.set_title(
        f"Full-modality prediction (GT = {target:+.3f})",
        fontsize=9.7,
        fontweight="bold",
        pad=3,
    )
    axis.set_xlabel("Sentiment score", fontsize=9.0, labelpad=3)
    axis.grid(axis="x", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    axis.tick_params(axis="x", labelsize=8.5, length=2, pad=2)
    axis.tick_params(axis="y", length=0, pad=3)
    axis.spines[["top", "right", "left"]].set_visible(False)


def teacher_selection_audit() -> dict:
    protocol = json.loads(
        (ROOT / "outputs/experiments/main_table_v1/mosei/assets/protocol.json").read_text()
    )
    teacher_rows = [
        json.loads(line)
        for line in Path(protocol["interaction_targets"]).read_text().splitlines()
        if line
    ]
    teacher_rows = [row for row in teacher_rows if row["split"] == "valid"]
    identities = [row["parent_sample_id"] for row in teacher_rows]
    values = np.asarray([row["mean"] for row in teacher_rows], dtype=np.float64)

    unimodal = values[:, :3]
    positive_peak = np.clip(unimodal, 0, None).max(axis=1)
    negative_peak = np.clip(-unimodal, 0, None).max(axis=1)
    mixed_sign = (positive_peak > 0) & (negative_peak > 0)
    opposition_strength = np.minimum(positive_peak, negative_peak)
    conflict_index = identities.index("130456[11]")
    conflict_order = np.argsort(-np.where(mixed_sign, opposition_strength, -np.inf))
    conflict_rank = int(np.where(conflict_order == conflict_index)[0][0]) + 1

    ta_strength = np.abs(values[:, 3])
    ta_index = identities.index("125676[7]")
    ta_order = np.argsort(-ta_strength)
    ta_rank = int(np.where(ta_order == ta_index)[0][0]) + 1
    return {
        "selection_uses_student_results": False,
        "cross_modal_conflict": {
            "definition": (
                "among mixed-sign teacher first-order coordinates T/A/V, rank by "
                "min(max positive magnitude, max negative magnitude)"
            ),
            "score": float(opposition_strength[conflict_index]),
            "rank": conflict_rank,
            "eligible_mixed_sign_samples": int(mixed_sign.sum()),
            "top_percent": float(100 * conflict_rank / mixed_sign.sum()),
        },
        "strong_text_audio": {
            "definition": "rank all validation samples by absolute teacher TA interaction",
            "score": float(ta_strength[ta_index]),
            "rank": ta_rank,
            "validation_samples": len(teacher_rows),
            "top_percent": float(100 * ta_rank / len(teacher_rows)),
        },
    }


def main() -> None:
    data, _ = load_data()
    figure = plt.figure(figsize=(10.6, 6.15), constrained_layout=True)
    full_grid = figure.add_gridspec(1, 2, wspace=0.08)

    for panel_index, panel in enumerate(PANELS):
        panel_grid = full_grid[panel_index].subgridspec(
            4, 1, height_ratios=[0.12, 0.82, 2.15, 0.92], hspace=0.10
        )
        title_axis = figure.add_subplot(panel_grid[0])
        title_axis.set_axis_off()
        title_axis.text(
            0,
            0.5,
            f"({panel['label']}) {panel['title']}",
            transform=title_axis.transAxes,
            ha="left",
            va="center",
            fontsize=10.8,
            fontweight="bold",
        )
        draw_frames_and_text(
            figure,
            panel_grid[1],
            panel["frame_dir"],
            panel["utterance"],
        )

        heatmap_grid = panel_grid[2].subgridspec(
            1, 2, width_ratios=[4.75, 0.12], wspace=0.08
        )
        heat_axis = figure.add_subplot(heatmap_grid[0])
        colorbar_axis = figure.add_subplot(heatmap_grid[1])
        prediction_axis = figure.add_subplot(panel_grid[3])
        rows = selected_rows(data[panel["sample_id"]]["rows"])
        draw_heatmap(
            figure,
            heat_axis,
            colorbar_axis,
            rows,
            panel["highlight"],
        )
        draw_prediction(
            prediction_axis,
            rows,
            data[panel["sample_id"]]["target_sentiment"],
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 300
        figure.savefig(OUTPUT_STEM.with_suffix(suffix), **kwargs)
    plt.close(figure)

    audit = teacher_selection_audit()
    manifest = {
        "schema": "qualitative-interaction-distillation-figure-v1",
        "dataset": "CMU-MOSEI",
        "split": "official valid",
        "layout": "horizontal side-by-side panels",
        "prediction_axis_range": [-3, 3],
        "displayed_frames": [1, 3, 5],
        "method_display_name": {"uniform_interaction": "Ours"},
        "selection_audit": audit,
        "outputs": {
            suffix[1:]: str(OUTPUT_STEM.with_suffix(suffix))
            for suffix in (".pdf", ".svg", ".png")
        },
    }
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_manifest").with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )

    caption = (
        "**Figure X: Qualitative analysis of interaction distillation.** "
        "Cases are selected using teacher-side interaction statistics, independently "
        "of student prediction errors. (a) Conflicting first-order modality effects: "
        "the teacher's text and audio coordinates are positive while its visual coordinate is negative, with "
        f"opposition strength ranked {audit['cross_modal_conflict']['rank']}/"
        f"{audit['cross_modal_conflict']['eligible_mixed_sign_samples']} among mixed-sign "
        "validation samples. Ours yields the lowest overall interaction reconstruction "
        "error; it does not necessarily minimize every coordinate error. (b) Strong "
        "text–audio interaction: the example is in the top 5% by absolute teacher TA "
        f"interaction (rank {audit['strong_text_audio']['rank']}/"
        f"{audit['strong_text_audio']['validation_samples']}). In this example, "
        "higher-order interaction supervision helps the student recover the strong TA "
        "coordinate and produces a prediction closer to the ground truth. The highlighted "
        "TA column denotes the dominant higher-order teacher interaction. Ours denotes "
        "Uniform Interaction Distillation. All prediction panels use the common sentiment "
        "range [-3, 3]. Values in brackets next to method names denote reconstruction "
        "MAE over all seven interaction coordinates; prediction labels show prediction "
        "[absolute error]. Frames show the "
        "beginning, middle, and end of each utterance."
    )
    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_caption").with_suffix(".md").write_text(
        caption + "\n"
    )
    print(json.dumps(manifest["outputs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
