#!/usr/bin/env python3
"""Plot interaction reconstruction and strength-stratified evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "project/reports/interaction_evidence_v1/"
    "mosei_interaction_evidence_valid.json"
)
DEFAULT_OUTPUT = ROOT / "project/reports/paper_figures"

LABELS = {
    "full_kd": "Full KD",
    "subset7": "Subset-7",
    "first_order_interaction": "First-order",
    "first_second_order_interaction": "First+Second",
    "uniform_interaction": "Uniform Interaction",
    "random_orthogonal": "Random Orthogonal",
}
COLORS = {
    "full_kd": "#6883A5",
    "subset7": "#78A66A",
    "first_order_interaction": "#C99A62",
    "first_second_order_interaction": "#9A89A8",
    "uniform_interaction": "#E45756",
    "random_orthogonal": "#74A5A1",
}
METHOD_ORDER = (
    "full_kd",
    "subset7",
    "first_order_interaction",
    "first_second_order_interaction",
    "random_orthogonal",
    "uniform_interaction",
)


def ordered_methods(results: dict) -> list[str]:
    return [method for method in METHOD_ORDER if method in results] + [
        method for method in results if method not in METHOD_ORDER
    ]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_all(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def dataset_stem(payload: dict, stem: str) -> str:
    return stem if payload["dataset"] == "mosei" else f"{stem}_{payload['dataset']}"


def overall_reconstruction_mae(payload: dict, method: str) -> float:
    """Return the equally weighted MAE over all seven Mobius coordinates."""
    result = payload["reconstruction"][method]
    if "all_coordinates" in result:
        return float(result["all_coordinates"])
    return (
        3 * result["first_order"]
        + 3 * result["second_order"]
        + result["third_order"]
    ) / 7


def bar_style(method: str) -> dict:
    return {
        "color": COLORS.get(method),
        "edgecolor": "#4A4A4A",
        "linewidth": 0.45,
        "alpha": 1.0 if method == "uniform_interaction" else 0.82,
    }


def reconstruction_figure(payload: dict, output: Path) -> None:
    methods = ordered_methods(payload["reconstruction"])
    orders = ("first_order", "second_order", "third_order")
    labels = ("First-order", "Second-order", "Third-order")
    x = np.arange(len(orders), dtype=float)
    width = 0.78 / len(methods)
    figure, axis = plt.subplots(figsize=(7.15, 3.35))
    for index, method in enumerate(methods):
        values = [payload["reconstruction"][method][order] for order in orders]
        axis.bar(
            x - 0.39 + width / 2 + index * width,
            values,
            width,
            label=LABELS.get(method, method),
            **bar_style(method),
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Teacher–student interaction MAE ↓")
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=len(methods), frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, 1.19), fontsize=6.6,
                columnspacing=0.85, handlelength=1.2, handletextpad=0.35)
    figure.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.78)
    # Keep the historical stem for downstream compatibility and expose the
    # paper-facing aggregate name used in the manuscript.
    save_all(figure, output / dataset_stem(payload, "fig_interaction_reconstruction_order"))
    save_all(figure, output / dataset_stem(payload, "fig_aggregate_interaction_reconstruction"))
    plt.close(figure)


def strength_figure(payload: dict, output: Path) -> None:
    results = payload["strength_stratified"]["methods"]
    methods = ordered_methods(results)
    groups = ("low", "medium", "high")
    x = np.arange(len(groups), dtype=float)
    width = 0.78 / len(methods)
    figure, axis = plt.subplots(figsize=(7.15, 3.35))
    for index, method in enumerate(methods):
        values = [results[method][group]["mae"] for group in groups]
        axis.bar(
            x - 0.39 + width / 2 + index * width,
            values,
            width,
            label=LABELS.get(method, method),
            **bar_style(method),
        )
    axis.set_xticks(x, ("Low", "Medium", "High"))
    axis.set_xlabel("Teacher higher-order interaction strength tercile")
    axis.set_ylabel("Sentiment MAE ↓")
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=len(methods), frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, 1.19), fontsize=6.6,
                columnspacing=0.85, handlelength=1.2, handletextpad=0.35)
    figure.subplots_adjust(left=0.1, right=0.99, bottom=0.18, top=0.78)
    save_all(figure, output / dataset_stem(payload, "fig_interaction_strength_stratified"))
    plt.close(figure)


def combined_reconstruction_figure(payloads: list[dict], output: Path) -> None:
    orders = ("first_order", "second_order", "third_order")
    category_labels = ("First-order", "Second-order", "Third-order")
    figure, axes = plt.subplots(1, len(payloads), figsize=(7.15, 3.15), squeeze=False)
    handles = None
    legend_labels = None
    for panel, (axis, payload) in enumerate(zip(axes[0], payloads)):
        methods = ordered_methods(payload["reconstruction"])
        x = np.arange(len(orders), dtype=float)
        width = 0.8 / len(methods)
        for index, method in enumerate(methods):
            values = [payload["reconstruction"][method][order] for order in orders]
            axis.bar(
                x - 0.4 + width / 2 + index * width,
                values,
                width,
                label=LABELS.get(method, method),
                **bar_style(method),
            )
        axis.set_xticks(x, category_labels)
        axis.set_title(
            f"({chr(ord('a') + panel)}) CMU-{payload['dataset'].upper()}",
            fontsize=9,
            pad=5,
        )
        axis.set_ylabel("Interaction reconstruction MAE ↓")
        axis.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if handles is None:
            handles, legend_labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        ncol=len(legend_labels),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=6.25,
        columnspacing=0.72,
        handlelength=1.15,
        handletextpad=0.3,
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.76, wspace=0.3)
    save_all(figure, output / "fig_aggregate_interaction_reconstruction_both")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--compare-input",
        type=Path,
        help="Optional second dataset report used to produce a two-panel aggregate figure.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.input.resolve().read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    reconstruction_figure(payload, output)
    strength_figure(payload, output)
    if args.compare_input is not None:
        comparison = json.loads(args.compare_input.resolve().read_text())
        combined_reconstruction_figure([payload, comparison], output)
    manifest = {
        "schema": "interaction-evidence-figures-v1",
        "source": str(args.input.resolve()),
        "dataset": payload["dataset"],
        "split": payload["split"],
        "seed": payload["seed"],
        "methods": ordered_methods(payload["reconstruction"]),
        "utterances": payload["utterances"],
        "checkpoint_policy": payload.get("checkpoint_policy"),
        "provisional": bool(payload.get("provisional", False)),
        "aggregation": {
            "first_order": ["t", "a", "v"],
            "second_order": ["ta", "tv", "av"],
            "third_order": ["tav"],
            "metric": "mean absolute teacher-student coordinate error",
            "overall_formula": "(3*first_order + 3*second_order + third_order) / 7",
            "overall_by_method": {
                method: overall_reconstruction_mae(payload, method)
                for method in ordered_methods(payload["reconstruction"])
            },
            "evaluation_coordinate_policy": (
                "All methods are evaluated post hoc in the same anchored Mobius "
                "coordinate space, irrespective of training supervision."
            ),
        },
        "figures": [
            dataset_stem(payload, "fig_aggregate_interaction_reconstruction"),
            dataset_stem(payload, "fig_interaction_reconstruction_order"),
            dataset_stem(payload, "fig_interaction_strength_stratified"),
        ],
    }
    manifest_name = (
        "interaction_evidence_manifest.json"
        if payload["dataset"] == "mosei"
        else f"interaction_evidence_manifest_{payload['dataset']}.json"
    )
    (output / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    method_count = len(payload["reconstruction"])
    caption_name = dataset_stem(
        payload, "fig_aggregate_interaction_reconstruction"
    ) + "_caption.md"
    provisional_note = (
        " This panel is provisional pending completion of the checkpoint pool."
        if payload.get("provisional", False)
        else ""
    )
    (output / caption_name).write_text(
        "**Figure X: Aggregate interaction reconstruction.** Mean absolute "
        "teacher--student interaction reconstruction error on "
        f"{payload['utterances']:,} CMU-{payload['dataset'].upper()} "
        "official-validation utterances "
        f"for {method_count} methods (seed 13). First-order averages "
        "$I_T$, $I_A$, and $I_V$; second-order averages $I_{TA}$, $I_{TV}$, "
        "and $I_{AV}$; third-order is $I_{TAV}$. Lower is better."
        " All methods are evaluated post hoc in the same Anchored M\u00f6bius "
        "coordinate space, irrespective of the coordinate space or interaction "
        "orders used during training. Overall seven-coordinate MAE is computed "
        "as $(3E_{\\rm first}+3E_{\\rm second}+E_{\\rm third})/7$."
        f"{provisional_note}\n",
        encoding="utf-8",
    )
    if args.compare_input is not None:
        comparison = json.loads(args.compare_input.resolve().read_text())
        combined_manifest = {
            "schema": "aggregate-interaction-reconstruction-both-v1",
            "sources": [str(args.input.resolve()), str(args.compare_input.resolve())],
            "datasets": [payload["dataset"], comparison["dataset"]],
            "split": "valid",
            "seed": 13,
            "checkpoint_policies": [
                payload.get("checkpoint_policy"),
                comparison.get("checkpoint_policy"),
            ],
            "methods": ordered_methods(payload["reconstruction"]),
            "independent_y_axes": True,
            "overall_reconstruction_mae": {
                item["dataset"]: {
                    method: overall_reconstruction_mae(item, method)
                    for method in ordered_methods(item["reconstruction"])
                }
                for item in (payload, comparison)
            },
            "evaluation_coordinate_policy": (
                "All methods are evaluated post hoc in the same anchored Mobius "
                "coordinate space, irrespective of training supervision."
            ),
            "provisional": any(
                item.get("provisional", False) for item in (payload, comparison)
            ),
            "figure": "fig_aggregate_interaction_reconstruction_both",
        }
        (output / "interaction_evidence_manifest_both.json").write_text(
            json.dumps(combined_manifest, ensure_ascii=False, indent=2) + "\n"
        )
        combined_provisional_note = (
            " The CMU-MOSEI panel is provisional pending completion of its "
            "checkpoint pool."
            if payload.get("provisional", False)
            else ""
        )
        overall_text = []
        for item in (payload, comparison):
            methods = ordered_methods(item["reconstruction"])
            values = "/".join(
                f"{overall_reconstruction_mae(item, method):.3f}"
                for method in methods
            )
            overall_text.append(f"CMU-{item['dataset'].upper()}: {values}")
        method_labels = "/".join(
            LABELS.get(method, method)
            for method in ordered_methods(payload["reconstruction"])
        )
        (output / "fig_aggregate_interaction_reconstruction_both_caption.md").write_text(
            "**Figure X: Aggregate interaction reconstruction.** Mean absolute "
            "teacher--student interaction reconstruction error on the official "
            "validation sets of (a) CMU-MOSEI and (b) CMU-MOSI (seed 13). "
            "Checkpoints are selected by validation MAE. First-, second-, and "
            "third-order bars average 3, 3, and 1 interaction coordinates, "
            "respectively. All methods are evaluated post hoc in the same Anchored "
            "M\u00f6bius coordinate space, irrespective of the coordinate space or "
            "interaction orders used during training. Overall seven-coordinate "
            "MAE is $(3E_{\\rm first}+3E_{\\rm second}+E_{\\rm third})/7$; in "
            f"legend order ({method_labels}), it is "
            f"{' and '.join(overall_text)}. Lower is better. The two panels use "
            "different y-axis ranges for readability."
            f"{combined_provisional_note}\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
