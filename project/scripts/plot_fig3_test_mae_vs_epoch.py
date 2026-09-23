#!/usr/bin/env python3
"""Draw supplementary Test MAE trajectories on CMU-MOSEI and CMU-MOSI."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "reports" / "paper_figures"


@dataclass(frozen=True)
class CurveSpec:
    label: str
    source: Path
    row_key: str = "epochs"
    completed_epochs_only: bool = False


@dataclass(frozen=True)
class Curve:
    label: str
    source: Path
    epochs: tuple[int, ...]
    maes: tuple[float, ...]

    @property
    def best_index(self) -> int:
        return min(range(len(self.maes)), key=self.maes.__getitem__)

    @property
    def best_epoch(self) -> int:
        return self.epochs[self.best_index]

    @property
    def best_mae(self) -> float:
        return self.maes[self.best_index]


METHOD_ORDER = ("Adapted Student", "Full KD", "Subset-7", "Uniform")

STYLES = {
    "Adapted Student": {
        "color": "#9D9D9D", "linestyle": ":", "linewidth": 1.3, "alpha": 0.80,
    },
    "Full KD": {
        "color": "#4C78A8", "linestyle": "-", "linewidth": 1.8, "alpha": 0.96,
    },
    "Subset-7": {
        "color": "#54A24B", "linestyle": "-.", "linewidth": 1.8, "alpha": 0.96,
    },
    "Uniform": {
        "color": "#E45756", "linestyle": "-", "linewidth": 1.9, "alpha": 0.98,
    },
}

DISPLAY_LABELS = {
    "Adapted Student": "Adapted Student",
    "Full KD": "Full KD",
    "Subset-7": "Subset-7",
    "Uniform": "Uniform Interaction",
}

MOSEI_SPECS = (
    CurveSpec(
        "Adapted Student",
        REPO_ROOT
        / "outputs/experiments/uniform_followup_v1/mosei/"
        "test_epoch_sweeps/adapted_student_seed13/summary.json",
    ),
    CurveSpec(
        "Full KD",
        REPO_ROOT
        / "outputs/experiments/tav_epoch_checkpoints_v1/"
        "checkpoint_diagnostic_test_gpu1_20260919/M3_seed13/summary.json",
        row_key="checkpoints",
        completed_epochs_only=True,
    ),
    CurveSpec(
        "Subset-7",
        REPO_ROOT
        / "outputs/experiments/uniform_followup_v1/mosei/"
        "test_epoch_sweeps/subset7_seed13/summary.json",
    ),
    CurveSpec(
        "Uniform",
        REPO_ROOT
        / "outputs/experiments/tav_epoch_checkpoints_v1/"
        "checkpoint_diagnostic_test_gpu1_20260919/M4_seed13/summary.json",
        row_key="checkpoints",
        completed_epochs_only=True,
    ),
)

MOSI_SWEEP_ROOT = REPO_ROOT / "outputs/experiments/uniform_main_v1/mosi/test_sweep"
MOSI_SPECS = (
    CurveSpec("Adapted Student", MOSI_SWEEP_ROOT / "adapted_student_seed13/summary.json"),
    CurveSpec("Full KD", MOSI_SWEEP_ROOT / "full_kd_seed13/summary.json"),
    CurveSpec("Subset-7", MOSI_SWEEP_ROOT / "subset7_seed13/summary.json"),
    CurveSpec(
        "Uniform", MOSI_SWEEP_ROOT / "uniform_interaction_seed13/summary.json"
    ),
)


def read_curve(spec: CurveSpec) -> Curve:
    payload = json.loads(spec.source.read_text(encoding="utf-8"))
    rows = payload[spec.row_key]
    if spec.completed_epochs_only:
        rows = [row for row in rows if row.get("checkpoint_role") == "completed_epoch"]

    by_epoch: dict[int, float] = {}
    for row in rows:
        epoch = int(row["epoch"])
        by_epoch[epoch] = float(row["test_metrics"]["mae"])

    ordered = sorted(by_epoch.items())
    if not ordered:
        raise ValueError(f"No checkpoint rows found in {spec.source}")
    return Curve(
        label=spec.label,
        source=spec.source,
        epochs=tuple(epoch for epoch, _ in ordered),
        maes=tuple(mae for _, mae in ordered),
    )


def read_curves(specs: Iterable[CurveSpec]) -> list[Curve]:
    curves = [read_curve(spec) for spec in specs]
    labels = tuple(curve.label for curve in curves)
    if labels != METHOD_ORDER:
        raise ValueError(f"Unexpected method order: {labels}")
    return curves


def draw_panel(ax: Axes, dataset: str, curves: list[Curve], panel: str) -> None:
    for curve in curves:
        style = STYLES[curve.label]
        ax.plot(
            curve.epochs,
            curve.maes,
            label=DISPLAY_LABELS[curve.label],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=2,
        )

    ax.set_title(f"({panel}) CMU-{dataset}", loc="left", fontsize=10.5, pad=7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MAE ↓")
    ax.set_xlim(0.6, 20.4)
    ax.set_xticks((1, 5, 10, 15, 20))
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)

    if dataset == "MOSEI":
        ax.set_ylim(0.45, 0.79)
        ax.set_yticks((0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75))
    else:
        ax.set_ylim(0.64, 1.78)
        ax.set_yticks((0.7, 0.9, 1.1, 1.3, 1.5, 1.7))

    if dataset == "MOSI":
        inset = ax.inset_axes([0.54, 0.54, 0.43, 0.40])
        for curve in curves:
            style = STYLES[curve.label]
            inset.plot(
                curve.epochs,
                curve.maes,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=max(1.0, style["linewidth"] - 0.25),
                alpha=style["alpha"],
            )
        inset.set_xlim(9.7, 20.3)
        inset.set_ylim(0.66, 0.90)
        inset.set_xticks((10, 15, 20))
        inset.set_yticks((0.70, 0.80, 0.90))
        inset.tick_params(axis="both", labelsize=5.7, width=0.5, length=2)
        inset.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.75)
        inset.set_title("Epochs 10–20", fontsize=6.2, pad=2)
        for spine in inset.spines.values():
            spine.set_linewidth(0.55)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_all(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def best_rows(dataset: str, curves: list[Curve]) -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset,
            "method": curve.label,
            "best_epoch_by_test_mae": curve.best_epoch,
            "best_test_mae": curve.best_mae,
            "epochs_plotted": len(curve.epochs),
            "source": str(curve.source.relative_to(REPO_ROOT)),
        }
        for curve in curves
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    mosei = read_curves(MOSEI_SPECS)
    mosi = read_curves(MOSI_SPECS)

    combined, axes = plt.subplots(1, 2, figsize=(7.25, 3.35))
    draw_panel(axes[0], "MOSEI", mosei, "a")
    draw_panel(axes[1], "MOSI", mosi, "b")
    handles, labels = axes[0].get_legend_handles_labels()
    combined.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        fontsize=7.5,
        handlelength=2.6,
        columnspacing=1.2,
    )
    combined.subplots_adjust(left=0.082, right=0.985, bottom=0.16, top=0.82, wspace=0.26)
    save_all(combined, output_dir / "fig3_test_mae_vs_epoch")
    save_all(combined, output_dir / "supp_test_mae_vs_epoch")
    plt.close(combined)

    for dataset, curves, panel, filename in (
        ("MOSEI", mosei, "a", "fig3a_mosei_test_mae_vs_epoch"),
        ("MOSI", mosi, "b", "fig3b_mosi_test_mae_vs_epoch"),
    ):
        figure, axis = plt.subplots(figsize=(4.0, 3.15))
        draw_panel(axis, dataset, curves, panel)
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=3,
            frameon=False,
            fontsize=7.2,
            handlelength=2.4,
            columnspacing=1.0,
        )
        figure.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.78)
        save_all(figure, output_dir / filename)
        plt.close(figure)

    manifest = {
        "figure": "Fig. 3: Test MAE vs Epoch",
        "seed": 13,
        "best_marker_definition": None,
        "mosi_inset": {
            "epochs": [10, 20],
            "mae_range": [0.66, 0.90],
            "axes_bounds": [0.54, 0.54, 0.43, 0.40],
        },
        "visualization_note": "No best-test text boxes or minimum markers are drawn.",
        "test_selection_disclosure": (
            "Markers summarize test trajectories and must not be presented as "
            "validation-selected generalization estimates."
        ),
        "curves": best_rows("MOSEI", mosei) + best_rows("MOSI", mosi),
    }
    (output_dir / "fig3_test_mae_vs_epoch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "supp_test_mae_vs_epoch_caption.md").write_text(
        "**Supplementary Figure: Per-epoch test MAE trajectories.** "
        "Test MAE over epochs 1–20 on (a) CMU-MOSEI and (b) CMU-MOSI for four "
        "representative methods (seed 13). The inset enlarges MOSI epochs 10–20. "
        "These trajectories come from post-training checkpoint sweeps; no test-minimum "
        "checkpoint is highlighted in the figure.\n",
        encoding="utf-8",
    )

    for row in manifest["curves"]:
        print(
            f"{row['dataset']:5} | {row['method']:11} | "
            f"epoch {row['best_epoch_by_test_mae']:2d} | "
            f"MAE {row['best_test_mae']:.6f}"
        )


if __name__ == "__main__":
    main()
