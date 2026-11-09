#!/usr/bin/env python3
"""Draw the main-paper per-epoch validation MAE trajectories."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "project/reports/paper_figures"
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


@dataclass(frozen=True)
class CurveSpec:
    label: str
    source: Path
    container: str


@dataclass(frozen=True)
class Curve:
    label: str
    source: Path
    epochs: tuple[int, ...]
    maes: tuple[float, ...]

    @property
    def minimum_epoch(self) -> int:
        index = min(range(len(self.maes)), key=self.maes.__getitem__)
        return self.epochs[index]

    @property
    def minimum_mae(self) -> float:
        return min(self.maes)


MOSEI = (
    CurveSpec(
        "Adapted Student",
        ROOT / "outputs/experiments/uniform_followup_v1/mosei/students/"
        "adapted_student_seed13/checkpoint_inventory.json",
        "inventory",
    ),
    CurveSpec(
        "Full KD",
        ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/students/"
        "M3_seed13/checkpoint_inventory.json",
        "inventory",
    ),
    CurveSpec(
        "Subset-7",
        ROOT / "outputs/experiments/uniform_followup_v1/mosei/students/"
        "subset7_seed13/checkpoint_inventory.json",
        "inventory",
    ),
    CurveSpec(
        "Uniform",
        ROOT / "outputs/experiments/tav_epoch_checkpoints_v1/students/"
        "M4_seed13/checkpoint_inventory.json",
        "inventory",
    ),
)

MOSI_ROOT = ROOT / "outputs/experiments/uniform_main_v1/mosi/students"
MOSI = (
    CurveSpec("Adapted Student", MOSI_ROOT / "adapted_student_seed13/history.json", "history"),
    CurveSpec("Full KD", MOSI_ROOT / "full_kd_seed13/history.json", "history"),
    CurveSpec("Subset-7", MOSI_ROOT / "subset7_seed13/history.json", "history"),
    CurveSpec(
        "Uniform", MOSI_ROOT / "uniform_interaction_seed13/history.json", "history"
    ),
)


def read_curve(spec: CurveSpec) -> Curve:
    payload = json.loads(spec.source.read_text(encoding="utf-8"))
    rows = payload["epochs"] if spec.container == "inventory" else payload
    by_epoch = {
        int(row["epoch"]): float(row["valid_metrics"]["mae"])
        for row in rows
    }
    ordered = sorted(by_epoch.items())
    if tuple(epoch for epoch, _ in ordered) != tuple(range(1, 21)):
        raise ValueError(f"validation coverage is not epochs 1–20: {spec.source}")
    return Curve(
        label=spec.label,
        source=spec.source,
        epochs=tuple(epoch for epoch, _ in ordered),
        maes=tuple(mae for _, mae in ordered),
    )


def read_curves(specs: Iterable[CurveSpec]) -> list[Curve]:
    curves = [read_curve(spec) for spec in specs]
    if tuple(curve.label for curve in curves) != METHOD_ORDER:
        raise ValueError("unexpected method order")
    return curves


def plot_curves(axis: Axes, curves: list[Curve], linewidth_delta: float = 0.0) -> None:
    for curve in curves:
        style = STYLES[curve.label]
        axis.plot(
            curve.epochs,
            curve.maes,
            label=DISPLAY_LABELS[curve.label],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=max(1.0, style["linewidth"] + linewidth_delta),
            alpha=style["alpha"],
            zorder=1 if curve.label == "Adapted Student" else 2,
        )


def draw_panel(axis: Axes, dataset: str, curves: list[Curve], panel: str) -> None:
    plot_curves(axis, curves)
    axis.set_title(f"({panel}) CMU-{dataset}", loc="left", fontsize=10.5, pad=7)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation MAE ↓")
    axis.set_xlim(0.6, 20.4)
    axis.set_xticks((1, 5, 10, 15, 20))
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=8)

    if dataset == "MOSEI":
        axis.set_ylim(0.44, 0.73)
        axis.set_yticks((0.45, 0.50, 0.55, 0.60, 0.65, 0.70))
        return

    axis.set_ylim(0.66, 2.68)
    axis.set_yticks((0.7, 1.0, 1.5, 2.0, 2.5))
    inset = axis.inset_axes([0.51, 0.53, 0.46, 0.41])
    plot_curves(inset, curves, linewidth_delta=-0.25)
    inset.set_xlim(9.7, 20.3)
    inset.set_ylim(0.70, 0.88)
    inset.set_xticks((10, 15, 20))
    inset.set_yticks((0.72, 0.78, 0.84))
    inset.tick_params(axis="both", labelsize=5.7, width=0.5, length=2)
    inset.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    inset.set_title("Epochs 10–20", fontsize=6.2, pad=2)
    for spine in inset.spines.values():
        spine.set_linewidth(0.55)


def configure_style() -> None:
    plt.rcParams.update({
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
    })


def save_all(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    configure_style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mosei = read_curves(MOSEI)
    mosi = read_curves(MOSI)

    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.35))
    draw_panel(axes[0], "MOSEI", mosei, "a")
    draw_panel(axes[1], "MOSI", mosi, "b")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
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
    figure.subplots_adjust(left=0.082, right=0.985, bottom=0.16, top=0.82, wspace=0.27)
    save_all(figure, OUTPUT_ROOT / "fig3_validation_mae_vs_epoch")
    plt.close(figure)

    manifest = {
        "figure": "Fig. 3: Per-epoch validation MAE trajectories",
        "dataset_split": "official validation",
        "seed": 13,
        "minimum_markers_drawn": False,
        "mosi_inset": {
            "epochs": [10, 20],
            "mae_range": [0.70, 0.88],
            "axes_bounds": [0.51, 0.53, 0.46, 0.41],
        },
        "curves": [
            {
                "dataset": dataset,
                "method": curve.label,
                "epochs_plotted": len(curve.epochs),
                "minimum_validation_epoch": curve.minimum_epoch,
                "minimum_validation_mae": curve.minimum_mae,
                "source": str(curve.source.relative_to(ROOT)),
            }
            for dataset, curves in (("MOSEI", mosei), ("MOSI", mosi))
            for curve in curves
        ],
    }
    (OUTPUT_ROOT / "fig3_validation_mae_vs_epoch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    caption = (
        "**Figure 3: Per-epoch MAE trajectories.** Validation MAE over epochs 1–20 "
        "on (a) CMU-MOSEI and (b) CMU-MOSI for four representative methods "
        "using seed 13. The inset enlarges MOSI epochs 10–20."
    )
    (OUTPUT_ROOT / "fig3_validation_mae_vs_epoch_caption.md").write_text(
        caption + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "fig3_caption.md").write_text(
        "# Fig. 3 caption\n\n" + caption + "\n\n## Reproduce\n\n```bash\n"
        "/home/wy/sjq/miniconda3/envs/kd/bin/python \\\n"
        "  project/scripts/plot_fig3_validation_mae_vs_epoch.py\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
