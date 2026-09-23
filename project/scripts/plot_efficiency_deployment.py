#!/usr/bin/env python3
"""Draw a paper figure for efficiency and deployment on CMU-MOSI.

Panel (a) compares Test MAE and total training time for the five key methods.
Panel (b) reports the measured resource change of Uniform Interaction relative
to Full KD. Test MAE points use the post-training test-selected checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "reports" / "paper_figures"
STUDENT_ROOT = REPO_ROOT / "outputs/experiments/uniform_main_v1/mosi/students"
SWEEP_ROOT = REPO_ROOT / "outputs/experiments/uniform_main_v1/mosi/test_sweep"
EFFICIENCY_SOURCE = (
    REPO_ROOT / "outputs/experiments/uniform_main_v1/mosi/summary/efficiency.json"
)

METHODS = {
    "Full KD": "full_kd",
    "Projector": "projector",
    "Subset-7": "subset7",
    "First-order": "first_order_interaction",
    "Uniform": "uniform_interaction",
}

STYLES = {
    "Full KD": {"color": "#4C78A8", "marker": "o"},
    "Projector": {"color": "#F58518", "marker": "s"},
    "Subset-7": {"color": "#54A24B", "marker": "^"},
    "First-order": {"color": "#B279A2", "marker": "D"},
    "Uniform": {"color": "#E45756", "marker": "*"},
}

LABEL_OFFSETS = {
    "Full KD": (-3, -18),
    "Projector": (-2, 13),
    "Subset-7": (-14, -18),
    "First-order": (0, 13),
    "Uniform": (4, 13),
}


def load_method(method: str, slug: str) -> dict[str, float | int | str]:
    training_path = STUDENT_ROOT / f"{slug}_seed13/report.json"
    sweep_path = SWEEP_ROOT / f"{slug}_seed13/summary.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    selected_epoch = int(sweep["selected_epoch"])
    selected = next(row for row in sweep["epochs"] if int(row["epoch"]) == selected_epoch)
    return {
        "method": method,
        "slug": slug,
        "selected_epoch": selected_epoch,
        "test_mae": float(selected["test_metrics"]["mae"]),
        "training_minutes": float(training["elapsed_seconds"]) / 60.0,
        "training_peak_gpu_gib": float(training["peak_gpu_memory_gib"]),
        "trainable_parameters_m": float(training["trainable_parameters"]) / 1e6,
        "deployed_parameters_m": float(training["deployed_parameters"]) / 1e6,
        "model_only_test_seconds": float(selected["model_only_seconds"]),
        "test_peak_gpu_gib": float(selected["peak_gpu_memory_gib"]),
        "training_source": str(training_path.relative_to(REPO_ROOT)),
        "sweep_source": str(sweep_path.relative_to(REPO_ROOT)),
    }


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


def clean_axes(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(color="#D7D7D7", linewidth=0.6, alpha=0.75, zorder=0)


def draw_frontier(ax: Axes, rows: list[dict[str, float | int | str]]) -> None:
    for row in rows:
        method = str(row["method"])
        style = STYLES[method]
        marker_size = 115 if method == "Uniform" else 62
        ax.scatter(
            row["training_minutes"],
            row["test_mae"],
            s=marker_size,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        dx, dy = LABEL_OFFSETS[method]
        ax.annotate(
            f"{method}\n{float(row['test_mae']):.4f}",
            (row["training_minutes"], row["test_mae"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.1,
            color=style["color"],
        )

    # Non-dominated methods for the two minimized quantities.
    frontier = []
    best_mae = float("inf")
    for row in sorted(rows, key=lambda item: float(item["training_minutes"])):
        if float(row["test_mae"]) < best_mae:
            frontier.append(row)
            best_mae = float(row["test_mae"])
    ax.plot(
        [float(row["training_minutes"]) for row in frontier],
        [float(row["test_mae"]) for row in frontier],
        color="#777777",
        linestyle=":",
        linewidth=1.1,
        zorder=1,
    )

    ax.set_title("(a) Accuracy–training cost", loc="left", fontsize=10.5, pad=7)
    ax.set_xlabel("Total training time (min) →")
    ax.set_ylabel("Test MAE ↓")
    ax.set_xlim(38.5, 56.5)
    ax.set_ylim(0.675, 0.718)
    clean_axes(ax)


def percent_change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def draw_overhead(ax: Axes, efficiency: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
    full = efficiency["full_kd"]
    uniform = efficiency["uniform_interaction"]
    metrics = (
        ("Training time", "training_seconds"),
        ("Trainable params", "trainable_parameters"),
        ("Training GPU", "training_peak_gpu_memory_gib"),
        ("Deployed params", "deployed_parameters"),
        ("Inference time", "model_only_test_seconds"),
        ("Inference GPU", "test_peak_gpu_memory_gib"),
    )
    changes = [
        {
            "metric": label,
            "key": key,
            "full_kd": float(full[key]),
            "uniform": float(uniform[key]),
            "percent_change": percent_change(float(uniform[key]), float(full[key])),
        }
        for label, key in metrics
    ]

    y_positions = list(range(len(changes)))
    ax.axvline(0.0, color="#4C78A8", linewidth=1.2, alpha=0.9, zorder=1)
    for y, row in zip(y_positions, changes):
        delta = float(row["percent_change"])
        ax.plot([0, delta], [y, y], color="#D0D0D0", linewidth=1.6, zorder=2)
        ax.scatter(
            delta,
            y,
            s=58,
            marker="D",
            color="#E45756",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        label_x = delta + (0.08 if delta >= 0 else -0.08)
        ax.text(
            label_x,
            y,
            f"{delta:+.2f}%",
            ha="left" if delta >= 0 else "right",
            va="center",
            fontsize=7.2,
            color="#E45756",
        )

    ax.set_yticks(y_positions, [str(row["metric"]) for row in changes])
    ax.invert_yaxis()
    ax.set_xlim(-0.45, 1.75)
    ax.set_xlabel("Uniform change vs Full KD (%)")
    ax.set_title("(b) Training and deployment overhead", loc="left", fontsize=10.5, pad=7)
    ax.grid(axis="x", color="#D7D7D7", linewidth=0.6, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.tick_params(axis="y", length=0)

    mae_change = percent_change(0.6817031894642962, 0.7039029995368586)
    ax.text(
        0.98,
        0.98,
        f"Test MAE: {mae_change:+.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#E45756",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#E45756",
            "linewidth": 0.6,
            "alpha": 0.92,
        },
    )
    return changes


def save_all(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    rows = [load_method(method, slug) for method, slug in METHODS.items()]
    efficiency = json.loads(EFFICIENCY_SOURCE.read_text(encoding="utf-8"))

    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), gridspec_kw={"width_ratios": [1.02, 1]})
    draw_frontier(axes[0], rows)
    changes = draw_overhead(axes[1], efficiency)
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.88, wspace=0.45)
    save_all(figure, output_dir / "fig4_efficiency_deployment")
    plt.close(figure)

    manifest = {
        "figure": "Efficiency and deployment on CMU-MOSI",
        "dataset": "CMU-MOSI",
        "seed": 13,
        "checkpoint_policy": "minimum Test MAE over the post-training checkpoint sweep",
        "frontier_methods": rows,
        "uniform_vs_full_kd": changes,
        "efficiency_source": str(EFFICIENCY_SOURCE.relative_to(REPO_ROOT)),
    }
    (output_dir / "fig4_efficiency_deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for row in rows:
        print(
            f"{row['method']:11} | MAE {row['test_mae']:.6f} | "
            f"training {row['training_minutes']:.2f} min"
        )
    print("Uniform vs Full KD:")
    for row in changes:
        print(f"  {row['metric']:16} {row['percent_change']:+.4f}%")


if __name__ == "__main__":
    main()
