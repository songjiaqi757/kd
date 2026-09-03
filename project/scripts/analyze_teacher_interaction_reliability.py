#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.interaction import SUBSET_NAMES, mobius_transform

INTERACTIONS = ("ta", "tv", "av", "tav")
INDICES = (3, 4, 5, 6)
COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize teacher interaction reliability across probe seeds")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    return parser.parse_args()


def load(path: Path) -> dict[str, dict[str, dict]]:
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[str(item["parent_sample_id"])][str(item["subset"])] = item
    return rows


def write_svg(path: Path, means: np.ndarray, variances: np.ndarray, snr: np.ndarray, stats: dict) -> None:
    width, height = 1200, 800
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;fill:#111827}.title{font-size:18px;font-weight:600}.label{font-size:13px}</style>',
    ]

    def histogram_panel(data: np.ndarray, title: str, x0: int, y0: int) -> None:
        panel_width, panel_height = 500, 260
        svg.append(f'<text class="title" x="{x0}" y="{y0 - 15}">{title}</text>')
        svg.append(f'<line x1="{x0}" y1="{y0 + panel_height}" x2="{x0 + panel_width}" y2="{y0 + panel_height}" stroke="#6b7280"/>')
        svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_height}" stroke="#6b7280"/>')
        low, high = float(np.min(data)), float(np.max(data))
        if high <= low:
            high = low + 1.0
        edges = np.linspace(low, high, 36)
        histograms = [np.histogram(data[:, index], bins=edges, density=True)[0] for index in range(4)]
        maximum = max(float(np.max(histogram)) for histogram in histograms) or 1.0
        for name, color, histogram in zip(INTERACTIONS, COLORS, histograms):
            points = []
            for bin_index, value in enumerate(histogram):
                x = x0 + (bin_index + 0.5) / len(histogram) * panel_width
                y = y0 + panel_height - float(value) / maximum * panel_height
                points.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>')
        for index, (name, color) in enumerate(zip(INTERACTIONS, COLORS)):
            x = x0 + index * 90
            svg.append(f'<line x1="{x}" y1="{y0 + panel_height + 28}" x2="{x + 20}" y2="{y0 + panel_height + 28}" stroke="{color}" stroke-width="3"/>')
            svg.append(f'<text class="label" x="{x + 25}" y="{y0 + panel_height + 33}">{name}</text>')
        svg.append(f'<text class="label" x="{x0}" y="{y0 + panel_height + 55}">range: {low:.3g} to {high:.3g}</text>')

    histogram_panel(means, "Teacher interaction mean", 70, 80)
    histogram_panel(np.log10(variances + 1e-4), "log10 probe variance", 650, 80)
    clipped = np.column_stack([np.minimum(snr[:, index], np.quantile(snr[:, index], 0.99)) for index in range(4)])
    histogram_panel(clipped, "SNR (per-term P99 clipped)", 70, 470)
    x0, y0, panel_width, panel_height = 650, 470, 500, 260
    svg.append(f'<text class="title" x="{x0}" y="{y0 - 15}">Probe sign agreement</text>')
    svg.append(f'<line x1="{x0}" y1="{y0 + panel_height}" x2="{x0 + panel_width}" y2="{y0 + panel_height}" stroke="#6b7280"/>')
    for index, (name, color) in enumerate(zip(INTERACTIONS, COLORS)):
        value = stats[name]["sign_agreement"]
        x = x0 + 45 + index * 115
        bar_height = value * panel_height
        svg.append(f'<rect x="{x}" y="{y0 + panel_height - bar_height:.1f}" width="60" height="{bar_height:.1f}" fill="{color}" opacity="0.85"/>')
        svg.append(f'<text class="label" x="{x + 15}" y="{y0 + panel_height + 22}">{name}</text>')
        svg.append(f'<text class="label" x="{x + 9}" y="{y0 + panel_height - bar_height - 7:.1f}">{value:.3f}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if len(args.inputs) < 2:
        raise ValueError("at least two probe seeds are required")
    probes = [load(path) for path in args.inputs]
    sample_ids = sorted(probes[0])
    values = []
    splits = []
    for sample_id in sample_ids:
        sample = []
        for probe in probes:
            items = probe.get(sample_id)
            if items is None or set(items) != set(SUBSET_NAMES):
                raise RuntimeError(f"incomplete targets for {sample_id}")
            subsets = torch.tensor([float(items[name]["probe_score"]) for name in SUBSET_NAMES])
            sample.append(mobius_transform(subsets, 0.0)[list(INDICES)])
        values.append(torch.stack(sample).numpy())
        splits.append(probes[0][sample_id]["tav"]["split"])
    array = np.asarray(values, dtype=np.float64)  # sample, probe, interaction
    means = array.mean(axis=1)
    variances = array.var(axis=1, ddof=1)
    snr = np.abs(means) / (np.sqrt(variances) + args.epsilon)
    signs = np.sign(array)
    sign_agreement = np.logical_or(np.all(signs >= 0, axis=1), np.all(signs <= 0, axis=1))

    stats = {}
    for index, name in enumerate(INTERACTIONS):
        stats[name] = {
            "mean": float(np.mean(means[:, index])),
            "std": float(np.std(means[:, index], ddof=1)),
            "median": float(np.median(means[:, index])),
            "mean_absolute_interaction": float(np.mean(np.abs(means[:, index]))),
            "mean_probe_variance": float(np.mean(variances[:, index])),
            "p90_probe_variance": float(np.quantile(variances[:, index], 0.9)),
            "sign_agreement": float(np.mean(sign_agreement[:, index])),
            "mean_snr": float(np.mean(snr[:, index])),
            "median_snr": float(np.median(snr[:, index])),
        }
    pair_mean_variance = float(np.mean(variances[:, :3]))
    triple_mean_variance = float(np.mean(variances[:, 3]))
    diagnosis = {
        "samples": len(sample_ids),
        "probe_count": len(probes),
        "split_counts": {split: splits.count(split) for split in sorted(set(splits))},
        "pair_mean_probe_variance": pair_mean_variance,
        "triple_mean_probe_variance": triple_mean_variance,
        "triple_to_pair_variance_ratio": triple_mean_variance / max(pair_mean_variance, 1e-12),
        "triple_variance_higher_than_each_pair": all(
            stats["tav"]["mean_probe_variance"] > stats[name]["mean_probe_variance"]
            for name in INTERACTIONS[:3]
        ),
        "interactions": stats,
    }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    figure_path = prefix.with_suffix(".svg")
    json_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 三 Probe 教师交互可靠性统计", "",
        f"样本数：{len(sample_ids)}；Probe 数：{len(probes)}。", "",
        "| 交互 | Mean | Std | Median | Mean abs(mu) | Mean variance | P90 variance | Sign agreement | Mean SNR | Median SNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in INTERACTIONS:
        row = stats[name]
        lines.append(
            f"| {name} | {row['mean']:.4f} | {row['std']:.4f} | {row['median']:.4f} | "
            f"{row['mean_absolute_interaction']:.4f} | {row['mean_probe_variance']:.4f} | "
            f"{row['p90_probe_variance']:.4f} | {row['sign_agreement']:.3f} | "
            f"{row['mean_snr']:.3f} | {row['median_snr']:.3f} |"
        )
    lines += [
        "", "## 噪声诊断", "",
        f"- Pair 平均 Probe 方差：{pair_mean_variance:.4f}",
        f"- Triple 平均 Probe 方差：{triple_mean_variance:.4f}",
        f"- Triple / Pair 方差比：{diagnosis['triple_to_pair_variance_ratio']:.3f}",
        f"- Triple 方差高于每个 pair：{diagnosis['triple_variance_higher_than_each_pair']}", "",
        f"![教师交互可靠性分布]({figure_path.name})", "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    write_svg(figure_path, means, variances, snr, stats)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "figure": str(figure_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
