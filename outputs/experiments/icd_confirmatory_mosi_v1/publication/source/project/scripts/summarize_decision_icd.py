#!/usr/bin/env python3
"""Summarize validation-only dual-checkpoint screening and paired uncertainty."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/src"))
from rdid_mosei.decision_icd import DECISION_MODES, clustered_paired_delta

POLICIES = ("valid_mae", "valid_acc2_nonzero")


def summarize(root, seeds=(13, 42), repetitions=2000):
    rows, missing, predictions = [], [], {}
    for mode in DECISION_MODES:
        for seed in seeds:
            run = root / f"icd_decision_{mode}_seed{seed}"
            complete = (run / "status.json").exists() and json.loads((run / "status.json").read_text()).get("status") == "complete"
            for policy in POLICIES:
                directory = run / "validation" / policy
                if not complete or not (directory / "report.json").exists():
                    missing.append(f"{run.name}/{policy}")
                    continue
                row = json.loads((directory / "report.json").read_text())
                if row.get("official_test_evaluated") is not False:
                    raise ValueError("screening must be validation only")
                rows.append({"mode": mode, **row})
                predictions[(mode, seed, policy)] = [json.loads(s) for s in (directory / "predictions.jsonl").read_text().splitlines()]
    aggregates, paired, gates = [], [], []
    for mode in DECISION_MODES:
        for policy in POLICIES:
            selected = [r for r in rows if r["mode"] == mode and r["checkpoint_selection"] == policy]
            if selected:
                metrics = {}
                for metric in ("mae", "pearson", "acc2_nonzero", "f1_weighted_nonzero"):
                    values = [r["metrics"][metric] for r in selected]
                    metrics[metric] = {"mean": float(np.mean(values)),
                                       "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
                aggregates.append({"mode": mode, "checkpoint_selection": policy, "seeds": len(selected), "metrics": metrics})
            if mode == "none":
                continue
            deltas = []
            for seed in seeds:
                if (mode, seed, policy) not in predictions or ("none", seed, policy) not in predictions:
                    continue
                delta = clustered_paired_delta(predictions[("none", seed, policy)], predictions[(mode, seed, policy)],
                                               repetitions=repetitions)
                paired.append({"mode": mode, "seed": seed, "checkpoint_selection": policy, **delta})
                deltas.append(delta)
            if len(deltas) == len(seeds):
                mae = float(np.mean([d["mae"]["candidate_minus_reference"] for d in deltas]))
                acc = float(np.mean([d["acc2_nonzero"]["candidate_minus_reference"] for d in deltas]))
                positive = all(d["acc2_nonzero"]["candidate_minus_reference"] > 0 for d in deltas)
                gates.append({"mode": mode, "checkpoint_selection": policy, "mean_delta_mae": mae,
                              "mean_delta_acc2": acc, "acc2_positive_all_seeds": positive,
                              "passes_screen": positive and acc >= .005 and mae <= .005})
    return {"schema": "decision-icd-validation-screen-v1", "complete": not missing,
            "official_test_evaluated": False, "rows": rows, "missing": missing,
            "aggregates": aggregates, "paired_video_bootstrap": paired, "screening_gates": gates,
            "gate_definition": "Acc2 mean gain >=0.005, positive in both seeds, mean MAE degradation <=0.005; primary policy valid_acc2_nonzero",
            "inference_limit": "validation-selected development evidence; bootstrap does not correct selection bias or estimate between-seed uncertainty"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42])
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    report = summarize(args.input_root, tuple(args.seeds), args.bootstrap_repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Decision ICD validation screening", "", "Both policies report all metrics; official test is not used.", "",
             "| Mode | Selection | Seeds | MAE | Acc2 NZ |", "|---|---|---:|---:|---:|"]
    for row in report["aggregates"]:
        lines.append(f"| {row['mode']} | {row['checkpoint_selection']} | {row['seeds']} | {row['metrics']['mae']['mean']:.6f} | {row['metrics']['acc2_nonzero']['mean']:.6f} |")
    lines.extend(["", f"Pending entries: {len(report['missing'])}.", "", report["gate_definition"], "", report["inference_limit"]])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": report["complete"], "rows": len(report["rows"]), "missing": len(report["missing"])}))


if __name__ == "__main__":
    main()
