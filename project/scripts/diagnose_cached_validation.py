#!/usr/bin/env python3
"""Cheap historical validation audit without reading test predictions or models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/src"))
from rdid_mosei.decision_icd import error_strata


def audit(run):
    history = json.loads((run / "history.json").read_text())
    report = json.loads((run / "report.json").read_text())
    records = [json.loads(s) for s in (run / "predictions.jsonl").read_text().splitlines()]
    strata = error_strata(records)
    predictions_epoch = int(report["best_epoch"])
    selected = {}
    for policy, metric, sign in (("valid_mae", "mae", 1), ("valid_acc2_nonzero", "acc2_nonzero", -1)):
        best = min(history, key=lambda r: (sign*float(r["valid_metrics"][metric]), r["epoch"]))
        selected[policy] = {"epoch": best["epoch"], "metrics": best["valid_metrics"],
                            "cached_predictions_available": best["epoch"] == predictions_epoch}
    return {"run": str(run), "prediction_epoch": predictions_epoch, "selections": selected,
            "cached_prediction_error_strata": strata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [audit(p.resolve()) for p in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "historical validation only; different experiment budgets are not pooled",
                                       "runs": results}, indent=2) + "\n")
    lines = ["# Historical validation error audit", "", "All strata use fixed |GT| <= 0.5; neutral is separate.", "",
             "| Run | MAE epoch | Acc2 epoch | Cached epoch | Near-zero wrong | Away wrong |", "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        groups = r["cached_prediction_error_strata"]["groups"]
        lines.append(f"| {Path(r['run']).name} | {r['selections']['valid_mae']['epoch']} | {r['selections']['valid_acc2_nonzero']['epoch']} | {r['prediction_epoch']} | {groups['near_zero_wrong_sign']['count']} | {groups['away_from_zero_wrong_sign']['count']} |")
    lines.extend(["", "Error counts refer to the cached prediction epoch only. Missing selected-epoch predictions are not inferred; the new trainer saves them for both policies."])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"runs": len(results), "output": str(args.output)}))


if __name__ == "__main__":
    main()
