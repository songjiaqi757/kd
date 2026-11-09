#!/usr/bin/env python3
"""Materialize the preregistered command matrix without launching training."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def combinations(tuning):
    if not tuning:
        return [dict()]
    keys = sorted(tuning)
    return [dict(zip(keys, values)) for values in itertools.product(*(tuning[key] for key in keys))]


def flag(name):
    return "--" + name.replace("_", "-")


def build(matrix, dataset, include_tier3=False):
    protocol = matrix["protocol"]
    dataset_config = matrix["datasets"][dataset]
    assets = ROOT / dataset_config["assets"]
    bert_model = ROOT / "model/bert/bert-base-uncased"
    output_root = ROOT / f"outputs/experiments/main_table_v1/{dataset}"
    runs = []
    methods = matrix["main_table_a"] + matrix["mechanism_ablations"]
    if include_tier3:
        methods += matrix.get("tier3_student_extensions", [])
    for specification in methods:
        candidates = combinations(specification["tuning"])
        for candidate_index, parameters in enumerate(candidates):
            tuning = len(candidates) > 1
            seeds = [protocol["seeds"][0]] if tuning else protocol["seeds"]
            for seed in seeds:
                suffix = "" if not tuning else f"_candidate{candidate_index + 1}"
                run_name = f"{specification['method']}{suffix}_seed{seed}"
                command = [
                    str(ROOT.parent / "miniconda3/envs/kd/bin/python"),
                    str(ROOT / "project/scripts/train_main_table_kd.py"),
                    "--method", specification["method"], "--assets", str(assets),
                    "--output", str(output_root / "students" / run_name), "--seed", str(seed),
                ]
                for key, value in parameters.items():
                    command.extend([flag(key), str(value)])
                # The frozen protocol is the source of truth.  This lets a
                # resumable asset-preparation service unlock MOSI without a
                # later manual edit to the experiment matrix.
                ready = assets.is_file()
                run = {
                    "kind": "student", "dataset": dataset, "name": run_name,
                    "method": specification["method"], "seed": seed, "parameters": parameters,
                    "stage": "valid_tuning" if tuning else "replication",
                    "command": command,
                    "ready": ready,
                    "tier": specification["tier"],
                    "optional": specification.get("optional", False),
                    "claim_driven": specification.get("claim_driven", False),
                }
                if not ready:
                    run["blocked_reason"] = dataset_config.get(
                        "blocking_prerequisite", f"student asset protocol is missing: {assets}"
                    )
                runs.append(run)
        if len(candidates) > 1:
            runs.append({
                "kind": "selection_gate", "dataset": dataset, "method": specification["method"],
                "rule": "choose minimum seed13 valid MAE; freeze candidate; then run seeds 42/2026 and retain seed13",
                "ready": False,
                "blocked_reason": "await all seed13 validation candidates for this method",
            })
            for seed in protocol["seeds"][1:]:
                runs.append({
                    "kind": "selected_replication", "dataset": dataset,
                    "name": f"{specification['method']}_selected_seed{seed}",
                    "method": specification["method"], "seed": seed,
                    "tier": specification["tier"], "stage": "post_selection_replication",
                    "parameters": "inherit seed13 minimum-valid-MAE candidate",
                    "ready": False,
                    "blocked_reason": "await seed13 candidate selection; materialize the selected hyperparameters before launch",
                })
    for external in matrix["main_table_b"]:
        if external["system"] in {"ours"} or (external["tier"] == 3 and not include_tier3):
            continue
        for seed in protocol["seeds"]:
            name = f"{external['system']}_seed{seed}"
            feature_file = ROOT / dataset_config["standard_features"]
            ready = (
                dataset_config["native_features_ready"]
                and feature_file.is_file()
                and bert_model.is_dir()
            )
            run = {
                "kind": "native_system", "dataset": dataset, "name": name,
                "system": external["system"], "seed": seed, "stage": "train_valid",
                "command": [
                    str(ROOT.parent / "miniconda3/envs/kd/bin/python"),
                    str(ROOT / "project/scripts/run_external_system.py"),
                    "--system", external["system"], "--dataset", dataset,
                    "--feature-file", str(feature_file),
                    "--bert-model", str(bert_model),
                    "--output", str(output_root / "native" / name), "--seed", str(seed),
                ],
                "ready": ready,
                "tier": external["tier"],
            }
            if not ready:
                missing = []
                if not dataset_config["native_features_ready"] or not feature_file.is_file():
                    missing.append(f"audited standard MMSA features: {feature_file}")
                if not bert_model.is_dir():
                    missing.append(f"native BERT checkpoint: {bert_model}")
                run["blocked_reason"] = "; ".join(missing)
            runs.append(run)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "project/configs/main_table_v1/experiments.json")
    parser.add_argument("--dataset", choices=("mosei", "mosi"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-tier3", action="store_true", help="Include enhancement experiments outside the minimum paper set")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    runs = build(matrix, args.dataset, include_tier3=args.include_tier3)
    output = args.output or ROOT / f"project/configs/main_table_v1/{args.dataset}_plan.jsonl"
    output.write_text("".join(json.dumps(run, ensure_ascii=False) + "\n" for run in runs))
    training_nodes = [run for run in runs if run["kind"] in {"student", "selected_replication", "native_system"}]
    print(json.dumps({
        "dataset": args.dataset,
        "nodes": len(runs),
        "tier3_included": args.include_tier3,
        "planned_training_runs": len(training_nodes),
        "materialized_commands": sum("command" in run for run in training_nodes),
        "ready_runs": sum(run["ready"] for run in training_nodes),
        "blocked_runs": sum(not run["ready"] for run in training_nodes),
        "pending_selection_gates": sum(run["kind"] == "selection_gate" for run in runs),
        "output": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
