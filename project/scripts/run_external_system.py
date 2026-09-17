#!/usr/bin/env python3
"""Audit data, configure, and run an official native MSA implementation."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from argparse import Namespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/src"))
from rdid_mosei.external_systems import (
    DATASETS,
    SYSTEMS,
    audit_standard_features,
    build_native_config,
    canonical_id,
    write_native_config,
)
from rdid_mosei.metrics import sentiment_metrics


REPOSITORIES = ROOT / "external/original"
SEEDS = (13, 42, 2026)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--feature-file", type=Path)
    parser.add_argument("--bert-model", type=Path, default=ROOT / "model/bert/bert-base-uncased")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--phase", choices=("train_valid", "final_test"), default="train_valid")
    parser.add_argument("--source-run", type=Path, help="Completed train_valid run whose locked checkpoint is used for final_test")
    parser.add_argument("--allow-official-test", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.feature_file is None:
        args.feature_file = ROOT / f"dataset/standard_features/{args.dataset}_aligned_50.pkl"
    if args.phase == "final_test" and not args.allow_official_test:
        parser.error("final_test requires --allow-official-test after configuration lock")
    if args.phase == "final_test" and args.source_run is None:
        parser.error("final_test requires --source-run with a completed locked train_valid run")
    if args.phase == "train_valid" and args.source_run is not None:
        parser.error("--source-run is only valid in final_test phase")
    if args.phase == "train_valid" and args.allow_official_test:
        parser.error("--allow-official-test is only valid in final_test phase")
    return args


def save_results(results, output: Path, phase: str, seed: int, system: str):
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("native runner must return exactly one seed result")
    result = results[0]
    ids = [canonical_id(value) for value in result.pop("Ids")]
    predictions = [float(value) for value in result.pop("SResults")]
    targets = [float(value) for value in result.pop("Labels")]
    result.pop("Features", None)
    deployed_parameters = int(result.pop("DeployedParameters"))
    if not (len(ids) == len(predictions) == len(targets) == len(set(ids))):
        raise ValueError("native result IDs/predictions are not one-to-one")
    metrics = sentiment_metrics(targets, predictions)
    split = "test" if phase == "final_test" else "valid"
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps({"parent_sample_id": sample_id, "split": split, "target_sentiment": target, "prediction": prediction}) + "\n" for sample_id, target, prediction in zip(ids, targets, predictions))
    )
    (output / "report.json").write_text(json.dumps({
        "system": system, "seed": seed, "split": split, "metrics": metrics,
        "native_metrics": result, "checkpoint_selection": "valid_mae",
        "deployed_parameters": deployed_parameters,
        "model_only_latency": None,
        "end_to_end_latency": None,
        "latency_unavailable_reason": "official native runner does not expose a protocol-compatible isolated inference timer; original feature-extraction chain is not part of aligned_50.pkl evaluation",
    }, indent=2, default=float) + "\n")


def normalize_dpdf_results(output: Path, phase: str, seed: int):
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines() if line.strip()]
    ids = [canonical_id(row["id"]) for row in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("DPDF-LQ returned duplicate IDs")
    targets = [float(row["target"]) for row in predictions]
    values = [float(row["prediction"]) for row in predictions]
    normalized = [
        {"parent_sample_id": sample_id, "split": "test" if phase == "final_test" else "valid", "target_sentiment": target, "prediction": prediction}
        for sample_id, target, prediction in zip(ids, targets, values)
    ]
    (output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in normalized))
    native = json.loads((output / "report.json").read_text())
    (output / "report.json").write_text(json.dumps({
        "system": "dpdf_lq", "seed": seed, "split": normalized[0]["split"],
        "metrics": sentiment_metrics(targets, values), "native_metrics": native.get("metrics", {}),
        "best_epoch": native.get("best_epoch"), "checkpoint_selection": "valid_mae",
        "deployed_parameters": native.get("deployed_parameters"),
        "model_only_latency": None, "end_to_end_latency": None,
        "latency_unavailable_reason": "official native runner does not expose a protocol-compatible isolated inference timer; original feature-extraction chain is not part of aligned_50.pkl evaluation",
    }, indent=2) + "\n")


def run_python_system(args, config_path, checkpoint):
    evaluate_test = args.phase == "final_test"
    native_overrides = {
        "evaluate_test": evaluate_test,
        "return_sample_results": True,
        "train_model": checkpoint is None,
    }
    if checkpoint is not None:
        native_overrides["checkpoint_path"] = str(checkpoint)
    if args.system == "dlf":
        repository = REPOSITORIES / "DLF"
        sys.path.insert(0, str(repository))
        module = importlib.import_module("run")
        return module.DLF_run(
            "DLF", args.dataset, config_file=str(config_path), seeds=[args.seed],
            is_training=True, mode="train", gpu_ids=[args.gpu], num_workers=args.num_workers,
            model_save_dir=str((args.source_run if checkpoint else args.output) / "checkpoints"), res_save_dir=str(args.output / "native_results"),
            log_dir=str(args.output / "logs"),
            config=native_overrides,
        )
    repository = REPOSITORIES / "GsiT/src/MMSA-GsiT"
    sys.path.insert(0, str(repository))
    module = importlib.import_module("run")
    command_args = Namespace(use_embedding=0, enhance_net=[0, 0])
    return module.MMSA_run(
        "gsit", args.dataset, config_file=str(config_path), seeds=[args.seed],
        gpu_ids=[args.gpu], num_workers=args.num_workers,
        model_save_dir=str((args.source_run if checkpoint else args.output) / "checkpoints"), res_save_dir=str(args.output / "native_results"),
        log_dir=str(args.output / "logs"), cmd_args=command_args,
        config=native_overrides,
    )


def locked_checkpoint(args):
    if args.phase != "final_test":
        return None, None
    source = args.source_run.resolve()
    status = json.loads((source / "status.json").read_text())
    source_protocol = json.loads((source / "run_config.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("native final_test requires a completed source run")
    expected = {"system": args.system, "dataset": args.dataset, "seed": args.seed, "phase": "train_valid"}
    if any(source_protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("source run system/dataset/seed/phase differs from final_test request")
    checkpoints = sorted((source / "checkpoints").glob("*.pth"))
    if len(checkpoints) != 1:
        raise ValueError(f"expected exactly one locked valid-best checkpoint in {source / 'checkpoints'}")
    return source, checkpoints[0]


def main():
    args = parse_args()
    source_run, checkpoint = locked_checkpoint(args)
    if source_run is not None:
        args.source_run = source_run
    feature_path = args.feature_file.resolve()
    bert_model = args.bert_model.resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(f"standard MMSA feature file required: {feature_path}")
    if not bert_model.is_dir():
        raise FileNotFoundError(f"native BERT checkpoint required: {bert_model}")
    dataset_root = ROOT / f"dataset/cmu_{args.dataset}"
    audit = audit_standard_features(feature_path, args.dataset, dataset_root)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"native output must be new: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    config, kind = build_native_config(args.system, args.dataset, feature_path, bert_model, REPOSITORIES)
    extension = "json" if kind == "json" else "yaml"
    config_path = output / f"native_config.{extension}"
    write_native_config(config, kind, config_path)
    sources = json.loads((ROOT / "external/SOURCES.json").read_text())
    source = next(item for item in sources["repositories"] if item["method"].lower().replace("-", "_") == args.system or (args.system == "dpdf_lq" and item["method"] == "DPDF-LQ"))
    protocol = {
        "system": args.system, "dataset": args.dataset, "seed": args.seed, "phase": args.phase,
        "official_test_evaluated": args.phase == "final_test", "checkpoint_selection": "valid_mae",
        "feature_file": str(feature_path), "feature_sha256": sha256(feature_path),
        "bert_model": str(bert_model), "native_config": str(config_path), "native_config_sha256": sha256(config_path),
        "upstream": source, "adaptation": "native architecture; strict ID/label coverage; valid-only selection; unified output",
    }
    if source_run is not None:
        protocol.update(
            source_run=str(source_run),
            source_run_config_sha256=sha256(source_run / "run_config.json"),
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint),
        )
    (output / "run_config.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps({"status": "ready", "protocol": protocol, "audit": audit}, indent=2))
        return
    if args.system == "dpdf_lq":
        command = [sys.executable, "train.py", "--config_file", str(config_path), "--seed", str(args.seed), "--gpu_id", str(args.gpu), "--output_dir", str(output)]
        if args.phase == "final_test":
            command.extend(["--evaluate_test", "--checkpoint", str(checkpoint)])
        subprocess.run(command, cwd=REPOSITORIES / "DPDF-LQ", check=True)
        normalize_dpdf_results(output, args.phase, args.seed)
    else:
        results = run_python_system(args, config_path, checkpoint)
        save_results(results, output, args.phase, args.seed, args.system)
    (output / "status.json").write_text(json.dumps({"status": "complete"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
