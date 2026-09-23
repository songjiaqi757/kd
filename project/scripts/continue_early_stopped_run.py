#!/usr/bin/env python3
"""Continue an early-stopped retained-checkpoint run to a fixed epoch.

The original run configuration remains the checkpoint protocol.  A separate
continuation record captures the requested epoch budget and execution host.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/scripts"))


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def gpu_description() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]


def backup(path: Path) -> None:
    if not path.is_file():
        return
    destination = path.with_name(f"{path.stem}.before_full20{path.suffix}")
    if not destination.exists():
        shutil.copy2(path, destination)


def common_arguments(config: dict, run: Path, assets: Path, target: int) -> list[str]:
    arguments = [
        "--method", str(config["method"]),
        "--assets", str(assets),
        "--output", str(run),
        "--seed", str(config["seed"]),
        "--device", "cuda:0",
        "--batch-size", str(config["batch_size"]),
        "--epochs", str(target),
        "--patience", "1000",
        "--learning-rate", str(config["learning_rate"]),
        "--weight-decay", str(config["weight_decay"]),
        "--num-workers", str(config["num_workers"]),
        "--video-layers", str(config["video_layers"]),
        "--video-rank", str(config["video_rank"]),
        "--video-alpha", str(config["video_alpha"]),
        "--video-dropout", str(config["video_dropout"]),
        "--progress-every", "100",
        "--resume",
    ]
    arguments.append(
        "--video-checkpointing" if config.get("video_checkpointing", True)
        else "--no-video-checkpointing"
    )
    if config.get("limit_per_split") is not None:
        arguments.extend(["--limit-per-split", str(config["limit_per_split"])])
    return arguments


def parse_main_arguments(config: dict, run: Path, assets: Path, target: int):
    import train_main_table_kd as trainer

    arguments = common_arguments(config, run, assets, target)
    arguments.extend([
        "--min-epochs", str(min(target, int(config.get("min_epochs", 1)))),
        "--checkpoint-selection", str(config.get("checkpoint_selection", "valid_mae")),
    ])
    if config.get("save_every_epoch", False):
        arguments.append("--save-every-epoch")
    for name in (
        "lambda_feature", "lambda_cafd", "cmad_tau", "lambda_subset",
        "lambda_interaction", "ea_entropy_temperature", "rld_temperature",
        "rld_alpha", "rld_beta", "rld_warmup_epochs", "skd_temperature",
        "skd_tik_factor",
    ):
        if config.get(name) is not None:
            arguments.extend(["--" + name.replace("_", "-"), str(config[name])])
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(trainer.__file__).resolve()), *arguments]
        return trainer, trainer.parse_args()
    finally:
        sys.argv = original_argv


def parse_tav_arguments(config: dict, run: Path, assets: Path, target: int):
    import train_tav_distillation as trainer

    arguments = common_arguments(config, run, assets, target)
    arguments.extend([
        "--teacher-subset", str(config.get("teacher_subset", "tav")),
        "--diagnostic-repetitions", str(config.get("diagnostic_repetitions", 5)),
        "--diagnostics" if config.get("diagnostics", False) else "--no-diagnostics",
    ])
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(trainer.__file__).resolve()), *arguments]
        return trainer, trainer.parse_args()
    finally:
        sys.argv = original_argv


def run_main(config: dict, run: Path, assets: Path, target: int) -> None:
    trainer, parsed = parse_main_arguments(config, run, assets, target)
    trainer.run_protocol = lambda _args, _assets: config
    if not config.get("save_every_epoch", False):
        import train_main_table_all_epochs as retained
        trainer.atomic_save = retained.atomic_save_with_epoch_retention
    with trainer.run_lock(run, True):
        trainer.train(parsed, run)
    if not config.get("save_every_epoch", False):
        import train_main_table_all_epochs as retained
        retained.audit(run)


def run_tav(config: dict, run: Path, assets: Path, target: int) -> None:
    trainer, parsed = parse_tav_arguments(config, run, assets, target)
    trainer.protocol = lambda _args: config
    import train_tav_distillation_all_epochs as retained
    trainer.atomic_save = retained.atomic_save_with_epoch_retention
    # The legacy trainer treats report+complete as an unconditional no-op.
    report = run / "report.json"
    if report.is_file():
        archived = run / "report.before_full20.json"
        if not archived.exists():
            os.replace(report, archived)
        else:
            os.replace(report, run / f"report.before_full20_retry_{int(time.time())}.json")
    with trainer.run_lock(run, True):
        trainer.train(parsed, run)
    retained.audit_completed_output(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("main_table", "tav"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--target-epoch", type=int, default=20)
    args = parser.parse_args()
    run, assets = args.run.resolve(), args.assets.resolve()
    config = json.loads((run / "run_config.json").read_text())
    history = json.loads((run / "history.json").read_text())
    if [int(row["epoch"]) for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("source history is not contiguous")
    if len(history) >= args.target_epoch:
        atomic_json({
            "schema": "full-epoch-continuation-v1", "status": "complete",
            "kind": args.kind, "target_epoch": args.target_epoch,
            "epochs": len(history), "already_complete": True,
        }, run / "full_epoch_continuation.json")
        return
    if args.target_epoch > int(config["epochs"]):
        raise ValueError("target exceeds the original configured epoch budget")
    backup(run / "report.json")
    backup(run / "status.json")
    started = time.time()
    record = {
        "schema": "full-epoch-continuation-v1",
        "status": "running",
        "kind": args.kind,
        "method": config["method"],
        "seed": config["seed"],
        "source_last_epoch": len(history),
        "target_epoch": args.target_epoch,
        "early_stopping_disabled_for_continuation": True,
        "host": platform.node(),
        "visible_gpus": gpu_description(),
        "torch_version": torch.__version__,
        "started_at_unix": started,
    }
    atomic_json(record, run / "full_epoch_continuation.json")
    try:
        if args.kind == "main_table":
            run_main(config, run, assets, args.target_epoch)
        else:
            run_tav(config, run, assets, args.target_epoch)
        final_history = json.loads((run / "history.json").read_text())
        if len(final_history) != args.target_epoch:
            raise ValueError(f"continuation ended at {len(final_history)}, expected {args.target_epoch}")
        atomic_json(record | {
            "status": "complete",
            "epochs": len(final_history),
            "completed_at_unix": time.time(),
        }, run / "full_epoch_continuation.json")
    except Exception as error:
        atomic_json(record | {
            "status": "failed", "error": repr(error),
            "updated_at_unix": time.time(),
        }, run / "full_epoch_continuation.json")
        raise


if __name__ == "__main__":
    main()
