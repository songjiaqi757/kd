#!/usr/bin/env python3
"""Rebuild the three CSV tables from the frozen MOSI First campaign outputs.

Usage:
    python source/build_tables.py \
      --campaign-root /ai/sjq/kd/outputs/experiments/mosi_first_20261111_v1 \
      --output tables
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

POLICIES = (
    ("valid_mae", "best_valid_mae.pt"),
    ("valid_acc2_nonzero", "best_valid_acc2_nonzero.pt"),
    ("last", "last.pt"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("tables"), type=Path)
    args = parser.parse_args()
    root = args.campaign_root
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    plan = json.loads((root / "plan.json").read_text())
    jobs = {job["id"]: job for job in plan["jobs"]}
    metric_rows, config_rows, transform_rows = [], [], []

    for job_id, job in jobs.items():
        run = root / "runs" / job_id
        config = json.loads((run / "run_config.json").read_text())
        inventory = json.loads((run / "checkpoint_inventory.json").read_text())
        checkpoints = {os.path.basename(c["path"]): c for c in inventory["checkpoints"]}
        transform = json.loads((run / "transform_metadata.json").read_text())

        config_rows.append({
            "method": job["name"], "mechanism": job["mechanism"],
            "task_ce": job["task_ce"], "lambda_interaction": job["lambda_interaction"],
            "loss_kind": job["loss_kind"], "matrix_seed": job["matrix_seed"],
            "seed": job["seed"], "epochs": config.get("epochs"),
            "min_epochs": config.get("min_epochs"), "patience": config.get("patience"),
            "batch_size": config.get("batch_size"), "learning_rate": config.get("learning_rate"),
            "weight_decay": config.get("weight_decay"), "video_layers": config.get("video_layers"),
            "video_rank": config.get("video_rank"), "video_alpha": config.get("video_alpha"),
            "video_dropout": config.get("video_dropout"), "lambda_feature": config.get("lambda_feature"),
            "lambda_subset": config.get("lambda_subset"), "kd_kl": config.get("kd_kl"),
            "coordinate_weights": json.dumps(config.get("coordinate_weights")),
            "icd_warmup": config.get("icd_warmup"), "lr_schedule": config.get("lr_schedule"),
            "transform_sha256": transform["sha256"],
            "transform_condition_number": transform["condition_number"],
            "transform_singular_values": json.dumps(transform["singular_values"]),
            "checkpoint_policies": "valid_mae;valid_acc2_nonzero;last",
        })
        transform_rows.append({
            "method": job["name"], "mechanism": job["mechanism"],
            "loss_kind": job["loss_kind"], "matrix_seed": job["matrix_seed"],
            "transform_sha256": transform["sha256"],
            "singular_values": json.dumps(transform["singular_values"]),
            "condition_number": transform["condition_number"],
            "matrix": json.dumps(transform["matrix"]),
        })

        for policy, filename in POLICIES:
            test_path = root / "tests" / f"{job_id}__{policy}" / "report.json"
            if not test_path.exists():
                continue
            test = json.loads(test_path.read_text())
            test_metrics = test["metrics"]
            valid_metrics = test["selected_valid_metrics"]
            checkpoint = checkpoints[filename]
            metric_rows.append({
                "method": job["name"], "mechanism": job["mechanism"],
                "task_ce": job["task_ce"], "lambda_interaction": job["lambda_interaction"],
                "loss_kind": job["loss_kind"], "policy": policy,
                "selected_epoch": test["selected_epoch"],
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_bytes": checkpoint["bytes"],
                "test_count": test_metrics["count"], "test_mae": test_metrics["mae"],
                "test_pearson": test_metrics["pearson"],
                "test_acc2_nonzero": test_metrics["acc2_nonzero"],
                "test_f1_weighted_nonzero": test_metrics["f1_weighted_nonzero"],
                "test_acc7": test_metrics["acc7"],
                "valid_count": valid_metrics["count"], "valid_mae": valid_metrics["mae"],
                "valid_pearson": valid_metrics["pearson"],
                "valid_acc2_nonzero": valid_metrics["acc2_nonzero"],
                "valid_f1_weighted_nonzero": valid_metrics["f1_weighted_nonzero"],
                "valid_acc7": valid_metrics["acc7"],
            })

    for filename, rows in (
        ("per_checkpoint_metrics.csv", metric_rows),
        ("method_configs.csv", config_rows),
        ("transform_matrices.csv", transform_rows),
    ):
        path = out / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
