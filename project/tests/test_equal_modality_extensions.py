from pathlib import Path
import json
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_equal_modality_method_extensions as summary
import evaluate_main_table_test as evaluator


def test_summary_requires_and_aggregates_all_methods_and_seeds(tmp_path, monkeypatch):
    input_root = tmp_path / "runs"
    for method_index, method in enumerate(summary.METHODS):
        for seed in summary.SEEDS:
            for policy in summary.CHECKPOINT_SELECTIONS:
                run = input_root / f"{method}_seed{seed}" / "test" / policy
                run.mkdir(parents=True)
                metrics = {
                    metric: 0.5 + method_index * 0.01 + seed * 1e-6
                    for metric in summary.METRICS
                }
                (run / "report.json").write_text(json.dumps({
                    "method": method,
                    "seed": seed,
                    "selected_epoch": 2,
                    "checkpoint_selection": policy,
                    "metrics": metrics,
                }))
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["summary", "--input-root", str(input_root), "--output", str(output)],
    )
    summary.main()
    payload = json.loads(output.read_text())
    assert len(payload["rows"]) == (
        len(summary.METHODS) * len(summary.SEEDS) * len(summary.CHECKPOINT_SELECTIONS)
    )
    assert output.with_suffix(".csv").is_file()
    delta = payload["paired_deltas"]["valid_mae"][summary.METHODS[1]]["f1_weighted_nonzero"]
    assert delta["mean_candidate_minus_icd"] == pytest.approx(0.01)


def test_checkpoint_selection_uses_all_retained_epochs(tmp_path):
    history = [
        {"epoch": 1, "valid_metrics": {"mae": 0.8, "acc2_nonzero": 0.70}},
        {"epoch": 2, "valid_metrics": {"mae": 0.7, "acc2_nonzero": 0.68}},
        {"epoch": 3, "valid_metrics": {"mae": 0.75, "acc2_nonzero": 0.74}},
    ]
    (tmp_path / "history.json").write_text(json.dumps(history))
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for row in history:
        torch.save(
            {"epoch": row["epoch"], "valid_metrics": row["valid_metrics"], "model": {}},
            checkpoints / f"epoch_{row['epoch']:03d}.pt",
        )

    mae_path, mae_epoch, _ = evaluator.selected_checkpoint(tmp_path, "valid_mae")
    acc_path, acc_epoch, _ = evaluator.selected_checkpoint(tmp_path, "valid_acc2_nonzero")
    assert mae_epoch == 2 and mae_path.name == "epoch_002.pt"
    assert acc_epoch == 3 and acc_path.name == "epoch_003.pt"
