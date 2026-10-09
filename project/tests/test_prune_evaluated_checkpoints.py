import hashlib
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prune_evaluated_checkpoints as prune
import manage_checkpoint_archive as archive


def setup_run(tmp_path, monkeypatch):
    root = tmp_path / "project"
    experiments = root / "outputs/experiments"
    archives = root / "outputs/checkpoint_archive_v1"
    monkeypatch.setattr(prune, "ROOT", root)
    monkeypatch.setattr(prune, "EXPERIMENTS", experiments)
    monkeypatch.setattr(prune, "ARCHIVES", archives)
    run = experiments / "example/students/run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "status.json").write_text('{"status":"complete"}')
    (run / "history.json").write_text(json.dumps([
        {"epoch": e, "valid_metrics": {"mae": mae, "acc2_nonzero": acc}}
        for e, mae, acc in [(1, .5, .6), (2, .6, .8), (3, .7, .7), (4, .8, .5)]
    ]))
    for epoch in range(1, 5):
        (run / "checkpoints" / f"epoch_{epoch:03d}.pt").write_bytes(bytes([epoch]) * 2**21)
    (run / "best.pt").symlink_to("checkpoints/epoch_001.pt")
    (run / "last.pt").write_bytes(b"resumable-state")
    # Even selected epochs have evaluation evidence: they must still survive.
    for epoch in [1, 2, 3]:
        evaluation = experiments / f"example/evaluation/epoch_{epoch:03d}"
        evaluation.mkdir(parents=True)
        checkpoint = run / "checkpoints" / f"epoch_{epoch:03d}.pt"
        (evaluation / "status.json").write_text('{"status":"complete"}')
        (evaluation / "report.json").write_text(json.dumps({
            "epoch": epoch, "test_metrics": {"count": 1, "mae": .2},
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }))
        (evaluation / "predictions.jsonl").write_text('{"split":"test","prediction":0.2}\n')
    return run, archives


def test_only_evaluated_nonbest_epoch_deleted_and_records_retained(tmp_path, monkeypatch):
    run, _ = setup_run(tmp_path, monkeypatch)
    plan = prune.plan()
    assert len(plan["candidates"]) == 1
    candidate = plan["candidates"][0]
    assert candidate["epoch"] == 3
    with (tmp_path / "operations.jsonl").open("w") as log:
        prune.prune_record(candidate, prune.linked_targets(prune.ROOT), log, tmp_path / "plan.json")
    assert not (run / "checkpoints/epoch_003.pt").exists()
    for name in ["best.pt", "last.pt", "history.json", "checkpoints/epoch_001.pt",
                 "checkpoints/epoch_002.pt", "checkpoints/epoch_004.pt"]:
        assert (run / name).is_file()
    assert Path(candidate["evaluation"]["report"]).is_file()
    assert all(Path(e["path"]).is_file() for e in candidate["evaluation"]["evidence"])


def test_verified_archive_pruning_leaves_nonrecoverable_tombstone(tmp_path, monkeypatch):
    run, archives = setup_run(tmp_path, monkeypatch)
    checkpoint = run / "checkpoints/epoch_003.pt"
    record = archive.compress_checkpoint(checkpoint, archives, prune.EXPERIMENTS)
    plan = prune.plan()
    assert len(plan["candidates"]) == 1
    with (tmp_path / "operations.jsonl").open("w") as log:
        prune.prune_record(plan["candidates"][0], set(), log, tmp_path / "plan.json")
    assert not Path(record["archive"]).exists()
    pointer = checkpoint.with_suffix(".pt.archive.json")
    metadata = json.loads(pointer.read_text())
    assert metadata["archive_available"] is False
    assert metadata["status"] == "permanently_deleted_after_evaluation"


@pytest.mark.parametrize("change", ["active", "history", "checkpoint", "evaluation"])
def test_changes_since_plan_block_deletion(tmp_path, monkeypatch, change):
    run, _ = setup_run(tmp_path, monkeypatch)
    candidate = prune.plan()["candidates"][0]
    checkpoint = run / "checkpoints/epoch_003.pt"
    if change == "active":
        (run / "status.json").write_text('{"status":"training"}')
    elif change == "history":
        with (run / "history.json").open("a") as stream:
            stream.write(" ")
    elif change == "checkpoint":
        checkpoint.write_bytes(b"changed")
    else:
        Path(candidate["evaluation"]["report"]).write_text('{}')
    with (tmp_path / "operations.jsonl").open("w") as log:
        with pytest.raises(ValueError):
            prune.prune_record(candidate, set(), log, tmp_path / "plan.json")
    assert checkpoint.exists()


@pytest.mark.parametrize("change", ["failed_evaluation", "hash_mismatch", "no_predictions", "missing_best", "missing_last"])
def test_incomplete_evidence_or_retention_excludes_candidate(tmp_path, monkeypatch, change):
    run, _ = setup_run(tmp_path, monkeypatch)
    evaluation = prune.EXPERIMENTS / "example/evaluation/epoch_003"
    if change == "failed_evaluation":
        (evaluation / "status.json").write_text('{"status":"failed"}')
    elif change == "hash_mismatch":
        report = json.loads((evaluation / "report.json").read_text())
        report["checkpoint_sha256"] = "0" * 64
        (evaluation / "report.json").write_text(json.dumps(report))
    elif change == "no_predictions":
        (evaluation / "predictions.jsonl").unlink()
    elif change == "missing_best":
        (run / "checkpoints/epoch_002.pt").unlink()
    else:
        (run / "last.pt").unlink()
    assert prune.plan()["candidate_count"] == 0
