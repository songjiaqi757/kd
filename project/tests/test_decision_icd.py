import json
from pathlib import Path
import sys

import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / name) for name in ("scripts", "src")]
from rdid_mosei.decision_icd import decision_distillation_loss, error_strata, clustered_paired_delta
from rdid_mosei.interaction import mobius_transform
from rdid_mosei.student import SUBSETS
import manage_checkpoint_archive as archive


def outputs(scores):
    return {s: {"regression": scores[:, i]} for i, s in enumerate(SUBSETS)}


@pytest.mark.parametrize("mode", ["none", "full", "subset", "interaction"])
def test_exact_teacher_match_and_finite_gradients(mode):
    scores = torch.randn(4, 7, requires_grad=True)
    loss = decision_distillation_loss(outputs(scores), scores.detach(), torch.ones(4), .17, mode=mode)
    assert loss.item() == pytest.approx(0, abs=1e-9)
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_full_control_only_supervises_tav_and_teacher_is_detached():
    scores = torch.zeros(2, 7, requires_grad=True)
    teacher = torch.ones(2, 7, requires_grad=True)
    loss = decision_distillation_loss(outputs(scores), teacher, torch.tensor([1., 0.]), .17, mode="full")
    loss.backward()
    assert scores.grad[0, -1] < 0
    assert torch.count_nonzero(scores.grad[:, :-1]) == 0
    assert torch.count_nonzero(scores.grad[1]) == 0
    assert teacher.grad is None


def test_interaction_transforms_decision_values_not_raw_coordinates():
    values = torch.tensor([[1., -1., .4, .7, -.1, .8, -.3]], requires_grad=True)
    target = torch.zeros(1, 7)
    baseline = torch.sigmoid(torch.tensor(.17))
    expected = torch.nn.functional.smooth_l1_loss(
        mobius_transform(values.sigmoid(), baseline),
        mobius_transform(target.sigmoid(), baseline))
    actual = decision_distillation_loss(outputs(values), target, torch.ones(1), .17, mode="interaction")
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad) == 7


def rows(predictions):
    return [{"parent_sample_id": str(i), "video_id": str(i//2), "split": "valid",
             "target_sentiment": y, "prediction": p}
            for i, (y, p) in enumerate(predictions)]


def test_strata_neutral_and_exact_zero_boundary_and_mae_accounting():
    records = rows([(0, .2), (.5, -.2), (-1, 0), (2, 1)])
    result = error_strata(records)
    groups = result["groups"]
    assert groups["near_zero_wrong_sign"]["count"] == 1
    assert groups["away_from_zero_wrong_sign"]["count"] == 1
    assert groups["correct_sign"]["count"] == 1
    assert sum(groups[g]["total_mae_contribution"] for g in
               ("neutral", "near_zero_wrong_sign", "away_from_zero_wrong_sign", "correct_sign")) == pytest.approx(result["metrics"]["mae"])
    records[0]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        error_strata(records)


def test_paired_bootstrap_alignment_and_delta_direction():
    reference = rows([(-1, 1), (1, -1), (-.5, -.2), (.5, .2)])
    candidate = rows([(-1, -1), (1, 1), (-.5, -.5), (.5, .5)])
    result = clustered_paired_delta(reference, list(reversed(candidate)), repetitions=100)
    assert result["video_clusters"] == 2
    assert result["mae"]["candidate_minus_reference"] < 0
    assert result["acc2_nonzero"]["candidate_minus_reference"] == .5
    with pytest.raises(ValueError):
        clustered_paired_delta(reference, candidate[:-1])


def make_run(tmp_path, state="complete"):
    run = tmp_path / "experiments" / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"status": state}))
    history = [{"epoch": e, "valid_metrics": {"mae": mae, "acc2_nonzero": acc}}
               for e, mae, acc in ((1, .7, .6), (2, .5, .7), (3, .6, .8), (4, .65, .65))]
    (run / "history.json").write_text(json.dumps(history))
    for e in range(1, 5):
        (run / "checkpoints" / f"epoch_{e:03d}.pt").write_bytes(bytes([e]) * 2**21)
    (run / "best.pt").symlink_to("checkpoints/epoch_002.pt")
    return run


def test_archive_protects_both_selected_epochs_and_running_runs(tmp_path):
    run = make_run(tmp_path)
    report = archive.inventory(run.parent)
    assert {Path(r["path"]).name for r in report["candidates"]} == {"epoch_001.pt", "epoch_004.pt"}
    (run / "status.json").write_text('{"status":"training"}')
    assert archive.inventory(run.parent)["candidate_count"] == 0


def test_archive_exact_roundtrip_and_no_overwrite(tmp_path):
    run = make_run(tmp_path)
    path = run / "checkpoints/epoch_001.pt"
    expected = path.read_bytes()
    record = archive.compress_checkpoint(path, tmp_path / "archive", run.parent)
    assert record["verified"] and not path.exists()
    assert record["saved_bytes"] > 2**20
    pointer = path.with_suffix(".pt.archive.json")
    archive.restore(pointer)
    assert path.read_bytes() == expected
    with pytest.raises(FileExistsError):
        archive.restore(pointer)
    with pytest.raises(ValueError, match="protected"):
        archive.compress_checkpoint(run / "checkpoints/epoch_003.pt", tmp_path / "archive", run.parent)


def test_corrupt_archive_never_restores_original_path(tmp_path):
    run = make_run(tmp_path)
    path = run / "checkpoints/epoch_001.pt"
    record = archive.compress_checkpoint(path, tmp_path / "archive", run.parent)
    Path(record["archive"]).write_bytes(b"corrupt")
    with pytest.raises(Exception):
        archive.restore(path.with_suffix(".pt.archive.json"))
    assert not path.exists()
    assert not path.with_suffix(".pt.restore.tmp").exists()
