import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_utility_crossfit_folds import assign


def test_video_groups_are_complete_disjoint_and_deterministic():
    rows = [
        {"split": "train", "video_id": f"v{video}", "parent_sample_id": f"v{video}[{utterance}]"}
        for video in range(11)
        for utterance in range(video % 3 + 1)
    ] + [{"split": "valid", "video_id": "heldout", "parent_sample_id": "heldout[0]"}]
    left = assign(rows, 5, 2026)
    right = assign(rows, 5, 2026)
    assert left == right
    assert set(left["video_to_fold"]) == {f"v{i}" for i in range(11)}
    assert "heldout" not in left["video_to_fold"]
    assert left["valid_or_test_assigned"] is False


def test_fold_count_must_be_valid():
    try:
        assign([], 1, 2026)
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("invalid fold count accepted")
