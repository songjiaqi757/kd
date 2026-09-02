import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_windowed_manifest import coverage_weights, window_intervals


def test_short_sample_is_not_split() -> None:
    assert window_intervals(7.5, 30.0, 5.0) == [(0.0, 7.5)]


def test_long_sample_has_bounded_windows_and_full_coverage() -> None:
    intervals = window_intervals(91.684, 30.0, 5.0)
    assert intervals[0][0] == 0.0
    assert intervals[-1][1] == pytest.approx(91.684)
    assert all(end - start <= 30.0 for start, end in intervals)
    assert all(right[0] <= left[1] for left, right in zip(intervals, intervals[1:]))


def test_overlap_corrected_weights_sum_to_duration() -> None:
    intervals = window_intervals(63.0, 30.0, 5.0)
    weights = coverage_weights(intervals)
    assert sum(weights) == pytest.approx(63.0)
    assert all(weight > 0 for weight in weights)
