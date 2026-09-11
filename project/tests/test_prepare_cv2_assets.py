from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_cv2_assets.py"
SPEC = importlib.util.spec_from_file_location("prepare_cv2_assets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_shifted_zero_baseline_matches_nonzero_baseline_mobius() -> None:
    values = np.asarray([[1.0, 2.0, 4.0, 3.5, 5.5, 7.0, 8.0]])
    baseline = 0.75
    expected = MODULE.mobius_numpy(values, baseline)
    actual = MODULE.mobius_numpy(values - baseline, 0.0)
    np.testing.assert_allclose(actual, expected)


def test_train_empty_baseline_uses_unique_train_tav_only() -> None:
    rows = [
        {"split": "train", "subset": "tav", "parent_sample_id": "a", "probe_score": 1.0},
        {"split": "train", "subset": "tav", "parent_sample_id": "b", "probe_score": 3.0},
        {"split": "valid", "subset": "tav", "parent_sample_id": "c", "probe_score": 99.0},
        {"split": "train", "subset": "ta", "parent_sample_id": "a", "probe_score": 20.0},
    ]
    assert MODULE.train_empty_baseline(rows) == 2.0


def test_shift_rows_preserves_original_and_seals_scope() -> None:
    rows = [{"split": "train", "subset": "t", "parent_sample_id": "a", "probe_score": 2.5}]
    shifted = MODULE.shift_rows(rows, 0.5)
    assert shifted[0]["probe_score"] == 2.0
    assert shifted[0]["probe_score_unshifted"] == 2.5
    assert shifted[0]["cv2_empty_baseline"] == 0.5
