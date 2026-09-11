from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from train_student_lora import configure_method


def arguments(method, ensemble=None):
    return SimpleNamespace(method=method, teacher_targets_ensemble=ensemble)


def test_task_only_lora_needs_no_ensemble():
    args = arguments("student")
    configure_method(args, [{"teacher_subset_scores": [0.0] * 7}])
    assert args.interaction_utility is None


def test_ensemble_pair_requires_multiple_teachers():
    with pytest.raises(ValueError, match="requires at least two"):
        configure_method(arguments("ensemble_pair"), [{"teacher_subset_scores": [0.0] * 7}])


def test_uniform_ensemble_pair_has_unit_utility_weights():
    args = arguments("ensemble_pair", [Path("a"), Path("b")])
    configure_method(args, [{"teacher_subset_scores": [0.0] * 7}])
    assert args.interaction_utility_normalized == [1.0, 1.0, 1.0]


def test_online_frozen_control_flag_is_available():
    from train_student_lora import parse_args
    import sys

    previous = sys.argv
    try:
        sys.argv = ["train_student_lora.py", "--output", "unused", "--freeze-text-audio"]
        assert parse_args().freeze_text_audio is True
    finally:
        sys.argv = previous
