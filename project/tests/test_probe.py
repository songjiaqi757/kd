from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.probe import TeacherProbe, last_valid_input_state


def test_last_valid_input_state_handles_left_and_right_padding() -> None:
    hidden = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])
    pooled = last_valid_input_state(hidden, mask)
    assert torch.equal(pooled[0], hidden[0, 2])
    assert torch.equal(pooled[1], hidden[1, 4])


def test_last_valid_input_state_rejects_generation_tokens() -> None:
    hidden_with_generated_tokens = torch.zeros(1, 6, 4)
    input_mask = torch.ones(1, 5)
    with pytest.raises(ValueError, match="generated tokens"):
        last_valid_input_state(hidden_with_generated_tokens, input_mask)


def test_last_valid_input_state_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="at least one valid"):
        last_valid_input_state(torch.zeros(1, 3, 4), torch.zeros(1, 3))


def test_teacher_probe_shapes_and_regression_range() -> None:
    probe = TeacherProbe(input_size=16, hidden_size=8, classes=7, dropout=0.0)
    outputs = probe(torch.randn(4, 16))
    assert outputs["classification_logits"].shape == (4, 7)
    assert outputs["regression"].shape == (4,)
    assert torch.all(outputs["regression"].abs() <= 3.0)
