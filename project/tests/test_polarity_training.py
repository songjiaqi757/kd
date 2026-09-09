from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("train_student_baseline", PROJECT_ROOT / "scripts/train_student_baseline.py")
assert SPEC is not None and SPEC.loader is not None
TRAINER = module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


def test_weighted_binary_task_loss_excludes_zero_labels() -> None:
    sentiment = torch.tensor([-1.0, 0.0, 1.0])
    weights = torch.ones(3)
    logits = torch.tensor([[4.0, -4.0], [100.0, -100.0], [-4.0, 4.0]], requires_grad=True)
    first = TRAINER.weighted_binary_classification_loss(logits, sentiment, weights)
    changed = logits.detach().clone()
    changed[1] = torch.tensor([-100.0, 100.0])
    second = TRAINER.weighted_binary_classification_loss(changed, sentiment, weights)
    assert torch.allclose(first, second)
    first.backward()
    assert torch.equal(logits.grad[1], torch.zeros(2))


def test_binary_kd_returns_one_value_per_sample_and_backpropagates() -> None:
    student = torch.randn(4, 2, requires_grad=True)
    teacher = torch.randn(4, 7)
    values = TRAINER.weighted_binary_kd_per_sample(
        student,
        teacher,
        calibration_temperature=0.9,
        distillation_temperature=2.0,
    )
    assert values.shape == (4,)
    assert torch.all(values >= 0.0)
    values.mean().backward()
    assert student.grad is not None


def test_masked_weighted_mean_excludes_zero_labels() -> None:
    values = torch.tensor([1.0, 1000.0, 3.0])
    sentiment = torch.tensor([-1.0, 0.0, 1.0])
    weights = torch.tensor([1.0, 10.0, 3.0])
    assert torch.allclose(TRAINER.masked_weighted_mean(values, sentiment, weights), torch.tensor(2.5))
