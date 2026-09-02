from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.student import QFormerPool, SUBSETS, SubsetFusion, student_task_loss


def test_qformer_pool_shape_and_mask() -> None:
    pool = QFormerPool(12, hidden_size=16, query_tokens=4, layers=2, heads=4, ffn_size=32, dropout=0.0)
    hidden = torch.randn(2, 7, 12)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    assert pool(hidden, mask).shape == (2, 4, 16)


def test_qformer_pool_rejects_empty_sequence() -> None:
    pool = QFormerPool(8, hidden_size=8, query_tokens=2, layers=1, heads=2, ffn_size=16)
    with pytest.raises(ValueError, match="at least one valid"):
        pool(torch.randn(1, 3, 8), torch.zeros(1, 3))


def test_subset_fusion_runs_all_seven_subsets_and_backward() -> None:
    fusion = SubsetFusion(
        hidden_size=16,
        tokens_per_modality=4,
        layers=2,
        heads=4,
        ffn_size=32,
        dropout=0.0,
    )
    encoded = {modality: torch.randn(2, 4, 16) for modality in "tav"}
    outputs = fusion.forward_subsets(encoded)
    assert tuple(outputs) == SUBSETS
    for output in outputs.values():
        assert output["regression"].shape == (2,)
        assert output["classification_logits"].shape == (2, 7)
        assert torch.all(output["regression"].abs() <= 3.0)
    sum(output["regression"].mean() for output in outputs.values()).backward()
    assert fusion.cls_token.grad is not None


def test_subset_fusion_rejects_unknown_subset() -> None:
    fusion = SubsetFusion(hidden_size=8, tokens_per_modality=2, layers=1, heads=2, ffn_size=16)
    encoded = {modality: torch.randn(1, 2, 8) for modality in "tav"}
    with pytest.raises(ValueError, match="unknown subset"):
        fusion(encoded, "at")


def test_student_task_loss_is_finite_and_backward_compatible() -> None:
    output = {
        "regression": torch.tensor([0.2, -0.3], requires_grad=True),
        "classification_logits": torch.randn(2, 7, requires_grad=True),
    }
    losses = student_task_loss(output, torch.tensor([1.0, -1.0]), torch.tensor([4, 2]))
    assert set(losses) == {"loss", "regression_loss", "classification_loss"}
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
