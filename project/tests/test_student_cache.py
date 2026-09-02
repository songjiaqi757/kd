from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.student import CachedStudentCore


def test_cached_student_core_all_subsets_and_backward() -> None:
    core = CachedStudentCore(
        text_hidden_size=12,
        audio_hidden_size=10,
        video_hidden_size=8,
        hidden_size=16,
        tokens_per_modality=2,
        qformer_layers=1,
        fusion_layers=1,
        heads=2,
        ffn_size=32,
        dropout=0.0,
    )
    states = {"t": torch.randn(2, 5, 12), "a": torch.randn(2, 7, 10), "v": torch.randn(2, 9, 8)}
    masks = {key: torch.ones(value.shape[:2], dtype=torch.bool) for key, value in states.items()}
    outputs = core(states, masks)
    assert len(outputs) == 7
    outputs["tav"]["regression"].sum().backward()
    assert core.pools["t"].queries.grad is not None
