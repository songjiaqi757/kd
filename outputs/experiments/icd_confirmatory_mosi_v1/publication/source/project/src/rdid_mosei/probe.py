from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def last_valid_input_state(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Select the last non-padding input token for each sample.

    The sequence lengths must match exactly, which intentionally rejects hidden
    states that contain appended generation tokens.
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be [batch, sequence, hidden], got {hidden_states.shape}")
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [batch, sequence], got {attention_mask.shape}")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError(
            "hidden-state sequence must exactly match the input attention mask; "
            "generated tokens are not allowed"
        )
    valid = attention_mask.to(dtype=torch.bool)
    if not torch.all(valid.any(dim=1)):
        raise ValueError("every sample must contain at least one valid input token")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
    last_indices = positions.masked_fill(~valid, -1).max(dim=1).values
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, last_indices.to(hidden_states.device)]


def extract_thinker_last_input_state(model: Any, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if "attention_mask" not in inputs:
        raise KeyError("processor inputs must include attention_mask")
    outputs = model.thinker(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if not outputs.hidden_states:
        raise RuntimeError("Thinker did not return hidden states")
    return last_valid_input_state(outputs.hidden_states[-1], inputs["attention_mask"])


class TeacherProbe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 512, classes: int = 7, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_size, classes)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(hidden_state)
        regression = 3.0 * torch.tanh(self.regressor(features).squeeze(-1))
        return {"classification_logits": self.classifier(features), "regression": regression}
