from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = ("t", "a", "v")
SUBSETS = ("t", "a", "v", "ta", "tv", "av", "tav")


def student_task_loss(
    output: Mapping[str, torch.Tensor],
    sentiment: torch.Tensor,
    class_7_index: torch.Tensor,
    classification_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    regression = F.smooth_l1_loss(output["regression"].float(), sentiment.float())
    classification = F.cross_entropy(output["classification_logits"].float(), class_7_index)
    return {
        "loss": regression + classification_weight * classification,
        "regression_loss": regression,
        "classification_loss": classification,
    }


class QFormerPool(nn.Module):
    """Compress a variable-length encoder sequence into learned query tokens."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        query_tokens: int = 4,
        layers: int = 2,
        heads: int = 8,
        ffn_size: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.queries = nn.Parameter(torch.empty(query_tokens, hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        memory = self.input_projection(hidden_states)
        queries = self.queries.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("attention_mask must match the first two hidden-state dimensions")
            if not torch.all(attention_mask.to(torch.bool).any(dim=1)):
                raise ValueError("every sample must contain at least one valid encoder token")
            padding_mask = ~attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        pooled = self.decoder(queries, memory, memory_key_padding_mask=padding_mask)
        return self.output_norm(pooled)


class SubsetFusion(nn.Module):
    """Build present/missing token sequences and predict all requested subsets."""

    def __init__(
        self,
        hidden_size: int = 512,
        tokens_per_modality: int = 4,
        layers: int = 4,
        heads: int = 8,
        ffn_size: int = 2048,
        dropout: float = 0.1,
        classes: int = 7,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tokens_per_modality = tokens_per_modality
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.missing_tokens = nn.ParameterDict(
            {modality: nn.Parameter(torch.empty(1, tokens_per_modality, hidden_size)) for modality in MODALITIES}
        )
        self.modality_embedding = nn.Embedding(4, hidden_size)
        self.presence_embedding = nn.Embedding(2, hidden_size)
        self.position_embedding = nn.Parameter(torch.empty(1, 1 + 3 * tokens_per_modality, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, classes)
        self.regressor = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        for token in self.missing_tokens.values():
            nn.init.normal_(token, mean=0.0, std=0.02)

    def forward(self, encoded: Mapping[str, torch.Tensor], subset: str) -> dict[str, torch.Tensor]:
        if subset not in SUBSETS:
            raise ValueError(f"unknown subset {subset!r}; expected one of {SUBSETS}")
        if set(encoded) != set(MODALITIES):
            raise ValueError("encoded must contain exactly t, a and v")
        batch_size = encoded["t"].shape[0]
        expected = (batch_size, self.tokens_per_modality, self.hidden_size)
        for modality in MODALITIES:
            if encoded[modality].shape != expected:
                raise ValueError(f"encoded[{modality!r}] must have shape {expected}")

        cls = self.cls_token.expand(batch_size, -1, -1)
        pieces = [cls + self.modality_embedding.weight[0].view(1, 1, -1)]
        for modality_index, modality in enumerate(MODALITIES, start=1):
            present = modality in subset
            tokens = encoded[modality] if present else self.missing_tokens[modality].expand(batch_size, -1, -1)
            tokens = tokens + self.modality_embedding.weight[modality_index].view(1, 1, -1)
            tokens = tokens + self.presence_embedding.weight[int(present)].view(1, 1, -1)
            pieces.append(tokens)
        sequence = torch.cat(pieces, dim=1) + self.position_embedding
        fused = self.final_norm(self.encoder(sequence)[:, 0])
        return {
            "regression": 3.0 * torch.tanh(self.regressor(fused).squeeze(-1)),
            "classification_logits": self.classifier(fused),
            "fused": fused,
        }

    def forward_subsets(
        self,
        encoded: Mapping[str, torch.Tensor],
        subsets: Iterable[str] = SUBSETS,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return {subset: self(encoded, subset) for subset in subsets}


class CachedStudentCore(nn.Module):
    """Trainable student modules operating on cached frozen-encoder sequences."""

    def __init__(
        self,
        text_hidden_size: int = 1024,
        audio_hidden_size: int = 768,
        video_hidden_size: int = 768,
        hidden_size: int = 512,
        tokens_per_modality: int = 4,
        qformer_layers: int = 2,
        fusion_layers: int = 4,
        heads: int = 8,
        ffn_size: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pools = nn.ModuleDict(
            {
                "t": QFormerPool(
                    text_hidden_size, hidden_size, tokens_per_modality,
                    qformer_layers, heads, ffn_size, dropout,
                ),
                "a": QFormerPool(
                    audio_hidden_size, hidden_size, tokens_per_modality,
                    qformer_layers, heads, ffn_size, dropout,
                ),
                "v": QFormerPool(
                    video_hidden_size, hidden_size, tokens_per_modality,
                    qformer_layers, heads, ffn_size, dropout,
                ),
            }
        )
        self.fusion = SubsetFusion(
            hidden_size=hidden_size,
            tokens_per_modality=tokens_per_modality,
            layers=fusion_layers,
            heads=heads,
            ffn_size=ffn_size,
            dropout=dropout,
        )

    def forward(
        self,
        encoder_hidden_states: Mapping[str, torch.Tensor],
        encoder_attention_masks: Mapping[str, torch.Tensor],
        subsets: Iterable[str] = SUBSETS,
    ) -> dict[str, dict[str, torch.Tensor]]:
        encoded = {
            modality: self.pools[modality](
                encoder_hidden_states[modality], encoder_attention_masks[modality]
            )
            for modality in MODALITIES
        }
        return self.fusion.forward_subsets(encoded, subsets)


class RDIDStudent(nn.Module):
    """Three frozen-or-trainable encoders, Q-Former pools, and shared subset fusion."""

    def __init__(
        self,
        text_encoder: nn.Module,
        audio_encoder: nn.Module,
        video_encoder: nn.Module,
        text_hidden_size: int,
        audio_hidden_size: int,
        video_hidden_size: int,
        hidden_size: int = 512,
        tokens_per_modality: int = 4,
        freeze_encoders: bool = True,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.freeze_encoders = freeze_encoders
        self.pools = nn.ModuleDict(
            {
                "t": QFormerPool(text_hidden_size, hidden_size, tokens_per_modality),
                "a": QFormerPool(audio_hidden_size, hidden_size, tokens_per_modality),
                "v": QFormerPool(video_hidden_size, hidden_size, tokens_per_modality),
            }
        )
        self.fusion = SubsetFusion(hidden_size=hidden_size, tokens_per_modality=tokens_per_modality)
        if freeze_encoders:
            for encoder in (self.text_encoder, self.audio_encoder, self.video_encoder):
                encoder.requires_grad_(False)

    @classmethod
    def from_local_pretrained(
        cls,
        text_path: str | Path,
        audio_path: str | Path,
        video_path: str | Path,
        *,
        dtype: torch.dtype = torch.bfloat16,
        freeze_encoders: bool = True,
    ) -> "RDIDStudent":
        from transformers import AutoModel

        paths = [Path(text_path), Path(audio_path), Path(video_path)]
        for path in paths:
            if not path.is_dir():
                raise FileNotFoundError(path)
        encoders = [
            AutoModel.from_pretrained(path, local_files_only=True, dtype=dtype)
            for path in paths
        ]
        sizes = [int(encoder.config.hidden_size) for encoder in encoders]
        return cls(*encoders, *sizes, freeze_encoders=freeze_encoders)

    def train(self, mode: bool = True) -> "RDIDStudent":
        super().train(mode)
        if self.freeze_encoders:
            self.text_encoder.eval()
            self.audio_encoder.eval()
            self.video_encoder.eval()
        return self

    def encode_modalities(
        self,
        *,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        input_values: torch.Tensor,
        audio_attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text = self.text_encoder(input_ids=input_ids, attention_mask=text_attention_mask, return_dict=True)
        audio = self.audio_encoder(
            input_values=input_values,
            attention_mask=audio_attention_mask,
            return_dict=True,
        )
        video = self.video_encoder(pixel_values=pixel_values, return_dict=True)

        audio_mask = None
        if audio_attention_mask is not None:
            audio_mask = self.audio_encoder._get_feature_vector_attention_mask(
                audio.last_hidden_state.shape[1], audio_attention_mask
            )
        return {
            "t": self.pools["t"](text.last_hidden_state, text_attention_mask),
            "a": self.pools["a"](audio.last_hidden_state, audio_mask),
            "v": self.pools["v"](video.last_hidden_state),
        }

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        input_values: torch.Tensor,
        audio_attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor,
        subsets: Iterable[str] = SUBSETS,
    ) -> dict[str, dict[str, torch.Tensor]]:
        encoded = self.encode_modalities(
            input_ids=input_ids,
            text_attention_mask=text_attention_mask,
            input_values=input_values,
            audio_attention_mask=audio_attention_mask,
            pixel_values=pixel_values,
        )
        return self.fusion.forward_subsets(encoded, subsets)
