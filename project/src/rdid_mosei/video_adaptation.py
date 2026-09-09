"""Controlled VideoMAE tail adaptation; legacy Stage-D models are unchanged."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import MethodType

import av
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .student import LoRALinear, LoRATextAudioCachedVideoStudent, inject_lora

MODES = ("frozen_video", "video_lora", "ta_only")


def _video_lora_attention(self, hidden_states):
    """VideoMAE's stock F.linear(weight=...) bypasses LoRALinear.forward."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.videomae.modeling_videomae import eager_attention_forward
    batch = hidden_states.shape[0]
    def project(module, bias):
        # Keep pretrained bias inside F.linear, including its bf16 rounding.
        base = nn.functional.linear(hidden_states, module.base.weight, bias)
        residual = module.lora_B(module.lora_A(module.dropout(hidden_states))) * module.scaling
        return base + residual
    k_bias = torch.zeros_like(self.v_bias) if self.q_bias is not None else None
    q = project(self.query, self.q_bias)
    k = project(self.key, k_bias)
    v = project(self.value, self.v_bias)
    def heads(x):
        return x.view(batch, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
    context, weights = interface(self, heads(q), heads(k), heads(v), None,
                                 is_causal=self.is_causal, scaling=self.scaling,
                                 dropout=self.dropout_prob if self.training else 0.0)
    return context.reshape(*context.shape[:-2], self.all_head_size), weights


def sample_video_frames(path: str | Path, frames: int = 16) -> list[np.ndarray]:
    """Same endpoint-inclusive indices as the legacy sampler, bounded RGB memory."""
    if frames <= 0:
        raise ValueError("frames must be positive")
    with av.open(str(path)) as container:
        count = int(container.streams.video[0].frames)
    if count <= 0:
        with av.open(str(path)) as container:
            count = sum(1 for _ in container.decode(video=0))
    if count <= 0:
        raise ValueError(f"empty video: {path}")
    indices = np.linspace(0, count - 1, frames).round().astype(np.int64)
    wanted = set(indices.tolist())
    selected = {}
    decoded_count = 0
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            decoded_count += 1
            if index in wanted:
                selected[index] = frame.to_ndarray(format="rgb24")
    if decoded_count != count:
        raise ValueError(f"video frame-count mismatch: {path}: {count} != {decoded_count}")
    return [selected[int(index)] for index in indices]


class VideoAdaptationStudent(LoRATextAudioCachedVideoStudent):
    def __init__(
        self, text_encoder: nn.Module, audio_encoder: nn.Module,
        video_encoder: nn.Module | None, *, mode: str = "video_lora",
        video_layers: int = 2, video_rank: int = 8, video_alpha: float = 16,
        video_dropout: float = 0.05, video_seed: int = 2026,
        video_checkpointing: bool = True, hidden_size: int = 512,
        video_hidden_size: int = 768,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        super().__init__(text_encoder, audio_encoder, hidden_size=hidden_size,
                         video_hidden_size=video_hidden_size)
        self.mode = mode
        self.video_encoder = video_encoder
        self.video_lora_modules: list[str] = []
        self.video_checkpointing = video_checkpointing
        self.video_layers = video_layers
        if mode == "ta_only":
            if video_encoder is not None:
                raise ValueError("TA-only must not load a video encoder")
            self.pools["v"].requires_grad_(False)
            self.video_start_layer = 0
        else:
            if video_encoder is None:
                raise ValueError("video encoder required")
            video_encoder.requires_grad_(False)
            if not 1 <= video_layers <= len(video_encoder.encoder.layer):
                raise ValueError("invalid number of video tail layers")
            self.video_start_layer = len(video_encoder.encoder.layer) - video_layers
            if mode == "video_lora":
                # Keep common T/A, pooling and fusion initialization/RNG identical to A.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(video_seed)
                    for index in range(self.video_start_layer, len(video_encoder.encoder.layer)):
                        names = inject_lora(
                            video_encoder.encoder.layer[index],
                            ("attention.attention.query", "attention.attention.key",
                             "attention.attention.value", "attention.output.dense"),
                            rank=video_rank, alpha=video_alpha, dropout=video_dropout,
                        )
                        self.video_lora_modules.extend(f"encoder.layer.{index}.{n}" for n in names)
                        attention = video_encoder.encoder.layer[index].attention.attention
                        attention.forward = MethodType(_video_lora_attention, attention)
            video_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.video_encoder is not None:
            self.video_encoder.eval()
            for module in self.video_encoder.modules():
                if isinstance(module, LoRALinear):
                    module.train(mode)
                    module.base.eval()
        return self

    def encode_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.video_encoder is None:
            raise ValueError("TA-only has no video encoder")
        encoder = self.video_encoder
        # The frozen prefix has no trainable parameters or upstream gradients.
        with torch.no_grad():
            hidden = encoder.embeddings(pixel_values, bool_masked_pos=None)
            for layer in encoder.encoder.layer[:self.video_start_layer]:
                hidden = layer(hidden)
        train_video = self.mode == "video_lora" and torch.is_grad_enabled()
        with nullcontext() if train_video else torch.no_grad():
            for layer in encoder.encoder.layer[self.video_start_layer:]:
                if train_video and self.training and self.video_checkpointing:
                    # Explicit checkpoint: works while frozen backbone dropout stays off.
                    hidden = checkpoint(layer, hidden, use_reentrant=False, preserve_rng_state=True)
                else:
                    hidden = layer(hidden)
            if encoder.layernorm is not None:
                hidden = encoder.layernorm(hidden)
        return hidden

    def forward(self, *, input_ids, text_attention_mask, input_values,
                audio_attention_mask, pixel_values=None):
        text = self.text_encoder(input_ids=input_ids, attention_mask=text_attention_mask,
                                 return_dict=True).last_hidden_state
        audio = self.audio_encoder(input_values=input_values, attention_mask=audio_attention_mask,
                                   return_dict=True).last_hidden_state
        audio_mask = None if audio_attention_mask is None else self.audio_encoder._get_feature_vector_attention_mask(
            audio.shape[1], audio_attention_mask)
        encoded = {"t": self.pools["t"](text, text_attention_mask),
                   "a": self.pools["a"](audio, audio_mask)}
        if self.mode == "ta_only":
            if pixel_values is not None:
                raise ValueError("TA-only must not receive video pixels")
            encoded["v"] = torch.zeros_like(encoded["t"])
            subset = "ta"
        else:
            if pixel_values is None:
                raise ValueError("video pixels required")
            video = self.encode_video(pixel_values)
            encoded["v"] = self.pools["v"](video, None)
            subset = "tav"
        return self.fusion(encoded, subset)
