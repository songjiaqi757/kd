"""M0--M6 share the timing-v2 online student and differ only in adaptation/loss."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .interaction import mobius_transform
from .student import SUBSETS
from .video_adaptation import VideoAdaptationStudent

METHODS = tuple(f"M{i}" for i in range(7))


class TAVDistillationStudent(VideoAdaptationStudent):
    def __init__(self, *args, method="M3", **kwargs):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__(*args, mode="frozen_video" if method in METHODS[:3] else "video_lora", **kwargs)
        self.method = method
        if method in ("M0", "M1"):
            self.text_encoder.requires_grad_(False)
            self.audio_encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.method in ("M0", "M1"):
            self.text_encoder.eval()
            self.audio_encoder.eval()
        return self

    def forward(self, *, input_ids, text_attention_mask, input_values,
                audio_attention_mask, pixel_values, all_subsets=False):
        text = self.text_encoder(input_ids=input_ids, attention_mask=text_attention_mask,
                                 return_dict=True).last_hidden_state
        audio = self.audio_encoder(input_values=input_values, attention_mask=audio_attention_mask,
                                   return_dict=True).last_hidden_state
        audio_mask = None if audio_attention_mask is None else self.audio_encoder._get_feature_vector_attention_mask(
            audio.shape[1], audio_attention_mask)
        encoded = {"t": self.pools["t"](text, text_attention_mask),
                   "a": self.pools["a"](audio, audio_mask),
                   "v": self.pools["v"](self.encode_video(pixel_values), None)}
        # The full prediction is evaluated first in every method. Extra subsets
        # reuse encoder graphs; they do not re-encode the expensive backbones.
        full = self.fusion(encoded, "tav")
        if not all_subsets:
            return full
        outputs = {"tav": full}
        outputs.update({s: self.fusion(encoded, s) for s in SUBSETS if s != "tav"})
        return outputs


def interaction_weights(mean, variance, utility, method):
    if method == "M4":
        return torch.ones_like(mean)
    if method not in ("M5", "M6"):
        raise ValueError(method)
    reliability = mean.abs() / torch.sqrt(variance.clamp_min(0) + 1e-4)
    reliability = reliability / reliability.mean(-1, keepdim=True).clamp_min(1e-8)
    weights = reliability.clamp(0.25, 4.0)
    if method == "M6":
        weights = weights * utility
    return weights / weights.mean(-1, keepdim=True).clamp_min(1e-8)


def interaction_loss(outputs, mean, variance, utility, sample_weights, empty_baseline, method):
    values = torch.stack([outputs[s]["regression"].float() for s in SUBSETS], dim=-1)
    coordinates = mobius_transform(values, empty_baseline)
    weights = interaction_weights(mean, variance, utility, method).detach()
    errors = F.smooth_l1_loss(coordinates, mean, reduction="none")
    per_sample = (weights * errors).mean(-1)
    loss = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
    return loss, weights
