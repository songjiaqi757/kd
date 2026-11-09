"""Explicit video timing for the local Qwen3-Omni single-clip pipeline."""
from __future__ import annotations

import math

VIDEO_TIMING_POLICIES = ("legacy_v1", "sampled_fps_v2")


def assert_timing_compatible(existing, requested):
    actual = existing.get("video_timing_policy", "legacy_v1")
    if actual != requested:
        raise ValueError(f"video timing policy differs: existing={actual}, requested={requested}; use a new output")


def sampled_video_kwargs(videos, metadata):
    if videos is None:
        return {}
    rates = metadata.get("fps")
    # Omni in Transformers 5.2.0 expects scalar FPS, not the utility's list.
    if len(videos) != 1 or not isinstance(rates, (list, tuple)) or len(rates) != 1:
        raise ValueError("this teacher pipeline requires exactly one sampled video and its FPS")
    fps = float(rates[0])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid sampled video FPS: {fps}")
    return {"fps": fps, "do_sample_frames": False}


def read_teacher_media(conversation, policy="sampled_fps_v2"):
    from qwen_omni_utils import process_mm_info

    if policy not in VIDEO_TIMING_POLICIES:
        raise ValueError(f"unknown video timing policy: {policy}")
    # Preserve legacy spatial preprocessing to isolate this temporal correction.
    if policy == "legacy_v1":
        a, i, v = process_mm_info(conversation, use_audio_in_video=False)
        return a, i, v, {}
    a, i, v, metadata = process_mm_info(
        conversation, use_audio_in_video=False, return_video_kwargs=True)
    return a, i, v, sampled_video_kwargs(v, metadata)


def video_preprocessing_audit(inputs, processor, video_kwargs):
    if "video_grid_thw" not in inputs:
        return {}
    seconds = inputs["video_second_per_grid"].detach().cpu().tolist()
    if "fps" in video_kwargs:
        expected = processor.video_processor.temporal_patch_size / video_kwargs["fps"]
        if not all(math.isclose(float(s), expected, rel_tol=1e-6, abs_tol=1e-6) for s in seconds):
            raise ValueError("processor video time spacing differs from actual sampled FPS")
    return {"sampled_fps": video_kwargs.get("fps"),
            "video_second_per_grid": seconds,
            "video_grid_thw": inputs["video_grid_thw"].detach().cpu().tolist()}
