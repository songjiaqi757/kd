from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import soundfile as sf
import torch


def load_manifest_row(path: str | Path, index: int = 0) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(f"manifest {path} has no row {index}")


def uniformly_sample_video(path: str | Path, frames: int = 16) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not decoded:
        raise ValueError(f"video contains no decodable frames: {path}")
    indices = np.linspace(0, len(decoded) - 1, num=frames).round().astype(np.int64)
    return [decoded[int(index)] for index in indices]


def prepare_student_sample(
    row: dict[str, Any],
    tokenizer: Any,
    audio_processor: Any,
    video_processor: Any,
    *,
    max_text_tokens: int = 256,
    video_frames: int = 16,
) -> dict[str, torch.Tensor]:
    text = tokenizer(
        row["text"],
        return_tensors="pt",
        truncation=True,
        max_length=max_text_tokens,
    )
    waveform, sample_rate = sf.read(row["audio_segment_path"], dtype="float32", always_2d=False)
    if waveform.ndim != 1:
        raise ValueError("student audio must be mono")
    if sample_rate != 16_000:
        raise ValueError(f"student audio must be 16 kHz, got {sample_rate}")
    audio = audio_processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    images = uniformly_sample_video(row["silent_video_path"], frames=video_frames)
    video = video_processor(images, return_tensors="pt")
    return {
        "input_ids": text["input_ids"],
        "text_attention_mask": text["attention_mask"],
        "input_values": audio["input_values"],
        "audio_attention_mask": audio.get("attention_mask"),
        "pixel_values": video["pixel_values"],
        "sentiment": torch.tensor([float(row["sentiment"])], dtype=torch.float32),
        "class_7_index": torch.tensor([int(row["class_7_index"])], dtype=torch.long),
    }
