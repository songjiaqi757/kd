from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str


def read_transcript(path: Path) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("___", 4)
        if len(parts) != 5:
            continue
        segments.append(
            TranscriptSegment(
                index=int(parts[1]),
                start=float(parts[2]),
                end=float(parts[3]),
                text=parts[4].strip(),
            )
        )
    return segments


def match_transcript_segment(
    segments: list[TranscriptSegment], start: float, end: float, tolerance: float = 1e-5
) -> TranscriptSegment:
    if not segments:
        raise ValueError("transcript contains no valid segments")
    match = min(segments, key=lambda item: abs(item.start - start) + abs(item.end - end))
    error = abs(match.start - start) + abs(match.end - end)
    if error > tolerance:
        raise ValueError(f"no transcript timestamp match: total error={error:.6f}s")
    return match


def round_sentiment_class(value: float) -> int:
    rounded = math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)
    return max(-3, min(3, rounded))


def sentiment_bucket(value: float) -> str:
    if value < -0.5:
        return "negative"
    if value > 0.5:
        return "positive"
    return "neutral"


def duration_bucket(duration: float) -> str:
    if duration < 5:
        return "lt5"
    if duration < 15:
        return "5to15"
    if duration <= 30:
        return "15to30"
    return "gt30"


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

