from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

SCORE_PATTERN = re.compile(r"(?<![\d.])([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?![\d.])")


def record_key(sample_id: str, subset: str) -> tuple[str, str]:
    return sample_id, subset


def load_completed_keys(path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not path.is_file():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if item.get("status") == "ok":
            completed.add(record_key(str(item["sample_id"]), str(item["subset"])))
    return completed


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def select_jobs(
    rows: Iterable[dict[str, Any]], subsets: Iterable[str], completed: set[tuple[str, str]]
) -> list[tuple[dict[str, Any], str]]:
    jobs = []
    for row in rows:
        for subset in subsets:
            if record_key(str(row["sample_id"]), subset) not in completed:
                jobs.append((row, subset))
    return jobs


def parse_sentiment_score(text: str) -> float:
    values = [float(match.group(1)) for match in SCORE_PATTERN.finditer(text)]
    in_range = [value for value in values if -3.0 <= value <= 3.0]
    if not in_range:
        raise ValueError(f"no sentiment score in [-3, 3]: {text!r}")
    return in_range[0]


def weighted_mean(values: Iterable[tuple[float, float]]) -> float:
    pairs = list(values)
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        raise ValueError("total weight must be positive")
    return sum(value * weight for value, weight in pairs) / total_weight
