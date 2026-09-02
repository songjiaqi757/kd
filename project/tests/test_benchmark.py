import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.benchmark import (
    append_jsonl,
    load_completed_keys,
    parse_sentiment_score,
    select_jobs,
    weighted_mean,
)


def test_resume_only_skips_successful_records(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    append_jsonl(output, {"sample_id": "one", "subset": "t", "status": "ok"})
    append_jsonl(output, {"sample_id": "one", "subset": "a", "status": "error"})
    completed = load_completed_keys(output)
    jobs = select_jobs([{"sample_id": "one"}], ["t", "a", "v"], completed)
    assert [(row["sample_id"], subset) for row, subset in jobs] == [("one", "a"), ("one", "v")]


def test_append_jsonl_is_valid(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    append_jsonl(output, {"sample_id": "样本", "subset": "t", "status": "ok"})
    assert json.loads(output.read_text(encoding="utf-8"))["sample_id"] == "样本"


def test_parse_sentiment_score() -> None:
    assert parse_sentiment_score("Sentiment: -1.75") == -1.75
    assert parse_sentiment_score("2") == 2.0


def test_weighted_mean() -> None:
    assert weighted_mean([(1.0, 1.0), (3.0, 3.0)]) == 2.5
