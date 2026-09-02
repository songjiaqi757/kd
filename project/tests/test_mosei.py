from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.mosei import duration_bucket, round_sentiment_class, sentiment_bucket
from rdid_mosei.subsets import SUBSETS, build_conversation


def test_round_sentiment_class() -> None:
    assert round_sentiment_class(-3.0) == -3
    assert round_sentiment_class(-0.6) == -1
    assert round_sentiment_class(0.0) == 0
    assert round_sentiment_class(0.6) == 1
    assert round_sentiment_class(3.0) == 3


def test_buckets() -> None:
    assert sentiment_bucket(-1.0) == "negative"
    assert sentiment_bucket(0.0) == "neutral"
    assert sentiment_bucket(1.0) == "positive"
    assert duration_bucket(4.9) == "lt5"
    assert duration_bucket(5.0) == "5to15"
    assert duration_bucket(15.0) == "15to30"
    assert duration_bucket(31.0) == "gt30"


def test_subset_conversations() -> None:
    sample = {
        "text": "example",
        "audio_segment_path": "/tmp/example.wav",
        "silent_video_path": "/tmp/example.mp4",
    }
    for subset in SUBSETS:
        conversation = build_conversation(sample, subset)
        user_content = conversation[1]["content"]
        types = [item["type"] for item in user_content]
        assert ("audio" in types) == ("a" in subset)
        assert ("video" in types) == ("v" in subset)
        prompt = user_content[-1]["text"]
        assert ("Transcript: example" in prompt) == ("t" in subset)


def test_affect_prompt_and_video_fps() -> None:
    sample = {
        "text": "example",
        "audio_segment_path": "/tmp/example.wav",
        "silent_video_path": "/tmp/example.mp4",
    }
    conversation = build_conversation(sample, "v", prompt_version="v2_affect", video_fps=4.0)
    user_content = conversation[1]["content"]
    assert user_content[0]["type"] == "video"
    assert user_content[0]["fps"] == 4.0
    assert "facial expression" in conversation[0]["content"][0]["text"]
