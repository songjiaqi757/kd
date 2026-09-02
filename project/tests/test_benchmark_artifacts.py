import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/wy/sjq/kd/dataset/cmu_mosei")


def test_benchmark500_is_frozen_balanced_and_leak_free() -> None:
    manifest = DATASET_ROOT / "manifests/benchmark500.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 500
    assert len({row["sample_id"] for row in rows}) == 500
    assert len({row["video_id"] for row in rows}) == 500
    assert Counter(row["split"] for row in rows) == {"train": 450, "valid": 50}
    assert Counter(row["duration_bucket"] for row in rows) == {
        "lt5": 124,
        "5to15": 250,
        "15to30": 110,
        "gt30": 16,
    }
    assert Counter(row["sentiment_bucket"] for row in rows) == {
        "negative": 167,
        "neutral": 167,
        "positive": 166,
    }


def test_benchmark500_prepared_media_paths_are_unique() -> None:
    manifest = DATASET_ROOT / "manifests/benchmark500_prepared.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    videos = [row["silent_video_path"] for row in rows]
    audios = [row["audio_segment_path"] for row in rows]
    assert len(set(videos)) == len(rows)
    assert len(set(audios)) == len(rows)
    assert all(Path(path).is_file() for path in videos + audios)


def test_windowed_manifest_preserves_500_parents_and_bounded_duration() -> None:
    manifest = DATASET_ROOT / "manifests/benchmark500_windowed.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 521
    assert len({row["sample_id"] for row in rows}) == 521
    assert len({row["parent_sample_id"] for row in rows}) == 500
    assert max(row["duration"] for row in rows) <= 30.0 + 1e-9

    parent_weights: dict[str, float] = {}
    parent_durations: dict[str, float] = {}
    for row in rows:
        parent_id = row["parent_sample_id"]
        parent_weights[parent_id] = parent_weights.get(parent_id, 0.0) + row["aggregation_weight"]
        parent_durations[parent_id] = row["utterance_duration"]
        assert Path(row["silent_video_path"]).is_file()
        assert Path(row["audio_segment_path"]).is_file()
    for parent_id, weight in parent_weights.items():
        assert abs(weight - parent_durations[parent_id]) < 1e-6
