import json
from collections import Counter
from pathlib import Path

MANIFEST = Path("/home/wy/sjq/kd/dataset/cmu_mosei/manifests/all.jsonl")


def test_full_manifest_preserves_official_rows_and_splits() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 22856
    assert len({row["sample_id"] for row in rows}) == 22856
    assert Counter(row["split"] for row in rows) == {
        "train": 16326,
        "valid": 1871,
        "test": 4659,
    }
    assert all(row["status"] == "valid" for row in rows)


def test_full_manifest_has_no_video_split_leakage() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    split_by_video: dict[str, str] = {}
    for row in rows:
        previous = split_by_video.setdefault(row["video_id"], row["split"])
        assert previous == row["split"]
    assert len(split_by_video) == 3225


def test_full_manifest_padding_is_explicit_and_bounded() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        assert 0 <= row["video_leading_padding_seconds"] <= 0.55
        assert 0 <= row["audio_leading_padding_seconds"] <= 0.55
        assert 0 <= row["video_tail_padding_seconds"] <= 0.55
        assert 0 <= row["audio_tail_padding_seconds"] <= 0.55
        assert Path(row["full_video_path"]).is_file()
        assert Path(row["full_audio_path"]).is_file()
