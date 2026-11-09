#!/usr/bin/env python3
"""Prepare portable MOSEI baseline inputs for /ai/sjq/kd without copying media."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE = Path("/ai/sjq/kd")
TRANSFER = ROOT / "transfer/qiii"
OVERLAY = TRANSFER / "overlay"
TRAIN_MANIFEST = Path("dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl")
TEST_MANIFEST = Path("dataset/cmu_mosei/manifests/official_test_windowed.jsonl")
BASE_PROTOCOL = Path("outputs/experiments/tav_main_v1/assets/protocol.json")
MAIN_PROTOCOL = Path("outputs/experiments/main_table_v1/mosei/assets/protocol.json")
MEDIA_KEYS = ("audio_segment_path", "silent_video_path")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_string(value: str) -> str:
    prefix = str(ROOT) + "/"
    return str(REMOTE) + "/" + value[len(prefix):] if value.startswith(prefix) else value


def relocate(value):
    if isinstance(value, str):
        return remote_string(value)
    if isinstance(value, list):
        return [relocate(item) for item in value]
    if isinstance(value, dict):
        return {remote_string(key): relocate(item) for key, item in value.items()}
    return value


def write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rewrite_manifest(relative: Path, media: set[Path]) -> tuple[int, str]:
    source, target = ROOT / relative, OVERLAY / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open() as input_stream, target.open("w") as output_stream:
        for line in input_stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in MEDIA_KEYS:
                path = Path(row[key])
                if not path.is_file():
                    raise FileNotFoundError(path)
                media.add(path.relative_to(ROOT))
            output_stream.write(json.dumps(relocate(row), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count, sha256(target)


def original_files(protocol: dict) -> set[Path]:
    result = set()
    for raw in protocol["input_sha256"]:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != protocol["input_sha256"][raw]:
            raise ValueError(f"source asset checksum changed: {path}")
        result.add(path.relative_to(ROOT))
    return result


def main() -> None:
    media: set[Path] = set()
    train_rows, train_hash = rewrite_manifest(TRAIN_MANIFEST, media)
    test_rows, test_hash = rewrite_manifest(TEST_MANIFEST, media)
    base_source = json.loads((ROOT / BASE_PROTOCOL).read_text())
    main_source = json.loads((ROOT / MAIN_PROTOCOL).read_text())
    inputs = original_files(base_source) | original_files(main_source)
    base = relocate(base_source)
    base["input_sha256"][str(REMOTE / TRAIN_MANIFEST)] = train_hash
    write_json(base, OVERLAY / BASE_PROTOCOL)
    base_hash = sha256(OVERLAY / BASE_PROTOCOL)
    main_protocol = relocate(main_source)
    main_protocol["input_sha256"][str(REMOTE / TRAIN_MANIFEST)] = train_hash
    main_protocol["input_sha256"][str(REMOTE / BASE_PROTOCOL)] = base_hash
    write_json(main_protocol, OVERLAY / MAIN_PROTOCOL)
    inputs.difference_update({TRAIN_MANIFEST, TEST_MANIFEST, BASE_PROTOCOL, MAIN_PROTOCOL})
    inputs.update(media)
    # Entire source tree and the three frozen backbones are small enough to
    # copy as directories; all other inputs are explicit files.
    directories = [Path("project"), Path("model/Qwen3-0.6B-Base"),
                   Path("model/WavLM-Base-Plus"), Path("model/VideoMAE-Base")]
    for directory in directories:
        if not (ROOT / directory).is_dir():
            raise FileNotFoundError(ROOT / directory)
    listed = [path for path in inputs if not any(path.is_relative_to(directory) for directory in directories)]
    files_from = TRANSFER / "files_from.txt"
    files_from.parent.mkdir(parents=True, exist_ok=True)
    files_from.write_text("".join(str(path) + "/\n" for path in directories)
                          + "".join(str(path) + "\n" for path in sorted(listed)))
    bytes_total = sum((ROOT / path).stat().st_size for path in listed)
    bytes_total += sum(path.stat().st_size for directory in directories for path in (ROOT / directory).rglob("*") if path.is_file())
    manifest = {
        "schema": "qiii-mosei-baselines-transfer-v1",
        "source_root": str(ROOT), "remote_root": str(REMOTE),
        "methods": ["projector", "ea_kd", "cmad_cafd"],
        "seed": 13, "max_epochs": 20, "early_stopping_patience": 7,
        "train_valid_windows": train_rows, "test_windows": test_rows,
        "media_files": len(media), "missing_media_files": 0,
        "listed_files": len(listed), "estimated_transfer_bytes": bytes_total,
        "overlays": {str(path.relative_to(OVERLAY)): sha256(path) for path in
                     (OVERLAY / TRAIN_MANIFEST, OVERLAY / TEST_MANIFEST,
                      OVERLAY / BASE_PROTOCOL, OVERLAY / MAIN_PROTOCOL)},
        "files_from": str(files_from),
    }
    write_json(manifest, TRANSFER / "transfer_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
