#!/usr/bin/env python3
"""Resume a retained-checkpoint run after a verified machine/path relocation.

This wrapper keeps the original run protocol embedded in every checkpoint.  It
only permits differences caused by the known Qiii-to-local path relocation and
by path-serialized JSON hashes; all training-code and target-data hashes must
still match.  Training itself is delegated to the unchanged retained-epoch
trainer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/scripts"))

import train_main_table_all_epochs as retained


REMOTE_ROOTS = ("/ai/sjq/kd_multiseed_20260924", "/ai/sjq/kd")
PATH_SERIALIZED_JSON = {
    "protocol.json",
    "official_train_valid_windowed.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def canonical_path(value: str) -> str:
    for prefix in REMOTE_ROOTS:
        if value == prefix or value.startswith(prefix + "/"):
            return "<ROOT>" + value[len(prefix):]
    local = str(ROOT)
    if value == local or value.startswith(local + "/"):
        return "<ROOT>" + value[len(local):]
    return value


def canonical_protocol(value, *, hash_map: bool = False):
    if isinstance(value, str):
        return canonical_path(value)
    if isinstance(value, list):
        return [canonical_protocol(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {}
    for key, item in value.items():
        normalized_key = canonical_path(key)
        if hash_map and Path(normalized_key).name in PATH_SERIALIZED_JSON:
            # These JSON/JSONL files contain absolute paths and were serialized
            # with different whitespace on Qiii and locally.  Their semantic
            # contents are checked independently before launch.
            normalized[normalized_key] = "<SEMANTIC_JSON_EQUIVALENCE>"
        else:
            normalized[normalized_key] = canonical_protocol(
                item, hash_map=(key == "input_sha256")
            )
    return normalized


def semantic_manifest_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = canonical_protocol(json.loads(line))
        digest.update(json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode())
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def validate_checkpoint_boundary(output: Path) -> dict:
    history = json.loads((output / "history.json").read_text())
    epochs = [int(row["epoch"]) for row in history]
    checkpoint_epochs = [
        int(path.stem.rsplit("_", 1)[1])
        for path in sorted((output / "checkpoints").glob("epoch_*.pt"))
    ]
    if not epochs or epochs != checkpoint_epochs:
        raise ValueError(
            f"history/checkpoint boundary differs: history={epochs}, files={checkpoint_epochs}"
        )
    saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    if int(saved["epoch"]) != epochs[-1] or saved["history"] != history:
        raise ValueError("last.pt does not match the durable history boundary")
    return {
        "completed_epochs": epochs,
        "resume_epoch": epochs[-1] + 1,
        "last_checkpoint_bytes": (output / "last.pt").stat().st_size,
        "last_checkpoint_sha256": sha256(output / "last.pt"),
        "best_checkpoint_sha256": sha256(output / "best.pt"),
    }


def main() -> None:
    output = retained.output_arg()
    original_protocol = json.loads((output / "run_config.json").read_text())
    boundary = validate_checkpoint_boundary(output)

    def relocated_protocol(args, assets):
        local_protocol = retained.protocol_with_epoch_retention(args, assets)
        if canonical_protocol(original_protocol) != canonical_protocol(local_protocol):
            raise ValueError(
                "Qiii protocol is not equivalent to the local protocol after "
                "the allowed path/serialization normalization"
            )
        manifest = Path(local_protocol["manifest"])
        row_count, semantic_hash = semantic_manifest_sha256(manifest)
        expected_hash = "1dd082e4137228764c61c2b6826470e805eb3b8f5a1de459d935a54957885c2b"
        if row_count != 18278 or semantic_hash != expected_hash:
            raise ValueError(
                f"local manifest semantic identity differs: rows={row_count}, sha256={semantic_hash}"
            )
        atomic_json({
            "schema": "rdid-msa-verified-run-relocation-v1",
            "source_machine": "Qiii",
            "destination_machine": "inspur-NF5280M6",
            "source_roots": list(REMOTE_ROOTS),
            "destination_root": str(ROOT),
            "protocol_policy": "retain original protocol after canonical equivalence check",
            "training_state_policy": "model, optimizer, RNG, history, and epoch checkpoints unchanged",
            "manifest_rows": row_count,
            "manifest_semantic_sha256": semantic_hash,
            "original_protocol_sha256": hashlib.sha256(
                json.dumps(original_protocol, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "local_equivalent_protocol_sha256": hashlib.sha256(
                json.dumps(local_protocol, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "relocation_wrapper": str(Path(__file__).resolve()),
            "relocation_wrapper_sha256": sha256(Path(__file__).resolve()),
            "verified_at_unix": time.time(),
            **boundary,
        }, output / "relocation_manifest.json")
        return original_protocol

    retained.trainer.run_protocol = relocated_protocol
    retained.trainer.atomic_save = retained.atomic_save_with_epoch_retention
    retained.trainer.main()
    retained.audit(output)


if __name__ == "__main__":
    main()
