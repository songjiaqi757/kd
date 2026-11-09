#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a frozen student encoder feature cache")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    config = json.loads((args.features / "run_config.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (args.features / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = Path(config["manifest"])
    if not manifest.is_absolute():
        manifest = (Path.cwd() / manifest).resolve()
    manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    expected_dimensions = {key: int(value) for key, value in config["hidden_sizes"].items()}
    failures: list[dict[str, object]] = []
    sequence_lengths: dict[str, list[int]] = {key: [] for key in expected_dimensions}
    started = time.time()

    if config.get("status") != "complete":
        failures.append({"kind": "config_status", "value": config.get("status")})
    if sha256(manifest) != config["manifest_sha256"]:
        failures.append({"kind": "manifest_sha256"})
    if len(rows) != int(config["count"]) or len(rows) != len(manifest_rows):
        failures.append({"kind": "row_count", "index": len(rows), "manifest": len(manifest_rows)})
    if len({row["index"] for row in rows}) != len(rows):
        failures.append({"kind": "duplicate_index"})
    if len({row["sample_id"] for row in rows}) != len(rows):
        failures.append({"kind": "duplicate_sample_id"})

    for position, (row, manifest_row) in enumerate(zip(rows, manifest_rows), start=1):
        if row["sample_id"] != manifest_row["sample_id"] or row["split"] != manifest_row["split"]:
            failures.append({"kind": "index_manifest_mismatch", "position": position - 1})
            continue
        path = Path(row["feature_path"])
        if not path.is_file():
            failures.append({"kind": "missing_file", "position": position - 1, "path": str(path)})
            continue
        try:
            item = torch.load(path, map_location="cpu", weights_only=True)
            if set(item) != set(expected_dimensions):
                raise ValueError(f"keys={sorted(item)}")
            for key, expected_dimension in expected_dimensions.items():
                tensor = item[key]
                if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] != expected_dimension:
                    raise ValueError(f"{key}_shape={tuple(tensor.shape)}")
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(f"{key}_dtype={tensor.dtype}")
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"{key}_non_finite")
                sequence_lengths[key].append(int(tensor.shape[0]))
        except Exception as exc:  # noqa: BLE001 - audit records corrupt items and continues
            failures.append({"kind": "invalid_file", "position": position - 1, "path": str(path), "error": repr(exc)})
        if position % args.progress_every == 0 or position == len(rows):
            print(json.dumps({"audited": position, "total": len(rows), "failures": len(failures)}), flush=True)

    report = {
        "status": "pass" if not failures else "fail",
        "features": str(args.features.resolve()),
        "count": len(rows),
        "manifest": str(manifest),
        "manifest_sha256": config["manifest_sha256"],
        "split_windows": dict(Counter(row["split"] for row in rows)),
        "parent_samples": len({row["parent_sample_id"] for row in rows}),
        "parent_splits": dict(Counter({row["parent_sample_id"]: row["split"] for row in rows}.values())),
        "hidden_sizes": expected_dimensions,
        "dtype": config["dtype"],
        "sequence_lengths": {
            key: {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
            for key, values in sequence_lengths.items() if values
        },
        "failures": failures[:100],
        "failure_count": len(failures),
        "elapsed_seconds": time.time() - started,
    }
    output = args.output or args.features / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
