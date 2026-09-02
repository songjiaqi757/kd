#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

DATASET_ROOT = Path("/home/wy/sjq/kd/dataset/cmu_mosei")


def main() -> int:
    manifests = sorted((DATASET_ROOT / "manifests").glob("*.jsonl"))
    for manifest in manifests:
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        checksum = manifest.with_suffix(manifest.suffix + ".sha256")
        checksum.write_text(f"{digest}  {manifest.name}\n", encoding="utf-8")
        print(f"{manifest.name} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
