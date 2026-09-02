#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

MODEL_DIR = Path("/home/wy/sjq/kd/model/Qwen3-Omni-30B-A3B-Instruct")


def main() -> int:
    expected = [MODEL_DIR / f"model-{index:05d}-of-00015.safetensors" for index in range(1, 16)]
    complete = [path for path in expected if path.is_file() and path.stat().st_size > 0]
    cache_dir = MODEL_DIR / ".cache/huggingface/download"
    cache_partials = list(cache_dir.glob("*.incomplete")) if cache_dir.exists() else []
    partials = cache_partials + list(MODEL_DIR.glob("model-*.safetensors.part"))
    report = {
        "model_dir": str(MODEL_DIR),
        "complete_shards": len(complete),
        "expected_shards": len(expected),
        "complete_bytes": sum(path.stat().st_size for path in complete),
        "partial_files": len(partials),
        "partial_bytes": sum(path.stat().st_size for path in partials),
        "missing_shards": [path.name for path in expected if path not in complete],
        "ready": len(complete) == len(expected),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
