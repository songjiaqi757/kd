#!/usr/bin/env python3
"""Atomically rewrite one absolute path prefix in JSON artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def rewrite(value, old: str, new: str):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [rewrite(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item, old, new) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()
    changed = 0
    for root in args.root:
        for path in sorted(root.rglob("*.json")):
            payload = json.loads(path.read_text())
            updated = rewrite(payload, args.old, args.new)
            if updated == payload:
                continue
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n")
            os.replace(temporary, path)
            changed += 1
    print(json.dumps({"status": "complete", "files_changed": changed}))


if __name__ == "__main__":
    main()
