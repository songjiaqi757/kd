#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and checksum the aligned CMU-MOSEI dataset")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/wy/sjq/kd/dataset/cmu_mosei"))
    parser.add_argument("--checksum-workers", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> tuple[Path, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return path, digest.hexdigest()


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    manifest = root / "manifests/all.jsonl"
    audit_report = root / "reports/full_media_audit.json"
    audit_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/audit_prepared_media.py"),
        "--manifest",
        str(manifest),
        "--output",
        str(audit_report),
    ]
    audit = subprocess.run(audit_command, text=True)
    report = json.loads(audit_report.read_text(encoding="utf-8"))

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    paths = sorted(
        [Path(row["silent_video_path"]) for row in rows]
        + [Path(row["audio_segment_path"]) for row in rows]
    )
    checksum_path = root / "reports/media_sha256.txt"
    temporary = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
    with ThreadPoolExecutor(max_workers=args.checksum_workers) as executor:
        checksums = list(executor.map(sha256_file, paths))
    with temporary.open("w", encoding="utf-8") as handle:
        for path, digest in checksums:
            handle.write(f"{digest}  {path.relative_to(root)}\n")
    temporary.replace(checksum_path)

    disk = shutil.disk_usage(root)
    summary = {
        "status": "complete" if audit.returncode == 0 else "audit_failed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "samples": len(rows),
        "video_files": len(rows),
        "audio_files": len(rows),
        "audit_passed": report["passed"],
        "audit_failed": report["failed"],
        "total_video_bytes": report["total_video_bytes"],
        "total_audio_bytes": report["total_audio_bytes"],
        "media_checksum_entries": len(checksums),
        "media_checksum_file": str(checksum_path),
        "disk_free_bytes_after": disk.free,
    }
    summary_path = root / "reports/final_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return audit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
