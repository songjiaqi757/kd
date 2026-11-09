#!/usr/bin/env python3
"""Assemble the text-only publication bundle for the confirmatory campaign.

Checkpoints, teacher tensors and per-run logs stay on the server. This script
copies the protocol, frozen source snapshot, summary tables and reports into
``publication/`` and writes a SHA256 manifest for review and GitHub sync.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

BASE = Path("/data/sjq/kd/outputs/experiments/icd_confirmatory_mosi_v1")
PUB = BASE / "publication"

REQUIRED = [
    "plan.md",
    "plan.json",
    "protocol_diff.md",
    "baseline_audit.json",
    "source_fingerprints.json",
    "completion_audit.json",
    "analysis/per_run.csv",
    "analysis/factorial_analysis.json",
    "analysis/interaction_analysis.json",
    "analysis/final_report.md",
    "analysis/paper_experiment_draft.md",
]
OPTIONAL = [
    "analysis/paired_source_video_analysis.json",
    "analysis/independent_metrics.json",
    "analysis/method_selection.json",
    "analysis/gradient_diagnostic_manifest.json",
    "selection.json",
    "budget_extension.json",
    "analysis/official_test_summary.json",
    "analysis/official_test_summary.csv",
    "analysis/official_test_summary.md",
]
SOURCE_DIRS = ["source", "assets"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy(relative: str, missing: list[str]) -> dict | None:
    src = BASE / relative
    if not src.exists():
        missing.append(relative)
        return None
    dst = PUB / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dict(path=str(dst), bytes=dst.stat().st_size, sha256=sha256(dst),
                origin=str(src))


def main() -> None:
    PUB.mkdir(parents=True, exist_ok=True)
    # Finalize the audit first: the bundle copy and the manifest must hash the
    # same bytes that stay on disk (no self-referential hash field).
    audit_path = BASE / "completion_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        audit.pop("publication_bundle_manifest_sha256", None)
        audit["publication_bundle"] = str(PUB)
        early_blockers = [name for name in REQUIRED if not (BASE / name).exists()]
        audit["github_sync_status"] = ("publication_bundle_ready"
                                       if not early_blockers else "publication_incomplete")
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    missing: list[str] = []
    entries = []
    for relative in REQUIRED + OPTIONAL:
        record = copy(relative, missing)
        if record is not None:
            entries.append(record)
    for directory in SOURCE_DIRS:
        src_dir = BASE / directory
        if not src_dir.exists():
            missing.append(directory + "/")
            continue
        for src in sorted(src_dir.rglob("*")):
            if not src.is_file() or src.suffix in {".pt", ".log", ".pyc"}:
                continue
            entries.append(copy(str(src.relative_to(BASE)), missing))

    blockers = [name for name in missing if name in REQUIRED]
    packager = Path(__file__).resolve()
    note = PUB / "README.md"
    note.write_text(
        "# Publication bundle\n\n"
        "Generated from `icd_confirmatory_mosi_v1` after the completion audit.\n"
        "Text assets only; checkpoints, teacher tensors and per-run logs stay on "
        "the server under the campaign root.\n\n"
        "Every file in this directory is hashed in `publication_manifest.json`.\n",
    )
    entries.append(dict(path=str(note), bytes=note.stat().st_size,
                        sha256=sha256(note), origin=str(note)))
    manifest = dict(
        schema="icd-confirmatory-publication-manifest-v1",
        bundle=str(PUB),
        packager=str(packager),
        packager_sha256=sha256(packager),
        required_missing=missing,
        blockers=blockers,
        checkpoint_bytes_policy="checkpoints and teacher assets remain on server",
        official_test_used_for_selection=False,
        official_test_reporting="completed after training (reporting only)",
        files=entries,
    )
    (PUB / "publication_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(dict(bundle=str(PUB), files=len(entries),
                          blockers=blockers), ensure_ascii=False))
    if blockers:
        raise SystemExit("Publication bundle incomplete: " + ", ".join(blockers))


if __name__ == "__main__":
    main()
