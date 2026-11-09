#!/usr/bin/env python3
"""Read-only identity and schema audit for the frozen C-v2 inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUBSETS = {"t", "a", "v", "ta", "tv", "av", "tav"}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl")
    parser.add_argument("--student-cache", type=Path, default=ROOT / "outputs/student/features/official_train_valid")
    parser.add_argument("--teacher-cache", type=Path, default=ROOT / "outputs/probe/features/official_train_valid")
    parser.add_argument("--output", type=Path, default=ROOT / "project/reports/c_v2_input_audit.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "project/reports/c_v2_input_audit.md")
    return parser.parse_args()


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    a = args()
    manifest = jsonl(a.manifest)
    student = jsonl(a.student_cache / "index.jsonl")
    teacher = jsonl(a.teacher_cache / "index.jsonl")
    student_config = json.loads((a.student_cache / "run_config.json").read_text())
    teacher_config = json.loads((a.teacher_cache / "run_config.json").read_text())
    actual_hash = sha256(a.manifest)
    errors: list[str] = []
    warnings: list[str] = []

    if len(student) != len(manifest):
        errors.append(f"student rows {len(student)} != manifest rows {len(manifest)}")
    if len(teacher) != len(manifest) * 7:
        errors.append(f"teacher rows {len(teacher)} != 7 * manifest rows {len(manifest)}")
    if student_config.get("manifest_sha256") != actual_hash:
        errors.append("student cache manifest SHA256 mismatch")
    if teacher_config.get("manifest_sha256") != actual_hash:
        errors.append("teacher cache manifest SHA256 mismatch")

    comparable = min(len(student), len(manifest))
    identity_fields = ("sample_id", "parent_sample_id", "split", "class_7_index")
    for index in range(comparable):
        source, cached = manifest[index], student[index]
        if cached.get("index") != index:
            errors.append(f"student index discontinuity at row {index}")
            break
        for field in identity_fields:
            if cached.get(field) != source.get(field):
                errors.append(f"student {field} mismatch at row {index}")
                break
        if not math.isclose(float(cached["sentiment"]), float(source["sentiment"]), abs_tol=1e-9):
            errors.append(f"student sentiment mismatch at row {index}")
            break

    teacher_groups: dict[str, set[str]] = defaultdict(set)
    for row in teacher:
        teacher_groups[str(row["sample_id"])].add(str(row["subset"]))
    bad_groups = [sample_id for sample_id, subsets in teacher_groups.items() if subsets != SUBSETS]
    if bad_groups:
        errors.append(f"teacher subset coverage invalid for {len(bad_groups)} sample IDs")
    manifest_ids = {str(row["sample_id"]) for row in manifest}
    if set(teacher_groups) != manifest_ids:
        errors.append("teacher sample IDs do not exactly match manifest")

    split_by_video: dict[str, str] = {}
    for row in manifest:
        sentiment = float(row["sentiment"])
        if not math.isfinite(sentiment) or not -3.0 <= sentiment <= 3.0:
            errors.append(f"invalid sentiment for {row['sample_id']}")
            break
        video_id, split = str(row["video_id"]), str(row["split"])
        previous = split_by_video.setdefault(video_id, split)
        if previous != split:
            errors.append(f"video leakage across splits: {video_id}")
            break

    student_items = list((a.student_cache / "items").glob("*.pt"))
    if len(student_items) != len(student):
        errors.append(f"student feature files {len(student_items)} != index rows {len(student)}")
    teacher_features = np.load(a.teacher_cache / "features.npy", mmap_mode="r")
    teacher_completed = np.load(a.teacher_cache / "completed.npy", mmap_mode="r")
    if teacher_features.shape != (len(teacher), 2048):
        errors.append(f"unexpected teacher feature shape {teacher_features.shape}")
    if teacher_completed.shape != (len(teacher),) or not bool(np.all(teacher_completed)):
        errors.append("teacher completion bitmap is incomplete or malformed")

    recommended_student_fields = {
        "video_id", "window_index", "window_start", "window_end", "manifest_sha256", "feature_sha256"
    }
    recommended_teacher_fields = {"window_start", "window_end", "manifest_sha256", "feature_sha256"}
    missing_student = sorted(recommended_student_fields - set(student[0])) if student else sorted(recommended_student_fields)
    missing_teacher = sorted(recommended_teacher_fields - set(teacher[0])) if teacher else sorted(recommended_teacher_fields)
    if missing_student:
        warnings.append("student index lacks v2 identity/fingerprint fields: " + ", ".join(missing_student))
    if missing_teacher:
        warnings.append("teacher index lacks v2 identity/fingerprint fields: " + ", ".join(missing_teacher))
    if not all(isinstance(student_config.get("models", {}).get(name), str) for name in ("text", "audio", "video")):
        errors.append("student model paths missing")
    warnings.append("model paths are recorded, but immutable model revision/file hashes are not recorded in the legacy cache config")

    report = {
        "status": "pass_with_v2_gaps" if not errors and warnings else ("pass" if not errors else "fail"),
        "manifest": str(a.manifest),
        "manifest_sha256": actual_hash,
        "counts": {
            "manifest_windows": len(manifest),
            "manifest_utterances": len({str(row["parent_sample_id"]) for row in manifest}),
            "manifest_videos": len(split_by_video),
            "splits": dict(Counter(str(row["split"]) for row in manifest)),
            "student_index_rows": len(student),
            "student_feature_files": len(student_items),
            "teacher_index_rows": len(teacher),
        },
        "teacher_feature_shape": list(teacher_features.shape),
        "errors": errors,
        "warnings": warnings,
        "legacy_cache_safe_to_read": not errors,
        "v2_rebuild_or_index_upgrade_required": bool(missing_student or missing_teacher),
        "official_test_evaluated": False,
    }
    lines = [
        "# C-v2 输入与缓存审计",
        "",
        f"状态：**{report['status']}**",
        "",
        f"- Manifest windows：{len(manifest):,}",
        f"- Student cache：{len(student):,} index rows / {len(student_items):,} feature files",
        f"- Teacher cache：{len(teacher):,} rows，feature shape={list(teacher_features.shape)}",
        f"- Manifest SHA256：`{actual_hash}`",
        f"- Legacy cache 可安全读取：{not errors}",
        f"- C-v2 前需升级索引或重建：{bool(missing_student or missing_teacher)}",
        "",
        "## Errors",
        "",
        *(f"- {item}" for item in errors),
        *( ["- None"] if not errors else [] ),
        "",
        "## V2 gaps",
        "",
        *(f"- {item}" for item in warnings),
        "",
        "Official test 未读取、未评估。",
        "",
    ]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    a.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(a.output)}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
