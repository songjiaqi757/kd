#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/wy/sjq/kd")
OUTPUT = ROOT / "project/reports/stage_c_v1_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    paths = [
        ROOT / "project/reports/stage_b_v1_freeze.json",
        ROOT / "project/reports/interaction_utility_diagnosis.json",
        ROOT / "project/reports/stage_c_seed13_screening.json",
        ROOT / "project/reports/stage_c_multiseed_results.json",
        ROOT / "project/scripts/train_student_baseline.py",
        ROOT / "project/scripts/infer_student_all_subsets.py",
        ROOT / "project/scripts/analyze_interaction_utility.py",
        ROOT / "project/scripts/analyze_stage_c_seed13_screening.py",
        ROOT / "project/scripts/summarize_stage_c_multiseed.py",
    ]
    for seed in (13, 42, 2026):
        full_kd = ROOT / f"outputs/student/fullscale_full_kd_seed{seed}"
        paths += [
            full_kd / "report.json",
            full_kd / "predictions.jsonl",
            full_kd / "all_subset_predictions.jsonl",
            full_kd / "all_subset_predictions.report.json",
        ]
        for method in (
            "ensemble_pair", "pair_snr", "reliability_utility_pair", "selective50_interaction4"
        ):
            directory = ROOT / f"outputs/student/fullscale_{method}_seed{seed}"
            paths += [directory / "report.json", directory / "predictions.jsonl"]
    utility = ROOT / "outputs/student/fullscale_utility_pair_seed13"
    paths += [utility / "report.json", utility / "predictions.jsonl"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"freeze inputs missing: {missing}")
    commit = subprocess.check_output(
        ["/home/wy/sjq/miniconda3/bin/git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = {
        "name": "rdid-stage-c-v1",
        "scope": "official train/valid only",
        "official_test_used": False,
        "git_commit_before_freeze_commit": commit,
        "decision": {
            "formal_multiseed_passed": ["reliability_utility_pair"],
            "recommended_effect_target_passed": [],
            "continue_conflict_gate": False,
        },
        "files": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths
        },
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "files": len(paths)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
