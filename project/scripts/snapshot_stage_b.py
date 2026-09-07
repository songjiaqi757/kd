#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/wy/sjq/kd")
OUTPUT = ROOT / "project/reports/stage_b_v1_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    paths = [
        ROOT / "dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl",
        ROOT / "outputs/student/features/official_train_valid/index.jsonl",
        ROOT / "outputs/student/features/official_train_valid/run_config.json",
        ROOT / "outputs/probe/official_train_valid_interaction_reliability.jsonl",
        ROOT / "project/reports/stage_b_baselines_three_seed.json",
        ROOT / "project/reports/fullscale_subset4_vs_pair_snr_three_seed.json",
    ]
    for probe_seed in (2026, 2027, 2028):
        paths += [
            ROOT / f"outputs/probe/official_train_valid_seed{probe_seed}/report.json",
            ROOT / f"outputs/probe/official_train_valid_seed{probe_seed}/predictions.jsonl",
        ]
    for method in ("student", "full_kd", "subset4", "pair_snr"):
        for seed in (13, 42, 2026):
            directory = ROOT / f"outputs/student/fullscale_{method}_seed{seed}"
            paths += [directory / "report.json", directory / "predictions.jsonl"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"freeze inputs missing: {missing}")
    commit = subprocess.check_output(
        ["/home/wy/sjq/miniconda3/bin/git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = {
        "name": "rdid-stage-b-v1",
        "scope": "official train/valid only",
        "official_test_used": False,
        "git_commit_before_freeze_commit": commit,
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
