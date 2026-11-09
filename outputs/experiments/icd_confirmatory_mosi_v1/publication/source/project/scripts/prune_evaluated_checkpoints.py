#!/usr/bin/env python3
"""Permanently prune evaluated historical epochs with an explicit audit plan.

Scope: completed runs under outputs/experiments, nonselected epoch_*.pt and
their verified Zstandard archives. Preserve all root checkpoints, both exact
and tolerance-based validation selections, all checkpoint symlink targets,
and every evaluation/config/history/prediction record. No recursive deletion.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

from manage_checkpoint_archive import atomic_json, protected_checkpoints, run_lock

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "outputs/experiments"
ARCHIVES = ROOT / "outputs/checkpoint_archive_v1"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024**2):
            result.update(block)
    return result.hexdigest()


def evaluated_hashes(root):
    """Only completed test evaluations with metrics AND extant predictions."""
    index = defaultdict(list)
    files = set(root.rglob("report.json")) | set(root.rglob("epoch_*.json"))
    for path in sorted(files):
        if path.name.endswith(".archive.json"):
            continue
        try:
            report = read(path)
            metrics = report.get("test_metrics", report.get("metrics", {}))
            if not isinstance(metrics, dict) or int(metrics.get("count", 0)) <= 0:
                continue
            if not math.isfinite(float(metrics.get("mae", float("nan")))):
                continue
            # Reports from ordinary training use valid_metrics, not these keys.
            if "test_metrics" not in report and not report.get("official_test_evaluated"):
                continue
            if path.name == "report.json":
                status_path = path.parent / "status.json"
                if read(status_path).get("status") != "complete":
                    continue
                predictions = path.parent / "predictions.jsonl"
            else:
                if not re.fullmatch(r"epoch_\d+\.json", path.name):
                    continue
                status_path = None
                predictions = path.with_name(path.stem + "_predictions.jsonl")
            if not predictions.is_file() or predictions.stat().st_size == 0:
                continue
            config_path = path.parent / "run_config.json"
            config = read(config_path) if config_path.exists() else {}
            checkpoint = config.get("checkpoint", {})
            checksum = report.get("checkpoint_sha256") or config.get("checkpoint_sha256")
            if not checksum and isinstance(checkpoint, dict):
                checksum = checkpoint.get("sha256")
            if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                continue
            epoch = report.get("epoch", report.get("selected_epoch", config.get("epoch")))
            if epoch is None and isinstance(checkpoint, dict):
                epoch = checkpoint.get("epoch")
            evidence_files = [path, predictions]
            if status_path:
                evidence_files.append(status_path)
            if config_path.exists():
                evidence_files.append(config_path)
            index[checksum].append({
                "report": str(path), "epoch": epoch, "count": metrics["count"],
                "evidence": [{"path": str(p), "bytes": p.stat().st_size,
                              "mtime_ns": p.stat().st_mtime_ns} for p in evidence_files],
            })
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return index


def linked_targets(root):
    return {p.resolve() for p in root.rglob("*.pt") if p.is_symlink()}


def retention(run, linked):
    if read(run / "status.json").get("status") != "complete":
        raise ValueError("run is not complete")
    protected = protected_checkpoints(run) | linked
    # Both history-derived best epochs must actually survive as readable files.
    local_protected = [p for p in protected if p.parent == run / "checkpoints"]
    missing = [str(p) for p in local_protected if not p.is_file()]
    if missing:
        raise ValueError(f"selected/aliased checkpoints missing: {missing}")
    if not (run / "last.pt").is_file():
        raise ValueError("last.pt missing; preserve this run until separately reviewed")
    return protected


def plan():
    evidence = evaluated_hashes(EXPERIMENTS)
    linked = linked_targets(ROOT / "outputs")
    candidates, skipped, runs = [], [], []
    for directory in sorted(EXPERIMENTS.rglob("checkpoints")):
        if not directory.is_dir() or directory.is_symlink():
            continue
        run = directory.parent
        if not (run / "history.json").is_file():
            continue
        try:
            protected = retention(run, linked)
        except (OSError, ValueError, KeyError, TypeError) as error:
            skipped.append({"run": str(run), "reason": str(error)})
            continue
        run_protected = [p for p in protected if p.is_relative_to(run)]
        runs.append({"run": str(run), "preserved": [str(p) for p in sorted(run_protected)],
                     "history_sha256": digest(run / "history.json")})
        sources = set(directory.glob("epoch_*.pt"))
        sources.update(Path(str(p).removesuffix(".archive.json")) for p in directory.glob("epoch_*.pt.archive.json"))
        for source in sorted(sources):
            if not re.fullmatch(r"epoch_\d+\.pt", source.name):
                continue
            if source.is_symlink() or source.resolve() in protected:
                continue
            epoch = int(source.stem.split("_")[-1])
            pointer = source.with_suffix(source.suffix + ".archive.json")
            archive_record = read(pointer) if pointer.exists() else None
            if source.exists():
                checksum = digest(source)
            elif archive_record and archive_record.get("verified") and archive_record.get("status") == "archived":
                checksum = archive_record["sha256"]
            else:
                continue
            matches = [e for e in evidence.get(checksum, []) if e["epoch"] is None or int(e["epoch"]) == epoch]
            if not matches:
                skipped.append({"path": str(source), "reason": "no completed evaluation with matching checkpoint SHA256 and epoch"})
                continue
            files = []
            if source.exists():
                files.append(source)
            if archive_record:
                archive = Path(archive_record["archive"])
                if archive.is_file():
                    if archive_record["sha256"] != checksum or not archive_record.get("verified"):
                        raise ValueError(f"archive/source checksum mismatch: {source}")
                    if not archive.resolve().is_relative_to(ARCHIVES.resolve()) or archive.is_symlink():
                        raise ValueError(f"archive outside approved directory: {archive}")
                    if archive.stat().st_size != archive_record["archive_bytes"]:
                        raise ValueError(f"archive size mismatch: {archive}")
                    files.append(archive)
            if not files or any(p.stat().st_nlink != 1 for p in files):
                continue
            candidates.append({"original": str(source), "run": str(run), "epoch": epoch,
                               "checkpoint_sha256": checksum, "evaluation": matches[0],
                               "history_sha256": runs[-1]["history_sha256"],
                               "files": [{"path": str(p), "bytes": p.stat().st_size,
                                          "allocated_bytes": p.stat().st_blocks * 512,
                                          "inode": p.stat().st_ino,
                                          "mtime_ns": p.stat().st_mtime_ns} for p in files]})
        print(json.dumps({"scanned_run": str(run), "candidate_count": len(candidates)}), flush=True)
    return {"schema": "evaluated-checkpoint-prune-v1", "candidates": candidates,
            "candidate_count": len(candidates),
            "reclaimable_bytes": sum(f["allocated_bytes"] for r in candidates for f in r["files"]),
            "retained_runs": runs, "skipped": skipped,
            "policy": "completed and SHA256-matched evaluation only; preserve validation-best MAE/Acc2, last, existing root best and links; keep all evaluation data"}


def prune_record(row, linked, log, plan_path):
    source, run = Path(row["original"]), Path(row["run"])
    if not source.is_relative_to(EXPERIMENTS) or source.parent != run / "checkpoints":
        raise ValueError("source outside approved scope")
    with run_lock(run):
        if source.resolve() in retention(run, linked):
            raise ValueError("checkpoint is now protected")
        if digest(run / "history.json") != row["history_sha256"]:
            raise ValueError("training history changed since audit")
        for evidence in row["evaluation"]["evidence"]:
            stat = Path(evidence["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (evidence["bytes"], evidence["mtime_ns"]):
                raise ValueError("evaluation evidence changed since audit")
        for item in row["files"]:
            path = Path(item["path"])
            if path != source and not path.is_relative_to(ARCHIVES):
                raise ValueError("deletion outside approved scope")
            stat = path.stat()
            if path.is_symlink() or stat.st_nlink != 1:
                raise ValueError("file became a link")
            if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != (item["inode"], item["bytes"], item["mtime_ns"]):
                raise ValueError("checkpoint changed since audit")
        # Write-ahead record makes interruptions after unlink auditable.
        log.write(json.dumps({"action": "deletion_intent", **row}) + "\n")
        log.flush()
        os.fsync(log.fileno())
        for item in row["files"]:
            Path(item["path"]).unlink()
        pointer = source.with_suffix(source.suffix + ".archive.json")
        if pointer.exists():
            record = read(pointer)
            record.update(status="permanently_deleted_after_evaluation", archive_available=False,
                          deletion_plan=str(plan_path), evaluation_report=row["evaluation"]["report"])
            atomic_json(record, pointer)
            archive_meta = Path(record["archive"] + ".json")
            if archive_meta.is_file():
                atomic_json(record, archive_meta)
        tombstone = source.with_suffix(source.suffix + ".deleted.json")
        atomic_json({"status": "permanently_deleted_after_evaluation", "checkpoint_sha256": row["checkpoint_sha256"],
                     "evaluation_report": row["evaluation"]["report"], "deletion_plan": str(plan_path)}, tombstone)
        log.write(json.dumps({"action": "deleted", "original": str(source),
                              "freed_bytes": sum(f["allocated_bytes"] for f in row["files"])}) + "\n")
        log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    if not args.apply:
        result = plan()
        atomic_json(result, args.plan)
        print(json.dumps({k: result[k] for k in ("candidate_count", "reclaimable_bytes")}), flush=True)
        return
    result = read(args.plan)
    if result.get("schema") != "evaluated-checkpoint-prune-v1":
        raise ValueError("unsupported audit plan")
    if args.plan.with_suffix(".result.json").exists():
        raise FileExistsError("this plan already has an execution result")
    linked = linked_targets(ROOT / "outputs")
    before = shutil.disk_usage(EXPERIMENTS).free
    completed, errors = [], []
    with args.plan.with_suffix(".operations.jsonl").open("a") as log:
        for row in result["candidates"]:
            try:
                prune_record(row, linked, log, args.plan)
                completed.append(row)
            except (OSError, ValueError, KeyError) as error:
                errors.append({"original": row["original"], "error": str(error)})
    missing = [p for r in result["retained_runs"] for p in r["preserved"] if not Path(p).is_file()]
    output = {"status": "complete" if not missing and not errors else "needs_review",
              "deleted_checkpoints": len(completed),
              "deleted_raw_files": sum(f["path"].endswith(".pt") for r in completed for f in r["files"]),
              "deleted_archives": sum(f["path"].endswith(".zst") for r in completed for f in r["files"]),
              "freed_bytes": sum(f["allocated_bytes"] for r in completed for f in r["files"]),
              "disk_free_before": before, "disk_free_after": shutil.disk_usage(EXPERIMENTS).free,
              "preserved_files_checked": sum(len(r["preserved"]) for r in result["retained_runs"]),
              "missing_preserved_files": missing, "errors": errors, "evaluation_records_deleted": 0}
    atomic_json(output, args.plan.with_suffix(".result.json"))
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
