#!/usr/bin/env python3
"""Inventory and reversibly compress completed, nonselected epoch checkpoints.

No pickle loading, lossy conversion, optimizer stripping or test-record deletion.
An original is unlinked only after full streaming decompression verifies SHA256.
Restore is explicit because torch.load cannot read the archive directly.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import zstandard as zstd

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 4 * 1024 * 1024


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(path.read_text())


def protected_checkpoints(run):
    """Fail closed if either validation policy cannot be reconstructed."""
    history = read_json(run / "history.json")
    if not history:
        raise ValueError("empty history")
    selected = [min(history, key=lambda r: (float(r["valid_metrics"]["mae"]), r["epoch"])),
                min(history, key=lambda r: (-float(r["valid_metrics"]["acc2_nonzero"]), r["epoch"]))]
    protected = {p.resolve() for p in run.glob("*.pt")}
    for row in selected:
        protected.add((run / "checkpoints" / f"epoch_{int(row['epoch']):03d}.pt").resolve())
    # Also preserve policy-specific checkpoints selected with the trainer's
    # 1e-5 improvement tolerance, even if exact argmin chooses another epoch.
    for metric, sign in (("mae", 1), ("acc2_nonzero", -1)):
        best = float("inf")
        epoch = None
        for row in sorted(history, key=lambda r: r["epoch"]):
            score = sign * float(row["valid_metrics"][metric])
            if score < best - 1e-5:
                best, epoch = score, int(row["epoch"])
        protected.add((run / "checkpoints" / f"epoch_{epoch:03d}.pt").resolve())
    return protected


def inventory(root):
    candidates, skipped = [], []
    # Protect every symlink target, including best aliases outside a run.
    linked = {p.resolve() for p in root.rglob("*.pt") if p.is_symlink()}
    for directory in sorted(root.rglob("checkpoints")):
        if not directory.is_dir() or directory.is_symlink():
            continue
        run = directory.parent
        try:
            if read_json(run / "status.json").get("status") != "complete":
                skipped.append({"run": str(run), "reason": "not complete"})
                continue
            protected = protected_checkpoints(run) | linked
        except (OSError, ValueError, KeyError, TypeError) as error:
            skipped.append({"run": str(run), "reason": str(error)})
            continue
        for path in sorted(directory.glob("epoch_*.pt")):
            if path.is_symlink() or path.resolve() in protected:
                continue
            stat = path.stat()
            if stat.st_nlink != 1:
                continue
            candidates.append({"path": str(path.absolute()), "run": str(run.absolute()),
                               "bytes": stat.st_size})
    candidates.sort(key=lambda r: (-r["bytes"], r["path"]))
    return {"schema": "reversible-checkpoint-archive-v1", "root": str(root.absolute()),
            "candidate_count": len(candidates), "candidate_bytes": sum(r["bytes"] for r in candidates),
            "note": "candidate bytes are original size, NOT expected disk savings",
            "protected": "all run-root checkpoints, both validation-best epochs, symlink targets, incomplete runs",
            "candidates": candidates, "skipped": skipped}


def digest_stream(stream, destination=None):
    digest, size = hashlib.sha256(), 0
    while block := stream.read(CHUNK):
        digest.update(block)
        size += len(block)
        if destination is not None:
            destination.write(block)
    return digest.hexdigest(), size


@contextmanager
def run_lock(run):
    with (run / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def compress_checkpoint(path, archive_root, root):
    path = path.absolute()
    relative = path.relative_to(root.absolute())
    run = path.parent.parent
    with run_lock(run):
        if read_json(run / "status.json").get("status") != "complete":
            raise ValueError("refusing active/incomplete run")
        if path.is_symlink() or path.resolve() in protected_checkpoints(run):
            raise ValueError("refusing protected checkpoint")
        before = path.stat()
        if before.st_nlink != 1:
            raise ValueError("refusing multiply linked checkpoint")
        archive = archive_root / relative.with_suffix(relative.suffix + ".zst")
        metadata = archive.with_suffix(archive.suffix + ".json")
        if archive.exists() or metadata.exists():
            raise FileExistsError(f"archive exists; inspect or restore first: {archive}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(archive.parent).free < before.st_size + 1024**3:
            raise OSError("insufficient space for temporary archive plus 1 GiB reserve")
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        try:
            with path.open("rb") as source, temporary.open("wb") as target:
                with zstd.ZstdCompressor(level=3, write_checksum=True).stream_writer(target, closefd=False) as compressor:
                    digest, size = digest_stream(source, compressor)
                target.flush()
                os.fsync(target.fileno())
            with temporary.open("rb") as stream, zstd.ZstdDecompressor().stream_reader(stream) as reader:
                check_digest, check_size = digest_stream(reader)
            if (check_digest, check_size) != (digest, size):
                raise ValueError("archive round-trip verification failed; original retained")
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("source changed while archiving; original retained")
            saved_bytes = size - temporary.stat().st_size
            if saved_bytes < 1024**2:
                temporary.unlink()
                return {"path": str(path), "status": "not_compressible", "saved_bytes": 0}
            os.replace(temporary, archive)
            record = {"schema": "lossless-checkpoint-archive-v1", "original": str(path),
                      "archive": str(archive.absolute()), "original_bytes": size,
                      "archive_bytes": archive.stat().st_size, "saved_bytes": saved_bytes,
                      "sha256": digest, "mode": before.st_mode & 0o777,
                      "mtime_ns": before.st_mtime_ns, "verified": True,
                      "codec": "zstd", "status": "verified_before_unlink"}
            atomic_json(record, metadata)
            pointer = path.with_suffix(path.suffix + ".archive.json")
            atomic_json(record, pointer)
            path.unlink()
            record["status"] = "archived"
            atomic_json(record, metadata)
            atomic_json(record, pointer)
            return record
        finally:
            if temporary.exists():
                temporary.unlink()


def restore(pointer):
    record = read_json(pointer)
    if record.get("status") == "permanently_deleted_after_evaluation":
        raise ValueError("archive was permanently deleted after evaluation; see deletion_plan in this record")
    if not record.get("verified") or record.get("codec") != "zstd":
        raise ValueError("unverified/unsupported archive")
    path, archive = Path(record["original"]), Path(record["archive"])
    with run_lock(path.parent.parent):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"will not overwrite existing checkpoint: {path}")
        if shutil.disk_usage(path.parent).free < record["original_bytes"] + 1024**3:
            raise OSError("restoration needs original size plus 1 GiB free")
        temporary = path.with_suffix(path.suffix + ".restore.tmp")
        try:
            with archive.open("rb") as source, zstd.ZstdDecompressor().stream_reader(source) as reader, temporary.open("wb") as destination:
                digest, size = digest_stream(reader, destination)
                destination.flush()
                os.fsync(destination.fileno())
            if (digest, size) != (record["sha256"], record["original_bytes"]):
                raise ValueError("restoration checksum mismatch")
            os.chmod(temporary, record["mode"])
            os.utime(temporary, ns=(record["mtime_ns"], record["mtime_ns"]))
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {"restored": str(path), "sha256": digest, "archive_retained": str(archive)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/experiments")
    parser.add_argument("--archive-root", type=Path, default=ROOT / "outputs/checkpoint_archive_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "project/reports/checkpoint_archive_inventory.json")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--target-free-gib", type=float, default=30)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--restore", type=Path, help="path to the .pt.archive.json pointer")
    args = parser.parse_args()
    if args.restore:
        print(json.dumps(restore(args.restore)), flush=True)
        return
    report = inventory(args.root)
    atomic_json(report, args.report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"candidates", "skipped"}}), flush=True)
    if not args.apply:
        return
    args.archive_root.mkdir(parents=True, exist_ok=True)
    with (args.archive_root / ".archive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved, count = 0, 0
        for item in report["candidates"]:
            if shutil.disk_usage(args.root).free >= args.target_free_gib * 1024**3:
                break
            if args.max_files and count >= args.max_files:
                break
            try:
                result = compress_checkpoint(Path(item["path"]), args.archive_root, args.root)
            except (OSError, ValueError) as error:
                result = {"path": item["path"], "status": "skipped", "error": str(error), "saved_bytes": 0}
            saved += result["saved_bytes"]
            count += 1
            with (args.archive_root / "operations.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps({"count": count, "saved_gib": saved/1024**3, **result}), flush=True)
            atomic_json({"status": "running", "processed": count, "saved_bytes": saved,
                         "disk_free_bytes": shutil.disk_usage(args.root).free}, args.archive_root / "status.json")
        atomic_json({"status": "complete", "processed": count, "saved_bytes": saved,
                     "disk_free_bytes": shutil.disk_usage(args.root).free,
                     "target_free_gib": args.target_free_gib}, args.archive_root / "status.json")


if __name__ == "__main__":
    main()
