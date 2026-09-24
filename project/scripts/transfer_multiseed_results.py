#!/usr/bin/env python3
"""Pull a completed Qiii experiment tree and verify every transferred file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def ssh_prefix(args: argparse.Namespace) -> list[str]:
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-p", str(args.port), "-i", str(args.identity), args.remote,
    ]


def remote_command(args: argparse.Namespace, command: str, *, capture: bool = False):
    return subprocess.run(
        [*ssh_prefix(args), command], check=True, text=True,
        capture_output=capture,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="root@172.23.166.144")
    parser.add_argument("--port", type=int, default=34989)
    parser.add_argument("--identity", type=Path, default=Path("/home/wy/sjq/private_key_sjq.pem"))
    parser.add_argument(
        "--remote-dir", type=Path,
        default=Path("/ai/sjq/kd/outputs/experiments/multiseed_controls_v1/mosei"),
    )
    parser.add_argument(
        "--local-dir", type=Path,
        default=ROOT / "outputs/experiments/multiseed_controls_v1/mosei",
    )
    parser.add_argument("--minimum-free-gib", type=int, default=40)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.identity = args.identity.resolve()
    if not args.identity.is_file():
        raise FileNotFoundError(args.identity)
    status_result = remote_command(args, f"cat {args.remote_dir}/queue_status.json", capture=True)
    status = json.loads(status_result.stdout)
    if status.get("status") != "complete" and not args.allow_partial:
        raise RuntimeError(f"remote queue is not complete: {status.get('status')}")

    local = args.local_dir.resolve()
    local.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(local.parent)
    if usage.free < args.minimum_free_gib * 1024**3:
        raise OSError(
            f"insufficient destination space: {usage.free / 1024**3:.1f} GiB free; "
            f"need at least {args.minimum_free_gib} GiB"
        )
    remote_command(
        args,
        f"cd {args.remote_dir} && "
        "find . -type f ! -name transfer_manifest.sha256 -print0 | "
        "sort -z | xargs -0 sha256sum > transfer_manifest.sha256",
    )
    producer = subprocess.Popen(
        [*ssh_prefix(args), f"tar -C {args.remote_dir.parent} -cf - {args.remote_dir.name}"],
        stdout=subprocess.PIPE,
    )
    assert producer.stdout is not None
    consumer = subprocess.run(
        ["tar", "-C", str(local.parent), "-xf", "-"],
        stdin=producer.stdout,
        check=False,
    )
    producer.stdout.close()
    producer_code = producer.wait()
    if producer_code or consumer.returncode:
        raise subprocess.CalledProcessError(
            producer_code or consumer.returncode,
            "remote tar transfer",
        )
    subprocess.run(
        ["sha256sum", "-c", "transfer_manifest.sha256"],
        cwd=local,
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    print(json.dumps({
        "status": "complete",
        "remote_dir": str(args.remote_dir),
        "local_dir": str(local),
        "remote_queue_status": status.get("status"),
    }, indent=2))


if __name__ == "__main__":
    main()
