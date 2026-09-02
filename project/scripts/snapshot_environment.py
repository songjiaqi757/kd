#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPORTS = Path("/home/wy/sjq/kd/environment")


def capture(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout
    if result.stderr:
        output += ("\n" if output else "") + result.stderr
    if result.returncode:
        output += f"\nexit_code={result.returncode}\n"
    return output.strip() + "\n"


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).with_name("ffmpeg"))
    conda = Path(sys.prefix).parents[1] / "bin/conda"
    metadata = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "environment_prefix": sys.prefix,
    }
    (REPORTS / "environment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (REPORTS / "pip_freeze.txt").write_text(
        capture([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8"
    )
    (REPORTS / "ffmpeg_version.txt").write_text(
        capture([ffmpeg, "-version"]), encoding="utf-8"
    )
    (REPORTS / "nvidia_smi.txt").write_text(capture(["nvidia-smi"]), encoding="utf-8")
    if conda.is_file():
        (REPORTS / "conda_explicit.txt").write_text(
            capture([str(conda), "list", "--explicit", "--prefix", sys.prefix]),
            encoding="utf-8",
        )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
