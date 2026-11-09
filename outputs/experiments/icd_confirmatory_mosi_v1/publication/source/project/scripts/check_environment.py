#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"ffmpeg={shutil.which('ffmpeg')}")
    print(f"ffprobe={shutil.which('ffprobe')}")

    modules = [
        "torch",
        "transformers",
        "accelerate",
        "qwen_omni_utils",
        "av",
        "cv2",
        "soundfile",
        "pandas",
    ]
    failed = False
    for name in modules:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
            print(f"{name}={version}")
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            failed = True
            print(f"{name}=ERROR:{type(exc).__name__}:{exc}")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip())
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        failed = True
        print(f"nvidia_smi=ERROR:{exc}")

    try:
        import torch

        print(f"torch_cuda_build={torch.version.cuda}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"cuda_device_count={torch.cuda.device_count()}")
        failed = failed or not torch.cuda.is_available() or torch.cuda.device_count() != 2
    except Exception:
        failed = True

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

