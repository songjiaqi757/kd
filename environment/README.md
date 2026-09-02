# Reproducibility Environment

实际 Conda 环境：`/home/wy/sjq/miniconda3/envs/kd`。

- `requirements.lock`：项目直接依赖版本；
- `conda_explicit.txt`：Conda 精确包 URL；
- `pip_freeze.txt`：pip 完整包列表；
- `environment_metadata.json`：Python、平台和环境前缀；
- `nvidia_smi.txt`：GPU 与驱动快照；
- `ffmpeg_version.txt`：媒体工具构建版本。

运行 `/home/wy/sjq/kd/project/scripts/snapshot_environment.py` 可刷新这些快照。
