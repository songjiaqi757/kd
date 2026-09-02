# RDID-MOSEI

Qwen3-Omni 教师驱动的 CMU-MOSEI 多模态情感交互坐标蒸馏实验工程。

## 当前阶段

全量数据与教师模型已经准备完成。数据统一位于 `/home/wy/sjq/kd/dataset/cmu_mosei`，教师模型位于 `/home/wy/sjq/kd/model/Qwen3-Omni-30B-A3B-Instruct`。

## 环境

本项目使用名为 `kd` 的独立 Conda 环境，环境内包含 CUDA 13 PyTorch 和项目依赖：

```bash
conda activate kd
python scripts/check_environment.py
```

## 500 条基准数据

统一数据目录中的关键文件：

- `/home/wy/sjq/kd/dataset/cmu_mosei/manifests/all.jsonl`：全量 22,856 条清单；
- `/home/wy/sjq/kd/dataset/cmu_mosei/manifests/benchmark500.jsonl`：固定 500 条基准；
- `/home/wy/sjq/kd/dataset/cmu_mosei/manifests/benchmark500_prepared.jsonl`：复用全量规范媒体的 500 条清单；
- `/home/wy/sjq/kd/dataset/cmu_mosei/manifests/benchmark500_windowed.jsonl`：教师推理使用的 521 个单元；
- `reports/benchmark500_media_audit.json` 与 `reports/benchmark500_windowed_media_audit.json`：媒体审计报告。

官方 `label.csv` 中的 split 原样保留，不重新划分数据。500 条样本来自 500 个不同源视频，训练/验证为 450/50；测试集不参与教师基准。16 条超过 30 秒的样本按最长 30 秒、重叠 5 秒切窗，并用重叠校正权重聚合。

Conda 环境提供 ffmpeg/ffprobe 8.0；媒体脚本使用 ffmpeg 重编码无声视频、SoundFile 裁剪独立音频。
Qwen 媒体读取固定使用 decord；Torchvision 0.27 已移除 `read_video`，不能作为当前后端。

## 教师模型下载与冒烟推理

下载器按顺序下载 15 个权重分片，中断后可重复执行以续传：

```bash
bash scripts/download_model.sh
python scripts/check_model_files.py
```

代理频繁断流时使用基于 HTTP Range 的下载器；它始终续传同一个 `.part` 文件：

```bash
bash scripts/download_model_curl.sh
```

全部分片就绪后先校验，再依次运行一条、20 条和完整基准。脚本按每个成功任务立即写盘，断开后执行同一命令会跳过已成功的任务：

```bash
python scripts/check_model_files.py
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_teacher_benchmark.py --limit 1
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_teacher_benchmark.py --limit 20
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_teacher_benchmark.py
python scripts/aggregate_teacher_outputs.py
```

仅检查任务规模、不加载模型：

```bash
python scripts/run_teacher_benchmark.py --dry-run
```

环境快照可随实验一起保存：

```bash
python scripts/snapshot_environment.py
```
