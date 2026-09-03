# KD / RDID-MSA Workspace

本目录是 RDID-MSA 实验的唯一工作区。

## 目录

- `project/`：实验代码、配置、测试与技术报告；
- `docs/`：完整研究方案；
- `dataset/cmu_mosei/`：整理并审计通过的 22,856 条训练数据；
- `dataset/cmu_mosei_source/`：只用于溯源和重新生成的原始数据；
- `model/`：教师模型，以及 Qwen3-0.6B-Base、WavLM-Base-Plus、VideoMAE-Base 三个学生编码器；
- `outputs/teacher_benchmark/`：教师 1/20/500 条基准结果；
- `outputs/teacher_cache/`：七模态子集教师缓存；
- `outputs/probe/`：教师 Probe 参数及校准结果；
- `outputs/checkpoints/`：学生模型检查点；
- `outputs/logs/`：训练日志；
- `environment/`：Conda、pip、GPU 和 ffmpeg 复现快照。

Conda 环境名为 `kd`，由 Conda 管理在 `/home/wy/sjq/miniconda3/envs/kd`。

## 当前状态

已完成数据整理、教师 benchmark500 七子集推理、教师 Probe、学生网络与特征缓存，
以及 B0–B4、三种子高阶交互对照、权重搜索和 Stage A v2 确定性可靠性交互诊断。当前单元测试 35/35 通过。

主要结论是：完整知识蒸馏 B1 明显优于仅任务监督 B0；Stage A v2 中 SNR pair-only
和 selective top50 的三种子平均 MAE 均约为 0.6221，优于 subset4 的 0.6362，工程
gate 为 Go。当前尚未进行官方 16,326 条训练集的全量学生训练。

完整进展、指标、限制和下一步见 [`docs/实验汇总报告.md`](docs/实验汇总报告.md)，
正式 19-run 表见 [`project/reports/stage_a_v2_results.md`](project/reports/stage_a_v2_results.md)。

## GitHub 内容边界

仓库只保存代码、配置、测试、文档和环境快照。`dataset/`、`model/`、`outputs/`
及模型权重、特征缓存、检查点、预测和日志均由 `.gitignore` 排除，不上传 GitHub。
