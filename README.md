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

已完成 official train/valid full-scale 的 B0 Student-only、B1 Full KD、B2 subset4
和 SNR pair-only 三种子实验，以及 paired bootstrap 和 subgroup analysis。当前单元
测试 44/44 通过。

official-valid 最低平均 MAE 为 B1 Full KD 的 0.5072 ± 0.0046。SNR pair-only 将
pair interaction reconstruction error 相对降低约 35.5%，但没有稳定提升 downstream
sentiment MAE。当前研究重点已转向 task-utility-aware interaction distillation。
Official test 尚未使用。

后续 Stage C 三种子复验中，Reliability × Utility 的 official-valid MAE 为
0.5055 ± 0.0023，比 B1 平均低 0.0017，但未达到预注册建议效果量 0.005，且逐 seed
bootstrap 区间均跨 0。当前停止继续叠加 conflict gate；B1 仍作为统计上最可靠的主基线。

Acc-2 分支已完成 P0 三 seed显著性，以及 E1 Binary Full KD / E2 Binary
subset4 KD 的 seeds 13、42 小规模验证。E2 相对 B2 的 Acc-2 在 2/2 seeds
提高，平均增量为 0.0063，但逐 seed bootstrap / McNemar 均不显著，且平均
MAE 退化 0.0249，未通过 regression safety gate。按预注册停止 Reliability、
Boundary、LoRA 和 binary 权重搜索；Official test 仍未使用。

Stage D 的 LoRA(T+A) 已完成 seeds 13、42：Full KD + LoRA 的两 seed 平均
MAE 为 0.4752，相同 seed 的冻结 B1 为 0.5081。RU + LoRA 的已完成 seed 42
进一步达到 0.4638；其余 seed 正在运行。第一版 cosine Hidden KD 未通过 Gate。
阶段性结果见 [`project/reports/stage_d_interim_results.md`](project/reports/stage_d_interim_results.md)。

完整进展、指标、限制和下一步见 [`docs/实验汇总报告.md`](docs/实验汇总报告.md)，
Stage B 主表见 [`project/reports/stage_b_baselines_three_seed.md`](project/reports/stage_b_baselines_three_seed.md)，
正式 19-run 诊断表见 [`project/reports/stage_a_v2_results.md`](project/reports/stage_a_v2_results.md)。

## GitHub 内容边界

仓库只保存代码、配置、测试、文档和环境快照。`dataset/`、`model/`、`outputs/`
及模型权重、特征缓存、检查点、预测和日志均由 `.gitignore` 排除，不上传 GitHub。
