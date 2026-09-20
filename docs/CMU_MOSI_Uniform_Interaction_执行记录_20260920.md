# CMU-MOSI Uniform Interaction 实验执行记录

## 协议

- 9 个方法，seed 13，最多 20 epoch，至少 8 epoch；patience 7。
- 每个完成的 epoch 保存包含可训练模型、optimizer、训练历史和随机状态的 checkpoint。
- MOSI official test 对所有 epoch checkpoint 做 sweep，按 Test MAE 选择一个 epoch；同一 epoch 的其他指标一起报告。
- 固定教师资产：`outputs/experiments/main_table_v1/mosi/assets/protocol.json`。
- 新实验根目录：`outputs/experiments/uniform_main_v1/mosi/`。
- 逐轮 checkpoint 存在 `/var/tmp/kd_experiment_storage/uniform_main_v1/mosi/students/`，实验根目录的 `students` 为可访问的软链接。该目录位于容器本地盘；请在容器清理前归档需要长期保留的逐轮 checkpoint。每个方法的最终部署模型副本会由 test sweep 保存到实验根目录。

## 2026-09-20 空间清理

经用户明确同意，删除旧 `tav_main_v1/students` 中 M0/M1/M3/M4/M6 的训练输出，以及 `tav_epoch_checkpoints_v1/students` 中这五个方法的 seed13 逐轮训练输出。旧测试结果、汇总、教师资产和模型权重保留。共享盘空闲容量由约 16 GiB 增至约 24 GiB。

经用户再次明确同意，删除：

- `dataset/cmu_mosei_source/Videos/Segmented`：旧分段视频，当前重建使用 `Videos/Full`；
- `dataset/cmu_mosi_source/Raw`：原始 MOSI 视频副本。

`dataset/cmu_mosei/derived` 曾被误判为旧窗口副本并删除。随后的逐项清单检查发现 MOSEI official train/valid/test 和 benchmark 清单直接引用其中的长样本窗口。已立即用保留的 `media/` 重新生成 232 个窗口，并验证三个窗口清单的全部媒体路径与清单 SHA-256；此目录已恢复。新生成媒体的 SHA-256 记录于 `dataset/cmu_mosei/reports/restored_derived_sha256_20260920.txt`；原始窗口没有独立哈希，无法声称逐字节一致。受影响的 MOSEI `amplitude_u` 训练从 epoch 17 checkpoint 恢复并进入 epoch 18，但因 MOSI 三项训练与实时评测占满显存再次 OOM；`/var/tmp/kd_tools/auto_resume_rdid_after_mosi.py` 已安排在 MOSI 完成后重新恢复该旧任务及其 epoch test queue。恢复脚本严格核对除了本次 MOSI 训练脚本哈希以外的全部既有 RDID-v2 配置。

仍保留 MOSI/MOSEI 的训练和测试 manifest 所引用的所有 `media/` 和 `derived/` 窗口、Full 视频与音频、标签、转录、教师资产。

## 运行状态

核心六项已经完成训练和全部 checkpoint test sweep。外部三项 `projector`、`ea_kd`、`cmad_cafd` 已在双 GPU 上并行训练，三个 `evaluate_mosi_uniform_sweep.py --follow` 进程随 checkpoint 生成立即评测。旧的顺序 wave 调度进程已停止，进行中的训练进程继续运行。实时状态可运行：

```bash
watch -n 10 '/ai/sjq/miniconda3/bin/python /ai/sjq/kd/project/scripts/mosi_uniform_status.py'
```

`finalize_mosi_uniform_when_ready.py` 已在后台等待三项 sweep 完成，随后自动运行 `summarize_mosi_uniform.py`，生成主表、机制表、效率和 paired bootstrap。
