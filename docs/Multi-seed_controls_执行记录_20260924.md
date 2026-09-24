# Multi-seed controls 执行记录（2026-09-24）

## 冻结口径

- 每个完成 epoch 均保留精简模型 checkpoint；`last.pt` 单独保存优化器和 RNG，用于恢复训练。
- checkpoint 只按 official validation MAE 选择。
- 训练结束后，仅将 valid-best `best.pt` 在 official test 上评测一次。
- 不使用逐 epoch test-best 作为论文主结果。
- Random Orthogonal 的坐标随机种子固定为 `20260922`，不随训练 seed 改变。
- 最大四个训练并行、每张 GPU 最多两个训练；单任务按 30 GiB、每卡保留 8 GiB 进行准入。

## 正式矩阵

### MOSEI / Qiii

复用已完成的 M3 Full KD 和 M4 Uniform Interaction 三 seed，不重跑。新增：

- `subset7`: seeds 42, 2026
- `random_orthogonal`: seeds 42, 2026

共 4 个新 run。代码快照位于：

`/ai/sjq/kd_multiseed_20260924`

持久化输出位于：

`/ai/sjq/kd/outputs/experiments/multiseed_controls_v1/mosei`

tmux session：

`rdid_ms_mosei_20260924`

### MOSI / 本地

- `full_kd`: seeds 42, 2026
- `subset7`: seeds 42, 2026
- `first_second_order_interaction`: seeds 42, 2026
- `random_orthogonal`: seeds 42, 2026
- `uniform_interaction`: seeds 42, 2026

共 10 个新 run。计划输出：

`/home/wy/sjq/kd/outputs/experiments/multiseed_controls_v1/mosi`

删除确认不被项目使用的三个 MOSEI 预计算特征目录后，根分区可用空间约 97 GiB。
本地 launcher 要求启动时至少有 80 GiB 可用空间，并在项目输出目录运行 MOSI 队列。

#### 七方法扩展

原五方法三 seed 已完成后，补充以下两种方法：

- `ensemble_full`: seeds 13, 42, 2026
- `first_order_interaction`: seeds 13, 42, 2026

seed 13 复用已有完整 20-epoch checkpoint，并重新评测 validation-MAE-best；seeds 42/2026
各新增训练一次，共四个新 run。输出位于：

`/home/wy/sjq/kd/outputs/experiments/multiseed_controls_v1/mosi_additional_controls`

训练 tmux 为 `rdid_ms_mosi_extra_20260924`。四个任务双卡并行，每卡最多两个；逐 epoch
保留 checkpoint，按 validation MAE 选模后只评测一次 official test。独立 watcher 在新增任务
完成后自动生成七方法 × 三 seed 的统一汇总。

## 远端结果回传

Qiii 队列完成后执行：

```bash
/home/wy/sjq/miniconda3/envs/kd/bin/python \
  project/scripts/transfer_multiseed_results.py
```

脚本要求远端队列为 `complete`，在远端为全部普通文件生成 SHA-256 清单，传回
`/home/wy/sjq/kd/outputs/experiments/multiseed_controls_v1/mosei` 后逐文件校验。

本地 transfer watcher 已配置为等待 Qiii 队列完成，之后自动执行上述回传与汇总。

### 2026-09-24 队列恢复

- `subset7_seed2026` 按 validation MAE 早停于 epoch 12，best epoch 5；训练和全部
  checkpoint 完整，仅 official-test 因 test manifest 保留本机绝对路径而失败。
- Qiii 的 4,687 条 test 媒体按远端路径核验：音频、视频缺失均为 0，
  所有音频均可读；恢复时只更正 manifest 路径前缀，不改变样本或标签。
- 后续队列首次启动误用 `/usr/bin/python`，缺少 PyTorch，尚未进入训练。恢复器
  `project/scripts/recover_qiii_multiseed_queues.py` 使用 Qiii 已验证的 Python 环境：
  先补做已训练 run 的 official-test，再恢复资源感知补位，最后重建两个队列的
  `complete` 状态以触发校验回传。
- 修复 Python 环境后的首次补位暴露了 CUDA 显存统计的启动滞后：控制器在第一个
  新任务尚未建立完整 CUDA 显存占用时，又向同卡投放了第二个，两者均在
  epoch 1 step 1 附近 OOM，未产生 checkpoint。这两个不可恢复的空壳目录保留到
  `followup_order_controls/archive/`；调度器增加同卡新任务 120 秒显存建立冷却，
  避免再次超发。
- `subset7_seed2026` 补测已完成：official-test MAE `0.480930`、Pearson `0.817778`、
  Acc-2 NZ `0.878371`、F1 NZ `0.876572`、Acc-2 HZ `0.862417`、F1 HZ `0.862960`、
  Acc-7 `0.574158`。
- 冷却修复后，`first_second_order_interaction_seed42` 已在 GPU0 稳定进入 epoch 1
  训练（已过 step 100）；该卡总显存约 60.4 GiB，队列未再超发，余下三项
  保持 pending 并将在资源释放后自动补位。

## 数据清理记录

2026-09-24 经用户明确批准，不备份并删除以下项目未使用的预计算特征：

- `dataset/cmu_mosei_source/Videos/Full/OpenFace2.0`：约 63.08 GiB
- `dataset/cmu_mosei_source/Audio/Full/COVAREP`：约 21.25 GiB
- `dataset/cmu_mosei_source/Videos/Full/FACET_4.2`：约 3.95 GiB

合计释放约 88.27 GiB；删除后再次检查 MOSEI/MOSI 四份活动 manifest，媒体缺失均为 0。
保留了原始 Combined 视频、WAV16k、Videos/Segmented、已准备的 MOSEI/MOSI 媒体。

## 验证记录

- 本地 `test_main_table_kd.py` 与 `test_interaction.py`：32 passed。
- Qiii 同一测试集：32 passed。
- Qiii 三个新增 MOSEI 方法的 frozen-asset dry-run：通过。
- 初始队列中的 First+Second-order 已按用户要求停止，未完成目录完整保存在
  `archive/switched_to_subset_random_20260924_0032/`，没有删除。
- 正式 Qiii 队列已更换为 Subset-7/Random Orthogonal × seeds 42/2026，共四项。

#### 七方法扩展

现有主队列的 Subset-7/Random Orthogonal × seeds 42/2026 保持运行。另建资源感知的
后续队列，按 GPU 实际空闲显存立即补位，不等待主队列四项全部结束：

- `first_second_order_interaction`: seeds 42, 2026
- `first_order_interaction`: seeds 42, 2026

后续队列输出位于主 MOSEI 结果树内：

`/ai/sjq/kd/outputs/experiments/multiseed_controls_v1/mosei/followup_order_controls`

tmux 为 `rdid_ms_mosei_order_followup_20260924`。Ensemble Full KD 的 seeds 42/2026
改在本机双卡运行，输出位于：

`/home/wy/sjq/kd/outputs/experiments/multiseed_controls_v1/mosei_ensemble_controls`

本机 tmux 为 `rdid_ms_mosei_ensemble_local_20260924`；每张 GPU 限一项，等待当前 MOSI
任务释放对应显存后立即启动。所有新增 run 均逐 epoch 保留 checkpoint、按 validation MAE
选模并只对所选 checkpoint 执行一次 official test。
