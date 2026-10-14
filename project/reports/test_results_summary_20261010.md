# 训练进度与已完成测评汇总（2026-10-10）

最新进度、坐标兼容加权三 seed 结果及全量历史快照见 [2026-10-11 更新](test_results_summary_20261011.md)。下文保留 2026-10-10 的快照。

快照时间：2026-10-10T22:54:03+08:00。本报告汇总当前 MOSEI 六方法 × 三 seed 队列，同时提供全部历史实验的新快照入口。

## 当前进度

- 18 次训练：完成 9 次、运行 4 次、等待 5 次。完成的 9 个 run 均已完成两种验证 checkpoint 测试，共 18 份测试报告；两卡队列无失败或阻塞任务。
- 本次新增完成梯度投影 seed42、seed2026 的训练和双策略测评，梯度投影三个 seed 现已齐全。

| GPU | 实验 | 已完成轮数 | 当前训练进度 | 预计剩余训练时间 |
|---|---|---:|---|---:|
| GPU1 | uniform_interaction_coordinate_compatible_seed13 | 9/20 | 第 10 轮，600/2050 步 | 约 15.4 小时 |
| GPU0 | uniform_interaction_coordinate_compatible_seed42 | 7/20 | 第 8 轮，400/2050 步 | 约 18.4 小时 |
| GPU0 | uniform_interaction_coordinate_compatible_seed2026 | 7/20 | 第 8 轮，400/2050 步 | 约 18.2 小时 |
| GPU1 | explicit_full_kd_seed13 | 2/20 | 第 3 轮，1/2050 步 | 约 25.3 小时 |

预计时间依据最近三个已完成 epoch 的中位耗时粗估；当前轮次按训练步比例折算，实际还受验证、I/O 与并行影响，不包含后续测试时间。状态文件按固定步间隔更新，表格不是实时逐步进度。

等待训练：显式 FullKD 的 seed42、seed2026；显式 ICD 的 seed13、seed42、seed2026。坐标兼容三个 seed 均在训练。

## MOSEI 已完成训练的验证结果

每个单元格来自同一验证选中的 epoch；两种策略分别聚合，± 为三 seed 样本标准差。未完成训练的中途验证结果不进入最终均值。

| 方法 | 选模策略 | Valid MAE ↓ | Valid Acc2 NZ ↑ (%) |
|---|---|---:|---:|
| 原始 ICD | valid_mae | 0.4680 ± 0.0039 | 88.32 ± 0.66 |
| 原始 ICD | valid_acc2_nonzero | 0.4740 ± 0.0045 | 89.04 ± 0.08 |
| 共享 dropout | valid_mae | 0.4693 ± 0.0009 | 87.60 ± 0.76 |
| 共享 dropout | valid_acc2_nonzero | 0.4740 ± 0.0035 | 88.78 ± 0.34 |
| 梯度投影 | valid_mae | 0.4666 ± 0.0028 | 88.02 ± 1.61 |
| 梯度投影 | valid_acc2_nonzero | 0.4719 ± 0.0077 | 88.76 ± 0.66 |

## MOSEI 测试：验证 MAE 选模（预设主策略）

| 方法 | seeds | Test MAE ↓ | Test Acc2 NZ ↑ (%) | Test Pearson ↑ | Test F1 NZ ↑ (%) |
|---|---|---:|---:|---:|---:|
| 原始 ICD | 13、42、2026 | 0.4762 ± 0.0045 | 87.65 ± 0.29 | 0.8244 ± 0.0002 | 87.61 ± 0.30 |
| 共享 dropout | 13、42、2026 | 0.4833 ± 0.0016 | 86.86 ± 0.36 | 0.8192 ± 0.0021 | 86.95 ± 0.32 |
| 梯度投影 | 13、42、2026 | 0.4769 ± 0.0026 | 87.91 ± 0.34 | 0.8216 ± 0.0023 | 87.89 ± 0.32 |

## MOSEI 测试：验证 Acc2_nonzero 选模（次策略）

| 方法 | seeds | Test MAE ↓ | Test Acc2 NZ ↑ (%) | Test Pearson ↑ | Test F1 NZ ↑ (%) |
|---|---|---:|---:|---:|---:|
| 原始 ICD | 13、42、2026 | 0.4824 ± 0.0063 | 87.69 ± 0.37 | 0.8200 ± 0.0060 | 87.61 ± 0.35 |
| 共享 dropout | 13、42、2026 | 0.4863 ± 0.0043 | 87.16 ± 0.47 | 0.8169 ± 0.0043 | 87.12 ± 0.38 |
| 梯度投影 | 13、42、2026 | 0.4805 ± 0.0035 | 88.02 ± 0.23 | 0.8202 ± 0.0027 | 87.92 ± 0.29 |

Test Acc2/F1 均排除真实标签 0；完整测试 utterance 数 4659，nonzero 数 3634。每种方法均为相同 seeds13、42、2026；不混入历史不同协议的同名方法。

## 梯度投影相对原始 ICD：逐 seed 配对

| 策略 | seed | 原始 Test MAE | 投影 Test MAE | MAE 差（投影 − 原始） | Acc2 差（百分点） |
|---|---:|---:|---:|---:|
| valid_mae | 13 | 0.477143 | 0.479765 | +0.002622 | -0.248 |
| valid_mae | 42 | 0.471247 | 0.474520 | +0.003274 | +0.605 |
| valid_mae | 2026 | 0.480118 | 0.476519 | -0.003600 | +0.413 |
| valid_acc2_nonzero | 13 | 0.488241 | 0.483164 | -0.005077 | +0.715 |
| valid_acc2_nonzero | 42 | 0.475739 | 0.481722 | +0.005984 | +0.083 |
| valid_acc2_nonzero | 2026 | 0.483211 | 0.476519 | -0.006692 | +0.193 |

- MAE 选模：梯度投影平均 Test MAE 比原始 ICD 高 0.000765（略差约 0.16%），Test Acc2 高 0.257 个百分点。MAE 仅 seed2026 改善，1/3 seed；Pearson 均值从 0.8244 降至 0.8216。不能称为 MOSEI 主策略回归收益复现。
- Acc2 选模：梯度投影平均 Test MAE 降低 0.001929，Test Acc2 提高 0.330 个百分点；Acc2 三个 seed 均改善，MAE 为 2/3 seed 改善。这是次策略下较一致的分类收益，尚无显著性检验。
- 共享 dropout 两种策略的 Test MAE 与 Acc2 均值均差于原始 ICD。坐标兼容、显式 FullKD 和显式 ICD 尚无完整 MOSEI 测试结果。

## MOSEI 全部已完成测试明细

| 方法 | seed | 策略 | selected epoch | Test MAE | Test Acc2 NZ (%) |
|---|---:|---|---:|---:|---:|
| 原始 ICD | 13 | valid_mae | 14 | 0.477143 | 87.810 |
| 原始 ICD | 13 | valid_acc2_nonzero | 3 | 0.488241 | 87.314 |
| 原始 ICD | 42 | valid_mae | 11 | 0.471247 | 87.314 |
| 原始 ICD | 42 | valid_acc2_nonzero | 8 | 0.475739 | 87.700 |
| 原始 ICD | 2026 | valid_mae | 11 | 0.480118 | 87.837 |
| 原始 ICD | 2026 | valid_acc2_nonzero | 12 | 0.483211 | 88.057 |
| 共享 dropout | 13 | valid_mae | 16 | 0.481979 | 87.204 |
| 共享 dropout | 13 | valid_acc2_nonzero | 18 | 0.490889 | 87.534 |
| 共享 dropout | 42 | valid_mae | 14 | 0.485058 | 86.874 |
| 共享 dropout | 42 | valid_acc2_nonzero | 8 | 0.485483 | 86.626 |
| 共享 dropout | 2026 | valid_mae | 14 | 0.482861 | 86.489 |
| 共享 dropout | 2026 | valid_acc2_nonzero | 13 | 0.482432 | 87.314 |
| 梯度投影 | 13 | valid_mae | 9 | 0.479765 | 87.562 |
| 梯度投影 | 13 | valid_acc2_nonzero | 17 | 0.483164 | 88.030 |
| 梯度投影 | 42 | valid_mae | 11 | 0.474520 | 87.920 |
| 梯度投影 | 42 | valid_acc2_nonzero | 16 | 0.481722 | 87.782 |
| 梯度投影 | 2026 | valid_mae | 13 | 0.476519 | 88.250 |
| 梯度投影 | 2026 | valid_acc2_nonzero | 13 | 0.476519 | 88.250 |

梯度投影 seed2026 两种策略均选 epoch13，仍各保留一份策略报告，不将其计为两个独立实验。

## MOSI 与全部历史结果

- MOSI 14 方法的 84 次双策略测试已齐全。主策略三 seed：方差残差纠正 Test MAE=0.700167、Acc2=84.959%；梯度投影 MAE=0.708886、Acc2=85.061%；原始 ICD MAE=0.733188、Acc2=84.299%。完整两策略表与逐 seed 来源见下方全量汇总。
- GPU0 第二批 61 次历史补测、1 次共享特征生成及1次最终汇总，共63个队列任务已全部完成。之前缺少所选 epoch 权重的66条记录仍按原缺失清单保留。
- 全历史新快照含276个有逐轮情感指标的学生目录、552条 run × 策略记录；已有匹配测试为 MAE 策略177条、Acc2策略119条。范围包含历史 pilot/smoke，不能把这些数量当成当前正式主表实验数。
- 全量目录审计：JSON解析问题0、未唯一匹配测试0；14组同run同epoch的已有测评数值差异均按既有来源优先级解决，未解决冲突0。

## 核验与文件

- 本次逐份核对18份MOSEI测试报告的run身份、选模策略、训练history选中epoch、同轮验证指标、成功标记及测试样本数，并与全量汇总重新聚合的均值逐项比对一致。未加载权重或重新运行测评。
- [MOSEI逐seed测试 CSV](mosei_progress_results_20261010.csv)；[进度、验证、测试与配对差值 JSON](mosei_progress_results_20261010.json)。
- [全部历史实验新快照](all_experiments_dual_checkpoint_20261010/README.md)；[MAE策略全表](all_experiments_dual_checkpoint_20261010/best_valid_mae.md)；[Acc2策略全表](all_experiments_dual_checkpoint_20261010/best_valid_acc2_nonzero.md)；[全部逐run CSV](all_experiments_dual_checkpoint_20261010/per_run.csv)。
- [GPU0迁移记录](gpu0_transfer_20261010.md)；[原MOSI分析](test_results_summary_20261009.md)；[历史缺失权重清单](historical_test_completion_blocked_20261009.csv)。
