# 全部历史实验：两种验证 checkpoint 选择汇总

快照时间：2026-10-09T19:29:55.703316+08:00

- 清点 272 个有逐轮情感指标的学生训练目录，状态分布：{'complete': 259, 'training': 2, 'paused': 7, 'partial_no_report': 4}。另有 8 个无已完成 epoch 的配置目录，见 no_history_runs.csv；6 个 history 不含逐轮情感指标（教师 Probe），独立列入 JSON 排除清单。
- 共 544 个 run × 选模记录。已有同轮测试：MAE 策略 123 个，Acc2 策略 108 个；缺失记录详见 missing_tests.csv。
- 主口径统一为验证 MAE 最小 / 验证 Acc2_nonzero 最大；并列取最早 epoch。从各自 history 选择同一轮的全部指标，绝不拼接不同轮次的最优值。
- 历史有一份 Acc2 汇总采用“Acc2 并列后 F1 更大、再 MAE 更小”的规则，已原样保存在 historical_acc2_original_tiebreak.json；与本表统一规则不同，不能混用其结果。
- 若有测试 sweep，只取由验证指标确定的那个 epoch；不取 test-best。历史 sweep 已经看过测试集的事实不会因重新选模而消失，故不称为新的盲测。每个测试来源和探索属性均保留。
- 同轮存在多次测评且数值略异时，按来源优先采用正式选模测评；仅有旧 sweep 时优先固定 epoch 路径而非可变 best 别名。不按成绩挑选，相同优先级仍不一致则标为冲突。
- 早期 history 只记录 MAE、Acc2、Pearson：未保存的 F1 等指标保持缺失，不能从另一个 checkpoint 补齐。CSV/JSON 对每项聚合保留实际 n 与 seeds。
- 同一方法名在不同轮次、预算、架构、数据子集下分组报告，不直接混成三 seed。当前未完成训练只进入逐 run 明细，不进入完成实验均值。
- ± 为 seed 样本标准差；单 seed 不显示标准差；不足三 seed 不能称为三 seed 结论。测试覆盖不齐时均值仅描述表中已有 seeds，不与完整三 seed直接排名。
- 路径和 epoch 能追溯的已有权重位置写入 checkpoint_paths；这不是加载权重验证。没有权重不代表没有历史指标；没有同轮测试也不代表一定没有权重。
- 本次只读取并汇总记录，没有重训、补测或更改 GPU 队列。教师 probe、效率、交互重建诊断不按学生两种情感 checkpoint 重新排名。

## 文件

- [逐 run 双策略明细](per_run.csv)；[各组均值和标准差](aggregates.csv)；[缺失测试](missing_tests.csv)；[完整来源与审计](summary.json)。
- 按策略单独阅读：[best_valid_mae 汇总](best_valid_mae.md)；[best_valid_acc2_nonzero 汇总](best_valid_acc2_nonzero.md)。
- [全部实验线分组表](all_campaigns.md)；[全部 run 的轮次与结果](all_runs.md)。

## historical_mosi_seven

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| ensemble_full | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8577 ± 0.1018 | 86.11 ± 0.46 | 0.7776 ± 0.0217 | 86.06 ± 0.57 |
| ensemble_full | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.8099 ± 0.1543 | 84.50 ± 0.23 | 0.7879 ± 0.0152 | 84.47 ± 0.37 |
| ensemble_full | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7412 ± 0.0390 | 84.26 ± 2.02 | 0.7888 ± 0.0241 | 84.12 ± 2.11 |
| ensemble_full | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7300 ± 0.0125 | 85.11 ± 0.09 | 0.7986 ± 0.0056 | 85.17 ± 0.09 |
| first_order_interaction | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7809 ± 0.0181 | 85.96 ± 1.07 | 0.7803 ± 0.0195 | 85.99 ± 1.06 |
| first_order_interaction | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7183 ± 0.0226 | 84.50 ± 1.10 | 0.7938 ± 0.0076 | 84.38 ± 1.19 |
| first_order_interaction | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7650 ± 0.0147 | 83.80 ± 0.46 | 0.7750 ± 0.0045 | 83.64 ± 0.66 |
| first_order_interaction | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7235 ± 0.0450 | 84.25 ± 1.50 | 0.7998 ± 0.0134 | 84.27 ± 1.48 |
| first_second_order_interaction | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7828 ± 0.0377 | 85.80 ± 0.27 | 0.7753 ± 0.0136 | 85.73 ± 0.16 |
| first_second_order_interaction | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7411 ± 0.0216 | 84.76 ± 0.66 | 0.7900 ± 0.0159 | 84.71 ± 0.80 |
| first_second_order_interaction | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7589 ± 0.0244 | 84.88 ± 0.96 | 0.7769 ± 0.0119 | 84.70 ± 0.94 |
| first_second_order_interaction | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7345 ± 0.0380 | 84.20 ± 1.48 | 0.7891 ± 0.0192 | 84.25 ± 1.47 |
| full_kd | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7873 ± 0.0191 | 85.49 ± 1.07 | 0.7813 ± 0.0020 | 85.52 ± 1.09 |
| full_kd | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7197 ± 0.0073 | 84.60 ± 1.00 | 0.7861 ± 0.0065 | 84.56 ± 1.02 |
| full_kd | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7522 ± 0.0192 | 84.41 ± 0.71 | 0.7877 ± 0.0099 | 84.41 ± 0.70 |
| full_kd | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7316 ± 0.0056 | 83.99 ± 0.70 | 0.7847 ± 0.0111 | 84.01 ± 0.71 |
| random_orthogonal | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7887 ± 0.0270 | 85.80 ± 0.71 | 0.7851 ± 0.0136 | 85.79 ± 0.74 |
| random_orthogonal | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7292 ± 0.0354 | 85.06 ± 0.30 | 0.7968 ± 0.0127 | 85.00 ± 0.37 |
| random_orthogonal | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7564 ± 0.0217 | 85.65 ± 0.93 | 0.7797 ± 0.0203 | 85.61 ± 0.88 |
| random_orthogonal | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7169 ± 0.0154 | 84.71 ± 0.89 | 0.7972 ± 0.0087 | 84.73 ± 0.87 |
| subset7 | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8265 ± 0.0247 | 85.80 ± 0.27 | 0.7718 ± 0.0051 | 85.84 ± 0.23 |
| subset7 | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7476 ± 0.0263 | 83.89 ± 0.63 | 0.7865 ± 0.0081 | 83.77 ± 0.74 |
| subset7 | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7702 ± 0.0136 | 84.10 ± 0.27 | 0.7812 ± 0.0045 | 83.91 ± 0.36 |
| subset7 | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7576 ± 0.0378 | 84.50 ± 1.62 | 0.7888 ± 0.0166 | 84.58 ± 1.59 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7849 ± 0.0072 | 85.49 ± 1.63 | 0.7790 ± 0.0070 | 85.50 ± 1.67 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7334 ± 0.0453 | 83.99 ± 0.70 | 0.7920 ± 0.0216 | 83.96 ± 0.63 |
| uniform_interaction | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7721 ± 0.0108 | 84.10 ± 0.96 | 0.7849 ± 0.0075 | 83.94 ± 0.96 |
| uniform_interaction | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7388 ± 0.0141 | 84.86 ± 0.75 | 0.7892 ± 0.0170 | 84.93 ± 0.76 |

## historical_mosei_seven

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| ensemble_full | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4967 | 88.94 | 0.7826 | 88.94 |
| ensemble_full | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5097 | 87.01 | 0.8048 | 87.00 |
| ensemble_full | mosei | full | valid_mae | validation | 13 (1/1) | 0.4718 | 88.04 | 0.7921 | 88.08 |
| ensemble_full | mosei | full | valid_mae | test | 13 (1/1) | 0.4869 | 87.84 | 0.8172 | 87.84 |
| first_order_interaction | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4694 ± 0.0013 | 88.80 ± 0.18 | 0.7939 ± 0.0041 | 88.70 ± 0.19 |
| first_order_interaction | mosei | full | valid_acc2_nonzero | test | 13 (1/3) | 0.4891 | 87.26 | 0.8186 | 87.09 |
| first_order_interaction | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4653 ± 0.0017 | 87.71 ± 0.84 | 0.7973 ± 0.0002 | 87.75 ± 0.77 |
| first_order_interaction | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4814 ± 0.0037 | 87.19 ± 0.73 | 0.8186 ± 0.0020 | 87.21 ± 0.67 |
| first_second_order_interaction | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4902 ± 0.0187 | 88.73 ± 0.30 | 0.7913 ± 0.0017 | 88.68 ± 0.27 |
| first_second_order_interaction | mosei | full | valid_acc2_nonzero | test | 13,2026 (2/3) | 0.4940 ± 0.0236 | 87.62 ± 0.12 | 0.8183 ± 0.0079 | 87.57 ± 0.06 |
| first_second_order_interaction | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4675 ± 0.0017 | 88.22 ± 0.47 | 0.7928 ± 0.0025 | 88.23 ± 0.40 |
| first_second_order_interaction | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4780 ± 0.0007 | 87.47 ± 0.21 | 0.8215 ± 0.0023 | 87.46 ± 0.14 |
| full_kd | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4792 ± 0.0174 | 88.66 ± 0.24 | 0.7904 ± 0.0046 | 88.62 ± 0.21 |
| full_kd | mosei | full | valid_acc2_nonzero | test | 13,42 (2/3) | 0.4949 ± 0.0242 | 87.66 ± 0.45 | 0.8143 ± 0.0040 | 87.65 ± 0.40 |
| full_kd | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4671 ± 0.0038 | 88.02 ± 0.66 | 0.7901 ± 0.0075 | 88.01 ± 0.66 |
| full_kd | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4798 ± 0.0019 | 87.28 ± 0.61 | 0.8169 ± 0.0042 | 87.28 ± 0.57 |
| random_orthogonal | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4730 ± 0.0043 | 88.94 ± 0.25 | 0.7917 ± 0.0050 | 88.85 ± 0.27 |
| random_orthogonal | mosei | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.4828 ± 0.0076 | 87.67 ± 0.57 | 0.8201 ± 0.0024 | 87.57 ± 0.59 |
| random_orthogonal | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4674 ± 0.0025 | 87.99 ± 0.38 | 0.7902 ± 0.0050 | 88.01 ± 0.36 |
| random_orthogonal | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4767 ± 0.0031 | 87.62 ± 0.77 | 0.8219 ± 0.0037 | 87.62 ± 0.75 |
| subset7 | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4763 ± 0.0223 | 88.66 ± 0.14 | 0.7925 ± 0.0081 | 88.55 ± 0.26 |
| subset7 | mosei | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.4876 ± 0.0189 | 87.64 ± 0.27 | 0.8178 ± 0.0064 | 87.52 ± 0.33 |
| subset7 | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4653 ± 0.0040 | 88.43 ± 0.53 | 0.7944 ± 0.0048 | 88.35 ± 0.55 |
| subset7 | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4780 ± 0.0044 | 87.61 ± 0.33 | 0.8196 ± 0.0039 | 87.52 ± 0.32 |
| uniform_interaction | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4852 ± 0.0222 | 88.83 ± 0.33 | 0.7933 ± 0.0049 | 88.72 ± 0.36 |
| uniform_interaction | mosei | full | valid_acc2_nonzero | test | 13 (1/3) | 0.5173 | 87.64 | 0.8132 | 87.48 |
| uniform_interaction | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4698 ± 0.0056 | 87.88 ± 0.11 | 0.7968 ± 0.0040 | 87.84 ± 0.15 |
| uniform_interaction | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4762 ± 0.0045 | 87.78 ± 0.44 | 0.8237 ± 0.0013 | 87.73 ± 0.42 |

## mosi_research_14

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| dgl_icd | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8322 ± 0.0196 | 85.65 ± 0.46 | 0.7584 ± 0.0132 | 85.59 ± 0.49 |
| dgl_icd | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.8341 ± 0.0065 | 83.49 ± 0.32 | 0.7566 ± 0.0213 | 83.54 ± 0.33 |
| dgl_icd | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7775 ± 0.0160 | 82.72 ± 1.63 | 0.7770 ± 0.0137 | 82.56 ± 1.78 |
| dgl_icd | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.8083 ± 0.0130 | 82.62 ± 0.26 | 0.7538 ± 0.0093 | 82.69 ± 0.31 |
| ensemble_source | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7661 ± 0.0121 | 86.27 ± 0.71 | 0.7824 ± 0.0096 | 86.20 ± 0.79 |
| ensemble_source | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7411 ± 0.0356 | 84.15 ± 1.68 | 0.7919 ± 0.0122 | 84.15 ± 1.78 |
| ensemble_source | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7505 ± 0.0230 | 85.65 ± 0.00 | 0.7919 ± 0.0005 | 85.50 ± 0.04 |
| ensemble_source | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7410 ± 0.0293 | 84.65 ± 0.63 | 0.7965 ± 0.0020 | 84.73 ± 0.62 |
| explicit_full_kd | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7931 ± 0.0445 | 84.88 ± 1.07 | 0.7699 ± 0.0150 | 84.83 ± 1.13 |
| explicit_full_kd | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7397 ± 0.0277 | 84.50 ± 0.87 | 0.7885 ± 0.0166 | 84.54 ± 0.86 |
| explicit_full_kd | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7681 ± 0.0251 | 83.18 ± 0.71 | 0.7715 ± 0.0153 | 83.06 ± 0.75 |
| explicit_full_kd | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7194 ± 0.0239 | 85.01 ± 0.84 | 0.7944 ± 0.0039 | 85.07 ± 0.81 |
| explicit_uniform_interaction | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7778 ± 0.0115 | 85.65 ± 0.93 | 0.7896 ± 0.0068 | 85.50 ± 0.92 |
| explicit_uniform_interaction | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7709 ± 0.0211 | 84.76 ± 0.81 | 0.8042 ± 0.0046 | 84.81 ± 0.77 |
| explicit_uniform_interaction | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7697 ± 0.0122 | 84.10 ± 1.93 | 0.7904 ± 0.0079 | 83.88 ± 1.92 |
| explicit_uniform_interaction | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7497 ± 0.0465 | 85.11 ± 1.22 | 0.8056 ± 0.0093 | 85.19 ± 1.20 |
| fusion_only | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8373 ± 0.0940 | 85.49 ± 1.34 | 0.7692 ± 0.0351 | 85.52 ± 1.36 |
| fusion_only | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7991 ± 0.0476 | 83.43 ± 0.77 | 0.7754 ± 0.0099 | 83.22 ± 0.91 |
| fusion_only | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7420 ± 0.0297 | 84.10 ± 1.75 | 0.7898 ± 0.0144 | 83.96 ± 1.84 |
| fusion_only | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7339 ± 0.0245 | 84.55 ± 2.41 | 0.7886 ± 0.0038 | 84.57 ± 2.36 |
| late_decay | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7914 ± 0.0288 | 85.65 ± 1.39 | 0.7851 ± 0.0112 | 85.65 ± 1.34 |
| late_decay | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7805 ± 0.0542 | 84.15 ± 1.22 | 0.7728 ± 0.0200 | 84.14 ± 1.21 |
| late_decay | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7659 ± 0.0239 | 83.80 ± 2.82 | 0.7945 ± 0.0070 | 83.57 ± 3.08 |
| late_decay | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7357 ± 0.0344 | 85.01 ± 1.14 | 0.7960 ± 0.0119 | 85.07 ± 1.10 |
| order_routed | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8134 ± 0.0623 | 85.49 ± 1.07 | 0.7731 ± 0.0271 | 85.48 ± 1.04 |
| order_routed | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7656 ± 0.0565 | 84.15 ± 1.15 | 0.7780 ± 0.0177 | 84.14 ± 1.11 |
| order_routed | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7558 ± 0.0170 | 84.26 ± 1.67 | 0.8003 ± 0.0040 | 84.14 ± 1.71 |
| order_routed | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7441 ± 0.0118 | 84.04 ± 0.84 | 0.7910 ± 0.0055 | 84.12 ± 0.83 |
| scrambled_routed | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7642 ± 0.0157 | 86.57 ± 0.46 | 0.7883 ± 0.0116 | 86.56 ± 0.51 |
| scrambled_routed | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7190 ± 0.0277 | 85.32 ± 1.07 | 0.7943 ± 0.0128 | 85.29 ± 1.05 |
| scrambled_routed | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7619 ± 0.0150 | 86.11 ± 0.46 | 0.7865 ± 0.0132 | 86.06 ± 0.50 |
| scrambled_routed | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7448 ± 0.0251 | 84.86 ± 0.49 | 0.7924 ± 0.0098 | 84.88 ± 0.52 |
| target_uniform | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8308 ± 0.0556 | 85.34 ± 0.27 | 0.7690 ± 0.0051 | 85.26 ± 0.14 |
| target_uniform | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.8175 ± 0.0965 | 84.10 ± 0.62 | 0.7834 ± 0.0300 | 84.12 ± 0.68 |
| target_uniform | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7411 ± 0.0179 | 84.10 ± 0.71 | 0.7797 ± 0.0151 | 84.05 ± 0.69 |
| target_uniform | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7103 ± 0.0240 | 85.37 ± 0.00 | 0.7992 ± 0.0058 | 85.39 ± 0.04 |
| target_variance | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7682 ± 0.0486 | 85.96 ± 0.71 | 0.7834 ± 0.0323 | 85.95 ± 0.66 |
| target_variance | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7164 ± 0.0202 | 84.76 ± 0.40 | 0.8013 ± 0.0037 | 84.73 ± 0.47 |
| target_variance | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7494 ± 0.0278 | 85.80 ± 0.71 | 0.7887 ± 0.0283 | 85.74 ± 0.65 |
| target_variance | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7002 ± 0.0194 | 84.96 ± 0.09 | 0.8060 ± 0.0050 | 85.00 ± 0.04 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | validation | 2026,13,42 (3/3) | 0.7839 ± 0.0343 | 85.96 ± 0.96 | 0.7847 ± 0.0077 | 85.96 ± 0.92 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | test | 2026,13,42 (3/3) | 0.7712 ± 0.0596 | 83.43 ± 0.54 | 0.7766 ± 0.0171 | 83.45 ± 0.52 |
| uniform_interaction | mosi | full | valid_mae | validation | 2026,13,42 (3/3) | 0.7595 ± 0.0185 | 84.88 ± 0.96 | 0.7924 ± 0.0077 | 84.82 ± 1.05 |
| uniform_interaction | mosi | full | valid_mae | test | 2026,13,42 (3/3) | 0.7332 ± 0.0167 | 84.30 ± 0.53 | 0.7873 ± 0.0048 | 84.35 ± 0.54 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7945 ± 0.0600 | 86.11 ± 0.46 | 0.7836 ± 0.0256 | 86.05 ± 0.37 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7648 ± 0.0471 | 84.76 ± 0.70 | 0.7881 ± 0.0111 | 84.76 ± 0.78 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7536 ± 0.0344 | 84.88 ± 0.53 | 0.7916 ± 0.0198 | 84.76 ± 0.60 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7257 ± 0.0084 | 84.65 ± 0.35 | 0.7941 ± 0.0011 | 84.71 ± 0.33 |
| uniform_interaction_gradient_projected | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7718 ± 0.0211 | 86.42 ± 0.53 | 0.8002 ± 0.0120 | 86.46 ± 0.56 |
| uniform_interaction_gradient_projected | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7162 ± 0.0132 | 84.35 ± 0.49 | 0.7990 ± 0.0066 | 84.26 ± 0.47 |
| uniform_interaction_gradient_projected | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7276 ± 0.0324 | 84.88 ± 0.71 | 0.8100 ± 0.0126 | 84.81 ± 0.69 |
| uniform_interaction_gradient_projected | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7089 ± 0.0109 | 85.06 ± 0.40 | 0.8048 ± 0.0055 | 85.08 ± 0.40 |
| uniform_interaction_shared_dropout | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7931 ± 0.0228 | 85.34 ± 0.71 | 0.7745 ± 0.0180 | 85.31 ± 0.73 |
| uniform_interaction_shared_dropout | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7686 ± 0.0071 | 83.59 ± 0.98 | 0.7860 ± 0.0155 | 83.60 ± 0.95 |
| uniform_interaction_shared_dropout | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7645 ± 0.0129 | 83.18 ± 0.96 | 0.7855 ± 0.0087 | 82.95 ± 1.04 |
| uniform_interaction_shared_dropout | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7433 ± 0.0296 | 84.76 ± 0.40 | 0.7970 ± 0.0069 | 84.84 ± 0.40 |

## mosei_extensions_v3

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| uniform_interaction | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4740 ± 0.0045 | 89.04 ± 0.08 | 0.7931 ± 0.0053 | 88.94 ± 0.12 |
| uniform_interaction | mosei | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.4824 ± 0.0063 | 87.69 ± 0.37 | 0.8200 ± 0.0060 | 87.61 ± 0.35 |
| uniform_interaction | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4680 ± 0.0039 | 88.32 ± 0.66 | 0.7973 ± 0.0032 | 88.28 ± 0.62 |
| uniform_interaction | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4762 ± 0.0045 | 87.65 ± 0.29 | 0.8244 ± 0.0002 | 87.61 ± 0.30 |
| uniform_interaction_gradient_projected | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4657 | 88.53 | 0.8007 | 88.46 |
| uniform_interaction_gradient_projected | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.4832 | 88.03 | 0.8175 | 87.95 |
| uniform_interaction_gradient_projected | mosei | full | valid_mae | validation | 13 (1/1) | 0.4638 | 88.25 | 0.7982 | 88.24 |
| uniform_interaction_gradient_projected | mosei | full | valid_mae | test | 13 (1/1) | 0.4798 | 87.56 | 0.8203 | 87.56 |
| uniform_interaction_shared_dropout | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4740 ± 0.0035 | 88.78 ± 0.34 | 0.7934 ± 0.0007 | 88.75 ± 0.21 |
| uniform_interaction_shared_dropout | mosei | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.4863 ± 0.0043 | 87.16 ± 0.47 | 0.8169 ± 0.0043 | 87.12 ± 0.38 |
| uniform_interaction_shared_dropout | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4693 ± 0.0009 | 87.60 ± 0.76 | 0.7933 ± 0.0034 | 87.70 ± 0.69 |
| uniform_interaction_shared_dropout | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4833 ± 0.0016 | 86.86 ± 0.36 | 0.8192 ± 0.0021 | 86.95 ± 0.32 |

## 历史原 Acc2 并列规则（独立附表）

以下保留当时已发布的 5 方法 × 3 seed 测试结果，按 Acc2、F1、MAE、最早轮次依次选模。它与上文统一规则不同，不能互相填补缺失。

| 方法 | seeds | Test MAE ↓ | Test Acc2 NZ ↑ (%) | Test Pearson ↑ | Test F1 NZ ↑ (%) |
|---|---|---:|---:|---:|---:|
| first_second_order_interaction | 13,42,2026 | 0.7411 ± 0.0216 | 84.76 ± 0.66 | 0.7900 ± 0.0159 | 84.71 ± 0.80 |
| full_kd | 13,42,2026 | 0.7148 ± 0.0126 | 84.35 ± 0.62 | 0.7875 ± 0.0068 | 84.28 ± 0.59 |
| random_orthogonal | 13,42,2026 | 0.7292 ± 0.0354 | 85.06 ± 0.30 | 0.7968 ± 0.0127 | 85.00 ± 0.37 |
| subset7 | 13,42,2026 | 0.7476 ± 0.0263 | 83.89 ± 0.63 | 0.7865 ± 0.0081 | 83.77 ± 0.74 |
| uniform_interaction | 13,42,2026 | 0.7334 ± 0.0453 | 83.99 ± 0.70 | 0.7920 ± 0.0216 | 83.96 ± 0.63 |

## 尚未完成的运行

| Run | 状态 | 已完成 epoch |
|---|---|---:|
| outputs/experiments/equal_modality_extensions_v3/mosei/uniform_interaction_gradient_projected_seed2026 | training | 1 |
| outputs/experiments/equal_modality_extensions_v3/mosei/uniform_interaction_gradient_projected_seed42 | training | 9 |
| outputs/experiments/multiseed_controls_v1/mosei_ensemble_controls/students/ensemble_full_seed2026 | paused | 7 |
| outputs/experiments/multiseed_controls_v1/mosei_ensemble_controls/students/ensemble_full_seed42 | paused | 7 |
| outputs/experiments/rdid_v2_mosei/students/amplitude_u_seed13 | paused | 19 |
| outputs/experiments/research_icd_smoke_20261008/dgl_icd | paused | 1 |
| outputs/experiments/research_icd_smoke_20261008/order_routed | paused | 1 |
| outputs/experiments/research_icd_smoke_20261008/target_variance | paused | 1 |
| outputs/student/polarity_e2_seed13 | partial_no_report | 13 |
| outputs/student/polarity_e2_seed13_gpu1 | partial_no_report | 3 |
| outputs/student/v2_pair_raw_seed2026 | partial_no_report | 12 |
| outputs/student/v2_snr_interaction4_seed2026 | partial_no_report | 16 |
| outputs/student/video_adaptation_smoke_v1/video_lora | paused | 1 |

## 匹配审计

- JSON 解析问题：0；未唯一匹配的测试记录：0；同 run 同 epoch 测评数值差异：14，其中未解决 0。详情见 summary.json。
