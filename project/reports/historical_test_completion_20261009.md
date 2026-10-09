# 历史双 checkpoint 补测（2026-10-09）

最新效果分析见 [当前测试结果汇总](test_results_summary_20261009.md)。

用户已明确要求补齐需要测试但尚未测试的实验。本轮覆盖已完成训练的全量 MOSI/MOSEI 学生实验，保持每个原 run 的验证选模规则、架构与权重。小样本 pilot、smoke/preflight 和教师探针不扩展为正式全量测试；尚未完成训练的 run 不用中途 checkpoint 代替最终结果。

## 已完成与已排队

- 第一批 67 次测试已全部完成：最新 MOSI 八方法 48 次、决策蒸馏 12 次、历史主表补测 7 次。MOSI 14 方法目前有 168 份完整验证/测试报告（42 run × 2 策略 × 2 split）。
- 第二批 61 次测试已接入 GPU0：3 个历史主表 checkpoint、26 个冻结特征模型 checkpoint、25 个 T/A LoRA checkpoint、7 个视频适配 checkpoint。两策略选中同一轮时仍保留各自完整报告。
- 第二批另外包含 1 次共享测试特征生成与 1 次最终汇总，共 63 个队列任务。缓存只包含 MOSEI official test，生成 4687 个窗口的冻结编码器特征，为早期模型共用；预算预留 25 GiB 磁盘。
- GPU0 同时最多 2 个测评任务，各预留 20000 MiB，额外保留 6000 MiB；GPU1 原训练/测试队列保持运行。GPU1 已有依赖任务不重复加入 GPU0。
- 队列完成后自动刷新 MOSI 14 方法与全部历史双策略汇总。

## 暂时无法补测的 66 条选模记录

这些记录在原训练目录没有对应选中 epoch 的权重，在现存压缩归档中也未找到。部分早期训练只保存了原 best.pt；仅有逐轮验证指标不能恢复该轮参数。不使用其他 epoch 冒充，不自动重训。这里是 66 条 run × 策略记录，不是 66 个独立实验；原有验证结果仍保留。

| Run | 策略 | 所需 epoch |
|---|---|---:|
| outputs/experiments/tav_main_v1/students/M0_seed13 | valid_acc2_nonzero | 8 |
| outputs/experiments/tav_main_v1/students/M1_seed13 | valid_acc2_nonzero | 8 |
| outputs/experiments/tav_main_v1/students/M1_seed2026 | valid_acc2_nonzero | 8 |
| outputs/experiments/tav_main_v1/students/M1_seed42 | valid_acc2_nonzero | 15 |
| outputs/experiments/tav_main_v1/students/M3_seed13 | valid_acc2_nonzero | 5 |
| outputs/experiments/tav_main_v1/students/M3_seed2026 | valid_acc2_nonzero | 14 |
| outputs/experiments/tav_main_v1/students/M4_seed13 | valid_acc2_nonzero | 4 |
| outputs/experiments/tav_main_v1/students/M4_seed2026 | valid_acc2_nonzero | 12 |
| outputs/experiments/tav_main_v1/students/M4_seed42 | valid_acc2_nonzero | 8 |
| outputs/experiments/tav_main_v1/students/M6_seed13 | valid_acc2_nonzero | 5 |
| outputs/experiments/tav_main_v1/students/M6_seed2026 | valid_acc2_nonzero | 8 |
| outputs/experiments/tav_main_v1/students/M6_seed42 | valid_acc2_nonzero | 6 |
| outputs/experiments/video_source_v2/students/video_lora_seed13 | valid_acc2_nonzero | 5 |
| outputs/experiments/video_source_v2/students/video_lora_seed2026 | valid_acc2_nonzero | 14 |
| outputs/student/fullscale_ensemble_pair_seed13 | valid_acc2_nonzero | 15 |
| outputs/student/fullscale_ensemble_pair_seed2026 | valid_acc2_nonzero | 6 |
| outputs/student/fullscale_ensemble_pair_seed42 | valid_acc2_nonzero | 10 |
| outputs/student/fullscale_full_kd_seed13 | valid_acc2_nonzero | 3 |
| outputs/student/fullscale_full_kd_seed2026 | valid_acc2_nonzero | 12 |
| outputs/student/fullscale_full_kd_seed42 | valid_acc2_nonzero | 20 |
| outputs/student/fullscale_pair_snr_seed13 | valid_acc2_nonzero | 15 |
| outputs/student/fullscale_pair_snr_seed42 | valid_acc2_nonzero | 7 |
| outputs/student/fullscale_reliability_utility_pair_seed13 | valid_acc2_nonzero | 21 |
| outputs/student/fullscale_reliability_utility_pair_seed2026 | valid_acc2_nonzero | 8 |
| outputs/student/fullscale_reliability_utility_pair_seed42 | valid_acc2_nonzero | 20 |
| outputs/student/fullscale_selective50_interaction4_seed13 | valid_acc2_nonzero | 4 |
| outputs/student/fullscale_selective50_interaction4_seed2026 | valid_acc2_nonzero | 3 |
| outputs/student/fullscale_selective50_interaction4_seed42 | valid_acc2_nonzero | 10 |
| outputs/student/fullscale_student_seed13 | valid_acc2_nonzero | 4 |
| outputs/student/fullscale_student_seed2026 | valid_acc2_nonzero | 2 |
| outputs/student/fullscale_student_seed42 | valid_acc2_nonzero | 7 |
| outputs/student/fullscale_subset4_seed2026 | valid_acc2_nonzero | 28 |
| outputs/student/fullscale_subset4_seed42 | valid_acc2_nonzero | 1 |
| outputs/student/fullscale_utility_pair_seed13 | valid_acc2_nonzero | 10 |
| outputs/student/polarity_e1_seed13 | valid_mae | 11 |
| outputs/student/polarity_e1_seed13 | valid_acc2_nonzero | 3 |
| outputs/student/polarity_e1_seed42 | valid_mae | 5 |
| outputs/student/polarity_e1_seed42 | valid_acc2_nonzero | 1 |
| outputs/student/polarity_e2_seed13_detached | valid_mae | 8 |
| outputs/student/polarity_e2_seed13_detached | valid_acc2_nonzero | 4 |
| outputs/student/polarity_e2_seed42 | valid_mae | 12 |
| outputs/student/polarity_e2_seed42 | valid_acc2_nonzero | 10 |
| outputs/student/stage_d_c0_task_only_lora_seed13 | valid_acc2_nonzero | 1 |
| outputs/student/stage_d_c0_task_only_lora_seed2026 | valid_acc2_nonzero | 2 |
| outputs/student/stage_d_c2_ensemble_pair_lora_seed13 | valid_acc2_nonzero | 11 |
| outputs/student/stage_d_c2_ensemble_pair_lora_seed42 | valid_acc2_nonzero | 12 |
| outputs/student/stage_d_cminus1_online_frozen_full_kd_seed13 | valid_acc2_nonzero | 4 |
| outputs/student/stage_d_cminus1_online_frozen_full_kd_seed2026 | valid_acc2_nonzero | 13 |
| outputs/student/stage_d_cminus1_online_frozen_full_kd_seed42 | valid_acc2_nonzero | 17 |
| outputs/student/stage_d_cv2_c1_full_kd_lora_tempfix_seed13 | valid_acc2_nonzero | 1 |
| outputs/student/stage_d_cv2_c1_full_kd_lora_tempfix_seed42 | valid_acc2_nonzero | 14 |
| outputs/student/stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean_seed13 | valid_acc2_nonzero | 7 |
| outputs/student/stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean_seed42 | valid_acc2_nonzero | 5 |
| outputs/student/stage_d_cv2_cminus1_online_frozen_full_kd_tempfix_seed2026 | valid_acc2_nonzero | 22 |
| outputs/student/stage_d_cv2_cminus1_online_frozen_full_kd_tempfix_seed42 | valid_acc2_nonzero | 20 |
| outputs/student/stage_d_d1_full_kd_lora_seed13 | valid_acc2_nonzero | 5 |
| outputs/student/stage_d_d1_full_kd_lora_seed2026 | valid_acc2_nonzero | 14 |
| outputs/student/stage_d_d1_full_kd_lora_seed42 | valid_acc2_nonzero | 1 |
| outputs/student/stage_d_d2_ru_lora_seed13 | valid_acc2_nonzero | 4 |
| outputs/student/stage_d_d2_ru_lora_seed2026 | valid_acc2_nonzero | 8 |
| outputs/student/stage_d_d2_ru_lora_seed42 | valid_acc2_nonzero | 1 |
| outputs/student/stage_d_d3_full_kd_hidden_seed13 | valid_acc2_nonzero | 14 |
| outputs/student/stage_d_d3_full_kd_hidden_seed42 | valid_acc2_nonzero | 1 |
| outputs/student/video_adaptation_v1/frozen_video_seed13 | valid_acc2_nonzero | 11 |
| outputs/student/video_adaptation_v1/ta_only_seed13 | valid_acc2_nonzero | 6 |
| outputs/student/video_adaptation_v1/video_lora_seed13 | valid_acc2_nonzero | 15 |

## 核验与来源

- 可用权重均核对 payload epoch 与验证选中 epoch；有内嵌验证指标则比对 history。早期视频权重只有 model+epoch，额外要求它是该 run 原 best.pt，且 report.best_epoch 与 report.valid_metrics 一致。
- 冻结特征模型在 CPU 上严格加载全部参数；T/A LoRA 和视频模型在测评时核对可训练参数集合。测评不需要教师预测或测试标签构造训练目标。
- 原训练器和训练模型源码未修改；新增独立历史测评脚本。测试特征生成沿用原冻结特征脚本和模型，测评仅前向。
- 两种策略分别输出 selected_epoch、源 run、权重 SHA256、测试 manifest SHA256、指标和逐样本预测。
- [第一批队列](../../outputs/experiments/historical_test_completion_20261009/gpu0_queue/queue_status.json)
- [第二批队列](../../outputs/experiments/historical_test_completion_20261009/legacy_gpu0_queue/queue_status.json)
- [全部双策略汇总](all_experiments_dual_checkpoint_20261009/README.md)
- [无法补测清单 CSV](historical_test_completion_blocked_20261009.csv)
