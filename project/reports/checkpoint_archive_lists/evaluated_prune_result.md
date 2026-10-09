# 已评测历史 checkpoint 清理结果

执行日期：2026-10-04（Asia/Shanghai）。仅清理与成功评测 SHA256 匹配的历史 epoch。

已永久删除 506 个历史 checkpoint：389 个压缩归档、117 个未压缩权重。释放 145.56 GiB。
复核 188 个受保护文件全部存在；错误数 0；测评记录删除数 0。

保留 MAE/Acc2 验证最优、last、原有 root best 与所有已发现的 checkpoint 软链接目标。保留全部指标、逐样本预测、配置和训练历史。缺少完整评测证据或未完成的实验不删除。

删除的压缩包不能再恢复，原位置的 .pt.deleted.json / .pt.archive.json 标明删除状态和评测依据。

逐文件路径见 [CSV](evaluated_deleted_checkpoints.csv)，完整保护清单与证据见 [审计计划](evaluated_prune_plan.json)，执行结果见 [JSON](evaluated_prune_plan.result.json)。

| 训练目录（相对于 outputs/experiments） | 删除数量 | 释放 GiB |
|---|---:|---:|
| `interaction_evidence_v1/mosei/students/first_second_order_interaction_seed13` | 18 | 2.643 |
| `interaction_evidence_v1/mosei/students/random_orthogonal_seed13` | 13 | 1.908 |
| `interaction_evidence_v1/mosi/students/first_second_order_interaction_seed13` | 18 | 2.638 |
| `interaction_evidence_v1/mosi/students/random_orthogonal_seed13` | 18 | 2.638 |
| `multiseed_controls_v1/mosi/students/full_kd_seed2026` | 1 | 0.159 |
| `rdid_v2_mosei/students/r_only_seed13` | 21 | 9.273 |
| `rdid_v2_mosei/students/soft_ru_a025_seed13` | 18 | 7.953 |
| `rdid_v2_mosei/students/soft_ru_a050_seed13` | 17 | 7.508 |
| `rdid_v2_mosei/students/soft_ru_a075_seed13` | 11 | 4.852 |
| `rdid_v2_mosei/students/u_only_seed13` | 19 | 8.392 |
| `tav_epoch_checkpoints_v1/students/M0_seed13` | 17 | 2.482 |
| `tav_epoch_checkpoints_v1/students/M1_seed13` | 10 | 1.460 |
| `tav_epoch_checkpoints_v1/students/M3_seed13` | 18 | 2.643 |
| `tav_epoch_checkpoints_v1/students/M4_seed13` | 18 | 2.643 |
| `tav_epoch_checkpoints_v1/students/M6_seed13` | 16 | 2.349 |
| `uniform_followup_v1/mosei/students/adapted_student_seed13` | 19 | 3.023 |
| `uniform_followup_v1/mosei/students/ensemble_full_seed13` | 16 | 2.545 |
| `uniform_followup_v1/mosei/students/first_order_interaction_seed13` | 18 | 2.864 |
| `uniform_followup_v1/mosei/students/subset7_seed13` | 18 | 2.864 |
| `uniform_followup_v1/mosei_qiii_baselines_20260920/students/cmad_cafd_seed13` | 15 | 2.256 |
| `uniform_followup_v1/mosei_qiii_baselines_20260920/students/ea_kd_seed13` | 18 | 2.864 |
| `uniform_followup_v1/mosei_qiii_baselines_20260920/students/projector_seed13` | 18 | 2.710 |
| `uniform_main_v1/mosi/students/adapted_student_seed13` | 18 | 7.931 |
| `uniform_main_v1/mosi/students/cmad_cafd_seed13` | 17 | 7.678 |
| `uniform_main_v1/mosi/students/ea_kd_seed13` | 14 | 6.170 |
| `uniform_main_v1/mosi/students/ensemble_full_seed13` | 17 | 7.488 |
| `uniform_main_v1/mosi/students/first_order_interaction_seed13` | 17 | 7.490 |
| `uniform_main_v1/mosi/students/full_kd_seed13` | 17 | 7.491 |
| `uniform_main_v1/mosi/students/projector_seed13` | 17 | 7.666 |
| `uniform_main_v1/mosi/students/subset7_seed13` | 17 | 7.493 |
| `uniform_main_v1/mosi/students/uniform_interaction_seed13` | 17 | 7.491 |
