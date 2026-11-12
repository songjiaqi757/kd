# ICD 文档导航

论文最终方法名称为 **Interaction Coordinate Distillation (ICD)**。 `RDID`、`TAV` 和 `Uniform Interaction` 保留为历史阶段及内部运行名称。

## 提交版本与结果依据

- [ICD 提交版 PDF](../ICD/ICD.pdf)：论文术语、实验口径和主张的最终依据。
- [ICD 实验结果总汇](paper/icd_experiment_summary.md)：正式结果及明确分开的历史诊断快照。
- [ICD 补充说明](paper/icd_supplementary_notes.md)：定性选择、训练成本、Teacher Probe 和 ensemble control。

## 汇总报告

- [实验汇总报告（2026-11-11 全量快照）](reports/实验汇总报告_20261111.md)：截至 2026-11-11 的全项目实验矩阵；合并 11-07 全量报告、11-08 MOSI 第一轮，以及 11-09 至 11-11 的确认性实验、A 系列、方法搜索、EA-N2 迁移、MOSI First 18 组几何矩阵与 SIMS v2 N2 结果。
- [MOSEI N2 / EA-N2 结果汇总（2026-11-12）](reports/MOSEI_N2_EA-N2_结果汇总_20261112.md)：补齐 EA-N2 迁移 campaign 的 MOSEI 结果；单 seed 探索性，不改变已确认结论。
- [MOSI First 复现包（2026-11-11）](../project/reports/mosi_first_20261111_v1/README.md)：几何矩阵公式、`diagonal_mse` 完整损失、matched-spectrum 校验、18 方法完整配置、54 checkpoint 指标与 SHA256。
- [四数据集 Transferability Audit（2026-11）](../project/reports/transferability_audit_202611/four_dataset_transferability_audit.md)：Phase 0 教师/学生可靠性、交互吸收与梯度兼容性审计；仅描述性证据，未启动 Phase 1–5。
- [实验汇总报告（2026-11-07 快照）](reports/实验汇总报告_20261107.md)：全项目实验矩阵、主结果、机制分析、限制与复现索引（数据截止 2026-10-30）。
- [MOSI ICD 第一轮实验汇总报告（2026-11-08）](reports/MOSI_ICD_第一轮实验汇总报告_20261108.md)：E0–E8 seed13 初筛与 Full KD 对照，共 12 个 validation run；未运行本轮 official-test。

## 统一实验记录

- [main_table_v1](experiments/main_table_v1.md)：主表对比方案、实现范围与复现命令。
- [tav_main_v1](experiments/tav_main_v1.md)：P0/TAV 方案、进展与三 seed 结果。
- [video_source_v2](experiments/video_source_v2.md)：视频链路审计、修正、适配与结果。
- [rdid_v2_mosei](experiments/rdid_v2_mosei.md)：RDID-v2 方法探索与代码落实。
- [uniform_main_v1](experiments/uniform_main_v1.md)：MOSI Uniform/ICD 主实验方案与执行记录。
- [uniform_followup_v1](experiments/uniform_followup_v1.md)：Uniform/ICD 后续对照方案。
- [multiseed_controls_v1](experiments/multiseed_controls_v1.md)：MOSEI/MOSI 多种子对照记录。

实验名称与 `outputs/experiments/<experiment_id>/` 对应。统一文档中的原文快照保留其日期和当时的协议，不能将历史计划或 test-best 诊断当作当前提交结果。

## 审计证据

- [项目复核 JSON](audit/project_review_20260909.json)
- [视频复核 JSON](audit/video_review_20260909.json)
- [25 条训练视频帧抽查](audit/video_frame_spotcheck_20260909.json)
- [视频后续统计](audit/video_followup_20260911.json)
- [教师视频时序审计](audit/teacher_video_timing_audit_20260911.json)
- [Video adaptation 验证](audit/video_adaptation_validation_v1.json)
- [Video source v2 预检](audit/video_source_v2_preflight.json)

JSON 保留当时的机器可读证据和指纹；其中运行状态仅代表记录日期。

## 历史归档

以下文件用于追溯决策，不作为当前执行指令。

- [实验补充内容方案](archive/plans/experiment方案.md)
- [早期 TAV 后续方案](archive/plans/后续方案.md)
- [统一实验与论文收敛路线（2026-09-09）](archive/plans/RDID-MSA_统一实验方案与论文收敛路线_20260909.md)
- [Student Absorbability 方案（2026-09-18）](archive/plans/RDID-MSA_Student_Absorbability_实验方案_20260918.md)
- [跨 Stage 历史实验报告](archive/reports/实验汇总报告.md)
- [Stage A 历史结果](archive/reports/RDID-MOSEI_StageA_实验结果.md)
- [项目复核说明（2026-09-09）](archive/reports/project_review_20260909.md)
- [工作区整理记录（2026-09-17）](archive/cleanup/WORKSPACE_CLEANUP_20260917.md)
- [工作区与 Docs 整理记录（2026-10-02）](archive/cleanup/WORKSPACE_CLEANUP_20261002.md)

## 产物与版本

- `paper/`：与 ICD 提交版对应的结果依据和补充口径。
- `experiments/`：按稳定实验 ID 组织的完整记录。
- `audit/`：机器可读审计证据；`archive/`：历史方案、报告与整理记录。
- `project/reports/`：脚本生成的表格、统计和论文图源。
- `outputs/`：checkpoint、预测、日志、状态与冻结实验产物。
- [2026-09-21 全量实验 catalog](../outputs/experiment_catalog_20260921/README.md)
- [2026-09-23 全量实验 catalog](../outputs/experiment_catalog_20260923/README.md)

`ICD/` 是提交包。Video source v2 的生成简表可由脚本重建，真实结果以统一记录和原始 JSON 为依据。
