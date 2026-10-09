# ICD 文档导航

论文最终方法名称为 **Interaction Coordinate Distillation (ICD)**。 `RDID`、`TAV` 和 `Uniform Interaction` 保留为历史阶段及内部运行名称。

## 提交版本与结果依据

- [ICD 提交版 PDF](../ICD/ICD.pdf)：论文术语、实验口径和主张的最终依据。
- [ICD 实验结果总汇](paper/icd_experiment_summary.md)：正式结果及明确分开的历史诊断快照。
- [ICD 补充说明](paper/icd_supplementary_notes.md)：定性选择、训练成本、Teacher Probe 和 ensemble control。

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
