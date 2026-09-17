# RDID-MSA 文档导航

本页是 `docs/` 的统一入口。文档保持原位，因为多个重算脚本、测试和报告已引用它们的稳定路径；通过本页区分当前规范、冻结结果、专项审计和历史快照。

## 当前规范

1. [主表对比与消融实验方案（2026-09-17）](RDID-MSA_主表对比与消融实验方案_20260917.md)：主表、消融、公平性和统计协议的当前规范。
2. [实验实现说明（2026-09-17）](RDID-MSA_实验实现说明_20260917.md)：对应配置、资产、入口和已实现范围。
3. [TAV 主实验执行方案](TAV主实验执行方案_20260914.md)：冻结的 P0 训练协议、调度和 Gate。
4. [详细后续方案](后续方案.md)：TAV M0–M6 与视频诊断的设计推导；若与前三项冲突，以日期更新的冻结规范为准。

## 当前结果

- [TAV/P0 结果汇总（2026-09-17）](TAV主实验结果汇总_20260917.md)：最新冻结快照。
- [实验汇总报告](实验汇总报告.md)：跨 Stage 的结果、限制和复现入口。
- [Stage A 实验结果](RDID-MOSEI_StageA_实验结果.md)：Stage A v1/v2 的冻结候选与 Gate。

## 历史快照

以下文件保留当时的决策语境和数值，不代替上述当前结果：

- [TAV 主实验 2026-09-15 进展](TAV主实验进展汇总_20260915.md)
- [TAV 主实验 seed13 2026-09-16 结果](TAV主实验seed13结果汇总_20260916.md)
- [统一实验方案与论文收敛路线（2026-09-09）](RDID-MSA_统一实验方案与论文收敛路线_20260909.md)

## 项目与视频审计

- [项目全量复核](project_review_20260909.md) / [机器可读结果](project_review_20260909.json)
- [视频实验复核](video_review_20260909.md) / [机器可读结果](video_review_20260909.json) / [帧抽查](video_frame_spotcheck_20260909.json)
- [视频后续分析](video_followup_20260911.md) / [机器可读结果](video_followup_20260911.json)
- [教师视频源修正](teacher_video_source_fix_20260911.md) / [时序审计](teacher_video_timing_audit_20260911.json)
- [视频分辨率复核](video_resolution_review_20260914.md)
- [Video adaptation v1 结果](video_adaptation_v1_results.md) / [验证记录](video_adaptation_validation_v1.json)
- [Video source v2 执行说明](video_source_v2_execution.md) / [预检](video_source_v2_preflight.json) / [结果](video_source_v2_results.md)

## 文档与产物约定

- `docs/*.md` 记录方案、解释和人类可读结论。
- `docs/*.json` 保存专项审计的小型机器可读证据。
- `project/reports/` 保存由脚本重算的正式表格与统计，JSON 与 Markdown 职责不同，不互相删除。
- 实时状态只位于 `outputs/`；文档中的日期快照不覆盖写。
- [2026-09-17 工作区整理记录](WORKSPACE_CLEANUP_20260917.md)保存本次判定边界、删除规则和验证结果。
