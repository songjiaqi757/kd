# KD 工作区整理记录（2026-10-02）

> 历史归档：原路径 `docs/WORKSPACE_CLEANUP_20261002.md`。本文保留当时的方案、结果与判断；其中“当前”“唯一口径”和运行状态均指原记录日期。现行论文口径见 [ICD 提交版](../../../ICD/ICD.pdf) 与 [ICD 实验结果总汇](../../paper/icd_experiment_summary.md)。

## 范围与批准

本次整理覆盖 `/home/wy/sjq/kd` 的全部非 Git 文件。操作前约有 144,151 个文件，数据、模型和实验产物约占 536 GiB。用户在查看分组方案后明确批准 A（系统垃圾与缓存）、B（已替代产物）和 C（文档合并与统一命名）。

`ICD/ICD.pdf` 被确定为提交版本。最终方法名称统一为 **Interaction Coordinate Distillation (ICD)**；`RDID`、`TAV` 和 `Uniform Interaction` 仅作为历史实验阶段或内部运行名称保留。

## 全量检查

- 逐路径检查了文件类型、大小、目录归属、空文件、符号链接和重复关系。
- 8,583 个 JSON、978 个 JSONL（5,189,547 行）和 180 个 YAML 完成首轮解析。唯一异常是 2,769 个伪装成 JSON 的 macOS 资源叉，以及第三方项目使用自定义 `!join` 标签的 YAML。
- MOSEI 的 45,712 个媒体文件和 MOSI 的全部媒体文件通过现有 SHA-256 清单复核。
- 44 个 PDF、1 个 PPTX、6 个 XLSX、20,101 个 PyTorch 权重、195 个 NPZ、12 个 NPY 和 94 张图片通过格式或容器结构检查。
- 853 个 Python 文件和 25 个 Shell 文件通过语法检查；30 个符号链接均有效。
- 整理前测试为 174 passed，20 条既有 PyTorch warning。

## 文档收敛

统一后的稳定入口如下：

| 实验或用途 | 统一文档 |
|---|---|
| TAV/P0 主实验 | `docs/experiments/tav_main_v1.md` |
| 视频链路与适配 | `docs/experiments/video_source_v2.md` |
| RDID-v2 方法探索 | `docs/experiments/rdid_v2_mosei.md` |
| MOSI Uniform/ICD 主实验 | `docs/experiments/uniform_main_v1.md` |
| Uniform/ICD 后续实验 | `docs/experiments/uniform_followup_v1.md` |
| 多种子对照 | `docs/experiments/multiseed_controls_v1.md` |
| ICD 结果总汇 | `docs/paper/icd_experiment_summary.md` |
| ICD 补充口径 | `docs/paper/icd_supplementary_notes.md` |

共 20 份旧文档被完整并入 8 份统一记录。每份原文保留文件名、日期、标题和全部正文，没有用摘要替代独有内容。两份 `docs/全部实验数据总账_*.md` 与对应 `outputs/experiment_catalog_*/README.md` 逐字节相同，因此只保留 catalog 内的副本。

`project/scripts/build_all_experiment_data_catalog.py` 不再默认向 `docs/` 复制第二份总账。Video source v2 的可重生成简表改写到 `docs/experiments/video_source_v2_results.generated.md`，不会覆盖统一记录。

## 已删除

- `dataset/cmu_mosei_source/Metadata/__MACOSX/` 中 2,772 个 macOS 资源叉文件。
- `ICD/.DS_Store`、Python/pytest 缓存、陈旧 `.lock`/`.pid` 和空日志。
- `project/reports/paper_figures/provisional_aggregate_20260923/` 和已被正式结果替代的 provisional 报告。
- RDID-v2 的两个失败启动目录和一个按用户要求停止的 batch-6 目录。
- TAV preflight、Video source v2 preflight 和 student smoke 临时目录。
- 三个已确认无内容的输出目录。

白名单清理脚本报告删除 3,398 个文件、8,988,234 字节。失败原因、修复情况和正式替代关系已写入对应统一实验记录。大型数据、模型、正式 checkpoint、预测、报告和 `ICD` 提交文件均保留。

## 故意保留的重复

`ICD/` 中三张提交图与 `project/reports/paper_figures/` 的最终图源逐字节相同。前者属于提交包，后者属于可重算图源，两者职责不同，故不去重。

两个 `outputs/experiment_catalog_*` 是不同日期的机器可读快照，也同时保留。

## 整理后验证

- 5,722 个 JSON、969 个 JSONL（5,189,444 行）重新解析，错误数为 0。
- Markdown 相对链接 100 个，断链数为 0。
- 符号链接损坏数为 0；已批准的缓存、锁、PID 和空日志残留数为 0。
- 全量测试 174 passed，20 条 warning 与整理前一致。
- `ICD/` 保留 `ICD.pdf`、四张 PDF 图、`framework.pptx`，仅移除 `.DS_Store`。

## Docs 二次收敛（2026-10-02）

用户在查看逐项检查结论后批准“保留核心入口、合并两份主表文档、归档历史、集中审计、删除测试残留”。本次仅处理文档组织及其引用。

- `docs/` 文件由 28 个收敛为 26 个；根目录仅保留 `README.md`。
- 两份主表方案与实现说明完整并入 [main_table_v1](../../experiments/main_table_v1.md)，保留原文件名与分节锚点，仅调整链接。
- 四份历史方案移入 `archive/plans/`，三份历史报告移入 `archive/reports/`，两份整理记录移入 `archive/cleanup/`。九份历史记录增加归档标记，正文与历史结论保留。
- 七份审计 JSON 移入 `audit/`，逐文件 SHA-256 与迁移前一致。历史来源字段与输入指纹不改写。
- 删除 `docs/experiments/video_source_v2_results.generated.md` 的测试残留：三个 seed 均为 MAE 0.5、Pearson 0.8、Acc-2 0.9，来源指向 `/tmp/pytest-of-sjq/pytest-54/`。真实结果在 [video_source_v2](../../experiments/video_source_v2.md) 及原始实验 JSON 中完整保留；生成入口继续可用。
- 根导航与 `docs/README.md` 更新至新路径；`workspace.yaml` 的方案入口改为统一主表文档。
- 审计脚本与 Stage A 汇总脚本的输出路径同步迁移；实验数据总账改为递归扫描 `docs/**/*.json`，原有 catalog 冻结快照保留。

迁移映射：

| 原路径 | 现路径 |
|---|---|
| `docs/experiment方案.md` | `docs/archive/plans/experiment方案.md` |
| `docs/后续方案.md` | `docs/archive/plans/后续方案.md` |
| `docs/RDID-MSA_统一实验方案与论文收敛路线_20260909.md` | `docs/archive/plans/RDID-MSA_统一实验方案与论文收敛路线_20260909.md` |
| `docs/RDID-MSA_Student_Absorbability_实验方案_20260918.md` | `docs/archive/plans/RDID-MSA_Student_Absorbability_实验方案_20260918.md` |
| `docs/实验汇总报告.md` | `docs/archive/reports/实验汇总报告.md` |
| `docs/RDID-MOSEI_StageA_实验结果.md` | `docs/archive/reports/RDID-MOSEI_StageA_实验结果.md` |
| `docs/project_review_20260909.md` | `docs/archive/reports/project_review_20260909.md` |
| `docs/WORKSPACE_CLEANUP_20260917.md` | `docs/archive/cleanup/WORKSPACE_CLEANUP_20260917.md` |
| `docs/WORKSPACE_CLEANUP_20261002.md` | `docs/archive/cleanup/WORKSPACE_CLEANUP_20261002.md` |
| `docs/video_source_v2_preflight.json` | `docs/audit/video_source_v2_preflight.json` |
| `docs/video_followup_20260911.json` | `docs/audit/video_followup_20260911.json` |
| `docs/project_review_20260909.json` | `docs/audit/project_review_20260909.json` |
| `docs/teacher_video_timing_audit_20260911.json` | `docs/audit/teacher_video_timing_audit_20260911.json` |
| `docs/video_adaptation_validation_v1.json` | `docs/audit/video_adaptation_validation_v1.json` |
| `docs/video_review_20260909.json` | `docs/audit/video_review_20260909.json` |
| `docs/video_frame_spotcheck_20260909.json` | `docs/audit/video_frame_spotcheck_20260909.json` |

### 本次验证

- 七份审计 JSON 的 SHA-256 与迁移前逐一相同；两份合并原文完整保留，仅重定位链接。
- 两份论文依据文档及未涉及本次整理的实验记录校验和不变。
- 根导航、项目 README 和 Docs 内共 123 个本地链接有效，合并分节锚点有效；所有审计 JSON 可解析。
- 四个修改脚本通过 Python 语法检查，`workspace.yaml` 可解析且新方案入口存在。
- 总账迁移回归测试、Video source v2 和 teacher video 相关测试共 24 passed；测试未重新生成临时报告。
