# 四数据集 Phase 0 Transferability Audit

状态：已完成。仅 Phase 0；训练和 optimizer step 均为 0；未使用 Test 预测设计方法。

代码基线：`b33992e0d5654ce67d2cc7920566d3efa5140df6`。资产、checkpoint SHA256、原配置、路径修复证明和 split 指纹见 `protocol/asset_manifest.json`。本次新增脚本单独保存在 `source/`。

## 审计范围与协议

- 四数据集 × Full KD / Subset-7 / ICD × seeds 13、42、2026；按各历史 run 的 validation MAE 最低、同分较早 epoch 选权重。MAE-best 的指标整行报告，不拼接 Acc2-best。
- 梯度诊断覆盖四数据集、三方法的 seed13，共 12 个 checkpoint；每个 6 个固定 Train 窗口、microbatch1。Eval 模式关闭 dropout，保留 autograd；因此这是 checkpoint 局部诊断，不能解释为完整训练轨迹。
- 参数组：Fusion（含模态池化）、Text/Audio/Video LoRA、Prediction head；同时保留全模型结果。ICD 按七坐标和三阶数分解。
- Base = SmoothL1 task + 原协议 alpha_ce × CE + seed2026 TAV SmoothL1 KD + calibrated KL；ICD = 7 个 anchored interaction SmoothL1 均值，报告原始未乘 λ 梯度。
- 教师质量同时给出 seed2026 与三 Probe ensemble。原始 Full KD base 仅使用 seed2026，而 ICD/Subset-7 的额外目标来自三 Probe；这个历史差异不能当成纯几何效应。
- 三 Probe 共享同一教师 hidden cache，只是不同拟合 seed 的预测头；一致性不等于正确性，也不等于独立教师不确定性。
- 每个 Probe 的 anchor 为自身 Train TAV 预测均值；Student 使用三 Probe anchor 的均值。交互坐标为 A(f−anchor)，所有 anchor 均未使用 Validation/Test 估计。
- 历史训练预算并非完全统一，MOSEI 各 campaign 的 epoch 上限/早停等有差异；本报告的配对差值只作描述性审计，不能当成等预算确认实验。差异见 `tables/training_protocol_differences.csv`。MOSI seed13 历史配置曾标记 test_mae；本次从 history 重新按 validation MAE 选择具体 epoch，不沿用 Test-best。
- 四象限容差在查看预测前固定为原生标签范围宽度的 1/12：英语 0.5，SIMS 1/6；按绝对回归误差判定，另按标签强度分组。MOSEI 按实际准备清单的音/视频填充记录分“有填充/无填充”；这只是预处理属性，不等于语义质量。其余数据集无对应可靠字段，明确不提供质量分组。
- SIMS 原生 [-1,1] / 5 类、Acc2 含零；英语 [-3,3] / 7 类、主 Acc2 排除真值零。绝对 MAE 不跨数据集排名。
- Test 仅检查 split ID/source-video 元数据；没有新增 Test 预测导出、权重估计或超参数选择。历史 Test 已多轮接触，后续不得称为完全盲测。

| 数据集 | Train utterances | Valid utterances | Train/Valid 源视频重叠 | Train/Test 源视频重叠 | Valid/Test 源视频重叠 |
|---|---:|---:|---:|---:|---:|
| mosi | 1284 | 229 | 0 | 0 | 0 |
| mosei | 16326 | 1871 | 0 | 0 | 0 |
| sims | 1368 | 456 | 59 | 59 | 58 |
| sims_v2 | 2722 | 647 | 119 | 127 | 117 |

SIMS/SIMS v2 保留官方 utterance split，但它们并非 source-video 独立划分；不能把共享视频的 utterance 当作独立视频样本。另据本地 Train/Validation 清单，SIMS 的 1824 个 parent IDs 全部也出现在 SIMS v2，标签和 split 一致；这两个版本不能视为独立 benchmark 重复。英语 manifest 的历史指纹差异已证明仅为 `/home/wy/sjq/kd` → `/data/sjq/kd` 路径替换，反向替换精确恢复原 SHA256。历史文件未改写。

## Teacher 与 Student 验证集指标

| 数据集 | 方法 | 完成 seeds | Student MAE ± SD | Teacher ensemble MAE | Student − Teacher |
|---|---|---:|---:|---:|
| mosi | full_kd | 13,42,2026 | 0.7523 ± 0.0179 | 0.4889 | 0.2634 |
| mosi | subset7 | 13,42,2026 | 0.7692 ± 0.0146 | 0.4889 | 0.2802 |
| mosi | uniform_interaction | 13,42,2026 | 0.7691 ± 0.0098 | 0.4889 | 0.2802 |
| mosei | full_kd | 13,42,2026 | 0.4673 ± 0.0044 | 0.4339 | 0.0333 |
| mosei | subset7 | 13,42,2026 | 0.4655 ± 0.0041 | 0.4339 | 0.0315 |
| mosei | uniform_interaction | 13,42,2026 | 0.4698 ± 0.0064 | 0.4339 | 0.0358 |
| sims | full_kd | 13,42,2026 | 0.4127 ± 0.0135 | 0.2700 | 0.1427 |
| sims | subset7 | 13,42,2026 | 0.4103 ± 0.0106 | 0.2700 | 0.1404 |
| sims | uniform_interaction | 13,42,2026 | 0.4038 ± 0.0069 | 0.2700 | 0.1338 |
| sims_v2 | full_kd | 13,42,2026 | 0.3782 ± 0.0134 | 0.2502 | 0.1280 |
| sims_v2 | subset7 | 13,42,2026 | 0.3668 ± 0.0021 | 0.2502 | 0.1166 |
| sims_v2 | uniform_interaction | 13,42,2026 | 0.3623 ± 0.0026 | 0.2502 | 0.1121 |

所有 seed 的 MAE、Acc2、Pearson、F1、checkpoint hash、Teacher fidelity 与标签任务指标保存在 `tables/student_task_quality.csv`、`student_absorbability.csv`。`checkpoint_validation_metrics.csv` 保留历史全部 epoch 的 validation 指标。

交互坐标归一化定义为 MAE(I_S,I_T) / Std_Train(I_T)，按坐标使用固定教师 Train 统计，分母下限 1e-8；阶数指标在该阶的坐标内取均值。标准差很小会放大归一化误差，因此必须同时审阅原始 MAE 和该分母。跨 Probe 方差及跨 seed SD 均采用 ddof=1。

## 当前证据与决策门

- mosi：ICD 验证集一/二/三阶归一化 MAE 为 0.6382 / 0.7173 / 0.9873。该误差衡量吸收，不直接代表坐标的任务效用。
- mosi：ICD seed13 的固定 Train 样本全参数 mean cosine(base, ICD) = 0.4066，负余弦比例 0.3333，raw gradient norm ratio = 0.9810；层组与坐标明细应优先用于定位，六个窗口不足以证明普遍训练冲突。
- mosei：ICD 验证集一/二/三阶归一化 MAE 为 0.3659 / 0.3935 / 0.4839。该误差衡量吸收，不直接代表坐标的任务效用。
- mosei：ICD seed13 的固定 Train 样本全参数 mean cosine(base, ICD) = 0.8100，负余弦比例 0.0000，raw gradient norm ratio = 0.1642；层组与坐标明细应优先用于定位，六个窗口不足以证明普遍训练冲突。
- sims：ICD 验证集一/二/三阶归一化 MAE 为 0.6912 / 0.6762 / 0.5984。该误差衡量吸收，不直接代表坐标的任务效用。
- sims：ICD seed13 的固定 Train 样本全参数 mean cosine(base, ICD) = -0.2570，负余弦比例 0.6667，raw gradient norm ratio = 0.9204；层组与坐标明细应优先用于定位，六个窗口不足以证明普遍训练冲突。
- sims_v2：ICD 验证集一/二/三阶归一化 MAE 为 0.6421 / 0.6078 / 0.5006。该误差衡量吸收，不直接代表坐标的任务效用。
- sims_v2：ICD seed13 的固定 Train 样本全参数 mean cosine(base, ICD) = 0.0667，负余弦比例 0.3333，raw gradient norm ratio = 0.7332；层组与坐标明细应优先用于定位，六个窗口不足以证明普遍训练冲突。

当前保持 Phase 1–5 关闭。Phase 0 只提供描述性机制证据：可靠性差异需结合任务质量，梯度范数与方向应分别判断，原始/归一化阶数误差应同时审阅。若教师拟合普遍差，先考虑表示/数据/容量；若 fidelity 高但任务差，再分析错误传递。不得根据本次 Test 元数据或历史 Test 排名调权重。

## 完成度与验证

- 七子集 Train/Validation 导出：72/72。梯度 checkpoint：12/12。
- 已完成验证集回放 36/36，MAE 最大绝对偏差 0.0056。容许 0.02 用于检测迁移后模型/权重不兼容；bf16 和不同 batch size 可导致小幅数值差异，详见 `validation_replay.csv`。
- 对每个导出断言样本 ID 完全覆盖、无重复、真实标签一致、预测有限；断言 (I_S − I_T) = A(f_S − f_T)，教师 ensemble / anchor / interaction mean 和 variance 可精确重建。
- 四数据集教师 hidden 缓存均逐字节匹配冻结 SHA256；12 个 Probe checkpoint 另在固定 16 Train + 16 Valid parents 的全部七子集、全部窗口上做 CPU 回放，误差须小于 1e-4。
- `checkpoint_inventory_audit.csv` 对审计 run 实际保留的每个权重文件计算当前 SHA256 并关联其 validation 指标；`checkpoint_history_coverage.csv` 区分历史 epoch 是否仍有权重。旧归档 SHA 另列，优化器剥离/重新序列化可能改变文件 SHA，不能将旧 SHA 当成当前验证。
- `paired_bootstrap.csv` 与 `paired_method_differences.csv` 使用 source-video cluster bootstrap（2000 次，固定 seed130013），保留每个源视频所有 utterances。区间仅为历史 checkpoint 的描述性不确定性，不修复跨 split 的视频重叠，也不替代跨训练 seed 确认。
- 每个诊断只用 autograd.grad；断言模型参数 version 未变，.grad 未填充；checkpoint 文件 size/mtime 未变。
- 原生 SIMS 导出初次因遗漏 class_5_index 字段失败，已修复并通过原生加权窗口聚合校验；失败日志与重试记录保留在 `protocol/native_export_schema_failed_attempts.json`，不把失败导出算作完成。
- 局限：只有一条 validation-selected 轨迹/seed；梯度只用 seed13 的少量 Train 窗口；无额外 Probe 重训；无新增 Test 因果比较；没有自动进入大规模训练。

## 产物

- `tables/protocol_audit.csv`、`protocol/asset_manifest.json`、`protocol/split_fingerprints.json`
- `tables/teacher_subset_quality.csv`、`teacher_reliability_by_subset.csv`、`probe_agreement.csv`、`probe_agreement_by_coordinate.csv`
- `tables/student_absorbability.csv`、`interaction_coordinate_error.csv`、`interaction_order_error.csv`、`sample_quadrants.csv`
- `tables/gradient_compatibility.csv`、`student_task_quality.csv`、`seed_aggregates.csv`、`validation_replay.csv`、`coverage.csv`
- `figures/`：Teacher MAE/Pearson 热图、交互阶数误差、梯度兼容性、Teacher–Student gap。
- `predictions/`、`gradients/`：逐样本原始诊断；`source/`、`logs/`、`status/`：脚本、运行证据和完成状态。
