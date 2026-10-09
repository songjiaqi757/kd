# `main_table_v1` 统一实验记录

> 文档状态：主表对比与消融实验的设计方案和实现说明。本文完整收录 2026-09-17 两份原文；计划项不代表已经完成，最终提交口径以 [ICD.pdf](../../ICD/ICD.pdf) 和 [ICD 结果总汇](../paper/icd_experiment_summary.md) 为准。

- [实验设计方案](#experiment-plan)
- [实现说明与复现命令](#implementation-notes)

<a id="experiment-plan"></a>

## 原文快照：`RDID-MSA_主表对比与消融实验方案_20260917.md`

# RDID-MSA 主表对比与消融实验方案

编制日期：2026-09-17。适用项目：`/home/wy/sjq/kd`。

本文整理本轮项目核查、2024—2026 年论文检索及最终讨论意见，是后续实验设计文档。文中的新增方法均为计划项，不代表已经实现、训练或取得结果。本文不修改当前冻结的 P0 训练协议；新增实验在独立配置及输出目录中实施。

本方案作为总体路线使用。新增外部方法的实现适配、调参预算和数据协议完成核查后，再冻结相应执行配置；不把论文名称清单直接视为已经可运行的实验协议。

## 1. 研究目标与总体安排

论文需要检验的核心命题是：**在相同学生架构和适配能力下，选择性迁移交互知识，能否比普通输出蒸馏、特征蒸馏及其他知识选择方法带来稳定收益。**

最终采用两张主表和一组机制消融：

| 实验部分 | 回答的问题 | 主要设置 |
|---|---|---|
| 主表 A：固定学生的蒸馏比较 | 收益是否来自蒸馏方法本身？ | 相同学生、输入、适配范围和评测协议；比较 4 种近年已发表 KD 方法及关键参照 |
| 主表 B：完整 MSA 系统比较 | 完整系统相对近年情感分析方法表现如何？ | 复跑 DLF、GsiT、DPDF-LQ，保留各自原生架构，统一评测 |
| 机制消融 | 收益来自哪些因素，是否存在更简单的解释？ | 控制七子集监督、Probe 集成、交互变换及 R/U 权重 |
| 效率与统计分析 | 收益是否稳定，训练与部署代价如何？ | 三种子、原视频聚类配对分析、参数和耗时报告 |

两张主表均覆盖 CMU-MOSI 和 CMU-MOSEI。最终完整 T/A/V 输入是默认测试场景；缺失模态鲁棒性作为有明确研究主张时才启动的扩展。

**自建控制实验不能代替已发表方法，但“属于消融”不构成排除它进入主表的充分理由。** Adapted Student-only 是必要参照；Subset-7 直接控制额外教师监督；M4 直接检验均匀迁移与选择性迁移的差异，三者均进入主表 A。Ensemble Full KD 列为关键必做控制，结果放入机制消融。

## 2. 项目现状与比较边界

当前项目采用冻结的大模型教师特征及任务 Probe，为学生提供情感回归分数和七分类 logits。学生使用 Qwen3-0.6B-Base、WavLM-Base-Plus、VideoMAE-Base 及融合网络。

| 现有方法 | 适配范围与损失 | 本方案中的用途 |
|---|---|---|
| M0 | 编码器冻结，仅任务损失 | 适配相关消融；不能直接充当 M6 的同能力无 KD 对照 |
| M1 | 编码器冻结，任务损失＋Full KD | 适配相关消融 |
| M3 | T/A/V 适配，任务损失＋Full KD | 主表 A 的普通蒸馏参照 |
| M4 | M3＋均匀交互蒸馏 | 主表 A 的直接机制参照，同时用于交互变换及权重消融 |
| M5 | M3＋R-only 交互蒸馏 | 可靠性权重消融，完成状态另行核实 |
| M6 | M3＋R×U 交互蒸馏 | Ours |

M3 的 Full KD 使用单个 Probe；M4/M5/M6 的交互目标使用同一教师特征上的三个 Probe，种子为 2026、2027、2028。三个 Probe 不等于三个独立的大模型教师。

代码依据：[适配范围与交互损失](../../project/src/rdid_mosei/tav_distillation.py)、[训练损失与评测入口](../../project/scripts/train_tav_distillation.py)、[学生融合表示](../../project/src/rdid_mosei/student.py)、[教师资产与 U 的构造](../../project/scripts/prepare_tav_protocol.py)。

后续新增的 Adapted Student-only 必须使用与 M6 相同的 T/A/V 适配范围。不能仅将现有 M0 改名后填入主表。

## 3. 主表 A：固定学生的蒸馏方法比较

### 3.1 最低充分集拟纳入的 8 行

下列编号仅用于本文组织实验，不代表代码中已经存在相应 method 标识。

| 编号 | 方法 | 额外监督或方法特点 | 实验角色 |
|---|---|---|---|
| A0 | Adapted Student-only | 与 M6 相同适配能力，仅任务损失 | 无蒸馏参照 |
| A1 | Full KD | 现有 M3 | 普通输出蒸馏参照 |
| A2 | Projector-based feature KD，AAAI 2024 | 投影、归一化及 LogSum 特征蒸馏 | 已发表特征蒸馏方法 |
| A3 | EA-KD，ICCV 2025 | 根据教师和学生熵进行样本价值加权 | 已发表自适应加权方法 |
| A4 | CMAD-style CAFD | 逐样本特征及跨样本相关结构蒸馏 | 领域相关 KD 组件适配 |
| A5 | Subset-7 KD | 公共 Full KD＋七子集预测直接匹配 | 同七子集教师监督参照 |
| A6 | Uniform Interaction | 现有 M4，与 M6 使用相同交互目标 | 均匀交互蒸馏参照 |
| A7 | Ours | 现有 M6 | 交互坐标上的 R×U 加权 |

最低充分集的外部方法覆盖通用特征迁移、知识价值加权及领域相关特征蒸馏。RLD 与 SKD
已经完成适配实现，但归入 Tier 3，不占最低充分集名额；不能根据测试成绩决定是否把它们
移回主表。

M4 同时出现在主表和消融表，复用同一协议下的结果，不重复训练。外部方法清单冻结前，完成第 11 节所列领域相关 KD 的适配核查；不能仅因任务名称相近就直接替换，也不能只比较通用 CV KD 而不讨论直接相关的 MER/MSA 蒸馏工作。

### 3.2 已发表方法、论文与官方代码

| 方法 | 论文 | 官方代码 | 适配要点 |
|---|---|---|---|
| Projector-based feature KD | [Understanding the Role of the Projector in Knowledge Distillation，AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28219) | [Simple-Recipe-Distillation](https://github.com/roymiles/Simple-Recipe-Distillation) | 学生融合表示经投影对齐缓存教师表示，保留线性投影、归一化和 LogSum 距离等核心设计；本文名称是描述性简称 |
| EA-KD | [EA-KD: Entropy-based Adaptive Knowledge Distillation，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Su_EA-KD_Entropy-based_Adaptive_Knowledge_Distillation_ICCV_2025_paper.html) | [EA-KD](https://github.com/cpsu00/EA-KD) | 用七分类预测构造教师、学生熵，并明确权重作用的损失项 |
| CMAD-style CAFD | [CMAD，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf) | [CMAD](https://github.com/YetZzzzzz/CMAD) | 仅适配 CAFD：保留逐样本特征 MSE、跨样本相关矩阵及双向 KL；不包含 MAR，不称为完整 CMAD |
| RLD（Tier 3） | [Knowledge Distillation with Refined Logits，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Knowledge_Distillation_with_Refined_Logits_ICCV_2025_paper.html) | [RLD](https://github.com/zju-SWJ/RLD) | 已实现完整核心损失及必要训练调度，资源充足时补充 |
| SKD（Tier 3） | [Streamlined Knowledge Distillation，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jeong_Streamlined_Knowledge_Distillation_CVPR_2026_paper.html) | [StreamLined](https://github.com/HyunJunSik/StreamLined) | 已实现实例知识与方向知识，资源充足时补充 |

### 3.3 公共损失与公平适配

记任务损失为 `L_task`，完整 TAV 的回归蒸馏为 `L_reg`，分类 KL 蒸馏为 `L_KL`，公共基线为：

```text
L_base = L_task + L_reg + L_KL
```

现有任务损失包含回归监督和系数为 0.5 的七分类 CE。正式实验沿用冻结协议或统一另建协议，不能只对 Ours 调整公共部分。

| 方法类型 | 建议损失组织 |
|---|---|
| Adapted Student-only | `L_task` |
| Full KD | `L_base` |
| RLD、SKD | 保留公共 `L_task + L_reg`，用相应方法替换分类蒸馏部分 |
| EA-KD 主表版本 | 保留公共 `L_task + L_reg`，对分类蒸馏使用原方法的样本加权 |
| Projector 特征法 | `L_base + λ_feature × L_feature` |
| CMAD-style CAFD | `L_base + λ_CAFD × L_CAFD` |
| Subset-7 | `L_base + λ_subset × L_subset7` |
| Uniform Interaction | `L_base + λ_interaction × L_interaction_uniform` |
| Ours | `L_base + λ_interaction × L_interaction_RU` |

另外补充 **EA-KD-Full（MSA 适配变体）**：利用相同熵权重同时加权 `L_reg + L_KL`，检验普通样本加权能否解释回归收益。该变体放在扩展比较中，明确标为本项目适配，不冒充原论文原始实验。

所有外部方法都标明“适配到统一 MSA 框架”。不能给外部方法只保留七分类蒸馏，而给 Ours 额外提供连续回归蒸馏后，仅以 MAE 判断优劣。

固定学生不等于教师信息量完全相同。主表增加 `Teacher subsets`、`Probe count` 等列或表注：普通输出 KD 使用 TAV、单 Probe；Subset-7 与 Ours 的额外监督使用七子集及三个 Probe。特征法还应注明使用教师隐藏表示。利用 A6 和机制消融进行信息来源归因。

## 4. Subset-7 KD 的具体定义

Subset-7 是本文提出的自建基线名称，不是某篇已发表论文的方法名。

### 4.1 七种模态组合

令 T、A、V 分别为文本、音频、视频，七个非空子集为：

```text
S = {T, A, V, TA, TV, AV, TAV}
```

对每个样本，学生在每种组合下的情感回归预测，分别匹配教师在同一组合下的预测。教师目标使用与 M6 相同的三个 Probe 的平均分数。

记学生分数为 `f_s(S)`，三个 Probe 的平均教师分数为 `mean_f_t(S)`，则：

```text
L_subset7 = (1 / 7) × Σ_S SmoothL1(f_s(S), mean_f_t(S))
L_Subset7_KD = L_base + λ_subset × L_subset7
```

公式省略了 batch 和窗口样本权重；实现沿用项目既有样本加权、窗口目标对应及 utterance 聚合规则。额外七子集项包括 TAV，因此完整 TAV 同时受到公共 Full KD 和额外项约束；这与 Ours 保留公共 Full KD 后添加七坐标监督的设计对应，并非遗漏去重。

### 4.2 与 M4、M6 的区别

Subset-7 匹配原始预测分数。交互方法先进行 Möbius 变换，例如：

```text
I(T)   = f(T) - b
I(TA)  = f(TA) - f(T) - f(A) + b
I(TAV) = f(TAV) - f(TA) - f(TV) - f(AV) + f(T) + f(A) + f(V) - b
```

`b` 使用项目当前定义的基线：每个 Probe 在 official train 上的完整 TAV 平均预测；学生使用这些基线的平均值。它不是实测的空输入预测，论文需如实说明。

| 方法 | 额外匹配的目标 | 坐标权重 |
|---|---|---|
| Subset-7 | 七个子集的原始回归预测 | 均匀 |
| M4 | 七个交互坐标，包含一阶项 | 均匀 |
| M6 | 与 M4 相同的交互坐标 | R×U |

给定基线，七个原始分数与七个交互坐标之间可以相互转换。变换本身不增加教师信息，改变的是监督组织和误差约束。因此，Subset-7 与 M4 检验交互坐标的作用；M4 与 M6 检验选择性加权的作用。

### 4.3 训练与推理

学生每个样本只需执行一次 T/A/V 编码，七种融合组合复用编码表示，使用与现有子集融合一致的模态屏蔽方式。最终测试只使用完整 TAV，不在推理时集成七个预测。

首次实现需核对七子集顺序、三个 Probe 的身份、目标均值、样本对应关系、损失平均方式和总权重，确保与 M6 的教师资产可比。

## 5. 主表 B：完整多模态情感分析系统比较

主表 B 纳入以下三个近年系统及 Ours。它回答完整系统的任务表现，不用于单独证明蒸馏机制的因果贡献。

| 方法 | 发表信息与论文 | 官方代码 | 选择理由 |
|---|---|---|---|
| DLF | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/35416) | [DLF](https://github.com/pwang322/DLF) | 共享/专属信息解耦、语言引导增强及分层预测，提供语言主导性与融合方式的竞争参照 |
| GsiT | [ACL 2025 主会](https://aclanthology.org/2025.acl-long.109/) | [GsiT](https://github.com/drewjin/GsiT) | 图结构、交错掩码与参数共享，兼顾融合效果和效率 |
| DPDF-LQ | [EMNLP 2025 主会](https://aclanthology.org/2025.emnlp-main.571/) | [DPDF-LQ](https://github.com/ZhouMiaoGX/DPDF-LQ) | 全局/局部双路径和动态可学习查询，提供较新的融合架构参照 |
| Ours | 本项目 | 本项目 | 完整教师蒸馏后的 TAV 学生系统 |

实施要求：

1. 优先基于官方实现复跑，保留各方法核心架构、必要损失和训练机制。
2. 核对 official split、测试 utterance ID、标签及可用样本覆盖。不能静默使用不同的测试子集。
3. 统一评测脚本与 checkpoint 选择原则；多个指标取同一个 checkpoint。
4. 报告各自输入特征、编码器、预训练资源和部署参数规模；评测统一不代表骨干相同。
5. 引用论文数字与本项目复跑数字分开标记，不能无标记混填。
6. 若把某方法接入本项目统一编码器，该结果另标为架构适配实验，不能与原生系统结果混称。

教师加 Probe 的性能可以另外提供为教师参照，但不得直接称为理论上界；需要说明 Probe 的监督训练与校准过程。

## 6. 核心机制消融与可选补充诊断

核心消融优先在 MOSEI 完成三种子；决定论文核心解释的对照在 MOSI 上同步验证。已有结果只有在数据、训练、目标和评测协议一致时才可复用。

| 对照组 | 具体设置 | 需要区分的解释 |
|---|---|---|
| Full KD / Subset-7 / Uniform Interaction | 依次加入七子集监督、交互坐标变换 | 收益来自额外监督，还是监督组织方式？ |
| Uniform / R-only / U-only / R×U | 相同交互目标，替换坐标权重 | R、U 各自是否有效，组合是否必要？ |
| Ensemble Full KD（关键必做） | 完整 TAV 回归目标取三 Probe 均值；分类目标对各 Probe 校准后的概率做平均 | 简单集成是否解释收益？ |
| 幅度替代 R | 用 `abs(interaction_mean)` 替换 R，保留 U 及相同的归一化、截断流程 | 收益是否需要 Probe 间稳定性信息，而非仅需交互幅度？ |
| 打乱 R＋保留 U | 跨训练样本打乱完整七维 R 向量，保留坐标结构及权重分布 | R 是否需要与正确样本对应？ |
| 仅一阶坐标 | 保留公共 Full KD，只增加 T/A/V 一阶坐标监督 | 是否需要跨模态高阶项？ |

权重变体沿用相同的最终归一化原则，避免仅通过放大附加损失获益。仅一阶项的附加损失按保留坐标取平均，并给予可比的系数调参预算。

### 6.1 U 的定义与报告范围

当前 U 是在训练集上估计后固定使用的七维任务关联权重，不是逐样本动态 utility。每个坐标按教师交互均值与训练标签的绝对 Pearson 相关系数计算，经下限截断和均值归一化得到：

```text
u_k = max(abs(Pearson(interaction_mean_k, y_train)), 0.01)
U_k = u_k / mean_j(u_j)
```

方案及论文正文不必逐项列出具体权重向量，但方法说明保留估计数据范围、计算公式、截断和归一化规则。实际权重及资产标识保留在实验配置中，并通过复现材料提供配置或重建方式；不删除已有冻结记录。

本轮实验集中验证任务关联加权的总体作用，以 Uniform、R-only、U-only、R×U 及其他关键参照支撑结论。不据此宣称精细七坐标权重不可替代，也不宣称 U 能逐样本识别有价值的交互。

### 6.2 R 的解释范围

当前 R 的基本形式为：

```text
R = |interaction_mean| / sqrt(interaction_variance + 1e-4)
```

随后执行逐样本均值归一化、截断及最终权重归一化。R 同时包含交互幅度和 Probe 间稳定性，不等于纯粹的方差估计，也不等于多个独立大模型教师的不确定性。

幅度替代 R 的对照纳入优先机制验证，比较时保留 U，避免同时改变两个因素。仅用逆标准差替代 R 的稳定性对照可作为补充；若强调乘法组合的独特性，再增加归一化后的 R+U 对照。幅度替代 R 的优先级高于 R+U。

U 应描述为训练标签关联代理，不作因果 utility 的解释。一阶坐标不应全部称为跨模态交互。

## 7. 复现前的具体核查

| 方法 | 已发现的适配问题 | 实施要求 |
|---|---|---|
| EA-KD | 论文定义熵上界为 `log(C)`；本轮检索的官方分类代码非 ImageNet 分支写为 `100.0` | 七分类任务按论文定义使用 `log(7)`，记录相对官方代码的差异；区分主表版本和 EA-KD-Full |
| SKD | 本轮检索的官方实现存在先把损失归约为标量再乘掩码的写法，实际效果不同于逐样本选择 | 固定具体 commit，核对论文与代码；预先声明实现依据及差异，不根据结果选择版本 |
| SKD | 包含协方差与 Cholesky 运算，当前训练 batch 较小 | 核查有限值、正则、梯度及 batch 敏感性，不能把梯度累积当成关系 batch 扩大 |
| Projector 特征法 | 核心设计含 BatchNorm，教师与学生表示维度和来源不同 | 核对特征位置、投影及归一化；不能随意替换为其他损失后仍称完整复现 |
| RLD | 原方法作用于分类 logits，七分类情感标签及类间关系不同于原基准 | 保留完整核心损失，核查标签掩码、归约、调度和数值稳定性 |

实现依据：[EA-KD 论文公式](https://arxiv.org/html/2311.13621v3)、[EA-KD 分类代码](https://github.com/cpsu00/EA-KD/blob/main/mdistiller/distillers/KD.py)、[SKD 代码](https://github.com/HyunJunSik/StreamLined/blob/main/Distiller/SKD.py)。以上为本轮检索所见，后续仓库变化以冻结的 commit 为准。

有公开仓库只表示有可参考实现，不代表已经在本项目上跑通。正式长训练前完成方法级核查与短程训练验证。

## 8. 统一训练、评测和统计协议

### 8.1 数据与训练

- 使用 official train/valid/test 划分；Probe 训练及 U 的估计不能使用 test 标签。
- 两个数据集分别构建和核验教师资产、校准与任务统计，不能把 MOSEI 的 U 直接当作 MOSI 主实验默认值。
- 主表 A 固定学生骨干、适配范围、原始媒体预处理、融合结构、任务损失和评测流程。特征法额外的训练投影器单独记录。
- 正式种子沿用 `13 / 42 / 2026`。每个方法获得预先声明的可比调参预算，只用 valid 选择配置。
- 当前公共训练参考为 batch size 8、学习率 1e-4、最多 30 epochs、patience 7；如后续修改公共协议，应建立新版本并保证相关行一致。
- 原论文默认超参数作为起点，不要求把 CIFAR/ImageNet 的温度、权重、训练轮数机械照搬到 MSA。
- 任何新增方法、Subset-7 系数或交互系数均先冻结搜索范围和选择规则，不能为得到预期排名不断扩展搜索。
- 如需要 batch 敏感性分析，单列扩展实验并报告预算变化，不把不同训练条件无标记混入主表。

### 8.2 指标与模型选择

| 项目 | 规则 |
|---|---|
| 主指标 | MAE，越低越好 |
| 其他指标 | Pearson Corr、Acc-2、Weighted F1、Acc-7 |
| Checkpoint | 统一按 valid MAE 选取；所有指标使用同一 checkpoint |
| Acc-2 / F1 | 沿用项目规则，排除真实标签为零的样本，并在论文注明 |
| Acc-7 | 沿用项目回归预测离散化规则，不混用分类头准确率 |
| 长样本 | 沿用窗口到 utterance 的聚合规则；最终在 utterance 层面评测 |
| 主表展示 | 每个方法、每个数据集报告三种子均值与标准差 |
| Official test | 配置锁定后统一评测全部主表方法，不只评测 Ours 或验证集赢家 |

### 8.3 关键比较与不确定性

预先声明核心比较：Ours vs Full KD、Ours vs Subset-7、Ours vs Uniform Interaction、Ours vs Ensemble Full KD、Ours vs 验证集选定的最强外部 KD 方法，以及核心消融中的配对比较。

每个种子使用自己的 best-valid checkpoint 独立生成测试预测，主表报告三个种子指标的均值与样本标准差（`ddof=1`），不先平均预测来形成未声明的模型集成。

关键比较按同一种子配对，分别进行 10,000 次原视频聚类 bootstrap，保持同一原视频内 utterance 一起重采样。报告逐种子差值及其区间，同时报告三种子平均差值。三个种子对同一测试集的预测不是三倍独立样本，不拼接后当作独立样本重采样。当前不额外构造跨种子的统一区间；若后续需要，应另行明确层级重采样设计。

样本 bootstrap 不替代训练随机性复验；若报告多项显著性检验，应预先说明多重比较处理。

不按单次测试指标决定哪些方法出现在论文中。训练数值故障与正常但较差的结果分别记录，不能删除正常的不利种子。

### 8.4 硬件与运行环境

同一种子下的配对方法尽量使用同型号 GPU，并记录 GPU、精度、batch size、软件版本和注意力后端。若混用硬件，应让不同方法在硬件分配上保持可比，避免外部方法与 Ours 系统性绑定不同 GPU。

不预先硬编码“某种子必须对应某 GPU 型号”；这种安排会把种子与硬件效应混在一起。根据实际资源进行匹配或平衡分配，并如实报告。不能仅因显存更大就单独增大某方法的 batch。正式效率测量在同一硬件、相同输入和一致计时边界下完成。

## 9. 效率报告

| 成本类别 | 需要报告的内容 |
|---|---|
| 学生部署 | 全部部署参数、实际推理耗时、峰值显存、测试输入规格 |
| 学生训练 | 可训练参数、训练耗时、峰值显存、七子集融合额外成本 |
| 教师离线处理 | 七子集特征提取次数、耗时及缓存规模 |
| Probe | 三 Probe 的训练和预测成本；说明共享教师特征 |
| 外部系统 | 统一计时边界，说明是否含编码器与媒体预处理 |

分别报告 `Model-only latency` 与 `End-to-end latency`。前者注明计时入口是原始输入还是预提取特征；后者尽可能包括媒体解码、特征提取和任务模型推理。若某外部系统的完整特征提取链不可复现，标为不可得，不将其特征模型耗时与本项目端到端耗时直接比较。

不能只报 LoRA 可训练参数作为学生总规模。最终只使用完整 TAV 的方法应明确推理时不需要教师及七子集集成；训练投影器如果推理时移除，也需说明。

## 10. 执行顺序与交付物

### 10.1 最低论文充分集

正式资源预算不再把完整扩展矩阵全部视为必做。每个数据集优先完成以下 53 次训练：

| 层级 | 方法 | Runs / dataset | 要求 |
|---|---|---:|---|
| Tier 1 | Adapted Student-only、Full KD、Uniform Interaction、Ours | 12 | 必须 |
| Tier 1 | R-only、U-only、Ensemble Full KD、Magnitude × U | 12 | 必须/强烈建议 |
| Tier 1 | Subset-7 | 5 | 必须；3 个 seed13 候选＋选中配置补 2 个种子 |
| Tier 1 | First-order Interaction | 3 | 固定 `lambda_interaction=1`，不单独调参 |
| Tier 2 | Projector Feature KD | 5 | 必须 |
| Tier 2 | EA-KD | 5 | 必须 |
| Tier 2 | CMAD-style CAFD component adaptation | 5 | 领域 KD；不得称为完整 CMAD |
| Tier 2 | DLF、DPDF-LQ 原生系统 | 6 | 各三种子 |
| 合计 |  | **53** | 两个数据集合计 106 次训练 |

领域 KD 优先选择 CMAD-style CAFD：其跨样本相关特征蒸馏能作用于当前融合表示。Light-MER
SWD-H 的核心是生成式回答 token 的隐藏状态几何；改为 pooled feature 会改变其核心机制，故暂不
作为正式统一框架 baseline。CAFD 适配保留特征 MSE、相关矩阵和双向 KL，并增加必要的学生表示
投影；完整 CMAD 的随机模态缺失和 MAR 不在该行中。

Tier 3 包含 RLD、SKD、Shuffled R＋U、EA-KD-Full、R×coarse U、R＋U 和第三个原生系统
GsiT。它们保留实现和冻结配置，但默认计划不启动，资源充足时通过显式 Tier 3 开关补充。

| 阶段 | 工作 | 交付物 |
|---|---|---|
| P0 收尾 | 完成现有主链三种子与结果核查 | 完整多种子结果及有效性记录 |
| P1：关键参照与基本机制 | 补 Adapted Student-only、Ensemble Full KD、Subset-7、R-only、U-only及幅度替代 R；复用同协议 M4/M6 | 排除适配、集成、额外监督及幅度权重的主要替代解释 |
| P2：已发表 KD | 推进 Projector、EA-KD、CMAD-style CAFD 的实现核查、验证调参和三种子训练；RLD/SKD 留作 Tier 3 | 主表 A、方法适配说明及冻结配置 |
| P3：第二数据集与系统比较 | 构建 MOSI 资产；完成两个数据集的主表行；复跑 DLF、DPDF-LQ，GsiT 留作 Tier 3 | 两个数据集的主表 A/B |
| P4：机制解释 | 完成其余核心消融及 EA-KD-Full；按论文主张和资源补充扩展 | 消融表、效应量与机制分析 |
| P5：最终评测 | 锁定方法与配置，统一 official test、统计和效率评测 | 最终表格、逐样本预测、统计报告、复现说明 |

MOSI 资产准备及外部实现核查可以在资源允许时提前开展；顺序不表示必须等待前一阶段所有任务结束后才做独立准备。不得因新方法接入而改写正在运行的冻结实验。

新增方法每项至少保存：方法名称和来源、官方 commit、适配差异、完整配置、数据与教师资产标识、种子、最佳 checkpoint 选择记录、逐样本预测、失败或异常记录。

## 11. 备选与有条件扩展

| 方法 | 发表信息与资源 | 本方案中的位置 |
|---|---|---|
| LSKD | [CVPR 2024 论文](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html)、[代码](https://github.com/sunshangquan/logit-standardization-KD) | 成本较低的 logits 扩展，不替代特征与同信息量对照 |
| WTTM | [ICLR 2024 论文](https://openreview.net/pdf?id=MJ3K7uDGGl)、[代码](https://github.com/zkxufo/TTM) | 温度与样本加权扩展，非首批必跑 |
| WKD | [NeurIPS 2024 论文](https://proceedings.nips.cc/paper_files/paper/2024/hash/78526d7ad4a2532bd91416e948b9644c-Abstract-Conference.html)、[代码](https://github.com/JiamingLv/WKD) | 需要进一步定义类别代价或特征分布适配，非首批必跑 |
| EBMC | [CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/html/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html)、[代码](https://github.com/kangverse/EBMC) | 如主张缺失模态鲁棒性则优先加入；先区分公开实现的分类与缺失率回归管线 |
| CMAD | [ICCV 2025 论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf)、[代码](https://github.com/YetZzzzzz/CMAD) | 缺失模态 MSA；公开代码含 MOSEI 连续回归头，但以缺失模态条件下的 Acc/F1 和完整 CAFD＋MAR 协议组织实验，不能直接把原论文数字并入当前完整 TAV 的 MAE 表 |
| TSD | [CVPR 2026，论文预印本](https://arxiv.org/abs/2602.19585) | 讨论全局共享、模态对共享和私有子空间；本轮未核实到作者公开代码，先纳入相关工作 |
| Light-MER / SWD-H | [论文](https://arxiv.org/abs/2607.12787)、[官方代码](https://github.com/GAIR-Lab/Light-MER)；官方仓库声明 ACM MM 2026 主会接收 | 补入直接相关工作并优先核查适配；先区分完整生成式系统与单独的 SWD-H 迁移实验 |
| CorrKD | [CVPR 2024 论文](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Correlation-Decoupled_Knowledge_Distillation_for_Multimodal_Sentiment_Analysis_with_Incomplete_Modalities_CVPR_2024_paper.html) | 缺失模态 MSA 蒸馏相关工作；公开实现及本项目协议适配仍需核实 |
| DHMD | [论文条目](https://arxiv.org/abs/2602.04260)标注 TPAMI 2026 | 领域相关候选；本轮未完成正式出版页、代码和评测协议核查，不直接列为顶会且已可复现的主表方法 |

不因年份更新就强行加入不同任务或输入条件的方法。也不能仅凭本轮未找到代码就断言作者不存在公开实现。

### 11.1 对领域相关 KD 建议的复核

Light-MER 的大教师到小学生设定与本项目相关，应补入相关工作。其公开方法以 MER-Caption+ 训练生成式模型，SWD-H 对齐回答 token 隐状态；MOSI/MOSEI 主要报告 WAF/ACC，并非本项目的连续回归 MAE 协议。因此，不直接把完整 Light-MER 填入当前回归主表。先核查 token 级监督、教师缓存和学生表示是否支持保留方法核心的迁移；如改为 pooled-feature SWD，需明确标注适配差异，不能称为完整 Light-MER 复现。[方法与评测协议](https://arxiv.org/html/2607.12787v1)

同理，只移植 CMAD 的 CAFD 子模块，应标为 `CAFD component adaptation`；不能据此声称复现了包含其他训练机制的完整 CMAD。领域名称更接近是优先核查的理由，不是跳过方法完整性和评测一致性的理由。

当前最低充分集保留 Projector、EA-KD 与 CMAD-style CAFD。RLD/SKD 固定为 Tier 3；若资源
允许，可按预注册配置补充，但不得依据测试成绩选择性纳入或删除。

## 12. 结果解释与结论边界

作为设计依据，2026-09-16 的历史 seed13 报告中：M6 相对 M3 的 valid MAE 改善为 0.007423，差值的 95% 区间为 `[-0.019538, +0.004635]`，跨越零；该结果不是三种子正式结论，也不是 official test 结果。实时运行进度不由本文更新。

来源：[TAV 主实验 seed13 结果汇总](tav_main_v1.md)。

后续按实际结果决定贡献表述：

- 若 Ours 稳定优于同适配 Full KD、Subset-7 和近年 KD 方法，可支持选择性交互蒸馏的有效性。
- 若 Ours 与 Subset-7 接近，应优先考虑七子集额外监督的解释。
- 对 U 的结论限定为所验证的任务关联加权作用，不将总体性能改善直接解释为精细七坐标权重不可替代。
- 若 EA-KD-Full 达到相近效果，应讨论普通样本价值加权是否足以解释回归收益。
- 若主要收益来自 T/A/V 适配，不应将其归因于交互蒸馏。
- 若第二数据集或训练种子表现不稳定，应如实报告适用范围，不预设全面优于所有方法或达到 SOTA。

本方案的目标是让实验能够区分这些解释；预期方法排序不是保留结果或判断实验有效性的条件。

<a id="implementation-notes"></a>

## 原文快照：`RDID-MSA_实验实现说明_20260917.md`

# RDID-MSA 实验实现说明

本实现对应《RDID-MSA 主表对比与消融实验方案》，不改写冻结的 P0 入口。新增代码位于独立的
`main_table_v1` 协议、训练器和输出目录。

## 1. 已实现范围

固定学生训练器覆盖 19 个方法标识。默认计划只生成最低论文充分集，Tier 3 需要显式解锁：

- Tier 1：`adapted_student`、`full_kd`、`subset7`、`uniform_interaction`、
  `ru_interaction`、`r_only_interaction`、`u_only_interaction`、`ensemble_full`、
  `amplitude_interaction`、`first_order_interaction`；
- Tier 2：`projector`、`ea_kd`、`cmad_cafd`；
- Tier 3：`rld`、`skd`、`shuffled_r_u_interaction`、`ea_kd_full`、
  `coarse_u_interaction`、`additive_r_u_interaction`。

所有方法使用 M3/M6 相同的 T/A/V LoRA 适配范围、任务损失、原始媒体输入、valid MAE
选模和窗口聚合。Projector 与 CAFD 的训练期线性投影器计入训练参数并在报告中单列；
推理仍只执行完整 TAV，不使用教师或七子集集成。

实现依据与差异已经冻结：

- EA-KD 使用论文的 `log(7)` 熵上界，不使用官方非 ImageNet 分支的 `100`；
- RLD 保留 SCD、MCD、标签掩码、`alpha=1`、`beta=8` 和 20 epoch warm-up；
- SKD 对逐样本 KL 应用低置信度 mask，方向项使用归一化 logits Gram 矩阵、Tikhonov
  正则和 Cholesky；没有沿用官方 commit 中“标量损失乘 mask”的归约错误；
- Projector 保留线性投影、无仿射 BatchNorm、四次幂和 LogSum；教师目标是冻结
  Qwen3-Omni 的 TAV、last-valid-token 2048 维表示；
- `cmad_cafd` 明确标为 CMAD-style CAFD 组件适配：保留逐样本特征 MSE、跨样本余弦
  相关矩阵和双向 KL，增加必要的 512→2048 学生投影；不冒充包含 MAR 的完整 CMAD；
- Subset-7 使用三个 Probe 的七子集原始回归均值；Ensemble Full KD 使用三 Probe 的
  TAV 回归均值及校准概率均值。

## 2. 冻结资产和实验矩阵

MOSEI 已生成：

```text
outputs/experiments/main_table_v1/mosei/assets/protocol.json
outputs/experiments/main_table_v1/mosei/assets/ensemble_targets.jsonl
outputs/experiments/main_table_v1/mosei/assets/teacher_feature_index.jsonl
```

完整预注册矩阵在 `project/configs/main_table_v1/experiments.json`。最低充分集每个数据集有
53 次训练：47 次固定学生和 6 次两个原生系统。计划文件共有 57 个节点，其中另有 4 个
valid 选择门；调参候选只使用 seed13 valid，按最小 valid MAE 冻结后再补 seeds 42/2026。
8 个补种子任务先作为 `selected_replication` 模板，选定超参数后才物化命令。

当前 MOSEI 和 MOSI 均已生成独立的冻结资产与执行计划。每个数据集的最低充分集包含 53 次
计划训练；当前计划汇总均为 39 个可直接启动、14 个受 valid 选择门或原生系统资产约束，另有
4 个选择门节点。每个不可执行节点都带 `blocked_reason`。截至 2026-09-18，两个数据集的新主表
训练均未自动启动，资产就绪不能写成已有实验成绩。

```bash
conda activate kd
python project/scripts/build_main_table_plan.py --dataset mosei --output /tmp/mosei_plan.jsonl
python project/scripts/build_main_table_plan.py --dataset mosi --output /tmp/mosi_plan.jsonl
# 资源充足时才加入 Tier 3
python project/scripts/build_main_table_plan.py --dataset mosei --include-tier3 --output /tmp/mosei_full_plan.jsonl
```

单项 MOSEI dry-run 或训练示例：

```bash
python project/scripts/train_main_table_kd.py \
  --method ea_kd --assets outputs/experiments/main_table_v1/mosei/assets/protocol.json \
  --output outputs/experiments/main_table_v1/mosei/students/ea_kd_candidate2_seed13 \
  --seed 13 --ea-entropy-temperature 3 --dry-run
```

## 3. MOSI 教师资产（已完成）

MOSI 媒体、official train/valid 窗口、七子集教师特征、三个 Probe、交互目标、ensemble targets、
MOSI 专属协议和执行计划均已完成。准备过程只读取 train/valid，没有读取 test 标签；最终状态为
`complete`，计划含 53 次训练、39 个 ready run、14 个 blocked run 和 4 个选择门。复建命令为：

```bash
python project/scripts/prepare_mosi_main_table.py --prepare-only
python project/scripts/prepare_mosi_main_table.py
```

第二条命令会等待两张 GPU 连续满足显存容量条件，随后可恢复地完成七子集特征、三个 Probe、MOSI 专属
交互协议、主表资产及 `mosi_plan.jsonl`。本轮 66 GiB 教师采用两卡模型并行，并以每卡至少
45,000 MiB 可用显存作为并发准入条件，不要求 GPU 独占；特征完成后三个轻量 Probe 并行
运行，其中 GPU0 同时运行两个、GPU1 运行一个。执行计划直接以冻结的
`assets/protocol.json` 为就绪
依据，不需要再手工修改实验矩阵。进度记录在
`outputs/experiments/main_table_v1/mosi/preparation_status.json`。
该流水线已在生成资产和计划文件后停止，没有自动启动任何主表学生或原生系统训练。机器可读状态在
`outputs/experiments/main_table_v1/mosi/preparation_status.json`，冻结资产在
`outputs/experiments/main_table_v1/mosi/assets/protocol.json`。

## 4. 原生系统主表

官方仓库和 commit 记录见 `external/SOURCES.json`。原生系统运行器会先审计标准 MMSA
`aligned_50.pkl` 的 train/valid/test 数量、ID、标签和特征维度，并生成各仓库原生配置。
当前工作区缺少该标准特征文件以及 `bert-base-uncased` 原生检查点，因此矩阵将默认的 DLF、
DPDF-LQ 标记为明确阻塞；GsiT 属于 Tier 3。运行器不会退化为不同测试子集或偷偷改用 RDID 编码器。

资产到位后的预检示例：

```bash
python project/scripts/run_external_system.py \
  --system dlf --dataset mosei \
  --feature-file dataset/standard_features/mosei_aligned_50.pkl \
  --bert-model model/bert/bert-base-uncased \
  --output outputs/experiments/main_table_v1/mosei/native/dlf_seed13 \
  --seed 13 --dry-run
```

DLF、GsiT 和 DPDF-LQ 的本地适配均已移除训练时 test 访问，并统一按 valid MAE 选模。
它们保留原生架构、输入特征和核心训练机制；结果应标为“官方实现复跑”，不能与统一编码器
架构适配混称。

原生系统的 `final_test` 必须用 `--source-run` 指向同系统、数据集、种子且状态为 complete 的
`train_valid` 目录。运行器要求其中只有一个 valid-best checkpoint，仅加载该 checkpoint，
不会在 official test 阶段重新训练。

## 5. Official test、统计与效率

冻结 P0 的 M0/M1/M3/M4/M6 × seeds13/42/2026 共 15 个 valid-best checkpoint 已于
2026-09-18 完成统一 MOSEI official-test 评测。该批次只测试 `best.pt`，不测试 `last.pt` 或逐
epoch checkpoint；测试结果不用于回头选择配置。机器可读汇总位于
`outputs/experiments/tav_main_v1/official_test_20260917/all_five_methods_summary.json`。这属于 P0
收尾证据，不表示下述新主表方法已经完成 official test。

两个数据集均已生成独立的 official-test 窗口清单。固定学生只有在配置锁定且 train/valid run
状态为 complete 后才能显式解锁一次测试：

```bash
python project/scripts/evaluate_main_table_test.py \
  --run outputs/experiments/main_table_v1/mosei/students/ru_interaction_seed13 \
  --test-manifest dataset/cmu_mosei/manifests/official_test_windowed.jsonl \
  --output outputs/experiments/main_table_v1/mosei/final_test/ru_interaction_seed13 \
  --allow-official-test
```

报告分别包含以预处理张量为入口的 model-only latency，以及含媒体读取、处理、模型前向和
窗口聚合的 end-to-end latency，同时记录全部部署参数和峰值显存。无法恢复完整特征链的
原生系统显式标记 latency 不可得。训练报告包含可训练参数、Projector 参数、总耗时与峰值显存。

主表汇总同时读取每个方法的 `teacher_subsets`、`probe_count` 和
`teacher_hidden_features`，避免把学生架构固定误写为教师信息预算完全相同。

## 6. 领域相关方法复核

Light-MER 官方仓库已按 commit 冻结，但 SWD-H 原实现匹配生成式回答 token 的末层隐藏状态，
训练模板使用 MER-Caption+，并非当前连续回归头。它暂不进入 MAE 主表；完整核查结论见
`external/LIGHT_MER_ADAPTATION_AUDIT.md`。CorrKD 本轮仍未识别到作者官方仓库，因此没有
绕过“先克隆原实现”的要求自行编写同名方法。

CMAD 官方仓库也已冻结并完成 CAFD/MAR 代码核查。其公开实现是带随机模态缺失和 MAR 的
完整系统，且训练循环会在 valid 改善时读取 test；因此既不能原样进入当前主表，也不能只移植
CAFD 后称为完整 CMAD。最低充分集采用明确命名的 `CMAD-style CAFD component adaptation`；
边界和后续独立扩展要求见 `external/CMAD_ADAPTATION_AUDIT.md`。

## 7. 统计汇总

三种子汇总和逐种子的原视频聚类配对 bootstrap：

```bash
python project/scripts/analyze_main_tables.py \
  --experiment-root outputs/experiments/main_table_v1/mosei \
  --manifest dataset/cmu_mosei/manifests/valid.jsonl --split valid \
  --comparison ru_interaction:full_kd \
  --comparison ru_interaction:subset7 \
  --resamples 10000 \
  --output project/reports/main_table_v1_mosei_valid.json
```

bootstrap 在每个训练种子内分别按原视频簇有放回抽取，报告各种子的差值与 95% 区间；另报
三个种子点估计的平均差值，但不为该平均值构造未预注册的跨种子统一区间。
