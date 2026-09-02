---
title: "Qwen3-Omni 教师驱动的 CMU-MOSEI 多模态情感交互坐标蒸馏"
short_title: "RDID-MOSEI"
version: "v2.1"
date: "2026-08-03"
language: "zh-CN"
status: "修订后的可执行研究方案"
teacher: "Qwen3-Omni-30B-A3B-Instruct（关闭 Talker）"
dataset: "CMU-MOSEI"
---

# Qwen3-Omni 教师驱动的 CMU-MOSEI 多模态情感交互坐标蒸馏

> **英文名称：** Rate-Distortion Constrained Interaction-Coordinate Distillation for Multimodal Sentiment Analysis  
> **方法简称：** RDID-MOSEI  
> **核心问题：** 在教师子集缓存、学生前向次数和训练预算相同的条件下，完整 Möbius/Harsanyi 重参数化是否改善优化；进一步选择高阶交互监督子空间，能否使有限信息率学生更有效地保留跨模态情感行为？  
> **教师模型：** Qwen3-Omni-30B-A3B-Instruct，关闭 Talker，只使用 Thinker。  
> **主数据集：** CMU-MOSEI。  
> **主学生：** Qwen3-0.6B + HuBERT-Base + VideoMAE-Small。  
> **关键修订：** 不再声称 Harsanyi 交互提供新的教师信息；删除与全量交互线性重复的 Unique loss；严格区分信息率、计算预算和模型容量。

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [研究问题与修订后的论文主张](#2-研究问题与修订后的论文主张)
3. [数学边界：交互坐标不增加信息](#3-数学边界交互坐标不增加信息)
4. [CMU-MOSEI 数据方案](#4-cmu-mosei-数据方案)
5. [Qwen3-Omni 教师方案](#5-qwen3-omni-教师方案)
6. [教师模态子集构造](#6-教师模态子集构造)
7. [教师 Probe 的精确定义](#7-教师-probe-的精确定义)
8. [学生模型](#8-学生模型)
9. [价值函数与交互坐标](#9-价值函数与交互坐标)
10. [率失真与变分信息瓶颈](#10-率失真与变分信息瓶颈)
11. [损失函数](#11-损失函数)
12. [训练流程](#12-训练流程)
13. [严格公平对照](#13-严格公平对照)
14. [实验设计与指标](#14-实验设计与指标)
15. [教师准入门槛](#15-教师准入门槛)
16. [本地服务器部署与成本](#16-本地服务器部署与成本)
17. [工程单元测试](#17-工程单元测试)
18. [理论结论与不可过度声明的内容](#18-理论结论与不可过度声明的内容)
19. [消融矩阵](#19-消融矩阵)
20. [阶段性实施计划](#20-阶段性实施计划)
21. [Go/No-Go 验收标准](#21-gono-go-验收标准)
22. [风险与应对](#22-风险与应对)
23. [论文贡献表述](#23-论文贡献表述)
24. [推荐论文结构](#24-推荐论文结构)
25. [配置文件草案](#25-配置文件草案)
26. [伪代码](#26-伪代码)
27. [参考资料](#27-参考资料)

---

# 1. 执行摘要

本方案研究多模态大模型到轻量多模态学生的情感知识蒸馏。教师固定为 Qwen3-Omni，数据集固定为 CMU-MOSEI。

教师对每个样本处理七种非空模态组合：

\[
t,\quad a,\quad v,\quad ta,\quad tv,\quad av,\quad tav,
\]

其中 \(t\)、\(a\)、\(v\) 分别表示文本、音频和无声视频。

教师子集输出可以表示成七维价值向量：

\[
\mathbf v_F=
[V_F(t),V_F(a),V_F(v),V_F(ta),V_F(tv),V_F(av),V_F(tav)]^\top.
\]

通过 Möbius 线性变换，可以得到七维交互坐标：

\[
\boldsymbol\mu_F=M\mathbf v_F+\mathbf b.
\]

由于该变换在固定基线下可逆，完整子集价值与完整交互坐标包含相同信息。本研究不把交互项描述成新的教师知识，而将实验拆成两个不同问题：

1. **完整坐标重参数化：** 在七维监督信息等价时，从原始子集坐标改写为完整交互坐标，是否改变有限容量学生的优化结果？
2. **高阶监督子空间选择：** 在使用相同七子集教师缓存、相同学生前向和相同训练预算时，只监督二阶、三阶交互方向，是否比等维子集方向更有效？

第二个问题不声称实际进入损失的监督信息完全等价；它研究的是从同一教师行为缓存中选择不同监督子空间的归纳偏置。

主方法不同时使用完整 Subset KD、完整 Interaction KD 和 Unique loss，避免线性重复。建议主目标为：

\[
\mathcal L_{\mathrm{main}}
=
\mathcal L_{\mathrm{task}}
+\lambda_f\mathcal L_{\mathrm{fullKD}}
+\lambda_i\mathcal L_{\mathrm{high-order}}
+\beta\mathcal L_{\mathrm{rate}}.
\]

其中：

- \(\mathcal L_{\mathrm{task}}\)：真实标签监督；
- \(\mathcal L_{\mathrm{fullKD}}\)：完整三模态输出蒸馏；
- \(\mathcal L_{\mathrm{high-order}}\)：二阶和三阶交互坐标蒸馏；
- \(\mathcal L_{\mathrm{rate}}\)：变分信息率上界代理。

直接 Subset KD 是必须比较的公平基线，而不是默认与全部交互损失叠加。

---

# 2. 研究问题与修订后的论文主张

## 2.1 原始问题

普通知识蒸馏只要求学生模仿教师的完整输入输出：

\[
p_S(y|tav)\approx p_T(y|tav).
\]

它不能直接约束学生在以下情况下的行为：

- 只有文本；
- 只有音频；
- 只有视频；
- 文本和音频组合；
- 文本和视频组合；
- 音频和视频组合。

对于讽刺等样本，完整预测正确并不意味着学生正确使用了音频和视频。

## 2.2 修订后的核心问题

本研究的核心问题不是：

> Harsanyi 交互是否包含子集输出之外的新知识？

答案是否定的。固定基线下，二者是可逆线性变换。

真正需要分别验证的是：

> 完整七维实验检验信息等价的坐标重参数化；四维实验检验在相同教师缓存、前向次数和训练预算下，高阶交互子空间是否比等维子集子空间更适合指导有限容量学生。

## 2.3 主假设

### H1：坐标系影响有限容量学生的优化

尽管 \(\mathbf v\) 与 \(\boldsymbol\mu\) 信息等价，不同坐标系对应不同的损失几何、梯度方向和权重结构。有限容量、非凸学生不一定对可逆变换保持优化等价。

### H2：高阶交互加权可缓解文本主导

对二阶、三阶项显式加权，可能使学生更重视：

- 文本—语调反讽；
- 文本—表情矛盾；
- 语音—表情一致；
- 三模态联合消歧。

### H3：VIB 使坐标选择更重要

当学生的信息率受到约束时，它必须选择保留哪些行为。交互坐标可能促使有限表示优先保留跨模态关系，而不是重复的单模态信息。

### H4：提升应集中在冲突或高交互样本

若方法确实学习高阶行为，性能提升应主要出现在教师高交互强度样本，而非所有样本均匀提升。

---

# 3. 数学边界：交互坐标不增加信息

## 3.1 可逆性

对三个模态，定义包含固定空基线的集合价值：

\[
V_F(\varnothing)=b.
\]

非空子集价值向量为：

\[
\mathbf v_F=
[V_F(t),V_F(a),V_F(v),V_F(ta),V_F(tv),V_F(av),V_F(tav)]^\top.
\]

对应 Harsanyi 项为：

\[
\boldsymbol\mu_F=
[\mu_F(t),\mu_F(a),\mu_F(v),\mu_F(ta),\mu_F(tv),\mu_F(av),\mu_F(tav)]^\top.
\]

Möbius 变换可以写成：

\[
\boldsymbol\mu_F=M\mathbf v_F+\mathbf c_b.
\]

逆变换为：

\[
\mathbf v_F=M^{-1}(\boldsymbol\mu_F-\mathbf c_b).
\]

因此：

\[
\mathbf v_F
\longleftrightarrow
\boldsymbol\mu_F
\]

在信息上等价。

## 3.2 正确的研究表述

可以表述为：

> Interaction-coordinate distillation reparameterizes the same subset behavioral supervision and introduces an interaction-oriented inductive bias through coordinate-specific weighting.

不应表述为：

> Harsanyi interaction reveals additional teacher knowledge absent from subset predictions.

## 3.3 为什么仍可能有效

如果采用未加权平方损失，并且变换为正交矩阵，那么两种坐标的误差可能等价。但 Möbius 变换一般不是正交变换；此外实际训练还包含：

- Huber 损失；
- 不同阶次权重；
- 子集采样；
- 教师不确定性权重；
- 有限学生容量；
- 非凸优化；
- VIB 信息率限制。

因此坐标变换可能改变梯度几何和优化结果，但必须通过随机正交变换等严格对照证明。

---

# 4. CMU-MOSEI 数据方案

## 4.1 数据集定位

CMU-MOSEI 是大规模自然场景多模态情感与情绪数据集，包含文本、音频和视频信息。原始论文报告约 2.3 万个句子级视频片段。

本地数据审计以服务器中的实际 `label.csv` 和官方划分文件为准。

## 4.2 本地数据统计

当前本地版本统计为：

| 指标 | 数值 |
|---|---:|
| utterance 数 | 22,856 |
| 平均时长 | 7.63 秒 |
| 中位数 | 6.49 秒 |
| P90 | 13.46 秒 |
| P95 | 16.60 秒 |
| P99 | 24.40 秒 |
| 总时长 | 48.4 小时 |
| 超过 15 秒 | 7.2% |
| 超过 20 秒 | 2.6% |
| 超过 30 秒 | 0.4% |
| 极端最长样本 | 约 109 秒 |

这些统计说明绝大多数样本较短，适合使用 Qwen3-Omni 生成教师缓存，但长尾样本仍需要独立策略。

## 4.3 划分

使用本地数据对应的官方标准划分。常见清洗版本约为：

- Train：16,326；
- Validation：1,871；
- Test：4,659。

正式实验开始前必须以本地 split 文件重新核对，不能仅依赖论文中的总数。

## 4.4 标签任务

CMU-MOSEI 情感强度通常表示为：

\[
y\in[-3,3].
\]

主任务：

1. 连续情感回归；
2. Acc-2 与 F1；
3. 可选 Acc-7。

第一版的分类 Probe 和学生分类头固定为七分类：

\[
y^{(7)}=\operatorname{clip}(\operatorname{round}(y),-3,3),
\]

并将 \(-3,-2,-1,0,1,2,3\) 依次映射为类别索引 \(0,\ldots,6\)。分类 CE、分类 KD、温度校准均使用这一七分类定义。Acc-2 和二分类 F1 不另设训练头，而由回归输出与真实连续标签的符号派生，并同时报告：

- Non-zero：排除真实标签为 0 的样本；
- Has-zero：将 0 归入非负类。

必须在论文中明确：

- Acc-2 是否排除标签为 0 的样本；
- 正负类边界；
- Acc-7 的离散化和四舍五入规则；
- 所有基线使用完全相同规则。

## 4.5 长视频处理

### 不超过 30 秒

直接处理完整音频，视频统一采样固定帧数。

建议：

- 视频帧数：16 或 24；
- 输入音频：完整 utterance；
- batch size：1；
- 按时长分桶。

### 超过 30 秒

这部分约占本地数据的 0.4%。采用以下策略之一：

**主策略：时间窗口聚合**

将样本分成不超过 30 秒的重叠窗口：

\[
W_1,W_2,\ldots,W_K.
\]

教师窗口预测聚合为：

\[
z_T(S)
=
\sum_{k=1}^{K}\alpha_k z_T(S,W_k),
\qquad
\sum_k\alpha_k=1.
\]

权重可按窗口时长或注意力池化确定。

**敏感性对照：截断**

对极端长样本只保留中心 30 秒或均匀选取 30 秒。报告窗口聚合与截断差异。

不得因为极少数长样本改变全部数据处理方式。

## 4.6 原始数据预处理

### 文本

- 使用官方转录；
- 保留否定词、感叹词、重复词和语气词；
- 不进行激进规范化；
- 保存原始文本与规范化文本两份。

### 音频

- 从视频中单独提取 WAV；
- 单声道；
- 16 kHz；
- 不做会改变情感韵律的强降噪；
- 子集输入只能使用独立 WAV 文件。

### 视频

必须生成**无声视频**：

```bash
ffmpeg -i input.mp4 -an -c:v copy silent.mp4
```

若容器格式不允许直接拷贝，则重新编码视频但不保留音轨。

不能把原始带音轨 MP4 直接当作视觉模态输入。

---

# 5. Qwen3-Omni 教师方案

## 5.1 固定模型

主教师固定为：

```text
Qwen/Qwen3-Omni-30B-A3B-Instruct
```

加载后执行：

```python
model.disable_talker()
```

只使用 Thinker 的文本理解能力，不生成音频。

## 5.2 为什么选择 Instruct + 关闭 Talker

- 同一模型原生处理文本、音频、视频；
- 可使用统一 Prompt；
- 关闭 Talker 可节省显存；
- 只需提取 Thinker 表示和情感输出；
- 不需要语音生成能力。

增强对照可使用：

```text
Qwen/Qwen3-Omni-30B-A3B-Thinking
```

但不作为主实验，以免教师变化成为额外变量。

## 5.3 官方显存参考与本地策略

官方 BF16 + FlashAttention 2 理论最低显存参考：

| 模型 | 15 秒视频 | 30 秒视频 |
|---|---:|---:|
| Qwen3-Omni-30B-A3B-Instruct | 78.85 GB | 88.52 GB |
| Qwen3-Omni-30B-A3B-Thinking | 68.74 GB | 77.79 GB |

Instruct 数值包含 Thinker 和 Talker。关闭 Talker 会降低显存占用，但本方案不预设固定节省量，实际峰值以本地实测为准。

本地两张 84 GB GPU 建议使用：

- 教师阶段先比较 Transformers 原生张量并行与 Accelerate `device_map="balanced"` 模型切分；
- BF16；
- FlashAttention 2；
- batch size 1 起步；
- 按时长分桶；
- 先对 500 个样本进行吞吐与显存基准；
- 不在学生训练时驻留教师。

`tensor_parallel_size` 不是通用的 Transformers 加载参数，不能直接写入配置后假定生效。阶段 B 必须记录实际采用的加载 API、两卡显存分布和吞吐；若当前模型/Transformers 组合支持原生 TP，则锁定其 `tp_plan` 配置，否则使用 `device_map="balanced"`，并将其明确称为模型切分而非张量并行。

## 5.4 软件版本锁定

建议创建独立环境：

```text
Python 3.11
PyTorch：与服务器 CUDA 匹配的固定版本
transformers >= 5.2.0，写入锁定版本
accelerate：固定版本
qwen-omni-utils：固定版本
flash-attn：固定版本
ffmpeg：记录系统版本
```

正式实验不能使用“近期稳定版本”这类不可复现表述。

建议保存：

```text
requirements.lock
conda-lock.yml
nvidia-smi.txt
pip-freeze.txt
git-commit.txt
```

## 5.5 推理后端

### Probe 训练与隐藏状态提取

使用 Hugging Face Transformers 后端，因为需要：

- 精确访问 Thinker hidden states；
- 注册 forward hook；
- 控制输入 token；
- 固定 pooling。

### 大规模缓存

若 vLLM 无法稳定返回所需 hidden state，则使用两阶段流程：

1. Transformers 提取并训练 Probe；
2. 固定 Probe 后，使用 Transformers 批量缓存 logits。

不要为了速度切换到无法保证等价输出或无法取隐藏状态的后端。

---

# 6. 教师模态子集构造

## 6.1 七个非空子集

| 子集 | 输入内容 | `use_audio_in_video` |
|---|---|---:|
| \(t\) | 文本 | False |
| \(a\) | 独立 WAV | False |
| \(v\) | 无声视频 | False |
| \(ta\) | 文本 + 独立 WAV | False |
| \(tv\) | 文本 + 无声视频 | False |
| \(av\) | 独立 WAV + 无声视频 | False |
| \(tav\) | 文本 + 独立 WAV + 无声视频 | False |

本方案在所有子集中固定：

```python
use_audio_in_video = False
```

即使 \(tav\) 输入，也通过独立音频文件提供音频，防止视频音轨重复输入。

## 6.2 强制要求

- \(v\) 子集不得包含任何音频；
- \(tv\) 子集不得从视频文件读取音频；
- \(av\) 和 \(tav\) 只使用一份独立音频；
- 所有子集使用同一 system prompt；
- 所有子集使用同一 user task instruction；
- 禁止因子集不同而改变语言风格或标签定义；
- 生成设置固定；
- Probe 在七个子集间完全共享。

## 6.3 Prompt

建议固定为：

```text
You are analyzing the sentiment expressed by the speaker in a short opinion clip.
Use only the modalities provided in this input.
Predict the speaker's sentiment intensity on a continuous scale from -3
(strongly negative) to +3 (strongly positive).
Do not infer unavailable modalities.
```

若使用 Probe 而不依赖生成文本，Prompt 的作用是明确任务语义和输入边界。

## 6.4 防止模态幻觉

对单模态输入额外加入明确提示：

```text
Audio and visual information not supplied in this input must be treated as unavailable.
Do not imagine missing cues.
```

但所有子集应使用结构一致的模板，只替换“available modalities”字段。

---

# 7. 教师 Probe 的精确定义

## 7.1 目标

教师不通过自由生成的数字或自然语言概率提供监督，而通过冻结 Thinker 表示上的共享 Probe 输出：

- 连续情感分数；
- 分类 logits；
- 置信度。

## 7.2 表示位置

主方案取：

> Thinker 最后一层中，“开始生成答案前的最后一个输入 token”对应的 hidden state。

记为：

\[
h_T(S)\in\mathbb R^{d_T}.
\]

必须满足：

- 不包含任何生成后的答案 token；
- 所有子集使用相同位置规则；
- padding 后按每个样本最后一个有效输入位置索引；
- 不因输入长度不同改变 pooling 算法。

## 7.3 后备 pooling

若最后输入 token 表示不稳定，预注册一个备选方案：

\[
h_T(S)
=
\mathrm{AttentionPool}(H_T^{\mathrm{input}}(S)).
\]

不能在观察测试结果后任意更换 pooling。应在验证集确定并冻结。

## 7.4 Probe 结构

共享 Probe：

\[
h_T
\rightarrow
\mathrm{LayerNorm}
\rightarrow
\mathrm{Linear}(d_T,512)
\rightarrow
\mathrm{GELU}
\rightarrow
\mathrm{Dropout}(0.1)
\rightarrow
\begin{cases}
z_T^{\mathrm{cls}}\in\mathbb R^C,\\
o_T^{\mathrm{reg}}\in\mathbb R.
\end{cases}
\]

回归：

\[
\hat y_T=3\tanh(o_T^{\mathrm{reg}}).
\]

七个子集共用同一个 Probe 参数。

## 7.5 Probe 训练

冻结 Qwen3-Omni Thinker。

训练数据中随机选择模态子集，确保 Probe 接触：

- 完整三模态；
- 单模态；
- 双模态。

建议子集采样概率：

| 类型 | 概率 |
|---|---:|
| tav | 0.40 |
| 单模态三者合计 | 0.30 |
| 双模态三者合计 | 0.30 |

Probe 任务损失：

\[
\mathcal L_{\mathrm{probe}}
=
\mathcal L_{\mathrm{SmoothL1}}
+\alpha \mathcal L_{\mathrm{CE}}.
\]

## 7.6 校准

分类 logits 在验证集做温度校准：

\[
p_T(c|S)
=
\mathrm{softmax}\left(
\frac{z_T^{\mathrm{cls}}(S)}{T_{\mathrm{cal}}}
\right).
\]

必须记录：

- 校准前后 NLL；
- ECE；
- Brier Score。

校准只调整概率，不改回归值。

---

# 8. 学生模型

## 8.1 主学生

| 模块 | 模型 |
|---|---|
| 文本编码器 | Qwen3-0.6B |
| 音频编码器 | WavLM-Base-Plus |
| 视频编码器 | VideoMAE-Base |
| 模态压缩 | 2 层 Q-Former × 3 |
| 信息瓶颈 | VIB × 3 |
| 融合器 | 4 层 Transformer Encoder |
| 输出 | 回归头 + 分类头 |

该组合保留“小型语言大模型学生”的定位，并使用已完成本地校验的三套编码器。

## 8.2 低成本诊断学生

用于方法快速筛选：

| 模块 | 模型 |
|---|---|
| 文本 | XLM-RoBERTa-Base |
| 音频 | HuBERT-Base |
| 视频 | VideoMAE-Small |

诊断学生不能替代最终主结果，但可用于先判断 Interaction 与 Subset 的差异。

## 8.3 编码器输出

\[
H_t=E_t(X_t),\qquad
H_a=E_a(X_a),\qquad
H_v=E_v(X_v).
\]

## 8.4 Q-Former

每个模态使用：

```yaml
layers: 2
hidden_size: 512
heads: 8
ffn_size: 2048
query_tokens: 4
dropout: 0.1
```

输出：

\[
\widetilde Z_m\in\mathbb R^{4\times512}.
\]

固定 token 数：

\[
K_t=K_a=K_v=4,
\qquad
K_{\mathrm{total}}=12.
\]

第一版不使用动态 token 门控。

## 8.5 VIB

\[
q_\theta(Z_m|X_m)
=
\mathcal N(\mu_m,\operatorname{diag}(\sigma_m^2)).
\]

\[
Z_m
=
\mu_m+\sigma_m\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
\]

建议 bottleneck 维度为 256。

## 8.6 融合器

输入：

```text
[CLS]
[T1][T2][T3][T4]
[A1][A2][A3][A4]
[V1][V2][V3][V4]
```

加入：

- modality embedding；
- position embedding；
- present/missing embedding。

融合器配置：

```yaml
layers: 4
hidden_size: 512
heads: 8
ffn_size: 2048
dropout: 0.1
```

## 8.7 缺失模态

缺失模态不运行对应编码器，使用可学习 token：

\[
[MISSING_m]_1,\ldots,[MISSING_m]_4.
\]

编码器完整输出可在一次 batch 内缓存，七个子集只重新组合 token 并运行融合器，降低前向成本。

---

# 9. 价值函数与交互坐标

## 9.1 主价值函数：连续情感输出

由于 CMU-MOSEI 的主标签是连续情感强度，主价值函数定义为：

\[
V_F(S)=\hat y_F(S).
\]

这直接分解模型实际输出的情感强度，不使用：

\[
-\left|\hat y_F(S)-y\right|.
\]

后者分解的是预测正确性，不是模型情感输出。

## 9.2 固定空基线

本方案不让 Qwen3-Omni处理“空输入”。

回归空基线固定为训练集平均情感：

\[
V_F(\varnothing)=b_{\mathrm{reg}}=\bar y_{\mathrm{train}}.
\]

教师和学生使用完全相同的固定基线。

可定义中心化价值：

\[
\widetilde V_F(S)=V_F(S)-b_{\mathrm{reg}},
\]

并令：

\[
\widetilde V_F(\varnothing)=0.
\]

## 9.3 辅助分类价值

对有标签样本，使用第 4.4 节定义的七分类目标 \(y^{(7)}\) 对应类别索引，分类 margin 为：

\[
V_F^{\mathrm{cls}}(S)
=
z_{F,y^{(7)}}(S)
-\log\sum_{c\neq y^{(7)}}\exp z_{F,c}(S).
\]

分类空基线由训练标签先验确定：

\[
b_{y^{(7)}}
=
\log\pi_{y^{(7)}}
-\log\sum_{c\neq y^{(7)}}\pi_c.
\]

分类交互仅作为辅助损失或分析，不替代连续情感主价值。

## 9.4 一阶项

\[
\mu_F(t)=V_F(t)-b,
\]

\[
\mu_F(a)=V_F(a)-b,
\]

\[
\mu_F(v)=V_F(v)-b.
\]

## 9.5 二阶项

\[
\mu_F(ta)
=
V_F(ta)-V_F(t)-V_F(a)+b.
\]

同理：

\[
\mu_F(tv),\qquad
\mu_F(av).
\]

## 9.6 三阶项

\[
\begin{aligned}
\mu_F(tav)
=&V_F(tav)
-V_F(ta)-V_F(tv)-V_F(av)\\
&+V_F(t)+V_F(a)+V_F(v)-b.
\end{aligned}
\]

## 9.7 主蒸馏对象

主方法只重点蒸馏高阶项：

\[
\mathcal A_{\mathrm{high}}
=
\{ta,tv,av,tav\}.
\]

这一步是从完整七维教师行为中选择四维监督子空间，不是信息等价的完整坐标变换。论文中必须将两类实验分开解释：

- Full Subset-7 vs Full Möbius-7：检验完整、可逆、信息等价的坐标重参数化；
- Subset-4 vs Interaction-4：检验等维但不同方向的监督子空间选择。

原因：

- 完整模态 KD 已约束最终输出；
- 一阶项与单模态子集值关系直接；
- 研究重点是跨模态组合行为；
- 避免同时叠加完整七维 Subset 和完整七维 Interaction。

## 9.8 删除贡献

例如：

\[
u_t=V(tav)-V(av).
\]

它满足：

\[
u_t
=
\mu(t)+\mu(ta)+\mu(tv)+\mu(tav).
\]

因此删除贡献不提供独立线性信息。

本方案中：

- 不设置 \(\mathcal L_{\mathrm{unique}}\)；
- deletion contribution 仅作为解释和评价指标。

---

# 10. 率失真与变分信息瓶颈

## 10.1 信息率

学生表示的信息率上界代理：

\[
R_{\mathrm{VIB}}
=
\mathbb E_X
\sum_{m\in\{t,a,v\}}
KL\left(
q(Z_m|X_m)\Vert p(Z_m)
\right).
\]

其中：

\[
p(Z_m)=\mathcal N(0,I).
\]

单位报告为：

```text
nats / sample
```

## 10.2 教师失真

回归蒸馏失真：

\[
D_{\mathrm{reg}}
=
\mathbb E
\rho\left(
\hat y_T(tav)-\hat y_S(tav)
\right).
\]

分类蒸馏失真：

\[
D_{\mathrm{cls}}
=
\mathbb E
KL\left(
p_T(tav)\Vert p_S(tav)
\right).
\]

高阶行为失真：

\[
D_{\mathrm{int}}
=
\mathbb E
\sum_{A\in\mathcal A_{\mathrm{high}}}
w_A
\rho\left(
\mu_T(A)-\mu_S(A)
\right).
\]

## 10.3 三类预算必须分开

### 信息率

- VIB KL；
- 单位 nats/sample；
- 通过 \(\beta\) 调节。

### 计算预算

- 情感 token 数；
- GFLOPs；
- 推理延迟；
- 峰值显存；
- 编码帧数。

### 模型容量

- 总参数量；
- 可训练参数量；
- 冻结参数量。

不得把“12 个 token”直接等同于“学生只读取了 12 个 token 的输入信息”。三个编码器已读取原始模态，12 个 token 主要约束融合接口和部分计算量。

## 10.4 率失真实验

扫描：

\[
\beta
\in
\{0,10^{-6},10^{-5},10^{-4},10^{-3}\}.
\]

报告实际测量值：

\[
R_{\mathrm{VIB}}
\rightarrow
D_{\mathrm{cls/reg/int}}
\rightarrow
\mathrm{MAE/F1}.
\]

不能只画 token 数曲线并称为率失真曲线。

---

# 11. 损失函数

## 11.1 任务损失

\[
\mathcal L_{\mathrm{task}}
=
\mathcal L_{\mathrm{SmoothL1}}
+
\alpha\mathcal L_{\mathrm{CE}}.
\]

## 11.2 完整输入蒸馏

\[
\mathcal L_{\mathrm{fullKD}}
=
\lambda_r
\rho\left(
\hat y_T(tav)-\hat y_S(tav)
\right)
+
\lambda_c\tau^2
KL\left(
p_T^\tau(tav)\Vert p_S^\tau(tav)
\right).
\]

## 11.3 直接子集坐标损失：公平基线

\[
\mathcal L_{\mathrm{subset-value}}
=
\sum_{S\in\mathcal P(\mathcal M)\setminus\{\varnothing\}}
\alpha_S
\rho\left(
V_T(S)-V_S(S)
\right).
\]

该项是基线，不与完整 Interaction 损失默认叠加。

## 11.4 主方法：高阶交互坐标损失

\[
\mathcal L_{\mathrm{high-order}}
=
\sum_{A\in\{ta,tv,av,tav\}}
w_A
\rho\left(
\mu_T(A)-\mu_S(A)
\right).
\]

权重可拆为：

\[
w_A=w_A^{\mathrm{order}}w_A^{\mathrm{confidence}}.
\]

第一版固定使用等权：

\[
w_{ta}=w_{tv}=w_{av}=1,
\qquad w_{tav}=1.
\]

不在第一版主结果中使用由熵拼接得到的交互置信权重。确定性重复前向的方差接近零，不能直接用于倒数加权。

教师不确定性作为后续消融时，必须预先指定方差来源，例如等价 Prompt ensemble 或启用 Probe dropout 的 MC 推理，并对倒数方差权重进行均值归一化和截断：

\[
w_A^{\mathrm{confidence}}
=
\operatorname{clip}\left(
\frac{1/(\widehat{\sigma}_T^2(A)+\epsilon)}
{\operatorname{mean}_{A'}[1/(\widehat{\sigma}_T^2(A')+\epsilon)]},
0.25,4
\right).
\]

该权重的定义、采样次数和随机种子必须在实验前固定；否则保持全部权重为 1。

## 11.5 VIB 信息率损失

\[
\mathcal L_{\mathrm{rate}}
=
\sum_m
KL\left(
q(Z_m|X_m)\Vert\mathcal N(0,I)
\right).
\]

使用 KL warm-up：

\[
\beta_e
=
\beta_{\max}
\min\left(1,\frac{e}{E_{\mathrm{warm}}}\right).
\]

## 11.6 主模型目标

\[
\boxed{
\mathcal L_{\mathrm{RDID}}
=
\mathcal L_{\mathrm{task}}
+\lambda_f\mathcal L_{\mathrm{fullKD}}
+\lambda_i\mathcal L_{\mathrm{high-order}}
+\beta\mathcal L_{\mathrm{rate}}
}
\]

## 11.7 不进入第一版主方法的损失

第一版不加入：

- Unique loss；
- 动态 token budget loss；
- 鲁棒一致性 loss；
- 学生 calibration loss；
- 全量 subset-value 与全量 interaction 同时叠加。

这些内容可在主效应明确后作为扩展实验。

---

# 12. 训练流程

## 阶段 A：数据与音轨审计

1. 检查 22,856 条样本文件完整性；
2. 对每条视频生成无声版本；
3. 提取独立 16 kHz WAV；
4. 检查视频音轨是否已删除；
5. 核对文本、音频、视频时长和 ID；
6. 固化 train/validation/test split。

## 阶段 B：Qwen3-Omni 基准测试

随机抽取 500 条，覆盖：

- 短于 5 秒；
- 5–15 秒；
- 15–30 秒；
- 超过 30 秒；
- 正面、负面、中性附近。

记录每个子集：

- 峰值显存；
- 前向时长；
- 输出稳定性；
- OOM 比例；
- 视频解码失败率。

基于实测吞吐重新估算全量缓存成本。

## 阶段 C：训练教师 Probe

1. 冻结 Qwen3-Omni；
2. 使用随机模态子集；
3. 训练共享分类/回归 Probe；
4. 验证集温度校准；
5. 进行教师准入评估。

## 阶段 D：全量教师缓存

先为训练集和验证集缓存七种子集：

```text
t, a, v, ta, tv, av, tav
```

只保存：

- 回归分数；
- 分类 logits；
- 熵；
- 可选池化 hidden state；
- 处理状态和版本信息。

默认不保存完整 token hidden sequence。

测试集缓存采用延迟协议：模型结构、Probe、所有超参数、高交互分组阈值和分析代码在验证集上冻结后，才对测试集生成一次七子集缓存。测试缓存只用于最终教师—学生行为指标和预注册分组分析，不得用于调参、阈值选择、模型筛选或提前停止。

## 阶段 E：学生无 VIB 预实验

先训练：

1. Student；
2. Full KD；
3. Subset-value KD；
4. Interaction-coordinate KD。

验证 Interaction 是否在公平条件下优于 Subset。

## 阶段 F：VIB 率失真实验

只有阶段 E 通过后，增加 VIB，扫描 \(\beta\)。

## 阶段 G：完整数据与随机种子

对最终核心方法运行至少 3 个随机种子，并在测试集只进行一次冻结评估。

---

# 13. 严格公平对照

## 13.1 必须控制的变量

Subset 与 Interaction 比较时保持：

- 相同教师；
- 相同七个子集；
- 相同训练样本；
- 相同学生初始化分布；
- 相同学生前向次数；
- 相同 optimizer steps；
- 相同 batch；
- 相同 loss 标量数量；
- 相同 Huber 参数；
- 相同总损失量级；
- 相同超参数搜索预算。

其中 Full Subset-7 与 Full Möbius-7 还必须保持进入损失的监督维数为 7，才能声称监督信息等价；Subset-4 与 Interaction-4 只能声称教师缓存、目标维数和计算预算相同，不能声称四维监督子空间包含相同信息。

## 13.2 核心对照组

### B0：Student

\[
\mathcal L=\mathcal L_{\mathrm{task}}.
\]

### B1：Full KD

\[
\mathcal L=
\mathcal L_{\mathrm{task}}
+\lambda_f\mathcal L_{\mathrm{fullKD}}.
\]

### B2：Subset-value KD

\[
\mathcal L=
\mathcal L_{\mathrm{task}}
+\lambda_f\mathcal L_{\mathrm{fullKD}}
+\lambda_s\mathcal L_{\mathrm{subset-value}}.
\]

### B3：Full Möbius Coordinate

蒸馏七个一至三阶坐标，不做阶次重加权。

### B4：High-order Interaction

只蒸馏：

\[
ta,tv,av,tav.
\]

### B5：Random Orthogonal Coordinate

对中心化七维子集向量使用固定随机正交矩阵：

\[
\mathbf r_F=Q\widetilde{\mathbf v}_F,
\qquad
Q^\top Q=I.
\]

蒸馏：

\[
\|\mathbf r_T-\mathbf r_S\|.
\]

用于排除“任何坐标变换都有效”。

### B6：Random Invertible Non-orthogonal Coordinate

选择条件数与 Möbius 矩阵接近的随机可逆矩阵，排除仅由缩放或非正交性造成的收益。

### B7：Interaction + VIB

最终 RDID。

## 13.3 等维对照

如果 High-order 只有 4 个标量，而 Subset 有 7 个标量，需设置等维版本：

- Subset-4：选择三个双模态值和完整值；
- Interaction-4：三个二阶项和一个三阶项。

同时保留全七维比较，不能只比较维数不同的目标。

---

# 14. 实验设计与指标

## 14.1 主任务

- MAE；
- Pearson Correlation；
- Acc-2；
- F1；
- Acc-7。

## 14.2 子集性能

分别报告：

\[
t,\ a,\ v,\ ta,\ tv,\ av,\ tav.
\]

以及七种输入平均性能。

## 14.3 交互拟合

\[
E_2
=
\frac{1}{3N}
\sum_i
\sum_{A\in\{ta,tv,av\}}
|\mu_T^{(i)}(A)-\mu_S^{(i)}(A)|.
\]

\[
E_3
=
\frac{1}{N}
\sum_i
|\mu_T^{(i)}(tav)-\mu_S^{(i)}(tav)|.
\]

## 14.4 高交互子集

按教师交互强度：

\[
I_T(x)
=
|\mu_T(ta)|
+|\mu_T(tv)|
+|\mu_T(av)|
+|\mu_T(tav)|.
\]

将测试集划分为：

- Low；
- Medium；
- High interaction。

Low/Medium/High 的两个分位点只根据验证集教师 \(I_T(x)\) 分布确定，例如验证集的 33% 和 67% 分位点；冻结阈值后应用到测试集。不得根据测试集自身分位数重新划分。

主方法应重点比较 High interaction 子集。

## 14.5 模态删除指标

仅作为解释指标：

\[
u_t=V(tav)-V(av),
\]

\[
u_a=V(tav)-V(tv),
\]

\[
u_v=V(tav)-V(ta).
\]

报告教师与学生 deletion contribution 的 MAE，但不作为独立主损失。

## 14.6 文本主导指标

\[
TDI
=
\frac{|u_t|}
{|u_t|+|u_a|+|u_v|+\epsilon}.
\]

在教师非文本贡献较高的样本中，检查学生是否仍过度依赖文本。

## 14.7 信息率

报告：

- 总 nats/sample；
- text nats/sample；
- audio nats/sample；
- video nats/sample。

## 14.8 计算与容量

### 计算

- 12 个情感 token；
- 视频帧数；
- GFLOPs；
- 单样本延迟；
- 峰值显存。

### 参数

- 总参数；
- 可训练参数；
- LoRA 参数；
- 冻结参数。

## 14.9 统计检验

每个核心模型至少 3 个随机种子，建议 5 个种子用于最终主表。

报告：

- mean；
- standard deviation；
- paired bootstrap 或 permutation test；
- effect size；
- 95% confidence interval。

测试必须在相同样本级预测上配对。

---

# 15. 教师准入门槛

在开始学生交互蒸馏前，Qwen3-Omni 教师必须通过以下检查。

## 15.1 性能门槛

- \(tav\) 显著优于普通学生；
- \(tav\) 不低于 text-only；
- 音频、视频子集不退化到标签先验；
- Probe 回归相关性为正且达到可用水平。

## 15.2 子集多样性

计算七个子集预测的样本内方差。若绝大多数样本七个输出几乎相同，交互目标无意义。

## 15.3 重复稳定性

在 500 个样本上使用：

- 相同输入重复前向；
- 不同等价 Prompt；
- 可选 Probe dropout 多次采样。

检查：

- 二阶交互符号一致率；
- 三阶交互符号一致率；
- 交互数值 ICC 或相关系数。

## 15.4 校准

温度校准后：

- NLL 应下降；
- ECE 应下降；
- 若没有改善，检查 Probe 和标签定义。

## 15.5 人工抽查

随机检查至少 100 条：

- 明显正面；
- 明显负面；
- 文本与语气可能冲突；
- 文本与表情可能冲突；
- 低质量音频；
- 人脸不可见。

若教师经常虚构缺失模态或不能利用非文本模态，不批准进入交互蒸馏。

---

# 16. 本地服务器部署与成本

## 16.1 服务器假设

当前规划基于：

- 2 × NVIDIA RTX 6000D；
- 每卡约 84 GB 显存；
- 125 GiB 系统内存；
- 约 321 GB 可用磁盘；
- CMU_MOSEI 原始数据约 169 GB。

正式运行前必须确认：

```bash
nvidia-smi
ls -l /dev/nvidia*
```

均正常。

## 16.2 教师部署

建议：

```yaml
precision: bf16
batch_size: 1
attn_implementation: flash_attention_2
disable_talker: true
use_audio_in_video: false
parallel_strategy: benchmark_then_lock
candidate_loaders:
  - transformers_native_tp
  - accelerate_device_map_balanced
```

双卡用于同一个教师实例，不建议每卡各部署一个完整 Qwen3-Omni 实例。阶段 B 完成后必须把 `parallel_strategy` 替换为实测选定的唯一加载方式，并保存可执行加载代码；若使用 `device_map="balanced"`，文中称为模型切分，不称为 tensor parallel。

## 16.3 预算估算

以下为规划区间，必须用 500 样本实测修正：

| 工作 | 预计 GPU·小时 | 双卡墙钟 |
|---|---:|---:|
| 500 样本、7 子集技术验证 | 8–20 | 4–10 小时 |
| 全量 7 子集教师缓存 | 100–250 | 2.5–6 天 |
| 单次主学生训练 | 20–50 | 10–25 小时 |
| 5 组核心配置 × 3 seeds | 300–600 | 7–14 天 |
| 最终完整核心实验 | 500–800 | 约 2 周 |
| 大规模完整消融 | 1,000–2,000 | 3–6 周 |

GPU·小时按照单卡小时累计；双卡运行一小时计 2 GPU·小时。

## 16.4 阶段预算上限

### 第一阶段

- 500–1,000 样本；
- 不超过 30 GPU·小时；
- 不超过 30 GB 新增磁盘；
- 验证教师、音轨和坐标对照。

### 第二阶段

只有第一阶段通过后：

- 全量训练和验证集教师缓存；
- 不超过 300 GPU·小时；
- 不超过 80 GB 新增磁盘。

### 最终阶段

核心论文实验总预算建议控制在：

- 500–800 GPU·小时；
- 150–180 GB 新增磁盘；
- 只保留 best、last 和 recovery checkpoint。

## 16.5 磁盘策略

不保存：

- 逐帧 PNG；
- 教师完整 token hidden sequence；
- 多份重复编码视频；
- 所有 epoch checkpoint。

保存：

- 无声视频或可重复生成脚本，二选一；
- 独立音频；
- 教师标量缓存；
- 可选单个池化 hidden vector；
- 数据 manifest；
- best/last/recovery checkpoint。

---

# 17. 工程单元测试

## 17.1 音轨泄漏测试

对每个无声视频：

```bash
ffprobe -v error -select_streams a \
-show_entries stream=index -of csv=p=0 silent.mp4
```

输出必须为空。

## 17.2 子集输入测试

记录每次调用的：

- 文本是否存在；
- 音频文件路径；
- 视频文件路径；
- `use_audio_in_video`；
- 实际 processor 输出的 modality tensor shape。

## 17.3 重复音频测试

对 \(av\)、\(tav\) 确认：

- 视频无音轨；
- 只有一份独立音频 tensor；
- 不存在视频内音频与外部 WAV 双重输入。

## 17.4 Probe token 测试

构造不同长度 batch，确认索引得到的是：

> 每个样本最后一个有效输入 token，而不是 padding token。

## 17.5 生成泄漏测试

确认 hidden state 只来自输入前向，不包含模型生成的标签文本。

## 17.6 Möbius 反演测试

随机生成价值向量，检查：

\[
\mathbf v
\rightarrow
\boldsymbol\mu
\rightarrow
\widehat{\mathbf v}
\]

数值误差小于 \(10^{-6}\)。

## 17.7 损失等规模测试

Subset、Interaction、Random Coordinate 的初始 loss 量级和梯度范数需要记录，防止某方法仅因数值尺度更大而获益。

## 17.8 数据泄漏测试

确认同一原始视频 ID 不跨 train/validation/test。

---

# 18. 理论结论与不可过度声明的内容

## 18.1 可以严格成立

### 完备性

\[
V_F(tav)-b
=
\sum_{\varnothing\neq A\subseteq\{t,a,v\}}
\mu_F(A).
\]

### 交互误差界

令：

\[
\delta_A=\mu_S(A)-\mu_T(A).
\]

则：

\[
\left|
[V_S(tav)-b]-[V_T(tav)-b]
\right|
\le
\sum_A|\delta_A|.
\]

若使用平方误差：

\[
|\Delta V_S-\Delta V_T|
\le
\sqrt7
\left(
\sum_A\delta_A^2
\right)^{1/2}.
\]

### KL 到概率差异

\[
\|p_T-p_S\|_{\mathrm{TV}}
\le
\sqrt{
\frac12 KL(p_T\Vert p_S)
}.
\]

### VIB 上界代理

\[
I(X;Z)
\le
\mathbb E_X
KL(q(Z|X)\Vert p(Z)).
\]

## 18.2 必须明确的限制

- Harsanyi 坐标不增加原始子集监督的信息；
- 模态删除不是因果 \(do\) 干预；
- deletion contribution 不是严格 PID unique information；
- VIB KL 是互信息上界代理，不是精确互信息；
- 12 个 token 不是完整模型容量；
- 教师更大不意味着所有子集知识可靠；
- 交互项描述模型行为，不等同真实心理机制。

---

# 19. 消融矩阵

| 编号 | Task | Full KD | Subset Value | Möbius Full | High-order | Random Coord | VIB |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | ✓ |  |  |  |  |  |  |
| E1 | ✓ | ✓ |  |  |  |  |  |
| E2 | ✓ | ✓ | ✓ |  |  |  |  |
| E3 | ✓ | ✓ |  | ✓ |  |  |  |
| E4 | ✓ | ✓ |  |  | ✓ |  |  |
| E5 | ✓ | ✓ |  |  |  | ✓ |  |
| E6 | ✓ | ✓ |  |  | ✓ |  | ✓ |
| E7 | ✓ | ✓ | ✓ |  |  |  | ✓ |

## 19.1 阶次消融

- 一阶；
- 二阶；
- 三阶；
- 一阶 + 二阶；
- 二阶 + 三阶；
- 全阶。

## 19.2 坐标矩阵消融

- Identity；
- Möbius；
- Random orthogonal；
- Random invertible；
- Whitened Möbius。

## 19.3 价值函数消融

- 回归分数；
- 分类 true-label margin；
- 校准 log-odds。

## 19.4 教师模型对照

主论文教师固定为 Qwen3-Omni Instruct。可选附录仅比较一次 Thinking 模型，不用于大规模消融。

---

# 20. 阶段性实施计划

## 阶段 1：数据与教师可行性

范围：

- 500–1,000 条样本；
- Qwen3-Omni；
- 七子集；
- Probe；
- 无 VIB 学生。

产出：

- 吞吐报告；
- 显存报告；
- 音轨单元测试；
- 教师子集稳定性；
- Full KD、Subset、Interaction 公平比较。

## 阶段 2：全量教师缓存

前提：

- 教师通过准入门槛；
- Interaction 至少未明显差于 Subset；
- 驱动和长视频策略稳定。

产出：

- train/validation 教师缓存；
- 缓存 checksum；
- 失败样本清单；
- 教师分布分析。

测试集七子集缓存不在本阶段提前生成；它在最终方案和验证集阈值全部冻结后生成一次，仅供最终评估。

## 阶段 3：核心学生实验

运行：

- E0–E7；
- 3 个种子；
- 高交互子集分析；
- 缺失模态测试。

## 阶段 4：率失真

扫描：

\[
\beta=\{0,10^{-6},10^{-5},10^{-4},10^{-3}\}.
\]

只在 E4 和最佳 Subset 基线运行完整扫描。

## 阶段 5：最终结果

- 5 个种子主表；
- 统计显著性；
- 计算成本；
- 理论和限制；
- 案例分析。

---

# 21. Go/No-Go 验收标准

## 21.1 第一阶段 Go 条件

同时满足：

1. Qwen3-Omni 七子集推理 OOM 和失败率可控；
2. 音轨泄漏单元测试全部通过；
3. 完整模态教师优于 text-only；
4. 教师子集输出具有明显差异；
5. 主要二阶交互符号具有基本重复稳定性；
6. Full Möbius-7 在等监督、等前向条件下至少不差于 Full Subset-7，且 Interaction-4 在等维、等前向条件下至少不差于 Subset-4；
7. Random coordinate 不能解释 Interaction 的全部收益。

## 21.2 全量实验 Go 条件

至少满足以下之一：

- Interaction 在 3 个种子上显著优于 Subset；
- Interaction 在 High-interaction 子集上有稳定优势；
- Interaction 在相同任务性能下需要更低 VIB rate；
- Interaction 在相同 rate 下有更低教师行为失真。

## 21.3 No-Go 条件

任一情况出现时停止全量交互蒸馏：

- 教师音频/视频子集接近随机；
- 子集输出近乎相同；
- Interaction 与随机坐标没有差异；
- 交互符号高度不稳定；
- VIB 不能形成可重复的 rate–distortion 趋势；
- 成本超过预算且前期无正向证据。

---

# 22. 风险与应对

## 风险 1：Qwen3-Omni 全量缓存成本高

应对：

- 500 样本先外推；
- 使用阶段 B 实测选定的双卡原生 TP 或模型切分方式；
- 关闭 Talker；
- 固定帧数；
- 标量缓存；
- 训练阶段只使用 train/validation 缓存；最终冻结后再生成一次 test 缓存；
- 测试集不参与超参选择。

## 风险 2：长视频 OOM

应对：

- 按时长分桶；
- batch 1；
- 超过 30 秒窗口聚合；
- 记录长视频专门结果。

## 风险 3：教师隐藏表示接口变化

应对：

- 锁定 Transformers 版本；
- 写 Probe extraction 单元测试；
- 保存模型 commit；
- 必要时对 Thinker 最后 block 注册 hook。

## 风险 4：交互仅是损失尺度变化

应对：

- 梯度范数归一化；
- 随机正交和随机可逆对照；
- whitened coordinate；
- 等维对照；
- 相同搜索预算。

## 风险 5：学生文本主导

应对：

- 模态 dropout；
- 高交互权重；
- 报告 TDI；
- 单模态/双模态辅助 forward；
- 在教师非文本贡献高的子集单独评估。

## 风险 6：VIB posterior collapse

应对：

- KL warm-up；
- free bits；
- \(\beta\) 小范围扫描；
- 先无 VIB 训练再开启；
- 分模态报告 KL。

## 风险 7：MOSEI 缺少人工单模态标签

应对：

- 不宣称模态贡献是真实人工解释；
- 把教师交互视为教师行为目标；
- 用缺失模态性能、人工案例和噪声实验做外部验证；
- 后续可在 CH-SIMS 作为附加外部验证，但不属于本方案主数据集。

---

# 23. 论文贡献表述

建议写成：

1. **行为坐标化蒸馏。**  
   将同一组多模态子集教师行为从原始价值坐标转换为 Möbius 交互坐标，并研究不同坐标对有限容量学生优化的影响。

2. **高阶交互导向的归纳偏置。**  
   从同一七子集教师缓存中选择二阶和三阶交互监督子空间，并与等维子集方向比较；该实验研究监督方向选择，不声称四维目标与完整七维目标信息等价。

3. **率失真联合分析。**  
   使用变分信息率代理，研究学生信息率、教师行为失真和任务性能之间的关系。

4. **严格公平评估。**  
   在相同子集、相同前向次数、相同损失维数下，与直接子集匹配、随机正交坐标和随机可逆坐标比较。

5. **Qwen3-Omni 全模态教师缓存。**  
   基于严格隔离的文本、独立音频和无声视频构造七种子集，避免视频音轨泄漏破坏交互分解。

## 23.1 不应写入贡献的内容

- “首次发现 Harsanyi 包含额外教师知识”；
- “模态删除得到因果作用”；
- “deletion contribution 就是 PID unique information”；
- “12 个 token 等于 12 单位信息容量”。

---

# 24. 推荐论文结构

## 1 Introduction

- 大模型教师强但部署昂贵；
- 普通完整输入 KD 难以保持子集行为；
- 子集行为和交互坐标信息等价，但优化偏置可能不同；
- 提出 RDID-MOSEI。

## 2 Related Work

- Multimodal Sentiment Analysis；
- Omni-modal Large Models；
- Knowledge Distillation；
- Interaction Attribution；
- Information Bottleneck。

## 3 Method

### 3.1 Problem Definition  
### 3.2 Qwen3-Omni Subset Teacher  
### 3.3 Fixed-baseline Value Function  
### 3.4 Interaction-coordinate Reparameterization  
### 3.5 Rate-constrained Student  
### 3.6 Training Objective  

## 4 Analysis

- Coordinate equivalence；
- error bounds；
- rate proxy；
- computational complexity。

## 5 Experiments

- CMU-MOSEI；
- teacher gate；
- fair baselines；
- main results；
- high-interaction subset；
- rate–distortion；
- missing modalities；
- statistics。

## 6 Limitations

明确写出：

- 不是因果方法；
- 教师行为不是人工真值；
- 交互坐标不增加信息；
- Qwen3-Omni 成本高；
- 仅以 MOSEI 为主数据集的外部效度有限。

---

# 25. 配置文件草案

```yaml
project:
  name: rdid_mosei
  version: "2.1"
  seed: 42

dataset:
  name: CMU_MOSEI
  root: /home/wy/sjq/kd/dataset/cmu_mosei_source
  use_official_split: true
  local_utterances: 22856
  audio_sample_rate: 16000
  video_frames: 16
  max_direct_duration_sec: 30
  long_sample_strategy: sliding_window
  create_silent_video: true

teacher:
  model: Qwen/Qwen3-Omni-30B-A3B-Instruct
  precision: bf16
  backend: transformers
  transformers_min_version: "5.2.0"
  parallel_strategy: benchmark_then_lock
  candidate_loaders: [transformers_native_tp, accelerate_device_map_balanced]
  attn_implementation: flash_attention_2
  disable_talker: true
  use_audio_in_video: false
  batch_size: 1
  prompt_version: v1
  subsets: [t, a, v, ta, tv, av, tav]

teacher_probe:
  backbone_frozen: true
  representation: last_valid_input_token
  generated_tokens_allowed: false
  shared_across_subsets: true
  hidden_size: 512
  dropout: 0.1
  tasks: [regression, classification]
  classification_classes: 7
  classification_target: clipped_rounded_sentiment
  calibration: temperature_scaling

student:
  text_encoder: /home/wy/sjq/kd/model/Qwen3-0.6B-Base
  audio_encoder: /home/wy/sjq/kd/model/WavLM-Base-Plus
  video_encoder: /home/wy/sjq/kd/model/VideoMAE-Base
  tokens_per_modality: 4
  bottleneck_dim: 256

  qformer:
    layers: 2
    hidden_size: 512
    heads: 8
    ffn_size: 2048

  fusion:
    layers: 4
    hidden_size: 512
    heads: 8
    ffn_size: 2048
    dropout: 0.1

value:
  primary: regression_score
  empty_baseline: train_label_mean
  auxiliary: true_label_logit_margin

interaction:
  primary_orders: [2, 3]
  pair_weight: 1.0
  triple_weight: 1.0
  confidence_weighting: disabled_in_v1
  robust_loss: huber
  use_unique_loss: false

vib:
  enabled: true
  prior: standard_normal
  beta_grid: [0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3]
  warmup_epochs: 5
  report_unit: nats_per_sample

training:
  precision: bf16
  optimizer: AdamW
  lr_new_modules: 1.0e-4
  lr_lora: 3.0e-5
  lr_encoder_top: 1.0e-5
  weight_decay: 0.01
  grad_clip: 1.0
  epochs_warmup: 5
  epochs_joint: 20
  seeds: [13, 42, 2026]

experiments:
  - student
  - full_kd
  - subset_value
  - mobius_full
  - high_order_interaction
  - random_orthogonal
  - high_order_interaction_vib

report:
  task_metrics: [mae, pearson, acc2, f1, acc7]
  acc2_zero_policies: [non_zero, zero_as_nonnegative]
  teacher_metrics: [ece, nll, brier]
  behavior_metrics: [pair_interaction_mae, triple_interaction_mae, tdi]
  rate_metrics: [total_nats, text_nats, audio_nats, video_nats]
  compute_metrics: [gflops, latency, peak_vram]
  capacity_metrics: [total_params, trainable_params, frozen_params]
```

---

# 26. 伪代码

## 26.1 教师缓存

```python
SUBSETS = ["t", "a", "v", "ta", "tv", "av", "tav"]

for sample in dataset:
    cache = {}

    for subset in SUBSETS:
        inputs = build_subset_input(
            sample=sample,
            subset=subset,
            silent_video=sample.silent_video,
            standalone_audio=sample.audio_wav,
            use_audio_in_video=False,
        )

        hidden = extract_last_valid_input_hidden(
            model=qwen3_omni_thinker,
            inputs=inputs,
            include_generated_tokens=False,
        )

        reg_score, cls_logits = shared_probe(hidden)

        cache[subset] = {
            "reg_score": float(reg_score),
            "cls_logits": cls_logits.cpu().tolist(),
            "entropy": entropy(softmax(cls_logits)),
        }

    save_cache(sample.id, cache)
```

## 26.2 交互计算

```python
def mobius_interactions(values: dict[str, torch.Tensor], baseline):
    mu = {}

    mu["t"] = values["t"] - baseline
    mu["a"] = values["a"] - baseline
    mu["v"] = values["v"] - baseline

    mu["ta"] = values["ta"] - values["t"] - values["a"] + baseline
    mu["tv"] = values["tv"] - values["t"] - values["v"] + baseline
    mu["av"] = values["av"] - values["a"] - values["v"] + baseline

    mu["tav"] = (
        values["tav"]
        - values["ta"]
        - values["tv"]
        - values["av"]
        + values["t"]
        + values["a"]
        + values["v"]
        - baseline
    )
    return mu
```

## 26.3 学生训练

```python
for batch in train_loader:
    # 编码器每个模态只运行一次
    encoded = student.encode_modalities(batch)

    # 共享编码结果，构造七个子集
    outputs = {
        subset: student.forward_from_encoded(encoded, subset)
        for subset in SUBSETS
    }

    teacher = load_teacher_cache(batch.ids)

    loss_task = task_loss(
        outputs["tav"],
        labels=batch.labels,
    )

    loss_full = full_kd_loss(
        student=outputs["tav"],
        teacher=teacher["tav"],
    )

    student_values = {
        s: outputs[s].reg_score for s in SUBSETS
    }
    teacher_values = {
        s: teacher[s].reg_score for s in SUBSETS
    }

    mu_s = mobius_interactions(student_values, train_label_mean)
    mu_t = mobius_interactions(teacher_values, train_label_mean)

    loss_interaction = sum(
        huber(mu_s[a], mu_t[a])
        for a in ["ta", "tv", "av", "tav"]
    )

    loss_rate = sum(outputs["tav"].rate_kl_by_modality.values())

    loss = (
        loss_task
        + lambda_full * loss_full
        + lambda_interaction * loss_interaction
        + beta * loss_rate
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
```

## 26.4 公平坐标对照

```python
# v_t 和 v_s 为相同顺序的 7 维中心化子集价值
loss_subset = huber(v_s, v_t).mean()

mu_t = mobius_matrix @ v_t
mu_s = mobius_matrix @ v_s
loss_mobius = huber(mu_s, mu_t).mean()

r_t = random_orthogonal_matrix @ v_t
r_s = random_orthogonal_matrix @ v_s
loss_random_orthogonal = huber(r_s, r_t).mean()
```

三个实验必须使用相同 batch、相同七子集 forward 和相同训练步数。

---

# 27. 参考资料

1. QwenLM. **Qwen3-Omni Official Repository.**  
   https://github.com/QwenLM/Qwen3-Omni

2. Xu, J., et al. **Qwen3-Omni Technical Report.**  
   https://arxiv.org/abs/2509.17765

3. Zadeh, A. B., et al. **Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph.** ACL 2018.  
   https://aclanthology.org/P18-1208/

4. Hsu, W.-N., et al. **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.**  
   https://arxiv.org/abs/2106.07447

5. Tong, Z., et al. **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.**  
   https://arxiv.org/abs/2203.12602

6. Yang, A., et al. **Qwen3 Technical Report.**  
   https://arxiv.org/abs/2505.09388

---

# 最终执行结论

本方案批准使用：

```text
教师：Qwen3-Omni-30B-A3B-Instruct（关闭 Talker）
数据集：CMU-MOSEI
学生：Qwen3-0.6B + HuBERT-Base + VideoMAE-Small
```

但执行顺序必须是：

```text
500–1,000 样本教师验证
→ 公平比较 Full Subset-7 / Full Möbius-7 / Subset-4 / Interaction-4 / Random Coordinate
→ 通过 Go 条件
→ 全量 train/validation 教师缓存
→ 核心学生实验
→ VIB 率失真扫描
→ 冻结方案后生成一次 test 教师缓存并最终评估
```

若实验支持，论文的中心结论限定为：

> 在完整七维监督下，交互坐标重参数化改善了有限信息率学生的优化结果；在相同教师缓存、目标维数和计算预算下，高阶交互监督子空间比等维直接子集方向更有效地保留跨模态情感行为。

只有当 Interaction 在严格公平对照下稳定优于 Subset 和随机坐标后，才进入完整论文实验。
