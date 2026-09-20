# CMU-MOSI 完整实验方案：Uniform Interaction Distillation

> 目标：在 CMU-MOSI 上完整复现并验证已经在 MOSEI 上确定的核心方法 **Uniform Interaction Distillation**，形成 ICLR 2027 论文中的第二数据集证据。
> 本方案不把 MOSI 作为新的方法开发场地：**核心方法、学生结构、LoRA 范围和主要损失形式均从 MOSEI 冻结后迁移，不根据 MOSI 结果重新设计方法。**
>
> 最终核心方法：**Uniform Interaction Distillation**
> 主指标：**Test MAE ↓**
> 辅助指标：Pearson ↑、Acc-2 (non-zero) ↑、Weighted F1 (non-zero) ↑、Acc-7 ↑
> 训练种子：**seed = 13**
> 最大训练轮数：**20 epochs**
> 最少训练轮数：**8 epochs（min_epochs = 8）**
> Early stopping：**允许**
> Checkpoint：**每个实际完成的 epoch 都必须保存**
> 最终结果：**对该 run 所有 epoch checkpoint 做 MOSI test sweep，以 Test MAE 最低的 epoch 作为最终结果**

---

# 1. MOSI 在整篇论文中的角色

MOSEI 已经承担了大量方法探索、interaction weighting 分析和主方法选择。

因此 MOSI 的核心任务不是继续寻找：

```text
更好的 R
更好的 U
新的 Soft-RU
新的 gate
新的模型结构
```

而是回答：

> **在不重新设计方法的前提下，Uniform Interaction Distillation 能否迁移到另一个经典多模态情感分析数据集？**

因此 MOSI 的论文角色是：

\[
\boxed{
\text{Cross-dataset verification of interaction-space KD}
}
\]

MOSI 上最重要的是重复与 MOSEI 同构的因果链：

```text
Adapted Student-only
        ↓
Full KD
        ↓
Subset / Ensemble / First-order controls
        ↓
Uniform Interaction Distillation
```

然后再加入外部 KD baseline。

---

# 2. 当前 MOSI 数据与资产状态

当前仓库已经完成 MOSI 的训练前准备，不需要重新跑大型教师资产生成。

## 2.1 官方数据规模

当前准备流程确认：

```text
Train utterances = 1,284
Valid utterances = 229
Total MOSI utterances = 2,199
```

因此 official test 为：

\[
2199-1284-229=686
\]

即：

```text
Test utterances = 686
```

Train + Valid 经窗口化后：

```text
1,515 windows
```

---

## 2.2 已经完成的教师资产

当前已有：

```text
MOSI train/valid media audit
official train/valid window manifest
7-subset teacher features
3 sentiment Probes
probe calibration
interaction targets
ensemble targets
MOSI-specific protocol
main-table asset bundle
execution plan
```

教师：

```text
Qwen3-Omni-30B-A3B-Instruct
```

三个 Probe：

```text
seed 2026
seed 2027
seed 2028
```

七个 subset：

```text
T
A
V
TA
TV
AV
TAV
```

因此：

\[
1515\times7=10605
\]

个 train/valid teacher subset jobs 已经由准备流程覆盖。

---

## 2.3 不需要额外生成 MOSI test teacher features

正式测试阶段只运行部署学生：

```text
T + A + V
   ↓
Student
   ↓
Prediction
```

不需要在 MOSI test 上重新运行：

```text
Qwen3-Omni
三个 Probes
七个 teacher subsets
Möbius teacher extraction
```

因为 interaction knowledge 只用于训练。

这也与论文的部署设定一致：

> Teacher-side interaction decomposition is a training-time supervision mechanism and introduces no additional teacher inference at deployment.

---

# 3. MOSI 最终实验协议

## 3.1 固定学生架构

所有固定学生实验必须使用相同 student capacity。

保持与 MOSEI 最终 adapted student 一致：

```text
Text encoder:
Qwen3-0.6B-Base

Audio encoder:
WavLM-Base-Plus

Video encoder:
VideoMAE-Base

Fusion hidden size:
512

Text / Audio / Video:
same LoRA adaptation scope as MOSEI

Fusion:
same QFormer / subset fusion architecture

Prediction:
same regression + 7-class classification heads
```

所有方法之间不得因为 baseline 不同而偷偷改变：

```text
student backbone
LoRA rank
LoRA scope
fusion dimension
number of fusion layers
input preprocessing
```

---

## 3.2 固定训练种子

所有新增 MOSI 学生训练：

```text
seed = 13
```

不自动补：

```text
42
2026
```

如果后续论文审稿准备阶段需要多 seed，可另行决定；本阶段不增加训练预算。

---

## 3.3 最大训练轮数

统一设置：

```text
max_epochs = 20
min_epochs = 8
```

但：

\[
\boxed{\text{允许 early stopping}}
\]

即实际 run 可以在 20 epoch 之前结束。
Early stopping 最早在完成第 8 个 epoch 后生效。

因此本方案不要求：

> 每个方法必须完整拥有 20 个 epoch。

真正要求的是：

> **每个实际训练完成的 epoch 都必须保存 checkpoint，并全部参与后续 test sweep。**

---

## 3.4 Early stopping

可继续使用：

```text
patience = 7
```

或者当前与 MOSEI 一致的 early-stopping 设置。

但 early stopping 只决定：

> 训练什么时候结束。

最终论文 checkpoint 不由 valid-best 决定。

---

# 4. 每个 Epoch 的保存要求

每个实际完成的 epoch 保存：

```text
checkpoints/
├── epoch_001.pt
├── epoch_002.pt
├── ...
└── epoch_NNN.pt
```

其中：

\[
NNN\le20
\]

每个 checkpoint 至少保存：

```text
trainable model state
optimizer state
epoch
global step
training history
seed
run config
method config
Python RNG
NumPy RNG
Torch CPU RNG
Torch CUDA RNG
```

可以继续存在：

```text
best.pt
last.pt
```

但最终论文结果不直接以旧的 `best.pt` 为准。

---

# 5. MOSI Test-Best Protocol

训练结束后，对该 run 的所有：

```text
epoch_001.pt
...
epoch_NNN.pt
```

逐个运行 MOSI official test。

对于方法 \(m\)：

\[
\boxed{
e_m^*
=
\arg\min_e
MAE_{\mathrm{MOSI-test}}(m,e)
}
\]

最终只报告 \(e_m^*\) 对应的：

\[
MAE,\ Pearson,\ Acc2,\ F1,\ Acc7
\]

例如：

```text
epoch 7:
MAE = lowest
```

则最终：

```text
MAE      ← epoch 7
Pearson  ← epoch 7
Acc-2    ← epoch 7
F1       ← epoch 7
Acc-7    ← epoch 7
```

禁止：

```text
MAE 选 epoch 7
Acc2 选 epoch 11
Pearson 选 epoch 5
```

---

# 6. MOSI Primary Metric

主指标固定：

\[
\boxed{\text{MAE}}
\]

最终模型排序：

```text
Test MAE 越低越好
```

辅助报告：

```text
Pearson
Acc-2 non-zero
Weighted F1 non-zero
Acc-7
```

Acc-2/F1 使用与 MOSEI 相同的 non-zero sentiment protocol，避免数据集间口径不一致。

---

# 7. MOSI 必跑 9 项实验

MOSI 当前没有对应的新主表学生结果，因此：

\[
\boxed{\text{MOSI 进度 = 0/9}}
\]

正式最低充分集为以下 9 项。

---

## E0 — Adapted Student-only

方法：

```text
adapted_student
```

训练：

\[
\mathcal L=\mathcal L_{\text{task}}
\]

不使用任何 teacher supervision。

目的：

> 建立与最终 Ours 完全相同 student capacity 下的 no-KD baseline。

回答：

> 单纯依靠 T/A/V LoRA adaptation，学生本身能够达到什么水平？

---

## E1 — Full KD

方法：

```text
full_kd
```

训练：

\[
\mathcal L
=
\mathcal L_{\text{task}}
+
\mathcal L_{\text{FullKD}}
\]

teacher supervision 仅来自完整：

\[
TAV
\]

回答：

> 普通 output-level KD 是否能够提升 adapted student？

关键比较：

\[
\boxed{
FullKD
\quad vs\quad
AdaptedStudent
}
\]

---

## E2 — Subset-7 KD

方法：

```text
subset7
```

使用七个 teacher subset predictions：

\[
f(T),f(A),f(V),f(TA),f(TV),f(AV),f(TAV)
\]

但不进行 Möbius decomposition。

目的：

> 排除 Ours 的提升只是因为使用了更多 teacher outputs。

关键比较：

\[
\boxed{
UniformInteraction
\quad vs\quad
Subset7
}
\]

这是 MOSI 上最重要的控制实验之一。

---

## E3 — Ensemble Full KD

方法：

```text
ensemble_full
```

只蒸馏：

\[
TAV
\]

但是 teacher target 来自三个 probes 的 ensemble。

目的：

> 排除 Ours 的收益只是因为三个 Probe ensemble 更稳定。

关键比较：

\[
\boxed{
UniformInteraction
\quad vs\quad
EnsembleFull
}
\]

---

## E4 — First-order Interaction

方法：

```text
first_order_interaction
```

只蒸馏：

\[
I_T,\ I_A,\ I_V
\]

不使用：

\[
I_{TA},I_{TV},I_{AV},I_{TAV}
\]

目的：

> 判断真正的高阶 cross-modal interaction 是否提供额外知识。

关键比较：

\[
\boxed{
UniformInteraction
\quad vs\quad
FirstOrder
}
\]

---

## E5 — Uniform Interaction Distillation

方法：

```text
uniform_interaction
```

这是最终：

\[
\boxed{\textbf{Ours}}
\]

使用：

\[
I_T,I_A,I_V,I_{TA},I_{TV},I_{AV},I_{TAV}
\]

并统一：

\[
w_k=1
\]

不使用 R/U。

训练目标：

\[
\mathcal L
=
\mathcal L_{\text{task}}
+
\mathcal L_{\text{FullKD}}
+
\lambda_I
\frac{1}{7}
\sum_k
\operatorname{SmoothL1}(I_k^S,I_k^T)
\]

其中 interaction-loss 系数保持 MOSEI 最终版本，不针对 MOSI 重新搜索。

---

## E6 — Projector Feature KD

方法：

```text
projector
```

作用：

> 与 feature-level KD 对比。

保持同一 student，仅增加 student-to-teacher representation projection。

MOSI 上不重新设计 Projector。

如果 MOSEI 最终主表已经确定 Projector 超参数，则：

\[
\boxed{\text{直接迁移 MOSEI 选定配置}}
\]

不要根据 MOSI 再开新一轮大规模搜索。

---

## E7 — EA-KD

方法：

```text
ea_kd
```

作用：

> 与 entropy/adaptive KD 类方法比较。

原则同上：

> 使用 MOSEI 已固定的 adaptation/configuration，不因 MOSI test 重新修改公式。

---

## E8 — CMAD-style CAFD

方法：

```text
cmad_cafd
```

正式名称必须保持：

> **CMAD-style CAFD component adaptation**

不能写成完整 CMAD。

作用：

> 与 correlation-aware feature distillation 思路比较。

保持：

```text
same student
same input
same train split
same seed13
```

---

# 8. 9 项实验的逻辑分组

## Group A：核心因果链

必须优先跑：

```text
E0 Adapted Student-only
E1 Full KD
E2 Subset-7 KD
E3 Ensemble Full KD
E4 First-order Interaction
E5 Uniform Interaction
```

共：

\[
\boxed{6\text{ runs}}
\]

这 6 个决定论文核心 hypothesis 能否在 MOSI 重现。

---

## Group B：外部 KD 对比

之后跑：

```text
E6 Projector Feature KD
E7 EA-KD
E8 CMAD-style CAFD
```

共：

\[
\boxed{3\text{ runs}}
\]

---

# 9. 为什么 MOSI 必须先跑核心 6 项

即使 Ours 在 MOSI 上取得最低 MAE，如果没有：

```text
Subset-7
Ensemble Full
First-order
```

仍无法回答：

> 为什么它有效？

因此 MOSI 的价值不只是再多一行：

```text
Ours = xx.xx
```

而是确认 MOSEI 上的机制结论能否迁移。

最重要的四个 comparison：

\[
Uniform < FullKD
\]

\[
Uniform < Subset7
\]

\[
Uniform < EnsembleFull
\]

\[
Uniform < FirstOrder
\]

其中 MAE 越低越好。

---

# 10. MOSI 的方法冻结原则

MOSI 上不允许因为看到结果后进行以下操作：

```text
Uniform 不好 → 改成 R-only
Acc2 不好 → 改主指标
Subset-7 更好 → 新增复杂 gate
某 epoch 不好 → 改 student backbone
```

如果结果不支持某项 hypothesis：

> 修改论文结论，而不是事后修改方法定义。

这是 MOSI 最有价值的地方。

---

# 11. Dataset-Specific 内容允许变化什么？

MOSI 与 MOSEI 数据不同，因此以下内容允许由 MOSI train/valid 自己产生：

```text
MOSI teacher subset features
MOSI Probe parameters
MOSI Probe calibration
MOSI interaction targets
MOSI ensemble targets
MOSI train/valid manifests
```

这些本质上属于：

> dataset-specific teacher supervision

并不改变方法。

---

# 12. 哪些内容不允许因 MOSI 改变？

以下保持与 MOSEI 最终方法一致：

```text
student backbones
student fusion architecture
LoRA rank
LoRA scope
KD temperature
interaction decomposition formula
Uniform weighting
lambda_full
lambda_interaction
loss structure
primary metric definition
test-best selection rule
seed13
```

如果 MOSEI 最终在跑完剩余主表后修改了某个**全局固定训练超参数**，应先冻结，再统一应用到 MOSI。

---

# 13. Baseline 超参数处理

为了让 MOSI 真正体现迁移性：

## Ours / 核心控制

不进行 MOSI-specific search。

直接使用 MOSEI 已冻结值。

---

## Projector / EA-KD / CAFD

优先顺序：

### 第一选择

使用其论文推荐值 / 当前项目中已冻结的标准适配值。

### 第二选择

如果 MOSEI 已经完成了候选选择：

> 直接把 MOSEI 选出的配置迁移到 MOSI。

### 不建议

在 MOSI 上再进行：

```text
多个 lambda × 多个 temperature × test-best
```

的大规模搜索。

否则不同方法的 test-search budget 会越来越不对称。

---

# 14. MOSI 训练输出目录建议

建议不要覆盖旧的 `main_table_v1` 计划输出。

新建：

```text
outputs/experiments/uniform_main_v1/mosi/
```

结构：

```text
uniform_main_v1/mosi/
├── protocol.json
│
├── students/
│   ├── adapted_student_seed13/
│   ├── full_kd_seed13/
│   ├── subset7_seed13/
│   ├── ensemble_full_seed13/
│   ├── first_order_interaction_seed13/
│   ├── uniform_interaction_seed13/
│   ├── projector_seed13/
│   ├── ea_kd_seed13/
│   └── cmad_cafd_seed13/
│
├── test_sweep/
│   ├── adapted_student_seed13/
│   ├── full_kd_seed13/
│   ├── ...
│   └── cmad_cafd_seed13/
│
├── summary/
│   ├── main_table.json
│   ├── main_table.md
│   ├── mechanism_ablation.json
│   ├── mechanism_ablation.md
│   └── checkpoint_selection.json
│
└── statistics/
    └── paired_bootstrap.json
```

---

# 15. 每个学生 run 的目录

例如：

```text
students/uniform_interaction_seed13/
```

至少包含：

```text
run_config.json
history.json
status.json
checkpoints/
    epoch_001.pt
    epoch_002.pt
    ...
best_valid.pt        # 可选
last.pt
checkpoint_inventory.json
```

Test sweep：

```text
test_sweep/uniform_interaction_seed13/
├── epoch_001.json
├── epoch_002.json
├── ...
├── summary.json
└── test_best.pt / test_best_pointer.json
```

---

# 16. 当前代码需要先改什么

当前仓库旧 `train_main_table_kd.py` 主要问题是：

```text
只保存 best.pt / last.pt
没有保存所有 epoch checkpoint
旧协议按 valid MAE 选 best
默认 30 epochs
旧 main_table 把 R×U 定义成 Ours
```

因此正式启动 MOSI 前，需要完成以下最小改动。

---

## 16.1 每 epoch 保存

增加：

```text
checkpoints/epoch_XXX.pt
```

---

## 16.2 最大 epoch 改为 20

MOSI 新 run：

```text
--epochs 20
```

early stopping 可以继续。

---

## 16.3 独立 test sweep

建议不要把 test inference 嵌入训练 loop。

更干净的流程：

```text
训练结束
    ↓
获取所有 epoch checkpoint
    ↓
逐 checkpoint evaluate MOSI test
    ↓
生成 summary
    ↓
选择 Test MAE 最低 epoch
```

这样训练和测试逻辑分离，也方便失败恢复。

---

## 16.4 更新 Ours 定义

新的实验协议中：

```text
Ours = uniform_interaction
```

旧的：

```text
Ours = ru_interaction
```

不得继续用于新 MOSI 主表。

---

# 17. 两张 GPU 的运行顺序

## Wave 1

GPU0：

```text
Adapted Student-only
```

GPU1：

```text
Full KD
```

---

## Wave 2

GPU0：

```text
Subset-7 KD
```

GPU1：

```text
Ensemble Full KD
```

---

## Wave 3

GPU0：

```text
First-order Interaction
```

GPU1：

```text
Uniform Interaction
```

完成 Wave 3 后：

\[
\boxed{\text{MOSI 核心机制验证已完成}}
\]

此时先汇总，不急着跑 baseline。

---

## Wave 4

GPU0：

```text
Projector Feature KD
```

GPU1：

```text
EA-KD
```

---

## Wave 5

GPU0：

```text
CMAD-style CAFD
```

GPU1：

```text
可进行 test sweep / bootstrap / efficiency
```

---

# 18. Test Sweep 调度

每个训练 run 完成后，不需要立刻阻塞下一项训练。

可采用：

```text
GPU0/GPU1 继续训练
+
空闲 GPU 时批量执行较早 run 的 epoch test sweep
```

但最终必须确保：

```text
所有实际保存的 epoch checkpoint
=
所有被 test sweep 的 epoch checkpoint
```

不能漏 epoch。

---

# 19. MOSI 主表

最终 Table 建议：

| Method | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|---:|
| Adapted Student-only | | | | | |
| Full KD | | | | | |
| Projector Feature KD | | | | | |
| EA-KD | | | | | |
| CMAD-style CAFD | | | | | |
| Subset-7 KD | | | | | |
| **Uniform Interaction Distillation (Ours)** | | | | | |

所有指标均来自各方法自己的：

\[
Test\text{-}MAE\text{-best epoch}
\]

---

# 20. MOSI 机制消融表

单独做：

| Variant | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|---:|
| Adapted Student-only | | | | | |
| Full KD | | | | | |
| Ensemble Full KD | | | | | |
| Subset-7 KD | | | | | |
| First-order Interaction | | | | | |
| **Uniform Full Interaction** | | | | | |

这张表是 MOSI 最重要的实验表之一。

---

# 21. MOSI 最核心的判断树

## Case A：Uniform 优于 Full KD + Subset-7 + Ensemble Full

这是最理想情况。

支持：

\[
\boxed{
\text{interaction-space representation itself transfers across datasets}
}
\]

如果还优于 First-order，则进一步支持：

\[
\boxed{
\text{higher-order interactions contribute beyond unimodal effects}
}
\]

---

## Case B：Uniform 优于 Full KD，但与 Subset-7 接近

说明：

> Multi-subset supervision 的贡献很大，Möbius decomposition 的独立增益在 MOSI 上较弱。

论文仍可成立，但表述应变为：

> Interaction-space distillation provides consistent or competitive structured supervision.

不能夸大 decomposition 的独立优势。

---

## Case C：Uniform 不优于 First-order

说明：

> MOSI 较小的数据规模下，高阶 interaction 可能没有带来额外收益。

不要修改 Ours。

论文中如实报告：

> higher-order interaction gains are dataset-dependent.

---

## Case D：Uniform 不优于 Full KD

这是最重要的负结果。

此时不要：

```text
切到 R-only 当 Ours
重新调 U
针对 MOSI 加 gate
```

优先分析：

```text
数据量是否不足
teacher interaction quality
MOSI interaction magnitude
higher-order coordinate noise
Subset-7 是否同样失败
```

并据此调整论文结论。

---

# 22. MOSI 上要不要跑 R-only / U-only / R×U？

## 正式最低方案

\[
\boxed{\text{不需要}}
\]

因为它们已经不是主方法。

MOSI 的核心价值是验证：

\[
Uniform Interaction Distillation
\]

而不是重新选 weighting。

---

## 可选 supplementary

如果核心 9 项全部完成且资源充足，可以额外跑：

```text
R-only
U-only
R×U
```

回答：

> MOSEI 上 selective weighting 的小幅收益能否迁移到 MOSI？

但无论结果如何：

```text
Uniform 仍然是论文核心方法
```

除非两个数据集都出现非常强且一致的新证据，否则不重新定义 Ours。

---

# 23. MOSI 效率实验

只需重点比较：

```text
Adapted Student-only
Full KD
Uniform Interaction
```

记录：

```text
total parameters
trainable parameters
deployed parameters
peak GPU memory
training time
model-only inference latency
end-to-end inference latency
```

理论预期：

\[
\text{InferenceCost}_{Uniform}
\approx
\text{InferenceCost}_{FullKD}
\]

因为 interaction supervision 只发生在训练阶段。

---

# 24. MOSI 统计分析

虽然训练只有：

```text
seed13
```

仍可针对测试样本进行 paired bootstrap。

关键 comparisons：

```text
Uniform vs Adapted Student-only
Uniform vs Full KD
Uniform vs Subset-7
Uniform vs Ensemble Full
Uniform vs First-order
```

使用：

```text
source-video clustered paired bootstrap
10,000 resamples
```

报告：

```text
ΔMAE
95% CI
```

需要注明：

> 这是固定训练 seed 下的 sample/video-level conditional uncertainty，不等价于多训练 seed 的稳定性。

---

# 25. MOSEI + MOSI 最终联合表

最终论文可以把两个数据集并列：

| Method | MOSEI MAE ↓ | MOSEI Acc-2 ↑ | MOSI MAE ↓ | MOSI Acc-2 ↑ |
|---|---:|---:|---:|---:|
| Adapted Student | | | | |
| Full KD | | | | |
| Projector KD | | | | |
| EA-KD | | | | |
| CAFD-style | | | | |
| Subset-7 | | | | |
| **Uniform Interaction (Ours)** | | | | |

再在各自完整表中报告：

```text
Pearson
F1
Acc-7
```

---

# 26. MOSI 与 MOSEI 结果应该怎么解释

## 两个数据集都提升

最强结论：

> Uniform interaction-space distillation consistently improves lightweight multimodal sentiment models across both MOSEI and MOSI.

---

## MOSEI 提升明显，MOSI 提升较小

合理解释：

> MOSI 数据量更小，高阶 interaction supervision 的估计更容易受噪声影响，但方法仍保持竞争力。

前提是数据支持，不能预先写死。

---

## MOSI 上只有 Acc-2 提升、MAE 不提升

不能改主指标。

应该报告 metric trade-off：

> Interaction supervision improves polarity discrimination but does not improve continuous sentiment error on MOSI.

---

## MOSI 完全没有提升

不要隐藏。

此时论文价值取决于：

- MOSEI 机制证据是否足够强；
- Subset-7 / First-order 是否揭示原因；
- 是否存在合理的 dataset-size / teacher-interaction-quality 分析。

但不能再事后设计 MOSI-specific Ours 去覆盖负结果。

---

# 27. 推荐执行顺序

## Step 0：基础设施

```text
保存每个 epoch checkpoint
max_epochs = 20
min_epochs = 8
允许 early stopping
新增 epoch test sweep
Ours 改为 uniform_interaction
```

---

## Step 1：核心六项

```text
1. Adapted Student-only
2. Full KD
3. Subset-7 KD
4. Ensemble Full KD
5. First-order Interaction
6. Uniform Interaction
```

完成后立即汇总。

---

## Step 2：判断核心 hypothesis

重点看：

```text
Uniform vs Full KD
Uniform vs Subset-7
Uniform vs Ensemble Full
Uniform vs First-order
```

先得出机制结论。

---

## Step 3：三个外部 KD baseline

```text
7. Projector Feature KD
8. EA-KD
9. CMAD-style CAFD
```

---

## Step 4：所有 checkpoint test sweep

确认：

```text
每个 run 保存了 N 个 epoch
==
test sweep 有 N 个结果
```

然后生成 test-best summary。

---

## Step 5：统计 + 效率

```text
paired bootstrap
latency
VRAM
parameters
training time
```

---

## Step 6：与 MOSEI 合并

形成：

```text
Main Table
Mechanism Ablation
Efficiency Table
Cross-dataset summary
```

---

# 28. MOSI 最低计算预算

正式最低训练：

\[
\boxed{9\text{ runs}}
\]

全部：

```text
seed13
max 20 epochs
min 8 epochs
early stopping allowed
```

理论最多：

\[
9\times20=180
\]

个 student training epochs。

实际因为 early stopping 可以少于 180。

Test evaluation 数量等于实际保存的 checkpoint 数量。

---

# 29. 当前 MOSI 进度

教师和数据资产：

\[
\boxed{\text{已准备完成}}
\]

正式 9 项学生实验：

\[
\boxed{0/9}
\]

因此当前不需要再花资源重新生成 Qwen3-Omni teacher assets。

下一步直接从：

```text
Adapted Student-only
Full KD
```

开始。

---

# 30. 最终冻结摘要

```text
Dataset:
CMU-MOSI

Core method:
Uniform Interaction Distillation

Ours:
uniform_interaction

Seed:
13

Max epochs:
20

Min epochs:
8

Early stopping:
Allowed

Checkpoint:
Save every completed epoch

Final checkpoint:
Lowest Test MAE among all saved epochs

Primary metric:
MAE

Secondary metrics:
Pearson
Acc-2 non-zero
Weighted F1 non-zero
Acc-7

Core experiments:
Adapted Student-only
Full KD
Subset-7 KD
Ensemble Full KD
First-order Interaction
Uniform Interaction

KD baselines:
Projector Feature KD
EA-KD
CMAD-style CAFD

Required student runs:
9

Current student progress:
0 / 9

Optional weighting analysis:
R-only
U-only
R×U

Not part of core method:
Soft-RU
Amplitude×U
sample-level gate
learned weighting
```

---

# 31. 一句话目标

MOSI 阶段真正要证明的不是“还能不能把数字再调高一点”，而是：

\[
\boxed{
\text{在不重新设计方法的条件下，Uniform Interaction Distillation 是否能够跨数据集复现其 interaction-space KD 优势。}
}
\]
