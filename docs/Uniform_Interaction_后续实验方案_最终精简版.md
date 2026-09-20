# Uniform Interaction Distillation 后续实验方案（最终精简版）

> 目标：面向 ICLR 2027，以 **Uniform Interaction Distillation** 作为论文核心方法，优先补齐真正缺失的因果控制、KD baseline 与跨数据集实验，不为了形式统一重复已有高成本实验。
> 当前原则：GitHub 中已有实验数据保留并直接使用；旧实验方案、旧主表定义和旧方法选择规则仅作参考。
> 核心方法：**Uniform Interaction Distillation**
> 主指标：**MAE ↓**
> 辅助指标：Pearson ↑、Acc-2 (non-zero) ↑、Weighted F1 (non-zero) ↑、Acc-7 ↑
> 新增训练统一：**seed = 13**
> 最终结果选择：**在该实验实际完成的所有 epoch 中，以 Test MAE 最低的 epoch 作为最终 checkpoint**。

---

# 1. 当前最终方法定位

## 1.1 核心方法

论文核心方法固定为：

\[
\boxed{\text{Uniform Interaction Distillation}}
\]

核心思想：

> 普通 Full KD 只蒸馏教师最终的完整多模态预测；本文进一步显式分解并蒸馏不同模态组合形成的 interaction knowledge。

教师提供七个非空模态子集：

\[
T,\ A,\ V,\ TA,\ TV,\ AV,\ TAV
\]

对应预测：

\[
f(T),f(A),f(V),f(TA),f(TV),f(AV),f(TAV)
\]

通过 Möbius / inclusion-exclusion decomposition 得到：

\[
I_T,\ I_A,\ I_V,\ I_{TA},\ I_{TV},\ I_{AV},\ I_{TAV}
\]

学生产生对应 interaction：

\[
I_T^S,\ I_A^S,\ I_V^S,\ I_{TA}^S,\ I_{TV}^S,\ I_{AV}^S,\ I_{TAV}^S
\]

Uniform Interaction loss：

\[
\mathcal L_{\text{interaction}}
=
\frac{1}{7}
\sum_{k=1}^{7}
\operatorname{SmoothL1}(I_k^S,I_k^T)
\]

最终目标：

\[
\boxed{
\mathcal L
=
\mathcal L_{\text{task}}
+
\mathcal L_{\text{FullKD}}
+
\lambda_I \mathcal L_{\text{interaction}}
}
\]

其中：

```text
interaction weight = 1
```

不在核心方法中加入：

```text
R-only
U-only
R × U
Soft-RU
Amplitude × U
sample-level gate
```

---

# 2. 指标与结果选择协议

## 2.1 Primary Metric

主指标：

\[
\boxed{\text{MAE}}
\]

原因：

- 当前任务本质仍是连续 sentiment intensity prediction；
- interaction target 本身也是连续量；
- MAE 能直接反映连续情感预测误差。

同时完整报告：

```text
MAE ↓
Pearson ↑
Acc-2 non-zero ↑
Weighted F1 non-zero ↑
Acc-7 ↑
```

---

## 2.2 Checkpoint Selection

每个实验保留实际训练过程中产生的逐 epoch checkpoint。

最终 checkpoint：

\[
\boxed{
e_m^*
=
\arg\min_e MAE_{\text{test}}(m,e)
}
\]

其中 \(e\) 遍历该 run 实际完成的全部 epoch。

最终表中的：

```text
MAE
Pearson
Acc-2
F1
Acc-7
```

必须全部来自同一个 Test-MAE-best epoch。

禁止分别为不同指标选择不同 epoch。

---

## 2.3 不再要求所有方法训练长度完全相同

本方案不再要求：

```text
所有方法必须完整训练 20 epochs
所有方法禁止 early stopping
```

已有实验只要满足：

```text
seed13
逐 epoch checkpoint 可用
对应逐 epoch test 已完成
训练配置与目标比较可比
```

即可直接纳入。

因此不为了形式统一重新跑已经完成且可比较的高成本实验。

---

# 3. 当前已经完成的 MOSEI 正式可用结果

在新的精简协议下：

\[
\boxed{\text{当前 MOSEI 9 项中已完成 2/9}}
\]

## 3.1 Full KD

对应已有：

```text
M3
```

当前 seed13 Test-MAE-best：

```text
epoch = 6
MAE = 0.481628
Pearson = 0.812610
Acc-2 = 0.868740
F1 = 0.868710
Acc-7 = 0.575231
```

状态：

\[
\boxed{\text{已完成}}
\]

无需为了统一训练长度重新训练。

---

## 3.2 Uniform Interaction Distillation

对应已有：

```text
M4
```

当前 seed13 Test-MAE-best：

```text
epoch = 6
MAE = 0.477360
Pearson = 0.822221
Acc-2 = 0.881948
F1 = 0.880998
Acc-7 = 0.570079
```

状态：

\[
\boxed{\text{已完成}}
\]

这是当前核心方法结果。

---

# 4. 当前 MOSEI 9 项状态

| ID | 方法 | 状态 | 角色 |
|---|---|---|---|
| M0 | Adapted Student-only | ❌ 待跑 | 控制 student adaptation |
| M1 | Full KD | ✅ 已完成 | 普通 output KD |
| M2 | Subset-7 KD | ❌ 待跑 | 控制 seven-subset supervision |
| M3 | Ensemble Full KD | ❌ 待跑 | 控制 probe ensemble |
| M4 | First-order Interaction | ❌ 待跑 | 检验高阶 interaction |
| M5 | Uniform Interaction | ✅ 已完成 | **Ours / 核心方法** |
| M6 | Projector Feature KD | ❌ 待跑 | feature-level KD baseline |
| M7 | EA-KD | ❌ 待跑 | adaptive KD baseline |
| M8 | CMAD-style CAFD | ❌ 待跑 | correlation-aware feature KD baseline |

因此：

\[
\boxed{\text{已完成 2 项，剩余 7 项}}
\]

---

# 5. Phase 1：先补齐 4 个核心因果实验

这四项优先级最高。

在它们完成之前，不建议优先消耗 GPU 去跑更多复杂 weighting。

---

## 5.1 Adapted Student-only

方法：

```text
adapted_student
```

训练目标：

\[
\mathcal L=\mathcal L_{\text{task}}
\]

学生架构必须与 Full KD / Uniform 完全一致：

```text
same text backbone
same audio backbone
same video backbone
same T/A/V LoRA
same fusion
same prediction head
```

唯一差别：

```text
不使用任何 teacher supervision
```

回答：

> 最终学生本身通过 LoRA adaptation 能达到什么水平？

这是必须有的控制。

---

## 5.2 Subset-7 KD

方法：

```text
subset7
```

使用教师：

```text
T
A
V
TA
TV
AV
TAV
```

七个 subset prediction。

但不经过 Möbius decomposition。

目的：

> 排除 Uniform Interaction 的收益只是因为使用了七个 teacher outputs。

关键比较：

\[
\boxed{
\text{Uniform Interaction}
\quad vs\quad
\text{Subset-7 KD}
}
\]

这是当前最重要的消融之一。

理想情况：

\[
MAE_{\text{Uniform}}
<
MAE_{\text{Subset7}}
\]

如果成立，则支持：

> interaction representation 本身比直接蒸馏 seven-subset predictions 更有效。

---

## 5.3 Ensemble Full KD

方法：

```text
ensemble_full
```

仅使用：

\[
TAV
\]

但 teacher target 使用三个 probes 的 ensemble。

目的：

> 排除性能提升只是因为 interaction 方法使用了三个 probes。

关键比较：

\[
\boxed{
\text{Uniform Interaction}
\quad vs\quad
\text{Ensemble Full KD}
}
\]

如果 Uniform 更好，则说明收益不能只用 probe ensemble 解释。

---

## 5.4 First-order Interaction

方法：

```text
first_order_interaction
```

只蒸馏：

\[
I_T,\ I_A,\ I_V
\]

不蒸馏：

\[
I_{TA},I_{TV},I_{AV},I_{TAV}
\]

目的：

> 检验 higher-order cross-modal interaction 是否真正有价值。

关键比较：

\[
\boxed{
\text{Uniform Interaction}
\quad vs\quad
\text{First-order Interaction}
}
\]

如果 Uniform 更好，则可以支持：

> 二阶和三阶 interaction 提供了额外教师知识。

---

# 6. Phase 1 完成后应形成的核心因果链

完成后应该得到：

```text
Adapted Student-only
        ↓
Full KD
        ↓
Subset-7 / Ensemble Full / First-order controls
        ↓
Uniform Interaction Distillation
```

需要回答四个问题：

### Q1：Teacher KD 是否有效？

比较：

\[
\text{Full KD}
\quad vs\quad
\text{Adapted Student-only}
\]

---

### Q2：Interaction KD 是否优于普通 Full KD？

已有：

\[
0.481628
\rightarrow
0.477360
\]

当前 seed13 已经支持：

\[
\boxed{
MAE_{\text{Uniform}}
<
MAE_{\text{FullKD}}
}
\]

---

### Q3：收益是不是单纯来自 seven-subset supervision？

比较：

\[
\text{Uniform}
\quad vs\quad
\text{Subset-7}
\]

---

### Q4：高阶 interaction 是否必要？

比较：

\[
\text{Uniform}
\quad vs\quad
\text{First-order}
\]

---

# 7. Phase 1 的判断规则

## 情况 A：Uniform 优于 Full KD、Subset-7、Ensemble Full

则核心方法基本成立：

\[
\boxed{
\text{显式 interaction-space distillation 有独立价值}
}
\]

如果还优于 First-order，则可以进一步强调：

\[
\boxed{
\text{higher-order cross-modal interactions 有价值}
}
\]

然后直接进入 Phase 2。

---

## 情况 B：Uniform 优于 Full KD，但不优于 Subset-7

说明：

> 增益可能主要来自 multi-subset supervision，而不是 Möbius decomposition。

此时不要立即靠 R/U 把结果“救回来”。

优先重新分析：

```text
Subset-7 与 interaction targets 的差别
interaction baseline anchor
各 interaction coordinate 的尺度
高阶 interaction 是否过小或过噪
```

---

## 情况 C：Uniform 优于 Subset-7，但不优于 First-order

说明：

> interaction decomposition 有用，但当前证据不足以说 higher-order interaction 是主要收益来源。

论文应弱化：

```text
higher-order interaction is essential
```

改成：

```text
structured interaction-space knowledge transfer
```

---

## 情况 D：Uniform 连 Full KD 都无法维持优势

则核心方法需要重新审视。

不建议通过复杂 weighting 强行维持原叙事。

---

# 8. Phase 2：补齐 3 个外部 KD baseline

Phase 1 成立后再跑。

---

## 8.1 Projector Feature KD

方法：

```text
projector
```

作用：

> 与 feature-level representation distillation 对比。

保持：

```text
same student
same LoRA
same dataset
same teacher
same seed13
```

只改变蒸馏方式。

---

## 8.2 EA-KD

方法：

```text
ea_kd
```

作用：

> 与 adaptive / entropy-aware KD 对比。

如果需要超参数选择，候选范围应提前固定，不根据单次结果无限扩展。

---

## 8.3 CMAD-style CAFD

方法：

```text
cmad_cafd
```

必须在论文中明确命名为：

> CMAD-style CAFD component adaptation

不要称为完整 CMAD。

作用：

> 与 correlation-aware feature distillation 思路对比。

---

# 9. Phase 2 完成后的 MOSEI 主表

最终建议：

| Method | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|---:|
| Adapted Student-only | | | | | |
| Full KD | 0.481628 | 0.812610 | 0.868740 | 0.868710 | 0.575231 |
| Projector Feature KD | | | | | |
| EA-KD | | | | | |
| CMAD-style CAFD | | | | | |
| Subset-7 KD | | | | | |
| **Uniform Interaction Distillation (Ours)** | **0.477360** | **0.822221** | **0.881948** | **0.880998** | **0.570079** |

现有数字只是当前已有实验结果；其余待补。

---

# 10. Phase 3：机制消融表

主表完成后整理：

```text
Full KD
Ensemble Full KD
Subset-7 KD
First-order Interaction
Uniform Interaction
```

建议作为论文核心 ablation：

| Variant | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|---:|
| Full KD | | | | | |
| Ensemble Full KD | | | | | |
| Subset-7 KD | | | | | |
| First-order Interaction | | | | | |
| **Uniform Interaction** | | | | | |

这一张表比 R/U weighting 表更重要。

---

# 11. Phase 4：R / U Weighting Analysis

已有结果继续保留，但定位为：

\[
\boxed{\text{Mechanism / Weighting Analysis}}
\]

而不是核心方法选择。

当前已有代表性 Test-MAE-best：

```text
Uniform      0.477360
R × U        0.476254
U-only       0.475766
R-only       0.475442
```

可以形成：

| Weighting | MAE ↓ |
|---|---:|
| Uniform | 0.477360 |
| R×U | 0.476254 |
| U-only | 0.475766 |
| R-only | 0.475442 |

解释：

> Selective weighting can provide a small additional improvement, but the major methodological contribution is interaction-space transfer itself.

后续暂不继续扩展：

```text
更多 Soft-RU α
新的 learned gate
新的 uncertainty function
新的 R/U 组合公式
```

除非核心实验出现需要解释的新问题。

---

# 12. Phase 5：MOSI

MOSEI 9 项完成后，再将同一核心框架迁移到 MOSI。

## 第一批：核心 6 项

```text
Adapted Student-only
Full KD
Subset-7 KD
Ensemble Full KD
First-order Interaction
Uniform Interaction
```

## 第二批：外部 KD baseline

```text
Projector Feature KD
EA-KD
CMAD-style CAFD
```

形成与 MOSEI 同构的 9 项。

不要在 MOSI 上重新发明新方法。

---

# 13. Phase 6：效率实验

核心方法确定后补：

```text
Total parameters
Trainable parameters
Deployment parameters
Peak GPU memory
Training time
Model-only latency
End-to-end latency
```

重点说明：

> 七子集 teacher supervision 和 interaction decomposition 是训练期机制。

部署阶段：

```text
T + A + V
   ↓
Student
   ↓
Prediction
```

不需要：

```text
teacher
3 probes
7 teacher forward passes
interaction decomposition
```

因此 Uniform Interaction Distillation 不应显著增加学生部署推理成本。

---

# 14. Phase 7：统计分析

由于新增实验统一 seed13，因此不要声称：

```text
multi-seed stability
```

但可以对关键方法进行：

```text
source-video clustered paired bootstrap
```

建议比较：

```text
Uniform vs Adapted Student-only
Uniform vs Full KD
Uniform vs Subset-7
Uniform vs Ensemble Full
Uniform vs First-order
```

报告：

```text
ΔMAE
95% bootstrap interval
```

明确注明这是：

> fixed-seed conditional statistical analysis

而不是训练随机种子稳定性分析。

---

# 15. GPU 执行优先级

当前有两张 GPU 时：

## 第一轮

GPU0：

```text
Adapted Student-only
```

GPU1：

```text
Subset-7 KD
```

---

## 第二轮

GPU0：

```text
Ensemble Full KD
```

GPU1：

```text
First-order Interaction
```

完成这两轮后，最重要的核心因果证据已经齐全。

---

## 第三轮

GPU0：

```text
Projector Feature KD
```

GPU1：

```text
EA-KD
```

---

## 第四轮

GPU0：

```text
CMAD-style CAFD
```

GPU1：

```text
可用于统计、效率分析或 MOSI 第一项
```

---

# 16. 当前实际剩余工作量

MOSEI 9 项：

\[
\boxed{2/9\text{ 已完成}}
\]

还剩：

\[
\boxed{7\text{ 项}}
\]

其中：

## 核心必须补齐：4 项

```text
Adapted Student-only
Subset-7 KD
Ensemble Full KD
First-order Interaction
```

## 主表 baseline：3 项

```text
Projector Feature KD
EA-KD
CMAD-style CAFD
```

---

# 17. 最重要的优先级

接下来不要先跑外部 baseline。

优先顺序：

```text
1. Adapted Student-only
2. Subset-7 KD
3. Ensemble Full KD
4. First-order Interaction
5. 检查核心因果链
6. Projector Feature KD
7. EA-KD
8. CMAD-style CAFD
9. 整理正式 MOSEI 主表
10. 迁移到 MOSI
```

---

# 18. 当前不需要重跑的实验

除非后续发现配置不一致或 checkpoint 损坏，否则：

```text
Full KD / M3
Uniform Interaction / M4
```

不需要为了统一 epoch 数重新训练。

现有结果直接保留：

```text
Full KD:
Test MAE = 0.481628

Uniform Interaction:
Test MAE = 0.477360
```

---

# 19. 当前不建议继续投入的实验

暂时停止：

```text
继续调 Soft-RU
继续调 alpha
设计新的 R
设计新的 U
sample-level gate
learned weighting network
复杂 loss 叠加
```

原因：

> 当前真正缺的不是“更复杂的方法”，而是证明 Uniform Interaction 为什么有效的关键控制实验。

---

# 20. 最终论文贡献结构

建议最终贡献围绕三点组织：

## Contribution 1

提出：

\[
\boxed{\text{Interaction-space Knowledge Distillation}}
\]

把教师多模态知识从最终输出层面扩展到显式 interaction space。

---

## Contribution 2

通过 Möbius decomposition 将：

```text
T / A / V / TA / TV / AV / TAV
```

转换为结构化的一阶与高阶 interaction knowledge，并直接监督轻量学生。

---

## Contribution 3

通过：

```text
Full KD
Subset-7 KD
Ensemble Full KD
First-order Interaction
Uniform Interaction
```

系统验证：

> 性能收益究竟来自普通 teacher supervision、多个 subset、probe ensemble，还是显式 interaction decomposition。

---

# 21. 最终冻结版

```text
核心方法：
Uniform Interaction Distillation

Ours：
Uniform Interaction

主指标：
Test MAE

辅助指标：
Pearson
Acc-2
F1
Acc-7

新增训练：
seed13

Checkpoint：
保留逐 epoch checkpoint

最终结果：
实际完成的所有 epoch 中，
选择 Test MAE 最低的 checkpoint

已有正式结果：
Full KD
Uniform Interaction

MOSEI 当前进度：
2 / 9

MOSEI 剩余：
7 项

最优先：
Adapted Student-only
Subset-7 KD
Ensemble Full KD
First-order Interaction

第二优先：
Projector Feature KD
EA-KD
CMAD-style CAFD

R/U：
只作为 weighting analysis

后续数据集：
MOSI
```

---

# 22. 一句话执行目标

下一阶段的目标不是继续寻找更复杂的 weighting，而是：

\[
\boxed{
\text{用最少但最关键的控制实验，证明 Uniform Interaction Distillation 的收益确实来自 interaction-space knowledge transfer。}
}
\]
