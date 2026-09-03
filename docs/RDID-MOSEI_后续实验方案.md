# RDID-MOSEI 后续实验方案

> **项目目标**：在现有 `Full KD / Subset KD / Möbius KD / interaction4 / subset4` 实验基础上，验证“可靠性加权的选择性交互蒸馏”是否能稳定优于直接子集蒸馏，并进一步研究其与信息瓶颈、缺失模态鲁棒性和端侧计算预算之间的关系。  
> **教师**：Qwen3-Omni-30B-A3B-Instruct + 共享情感 Probe  
> **数据集**：CMU-MOSEI  
> **学生**：Qwen3-0.6B-Base + WavLM-Base-Plus + VideoMAE-Base  
> **当前已知结论**：
>
> 1. Student-only 明显弱于 Full KD；
> 2. Full KD 有效；
> 3. `subset4` 当前优于 `interaction4`；
> 4. raw Möbius 高阶交互没有直接带来优势；
> 5. interaction4 的种子方差高于 subset4；
> 6. 下一阶段不再继续盲目调 interaction 标量权重，而优先验证“教师交互噪声被 Möbius 差分放大”这一假设。

---

## 1. 后续研究问题

后续实验围绕 5 个问题展开：

### RQ1：raw interaction4 为什么输给 subset4？

重点区分：

- 数值尺度问题；
- 高阶差分噪声放大；
- 三阶交互过于不稳定；
- Möbius 坐标本身无优势；
- 仅仅是额外损失带来的正则化差异。

### RQ2：教师交互可靠性加权是否有效？

验证：

\[
\text{Reliable Interaction KD}
>
\text{Raw Interaction KD}
\]

以及能否达到或超过：

\[
\text{Subset4 KD}.
\]

### RQ3：二阶交互和三阶交互谁更有价值？

分别比较：

- pair-only；
- triple-only；
- pair + triple。

### RQ4：交互蒸馏能否在有限信息率下更有优势？

在 VIB 下比较：

\[
\text{Subset4+VIB}
\quad vs \quad
\text{Reliable Interaction+VIB}.
\]

### RQ5：交互蒸馏的收益是否集中在高交互、模态冲突和缺失模态场景？

如果方法确实学习跨模态关系，那么提升应主要出现在：

- 高教师交互强度样本；
- 文本和非文本模态冲突样本；
- 缺失模态样本；
- 音频或视觉更关键的样本。

---

# 2. 总体实验路线

建议严格分为 5 个阶段：

```text
Stage A
benchmark500 诊断实验
    ↓
Stage B
可靠性交互方法筛选
    ↓
Stage C
全量 CMU-MOSEI 3-seed 核心实验
    ↓
Stage D
VIB / Rate-Distortion 实验
    ↓
Stage E
鲁棒性、缺失模态、端侧效率与最终论文实验
```

原则：

> **每一阶段只有通过 Go 条件，才进入下一阶段。**

---

# 3. Stage A：benchmark500 诊断实验

## 3.1 目标

先解释 raw interaction4 为什么弱于 subset4。

本阶段只用已有 benchmark500，不做全量数据。

---

## 3.2 A1：Subset4 强基线复现

已有配置：

\[
\mathcal L
=
\mathcal L_{\text{task}}
+
\lambda_f\mathcal L_{\text{fullKD}}
+
\lambda_s\mathcal L_{\text{subset4}}.
\]

其中：

\[
\mathcal S_4
=
\{ta,tv,av,tav\}.
\]

必须固定：

- 训练集；
- 验证集；
- seed；
- batch size；
- optimizer；
- 学习率；
- epoch；
- teacher cache；
- student initialization。

建议确认 3 seeds：

```text
13
42
2026
```

记录：

- MAE；
- Pearson；
- Acc-2；
- F1；
- Acc-7；
- validation best epoch；
- gradient norm；
- loss scale。

---

## 3.3 A2：Raw Interaction4

保持与 subset4 完全相同的 4 个标量监督维数：

\[
\mathcal A_4
=
\{ta,tv,av,tav\}.
\]

损失：

\[
\mathcal L_{\text{raw-int}}
=
\sum_{A\in\mathcal A_4}
\rho(
\mu_S(A)-\mu_T(A)
).
\]

目的：

> 固化当前失败基线，后续所有新方法均与该结果比较。

---

# 4. A3：Z-score Interaction4

### Motivation

Möbius 不同阶次的数值尺度可能显著不同。

先统计训练集：

\[
m_A
=
\mathbb E[\mu_T(A)],
\]

\[
s_A
=
\sqrt{
\operatorname{Var}[\mu_T(A)]
}.
\]

标准化：

\[
\tilde \mu_T(A)
=
\frac{
\mu_T(A)-m_A
}{
s_A+\epsilon
}.
\]

学生同样：

\[
\tilde \mu_S(A)
=
\frac{
\mu_S(A)-m_A
}{
s_A+\epsilon
}.
\]

损失：

\[
\mathcal L_{\text{z-int}}
=
\sum_A
\rho(
\tilde\mu_S(A)-\tilde\mu_T(A)
).
\]

### 判断

如果：

\[
\text{Z-score interaction}
\gg
\text{Raw interaction},
\]

说明原结果可能主要来自量纲/尺度问题。

如果没有改善，则继续验证噪声假设。

---

# 5. 教师交互噪声估计

## 5.1 利用已有多 Probe seed

假设已有 3 个教师 Probe：

```text
teacher_probe_seed_2026
teacher_probe_seed_2027
teacher_probe_seed_2028
```

对同一样本和交互项：

\[
\mu_T^{(1)}(A),
\mu_T^{(2)}(A),
\mu_T^{(3)}(A).
\]

计算：

\[
\bar\mu_T(A)
=
\frac{1}{K}
\sum_{k=1}^{K}
\mu_T^{(k)}(A),
\]

\[
\sigma_T^2(A)
=
\frac{1}{K-1}
\sum_{k=1}^{K}
\left[
\mu_T^{(k)}(A)-\bar\mu_T(A)
\right]^2.
\]

保存新 cache：

```json
{
  "sample_id": "...",
  "interaction_mean": {
    "ta": 0.32,
    "tv": -0.18,
    "av": 0.09,
    "tav": -0.44
  },
  "interaction_var": {
    "ta": 0.01,
    "tv": 0.08,
    "av": 0.04,
    "tav": 0.27
  }
}
```

---

# 6. A4：Inverse-Variance Interaction KD

定义：

\[
w_A(x)
=
\frac{
1
}{
\sigma_T^2(A|x)+\epsilon
}.
\]

建议做归一化和截断：

\[
\tilde w_A
=
\frac{
w_A
}{
\frac1{|\mathcal A|}
\sum_B w_B
},
\]

并：

\[
\tilde w_A
\leftarrow
\operatorname{clip}
(\tilde w_A,w_{\min},w_{\max}).
\]

初始：

```yaml
epsilon: 1.0e-4
w_min: 0.25
w_max: 4.0
```

损失：

\[
\mathcal L_{\text{IV-int}}
=
\sum_A
\tilde w_A
\rho(
\mu_S(A)-\bar\mu_T(A)
).
\]

### 目的

验证：

> 教师越稳定的交互，越值得被蒸馏。

---

# 7. A5：SNR-weighted Interaction KD

仅方差小不代表交互有意义。

定义：

\[
SNR_A
=
\frac{
|\bar\mu_T(A)|
}{
\sigma_T(A)+\epsilon
}.
\]

权重：

\[
w_A
=
\operatorname{clip}
\left(
\frac{SNR_A}
{\operatorname{mean}_B SNR_B},
w_{\min},
w_{\max}
\right).
\]

损失：

\[
\mathcal L_{\text{SNR-int}}
=
\sum_A
w_A
\rho(
\mu_S(A)-\bar\mu_T(A)
).
\]

这应该成为当前最重要的新候选方法。

---

# 8. A6：Threshold Selective Interaction

只学习：

> **强且可靠的交互。**

定义：

\[
g_A(x)
=
\mathbf 1
[
SNR_A(x)>\tau
].
\]

损失：

\[
\mathcal L_{\text{selective}}
=
\frac{
\sum_A
g_A(x)
\rho(
\mu_S(A)-\bar\mu_T(A)
)
}{
\sum_Ag_A(x)+\epsilon
}.
\]

阈值不要大规模网格搜索。

使用训练集 SNR 分位数：

```text
top 25%
top 50%
top 75%
```

三个设置即可。

---

# 9. A7：Pair-only Interaction

只使用：

\[
\mathcal A_2
=
\{ta,tv,av\}.
\]

损失：

\[
\mathcal L_{\text{pair}}
=
\sum_{A\in\mathcal A_2}
w_A
\rho(
\mu_S(A)-\bar\mu_T(A)
).
\]

优先使用：

- raw pair；
- SNR-weighted pair。

### 核心假设

二阶交互只包含 3 个集合值差分，理论噪声通常低于三阶交互。

如果：

\[
\text{Pair-only}
>
\text{Pair+Triple},
\]

说明三阶目标可能是主要噪声来源。

---

# 10. A8：Triple-only Interaction

仅：

\[
\mu(tav).
\]

用于诊断，不预期成为最终方法。

比较：

```text
pair-only
triple-only
pair+triple
```

记录：

- MAE；
- Pearson；
- seed variance；
- interaction target variance；
- teacher disagreement。

---

# 11. A9：Random Orthogonal Coordinate

这是必须做的公平实验。

对中心化的 7 维 subset value：

\[
\widetilde{\mathbf v}
=
\mathbf v-b.
\]

生成固定随机正交矩阵：

\[
Q^\top Q=I.
\]

变换：

\[
\mathbf r=Q\widetilde{\mathbf v}.
\]

蒸馏：

\[
\mathcal L_{\text{random-orth}}
=
\rho(
\mathbf r_S-\mathbf r_T
).
\]

至少固定三个随机矩阵 seed：

```text
100
200
300
```

### 判断

如果：

\[
\text{Möbius}
\approx
\text{Random Orthogonal},
\]

则不能声称交互坐标本身具有特殊意义。

如果可靠性加权的 Möbius 明显优于随机坐标，则方法论更有说服力。

---

# 12. A10：Random Non-Orthogonal Coordinate

构造随机可逆矩阵：

\[
R.
\]

要求：

\[
\kappa(R)
\approx
\kappa(M_{\text{Mobius}}),
\]

即 condition number 接近 Möbius 矩阵。

目的：

> 排除 Möbius 的收益或劣势仅来自非正交变换与数值缩放。

---

# 13. Stage A 最终实验表

建议 benchmark500 最终只补以下项目：

| ID | 方法 | 必做 |
|---|---|---:|
| A1 | subset4 | 已完成/复现 |
| A2 | raw interaction4 | 已完成/复现 |
| A3 | z-score interaction4 | ✓ |
| A4 | inverse-variance interaction4 | ✓ |
| A5 | SNR-weighted interaction4 | ✓ |
| A6 | selective interaction | ✓ |
| A7 | pair-only raw/SNR | ✓ |
| A8 | triple-only | ✓ |
| A9 | random orthogonal | ✓ |
| A10 | random non-orthogonal | ✓ |

每个方法：

- 先 1 seed 快速筛选；
- 只有明显有潜力的方法再运行 3 seeds。

---

# 14. Stage A Go 条件

只有满足以下条件之一，才进入全量数据：

### 条件 1

\[
MAE_{\text{new-int}}
<
MAE_{\text{subset4}}
\]

并且不是单 seed 偶然。

### 条件 2

整体 MAE 与 subset4 相当，但：

\[
MAE_{\text{High-Interaction}}
<
MAE_{\text{subset4, High-Interaction}}
\]

明显。

### 条件 3

相同 MAE 下，新方法具有更低：

\[
D_{\text{interaction}}.
\]

### 条件 4

方法与 subset4 性能相当，但 seed 方差显著更低。

若 A3–A10 全部无法接近 subset4，则：

> **停止把 interaction 作为主训练目标。**

此时论文方向改为：

> 分析高阶交互蒸馏为何受教师噪声限制，并将 interaction 作为分析工具而非核心 loss。

---

# 15. Stage B：全量 CMU-MOSEI 核心实验

只有 Stage A 通过后进行。

## 15.1 推荐保留的模型

最多保留：

```text
B0 Student-only
B1 Full KD
B2 subset4
B3 best interaction method
B4 best interaction + VIB
```

不要把 benchmark500 的全部诊断实验搬到全量数据。

## 15.2 随机种子

至少：

```text
13
42
2026
```

最终论文主表建议：

```text
5 seeds
```

若资源不足：

- 全量主表 3 seeds；
- benchmark500 方法诊断 5 seeds。

---

# 16. Stage B 统计检验

对同一 test 样本进行 paired bootstrap。

例如 MAE 差值：

\[
\Delta_i
=
|\hat y_i^{A}-y_i|
-
|\hat y_i^{B}-y_i|.
\]

Bootstrap：

```text
10000 resamples
```

报告：

- mean difference；
- 95% CI；
- p-value；
- paired standardized effect size。

不能只报告平均值。

---

# 17. Stage B 分组分析：高交互样本

定义：

\[
I_T(x)
=
|\bar\mu_T(ta)|
+
|\bar\mu_T(tv)|
+
|\bar\mu_T(av)|
+
|\bar\mu_T(tav)|.
\]

分为：

```text
Low:    0–33%
Medium: 33–66%
High:   66–100%
```

分别报告：

- MAE；
- Pearson；
- Acc-2。

### 关键期望

若 Interaction 方法成立：

\[
\Delta_{\text{High}}
>
\Delta_{\text{Low}}.
\]

---

# 18. 教师可靠性分组

定义：

\[
U_T(x)
=
\frac1{4}
\sum_A
\sigma_T^2(A|x).
\]

分为：

- Low uncertainty；
- Medium uncertainty；
- High uncertainty。

比较：

```text
raw interaction
vs
reliability-aware interaction
```

预期：

> reliability-aware 方法主要在高不确定性交互样本上减少性能损失。

---

# 19. Stage C：VIB / Rate-Distortion 实验

## 19.1 只比较两个核心方法

不要对所有基线扫描 VIB。

只比较：

```text
subset4 + VIB
best interaction + VIB
```

## 19.2 Beta 网格

\[
\beta
\in
\{
0,
10^{-6},
10^{-5},
10^{-4},
10^{-3}
\}.
\]

如果：

\[
10^{-3}
\]

导致明显 posterior collapse，再补：

\[
3\times10^{-4}.
\]

## 19.3 每个 beta 必须报告

### 信息率

\[
R
=
\mathbb E
KL(
q(Z|X)\Vert p(Z)
).
\]

分别报告：

\[
R_t,\quad R_a,\quad R_v,\quad R_{\text{total}}.
\]

单位：

```text
nats/sample
```

### 教师输出失真

\[
D_{\text{full}}
=
\mathbb E
|\hat y_T(tav)-\hat y_S(tav)|.
\]

### 交互失真

\[
D_{\text{int}}
=
\mathbb E
\sum_A
|\mu_T(A)-\mu_S(A)|.
\]

### 任务性能

- MAE；
- Pearson；
- Acc-2；
- F1。

---

# 20. Rate-Distortion 关键图

至少绘制 4 张：

### 图 1

```text
Total VIB Rate
vs
Test MAE
```

### 图 2

```text
Total VIB Rate
vs
Teacher Full-output Distortion
```

### 图 3

```text
Total VIB Rate
vs
Interaction Distortion
```

### 图 4

```text
R_t / R_a / R_v
vs
beta
```

重点观察：

> Interaction 方法是否在相同 rate 下取得更低任务误差或更低交互失真。

---

# 21. Stage D：缺失模态鲁棒性

测试：

\[
t,\quad a,\quad v,\quad ta,\quad tv,\quad av,\quad tav.
\]

对最终三种模型：

```text
Full KD
subset4
best reliable interaction
```

报告：

- MAE；
- Pearson；
- Acc-2；
- F1。

定义：

\[
\Delta_{\text{missing}}(S)
=
MAE(S)-MAE(tav).
\]

越小越好。

---

# 22. 缺失模态平均性能

\[
MAE_{\text{avg}}
=
\frac1{7}
\sum_{S}
MAE(S).
\]

并单独计算非完整模态：

\[
MAE_{\text{missing-avg}}
=
\frac1{6}
\sum_{S\neq tav}
MAE(S).
\]

如果交互蒸馏有实际机器人部署价值，应至少在 missing-average 上有优势。

---

# 23. Stage D：模态噪声鲁棒性

## 23.1 Audio

加入：

```text
SNR = 30 dB
SNR = 20 dB
SNR = 10 dB
```

以及随机静音：

```text
5%
10%
20%
```

## 23.2 Video

加入：

```text
10% frame drop
25% frame drop
50% frame drop
```

以及：

- Gaussian blur；
- face-region occlusion。

## 23.3 Text

加入：

- 5% token deletion；
- 10% token deletion；
- ASR-like substitution。

主指标：

\[
\text{Performance Degradation}
=
Perf_{\text{clean}}
-
Perf_{\text{noise}}.
\]

---

# 24. Stage E：端侧部署实验

如果论文 motivation 强调家用机器人，必须补充真正的端侧指标。

## 24.1 报告三类预算

### 模型容量

- total parameters；
- trainable parameters；
- model size FP16；
- model size INT8。

### 计算

- GFLOPs/sample；
- 16-frame latency；
- peak VRAM；
- peak RAM。

### 信息率

- VIB nats/sample。

不能混为一个“capacity”。

---

# 25. 学生模型部署版本

主学生：

```text
Qwen3-0.6B-Base
WavLM-Base-Plus
VideoMAE-Base
```

另外增加一个可选 Tiny Student：

```text
DeBERTa-v3-small
WavLM-Base
VideoMAE-Small
```

目的不是重新做全部实验，而是验证：

> 方法是否在更强压缩场景下仍有效。

Tiny Student 只跑：

```text
Student
Full KD
subset4
best interaction
```

---

# 26. Teacher-vs-Student 实验

正式报告：

\[
Gap_{\text{teacher}}
=
Perf(S)-Perf(T).
\]

对于 MAE：

\[
Gap_{\text{teacher}}^{MAE}
=
MAE_T-MAE_S.
\]

比较：

```text
Teacher
Student
Full KD
Subset KD
Reliable Interaction
Reliable Interaction + VIB
```

目的：

> 验证任务专用小模型是否能通过真实标签 + 选择性教师知识超过通用大教师。

---

# 27. 教师错误分组

根据教师完整输入误差：

\[
e_T
=
|\hat y_T-y|.
\]

分为：

```text
Teacher-easy
Teacher-medium
Teacher-hard
```

检查：

> 学生是否只在教师正确样本上提升，还是能够利用真实标签纠正教师错误。

如果后续需要加入 teacher-output reliability，可定义：

\[
w_i^{out}
=
\exp(-\gamma e_{T,i}).
\]

但这属于第二优先级，不要与 interaction reliability 同时引入过多新变量。

---

# 28. 推荐的最终核心方法

如果 Stage A 支持噪声假设，最终方法建议写成：

## Reliability-Aware Selective Interaction Distillation

\[
\boxed{
\mathcal L
=
\mathcal L_{\text{task}}
+
\lambda_f\mathcal L_{\text{fullKD}}
+
\lambda_i
\sum_A
w_A
\rho(
\mu_S(A)-\bar\mu_T(A)
)
+
\beta\mathcal L_{\text{rate}}
}
\]

其中：

\[
w_A
=
f(
|\bar\mu_T(A)|,
\sigma_T(A)
).
\]

推荐优先：

\[
w_A
\propto
\frac{
|\bar\mu_T(A)|
}{
\sigma_T(A)+\epsilon
}.
\]

---

# 29. 更严格的协方差版本

若实验支持 reliability-aware 方法，可以再做一个理论增强实验。

设教师 subset-value 误差协方差：

\[
\Sigma_V.
\]

Möbius 矩阵：

\[
M.
\]

则：

\[
\Sigma_\mu
=
M\Sigma_VM^\top.
\]

使用：

\[
\mathcal L_{\text{Mahalanobis}}
=
(\mu_S-\bar\mu_T)^\top
(
\Sigma_\mu+\lambda I
)^{-1}
(\mu_S-\bar\mu_T).
\]

注意：

> 这一版只有在简单 SNR/variance weighting 已经有效后再做。

不要一开始就增加矩阵估计复杂度。

---

# 30. 建议实验优先级

## P0：必须立刻做

1. z-score interaction4；
2. inverse-variance interaction4；
3. SNR-weighted interaction4；
4. pair-only；
5. random orthogonal。

## P1：P0 有正结果后

6. selective interaction；
7. random non-orthogonal；
8. pair/triple 分解；
9. 3 seeds；
10. benchmark500 高交互分组。

## P2：方法通过后

11. 全量 CMU-MOSEI；
12. 3 seeds 主结果；
13. VIB beta 扫描；
14. 缺失模态；
15. rate–distortion 曲线。

## P3：论文完整性

16. 5 seeds 最终主表；
17. 噪声鲁棒性；
18. Tiny Student；
19. latency / GFLOPs / memory；
20. teacher-vs-student。

---

# 31. 建议停止继续做的实验

暂时不要继续：

- 反复搜索 raw interaction4 的单个 \(\lambda\)；
- 同时加入 unique loss；
- 同时加入 robustness + calibration + dynamic budget；
- 把七维 subset 和七维 interaction 全部叠加；
- 一开始就做 Mahalanobis 全协方差；
- 大规模更换学生骨干；
- 立即对所有 baseline 做 VIB 扫描。

原因：

> 当前最核心的不确定性不是 backbone，而是 interaction target 本身是否可靠。

---

# 32. 推荐结果表 1：benchmark500 方法诊断

| Method | MAE ↓ | Pearson ↑ | Acc-2 ↑ | Pair Error ↓ | Triple Error ↓ | Seed Std ↓ |
|---|---:|---:|---:|---:|---:|---:|
| subset4 |  |  |  |  |  |  |
| raw-int4 |  |  |  |  |  |  |
| z-int4 |  |  |  |  |  |  |
| IV-int4 |  |  |  |  |  |  |
| SNR-int4 |  |  |  |  |  |  |
| pair-only |  |  |  |  | — |  |
| random-orth |  |  |  | — | — |  |

---

# 33. 推荐结果表 2：全量主结果

| Method | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|---:|
| Qwen3-Omni Teacher |  |  |  |  |  |
| Student |  |  |  |  |  |
| Full KD |  |  |  |  |  |
| Subset4 |  |  |  |  |  |
| Reliable Interaction |  |  |  |  |  |
| Reliable Interaction + VIB |  |  |  |  |  |

---

# 34. 推荐结果表 3：交互强度分组

| Method | Low MAE | Mid MAE | High MAE |
|---|---:|---:|---:|
| Full KD |  |  |  |
| Subset4 |  |  |  |
| Reliable Interaction |  |  |  |

最重要的是 High 列。

---

# 35. 推荐结果表 4：缺失模态

| Method | t | a | v | ta | tv | av | tav | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full KD |  |  |  |  |  |  |  |  |
| Subset4 |  |  |  |  |  |  |  |  |
| Reliable Interaction |  |  |  |  |  |  |  |  |

---

# 36. 推荐结果表 5：率失真

| Method | β | Rate ↓ | Full Distortion ↓ | Interaction Distortion ↓ | MAE ↓ |
|---|---:|---:|---:|---:|---:|
| Subset4+VIB | 0 |  |  |  |  |
| Subset4+VIB | 1e-6 |  |  |  |  |
| ... | ... | ... | ... | ... | ... |
| ReliableInt+VIB | 0 |  |  |  |  |
| ReliableInt+VIB | 1e-6 |  |  |  |  |

---

# 37. 最终论文可能形成的故事线

如果 reliability-aware interaction 成功，论文故事可以写成：

```text
1. Generalist omni-modal teacher improves compact sentiment students.

2. Direct high-order Möbius interaction distillation unexpectedly
   underperforms subset matching.

3. We show that Möbius differencing amplifies uncertainty in noisy
   teacher subset predictions, especially for high-order terms.

4. We therefore propose reliability-aware selective interaction
   distillation, which transfers only strong and stable cross-modal
   interactions.

5. Under constrained information rate, reliable interaction knowledge
   yields a better task-performance / teacher-distortion trade-off.

6. The gains are concentrated on high-interaction and missing-modality
   samples, matching the intended edge-robotics use case.
```

这条故事比：

> “Harsanyi 一定比 subset 更好”

更加符合当前真实实验结果。

---

# 38. 如果最终 Reliable Interaction 仍输给 Subset4

也不要继续无限调参。

此时建议正式停止交互 loss 主线，并将结果转化为：

> **A systematic study of when high-order multimodal interaction distillation fails under noisy omni-modal teachers.**

可以形成以下结论：

1. Subset behavior distillation 比高阶差分目标更稳定；
2. Möbius 差分放大 teacher uncertainty；
3. 高阶项尤其脆弱；
4. 交互指标更适合作为诊断或 sample-selection 工具，而非直接 supervision；
5. 最终方法可以改为“interaction-aware sample weighting”：

\[
w_i
=
f(I_T(x_i),U_T(x_i)),
\]

但仍蒸馏 subset values。

这仍然是一个合理的后备研究路线。

---

# 39. 推荐近期执行顺序

最短路径：

```text
Day 1
统计 3 个 teacher probe 的 interaction mean / variance / SNR

Day 2
跑 z-score / inverse-variance / SNR-weighted（单 seed）

Day 3
跑 pair-only + random orthogonal

Day 4
筛选 Top-2 方法，跑 3 seeds

Day 5
做 high-interaction / uncertainty 分组分析

Gate
若 Top-1 ≥ subset4 → 进入全量
否则 → 停止 direct interaction loss，切换后备路线
```

---

# 40. 最终实验决策原则

后续不要以：

> “能不能把 interaction 调赢”

作为目标。

而应该以：

> **“实验数据究竟支持哪一种教师知识表示和蒸馏方式？”**

作为判断标准。

当前最值得验证的假设是：

\[
\boxed{
\text{Raw high-order interaction is noisy}
}
\]

以及：

\[
\boxed{
\text{Strong + Stable Interaction}
\text{ may be more useful than }
\text{All Interaction}
}
\]

如果这一假设成立，就形成新的主方法：

> **Reliability-Aware Selective Interaction Distillation**

如果不成立，则 subset4 本身就是当前最可靠的答案，应据此重新定位论文。
