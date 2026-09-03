# RDID-MOSEI 后续实验执行方案

> **项目**：Qwen3-Omni → 轻量多模态学生的 CMU-MOSEI 情感蒸馏
> **教师**：Qwen3-Omni-30B-A3B-Instruct + 共享情感 Probe
> **学生**：Qwen3-0.6B-Base + WavLM-Base-Plus + VideoMAE-Base
> **当前阶段目标**：解释 `subset4 > raw interaction4` 的原因，并验证“可靠性加权的选择性交互蒸馏”是否能成为新的主方法。

---

# 1. 当前状态摘要

目前已经完成或基本确认：

- Student-only 基线；
- Full TAV KD；
- 7-subset KD；
- Möbius 全交互 KD；
- subset4；
- interaction4；
- 多随机种子实验；
- interaction loss 权重搜索；
- Qwen3-Omni 教师七子集推理；
- 共享情感 Probe；
- Qwen3-0.6B-Base + WavLM-Base-Plus + VideoMAE-Base 学生骨干；
- Möbius / inverse Möbius 实现；
- 随机正交矩阵工具。

当前关键观察：

\[
\text{Full KD} > \text{Student-only}
\]

说明教师知识有效。

同时：

\[
\text{subset4} > \text{interaction4}
\]

并且 interaction4 的随机种子方差更大。

因此下一阶段不再继续单纯搜索一个全局 interaction 权重，而重点验证：

\[
\boxed{
\text{Möbius 高阶差分是否放大了教师子集预测噪声}
}
\]

以及：

\[
\boxed{
\text{只蒸馏强且稳定的交互是否优于 raw interaction}
}
\]

---

# 2. 后续实验总路线

```text
Stage A：benchmark500 诊断
        ↓
确定 interaction 是否值得继续
        ↓
Stage B：全量 CMU-MOSEI 核心实验
        ↓
Stage C：VIB / Rate-Distortion
        ↓
Stage D：缺失模态与噪声鲁棒性
        ↓
Stage E：端侧部署与最终论文实验
```

原则：

> 每一阶段设置 Go/No-Go 条件。前一阶段未通过，不进入下一阶段的大规模计算。

---

# 3. Stage A：benchmark500 诊断实验

## 3.1 目的

回答四个问题：

1. interaction4 输给 subset4 是否只是数值尺度问题？
2. 三阶交互是否是主要噪声来源？
3. 教师交互的不确定性加权是否有效？
4. Möbius 坐标是否优于随机坐标？

---

# 4. A1：Z-score Interaction4

## 4.1 Motivation

不同交互项的数值尺度可能不同，例如：

\[
|\mu(ta)| \ll |\mu(tav)|
\]

或者相反。

若直接使用 Huber / MSE：

\[
\sum_A \rho(\mu_S(A)-\mu_T(A))
\]

大尺度项会主导梯度。

因此先排除纯尺度问题。

## 4.2 方法

对训练集每个交互项：

\[
m_A = \mathbb E[\mu_T(A)]
\]

\[
s_A = \sqrt{\operatorname{Var}[\mu_T(A)]}
\]

定义：

\[
\tilde\mu_T(A)
=
\frac{\mu_T(A)-m_A}{s_A+\epsilon}
\]

学生使用同一教师统计量：

\[
\tilde\mu_S(A)
=
\frac{\mu_S(A)-m_A}{s_A+\epsilon}
\]

损失：

\[
\mathcal L_{\text{z-int}}
=
\sum_{A\in\{ta,tv,av,tav\}}
\rho(
\tilde\mu_S(A)-\tilde\mu_T(A)
)
\]

## 4.3 判断

若：

\[
\text{Z-score Interaction4}
\gg
\text{Raw Interaction4}
\]

说明原失败可能主要来自数值尺度。

若几乎无提升，则继续验证噪声假设。

---

# 5. A2：Pair-only Interaction

## 5.1 Motivation

二阶交互：

\[
\mu(ta)
=
V(ta)-V(t)-V(a)+b
\]

包含 3 个有噪声的子集输出。

三阶交互：

\[
\begin{aligned}
\mu(tav)
=&V(tav)-V(ta)-V(tv)-V(av)\\
&+V(t)+V(a)+V(v)-b
\end{aligned}
\]

包含 7 个子集输出的加减。

若每个教师子集值带误差：

\[
\hat V(S)=V^*(S)+\epsilon_S
\]

且简单假设：

\[
\operatorname{Var}(\epsilon_S)=\sigma^2
\]

则二阶交互噪声约为：

\[
3\sigma^2
\]

而三阶可达到：

\[
7\sigma^2
\]

量级。

因此必须单独测试三阶项是否在破坏训练。

## 5.2 方法

只蒸馏：

\[
\mu(ta),\mu(tv),\mu(av)
\]

即：

\[
\mathcal L_{\text{pair}}
=
\sum_{A\in\{ta,tv,av\}}
\rho(
\mu_S(A)-\mu_T(A)
)
\]

同时单独跑：

\[
\mathcal L_{\text{triple}}
=
\rho(
\mu_S(tav)-\mu_T(tav)
)
\]

比较：

```text
pair-only
triple-only
pair + triple
```

## 5.3 关键判断

若：

\[
\text{pair-only} > \text{pair+triple}
\]

说明三阶交互很可能是不稳定监督源。

---

# 6. A3：教师多 Probe 不确定性统计

## 6.1 使用已有 Probe seeds

使用已有多个教师 Probe，例如：

```text
seed 2026
seed 2027
seed 2028
```

对每个样本 \(x\) 和交互项 \(A\)，计算：

\[
\mu_T^{(1)}(A|x),
\mu_T^{(2)}(A|x),
\mu_T^{(3)}(A|x)
\]

均值：

\[
\bar\mu_T(A|x)
=
\frac1K
\sum_{k=1}^K
\mu_T^{(k)}(A|x)
\]

方差：

\[
\sigma_T^2(A|x)
=
\frac1{K-1}
\sum_k
\left[
\mu_T^{(k)}(A|x)
-
\bar\mu_T(A|x)
\right]^2
\]

## 6.2 必须先做统计图

对：

\[
ta,tv,av,tav
\]

分别统计：

- mean；
- std；
- median；
- P90 variance；
- sign agreement；
- SNR。

特别检查：

\[
\sigma^2_{tav}
\]

是否显著高于三个 pair 项。

---

# 7. A4：Inverse-Variance Interaction KD

## 7.1 方法

定义：

\[
w_A(x)
=
\frac{1}{\sigma_T^2(A|x)+\epsilon}
\]

为了避免极端权重，归一化：

\[
\tilde w_A
=
\frac{w_A}
{\frac1{|\mathcal A|}\sum_B w_B}
\]

再裁剪：

\[
\tilde w_A
\leftarrow
\operatorname{clip}
(
\tilde w_A,
w_{\min},
w_{\max}
)
\]

初始设置：

```yaml
epsilon: 1e-4
w_min: 0.25
w_max: 4.0
```

损失：

\[
\mathcal L_{\text{IV}}
=
\sum_A
\tilde w_A
\rho(
\mu_S(A)-\bar\mu_T(A)
)
\]

## 7.2 含义

教师不同 Probe 对某个交互越一致：

\[
\sigma_T^2(A)\downarrow
\]

学生越应该学习。

---

# 8. A5：SNR-weighted Interaction KD

这是当前最值得优先验证的方法。

## 8.1 Motivation

只看方差有一个问题。

例如：

\[
\mu=0.01\pm0.001
\]

虽然很稳定，但该交互几乎没有实际作用。

而：

\[
\mu=-1.0\pm0.1
\]

虽然方差更大，但信号非常强。

因此使用：

\[
SNR_A(x)
=
\frac{
|\bar\mu_T(A|x)|
}{
\sigma_T(A|x)+\epsilon
}
\]

## 8.2 权重

\[
w_A
=
\operatorname{clip}
\left(
\frac{SNR_A}
{\operatorname{mean}_B SNR_B},
w_{\min},
w_{\max}
\right)
\]

损失：

\[
\boxed{
\mathcal L_{\text{SNR}}
=
\sum_A
w_A
\rho(
\mu_S(A)-\bar\mu_T(A)
)
}
\]

## 8.3 推荐版本

至少比较：

```text
SNR pair-only
SNR pair+triple
```

如果 `SNR pair-only` 最好，可以直接把三阶项从最终主方法中移除。

---

# 9. A6：Selective Interaction

进一步从“加权”变成“选择”。

## 9.1 Gate

定义：

\[
g_A(x)
=
\mathbf 1[
SNR_A(x)>\tau
]
\]

损失：

\[
\mathcal L_{\text{selective}}
=
\frac{
\sum_A
g_A
\rho(
\mu_S(A)-\bar\mu_T(A)
)
}{
\sum_Ag_A+\epsilon
}
\]

## 9.2 阈值

不要做大规模超参搜索。

使用训练集 SNR 分位数：

```text
Top 25%
Top 50%
Top 75%
```

只测试三个即可。

---

# 10. A7：Random Orthogonal Coordinate

这是必须补的公平性实验。

## 10.1 方法

令中心化 subset value：

\[
\tilde{\mathbf v}
=
[
V(t)-b,
V(a)-b,
V(v)-b,
V(ta)-b,
V(tv)-b,
V(av)-b,
V(tav)-b
]^\top
\]

生成随机正交矩阵：

\[
Q^\top Q=I
\]

定义：

\[
\mathbf r=Q\tilde{\mathbf v}
\]

教师学生损失：

\[
\mathcal L_{\text{orth}}
=
\rho(
Q\tilde{\mathbf v}_S
-
Q\tilde{\mathbf v}_T
)
\]

建议至少三个固定矩阵 seed：

```text
100
200
300
```

## 10.2 目的

如果：

\[
\text{Möbius}
\approx
\text{Random Orthogonal}
\]

说明换坐标本身没有特殊价值。

如果：

\[
\text{Reliable Möbius}
>
\text{Random Orthogonal}
\]

才支持“交互结构 + 可靠性”确实带来针对性归纳偏置。

---

# 11. A8：Random Non-Orthogonal Coordinate

构造随机可逆矩阵：

\[
R
\]

并控制：

\[
\kappa(R)
\approx
\kappa(M_{\text{Mobius}})
\]

其中 \(\kappa\) 是 condition number。

目的：

> 排除 Möbius 的表现仅仅来自非正交缩放。

该实验优先级低于 Random Orthogonal，但最终论文建议补。

---

# 12. Stage A 执行顺序

## 第一轮：1 seed 筛选

固定：

```text
seed = 42
```

运行：

| ID | 方法 |
|---|---|
| A0 | subset4 |
| A1 | raw interaction4 |
| A2 | z-score interaction4 |
| A3 | pair-only |
| A4 | inverse-variance interaction4 |
| A5 | SNR interaction4 |
| A6 | SNR pair-only |
| A7 | random orthogonal |

## 第二轮

选表现最好的 2 个新 interaction 方法，运行：

```text
seed 13
seed 42
seed 2026
```

同时补：

- subset4；
- raw interaction4。

---

# 13. Stage A 必须记录的指标

主任务：

- MAE；
- Pearson；
- Acc-2；
- F1；
- Acc-7。

交互：

\[
E_{\text{pair}}
=
\frac1{3N}
\sum_i
\sum_{A\in\{ta,tv,av\}}
|\mu_S^i(A)-\bar\mu_T^i(A)|
\]

\[
E_{\text{triple}}
=
\frac1N
\sum_i
|\mu_S^i(tav)-\bar\mu_T^i(tav)|
\]

训练稳定性：

- best epoch；
- loss mean/std；
- gradient norm；
- seed std。

---

# 14. Stage A Go / No-Go

## Go 条件

满足以下任意一条：

### G1

\[
MAE_{\text{new-int}}
<
MAE_{\text{subset4}}
\]

且 3 seeds 均值优势稳定。

### G2

整体 MAE 基本相同，但：

\[
MAE_{\text{HighInteraction,new}}
<
MAE_{\text{HighInteraction,subset4}}
\]

明显。

### G3

相同 MAE 下：

\[
E_{\text{pair/new}}
<
E_{\text{pair/subset4}}
\]

且差距明显。

### G4

在后续 VIB 下，相同信息率时新方法更优。

---

## No-Go 条件

如果：

- z-score 无效；
- pair-only 无效；
- variance/SNR weighting 无效；
- random coordinate 与 Möbius 相同；
- 新方法 3-seed 仍明显弱于 subset4；

则停止 direct interaction loss 主线。

不要继续无限调 \(\lambda\)。

---

# 15. No-Go 后备路线

如果 direct interaction KD 失败，则把 interaction 改成：

> **分析工具 / 样本权重，而不是直接监督目标。**

例如继续蒸馏：

\[
V(ta),V(tv),V(av),V(tav)
\]

但根据教师交互强度和可靠性给样本权重：

\[
I_T(x)
=
\sum_A|\bar\mu_T(A|x)|
\]

\[
U_T(x)
=
\frac1{|\mathcal A|}
\sum_A\sigma_T^2(A|x)
\]

定义：

\[
w_i
=
f(I_T(x_i),U_T(x_i))
\]

最终：

\[
\mathcal L
=
\mathcal L_{\text{task}}
+
\lambda_f\mathcal L_{\text{fullKD}}
+
\lambda_s
w_i
\mathcal L_{\text{subset4}}
\]

该路线更稳，也与当前实验事实一致。

---

# 16. Stage B：全量 CMU-MOSEI 核心实验

只有 Stage A 通过后进行。

## 16.1 全量实验只保留 5 组

```text
B0 Student-only
B1 Full KD
B2 subset4
B3 Best Reliable Interaction
B4 Best Reliable Interaction + VIB
```

不要把 benchmark500 的全部诊断方法搬到全量。

---

# 17. Stage B 训练顺序

## 第一步

每个模型先运行：

```text
1 seed
```

验证：

- 显存；
- wall time；
- loss；
- validation convergence；
- teacher cache 完整性。

## 第二步

保留有效方法运行：

```text
3 seeds:
13
42
2026
```

## 第三步

最终主结果如果资源允许：

```text
5 seeds
```

---

# 18. Stage B 统计检验

必须做 paired test。

对于两种模型 A、B：

\[
d_i
=
|\hat y_i^A-y_i|
-
|\hat y_i^B-y_i|
\]

进行：

```text
paired bootstrap
10000 resamples
```

报告：

- mean difference；
- 95% CI；
- p-value；
- effect size。

不能只比较单次最佳结果。

---

# 19. 高交互分组实验

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
|\bar\mu_T(tav)|
\]

按分位数划分：

```text
Low:    0–33%
Medium: 33–66%
High:   66–100%
```

比较：

```text
Full KD
subset4
Reliable Interaction
```

重点报告：

\[
MAE_{\text{High}}
\]

如果方法确实学习跨模态关系，最理想结果是：

\[
\Delta_{\text{High}}
>
\Delta_{\text{Low}}
\]

---

# 20. 教师不确定性分组

定义：

\[
U_T(x)
=
\frac1{4}
\sum_A
\sigma_T^2(A|x)
\]

划分：

```text
Low uncertainty
Medium uncertainty
High uncertainty
```

比较：

```text
raw interaction
reliable interaction
subset4
```

重点验证：

> reliability-aware 方法是否在高教师不确定性样本上避免负迁移。

---

# 21. Stage C：VIB / Rate-Distortion

只有 Stage B 的可靠性交互方法成立后再做。

## 21.1 只比较

```text
subset4 + VIB
best reliable interaction + VIB
```

---

# 22. Beta 网格

\[
\beta
\in
\{
0,
10^{-6},
10^{-5},
10^{-4},
10^{-3}
\}
\]

若 \(10^{-3}\) 出现 posterior collapse，可补：

\[
3\times10^{-4}
\]

---

# 23. 信息率

报告：

\[
R_t
=
\mathbb E KL(q(Z_t|X_t)\Vert p(Z_t))
\]

\[
R_a
=
\mathbb E KL(q(Z_a|X_a)\Vert p(Z_a))
\]

\[
R_v
=
\mathbb E KL(q(Z_v|X_v)\Vert p(Z_v))
\]

总 rate：

\[
R=R_t+R_a+R_v
\]

单位：

```text
nats/sample
```

---

# 24. Rate-Distortion 图

必须至少有：

## Figure 1

```text
Rate vs Test MAE
```

## Figure 2

```text
Rate vs Full Teacher Distortion
```

其中：

\[
D_{\text{full}}
=
\mathbb E
|\hat y_T(tav)-\hat y_S(tav)|
\]

## Figure 3

```text
Rate vs Interaction Distortion
```

## Figure 4

```text
beta vs R_t / R_a / R_v
```

---

# 25. Stage C 的核心问题

最终希望回答：

> 在相同信息率 \(R\) 下，Reliable Interaction 是否取得更低 MAE？

即：

\[
MAE_{\text{Reliable}}(R)
<
MAE_{\text{Subset}}(R)
\]

或者：

> 在相同 MAE 下，是否需要更低 rate？

如果成立，率失真主线才真正有价值。

---

# 26. Stage D：缺失模态

测试：

\[
t,a,v,ta,tv,av,tav
\]

比较：

```text
Full KD
subset4
best reliable interaction
```

指标：

- MAE；
- Pearson；
- Acc-2；
- F1。

定义：

\[
MAE_{\text{missing-avg}}
=
\frac1{6}
\sum_{S\neq tav}
MAE(S)
\]

这是家用机器人端侧场景的重要指标。

---

# 27. Stage D：音频噪声

测试：

```text
Clean
30 dB SNR
20 dB SNR
10 dB SNR
```

以及：

```text
5% random silence
10% random silence
20% random silence
```

记录：

\[
\Delta MAE
=
MAE_{\text{noise}}
-
MAE_{\text{clean}}
\]

---

# 28. Stage D：视频噪声

测试：

```text
10% frame drop
25% frame drop
50% frame drop
```

以及：

- blur；
- face occlusion。

---

# 29. Stage D：文本噪声

测试：

```text
5% token deletion
10% token deletion
```

以及少量 ASR-like substitution。

不建议加入大量文本增强，避免把论文主线带偏。

---

# 30. Stage E：端侧部署指标

如果论文 Motivation 强调家用机器人，最终必须报告真实部署相关指标。

分三类。

## 30.1 模型容量

- total parameters；
- trainable parameters；
- FP16 model size；
- INT8 model size。

## 30.2 计算预算

- GFLOPs/sample；
- latency/sample；
- peak VRAM；
- peak RAM；
- 视频帧数；
- multimodal token 数。

## 30.3 信息预算

- VIB nats/sample。

这三类指标不能混称为“capacity”。

---

# 31. Tiny Student 可选实验

只有主方法已经成立后再做。

例如：

```text
Text:
DeBERTa-v3-small

Audio:
WavLM-Base

Video:
VideoMAE-Small
```

只比较：

```text
Student
Full KD
subset4
best reliable interaction
```

目的：

> 验证方法在更强端侧压缩条件下是否仍有效。

---

# 32. Teacher vs Student

正式主表建议加入 Qwen3-Omni Teacher。

定义：

对于 MAE：

\[
G_T
=
MAE_T-MAE_S
\]

如果：

\[
G_T>0
\]

表示学生在 CMU-MOSEI 专项任务上超过教师。

主表：

| Method | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ |
|---|---:|---:|---:|---:|
| Qwen3-Omni Teacher | | | | |
| Student | | | | |
| Full KD | | | | |
| Subset4 | | | | |
| Reliable Interaction | | | | |
| Reliable Interaction + VIB | | | | |

---

# 33. 实验优先级

## P0：立即做

1. 三 Probe 交互 mean / variance / SNR 统计；
2. z-score interaction4；
3. pair-only；
4. inverse-variance；
5. SNR-weighted；
6. random orthogonal。

## P1：P0 有希望后

7. selective interaction；
8. triple-only；
9. random non-orthogonal；
10. Top-2 方法 3 seeds；
11. High-interaction 分组；
12. uncertainty 分组。

## P2：方法通过后

13. 全量 CMU-MOSEI；
14. 3 seeds；
15. paired bootstrap；
16. VIB beta 扫描；
17. Rate-Distortion 曲线；
18. 缺失模态。

## P3：论文完整性

19. 5 seeds；
20. 音频/视频/文本噪声；
21. latency / GFLOPs / VRAM；
22. Tiny Student；
23. Teacher-vs-Student。

---

# 34. 当前不建议做的实验

暂时停止：

- 继续反复搜索 raw interaction4 的一个全局 \(\lambda\)；
- Unique loss；
- Dynamic token budget；
- 大规模 robustness loss；
- Calibration loss；
- 同时叠加全 7 subset KD + 全 7 Möbius KD；
- 一开始就做 Mahalanobis 全协方差；
- 继续更换学生 backbone；
- 在方法未通过前直接做所有全量消融。

当前最核心的问题是：

\[
\boxed{
\text{interaction target 是否可靠}
}
\]

而不是模型规模。

---

# 35. 如果可靠性交互成功

论文主线：

> **Reliability-Aware Selective Interaction Distillation for Resource-Constrained Multimodal Sentiment Analysis**

核心方法：

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
\propto
\frac{
|\bar\mu_T(A)|
}{
\sigma_T(A)+\epsilon
}
\]

论文逻辑：

```text
大模型跨模态能力强
↓
但教师子集预测存在不确定性
↓
Möbius 高阶差分会放大这种不确定性
↓
raw interaction KD 出现负迁移
↓
只选择强且稳定的交互蒸馏
↓
有限信息率学生获得更好的 rate–distortion trade-off
↓
适用于家用机器人等端侧多模态情感理解
```

---

# 36. 如果可靠性交互仍失败

不要继续硬调。

转为第二条路线：

> **Interaction-Aware Reliable Subset Distillation**

仍蒸馏稳定的：

\[
V(ta),V(tv),V(av),V(tav)
\]

但利用交互强度和教师不确定性决定：

- 哪些样本重点蒸馏；
- 哪些 subset 权重大；
- 哪些教师目标应降低权重。

这样 interaction 变成：

> teacher reliability / sample difficulty estimator

而不是直接 prediction target。

该路线通常比强行使用高阶差分监督更稳。

---

# 37. 近期实际执行顺序

## Day 1

计算 3 个教师 Probe：

\[
\bar\mu_T,\quad
\sigma_T,\quad
SNR_T
\]

并输出四种交互：

```text
ta
tv
av
tav
```

的统计表与分布。

## Day 2

单 seed 跑：

```text
z-score interaction4
pair-only
inverse-variance
SNR interaction4
SNR pair-only
```

## Day 3

运行：

```text
random orthogonal
triple-only
```

整理 MAE / Pearson / interaction error。

## Day 4

选择 Top-2 方法运行：

```text
seed 13
seed 42
seed 2026
```

## Day 5

做：

```text
High-interaction split
Teacher-uncertainty split
```

并做 Go/No-Go 决策。

---

# 38. 最终 Go / No-Go 决策

## 如果：

\[
\text{Reliable Interaction}
\ge
\text{Subset4}
\]

或者在 High-interaction / Rate-Distortion 上有明显优势：

> **Go：进入全量 CMU-MOSEI。**

## 如果：

\[
\text{Reliable Interaction}
<
\text{Subset4}
\]

并且随机坐标结果也无法支持 Möbius 特殊性：

> **No-Go：停止 direct interaction KD。**

改用：

> Interaction-Aware Reliable Subset Distillation。

---

# 39. 当前最重要的研究原则

后续实验目标不是：

> “想办法把 interaction 调赢。”

而是：

> **判断实验数据究竟支持哪一种教师知识形式。**

当前最值得验证的两个假设：

\[
\boxed{
\text{高阶 Möbius 差分放大教师噪声}
}
\]

以及：

\[
\boxed{
\text{强且稳定的交互比全部交互更值得蒸馏}
}
\]

如果成立，就形成可靠性交互蒸馏的新方法。

如果不成立，则 `subset4` 是当前更可靠的监督形式，应围绕它重新组织论文方法。
