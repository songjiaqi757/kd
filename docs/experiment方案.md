# Uniform Interaction Distillation：实验部分新增内容方案

## 1. 目标

当前主性能实验已经基本完整，后续新增内容应重点补足：

1. interaction-level mechanism evidence；
2. interaction order ablation；
3. coordinate-space control；
4. efficiency evidence；
5. training dynamics 与 qualitative analysis。

不再优先增加大量新的 KD baseline，也不继续扩展复杂的 Reliability / Utility weighting 变体。

---

# 2. 新增内容总览

| 优先级 | 内容                                       | 是否需要重新训练 | 主要目的                                            | 建议位置    |
| --- | ---------------------------------------- | -------: | ----------------------------------------------- | ------- |
| P0  | Interaction Reconstruction Analysis      |        否 | 验证 student 是否真正逼近 teacher interaction structure | 正文      |
| P0  | Interaction-Strength Stratified Analysis |        否 | 分析方法在哪类样本上收益更明显                                 | 正文      |
| P1  | First + Second-order Interaction         |        是 | 分离 second-order 和 third-order interaction 的贡献   | 正文      |
| P1  | Random Orthogonal Basis Distillation     |        是 | 排除“任意可逆变换都有效”的解释                                | 正文      |
| P1  | Efficiency Analysis                      |      基本否 | 验证 interaction supervision 不增加部署复杂度             | 正文      |
| P2  | Teacher Subset / Interaction Statistics  |        否 | 展示七子集及 interaction 本身的结构                        | 正文或附加材料 |
| P2  | Epoch-wise Training Dynamics             |        否 | 利用逐 checkpoint 结果展示训练轨迹                         | 正文或附加材料 |
| P3  | Interaction Case Study                   |        否 | 直观展示 teacher/student interaction profile        | 附加材料    |
| P3  | Hyperparameter Sensitivity               |       可能 | 检查 $\lambda_{\mathrm{int}}$ 稳定性                 | 附加材料    |

---

# 3. P0：Interaction Reconstruction Analysis

## 3.1 目的

直接衡量不同学生对 teacher interaction coordinates 的恢复程度。

对于每个 interaction coordinate

$$
S\in
\{
\mathrm{T},\mathrm{A},\mathrm{V},
\mathrm{TA},\mathrm{TV},\mathrm{AV},\mathrm{TAV}
\},
$$

定义 teacher--student interaction error：

$$
E(S)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
I_i^{\mathcal S}(S)
-
I_i^{\mathcal T}(S)
\right|.
$$

同时计算不同阶数的平均误差：

$$
E_{\mathrm{1st}}
=
\frac{
E(\mathrm T)+E(\mathrm A)+E(\mathrm V)
}{3},
$$

$$
E_{\mathrm{2nd}}
=
\frac{
E(\mathrm{TA})+E(\mathrm{TV})+E(\mathrm{AV})
}{3},
$$

$$
E_{\mathrm{3rd}}
=
E(\mathrm{TAV}).
$$

---

## 3.2 对比方法

至少包含：

* Full KD
* Subset-7 KD
* First-order Interaction
* Uniform Interaction Distillation

如果后续完成 First+Second-order，则同时加入：

* First + Second-order Interaction

---

## 3.3 所需数据

使用最终 checkpoint 对测试集进行七 subset inference：

```text
T
A
V
TA
TV
AV
TAV
```

得到：

```text
student subset predictions
teacher subset predictions
```

然后使用与 Method 完全相同的 anchor $b$ 和 Anchored Möbius Decomposition 计算七个 interaction coordinates。

---

## 3.4 建议表格

### Table: Interaction Reconstruction Error

| Method              | $I_T$ | $I_A$ | $I_V$ | $I_{TA}$ | $I_{TV}$ | $I_{AV}$ | $I_{TAV}$ | Avg. ↓ |
| ------------------- | ----: | ----: | ----: | -------: | -------: | -------: | --------: | -----: |
| Full KD             |       |       |       |          |          |          |           |        |
| Subset-7 KD         |       |       |       |          |          |          |           |        |
| First-order         |       |       |       |          |          |          |           |        |
| Uniform Interaction |       |       |       |          |          |          |           |        |

---

## 3.5 建议图

优先将结果画成三阶误差图：

```text
First-order
Second-order
Third-order
```

纵轴：

```text
Teacher–Student Interaction Error ↓
```

方法：

```text
Full KD
Subset-7
First-order
Uniform Interaction
```

正文优先使用图，七个 coordinate 的完整数值可保留为表格。

---

# 4. P0：Interaction-Strength Stratified Analysis

## 4.1 目的

分析 Uniform Interaction 是否在跨模态 interaction 较强的样本上获得更明显的收益。

---

## 4.2 Interaction Strength

推荐使用 teacher high-order interaction magnitude：

$$
H_i
=
\frac{
|I_i(\mathrm{TA})|
+
|I_i(\mathrm{TV})|
+
|I_i(\mathrm{AV})|
+
|I_i(\mathrm{TAV})|
}{4}.
$$

按照 $H_i$ 将测试样本划分为三个等规模区间：

```text
Low
Medium
High
```

划分阈值由 teacher interaction strength 的 tercile 决定。

---

## 4.3 对比方法

建议：

* Full KD
* Subset-7 KD
* Uniform Interaction Distillation

可选：

* Adapted Student

---

## 4.4 汇报指标

每个 interaction-strength group 分别计算：

```text
MAE
Pearson
```

正文主图优先只展示 MAE。

---

## 4.5 建议图

横轴：

```text
Low        Medium        High
```

纵轴：

```text
MAE ↓
```

比较：

```text
Full KD
Subset-7 KD
Uniform Interaction
```

也可以额外计算：

$$
\Delta_i
=
\mathrm{MAE}_{\mathrm{Uniform}}
-
\mathrm{MAE}_{\mathrm{FullKD}}.
$$

并画：

```text
Relative MAE improvement
```

---

# 5. P1：First + Second-order Interaction

## 5.1 目的

当前已有：

```text
First-order
vs.
First + Second + Third-order
```

但无法判断性能增益主要来自：

```text
second-order interaction
```

还是：

```text
third-order interaction
```

因此增加：

```text
First + Second-order Interaction
```

---

## 5.2 Loss

监督：

$$
I_T,\ I_A,\ I_V,\ I_{TA},\ I_{TV},\ I_{AV}
$$

不监督：

$$
I_{TAV}.
$$

定义：

$$
\mathcal L_{\mathrm{int}}^{1+2}
=
\frac{1}{6}
\sum_{
S\in
\{
T,A,V,TA,TV,AV
\}
}
\ell_{\mathrm{Huber}}
\left(
I^{\mathcal S}(S),
I^{\mathcal T}(S)
\right).
$$

其余训练配置与 Uniform Interaction 完全一致。

---

## 5.3 最终消融

| Method              | First-order | Second-order | Third-order | MAE ↓ |
| ------------------- | :---------: | :----------: | :---------: | ----: |
| First-order         |      ✓      |              |             |       |
| First + Second      |      ✓      |       ✓      |             |       |
| Uniform Interaction |      ✓      |       ✓      |      ✓      |       |

建议同时报告：

```text
Pearson
Acc-2
F1
Acc-7
```

正文表格可只保留：

```text
MAE
Pearson
Acc-2
```

---

# 6. P1：Random Orthogonal Basis Distillation

## 6.1 目的

验证性能提升是否来自 Möbius interaction coordinates 本身，而不是任意坐标变换。

由于固定 anchor 下：

$$
\mathbf r
\longleftrightarrow
\mathbf I
$$

是可逆变换，因此增加随机正交坐标系作为控制。

---

## 6.2 构造

将七 subset responses 写为：

$$
\mathbf r_i
=
[
r_i(T),
r_i(A),
r_i(V),
r_i(TA),
r_i(TV),
r_i(AV),
r_i(TAV)
]^\top.
$$

采样固定随机矩阵并进行 QR 分解得到：

$$
Q^\top Q=I.
$$

教师和学生均使用相同的 $Q$：

$$
\mathbf z_i^{\mathcal T}
=
Q\mathbf r_i^{\mathcal T},
$$

$$
\mathbf z_i^{\mathcal S}
=
Q\mathbf r_i^{\mathcal S}.
$$

蒸馏目标：

$$
\mathcal L_{\mathrm{rand}}
=
\frac{1}{7}
\sum_{k=1}^{7}
\ell_{\mathrm{Huber}}
\left(
z_{i,k}^{\mathcal S},
z_{i,k}^{\mathcal T}
\right).
$$

---

## 6.3 控制条件

Random Basis 和 Uniform Interaction 保持完全一致：

```text
student architecture
teacher predictions
seven subset responses
loss type
loss weight
number of supervised coordinates
training epochs
optimizer
random seed
```

唯一变化：

```text
coordinate transformation
```

---

## 6.4 建议表格

| Distillation Space | Same 7 Subsets | Invertible Transform | Structured Interaction | MAE ↓ |
| ------------------ | :------------: | :------------------: | :--------------------: | ----: |
| Subset-7           |        ✓       |           –          |            –           |       |
| Random Orthogonal  |        ✓       |           ✓          |            –           |       |
| Möbius Interaction |        ✓       |           ✓          |            ✓           |       |

这是验证方法核心归纳偏置的重要控制实验。

---

# 7. P1：Efficiency Analysis

## 7.1 目的

验证 interaction supervision 仅增加训练阶段监督，不改变部署模型结构。

---

## 7.2 汇报内容

至少统计：

```text
Total parameters
Trainable parameters
Deployment parameters
Peak training memory
Training time / epoch
Model-only inference latency
End-to-end inference latency
```

---

## 7.3 对比方法

建议：

* Adapted Student
* Full KD
* Subset-7 KD
* Uniform Interaction

---

## 7.4 建议表格

| Method              | Total Params | Trainable Params | Deploy Params | Train Mem. ↓ | Time/Epoch ↓ | Latency ↓ |
| ------------------- | -----------: | ---------------: | ------------: | -----------: | -----------: | --------: |
| Adapted Student     |              |                  |               |              |              |           |
| Full KD             |              |                  |               |              |              |           |
| Subset-7 KD         |              |                  |               |              |              |           |
| Uniform Interaction |              |                  |               |              |              |           |

额外增加：

```text
Inference branches
```

所有最终学生均为：

```text
1 × TAV
```

---

# 8. P2：Teacher Subset Behavior

## 8.1 目的

展示 teacher 在不同 modality subsets 上的预测能力和模态贡献差异。

---

## 8.2 建议表格

| Subset | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ |
| ------ | ----: | --------: | ------: | ---: | ------: |
| T      |       |           |         |      |         |
| A      |       |           |         |      |         |
| V      |       |           |         |      |         |
| TA     |       |           |         |      |         |
| TV     |       |           |         |      |         |
| AV     |       |           |         |      |         |
| TAV    |       |           |         |      |         |

MOSEI 和 MOSI 可以分别计算。

正文若空间不足，只保留：

```text
MAE
Pearson
```

---

# 9. P2：Teacher Interaction Statistics

除 subset-level performance 外，建议统计 teacher interaction coordinates 的基本分布。

对于每个：

$$
I_T,
I_A,
I_V,
I_{TA},
I_{TV},
I_{AV},
I_{TAV},
$$

计算：

```text
Mean absolute magnitude
Standard deviation
Median absolute magnitude
```

建议表格：

| Coordinate | Mean $|I|$ | Median $|I|$ | Std. |
|---|---:|---:|---:|
| $I_T$ | | | |
| $I_A$ | | | |
| $I_V$ | | | |
| $I_{TA}$ | | | |
| $I_{TV}$ | | | |
| $I_{AV}$ | | | |
| $I_{TAV}$ | | | |

也可将其画为：

```text
violin plot / box plot
```

用于展示不同 interaction order 的尺度和分布。

---

# 10. P2：Epoch-wise Training Dynamics

## 10.1 现有数据

当前 MOSEI 和 MOSI 已保存逐 epoch checkpoints，并完成对应测试。

因此无需重新训练。

---

## 10.2 推荐方法

仅选择：

* Adapted Student
* Full KD
* Subset-7
* Uniform Interaction

避免同时绘制全部 9 个方法。

---

## 10.3 建议图

横轴：

```text
Training Epoch
```

纵轴：

```text
MAE ↓
```

可以做：

```text
(a) CMU-MOSEI
(b) CMU-MOSI
```

若正文版面有限，则放置在补充实验中。

---

# 11. P3：Interaction Case Study

## 11.1 目的

通过具体样本直观展示不同方法对 teacher interaction profile 的恢复能力。

---

## 11.2 样本选择

选取 2--4 个具有代表性的 test samples：

```text
1. strong TA interaction
2. strong TV interaction
3. strong AV interaction
4. strong TAV residual
```

避免只挑预测完全正确的样本。

---

## 11.3 展示内容

每个样本展示：

```text
Ground-truth sentiment
Teacher prediction
Full KD prediction
Subset-7 prediction
Uniform prediction
```

以及七个 interaction coordinates：

$$
I_T,\ I_A,\ I_V,\ I_{TA},\ I_{TV},\ I_{AV},\ I_{TAV}.
$$

推荐使用 grouped bar chart。

---

# 12. P3：Hyperparameter Sensitivity

优先级较低。

如果最终需要补充，可以测试：

$$
\lambda_{\mathrm{int}}
\in
\{
0.25,\ 0.5,\ 1,\ 2,\ 4
\}.
$$

保持：

$$
\lambda_{\mathrm{full}}=1.
$$

汇报：

```text
MAE
Pearson
```

如果 $\lambda_{\mathrm{int}}=1$ 附近表现稳定，即可作为 uniform weighting 的补充证据。

该实验不优先于：

```text
Random Basis
First+Second-order
Interaction Reconstruction
```

---

# 13. 不建议继续扩展的实验

当前不再优先增加：

```text
R-only
U-only
R × U
Soft-RU
Amplitude × U
更多 selective thresholds
更多复杂 interaction weighting
```

这些实验保留作为历史分析或补充材料即可。

同样不建议继续大量增加新的 KD baseline。现有：

```text
Full KD
Projector Feature KD
EA-KD
CMAD-style CAFD
Subset-7 KD
```

已经覆盖：

```text
output KD
feature KD
adaptive KD
correlation-aware feature KD
subset-level KD
```

---

# 14. 最终正文建议表图结构

## Tables

### Table 1

**Overall comparison on CMU-MOSEI**

```text
Adapted Student
Full KD
Projector
EA-KD
CMAD-style CAFD
Subset-7
Uniform Interaction
```

指标：

```text
MAE
Pearson
Acc-2
F1
Acc-7
```

---

### Table 2

**Overall comparison on CMU-MOSI**

与 Table 1 相同。

---

### Table 3

**Controlled distillation ablation**

```text
Adapted Student
Full KD
Ensemble Full KD
Subset-7 KD
First-order Interaction
First+Second-order Interaction
Uniform Interaction
```

重点：

```text
supervision type
interaction order
MAE
```

---

### Table 4

**Coordinate-space comparison**

```text
Subset-7
Random Orthogonal Basis
Möbius Interaction
```

---

### Table 5

**Efficiency comparison**

```text
Parameters
Trainable parameters
Peak memory
Training time
Inference latency
```

---

# 15. 最终正文建议 Figures

### Figure 1

Method Framework
已有。

### Figure 2

**Teacher--Student Interaction Reconstruction Error**

```text
First-order
Second-order
Third-order
```

### Figure 3

**Performance under Different Interaction Strengths**

```text
Low
Medium
High
```

### Figure 4

**Training Dynamics**

```text
Epoch vs MAE
MOSEI / MOSI
```

Figure 4 可根据版面移入补充材料。

---

# 16. 推荐执行顺序

## 第一阶段：无需训练，立即完成

```text
1. Interaction Reconstruction Analysis
2. Interaction-Strength Stratified Analysis
3. Efficiency Table
4. Teacher Subset Table
5. Teacher Interaction Statistics
6. Epoch-wise Training Curves
```

这些主要依赖已有 checkpoint、teacher predictions 和逐 epoch 测试结果。

---

## 第二阶段：新增训练实验

优先：

```text
7. First + Second-order Interaction
8. Random Orthogonal Basis
```

建议先在 MOSEI seed13 上完成。

如果结果有明显解释价值，再同步到 MOSI。

---

## 第三阶段：论文完善

```text
9. Interaction Case Studies
10. Hyperparameter Sensitivity（可选）
```

---

# 17. 最终最低完成标准

如果计算资源有限，至少补齐以下四项：

```text
✓ Interaction Reconstruction Analysis
✓ Interaction-Strength Stratified Analysis
✓ First + Second-order Interaction
✓ Random Orthogonal Basis
```

其中前两项属于分析实验，通常无需重新训练；后两项需要新增训练。

完成这四项后，实验部分即可形成：

$$
\text{Overall Performance}
\rightarrow
\text{Controlled Ablation}
\rightarrow
\text{Interaction Order}
\rightarrow
\text{Coordinate-Space Control}
\rightarrow
\text{Mechanism Analysis}
\rightarrow
\text{Efficiency}
$$

的完整证据链。

---

# 18. 审阅与执行记录（2026-09-22）

## 18.1 审阅结论

1. 方案的最低完成集合合理：交互重建、交互强度分层、一+二阶交互和随机正交基对照可形成闭环。
2. 交互重建与交互强度分层在 official validation split 上完成。原因是现有冻结 teacher interaction target 覆盖 train/valid，而重新产生 test 上全部七子集 teacher target 需要 32,809 次多模态 teacher 任务，不属于“无需重新训练”的快速分析。
3. Random Orthogonal Basis 使用坐标级 SmoothL1，损失对正交旋转并不严格不变。因此该对照检验的是“Möbius 坐标系+坐标级鲁棒损失”的联合作用，不应单独归因为交互基的唯一性。
4. seed 13 对每个已完成 epoch checkpoint 评估 Test MAE，test-best 结果仅作诊断口径单独保留。Aggregate interaction reconstruction 与交互强度分层统一按 official-validation MAE 选 checkpoint，不使用 test 误差选轮。

## 18.2 已完成

- 已实现 `first_second_order_interaction` 与 `random_orthogonal` 训练目标，后者使用固定 `coordinate_seed=20260922` 生成可复现正交矩阵。
- 已生成 MOSEI/MOSI official validation 的 Teacher Subset / Interaction Statistics，输出位于 `project/reports/interaction_evidence_v1/`。
- 已将正文 Fig. 3 调整为 Adapted Student、Full KD、Subset-7 和 Uniform
  四种代表方法的 official-validation MAE 轨迹（epochs 1--20）；不显示
  best-test 文本框或最优星号，MOSI 使用 inset 放大 epochs 10--20。逐 epoch
  Test MAE 轨迹保留为补充图，并同样移除最优 test checkpoint 的视觉强调。
- 已有 Fig. 4 效率/部署证据；interaction supervision 只影响训练，部署时使用同一 student 前向图。
- 全部自动化测试通过：174 passed。

## 18.3 已完成的训练、测评与自动收尾

- MOSEI `first_second_order_interaction` 已完成 20/20 epoch 训练与 20/20 个 checkpoint 测评。
- MOSEI `random_orthogonal` 已完成 15/15 epoch 训练与 15/15 个 checkpoint 测评。
- 自动收尾已完成。Aggregate interaction evidence 对六种方法统一按 validation MAE 选 checkpoint：Full KD e6、Subset-7 e9、First-order e14、First+Second e14、Random Orthogonal e8、Uniform e6。
- 旧 test-selected finalizer 入口已改为转发到 validation-selected finalizer，避免再次覆盖正式报告。

主监督状态：

```text
outputs/experiments/interaction_evidence_v1/mosei/status.json
```

最终收尾状态：

```text
project/reports/interaction_evidence_v1/finalizer_status.json
```

最终分析报告：

```text
project/reports/interaction_evidence_v1/mosei_interaction_evidence_valid.{json,md}
```

## 18.4 MOSI 对称对照与增量测评（2026-09-23）

- MOSI `first_second_order_interaction` 与 `random_orthogonal`（seed 13）均已完成 20/20 epoch 训练和 20/20 checkpoint 测评。
- 测评改为 checkpoint 生成即入队，不再等待整次训练结束。训练并行期间先设置
  两个 evaluation workers；MOSI 两项训练跑满 20 epochs 并释放显存后，扩展为
  MOSI 测评阶段使用八个并行 workers，每个方法两个 evaluator，按奇偶 epoch
  分片。MOSI 完成并释放显存后，MOSEI 两个方法各扩展为四个 evaluator，按
  epoch 模 4 分片；共享汇总锁保证结果不重复且原子更新。
- live evaluator 与原最终 evaluator 共用输出锁；最终收尾只补缺失 epoch，已经
  完成的 checkpoint 不重复测评。
- 本轮 GPU 训练与测评队列已全部结束，当前无活跃 evaluator。

队列状态：

```text
outputs/experiments/interaction_evidence_v1/live_evaluation_queue_status.json
```

MOSI 训练与逐 epoch 测评状态：

```text
outputs/experiments/interaction_evidence_v1/mosi/students/*/status.json
outputs/experiments/interaction_evidence_v1/mosi/test_epoch_sweeps/*/live_status.json
outputs/experiments/interaction_evidence_v1/mosi/test_epoch_sweeps/*/summary.json
```

MOSI 自动收尾状态与最终输出：

```text
project/reports/interaction_evidence_v1/mosi_finalizer_status.json
project/reports/interaction_evidence_v1/mosi_interaction_evidence_valid.{json,md}
project/reports/paper_figures/fig_aggregate_interaction_reconstruction_mosi.pdf
```
