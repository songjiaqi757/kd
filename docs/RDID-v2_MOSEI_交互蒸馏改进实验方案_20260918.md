# RDID-v2：基于 MOSEI Official-Test 反馈的交互蒸馏改进方案

> 版本：2026-09-18
> 适用项目：`songjiaqi757/kd`
> 当前开发数据集：CMU-MOSEI
> 核心约束：**后续新增训练全部只使用 `seed=13`；每个完整 epoch 都保存独立、可恢复 checkpoint。**
> 本方案允许使用 MOSEI official-test 结果指导方法改进，因此 **MOSEI 从最终盲测集调整为 development benchmark**。后续真正独立的最终确认应由 MOSI official-test 或新的未参与方法开发的数据集承担。

---

## 1. 背景与当前问题

当前冻结 P0 在 MOSEI 上得到：

| 方法 | Valid MAE ↓ | Official-Test MAE ↓ |
|---|---:|---:|
| M3：Adapted Full KD | 0.467097 ± 0.003826 | 0.479825 ± 0.001906 |
| M4：Uniform Interaction | 0.469803 ± 0.005565 | **0.476242 ± 0.004540** |
| M6：R × U Interaction | **0.463968 ± 0.001598** | 0.481887 ± 0.000657 |

核心现象：

```text
Valid:  M6 < M3 < M4
Test:   M4 < M3 < M6
```

其中 MAE 越低越好。

这说明当前需要优先解决的问题并不是“interaction distillation 是否有效”，而是：

> **为什么在 Uniform Interaction 的基础上加入 R × U 选择性加权后，validation 变好，但 official-test 泛化反而变差？**

因此后续开发目标调整为：

1. 保留并继续验证 interaction-space distillation；
2. 诊断当前 `R × U` 是否存在过强选择、权重尖锐化或错误 utility/reliability 归因；
3. 优先尝试 **Soft Selective Interaction**，即在 Uniform Interaction 和原始 R×U 之间连续插值；
4. 只有在证据支持时，再增加 sample-level interaction gate；
5. 暂缓完整主表和大规模外部 baseline，直到最终 Ours 定义稳定。

---

# 2. 后续统一训练协议

## 2.1 种子

从本方案开始，所有新增实验统一：

```text
seed = 13
```

不再自动补：

```text
42
2026
```

现有历史三 seed 结果保留用于背景和稳定性参考，但 **RDID-v2 所有新增开发实验均只训练 seed13**。

---

## 2.2 Checkpoint 保存策略

以后任何正式训练都必须保存：

```text
epoch_001.pt
epoch_002.pt
epoch_003.pt
...
epoch_NNN.pt
```

而不是仅保存：

```text
best.pt
last.pt
```

### 每个 epoch checkpoint 必须是完整可恢复 checkpoint

每个 `epoch_XXX.pt` 至少包含：

```text
model_state_dict
optimizer_state_dict
lr_scheduler_state_dict（如使用）
grad_scaler_state_dict（如使用 AMP scaler）
epoch
global_step
best_valid_mae_so_far
best_epoch_so_far
stale_epochs
training_history_so_far
run_config
method_config
random_seed
Python RNG state
NumPy RNG state
Torch CPU RNG state
Torch CUDA RNG state
```

如果存在：

```text
LoRA adapter state
trainable projector state
额外 loss module state
```

也必须包含在 checkpoint 中。

### `best.pt` 与 `last.pt` 的定位

仍然可以保留：

```text
best.pt
last.pt
```

但它们只作为便利别名：

- `best.pt`：指向 / 复制 valid MAE 最佳 epoch；
- `last.pt`：指向 / 复制最后完成 epoch；
- **不能替代 `epoch_XXX.pt` 全量保存。**

### Checkpoint 保存时机

每个 epoch 完成后按以下顺序执行：

```text
1. 完成本 epoch train
2. 完成 official-valid evaluation
3. 写入 history
4. 保存 epoch_XXX.pt
5. 如果该 epoch 是当前最佳，再更新 best.pt
6. 更新 last.pt
7. 原子写入 status.json
```

避免出现：

> 指标已经写盘，但 epoch checkpoint 丢失。

---

## 2.3 Checkpoint 审计

每个训练 run 结束后生成：

```text
checkpoint_inventory.json
```

至少记录：

```json
{
  "epoch": 1,
  "path": ".../checkpoints/epoch_001.pt",
  "sha256": "...",
  "bytes": 0,
  "valid_mae": 0.0,
  "is_best_epoch": false,
  "global_step": 0
}
```

并检查：

```text
history 中有多少完整 epoch
==
checkpoints/ 中有多少 epoch_XXX.pt
```

否则 run 不允许标记为 `complete`。

---

# 3. RDID-v2 核心假设

当前 M4：

\[
W_{ik}=1
\]

当前 M6：

\[
Q_{ik}
=
\frac{R_{ik}U_k}
{\frac{1}{7}\sum_jR_{ij}U_j}
\]

其中：

\[
R_{ik}
=
\frac{|\mu_{ik}|}
{\sqrt{\sigma^2_{ik}+\epsilon}}
\]

而：

\[
U_k
=
\left|
\mathrm{Pearson}
(
I_k,
y_{\text{train}}
)
\right|
\]

当前问题可能来自：

> `R × U` 将原本稳定的七坐标均匀监督变成了过于尖锐的坐标选择。

因此 RDID-v2 第一阶段不重新设计 teacher/student，而只修改：

> **选择性权重的强度。**

---

# 4. Phase A：零训练成本诊断

在新增训练前，先完成以下分析。

---

## A1. R×U 权重尖锐度分析

对全部 MOSEI train 样本计算七维权重：

\[
Q_i=[q_{i1},...,q_{i7}]
\]

归一化为概率：

\[
p_{ik}
=
\frac{q_{ik}}
{\sum_jq_{ij}}
\]

计算：

### Entropy

\[
H_i
=
-\sum_kp_{ik}\log p_{ik}
\]

### Effective number of coordinates

\[
N_{\mathrm{eff},i}
=
\frac{1}
{\sum_kp_{ik}^2}
\]

范围：

```text
1 ≤ Neff ≤ 7
```

若大量样本：

```text
Neff ≈ 1~3
```

则说明当前 R×U 确实把七坐标监督压缩成非常强的少数坐标选择。

需要报告：

```text
mean / median / p10 / p25 / p75 / p90
```

并按：

```text
train
valid
```

分别统计。

---

## A2. R 的组成诊断

分别计算：

```text
R
abs(mean)
1 / sqrt(variance + eps)
```

分析：

```text
corr(R, abs(mean))
corr(R, inverse_std)
```

同时统计：

```text
R clip 到 0.25 的比例
R clip 到 4.0 的比例
```

如果：

```text
corr(R, abs(mean)) >> corr(R, inverse_std)
```

说明当前所谓 Reliability 主要实际上由 interaction amplitude 决定。

---

## A3. 七坐标的权重分布

对：

```text
T
A
V
TA
TV
AV
TAV
```

分别统计：

```text
mean R
mean U
mean R×U
median R×U
P90 R×U
被选为最大权重的频率
```

重点检查是否存在：

> 某一两个 interaction coordinate 长期支配训练。

---

## A4. MOSEI Test Utility 分析

既然本方案允许使用 MOSEI official-test 开发方法，则新增一套 **分析专用** teacher subset extraction：

```text
official-test × 7 subsets
```

为每个 test utterance 得到三个 Probe 的 interaction mean。

分别计算：

\[
U^{train}
\]

\[
U^{valid}
\]

\[
U^{test}
\]

比较：

```text
Pearson(U_train, U_valid)
Pearson(U_train, U_test)
Spearman(U_train, U_valid)
Spearman(U_train, U_test)
```

并列出七坐标具体值。

目的：

> 判断全局 train utility 是否能够稳定迁移到 valid/test。

注意：

该分析从现在开始使 MOSEI test 正式属于 development information。

---

# 5. Phase B：Soft-RU 核心实验

## 5.1 新权重定义

定义：

\[
Q_{ik}
=
\frac{R_{ik}U_k}
{\frac{1}{7}\sum_jR_{ij}U_j}
\]

Soft-RU：

\[
\boxed{
W_{ik}^{(\alpha)}
=
(1-\alpha)
+
\alpha Q_{ik}
}
\]

其中：

\[
0\le\alpha\le1
\]

由于：

\[
mean_k(Q_{ik})=1
\]

因此：

\[
mean_k(W_{ik}^{(\alpha)})=1
\]

不会因为改变 α 而改变 interaction loss 的平均尺度。

---

## 5.2 两个已有端点

### α = 0

\[
W=1
\]

等价于：

```text
M4 / Uniform Interaction
```

### α = 1

\[
W=Q
\]

等价于：

```text
M6 / Original R×U
```

---

## 5.3 新增三个候选

只训练：

```text
soft_ru_alpha025_seed13
soft_ru_alpha050_seed13
soft_ru_alpha075_seed13
```

即：

\[
\alpha\in\{0.25,0.50,0.75\}
\]

全部：

```text
seed = 13
```

不跑其他 seed。

---

## 5.4 其他训练条件完全固定

与当前 M4/M6 保持一致：

```text
student backbone: same M3-capacity student
text: Qwen3-0.6B-Base
audio: WavLM-Base-Plus
video: VideoMAE-Base
T/A/V LoRA scope: unchanged
raw T/A/V online input
batch size = 8
learning rate = 1e-4
weight decay = 0.01
max epochs = 30
patience = 7
lambda_task = unchanged
lambda_full = 1.0
lambda_interaction = 1.0
KD temperature = 2.0
teacher targets = unchanged
interaction targets = same 3-Probe mean
checkpoint selection = valid MAE
seed = 13
```

唯一变化：

```text
alpha
```

---

# 6. Phase C：R/U 机制拆解

与 Soft-RU 同一阶段或紧随其后运行。

全部：

```text
seed = 13
```

---

## C1. R-only

\[
W=Normalize(R)
\]

实验名：

```text
r_only_interaction_seed13
```

回答：

> Probe-consistency / amplitude 混合 reliability 本身是否有价值？

---

## C2. U-only

\[
W=Normalize(U)
\]

实验名：

```text
u_only_interaction_seed13
```

回答：

> train-level task association utility 本身是否能够泛化？

---

## C3. Amplitude × U

\[
W=Normalize(|\mu|U)
\]

实验名：

```text
amplitude_u_interaction_seed13
```

回答：

> 当前 R×U 的作用是否主要来自 interaction amplitude，而不是真正的 Probe consistency？

---

## C4. 暂缓项

第一轮先不跑：

```text
R + U
coarse-U
shuffled-R
first-order-only
new learned weighting network
```

等前 6 个实验得到结果后再决定。

---

# 7. Phase D：MOSEI Development Selection

由于 MOSEI test 已参与方法开发，不再只按 valid MAE 选 RDID-v2。

建议同时记录：

\[
MAE_{valid}
\]

\[
MAE_{test}
\]

以及：

\[
GeneralizationGap
=
MAE_{test}-MAE_{valid}
\]

---

## 7.1 主开发指标

建议使用：

\[
\boxed{
S=
\max
(
MAE_{valid},
MAE_{test}
)
}
\]

越低越好。

它避免再次出现：

> validation 极好，但 test 明显退化。

---

## 7.2 辅助判断

同时检查：

```text
Pearson
Acc-2
Weighted F1
Acc-7
Test-Valid gap
```

不能只看一个 test MAE 小数点差异。

---

## 7.3 Phase B/C 完成后的判断树

### 情况 A：Soft-RU 中间 α 最好

例如：

```text
α=0.25 或 0.50 最优
```

则说明：

> R/U 中存在有效信息，但原始 M6 选择强度过大。

此时进入：

```text
RDID-v2 = Soft Selective Interaction KD
```

然后考虑 Phase E 的 sample-level gate。

---

### 情况 B：α=0 始终最好

即：

```text
Uniform Interaction > 所有 Soft-RU / R-only / U-only
```

则停止继续“救” R×U。

主方法改为：

```text
Uniform Interaction Distillation
```

R/U 作为机制分析或负结果保留。

---

### 情况 C：R-only 好，U-only 差

说明：

```text
U 是主要问题
```

后续改进 Utility。

优先尝试：

### Cross-Fold Robust Utility

在 official train 内按 source-video 划分 K folds。

每个 fold：

\[
u_k^{(j)}
=
|
Corr(I_k,y)
|
\]

最终：

\[
\tilde U_k
=
median_j(u_k^{(j)})
\]

再 mean normalize。

避免由一次全局 Pearson 决定 U。

---

### 情况 D：U-only 好，R-only 差

说明：

```text
R 定义需要修改
```

重点比较：

```text
R
abs(mean)
inverse_std
```

可尝试 bounded reliability：

\[
R^\text{bounded}
=
\frac{\mu^2}
{\mu^2+\sigma^2+\epsilon}
\]

范围天然：

\[
0\le R^\text{bounded}\le1
\]

避免原始 SNR 在 variance 很小时产生极端值。

---

### 情况 E：R-only、U-only 都好，但 R×U 差

说明：

> 乘法组合导致过度集中。

此时优先：

```text
Soft-RU
```

其次：

```text
R + U
```

不优先继续增强乘法。

---

# 8. Phase E：Sample-Level Interaction Gate（仅在 Soft-RU 有希望时启动）

如果 Soft-RU 相比 Uniform 出现稳定改善，再增加第二层选择：

> 判断“这个样本整体上是否值得进行 interaction distillation”。

---

## 8.1 Interaction Strength

定义高阶 interaction strength：

\[
E_i
=
\frac{1}{4}
(
|I_{TA}|
+
|I_{TV}|
+
|I_{AV}|
+
|I_{TAV}|
)
\]

---

## 8.2 Gate

第一版使用非学习式 gate：

\[
g_i
=
clip
\left(
\frac{E_i}
{median(E_{train})},
0,
1
\right)
\]

---

## 8.3 最终 loss

\[
\boxed{
L=
L_{task}
+
L_{FullKD}
+
\lambda_{interaction}
g_i
\frac{1}{7}
\sum_k
W_{ik}^{(\alpha)}
L_{ik}
}
\]

其中：

```text
sample-level gate g_i
```

回答：

> 这个样本是否值得蒸馏 interaction？

而：

```text
coordinate weight W_ik
```

回答：

> 如果值得蒸馏，七个 interaction 哪些更重要？

---

## 8.4 Gate 第一轮只跑一个版本

只运行：

```text
best_soft_ru + interaction_strength_gate
seed13
```

不要一次设计多个 gate。

---

# 9. Phase F：确定最终 Ours

完成 Phase A-E 后，在 MOSEI development 上确定最终方法定义。

最终候选只能从以下三类中选择：

```text
1. Uniform Interaction
2. Soft-RU
3. Soft-RU + Sample-Level Gate
```

选择后：

> **冻结公式，不再继续根据 MOSEI test 修改。**

---

# 10. MOSI 的角色

由于 MOSEI official-test 已参与方法开发：

> MOSEI 不能再作为完全独立的最终 confirmatory benchmark。

因此最终方法冻结后：

## MOSI train/valid

用于：

```text
训练
checkpoint selection
必要的 dataset-specific teacher utility construction
```

## MOSI official-test

必须保持：

```text
直到最终 RDID-v2 方法完全冻结之后才评测
```

MOSI official-test 将成为：

\[
\boxed{
\text{主要独立泛化证据}
}
\]

---

# 11. 暂缓完整 Main Table

当前先暂停：

```text
Projector KD
EA-KD
CMAD-style CAFD
RLD
SKD
DLF
GsiT
DPDF-LQ
```

原因：

> Ours 本身仍在开发阶段。

只有最终 RDID-v2 冻结后，才重新启动：

```text
main_table_v2
```

避免在方法定义持续变化时浪费大量 GPU 预算。

---

# 12. 第一阶段新增训练清单

全部只跑：

```text
seed13
```

| ID | 方法 | 新训练 |
|---|---|---:|
| B1 | Soft-RU α=0.25 | 1 |
| B2 | Soft-RU α=0.50 | 1 |
| B3 | Soft-RU α=0.75 | 1 |
| C1 | R-only | 1 |
| C2 | U-only | 1 |
| C3 | Amplitude × U | 1 |

合计：

\[
\boxed{6\text{ runs}}
\]

已有 M4 和 M6 的 seed13 结果作为两个端点参考，不重复训练，除非需要统一新的 checkpoint-retention 协议做 trajectory analysis。

---

# 13. 如果需要统一训练轨迹，M4/M6 可重新跑 seed13

为了让所有候选都有：

```text
epoch_001.pt
epoch_002.pt
...
```

可以重新训练：

```text
Uniform Interaction seed13
Original R×U seed13
```

但这两次属于：

```text
trajectory reproduction
```

不是新增方法。

如果当前正在运行的 `tav_epoch_checkpoints_v1` 已经覆盖 M4/M6，则直接复用，不再重复。

---

# 14. 推荐输出目录

```text
outputs/experiments/rdid_v2_mosei/
├── diagnostics/
│   ├── weight_sharpness.json
│   ├── reliability_decomposition.json
│   ├── utility_split_stability.json
│   └── test_teacher_interactions/
│
├── students/
│   ├── soft_ru_a025_seed13/
│   │   ├── checkpoints/
│   │   │   ├── epoch_001.pt
│   │   │   ├── epoch_002.pt
│   │   │   └── ...
│   │   ├── best.pt
│   │   ├── last.pt
│   │   ├── checkpoint_inventory.json
│   │   ├── history.json
│   │   ├── run_config.json
│   │   ├── report.json
│   │   └── status.json
│   │
│   ├── soft_ru_a050_seed13/
│   ├── soft_ru_a075_seed13/
│   ├── r_only_seed13/
│   ├── u_only_seed13/
│   └── amplitude_u_seed13/
│
├── development_summary.json
└── development_summary.md
```

---

# 15. 每个 run 必须记录的指标

每个 epoch：

```text
train total loss
train task loss
train full KD regression loss
train full KD KL loss
train interaction loss

valid MAE
valid Pearson
valid Acc-2
valid Weighted F1
valid Acc-7

learning rate
epoch wall time
peak GPU memory
```

interaction 方法额外记录：

```text
mean interaction weight
weight std
weight min/max
weight entropy
effective coordinate count
```

Soft-RU 还必须记录：

```text
alpha
```

---

# 16. 每个 run 完成后的分析

至少输出：

### Validation trajectory

```text
epoch vs valid MAE
```

### Training trajectory

```text
epoch vs task loss
epoch vs full KD loss
epoch vs interaction loss
```

### Generalization

如果该候选进入 MOSEI development test：

```text
best-valid epoch
valid metrics
test metrics
test-valid gap
```

---

# 17. 不允许的操作

后续开发阶段禁止：

```text
因为某个 epoch 的 test 更好而选该 epoch
```

即使 MOSEI test 已用于方法开发，checkpoint 仍统一：

```text
按 valid MAE 选择
```

MOSEI test 只用于：

```text
方法级开发
```

不能用于：

```text
epoch-level checkpoint selection
```

否则会把 test 直接变成第二 validation set。

同样禁止：

```text
逐 epoch 跑 MOSEI official-test 然后选 epoch
```

---

# 18. 当前最重要的研究问题

后续实验依次回答：

## Q1

```text
R×U 是否因为选择过强而损害泛化？
```

对应：

```text
Soft-RU α curve
```

---

## Q2

```text
R 和 U 哪一个真正有用？
```

对应：

```text
R-only
U-only
```

---

## Q3

```text
所谓 reliability 的收益是不是其实来自 interaction amplitude？
```

对应：

```text
Amplitude × U
```

---

## Q4

```text
是否所有样本都应该接受 interaction distillation？
```

对应：

```text
Sample-Level Interaction Gate
```

---

# 19. 最终可能形成的论文方法

如果 Soft-RU + Gate 成立：

\[
\boxed{
\text{RDID-v2:
Reliability/Utility-guided Soft Interaction Distillation}
}
\]

核心包含：

```text
1. 七子集教师预测
2. Möbius interaction decomposition
3. Uniform interaction structural prior
4. Soft selective coordinate reweighting
5. Sample-level interaction transfer gate
```

推理阶段仍然：

```text
只输入完整 T/A/V
不使用教师
不使用三个 Probe
不运行七子集 ensemble
```

因此额外机制全部只发生在训练阶段。

---

# 20. 立即执行顺序

建议严格按以下顺序：

```text
Step 1
完成当前 seed13 全 epoch checkpoint 复训和 checkpoint 审计

Step 2
完成 R×U weight sharpness / R decomposition 分析

Step 3
为 MOSEI official-test 构建分析专用七子集 teacher interaction targets

Step 4
计算 U_train / U_valid / U_test 稳定性

Step 5
训练 Soft-RU:
α = 0.25
α = 0.50
α = 0.75

Step 6
训练:
R-only
U-only
Amplitude × U

Step 7
统一跑 MOSEI development evaluation

Step 8
根据结果决定：
Uniform
Soft-RU
或 Soft-RU + Gate

Step 9
若需要，额外训练一个 Sample-Level Gate 版本

Step 10
冻结 RDID-v2

Step 11
转入 MOSI 独立验证

Step 12
最终方法冻结后再启动完整 external KD / native MSA main table
```

---

# 21. 本阶段最终判定原则

这一阶段不要追求：

> “一定把 R×U 救回来。”

真正目标是：

\[
\boxed{
\text{找到 interaction distillation 中真正稳定、可泛化的选择机制}
}
\]

如果最终结果证明：

```text
Uniform Interaction
```

仍然最好，那么就正式停止 R/U 主线。

如果：

```text
Soft-RU
```

最好，则证明：

> Reliability/Utility 信息有价值，但必须作为对 Uniform structural supervision 的软偏置，而不能完全取代均匀监督。

这是当前最值得优先验证的假设。
