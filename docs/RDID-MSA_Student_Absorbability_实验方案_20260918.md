# RDID-MSA 后续实验方案：Student-Aware Selective Interaction Distillation

> 版本：2026-09-18
> 适用项目：`https://github.com/songjiaqi757/kd`
> 当前原则：**先验证 Student Absorbability 是否真实存在并能解释蒸馏收益，再冻结最终 Ours；在方法冻结前暂停大规模主表铺开。**
> 训练约束：**方法探索阶段统一只跑 seed=13；保存每一个 epoch checkpoint，不只保存 best/last。**

---

## 0. 方案目的

当前项目已经证明：

1. 大模型教师提供的多模态知识具有明显异质性；
2. 普通 Full KD 可以优于 Student-only；
3. 给轻量学生增加适配能力（LoRA）带来的收益非常显著；
4. Uniform Interaction 并不会稳定优于 Adapted Full KD；
5. Reliability × Utility（R×U）在 valid 上只有较小收益，且当前 P0 official test 不支持其稳定优于 Uniform 或 Adapted Full KD；
6. 因此，**“教师知识是否可靠 / 是否与任务相关”并不足以决定该知识是否适合当前学生学习。**

后续实验不再默认 `R×U Interaction = Ours`，而是检验新的核心假设：

> **真正决定轻量学生是否从异质教师知识中获益的，不仅是 teacher-side knowledge quality，还包括 student-side absorbability / transferability。**

目标是将论文主线从：

```text
Teacher knowledge
→ interaction decomposition
→ Reliability × Utility
→ distillation
```

升级为：

```text
Teacher knowledge
→ interaction decomposition
→ teacher reliability
→ task relevance
→ student absorbability
→ selective distillation
```

---

# 1. 当前结果冻结与使用边界

## 1.1 P0 结果全部冻结

现有 P0：

- M0：Frozen TAV Student-only
- M1：Frozen TAV Full KD
- M3：Adapted Full KD
- M4：Uniform Interaction
- M6：Reliability × Utility Interaction

以及：

- 三 seed valid 结果；
- clustered paired bootstrap；
- 已完成的 P0 MOSEI official-test；
- hidden KD / polarity KD / video 等历史失败或探索结果；

全部作为 **motivation / diagnosis evidence** 冻结。

不得：

- 根据 P0 official-test 结果重新修改 P0 配置；
- 因 M4 test 最好而回头选择 M4 超参；
- 因 M6 test 不理想而在同一 P0 协议上反复调 R/U；
- 将 P0 official-test 用作新方法的开发集。

## 1.2 新方法开发仍只允许使用 train / valid

新的 Absorbability 方法：

- 只使用 MOSEI official train / valid；
- 只用 train 构造任何可能涉及标签的统计量；
- valid 只用于 checkpoint 选择和方法筛选；
- **新方法 official-test 在最终方法冻结前保持封存。**

P0 已经使用过 test，不影响新协议单独执行严格的 test-lock，但论文中必须明确区分：

```text
P0 diagnostic test
vs.
Final method evaluation test
```

不得混称。

---

# 2. 核心研究问题

后续实验只围绕四个问题展开。

## RQ1：是否真的存在 Student Absorbability？

对于同一个 teacher interaction：

- 有些监督是否能够帮助学生主任务；
- 有些监督是否虽然能降低 interaction reconstruction error，却不能降低 task error；
- 这种差异是否与 student-side optimization / learning state 有系统关系。

如果答案是否定的，则停止 Student Absorbability 主线，不继续增加复杂模块。

---

## RQ2：Absorbability 能否被训练时可见信号预测？

至少比较三类候选：

1. **Gradient Compatibility**
2. **Learning Progress**
3. **Student–Teacher Residual / Difficulty**

目标不是立即追求最高 MAE，而是先判断：

> 哪类 student-side signal 与“该 interaction 最终是否真正帮助任务”关系最强。

---

## RQ3：Absorbability 是否提供独立于 R 和 U 的信息？

必须证明：

```text
A ≠ R
A ≠ U
```

并且：

```text
R × U × A
```

相对于：

```text
R × U
```

存在独立收益。

如果 A 只是在重复 Reliability 或 Utility，则不能作为核心创新。

---

## RQ4：最终方法能否在统一公平设置下稳定超过强基线？

最终冻结方法至少需要比较：

- Adapted Student-only
- Adapted Full KD
- Uniform Interaction
- R-only
- U-only
- R×U
- A-only
- R×A
- U×A
- R×U×A

并在最终阶段加入外部 KD：

- Projector-based KD
- EA-KD
- CMAD-style CAFD
- Subset-7 KD
- 必要时 RLD / SKD

---

# 3. 总体执行顺序

后续实验分成六个阶段。

```text
Phase A  Absorbability 诊断
    ↓
Phase B  A proxy 筛选
    ↓
Phase C  新 Ours 小规模验证
    ↓
Phase D  方法冻结 + 完整消融
    ↓
Phase E  外部主表 + MOSI + 鲁棒性 + 效率
    ↓
Phase F  新方法一次性 official-test
```

在 Phase C 完成前：

> **暂停 main_table_v1 的 MOSEI/MOSI 全量 53-run 铺开。**

已有代码、资产和 dry-run 保留，不删除。

---

# 4. Phase A：Student Absorbability 诊断

## 4.1 数据集

第一阶段只使用：

```text
CMU-MOSEI
```

原因：

- 当前证据最完整；
- P0 与 Interaction 资产已经齐全；
- 可以快速验证核心假设；
- 避免在方法尚未确定时同时消耗 MOSI 计算资源。

---

## 4.2 固定模型

固定使用：

```text
Student backbone:
- Qwen3-0.6B
- WavLM-Base-Plus
- VideoMAE-Base

Adaptation:
- 与当前 M3/M4/M6 相同的 T/A/V LoRA 范围

Teacher:
- 当前冻结 Qwen3-Omni 教师
- 三 Probe interaction targets
```

第一阶段禁止更换学生 backbone。

否则无法区分：

```text
Absorbability effect
vs.
student architecture effect
```

---

# 5. Absorbability 的定义候选

不直接假定最终公式。

第一轮至少比较以下三类。

---

## 5.1 A1：Gradient Compatibility

对于第 `k` 个 interaction：

```text
L_task
L_int,k
```

定义：

\[
A^{grad}_{k}
=
\cos(
\nabla_{\theta_s}L_{task},
\nabla_{\theta_s}L_{int,k}
)
\]

其中只在预先指定的一小组 student 参数上计算，例如：

```text
fusion layer
+ regression head
+ classification head
```

不建议第一版直接对所有 backbone 参数计算，避免：

- 显存过高；
- 噪声过大；
- 计算成本不可控。

解释：

```text
A > 0:
interaction supervision 与任务优化方向一致

A ≈ 0:
interaction 对当前 task optimization 作用弱

A < 0:
interaction supervision 与 task objective 冲突
```

第一版推荐映射：

```text
A_grad = clamp((cos + 1) / 2, 0, 1)
```

或：

```text
A_grad = ReLU(cos)
```

两者只允许在 train/valid 上比较一次，不能无限搜索。

---

## 5.2 A2：Learning Progress

考察某个 interaction 是否是学生当前“可学”的。

定义一种简单形式：

\[
LP_{k}^{(t)}
=
E_{k}^{(t-\Delta)}
-
E_{k}^{(t)}
\]

其中：

```text
E_k = student 与 teacher interaction 的 SmoothL1 error
```

若：

```text
LP > 0
```

说明最近训练中学生确实能够吸收该 interaction。

可以进一步归一化：

\[
A^{LP}_{k}
=
\sigma(
LP_k / (\bar{|LP|}+\epsilon)
)
\]

第一版不引入额外 MLP。

---

## 5.3 A3：Student–Teacher Residual / Difficulty

定义：

\[
D_k
=
|I^S_k - I^T_k|
\]

最简单的 absorbability proxy 可以是：

```text
过小：
已经学会，继续蒸馏价值有限

适中：
可能是最可学、最有效区域

过大：
可能超出学生当前能力
```

因此不能简单使用：

```text
A = 1 / D
```

建议诊断阶段先分析：

```text
D_k 与后续 task gain 的关系
```

确认是否存在：

```text
easy / medium / too-hard
```

结构，再决定映射。

---

# 6. Phase A 具体实验

所有 Phase A 实验：

```text
seed = 13
```

必须保存：

```text
checkpoint_epoch_001.pt
checkpoint_epoch_002.pt
...
checkpoint_epoch_N.pt
best.pt
last.pt
```

同时每个 epoch 保存：

```text
train predictions
valid predictions
interaction reconstruction error
task metrics
R
U
A candidate values
gradient compatibility statistics
```

---

## 6.1 A0：Adapted Full KD trajectory

对应当前 M3。

目的：

> 获取没有 interaction supervision 时的学生训练轨迹。

保存每 epoch：

- valid MAE
- Pearson
- Acc-2
- F1
- Acc-7
- 7 interaction 的 student prediction
- teacher interaction reconstruction gap

---

## 6.2 A1：Uniform Interaction trajectory

对应当前 M4。

目的：

> 研究所有 interaction 被同等蒸馏时，哪些坐标真正帮助任务、哪些坐标可能产生冲突。

必须保存：

```text
per-coordinate interaction loss
per-coordinate gradient compatibility
per-coordinate student-teacher residual
```

---

## 6.3 A2：R×U trajectory

对应当前 M6。

目的：

> 研究 R/U 高权重的 interaction 是否真的具有更高 student transferability。

重点计算相关性：

```text
corr(R, actual_gain)
corr(U, actual_gain)
corr(R×U, actual_gain)
corr(A_grad, actual_gain)
corr(A_LP, actual_gain)
corr(D, actual_gain)
```

---

# 7. “Actual Transfer Gain” 的诊断定义

Absorbability 必须有一个被预测对象。

不能直接用 test。

建议在 train/valid 分析中构造：

## 7.1 Epoch-level gain

对于 epoch `t`：

\[
G_t
=
MAE_{baseline,t}
-
MAE_{candidate,t}
\]

用于整体方法比较。

---

## 7.2 Interaction-level gain

第一版使用 leave-one-coordinate-out 诊断。

对于 7 个 interaction：

```text
all interaction
minus
interaction k
```

比较 valid MAE：

\[
G_k
=
MAE_{-k}
-
MAE_{all}
\]

解释：

```text
G_k > 0:
保留 interaction k 有帮助

G_k < 0:
保留 interaction k 反而伤害任务
```

由于只有 7 个坐标，该实验成本可接受。

---

## 7.3 Oracle ranking

根据 `G_k` 得到：

```text
Oracle beneficial interaction ranking
```

注意：

- Oracle 只用于 mechanism analysis；
- 不参与 final model inference；
- 不在 official test 上构造；
- 不允许使用 test label。

然后比较：

```text
R ranking
U ranking
R×U ranking
A ranking
```

与 Oracle ranking 的：

- Spearman
- Kendall tau
- Top-k precision
- beneficial / harmful interaction AUROC

这是后续论文机制证据的重要组成部分。

---

# 8. Phase A Gate

只有满足至少以下两项，才允许进入 Student Absorbability 主线：

1. 至少一种 A proxy 与 Oracle interaction gain 的相关性明显高于 R / U；
2. A 能较稳定地区分 beneficial interaction 与 harmful interaction；
3. A 与 R/U 的相关性不是接近 1，说明提供独立信息；
4. 高 A interaction 的蒸馏比低 A interaction 更容易降低 valid task error；
5. interaction fidelity 与 task gain 的非单调关系能够被 A 更好解释。

建议 Gate：

```text
Spearman(A, Oracle Gain) >= 0.40
```

或：

```text
beneficial-vs-harmful AUROC >= 0.65
```

至少一个指标通过，并且结果方向与 qualitative case study 一致。

如果所有 A proxy 都明显失败：

> 停止 Student Absorbability 方法线，不继续堆模块。

---

# 9. Phase B：A Proxy 筛选

若 Phase A 通过，只保留最多两个 A 候选进入训练比较。

例如：

```text
B1 Gradient Compatibility
B2 Learning Progress
```

统一：

```text
seed = 13
same initialization
same batch order
same train budget
same task loss
same teacher targets
same LoRA
```

---

## 9.1 必做方法

| 编号 | 方法 | 权重 |
|---|---|---|
| B0 | Full KD | 无 interaction |
| B1 | Uniform | 1 |
| B2 | R×U | `R*U` |
| B3 | A-grad | `A_grad` |
| B4 | R×U×A-grad | `R*U*A_grad` |
| B5 | A-LP | `A_LP` |
| B6 | R×U×A-LP | `R*U*A_LP` |

第一轮禁止再增加复杂 gating network。

---

# 10. Phase B Gate

优先看：

```text
Valid MAE
```

其次：

```text
Pearson
Acc-2
F1
Acc-7
```

候选 A 进入最终方法需要：

1. `R×U×A` 优于 `R×U`；
2. `R×U×A` 优于 Full KD；
3. 相比 Uniform 不出现明显退化；
4. 至少不是仅靠某一个 epoch 偶然产生；
5. training curve 中效果可解释。

建议第一阶段效果门槛：

```text
seed13:
ΔMAE <= -0.005
```

如果达不到 0.005，但机制证据非常稳定，可以保留为候选，不立即补更多 seed。

---

# 11. Phase C：最终 Ours 小规模验证

从 Phase B 只选一个最终 A。

最终形式优先保持简单：

\[
w_{ik}
=
Normalize(
R_{ik}
\cdot
U_k
\cdot
A_{ik}
)
\]

论文暂定名称可使用：

```text
Student-Aware Selective Interaction Distillation
```

简称可以后续再定。

---

## 11.1 seed13 完整机制矩阵

| 方法 | 目的 |
|---|---|
| Adapted Student-only | 无 KD |
| Full KD | 普通 KD |
| Uniform Interaction | interaction 本身 |
| R-only | teacher reliability |
| U-only | task relevance |
| R×U | 当前方法 |
| A-only | student absorbability |
| R×A | reliability + absorbability |
| U×A | utility + absorbability |
| R×U×A | 最终 Ours |

---

## 11.2 必做控制

### shuffled-A

随机打乱样本或 coordinate 对应的 A：

```text
R × U × shuffled(A)
```

回答：

> 收益是否真的来自 absorbability information，而不是额外随机 reweighting。

---

### detached-A

如果 A 来源涉及 gradient：

```text
A.detach()
```

不允许 A 自身形成隐式二阶优化路径。

第一版必须避免二阶梯度。

---

### coarse-A

只区分：

```text
main effect
pair interaction
high-order interaction
```

或：

```text
video-related
non-video-related
```

用于证明细粒度 student-aware 信息是否必要。

---

# 12. Phase C 方法冻结条件

最终 Ours 冻结前至少满足：

```text
Ours < Full KD
Ours < R×U
```

其中 `<` 表示 MAE 更低。

并且：

- shuffled-A 无法复制收益；
- A-only / R×A / U×A 的结果能解释最终组合；
- interaction reconstruction improvement 与 task improvement 的关系得到机制支持；
- 没有发现明显数据泄漏；
- A 不使用 valid/test label；
- 推理时不需要 teacher。

满足后冻结：

```text
method formula
A definition
lambda
normalization
LoRA config
student backbone
teacher assets
early stop
checkpoint selection
```

冻结后禁止基于 final test 再修改。

---

# 13. Phase D：正式主表与完整消融

方法冻结后，再恢复 `main_table_v1`。

---

## 13.1 固定学生主表 A

建议最终主表：

| 方法 | 类型 |
|---|---|
| Adapted Student-only | no KD |
| Full KD | output KD |
| Projector-based KD | feature KD |
| EA-KD | adaptive KD |
| CMAD-style CAFD | MSA-related KD |
| Subset-7 KD | same teacher subset budget |
| Uniform Interaction | interaction control |
| R×U | previous selective baseline |
| **Ours: R×U×A** | final method |

RLD / SKD：

- 资源允许时加入；
- 或放附录扩展表；
- 不能因为结果不利而删除。

---

## 13.2 正式消融

最终消融至少包含：

```text
Full KD
Uniform
R
U
R×U
A
R×A
U×A
R×U×A
shuffled-A
coarse-A
```

如计算资源允许，再加入：

```text
amplitude interaction
first-order only
additive R+U+A
```

但这些不优先于核心机制矩阵。

---

# 14. 多 seed 策略

## 方法探索阶段

统一：

```text
seed = 13
```

原因：

- 降低方法探索成本；
- 避免方法尚未冻结时大量消耗算力；
- 便于保存全部 epoch checkpoint 做动态分析。

---

## 方法冻结后

正式主表：

```text
seed = 13 / 42 / 2026
```

要求：

- 相同初始化规则；
- 相同 batch 顺序控制；
- 相同训练预算；
- 相同 valid MAE checkpoint selection；
- 不允许仅给 Ours 增加额外调参预算。

---

# 15. Checkpoint 保留规则

从现在开始，Absorbability 相关训练必须保留：

```text
checkpoints/
  epoch_001.pt
  epoch_002.pt
  ...
  epoch_N.pt
  best.pt
  last.pt
```

每个 epoch checkpoint 至少包含：

```text
model
LoRA adapters
fusion/head
optimizer
scheduler
epoch
global_step
RNG states
best metric
full config
git commit
```

并配套：

```text
epoch_XXX_predictions.jsonl
epoch_XXX_interactions.jsonl
epoch_XXX_absorbability.jsonl
```

不得仅保留 best / last。

---

# 16. 训练日志新增字段

每个 batch / epoch 至少记录：

```text
task_loss
regression_kd_loss
classification_kd_loss
interaction_loss_total

interaction_loss_T
interaction_loss_A
interaction_loss_V
interaction_loss_TA
interaction_loss_TV
interaction_loss_AV
interaction_loss_TAV

R_k
U_k
A_k

student_teacher_residual_k
gradient_cosine_k
learning_progress_k
```

正式报告至少汇总：

```text
mean
std
median
P10
P90
negative-gradient-ratio
```

---

# 17. 防止 A 变成新的 heuristic

最终论文不能只写：

```text
we multiply another score A
```

必须至少提供以下三类证据。

## 17.1 Predictive evidence

A 能预测：

```text
which interaction will help task performance
```

---

## 17.2 Optimization evidence

高 A interaction：

- task gradient 更一致；
- 学习进度更稳定；
- 不容易产生 gradient conflict。

---

## 17.3 Intervention evidence

人为：

```text
保留高 A
删除低 A
```

优于：

```text
保留低 A
删除高 A
```

这是证明因果方向最重要的控制之一。

---

# 18. High-A / Low-A Intervention

新增一个重要实验：

## High-A Distill

只蒸馏：

```text
Top-K A interactions
```

## Low-A Distill

只蒸馏：

```text
Bottom-K A interactions
```

建议：

```text
K = 3
```

只预注册一个 K，避免 valid 上搜索。

目标：

```text
High-A > Low-A
```

且：

```text
High-A >= Random-K
```

如果成立，Student Absorbability 的解释会非常有说服力。

---

# 19. 与 DTO-KD 等 gradient-conflict 方法的边界

如果最终 A 使用 gradient compatibility，论文中必须明确：

本方法不是一般的：

```text
task loss vs KD loss gradient balancing
```

而是：

```text
task loss
vs.
individual multimodal interaction knowledge atoms
```

核心区别在于：

> **对异质、多粒度、多模态 teacher knowledge 做 coordinate-wise student transferability estimation。**

因此实验中建议增加：

```text
Global gradient compatibility baseline
```

即：

```text
只计算整个 interaction loss 与 task loss 的 cosine
```

对比：

```text
coordinate-wise A
```

如果 coordinate-wise 明显更好，能够直接支撑方法必要性。

---

# 20. Phase E：MOSI 泛化

只有最终 Ours 冻结后才启动 MOSI。

MOSI 不重新搜索大规模超参。

至少跑：

```text
Adapted Student-only
Full KD
Uniform Interaction
R×U
Ours
```

如主表预算允许，再补外部 KD。

目标不是要求与 MOSEI 完全相同的绝对提升，而是：

```text
方向一致
```

尤其检查：

```text
Ours vs R×U
Ours vs Full KD
```

---

# 21. Robustness

最终只比较：

```text
Student-only
Full KD
R×U
Ours
```

测试：

- drop T
- drop A
- drop V
- audio noise
- video corruption
- text masking / ASR-like corruption

重点不仅看最终性能，还看：

```text
A 是否会对受损 modality interaction 自动降低权重
```

如果能成立，这是 Student-Aware 方法非常强的附加证据。

---

# 22. Efficiency

报告：

```text
Total params
Trainable params
Peak VRAM
Training time
Inference latency
End-to-end latency
```

如果 A 只在训练期使用：

> 必须明确说明部署时无额外 teacher / A module 成本。

这是论文的重要优势。

---

# 23. Final Official Test

新方法在以下条件全部满足前不得测试：

1. 最终方法公式冻结；
2. A 定义冻结；
3. 所有 lambda 冻结；
4. 三 seed MOSEI valid 完成；
5. 主表完成；
6. 关键消融完成；
7. MOSI 完成；
8. checkpoint hash 完成；
9. git commit 冻结；
10. 生成：

```text
FINAL_METHOD_TEST_UNLOCK.md
```

official-test 只评估预注册方法：

```text
Adapted Student-only
Full KD
Uniform Interaction
R×U
Ours
```

不得根据 test 再：

- 调 A；
- 调 lambda；
- 改 normalization；
- 改 checkpoint；
- 改 method。

---

# 24. 停止规则

## Stop-A

如果：

```text
所有 A proxy 与 Oracle Gain 都没有明显关系
```

则停止 Student Absorbability。

---

## Stop-B

如果：

```text
A 能解释现象
但 R×U×A 无法优于 R×U / Full KD
```

则 A 只作为分析结论，不作为主方法。

---

## Stop-C

如果：

```text
R×U×A 只在 seed13 有效
多 seed 不稳定
```

则不得写成稳定方法优势。

---

## Stop-D

如果：

```text
MOSEI 有效
MOSI 严重负迁移
```

论文主张限定为：

```text
MOSEI evidence
```

不强行宣称跨数据集普适。

---

# 25. 推荐执行队列

## P0：立即执行

```text
[1] 冻结当前 P0 / main_table_v1 状态
[2] 不启动两数据集全量主表
[3] 增加 epoch checkpoint 保存
[4] 增加 per-coordinate diagnostic logging
[5] 跑 seed13 Full KD trajectory
[6] 跑 seed13 Uniform trajectory
[7] 跑 seed13 R×U trajectory
```

---

## P1：Absorbability diagnosis

```text
[1] 计算 gradient compatibility
[2] 计算 learning progress
[3] 计算 residual/difficulty
[4] 做 7-coordinate leave-one-out
[5] 构造 Oracle Gain
[6] 比较 R / U / RU / A 对 Oracle 的预测能力
```

---

## P2：A candidate screening

```text
A-only
RU+A
shuffled-A
global-gradient baseline
```

只 seed13。

---

## P3：方法冻结

若 Gate 通过：

```text
Full KD
Uniform
R
U
RU
A
RA
UA
RUA
shuffled-A
High-A
Low-A
Random-A
```

仍先 seed13。

---

## P4：正式多 seed

最终冻结后：

```text
13 / 42 / 2026
```

完成：

- internal baselines
- external KD
- core ablations
- bootstrap
- paired tests

---

## P5：泛化与收尾

```text
MOSI
robustness
efficiency
final official-test
```

---

# 26. 论文最终目标叙事

如果实验成立，论文主线建议收敛为：

## Observation

```text
Large multimodal teachers contain rich heterogeneous knowledge,
but transferring more knowledge does not necessarily improve a lightweight student.
```

## Diagnosis

```text
Teacher reliability and task relevance are insufficient:
knowledge must also be compatible with the student's current learning capacity.
```

## Method

```text
Decompose multimodal knowledge into interaction atoms,
estimate their reliability, relevance, and student absorbability,
and selectively distill only transferable knowledge.
```

## Evidence

```text
1. Adaptation strongly changes what knowledge the student can exploit.
2. Interaction fidelity is not monotonically related to task performance.
3. Student-side absorbability predicts beneficial interactions better than R/U alone.
4. Selective R×U×A distillation outperforms R×U and Full KD.
5. The effect transfers to MOSI and remains robust under modality degradation.
```

---

# 27. 最终完成定义

项目只有满足以下条件才认为“方法已完成”：

- Absorbability hypothesis 被实验证据支持；
- A definition 完全冻结；
- Ours 相对 Full KD 与 R×U 有独立收益；
- shuffled / high-low intervention 支持机制解释；
- 三 seed 结果完整；
- 主表包含近期 KD；
- MOSI 完成；
- robustness 完成；
- efficiency 完成；
- final official-test 只执行一次；
- 所有 checkpoint、预测、配置、commit、hash 可追溯；
- 所有失败实验保留，不选择性隐藏。

---

# 28. 当前最重要的执行原则

现阶段不要把算力优先用于：

```text
把现有 53-run × MOSEI/MOSI 主表全部跑完
```

而应优先回答：

```text
Student Absorbability 是否真实存在？
它能否预测哪些 interaction 对当前学生真正有帮助？
它是否能提供 R/U 之外的独立信息？
```

如果这三个问题成立，再进行大规模主表，论文主方法才真正值得冻结。

否则，应接受负结果并重新确定主方法，而不是继续围绕 R×U 增加更多经验性权重。
