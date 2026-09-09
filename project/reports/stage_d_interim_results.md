# Stage D 性能提升实验阶段性结果

> 数据快照：2026-09-09 10:02 CST
>
> 评估范围：CMU-MOSEI official train + official valid
>
> Official test：未读取、未评估

## 1. 实验状态

| 实验 | seed 13 | seed 42 | seed 2026 |
|---|---|---|---|
| D1 Full KD + LoRA(T+A) | 完成 | 完成 | 运行中 |
| D2 RU + LoRA(T+A) | 运行中 | 完成 | 已排队 |
| D3 Full KD + Hidden KD | 完成 | 完成 | Gate 失败，不运行 |
| D4 RU + Hidden KD | D3 Gate 失败，不运行 | D3 Gate 失败，不运行 | 不运行 |
| D5 RU + Hidden KD + LoRA | Hidden KD 无效，不运行 | Hidden KD 无效，不运行 | 不运行 |

运行中的 checkpoint 只用于进度说明，不纳入正式均值、显著性或论文结论。

## 2. D1：Full KD + LoRA(T+A)

LoRA 配置固定为 rank 8、alpha 16、dropout 0.05。Qwen3-0.6B 的目标模块为
`q_proj/k_proj/v_proj/o_proj`，WavLM 按实际模块名映射为
`q_proj/k_proj/v_proj/out_proj`；VideoMAE 保持冻结并复用缓存。总参数量为
732,492,152，可训练参数量为 42,060,296。

| seed | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ | best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | **0.4712** | **0.7918** | **0.8818** | **0.8822** | **0.5783** | 13 |
| 42 | **0.4791** | **0.7877** | **0.8602** | **0.8604** | **0.5676** | 6 |
| 两 seed | **0.4752 ± 0.0056** | **0.7897 ± 0.0029** | **0.8710 ± 0.0152** | **0.8713 ± 0.0154** | **0.5730 ± 0.0076** | — |

相同两 seed 的冻结 B1 平均 MAE 为 0.5081 ± 0.0061。D1 的平均 MAE 改善
0.0329；seed 13 和 42 分别改善 0.0412 和 0.0247，均超过预注册的 0.003
单 seed Gate。因此 LoRA 路线明确通过 Gate，正在补 seed 2026。

## 3. D2：RU + LoRA(T+A)

seed 42 已完成：

| seed | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ | best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | **0.4638** | **0.7995** | **0.8762** | **0.8770** | **0.5794** | 13 |

相对相同 seed 的 D1，D2 的 MAE 改善 0.0153、Pearson 提高 0.0118、Acc-2
提高 0.0160、F1 提高 0.0166、Acc-7 提高 0.0118。该结果超过
`D2 vs D1: ΔMAE ≤ -0.005` 的效果量目标，是当前最强的已完成单 seed 结果。

seed 13 截至快照已运行到 epoch 12，当前最佳 checkpoint 位于 epoch 9，MAE
为 0.4681；该值仅作进度记录。seed 13 完成后，持久队列将自动启动 seed 2026。

## 4. D3：Full KD + Hidden KD

D3 使用教师 TAV 的 2048-d hidden、学生 TAV fused 512-d representation、单层
`512→2048` projection 和 cosine loss，权重固定为 1.0。学生三个 encoder 均冻结。

| seed | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ | best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 0.5075 | 0.7464 | 0.8602 | 0.8573 | 0.5420 | 15 |
| 42 | 0.5153 | 0.7387 | 0.8491 | 0.8483 | 0.5452 | 4 |
| 两 seed | 0.5114 ± 0.0055 | 0.7426 ± 0.0055 | 0.8547 ± 0.0079 | 0.8528 ± 0.0064 | 0.5436 ± 0.0023 | — |

相同两 seed 的 B1 平均 MAE 为 0.5081，D3 反而退化 0.0033，未通过 Hidden
KD Gate。按预注册规则停止 D3 seed 2026，不执行 D4 或 D5，也不搜索 hidden
loss 类型或权重。

## 5. 当前结论

1. Frozen student representation 是主要瓶颈之一；受限的 Text+Audio LoRA 带来远大于此前 interaction 权重调整的增益。
2. 已完成的 D2 seed 42 表明 RU 与 LoRA 可能互补，但必须等待 seed 13/2026 完成后再作稳定性结论。
3. 第一版 cosine Hidden KD 没有改善 B1，按 Gate 停止该路线。
4. 下一步是完成 D1/D2 seed 2026，随后执行三 seed mean ± sample std、10,000 次 paired bootstrap 以及 Acc-2/F1/Acc-7 显著性分析。
5. Official test 继续保持未触碰；当前所有选择只使用 official train/valid。
