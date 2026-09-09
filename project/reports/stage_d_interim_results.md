# Stage D 性能提升实验阶段性结果

> 数据快照：2026-09-09 14:46 CST
>
> 评估范围：CMU-MOSEI official train + official valid
>
> Official test：未读取、未评估

## 1. 实验状态

| 实验 | seed 13 | seed 42 | seed 2026 |
|---|---|---|---|
| D1 Full KD + LoRA(T+A) | 完成 | 完成 | 完成 |
| D2 RU + LoRA(T+A) | 完成 | 完成 | 运行中 |
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
| 2026 | **0.4681** | **0.7925** | **0.8797** | **0.8792** | **0.5772** | 9 |
| 三 seed | **0.4728 ± 0.0057** | **0.7907 ± 0.0026** | **0.8739 ± 0.0119** | **0.8739 ± 0.0118** | **0.5744 ± 0.0059** | — |

冻结 B1 三 seed 平均 MAE 为 0.5072 ± 0.0046。D1 的平均 MAE 改善 0.0344；
seed 13、42、2026 分别改善 0.0412、0.0247、0.0374，3/3 seeds 均超过
预注册的 0.003 单 seed Gate。因此 LoRA 路线明确通过 Gate。

## 3. D2：RU + LoRA(T+A)

seeds 13、42 已完成：

| seed | MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ | best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | **0.4645** | **0.7929** | **0.8748** | **0.8744** | **0.5895** | 14 |
| 42 | **0.4638** | **0.7995** | **0.8762** | **0.8770** | **0.5794** | 13 |
| 两 seed | **0.4641 ± 0.0005** | **0.7962 ± 0.0047** | **0.8755 ± 0.0010** | **0.8757 ± 0.0019** | **0.5844 ± 0.0072** | — |

相对相同 seed 的 D1，D2 的 MAE 在 seed 13、42 分别改善 0.0067、0.0153，
2/2 seeds 均超过 `D2 vs D1: ΔMAE ≤ -0.005` 的效果量目标。两 seed 平均
MAE 相对 D1 改善 0.0110，是当前最强的候选方法。

seed 2026 截至快照已运行到 epoch 3，当前最佳 checkpoint 位于 epoch 1，MAE
为 0.5062；该值仅作进度记录，不进入均值或正式结论。

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
2. D2 在已完成的 seeds 13、42 上均以超过 0.005 的 MAE 幅度优于对应 D1，表明 RU 与 LoRA 存在稳定互补信号；仍需等待 seed 2026 后形成三 seed 结论。
3. 第一版 cosine Hidden KD 没有改善 B1，按 Gate 停止该路线。
4. 下一步是完成 D2 seed 2026，随后执行三 seed mean ± sample std、10,000 次 paired bootstrap 以及 Acc-2/F1/Acc-7 显著性分析。
5. Official test 继续保持未触碰；当前所有选择只使用 official train/valid。
