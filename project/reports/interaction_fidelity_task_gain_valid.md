# Cross-modal interaction fidelity and polarity correction (official validation)

比较 seed13、按 validation MAE 选中的 Full KD 与 Uniform Interaction。正的 `ΔE_cross = E_cross,Full − E_cross,Uniform` 表示 Uniform 对 TA/TV/AV/TAV 的 cross-modal interaction reconstruction 更好；正的 `ΔM = M_Full − M_Uniform` 表示 Uniform 的逐样本情感绝对误差更小。

## Continuous prediction diagnostic (auxiliary)

| Dataset | N / video clusters | E_cross Full | E_cross Uniform | Mean ΔE_cross | Mean ΔM | Spearman ρ | Asymptotic p | Video-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOSEI | 1,871 / 300 | 0.448671 | 0.236799 | +0.211872 | -0.003293 | +0.0118 | 0.6105 | [-0.0333, +0.0589] |
| MOSI | 229 / 10 | 1.208071 | 0.472039 | +0.736032 | -0.031203 | +0.0271 | 0.6828 | [-0.0916, +0.1379] |

Asymptotic p 值把 utterance 当作独立观测，仅作参考；主要不确定性结果是按原视频聚类的 10,000 次 percentile bootstrap CI。

## Polarity correction analysis (primary)

采用 Acc-2 non-zero 口径：排除真实标签为 0 的样本；真实标签 `>0` 为 positive，回归预测 `>=0` 为 predicted positive。纵轴统计量是组内 mean ΔE_cross。

### MOSEI polarity outcomes

Full/Uniform Acc-2 = 0.872740 / 0.877608，ΔAcc-2 = +0.004868，exact McNemar p = 0.5507；non-zero N=1,438。

| Outcome | N | Video clusters | Mean ΔE_cross | Median ΔE_cross | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Full wrong → Uniform correct | 54 | 44 | +0.128126 | +0.143511 | [+0.090106, +0.171520] |
| Both correct | 1208 | 273 | +0.248418 | +0.256571 | [+0.231158, +0.266226] |
| Both wrong | 129 | 95 | +0.089356 | +0.100086 | [+0.057728, +0.119436] |
| Full correct → Uniform wrong | 47 | 42 | +0.060401 | +0.050317 | [+0.016308, +0.109611] |

Corrected minus harmed contrast: +0.067724, cluster-bootstrap 95% CI [+0.009628, +0.125607]。

### MOSI polarity outcomes

Full/Uniform Acc-2 = 0.851852 / 0.851852，ΔAcc-2 = +0.000000，exact McNemar p = 1.0000；non-zero N=216。

| Outcome | N | Video clusters | Mean ΔE_cross | Median ΔE_cross | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Full wrong → Uniform correct | 9 | 6 | +0.612899 | +0.487330 | [+0.419113, +0.830056] |
| Both correct | 175 | 10 | +0.838131 | +0.869617 | [+0.602569, +1.013536] |
| Both wrong | 23 | 9 | +0.392014 | +0.531390 | [+0.061889, +0.632542] |
| Full correct → Uniform wrong | 9 | 6 | +0.194179 | +0.199570 | [-0.173482, +0.499121] |

Corrected minus harmed contrast: +0.418720, cluster-bootstrap 95% CI [+0.220063, +0.735306]。

## Seven-coordinate overall fidelity (auxiliary audit)

| Dataset | E_all Full | E_all Uniform | Mean ΔE_all | Spearman(ΔE_all, ΔM) | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| MOSEI | 0.448602 | 0.266878 | +0.181725 | +0.0333 | [-0.0141, +0.0806] |
| MOSI | 1.143837 | 0.538440 | +0.605396 | +0.0739 | [-0.0475, +0.2069] |

## Interpretation boundary

这是固定 seed13、固定 validation-selected checkpoint 的样本级相关分析。Validation 同时参与了 checkpoint 选择，cluster bootstrap 只覆盖原视频抽样不确定性，不覆盖训练随机性或模型选择不确定性。无论相关系数是否显著，都不能解释为 interaction fidelity 对任务改善的因果效应。
逐样本任务误差使用 interaction evidence 已保存的同一次七子集 forward 中的 TAV 输出，以确保 interaction 与任务预测逐样本配对；它不是从训练 history 的汇总 MAE 反推得到。
MOSI validation 只有 10 个原视频 cluster，因此其 cluster bootstrap 区间尤其不精确。

Figure: `/home/wy/sjq/kd/project/reports/paper_figures/fig_interaction_fidelity_task_gain_valid.pdf`
Caption: `/home/wy/sjq/kd/project/reports/paper_figures/fig_interaction_fidelity_task_gain_valid_caption.md`
