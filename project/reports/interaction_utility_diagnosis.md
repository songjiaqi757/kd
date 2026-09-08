# Interaction Utility Diagnosis

范围：official train/valid；official test 未使用。比较方向均为 pair-SNR − B1 Full KD。

## Official-train global utility

| Interaction | abs corr(I, y) | normalized utility |
|---|---:|---:|
| TA | 0.820008 | 1.324889 |
| TV | 0.518997 | 0.838545 |
| AV | 0.517772 | 0.836566 |

## Fidelity → task utility（三 seed）

| Interaction | Teacher mean SNR | Δ interaction MAE | Corr(ΔE_I, ΔE_task) | Interaction improved rate | Task-useful rate | Δ task when I improves |
|---|---:|---:|---:|---:|---:|---:|
| TA | 9.3079 | -0.6028 ± 0.2222 | 0.0341 ± 0.0095 | 0.8128 ± 0.0781 | 0.4948 ± 0.0111 | -0.0041 ± 0.0043 |
| TV | 4.9227 | -0.2736 ± 0.0305 | -0.0225 ± 0.0735 | 0.7527 ± 0.0074 | 0.4904 ± 0.0089 | 0.0015 ± 0.0105 |
| AV | 6.1272 | -0.5282 ± 0.2312 | -0.0076 ± 0.0477 | 0.8719 ± 0.0820 | 0.4865 ± 0.0021 | 0.0007 ± 0.0059 |
| TAV | 5.0780 | -0.2639 ± 0.1663 | 0.0065 ± 0.0221 | 0.7125 ± 0.0773 | 0.4902 ± 0.0144 | -0.0013 ± 0.0033 |

## Pair-SNR vs Full KD：MAE paired bootstrap

| Seed | ΔMAE | 95% CI | P(pair-SNR better) |
|---:|---:|---:|---:|
| 13 | -0.003336 | [-0.014891, +0.008027] | 0.7147 |
| 42 | +0.003240 | [-0.008018, +0.014613] | 0.2910 |
| 2026 | +0.003912 | [-0.007002, +0.015301] | 0.2382 |

## Interaction strength tertile

| Group | Method | MAE | Pearson | Acc-2 |
|---|---|---:|---:|---:|
| low | B1 Full KD | 0.4617 ± 0.0083 | 0.4460 ± 0.0062 | 0.7188 ± 0.0073 |
| low | pair-SNR | 0.4626 ± 0.0104 | 0.4635 ± 0.0112 | 0.7213 ± 0.0233 |
| middle | B1 Full KD | 0.4696 ± 0.0091 | 0.6871 ± 0.0198 | 0.8549 ± 0.0110 |
| middle | pair-SNR | 0.4775 ± 0.0060 | 0.6679 ± 0.0107 | 0.8563 ± 0.0091 |
| high | B1 Full KD | 0.5903 ± 0.0067 | 0.8143 ± 0.0049 | 0.9476 ± 0.0070 |
| high | pair-SNR | 0.5854 ± 0.0123 | 0.8158 ± 0.0090 | 0.9446 ± 0.0044 |

## Teacher uncertainty tertile

| Group | Method | MAE | Pearson | Acc-2 |
|---|---|---:|---:|---:|
| low | B1 Full KD | 0.4518 ± 0.0094 | 0.6386 ± 0.0088 | 0.8401 ± 0.0118 |
| low | pair-SNR | 0.4596 ± 0.0071 | 0.6232 ± 0.0192 | 0.8366 ± 0.0114 |
| middle | B1 Full KD | 0.5122 ± 0.0049 | 0.6449 ± 0.0063 | 0.8244 ± 0.0087 |
| middle | pair-SNR | 0.5115 ± 0.0085 | 0.6501 ± 0.0080 | 0.8272 ± 0.0126 |
| high | B1 Full KD | 0.5576 ± 0.0057 | 0.8142 ± 0.0096 | 0.8911 ± 0.0093 |
| high | pair-SNR | 0.5544 ± 0.0083 | 0.8126 ± 0.0073 | 0.8917 ± 0.0086 |

## Modality conflict tertile

| Group | Method | MAE | Pearson | Acc-2 |
|---|---|---:|---:|---:|
| low | B1 Full KD | 0.4194 ± 0.0026 | 0.6053 ± 0.0119 | 0.8280 ± 0.0048 |
| low | pair-SNR | 0.4284 ± 0.0059 | 0.5860 ± 0.0201 | 0.8295 ± 0.0132 |
| middle | B1 Full KD | 0.4655 ± 0.0083 | 0.6754 ± 0.0111 | 0.8519 ± 0.0103 |
| middle | pair-SNR | 0.4653 ± 0.0066 | 0.6797 ± 0.0083 | 0.8439 ± 0.0141 |
| high | B1 Full KD | 0.6366 ± 0.0076 | 0.7909 ± 0.0058 | 0.8742 ± 0.0056 |
| high | pair-SNR | 0.6318 ± 0.0027 | 0.7909 ± 0.0067 | 0.8797 ± 0.0138 |

Utility 权重只由 official-train label 计算；valid label 只用于诊断和方法选择。
