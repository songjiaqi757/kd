# best_valid_acc2_nonzero：全部实验汇总

[口径、缺失项说明及来源审计](README.md)。所有表均采用同一选中 epoch 的完整指标，± 为样本标准差。

## historical_mosi_seven

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| ensemble_full | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8577 ± 0.1018 | 86.11 ± 0.46 | 0.7776 ± 0.0217 | 86.06 ± 0.57 |
| ensemble_full | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.8099 ± 0.1543 | 84.50 ± 0.23 | 0.7879 ± 0.0152 | 84.47 ± 0.37 |
| first_order_interaction | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7809 ± 0.0181 | 85.96 ± 1.07 | 0.7803 ± 0.0195 | 85.99 ± 1.06 |
| first_order_interaction | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7183 ± 0.0226 | 84.50 ± 1.10 | 0.7938 ± 0.0076 | 84.38 ± 1.19 |
| first_second_order_interaction | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7828 ± 0.0377 | 85.80 ± 0.27 | 0.7753 ± 0.0136 | 85.73 ± 0.16 |
| first_second_order_interaction | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7411 ± 0.0216 | 84.76 ± 0.66 | 0.7900 ± 0.0159 | 84.71 ± 0.80 |
| full_kd | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7873 ± 0.0191 | 85.49 ± 1.07 | 0.7813 ± 0.0020 | 85.52 ± 1.09 |
| full_kd | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7197 ± 0.0073 | 84.60 ± 1.00 | 0.7861 ± 0.0065 | 84.56 ± 1.02 |
| random_orthogonal | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7887 ± 0.0270 | 85.80 ± 0.71 | 0.7851 ± 0.0136 | 85.79 ± 0.74 |
| random_orthogonal | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7292 ± 0.0354 | 85.06 ± 0.30 | 0.7968 ± 0.0127 | 85.00 ± 0.37 |
| subset7 | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8265 ± 0.0247 | 85.80 ± 0.27 | 0.7718 ± 0.0051 | 85.84 ± 0.23 |
| subset7 | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7476 ± 0.0263 | 83.89 ± 0.63 | 0.7865 ± 0.0081 | 83.77 ± 0.74 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7849 ± 0.0072 | 85.49 ± 1.63 | 0.7790 ± 0.0070 | 85.50 ± 1.67 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7334 ± 0.0453 | 83.99 ± 0.70 | 0.7920 ± 0.0216 | 83.96 ± 0.63 |

## historical_mosei_seven

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| ensemble_full | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4967 | 88.94 | 0.7826 | 88.94 |
| ensemble_full | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5097 | 87.01 | 0.8048 | 87.00 |
| first_order_interaction | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4694 ± 0.0013 | 88.80 ± 0.18 | 0.7939 ± 0.0041 | 88.70 ± 0.19 |
| first_order_interaction | mosei | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.4826 ± 0.0057 | 87.86 ± 0.63 | 0.8187 ± 0.0019 | 87.76 ± 0.67 |
| first_second_order_interaction | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4902 ± 0.0187 | 88.73 ± 0.30 | 0.7913 ± 0.0017 | 88.68 ± 0.27 |
| first_second_order_interaction | mosei | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.4957 ± 0.0169 | 87.77 ± 0.28 | 0.8178 ± 0.0057 | 87.71 ± 0.24 |
| full_kd | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4792 ± 0.0174 | 88.66 ± 0.24 | 0.7904 ± 0.0046 | 88.62 ± 0.21 |
| full_kd | mosei | full | valid_acc2_nonzero | test | 13,42 (2/3) | 0.4949 ± 0.0242 | 87.66 ± 0.45 | 0.8143 ± 0.0040 | 87.65 ± 0.40 |
| random_orthogonal | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4730 ± 0.0043 | 88.94 ± 0.25 | 0.7917 ± 0.0050 | 88.85 ± 0.27 |
| random_orthogonal | mosei | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.4828 ± 0.0076 | 87.67 ± 0.57 | 0.8201 ± 0.0024 | 87.57 ± 0.59 |
| subset7 | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4763 ± 0.0223 | 88.66 ± 0.14 | 0.7925 ± 0.0081 | 88.55 ± 0.26 |
| subset7 | mosei | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.4876 ± 0.0189 | 87.64 ± 0.27 | 0.8178 ± 0.0064 | 87.52 ± 0.33 |
| uniform_interaction | mosei | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.4852 ± 0.0222 | 88.83 ± 0.33 | 0.7933 ± 0.0049 | 88.72 ± 0.36 |
| uniform_interaction | mosei | full | valid_acc2_nonzero | test | 13 (1/3) | 0.5173 | 87.64 | 0.8132 | 87.48 |

## mosi_research_14

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| dgl_icd | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8322 ± 0.0196 | 85.65 ± 0.46 | 0.7584 ± 0.0132 | 85.59 ± 0.49 |
| dgl_icd | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.8341 ± 0.0065 | 83.49 ± 0.32 | 0.7566 ± 0.0213 | 83.54 ± 0.33 |
| ensemble_source | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7661 ± 0.0121 | 86.27 ± 0.71 | 0.7824 ± 0.0096 | 86.20 ± 0.79 |
| ensemble_source | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7411 ± 0.0356 | 84.15 ± 1.68 | 0.7919 ± 0.0122 | 84.15 ± 1.78 |
| explicit_full_kd | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7931 ± 0.0445 | 84.88 ± 1.07 | 0.7699 ± 0.0150 | 84.83 ± 1.13 |
| explicit_full_kd | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7397 ± 0.0277 | 84.50 ± 0.87 | 0.7885 ± 0.0166 | 84.54 ± 0.86 |
| explicit_uniform_interaction | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7778 ± 0.0115 | 85.65 ± 0.93 | 0.7896 ± 0.0068 | 85.50 ± 0.92 |
| explicit_uniform_interaction | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7709 ± 0.0211 | 84.76 ± 0.81 | 0.8042 ± 0.0046 | 84.81 ± 0.77 |
| fusion_only | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8373 ± 0.0940 | 85.49 ± 1.34 | 0.7692 ± 0.0351 | 85.52 ± 1.36 |
| fusion_only | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7991 ± 0.0476 | 83.43 ± 0.77 | 0.7754 ± 0.0099 | 83.22 ± 0.91 |
| late_decay | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7914 ± 0.0288 | 85.65 ± 1.39 | 0.7851 ± 0.0112 | 85.65 ± 1.34 |
| late_decay | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7805 ± 0.0542 | 84.15 ± 1.22 | 0.7728 ± 0.0200 | 84.14 ± 1.21 |
| order_routed | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8134 ± 0.0623 | 85.49 ± 1.07 | 0.7731 ± 0.0271 | 85.48 ± 1.04 |
| order_routed | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7656 ± 0.0565 | 84.15 ± 1.15 | 0.7780 ± 0.0177 | 84.14 ± 1.11 |
| scrambled_routed | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7642 ± 0.0157 | 86.57 ± 0.46 | 0.7883 ± 0.0116 | 86.56 ± 0.51 |
| scrambled_routed | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7190 ± 0.0277 | 85.32 ± 1.07 | 0.7943 ± 0.0128 | 85.29 ± 1.05 |
| target_uniform | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.8308 ± 0.0556 | 85.34 ± 0.27 | 0.7690 ± 0.0051 | 85.26 ± 0.14 |
| target_uniform | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.8175 ± 0.0965 | 84.10 ± 0.62 | 0.7834 ± 0.0300 | 84.12 ± 0.68 |
| target_variance | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7682 ± 0.0486 | 85.96 ± 0.71 | 0.7834 ± 0.0323 | 85.95 ± 0.66 |
| target_variance | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7164 ± 0.0202 | 84.76 ± 0.40 | 0.8013 ± 0.0037 | 84.73 ± 0.47 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | validation | 2026,13,42 (3/3) | 0.7839 ± 0.0343 | 85.96 ± 0.96 | 0.7847 ± 0.0077 | 85.96 ± 0.92 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | test | 2026,13,42 (3/3) | 0.7712 ± 0.0596 | 83.43 ± 0.54 | 0.7766 ± 0.0171 | 83.45 ± 0.52 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7945 ± 0.0600 | 86.11 ± 0.46 | 0.7836 ± 0.0256 | 86.05 ± 0.37 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7648 ± 0.0471 | 84.76 ± 0.70 | 0.7881 ± 0.0111 | 84.76 ± 0.78 |
| uniform_interaction_gradient_projected | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7718 ± 0.0211 | 86.42 ± 0.53 | 0.8002 ± 0.0120 | 86.46 ± 0.56 |
| uniform_interaction_gradient_projected | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7162 ± 0.0132 | 84.35 ± 0.49 | 0.7990 ± 0.0066 | 84.26 ± 0.47 |
| uniform_interaction_shared_dropout | mosi | full | valid_acc2_nonzero | validation | 13,42,2026 (3/3) | 0.7931 ± 0.0228 | 85.34 ± 0.71 | 0.7745 ± 0.0180 | 85.31 ± 0.73 |
| uniform_interaction_shared_dropout | mosi | full | valid_acc2_nonzero | test | 13,42,2026 (3/3) | 0.7686 ± 0.0071 | 83.59 ± 0.98 | 0.7860 ± 0.0155 | 83.60 ± 0.95 |

## mosei_extensions_v3

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| uniform_interaction | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4740 ± 0.0045 | 89.04 ± 0.08 | 0.7931 ± 0.0053 | 88.94 ± 0.12 |
| uniform_interaction | mosei | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.4824 ± 0.0063 | 87.69 ± 0.37 | 0.8200 ± 0.0060 | 87.61 ± 0.35 |
| uniform_interaction_coordinate_compatible | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4796 ± 0.0144 | 89.31 ± 0.04 | 0.7983 ± 0.0047 | 89.22 ± 0.13 |
| uniform_interaction_coordinate_compatible | mosei | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.4905 ± 0.0190 | 87.89 ± 0.18 | 0.8207 ± 0.0034 | 87.78 ± 0.25 |
| uniform_interaction_gradient_projected | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4719 ± 0.0077 | 88.76 ± 0.66 | 0.7910 ± 0.0087 | 88.65 ± 0.70 |
| uniform_interaction_gradient_projected | mosei | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.4805 ± 0.0035 | 88.02 ± 0.23 | 0.8202 ± 0.0027 | 87.92 ± 0.29 |
| uniform_interaction_shared_dropout | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4740 ± 0.0035 | 88.78 ± 0.34 | 0.7934 ± 0.0007 | 88.75 ± 0.21 |
| uniform_interaction_shared_dropout | mosei | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.4863 ± 0.0043 | 87.16 ± 0.47 | 0.8169 ± 0.0043 | 87.12 ± 0.38 |

## 全部历史实验线（包括 pilot 与 smoke）

### experiments/decision_icd_v1

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| smoke_interaction | mosi | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 4.2000 | 0.00 | 0.1111 | 0.00 |
| smoke_interaction | mosi | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |

### experiments/decision_icd_v1/mosi

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| icd_decision_full | mosi | full | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.7947 ± 0.0345 | 85.65 ± 1.31 | 0.7745 ± 0.0306 | 85.64 ± 1.31 |
| icd_decision_full | mosi | full | valid_acc2_nonzero | test | 13,42 (2/2) | 0.6979 ± 0.0142 | 85.75 ± 0.11 | 0.8073 ± 0.0021 | 85.69 ± 0.04 |
| icd_decision_interaction | mosi | full | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.7997 ± 0.0069 | 85.65 ± 0.65 | 0.7792 ± 0.0124 | 85.65 ± 0.73 |
| icd_decision_interaction | mosi | full | valid_acc2_nonzero | test | 13,42 (2/2) | 0.7298 ± 0.0029 | 84.30 ± 0.22 | 0.7807 ± 0.0014 | 84.24 ± 0.15 |
| icd_decision_none | mosi | full | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.8012 ± 0.0235 | 86.34 ± 0.98 | 0.7809 ± 0.0058 | 86.33 ± 0.95 |
| icd_decision_none | mosi | full | valid_acc2_nonzero | test | 13,42 (2/2) | 0.7994 ± 0.0484 | 83.16 ± 0.32 | 0.7698 ± 0.0173 | 83.17 ± 0.25 |
| icd_decision_subset | mosi | full | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.8889 ± 0.0818 | 84.49 ± 0.33 | 0.7733 ± 0.0269 | 84.56 ± 0.31 |
| icd_decision_subset | mosi | full | valid_acc2_nonzero | test | 13,42 (2/2) | 0.7777 ± 0.1116 | 83.77 ± 1.83 | 0.8048 ± 0.0098 | 83.46 ± 2.01 |

### experiments/equal_modality_extensions_v3/mosei

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| uniform_interaction | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4740 ± 0.0045 | 89.04 ± 0.08 | 0.7931 ± 0.0053 | 88.94 ± 0.12 |
| uniform_interaction | mosei | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.4824 ± 0.0063 | 87.69 ± 0.37 | 0.8200 ± 0.0060 | 87.61 ± 0.35 |
| uniform_interaction_coordinate_compatible | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4796 ± 0.0144 | 89.31 ± 0.04 | 0.7983 ± 0.0047 | 89.22 ± 0.13 |
| uniform_interaction_coordinate_compatible | mosei | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.4905 ± 0.0190 | 87.89 ± 0.18 | 0.8207 ± 0.0034 | 87.78 ± 0.25 |
| uniform_interaction_gradient_projected | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4719 ± 0.0077 | 88.76 ± 0.66 | 0.7910 ± 0.0087 | 88.65 ± 0.70 |
| uniform_interaction_gradient_projected | mosei | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.4805 ± 0.0035 | 88.02 ± 0.23 | 0.8202 ± 0.0027 | 87.92 ± 0.29 |
| uniform_interaction_shared_dropout | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4740 ± 0.0035 | 88.78 ± 0.34 | 0.7934 ± 0.0007 | 88.75 ± 0.21 |
| uniform_interaction_shared_dropout | mosei | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.4863 ± 0.0043 | 87.16 ± 0.47 | 0.8169 ± 0.0043 | 87.12 ± 0.38 |

### experiments/interaction_evidence_v1/mosei

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| first_second_order_interaction | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.5057 | 88.53 | 0.7893 | 88.53 |
| first_second_order_interaction | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5106 | 87.53 | 0.8127 | 87.53 |
| random_orthogonal | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4779 | 89.22 | 0.7866 | 89.13 |
| random_orthogonal | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.4897 | 87.51 | 0.8176 | 87.42 |

### experiments/interaction_evidence_v1/mosi

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| first_second_order_interaction | mosi | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.8041 | 85.65 | 0.7706 | 85.69 |
| first_second_order_interaction | mosi | full | valid_acc2_nonzero | test | 13 (1/1) | 0.7166 | 84.30 | 0.7947 | 84.09 |
| random_orthogonal | mosi | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.7906 | 85.65 | 0.7844 | 85.69 |
| random_orthogonal | mosi | full | valid_acc2_nonzero | test | 13 (1/1) | 0.6934 | 85.06 | 0.8114 | 84.88 |

### experiments/mosi_research_v1

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| dgl_icd | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.8322 ± 0.0196 | 85.65 ± 0.46 | 0.7584 ± 0.0132 | 85.59 ± 0.49 |
| dgl_icd | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.8341 ± 0.0065 | 83.49 ± 0.32 | 0.7566 ± 0.0213 | 83.54 ± 0.33 |
| ensemble_source | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.7661 ± 0.0121 | 86.27 ± 0.71 | 0.7824 ± 0.0096 | 86.20 ± 0.79 |
| ensemble_source | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.7411 ± 0.0356 | 84.15 ± 1.68 | 0.7919 ± 0.0122 | 84.15 ± 1.78 |
| explicit_full_kd | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.7931 ± 0.0445 | 84.88 ± 1.07 | 0.7699 ± 0.0150 | 84.83 ± 1.13 |
| explicit_full_kd | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.7397 ± 0.0277 | 84.50 ± 0.87 | 0.7885 ± 0.0166 | 84.54 ± 0.86 |
| explicit_uniform_interaction | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.7778 ± 0.0115 | 85.65 ± 0.93 | 0.7896 ± 0.0068 | 85.50 ± 0.92 |
| explicit_uniform_interaction | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.7709 ± 0.0211 | 84.76 ± 0.81 | 0.8042 ± 0.0046 | 84.81 ± 0.77 |
| fusion_only | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.8373 ± 0.0940 | 85.49 ± 1.34 | 0.7692 ± 0.0351 | 85.52 ± 1.36 |
| fusion_only | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.7991 ± 0.0476 | 83.43 ± 0.77 | 0.7754 ± 0.0099 | 83.22 ± 0.91 |
| late_decay | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.7914 ± 0.0288 | 85.65 ± 1.39 | 0.7851 ± 0.0112 | 85.65 ± 1.34 |
| late_decay | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.7805 ± 0.0542 | 84.15 ± 1.22 | 0.7728 ± 0.0200 | 84.14 ± 1.21 |
| order_routed | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.8134 ± 0.0623 | 85.49 ± 1.07 | 0.7731 ± 0.0271 | 85.48 ± 1.04 |
| order_routed | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.7656 ± 0.0565 | 84.15 ± 1.15 | 0.7780 ± 0.0177 | 84.14 ± 1.11 |
| scrambled_routed | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.7642 ± 0.0157 | 86.57 ± 0.46 | 0.7883 ± 0.0116 | 86.56 ± 0.51 |
| scrambled_routed | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.7190 ± 0.0277 | 85.32 ± 1.07 | 0.7943 ± 0.0128 | 85.29 ± 1.05 |
| target_uniform | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.8308 ± 0.0556 | 85.34 ± 0.27 | 0.7690 ± 0.0051 | 85.26 ± 0.14 |
| target_uniform | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.8175 ± 0.0965 | 84.10 ± 0.62 | 0.7834 ± 0.0300 | 84.12 ± 0.68 |
| target_variance | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.7682 ± 0.0486 | 85.96 ± 0.71 | 0.7834 ± 0.0323 | 85.95 ± 0.66 |
| target_variance | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.7164 ± 0.0202 | 84.76 ± 0.40 | 0.8013 ± 0.0037 | 84.73 ± 0.47 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | validation | 2026 (1/1) | 0.7493 | 85.19 | 0.7922 | 85.22 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | test | 2026 (1/1) | 0.7149 | 83.99 | 0.7904 | 84.01 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.7945 ± 0.0600 | 86.11 ± 0.46 | 0.7836 ± 0.0256 | 86.05 ± 0.37 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.7648 ± 0.0471 | 84.76 ± 0.70 | 0.7881 ± 0.0111 | 84.76 ± 0.78 |
| uniform_interaction_gradient_projected | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.7718 ± 0.0211 | 86.42 ± 0.53 | 0.8002 ± 0.0120 | 86.46 ± 0.56 |
| uniform_interaction_gradient_projected | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.7162 ± 0.0132 | 84.35 ± 0.49 | 0.7990 ± 0.0066 | 84.26 ± 0.47 |
| uniform_interaction_shared_dropout | mosi | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.7931 ± 0.0228 | 85.34 ± 0.71 | 0.7745 ± 0.0180 | 85.31 ± 0.73 |
| uniform_interaction_shared_dropout | mosi | full | valid_acc2_nonzero | test | 13,2026,42 (3/3) | 0.7686 ± 0.0071 | 83.59 ± 0.98 | 0.7860 ± 0.0155 | 83.60 ± 0.95 |

### experiments/multiseed_controls_v1/mosei

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| random_orthogonal | mosei | full | valid_acc2_nonzero | validation | 2026,42 (2/2) | 0.4705 ± 0.0001 | 88.80 ± 0.10 | 0.7943 ± 0.0035 | 88.71 ± 0.16 |
| random_orthogonal | mosei | full | valid_acc2_nonzero | test | 2026,42 (2/2) | 0.4793 ± 0.0066 | 87.75 ± 0.78 | 0.8213 ± 0.0018 | 87.65 ± 0.81 |
| subset7 | mosei | full | valid_acc2_nonzero | validation | 2026,42 (2/2) | 0.4635 ± 0.0033 | 88.73 ± 0.10 | 0.7971 ± 0.0017 | 88.65 ± 0.28 |
| subset7 | mosei | full | valid_acc2_nonzero | test | 2026,42 (2/2) | 0.4769 ± 0.0057 | 87.80 ± 0.06 | 0.8210 ± 0.0045 | 87.70 ± 0.07 |

### experiments/multiseed_controls_v1/mosei/followup_order_controls

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| first_order_interaction | mosei | full | valid_acc2_nonzero | validation | 2026,42 (2/2) | 0.4694 ± 0.0019 | 88.84 ± 0.25 | 0.7915 ± 0.0009 | 88.78 ± 0.20 |
| first_order_interaction | mosei | full | valid_acc2_nonzero | test | 2026,42 (2/2) | 0.4793 ± 0.0013 | 88.17 ± 0.51 | 0.8187 ± 0.0027 | 88.10 ± 0.47 |
| first_second_order_interaction | mosei | full | valid_acc2_nonzero | validation | 2026,42 (2/2) | 0.4824 ± 0.0184 | 88.84 ± 0.34 | 0.7922 ± 0.0004 | 88.75 ± 0.34 |
| first_second_order_interaction | mosei | full | valid_acc2_nonzero | test | 2026,42 (2/2) | 0.4883 ± 0.0155 | 87.89 ± 0.27 | 0.8203 ± 0.0051 | 87.80 ± 0.26 |

### experiments/multiseed_controls_v1/mosi

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| first_second_order_interaction | mosi | full | valid_acc2_nonzero | validation | 2026,42 (2/2) | 0.7722 ± 0.0465 | 85.88 ± 0.33 | 0.7776 ± 0.0184 | 85.75 ± 0.23 |
| first_second_order_interaction | mosi | full | valid_acc2_nonzero | test | 2026,42 (2/2) | 0.7534 ± 0.0057 | 84.98 ± 0.75 | 0.7876 ± 0.0217 | 85.02 ± 0.83 |
| full_kd | mosi | full | valid_acc2_nonzero | validation | 2026,42 (2/2) | 0.7979 ± 0.0074 | 85.19 ± 1.31 | 0.7804 ± 0.0016 | 85.20 ± 1.33 |
| full_kd | mosi | full | valid_acc2_nonzero | test | 2026,42 (2/2) | 0.7226 ± 0.0074 | 84.68 ± 1.40 | 0.7828 ± 0.0042 | 84.64 ± 1.43 |
| random_orthogonal | mosi | full | valid_acc2_nonzero | validation | 2026,42 (2/2) | 0.7878 ± 0.0382 | 85.88 ± 0.98 | 0.7854 ± 0.0191 | 85.85 ± 1.04 |
| random_orthogonal | mosi | full | valid_acc2_nonzero | test | 2026,42 (2/2) | 0.7471 ± 0.0243 | 85.06 ± 0.43 | 0.7895 ± 0.0008 | 85.06 ± 0.50 |
| subset7 | mosi | full | valid_acc2_nonzero | validation | 2026,42 (2/2) | 0.8132 ± 0.0124 | 85.88 ± 0.33 | 0.7709 ± 0.0069 | 85.91 ± 0.29 |
| subset7 | mosi | full | valid_acc2_nonzero | test | 2026,42 (2/2) | 0.7421 ± 0.0347 | 84.15 ± 0.65 | 0.7887 ± 0.0102 | 84.04 ± 0.81 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | validation | 2026,42 (2/2) | 0.7884 ± 0.0054 | 84.72 ± 1.31 | 0.7782 ± 0.0097 | 84.71 ± 1.34 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | test | 2026,42 (2/2) | 0.7594 ± 0.0081 | 83.61 ± 0.32 | 0.7796 ± 0.0035 | 83.63 ± 0.40 |

### experiments/multiseed_controls_v1/mosi_additional_controls

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| ensemble_full | mosi | full | valid_acc2_nonzero | validation | 2026,42 (2/2) | 0.8841 ± 0.1286 | 85.88 ± 0.33 | 0.7663 ± 0.0130 | 85.77 ± 0.38 |
| ensemble_full | mosi | full | valid_acc2_nonzero | test | 2026,42 (2/2) | 0.8568 ± 0.1855 | 84.60 ± 0.22 | 0.7814 ± 0.0144 | 84.66 ± 0.24 |
| first_order_interaction | mosi | full | valid_acc2_nonzero | validation | 2026,42 (2/2) | 0.7790 ± 0.0252 | 85.65 ± 1.31 | 0.7915 ± 0.0028 | 85.68 ± 1.30 |
| first_order_interaction | mosi | full | valid_acc2_nonzero | test | 2026,42 (2/2) | 0.7219 ± 0.0307 | 84.22 ± 1.40 | 0.7962 ± 0.0091 | 84.09 ± 1.52 |

### experiments/rdid_v2_mosei

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| r_only | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4783 | 88.87 | 0.7937 | 88.74 |
| r_only | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.4887 | 87.97 | 0.8201 | 87.83 |
| soft_ru_a025 | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.5030 | 88.32 | 0.7813 | 88.02 |
| soft_ru_a025 | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5108 | 86.68 | 0.8134 | 86.38 |
| soft_ru_a050 | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4723 | 89.08 | 0.7907 | 88.94 |
| soft_ru_a050 | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.4785 | 87.34 | 0.8204 | 87.22 |
| soft_ru_a075 | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4718 | 89.15 | 0.7936 | 89.01 |
| soft_ru_a075 | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.4837 | 87.97 | 0.8175 | 87.82 |
| u_only | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4638 | 88.60 | 0.7957 | 88.60 |
| u_only | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.4767 | 87.95 | 0.8249 | 87.96 |

### experiments/tav_epoch_checkpoints_v1

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| M0 | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.5531 | 85.88 | 0.7231 | 85.96 |
| M0 | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5601 | 85.55 | 0.7618 | 85.58 |
| M1 | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.5377 | 85.81 | 0.7363 | 85.61 |
| M1 | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5366 | 86.38 | 0.7797 | 86.26 |
| M3 | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4990 | 88.53 | 0.7923 | 88.55 |
| M3 | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5120 | 87.34 | 0.8115 | 87.36 |
| M4 | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.5104 | 88.46 | 0.7877 | 88.33 |
| M4 | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5173 | 87.64 | 0.8132 | 87.48 |
| M6 | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.5201 | 88.53 | 0.7939 | 88.28 |
| M6 | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5321 | 86.65 | 0.8132 | 86.36 |

### experiments/tav_main_v1

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| M0 | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5339 ± 0.0168 | 86.11 ± 0.24 | 0.7319 ± 0.0076 | 86.08 ± 0.15 |
| M0 | mosei | full | valid_acc2_nonzero | test | 2026,42 (2/3) | 0.5318 ± 0.0030 | 85.70 ± 0.21 | 0.7710 ± 0.0003 | 85.61 ± 0.15 |
| M1 | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5241 ± 0.0129 | 86.39 ± 0.51 | 0.7407 ± 0.0039 | 86.33 ± 0.64 |
| M1 | mosei | full | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| M3 | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4792 ± 0.0174 | 88.66 ± 0.24 | 0.7904 ± 0.0046 | 88.62 ± 0.21 |
| M3 | mosei | full | valid_acc2_nonzero | test | 42 (1/3) | 0.4778 | 87.97 | 0.8171 | 87.93 |
| M4 | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4852 ± 0.0222 | 88.83 ± 0.33 | 0.7933 ± 0.0049 | 88.72 ± 0.36 |
| M4 | mosei | full | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| M6 | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4863 ± 0.0293 | 88.80 ± 0.48 | 0.7961 ± 0.0041 | 88.71 ± 0.61 |
| M6 | mosei | full | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |

### experiments/uniform_followup_v1/mosei

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| adapted_student | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.5106 | 87.27 | 0.7575 | 87.34 |
| adapted_student | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5284 | 85.64 | 0.7830 | 85.71 |
| ensemble_full | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4967 | 88.94 | 0.7826 | 88.94 |
| ensemble_full | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5097 | 87.01 | 0.8048 | 87.00 |
| first_order_interaction | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4694 | 88.73 | 0.7985 | 88.56 |
| first_order_interaction | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.4891 | 87.26 | 0.8186 | 87.09 |
| subset7 | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.5018 | 88.53 | 0.7833 | 88.35 |
| subset7 | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.5090 | 87.34 | 0.8114 | 87.14 |

### experiments/uniform_followup_v1/mosei_qiii_baselines_20260920

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| cmad_cafd | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4722 | 89.01 | 0.7883 | 88.91 |
| cmad_cafd | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.4838 | 87.75 | 0.8167 | 87.65 |
| ea_kd | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4798 | 88.60 | 0.7848 | 88.55 |
| ea_kd | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.4864 | 87.62 | 0.8137 | 87.57 |
| projector | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4713 | 88.53 | 0.7939 | 88.37 |
| projector | mosei | full | valid_acc2_nonzero | test | 13 (1/1) | 0.4787 | 87.64 | 0.8241 | 87.49 |

### experiments/uniform_main_v1/mosi

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| adapted_student | mosi | full | valid_acc2_nonzero | validation | 13 (1/1) | 1.0161 | 85.19 | 0.7281 | 85.11 |
| adapted_student | mosi | full | valid_acc2_nonzero | test | 13 (1/1) | 0.8879 | 82.01 | 0.7394 | 82.08 |
| cmad_cafd | mosi | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.7618 | 84.72 | 0.7791 | 84.75 |
| cmad_cafd | mosi | full | valid_acc2_nonzero | test | 13 (1/1) | 0.7209 | 85.06 | 0.7883 | 84.96 |
| ea_kd | mosi | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.7967 | 87.04 | 0.7887 | 87.09 |
| ea_kd | mosi | full | valid_acc2_nonzero | test | 13 (1/1) | 0.7310 | 83.54 | 0.7875 | 83.40 |
| ensemble_full | mosi | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.8050 | 86.57 | 0.8004 | 86.63 |
| ensemble_full | mosi | full | valid_acc2_nonzero | test | 13 (1/1) | 0.7160 | 84.30 | 0.8010 | 84.09 |
| first_order_interaction | mosi | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.7847 | 86.57 | 0.7579 | 86.60 |
| first_order_interaction | mosi | full | valid_acc2_nonzero | test | 13 (1/1) | 0.7111 | 85.06 | 0.7891 | 84.96 |
| full_kd | mosi | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.7661 | 86.11 | 0.7832 | 86.16 |
| full_kd | mosi | full | valid_acc2_nonzero | test | 13 (1/1) | 0.7138 | 84.45 | 0.7929 | 84.41 |
| projector | mosi | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.7427 | 87.04 | 0.8090 | 87.00 |
| projector | mosi | full | valid_acc2_nonzero | test | 13 (1/1) | 0.7615 | 84.60 | 0.7887 | 84.69 |
| subset7 | mosi | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.8531 | 85.65 | 0.7736 | 85.72 |
| subset7 | mosi | full | valid_acc2_nonzero | test | 13 (1/1) | 0.7585 | 83.38 | 0.7822 | 83.22 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.7778 | 87.04 | 0.7808 | 87.08 |
| uniform_interaction | mosi | full | valid_acc2_nonzero | test | 13 (1/1) | 0.6815 | 84.76 | 0.8168 | 84.61 |

### experiments/video_source_v2

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| video_lora | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4792 ± 0.0174 | 88.66 ± 0.24 | 0.7904 ± 0.0046 | 88.62 ± 0.21 |
| video_lora | mosei | full | valid_acc2_nonzero | test | 42 (1/3) | 0.4778 | 87.97 | 0.8171 | 87.93 |

### student

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| fullscale_ensemble_pair | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5277 ± 0.0146 | 86.39 ± 0.11 | 0.7473 ± 0.0031 | — |
| fullscale_ensemble_pair | mosei | full | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| fullscale_full_kd | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5188 ± 0.0166 | 86.37 ± 0.49 | 0.7451 ± 0.0073 | — |
| fullscale_full_kd | mosei | full | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| fullscale_pair_snr | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5156 ± 0.0062 | 86.21 ± 0.46 | 0.7387 ± 0.0046 | 86.10 |
| fullscale_pair_snr | mosei | full | valid_acc2_nonzero | test | 2026 (1/3) | 0.5173 | 86.05 | 0.7830 | 86.01 |
| fullscale_reliability_utility_pair | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5175 ± 0.0121 | 86.30 ± 0.61 | 0.7448 ± 0.0052 | — |
| fullscale_reliability_utility_pair | mosei | full | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| fullscale_selective50_interaction4 | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5608 ± 0.0436 | 85.81 ± 0.39 | 0.7333 ± 0.0080 | — |
| fullscale_selective50_interaction4 | mosei | full | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| fullscale_student | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5572 ± 0.0144 | 85.23 ± 0.46 | 0.7143 ± 0.0144 | — |
| fullscale_student | mosei | full | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| fullscale_subset4 | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5231 ± 0.0117 | 86.30 ± 0.67 | 0.7355 ± 0.0126 | 85.67 |
| fullscale_subset4 | mosei | full | valid_acc2_nonzero | test | 13 (1/3) | 0.5339 | 85.77 | 0.7750 | 85.87 |
| fullscale_utility_pair | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.5231 | 86.09 | 0.7418 | — |
| fullscale_utility_pair | mosei | full | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| polarity_e1 | mosei | full | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.5535 ± 0.0480 | 86.61 ± 0.64 | 0.7208 ± 0.0266 | — |
| polarity_e1 | mosei | full | valid_acc2_nonzero | test | — (0/2) | — | — | — | — |
| polarity_e2 | mosei | full | valid_acc2_nonzero | validation | 42 (1/1) | 0.5536 | 86.02 | 0.7483 | — |
| polarity_e2 | mosei | full | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| polarity_e2_detached | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.5490 | 86.16 | 0.7518 | — |
| polarity_e2_detached | mosei | full | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| stage_d_c0_task_only_lora | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5278 ± 0.0245 | 87.20 ± 0.72 | 0.7600 ± 0.0036 | 87.86 |
| stage_d_c0_task_only_lora | mosei | full | valid_acc2_nonzero | test | 42 (1/3) | 0.5119 | 86.65 | 0.7941 | 86.51 |
| stage_d_c2_ensemble_pair_lora | mosei | full | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.4732 ± 0.0023 | 89.36 ± 0.30 | 0.7908 ± 0.0004 | — |
| stage_d_c2_ensemble_pair_lora | mosei | full | valid_acc2_nonzero | test | — (0/2) | — | — | — | — |
| stage_d_cminus1_online_frozen_full_kd | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5477 ± 0.0361 | 86.58 ± 0.55 | 0.7431 ± 0.0016 | — |
| stage_d_cminus1_online_frozen_full_kd | mosei | full | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| stage_d_cv2_c1_full_kd_lora_tempfix | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5152 ± 0.0813 | 88.85 ± 0.35 | 0.7816 ± 0.0239 | 89.17 |
| stage_d_cv2_c1_full_kd_lora_tempfix | mosei | full | valid_acc2_nonzero | test | 2026 (1/3) | 0.4782 | 87.84 | 0.8213 | 87.79 |
| stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean | mosei | full | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.4758 ± 0.0039 | 88.42 ± 0.05 | 0.7924 ± 0.0007 | — |
| stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean | mosei | full | valid_acc2_nonzero | test | — (0/2) | — | — | — | — |
| stage_d_cv2_cminus1_online_frozen_full_kd_tempfix | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.5205 ± 0.0132 | 86.65 ± 0.57 | 0.7464 ± 0.0059 | 86.82 |
| stage_d_cv2_cminus1_online_frozen_full_kd_tempfix | mosei | full | valid_acc2_nonzero | test | 13 (1/3) | 0.5277 | 85.88 | 0.7791 | 85.89 |
| stage_d_d1_full_kd_lora | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4959 ± 0.0323 | 89.11 ± 0.32 | 0.7848 ± 0.0110 | — |
| stage_d_d1_full_kd_lora | mosei | full | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| stage_d_d2_ru_lora | mosei | full | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.4795 ± 0.0085 | 88.41 ± 0.14 | 0.7847 ± 0.0115 | — |
| stage_d_d2_ru_lora | mosei | full | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| stage_d_d3_full_kd_hidden | mosei | full | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.5319 ± 0.0258 | 85.95 ± 0.20 | 0.7267 ± 0.0148 | — |
| stage_d_d3_full_kd_hidden | mosei | full | valid_acc2_nonzero | test | — (0/2) | — | — | — | — |
| a10_random_nonorthogonal_cseed100 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6701 | 80.95 | 0.6206 | 80.95 |
| a10_random_nonorthogonal_cseed100 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| a3_zscore_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 2026,2027,2028 (3/3) | 0.5991 ± 0.0183 | 84.92 ± 1.37 | 0.7011 ± 0.0299 | 85.73 ± 0.02 |
| a3_zscore_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| a4_inverse_variance_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6816 | 80.95 | 0.6853 | — |
| a4_inverse_variance_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| a5_snr_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6664 | 80.95 | 0.6735 | 80.82 |
| a5_snr_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| a6_selective25_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6842 | 83.33 | 0.6782 | — |
| a6_selective25_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| a6_selective50_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 2026,2027,2028 (3/3) | 0.7049 ± 0.0281 | 81.75 ± 3.64 | 0.6467 ± 0.0330 | — |
| a6_selective50_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| a6_selective75_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6362 | 83.33 | 0.6687 | 83.36 |
| a6_selective75_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| a7_pair_raw | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.7587 | 80.95 | 0.5543 | — |
| a7_pair_raw | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| a7_pair_snr | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6675 | 85.71 | 0.6923 | — |
| a7_pair_snr | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| a8_triple_raw | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6429 | 85.71 | 0.6370 | — |
| a8_triple_raw | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| a9_random_orthogonal_cseed100 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.7277 | 80.95 | 0.5908 | — |
| a9_random_orthogonal_cseed100 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| a9_random_orthogonal_cseed200 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.7117 | 83.33 | 0.6913 | — |
| a9_random_orthogonal_cseed200 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| a9_random_orthogonal_cseed300 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6486 | 83.33 | 0.6608 | 83.29 |
| a9_random_orthogonal_cseed300 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| baseline_benchmark500 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.7309 | 78.57 | 0.6430 | — |
| baseline_benchmark500 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| full_kd_benchmark500 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.7556 | 85.71 | 0.6234 | — |
| full_kd_benchmark500 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| high_order_interaction_benchmark500 | mosei | pilot | valid_acc2_nonzero | validation | 2026,2027,2028 (3/3) | 0.6469 ± 0.0494 | 80.95 ± 2.38 | 0.6650 ± 0.0805 | 80.90 ± 3.38 |
| high_order_interaction_benchmark500 | mosei | pilot | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| mobius_full_benchmark500 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6586 | 85.71 | 0.6616 | — |
| mobius_full_benchmark500 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| subset_value_4_benchmark500 | mosei | pilot | valid_acc2_nonzero | validation | 2026,2027,2028 (3/3) | 0.6944 ± 0.0729 | 84.13 ± 1.37 | 0.6703 ± 0.0285 | — |
| subset_value_4_benchmark500 | mosei | pilot | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| subset_value_benchmark500 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6493 | 85.71 | 0.6490 | — |
| subset_value_benchmark500 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2_inverse_variance | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.7011 | 83.33 | 0.6061 | — |
| v2_inverse_variance | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2_pair_raw | mosei | pilot | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.6878 ± 0.0624 | 85.71 ± 3.37 | 0.6372 ± 0.0570 | 88.12 |
| v2_pair_raw | mosei | pilot | valid_acc2_nonzero | test | — (0/2) | — | — | — | — |
| v2_pair_snr | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.7223 | 80.95 | 0.6017 | — |
| v2_pair_snr | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2_random_orthogonal_cseed100 | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.7073 | 83.33 | 0.6043 | — |
| v2_random_orthogonal_cseed100 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2_raw_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.7355 ± 0.0450 | 82.14 ± 1.68 | 0.5846 ± 0.0039 | — |
| v2_raw_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/2) | — | — | — | — |
| v2_repro_pair_raw_gpu0 | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.7963 | 66.67 | 0.5230 | 64.78 |
| v2_repro_pair_raw_gpu0 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2_repro_pair_raw_gpu1 | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.7963 | 66.67 | 0.5230 | 64.78 |
| v2_repro_pair_raw_gpu1 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2_selective50 | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.6414 | 83.33 | 0.6539 | — |
| v2_selective50 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2_snr_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.6526 ± 0.0573 | 84.52 ± 1.68 | 0.6571 ± 0.0150 | 85.71 |
| v2_snr_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/2) | — | — | — | — |
| v2_subset4 | mosei | pilot | valid_acc2_nonzero | validation | 13,42 (2/2) | 0.7257 ± 0.0869 | 82.14 ± 5.05 | 0.5742 ± 0.0708 | — |
| v2_subset4 | mosei | pilot | valid_acc2_nonzero | test | — (0/2) | — | — | — | — |
| v2_triple_raw | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.6823 | 80.95 | 0.6234 | 80.82 |
| v2_triple_raw | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2_zscore_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.8320 | 78.57 | 0.5819 | — |
| v2_zscore_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2d_inverse_variance_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.6547 | 85.71 | 0.6425 | 85.75 |
| v2d_inverse_variance_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2d_pair_raw | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.7043 | 83.33 | 0.6105 | — |
| v2d_pair_raw | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2d_pair_snr | mosei | pilot | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.6616 ± 0.0240 | 84.13 ± 1.37 | 0.6295 ± 0.0439 | — |
| v2d_pair_snr | mosei | pilot | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| v2d_random_nonorthogonal | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.6936 | 85.71 | 0.6386 | — |
| v2d_random_nonorthogonal | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2d_random_orthogonal | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.7188 | 83.33 | 0.5881 | — |
| v2d_random_orthogonal | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2d_raw_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.7003 ± 0.0159 | 83.33 ± 0.00 | 0.6233 ± 0.0687 | — |
| v2d_raw_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| v2d_selective50_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.6437 ± 0.0337 | 84.92 ± 1.37 | 0.6776 ± 0.0279 | — |
| v2d_selective50_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| v2d_snr_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.7487 | 83.33 | 0.5840 | — |
| v2d_snr_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2d_subset4 | mosei | pilot | valid_acc2_nonzero | validation | 13,2026,42 (3/3) | 0.6859 ± 0.0530 | 84.13 ± 1.37 | 0.6338 ± 0.0589 | — |
| v2d_subset4 | mosei | pilot | valid_acc2_nonzero | test | — (0/3) | — | — | — | — |
| v2d_triple_raw | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.6625 | 85.71 | 0.6459 | — |
| v2d_triple_raw | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| v2d_zscore_interaction4 | mosei | pilot | valid_acc2_nonzero | validation | 42 (1/1) | 0.6762 | 83.33 | 0.6069 | — |
| v2d_zscore_interaction4 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| stage_d_batch8_probe | mosei | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 3.1667 | 38.46 | -0.2428 | 21.37 |
| stage_d_batch8_probe | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| stage_d_c0_task_only_lora_smoke | mosei | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 2.8984 | 0.00 | -1.0000 | 0.00 |
| stage_d_c0_task_only_lora_smoke | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| stage_d_c2_ensemble_pair_lora_smoke | mosei | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 3.3828 | 0.00 | -1.0000 | 0.00 |
| stage_d_c2_ensemble_pair_lora_smoke | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| stage_d_cminus1_online_frozen_full_kd_smoke | mosei | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 2.9844 | 0.00 | -1.0000 | 0.00 |
| stage_d_cminus1_online_frozen_full_kd_smoke | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| stage_d_smoke_full_kd_lora | mosei | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 0.5527 | 100.00 | -1.0000 | 100.00 |
| stage_d_smoke_full_kd_lora | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| stage_d_smoke_hidden_kd | mosei | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 2.8333 | 61.54 | 0.2428 | 46.89 |
| stage_d_smoke_hidden_kd | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| stage_d_smoke_ru_lora | mosei | smoke/preflight | valid_acc2_nonzero | validation | 42 (1/1) | 2.5000 | 100.00 | — | — |
| stage_d_smoke_ru_lora | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| video_all_layers_preflight_20260911 | mosei | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 2.8854 | 57.14 | — | 41.56 |
| video_all_layers_preflight_20260911 | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |

### student/tuning

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| interaction4_lambda0.25 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.7150 | 83.33 | 0.6315 | — |
| interaction4_lambda0.25 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| interaction4_lambda0.5 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6936 | 88.10 | 0.6022 | — |
| interaction4_lambda0.5 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| interaction4_lambda2 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.7370 | 85.71 | 0.5614 | — |
| interaction4_lambda2 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| subset4_lambda0.25 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6776 | 83.33 | 0.6048 | — |
| subset4_lambda0.25 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| subset4_lambda0.5 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6315 | 90.48 | 0.6567 | — |
| subset4_lambda0.5 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| subset4_lambda2 | mosei | pilot | valid_acc2_nonzero | validation | 2026 (1/1) | 0.6534 | 88.10 | 0.6450 | — |
| subset4_lambda2 | mosei | pilot | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |

### student/video_adaptation_smoke_v2

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| frozen_video | mosei | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 3.3828 | 0.00 | -1.0000 | 0.00 |
| frozen_video | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| ta_only | mosei | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 2.5000 | 100.00 | — | 100.00 |
| ta_only | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| video_lora | mosei | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 3.4688 | 0.00 | — | 0.00 |
| video_lora | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| video_lora_resume | mosei | smoke/preflight | valid_acc2_nonzero | validation | 13 (1/1) | 3.4688 | 0.00 | — | 0.00 |
| video_lora_resume | mosei | smoke/preflight | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |

### student/video_adaptation_v1

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| frozen_video | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4750 | 88.73 | 0.7920 | 88.66 |
| frozen_video | mosei | full | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| ta_only | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4787 | 88.60 | 0.7801 | 88.58 |
| ta_only | mosei | full | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
| video_lora | mosei | full | valid_acc2_nonzero | validation | 13 (1/1) | 0.4736 | 88.32 | 0.7917 | 88.33 |
| video_lora | mosei | full | valid_acc2_nonzero | test | — (0/1) | — | — | — | — |
