# best_valid_mae：全部实验汇总

[口径、缺失项说明及来源审计](README.md)。所有表均采用同一选中 epoch 的完整指标，± 为样本标准差。

## historical_mosi_seven

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| ensemble_full | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7412 ± 0.0390 | 84.26 ± 2.02 | 0.7888 ± 0.0241 | 84.12 ± 2.11 |
| ensemble_full | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7300 ± 0.0125 | 85.11 ± 0.09 | 0.7986 ± 0.0056 | 85.17 ± 0.09 |
| first_order_interaction | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7650 ± 0.0147 | 83.80 ± 0.46 | 0.7750 ± 0.0045 | 83.64 ± 0.66 |
| first_order_interaction | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7235 ± 0.0450 | 84.25 ± 1.50 | 0.7998 ± 0.0134 | 84.27 ± 1.48 |
| first_second_order_interaction | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7589 ± 0.0244 | 84.88 ± 0.96 | 0.7769 ± 0.0119 | 84.70 ± 0.94 |
| first_second_order_interaction | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7345 ± 0.0380 | 84.20 ± 1.48 | 0.7891 ± 0.0192 | 84.25 ± 1.47 |
| full_kd | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7522 ± 0.0192 | 84.41 ± 0.71 | 0.7877 ± 0.0099 | 84.41 ± 0.70 |
| full_kd | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7316 ± 0.0056 | 83.99 ± 0.70 | 0.7847 ± 0.0111 | 84.01 ± 0.71 |
| random_orthogonal | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7564 ± 0.0217 | 85.65 ± 0.93 | 0.7797 ± 0.0203 | 85.61 ± 0.88 |
| random_orthogonal | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7169 ± 0.0154 | 84.71 ± 0.89 | 0.7972 ± 0.0087 | 84.73 ± 0.87 |
| subset7 | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7702 ± 0.0136 | 84.10 ± 0.27 | 0.7812 ± 0.0045 | 83.91 ± 0.36 |
| subset7 | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7576 ± 0.0378 | 84.50 ± 1.62 | 0.7888 ± 0.0166 | 84.58 ± 1.59 |
| uniform_interaction | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7721 ± 0.0108 | 84.10 ± 0.96 | 0.7849 ± 0.0075 | 83.94 ± 0.96 |
| uniform_interaction | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7388 ± 0.0141 | 84.86 ± 0.75 | 0.7892 ± 0.0170 | 84.93 ± 0.76 |

## historical_mosei_seven

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| ensemble_full | mosei | full | valid_mae | validation | 13 (1/1) | 0.4718 | 88.04 | 0.7921 | 88.08 |
| ensemble_full | mosei | full | valid_mae | test | 13 (1/1) | 0.4869 | 87.84 | 0.8172 | 87.84 |
| first_order_interaction | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4653 ± 0.0017 | 87.71 ± 0.84 | 0.7973 ± 0.0002 | 87.75 ± 0.77 |
| first_order_interaction | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4814 ± 0.0037 | 87.19 ± 0.73 | 0.8186 ± 0.0020 | 87.21 ± 0.67 |
| first_second_order_interaction | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4675 ± 0.0017 | 88.22 ± 0.47 | 0.7928 ± 0.0025 | 88.23 ± 0.40 |
| first_second_order_interaction | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4780 ± 0.0007 | 87.47 ± 0.21 | 0.8215 ± 0.0023 | 87.46 ± 0.14 |
| full_kd | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4671 ± 0.0038 | 88.02 ± 0.66 | 0.7901 ± 0.0075 | 88.01 ± 0.66 |
| full_kd | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4798 ± 0.0019 | 87.28 ± 0.61 | 0.8169 ± 0.0042 | 87.28 ± 0.57 |
| random_orthogonal | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4674 ± 0.0025 | 87.99 ± 0.38 | 0.7902 ± 0.0050 | 88.01 ± 0.36 |
| random_orthogonal | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4767 ± 0.0031 | 87.62 ± 0.77 | 0.8219 ± 0.0037 | 87.62 ± 0.75 |
| subset7 | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4653 ± 0.0040 | 88.43 ± 0.53 | 0.7944 ± 0.0048 | 88.35 ± 0.55 |
| subset7 | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4780 ± 0.0044 | 87.61 ± 0.33 | 0.8196 ± 0.0039 | 87.52 ± 0.32 |
| uniform_interaction | mosei | full | valid_mae | validation | 13,42,2026 (3/3) | 0.4698 ± 0.0056 | 87.88 ± 0.11 | 0.7968 ± 0.0040 | 87.84 ± 0.15 |
| uniform_interaction | mosei | full | valid_mae | test | 13,42,2026 (3/3) | 0.4762 ± 0.0045 | 87.78 ± 0.44 | 0.8237 ± 0.0013 | 87.73 ± 0.42 |

## mosi_research_14

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| dgl_icd | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7775 ± 0.0160 | 82.72 ± 1.63 | 0.7770 ± 0.0137 | 82.56 ± 1.78 |
| dgl_icd | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.8083 ± 0.0130 | 82.62 ± 0.26 | 0.7538 ± 0.0093 | 82.69 ± 0.31 |
| ensemble_source | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7505 ± 0.0230 | 85.65 ± 0.00 | 0.7919 ± 0.0005 | 85.50 ± 0.04 |
| ensemble_source | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7410 ± 0.0293 | 84.65 ± 0.63 | 0.7965 ± 0.0020 | 84.73 ± 0.62 |
| explicit_full_kd | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7681 ± 0.0251 | 83.18 ± 0.71 | 0.7715 ± 0.0153 | 83.06 ± 0.75 |
| explicit_full_kd | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7194 ± 0.0239 | 85.01 ± 0.84 | 0.7944 ± 0.0039 | 85.07 ± 0.81 |
| explicit_uniform_interaction | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7697 ± 0.0122 | 84.10 ± 1.93 | 0.7904 ± 0.0079 | 83.88 ± 1.92 |
| explicit_uniform_interaction | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7497 ± 0.0465 | 85.11 ± 1.22 | 0.8056 ± 0.0093 | 85.19 ± 1.20 |
| fusion_only | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7420 ± 0.0297 | 84.10 ± 1.75 | 0.7898 ± 0.0144 | 83.96 ± 1.84 |
| fusion_only | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7339 ± 0.0245 | 84.55 ± 2.41 | 0.7886 ± 0.0038 | 84.57 ± 2.36 |
| late_decay | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7659 ± 0.0239 | 83.80 ± 2.82 | 0.7945 ± 0.0070 | 83.57 ± 3.08 |
| late_decay | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7357 ± 0.0344 | 85.01 ± 1.14 | 0.7960 ± 0.0119 | 85.07 ± 1.10 |
| order_routed | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7558 ± 0.0170 | 84.26 ± 1.67 | 0.8003 ± 0.0040 | 84.14 ± 1.71 |
| order_routed | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7441 ± 0.0118 | 84.04 ± 0.84 | 0.7910 ± 0.0055 | 84.12 ± 0.83 |
| scrambled_routed | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7619 ± 0.0150 | 86.11 ± 0.46 | 0.7865 ± 0.0132 | 86.06 ± 0.50 |
| scrambled_routed | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7448 ± 0.0251 | 84.86 ± 0.49 | 0.7924 ± 0.0098 | 84.88 ± 0.52 |
| target_uniform | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7411 ± 0.0179 | 84.10 ± 0.71 | 0.7797 ± 0.0151 | 84.05 ± 0.69 |
| target_uniform | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7103 ± 0.0240 | 85.37 ± 0.00 | 0.7992 ± 0.0058 | 85.39 ± 0.04 |
| target_variance | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7494 ± 0.0278 | 85.80 ± 0.71 | 0.7887 ± 0.0283 | 85.74 ± 0.65 |
| target_variance | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7002 ± 0.0194 | 84.96 ± 0.09 | 0.8060 ± 0.0050 | 85.00 ± 0.04 |
| uniform_interaction | mosi | full | valid_mae | validation | 2026,13,42 (3/3) | 0.7595 ± 0.0185 | 84.88 ± 0.96 | 0.7924 ± 0.0077 | 84.82 ± 1.05 |
| uniform_interaction | mosi | full | valid_mae | test | 2026,13,42 (3/3) | 0.7332 ± 0.0167 | 84.30 ± 0.53 | 0.7873 ± 0.0048 | 84.35 ± 0.54 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7536 ± 0.0344 | 84.88 ± 0.53 | 0.7916 ± 0.0198 | 84.76 ± 0.60 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7257 ± 0.0084 | 84.65 ± 0.35 | 0.7941 ± 0.0011 | 84.71 ± 0.33 |
| uniform_interaction_gradient_projected | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7276 ± 0.0324 | 84.88 ± 0.71 | 0.8100 ± 0.0126 | 84.81 ± 0.69 |
| uniform_interaction_gradient_projected | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7089 ± 0.0109 | 85.06 ± 0.40 | 0.8048 ± 0.0055 | 85.08 ± 0.40 |
| uniform_interaction_shared_dropout | mosi | full | valid_mae | validation | 13,42,2026 (3/3) | 0.7645 ± 0.0129 | 83.18 ± 0.96 | 0.7855 ± 0.0087 | 82.95 ± 1.04 |
| uniform_interaction_shared_dropout | mosi | full | valid_mae | test | 13,42,2026 (3/3) | 0.7433 ± 0.0296 | 84.76 ± 0.40 | 0.7970 ± 0.0069 | 84.84 ± 0.40 |

## mosei_extensions_v3

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| uniform_interaction | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4680 ± 0.0039 | 88.32 ± 0.66 | 0.7973 ± 0.0032 | 88.28 ± 0.62 |
| uniform_interaction | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4762 ± 0.0045 | 87.65 ± 0.29 | 0.8244 ± 0.0002 | 87.61 ± 0.30 |
| uniform_interaction_gradient_projected | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4666 ± 0.0028 | 88.02 ± 1.61 | 0.7940 ± 0.0052 | 88.01 ± 1.53 |
| uniform_interaction_gradient_projected | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4769 ± 0.0026 | 87.91 ± 0.34 | 0.8216 ± 0.0023 | 87.89 ± 0.32 |
| uniform_interaction_shared_dropout | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4693 ± 0.0009 | 87.60 ± 0.76 | 0.7933 ± 0.0034 | 87.70 ± 0.69 |
| uniform_interaction_shared_dropout | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4833 ± 0.0016 | 86.86 ± 0.36 | 0.8192 ± 0.0021 | 86.95 ± 0.32 |

## 全部历史实验线（包括 pilot 与 smoke）

### experiments/decision_icd_v1

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| smoke_interaction | mosi | smoke/preflight | valid_mae | validation | 13 (1/1) | 4.2000 | 0.00 | 0.1111 | 0.00 |
| smoke_interaction | mosi | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |

### experiments/decision_icd_v1/mosi

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| icd_decision_full | mosi | full | valid_mae | validation | 13,42 (2/2) | 0.7669 ± 0.0307 | 85.19 ± 1.31 | 0.7762 ± 0.0329 | 85.08 ± 1.27 |
| icd_decision_full | mosi | full | valid_mae | test | 13,42 (2/2) | 0.7201 ± 0.0143 | 83.99 ± 0.22 | 0.7989 ± 0.0109 | 84.05 ± 0.21 |
| icd_decision_interaction | mosi | full | valid_mae | validation | 13,42 (2/2) | 0.7862 ± 0.0110 | 84.72 ± 0.65 | 0.7628 ± 0.0063 | 84.70 ± 0.69 |
| icd_decision_interaction | mosi | full | valid_mae | test | 13,42 (2/2) | 0.7217 ± 0.0174 | 84.98 ± 0.54 | 0.7869 ± 0.0006 | 84.99 ± 0.51 |
| icd_decision_none | mosi | full | valid_mae | validation | 13,42 (2/2) | 0.7646 ± 0.0229 | 84.72 ± 1.31 | 0.7925 ± 0.0109 | 84.62 ± 1.41 |
| icd_decision_none | mosi | full | valid_mae | test | 13,42 (2/2) | 0.7423 ± 0.0076 | 84.45 ± 0.65 | 0.7858 ± 0.0057 | 84.52 ± 0.64 |
| icd_decision_subset | mosi | full | valid_mae | validation | 13,42 (2/2) | 0.7750 ± 0.0097 | 84.03 ± 0.33 | 0.7798 ± 0.0099 | 84.01 ± 0.35 |
| icd_decision_subset | mosi | full | valid_mae | test | 13,42 (2/2) | 0.7207 ± 0.0333 | 84.45 ± 0.00 | 0.8018 ± 0.0100 | 84.44 ± 0.01 |

### experiments/equal_modality_extensions_v3/mosei

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| uniform_interaction | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4680 ± 0.0039 | 88.32 ± 0.66 | 0.7973 ± 0.0032 | 88.28 ± 0.62 |
| uniform_interaction | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4762 ± 0.0045 | 87.65 ± 0.29 | 0.8244 ± 0.0002 | 87.61 ± 0.30 |
| uniform_interaction_gradient_projected | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4666 ± 0.0028 | 88.02 ± 1.61 | 0.7940 ± 0.0052 | 88.01 ± 1.53 |
| uniform_interaction_gradient_projected | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4769 ± 0.0026 | 87.91 ± 0.34 | 0.8216 ± 0.0023 | 87.89 ± 0.32 |
| uniform_interaction_shared_dropout | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4693 ± 0.0009 | 87.60 ± 0.76 | 0.7933 ± 0.0034 | 87.70 ± 0.69 |
| uniform_interaction_shared_dropout | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4833 ± 0.0016 | 86.86 ± 0.36 | 0.8192 ± 0.0021 | 86.95 ± 0.32 |

### experiments/interaction_evidence_v1/mosei

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| first_second_order_interaction | mosei | full | valid_mae | validation | 13 (1/1) | 0.4660 | 88.39 | 0.7905 | 88.41 |
| first_second_order_interaction | mosei | full | valid_mae | test | 13 (1/1) | 0.4788 | 87.42 | 0.8192 | 87.43 |
| random_orthogonal | mosei | full | valid_mae | validation | 13 (1/1) | 0.4665 | 87.55 | 0.7917 | 87.59 |
| random_orthogonal | mosei | full | valid_mae | test | 13 (1/1) | 0.4783 | 87.04 | 0.8202 | 87.07 |

### experiments/interaction_evidence_v1/mosi

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| first_second_order_interaction | mosi | full | valid_mae | validation | 13 (1/1) | 0.7642 | 85.65 | 0.7761 | 85.38 |
| first_second_order_interaction | mosi | full | valid_mae | test | 13 (1/1) | 0.7326 | 82.93 | 0.7981 | 83.03 |
| random_orthogonal | mosi | full | valid_mae | validation | 13 (1/1) | 0.7566 | 85.65 | 0.7948 | 85.61 |
| random_orthogonal | mosi | full | valid_mae | test | 13 (1/1) | 0.7194 | 85.37 | 0.8046 | 85.40 |

### experiments/mosi_research_v1

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| dgl_icd | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7775 ± 0.0160 | 82.72 ± 1.63 | 0.7770 ± 0.0137 | 82.56 ± 1.78 |
| dgl_icd | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.8083 ± 0.0130 | 82.62 ± 0.26 | 0.7538 ± 0.0093 | 82.69 ± 0.31 |
| ensemble_source | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7505 ± 0.0230 | 85.65 ± 0.00 | 0.7919 ± 0.0005 | 85.50 ± 0.04 |
| ensemble_source | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7410 ± 0.0293 | 84.65 ± 0.63 | 0.7965 ± 0.0020 | 84.73 ± 0.62 |
| explicit_full_kd | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7681 ± 0.0251 | 83.18 ± 0.71 | 0.7715 ± 0.0153 | 83.06 ± 0.75 |
| explicit_full_kd | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7194 ± 0.0239 | 85.01 ± 0.84 | 0.7944 ± 0.0039 | 85.07 ± 0.81 |
| explicit_uniform_interaction | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7697 ± 0.0122 | 84.10 ± 1.93 | 0.7904 ± 0.0079 | 83.88 ± 1.92 |
| explicit_uniform_interaction | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7497 ± 0.0465 | 85.11 ± 1.22 | 0.8056 ± 0.0093 | 85.19 ± 1.20 |
| fusion_only | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7420 ± 0.0297 | 84.10 ± 1.75 | 0.7898 ± 0.0144 | 83.96 ± 1.84 |
| fusion_only | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7339 ± 0.0245 | 84.55 ± 2.41 | 0.7886 ± 0.0038 | 84.57 ± 2.36 |
| late_decay | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7659 ± 0.0239 | 83.80 ± 2.82 | 0.7945 ± 0.0070 | 83.57 ± 3.08 |
| late_decay | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7357 ± 0.0344 | 85.01 ± 1.14 | 0.7960 ± 0.0119 | 85.07 ± 1.10 |
| order_routed | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7558 ± 0.0170 | 84.26 ± 1.67 | 0.8003 ± 0.0040 | 84.14 ± 1.71 |
| order_routed | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7441 ± 0.0118 | 84.04 ± 0.84 | 0.7910 ± 0.0055 | 84.12 ± 0.83 |
| scrambled_routed | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7619 ± 0.0150 | 86.11 ± 0.46 | 0.7865 ± 0.0132 | 86.06 ± 0.50 |
| scrambled_routed | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7448 ± 0.0251 | 84.86 ± 0.49 | 0.7924 ± 0.0098 | 84.88 ± 0.52 |
| target_uniform | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7411 ± 0.0179 | 84.10 ± 0.71 | 0.7797 ± 0.0151 | 84.05 ± 0.69 |
| target_uniform | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7103 ± 0.0240 | 85.37 ± 0.00 | 0.7992 ± 0.0058 | 85.39 ± 0.04 |
| target_variance | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7494 ± 0.0278 | 85.80 ± 0.71 | 0.7887 ± 0.0283 | 85.74 ± 0.65 |
| target_variance | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7002 ± 0.0194 | 84.96 ± 0.09 | 0.8060 ± 0.0050 | 85.00 ± 0.04 |
| uniform_interaction | mosi | full | valid_mae | validation | 2026 (1/1) | 0.7493 | 85.19 | 0.7922 | 85.22 |
| uniform_interaction | mosi | full | valid_mae | test | 2026 (1/1) | 0.7149 | 83.99 | 0.7904 | 84.01 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7536 ± 0.0344 | 84.88 ± 0.53 | 0.7916 ± 0.0198 | 84.76 ± 0.60 |
| uniform_interaction_coordinate_compatible | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7257 ± 0.0084 | 84.65 ± 0.35 | 0.7941 ± 0.0011 | 84.71 ± 0.33 |
| uniform_interaction_gradient_projected | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7276 ± 0.0324 | 84.88 ± 0.71 | 0.8100 ± 0.0126 | 84.81 ± 0.69 |
| uniform_interaction_gradient_projected | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7089 ± 0.0109 | 85.06 ± 0.40 | 0.8048 ± 0.0055 | 85.08 ± 0.40 |
| uniform_interaction_shared_dropout | mosi | full | valid_mae | validation | 13,2026,42 (3/3) | 0.7645 ± 0.0129 | 83.18 ± 0.96 | 0.7855 ± 0.0087 | 82.95 ± 1.04 |
| uniform_interaction_shared_dropout | mosi | full | valid_mae | test | 13,2026,42 (3/3) | 0.7433 ± 0.0296 | 84.76 ± 0.40 | 0.7970 ± 0.0069 | 84.84 ± 0.40 |

### experiments/multiseed_controls_v1/mosei

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| random_orthogonal | mosei | full | valid_mae | validation | 2026,42 (2/2) | 0.4679 ± 0.0033 | 88.21 ± 0.05 | 0.7894 ± 0.0068 | 88.21 ± 0.05 |
| random_orthogonal | mosei | full | valid_mae | test | 2026,42 (2/2) | 0.4759 ± 0.0039 | 87.91 ± 0.84 | 0.8228 ± 0.0047 | 87.89 ± 0.83 |
| subset7 | mosei | full | valid_mae | validation | 2026,42 (2/2) | 0.4635 ± 0.0033 | 88.73 ± 0.10 | 0.7971 ± 0.0017 | 88.65 ± 0.28 |
| subset7 | mosei | full | valid_mae | test | 2026,42 (2/2) | 0.4769 ± 0.0057 | 87.80 ± 0.06 | 0.8210 ± 0.0045 | 87.70 ± 0.07 |

### experiments/multiseed_controls_v1/mosei/followup_order_controls

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| first_order_interaction | mosei | full | valid_mae | validation | 2026,42 (2/2) | 0.4644 ± 0.0001 | 88.11 ± 0.69 | 0.7974 ± 0.0000 | 88.10 ± 0.69 |
| first_order_interaction | mosei | full | valid_mae | test | 2026,42 (2/2) | 0.4793 ± 0.0007 | 87.56 ± 0.47 | 0.8197 ± 0.0007 | 87.54 ± 0.48 |
| first_second_order_interaction | mosei | full | valid_mae | validation | 2026,42 (2/2) | 0.4682 ± 0.0016 | 88.14 ± 0.64 | 0.7940 ± 0.0020 | 88.14 ± 0.51 |
| first_second_order_interaction | mosei | full | valid_mae | test | 2026,42 (2/2) | 0.4776 ± 0.0003 | 87.49 ± 0.29 | 0.8226 ± 0.0019 | 87.48 ± 0.19 |

### experiments/multiseed_controls_v1/mosi

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| first_second_order_interaction | mosi | full | valid_mae | validation | 2026,42 (2/2) | 0.7562 ± 0.0339 | 84.49 ± 0.98 | 0.7772 ± 0.0168 | 84.37 ± 1.05 |
| first_second_order_interaction | mosi | full | valid_mae | test | 2026,42 (2/2) | 0.7354 ± 0.0537 | 84.83 ± 1.40 | 0.7846 ± 0.0249 | 84.86 ± 1.45 |
| full_kd | mosi | full | valid_mae | validation | 2026,42 (2/2) | 0.7627 ± 0.0086 | 84.03 ± 0.33 | 0.7833 ± 0.0090 | 84.02 ± 0.27 |
| full_kd | mosi | full | valid_mae | test | 2026,42 (2/2) | 0.7332 ± 0.0069 | 83.69 ± 0.65 | 0.7790 ± 0.0072 | 83.69 ± 0.64 |
| random_orthogonal | mosi | full | valid_mae | validation | 2026,42 (2/2) | 0.7563 ± 0.0306 | 85.65 ± 1.31 | 0.7722 ± 0.0221 | 85.61 ± 1.24 |
| random_orthogonal | mosi | full | valid_mae | test | 2026,42 (2/2) | 0.7157 ± 0.0215 | 84.38 ± 0.97 | 0.7934 ± 0.0082 | 84.39 ± 0.91 |
| subset7 | mosi | full | valid_mae | validation | 2026,42 (2/2) | 0.7634 ± 0.0097 | 84.03 ± 0.33 | 0.7801 ± 0.0059 | 83.80 ± 0.44 |
| subset7 | mosi | full | valid_mae | test | 2026,42 (2/2) | 0.7593 ± 0.0533 | 84.38 ± 2.26 | 0.7842 ± 0.0207 | 84.45 ± 2.22 |
| uniform_interaction | mosi | full | valid_mae | validation | 2026,42 (2/2) | 0.7781 ± 0.0038 | 83.56 ± 0.33 | 0.7812 ± 0.0052 | 83.40 ± 0.32 |
| uniform_interaction | mosi | full | valid_mae | test | 2026,42 (2/2) | 0.7469 ± 0.0011 | 84.60 ± 0.86 | 0.7795 ± 0.0032 | 84.67 ± 0.86 |

### experiments/multiseed_controls_v1/mosi_additional_controls

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| ensemble_full | mosi | full | valid_mae | validation | 2026,42 (2/2) | 0.7590 ± 0.0339 | 83.10 ± 0.33 | 0.7768 ± 0.0173 | 82.91 ± 0.38 |
| ensemble_full | mosi | full | valid_mae | test | 2026,42 (2/2) | 0.7367 ± 0.0069 | 85.06 ± 0.00 | 0.7989 ± 0.0079 | 85.12 ± 0.04 |
| first_order_interaction | mosi | full | valid_mae | validation | 2026,42 (2/2) | 0.7566 ± 0.0033 | 84.03 ± 0.33 | 0.7773 ± 0.0028 | 84.00 ± 0.34 |
| first_order_interaction | mosi | full | valid_mae | test | 2026,42 (2/2) | 0.6986 ± 0.0184 | 84.68 ± 1.83 | 0.8037 ± 0.0164 | 84.66 ± 1.85 |

### experiments/rdid_v2_mosei

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| r_only | mosei | full | valid_mae | validation | 13 (1/1) | 0.4689 | 88.60 | 0.7927 | 88.55 |
| r_only | mosei | full | valid_mae | test | 13 (1/1) | 0.4823 | 87.59 | 0.8159 | 87.56 |
| soft_ru_a025 | mosei | full | valid_mae | validation | 13 (1/1) | 0.4695 | 86.51 | 0.7934 | 86.67 |
| soft_ru_a025 | mosei | full | valid_mae | test | 13 (1/1) | 0.4830 | 86.74 | 0.8186 | 86.85 |
| soft_ru_a050 | mosei | full | valid_mae | validation | 13 (1/1) | 0.4695 | 88.80 | 0.7911 | 88.72 |
| soft_ru_a050 | mosei | full | valid_mae | test | 13 (1/1) | 0.4796 | 87.97 | 0.8187 | 87.89 |
| soft_ru_a075 | mosei | full | valid_mae | validation | 13 (1/1) | 0.4664 | 88.60 | 0.7982 | 88.59 |
| soft_ru_a075 | mosei | full | valid_mae | test | 13 (1/1) | 0.4891 | 87.62 | 0.8159 | 87.58 |
| u_only | mosei | full | valid_mae | validation | 13 (1/1) | 0.4638 | 88.60 | 0.7957 | 88.60 |
| u_only | mosei | full | valid_mae | test | 13 (1/1) | 0.4767 | 87.95 | 0.8249 | 87.96 |

### experiments/tav_epoch_checkpoints_v1

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| M0 | mosei | full | valid_mae | validation | 13 (1/1) | 0.5333 | 83.24 | 0.7221 | 83.41 |
| M0 | mosei | full | valid_mae | test | 13 (1/1) | 0.5465 | 84.31 | 0.7610 | 84.41 |
| M1 | mosei | full | valid_mae | validation | 13 (1/1) | 0.5166 | 85.05 | 0.7404 | 84.86 |
| M1 | mosei | full | valid_mae | test | 13 (1/1) | 0.5280 | 85.36 | 0.7776 | 85.15 |
| M3 | mosei | full | valid_mae | validation | 13 (1/1) | 0.4713 | 87.27 | 0.7862 | 87.25 |
| M3 | mosei | full | valid_mae | test | 13 (1/1) | 0.4816 | 86.87 | 0.8126 | 86.87 |
| M4 | mosei | full | valid_mae | validation | 13 (1/1) | 0.4746 | 87.76 | 0.7921 | 87.67 |
| M4 | mosei | full | valid_mae | test | 13 (1/1) | 0.4774 | 88.19 | 0.8222 | 88.10 |
| M6 | mosei | full | valid_mae | validation | 13 (1/1) | 0.4638 | 87.13 | 0.7971 | 87.25 |
| M6 | mosei | full | valid_mae | test | 13 (1/1) | 0.4811 | 86.96 | 0.8202 | 87.04 |

### experiments/tav_main_v1

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| M0 | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5273 ± 0.0058 | 85.23 ± 1.73 | 0.7315 ± 0.0082 | 85.23 ± 1.58 |
| M0 | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5367 ± 0.0088 | 85.24 ± 0.82 | 0.7676 ± 0.0058 | 85.21 ± 0.70 |
| M1 | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5142 ± 0.0032 | 84.24 ± 0.74 | 0.7413 ± 0.0022 | 84.31 ± 0.51 |
| M1 | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5273 ± 0.0021 | 84.81 ± 0.53 | 0.7769 ± 0.0042 | 84.83 ± 0.38 |
| M3 | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4671 ± 0.0038 | 88.02 ± 0.66 | 0.7901 ± 0.0075 | 88.01 ± 0.66 |
| M3 | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4798 ± 0.0019 | 87.28 ± 0.61 | 0.8169 ± 0.0042 | 87.28 ± 0.57 |
| M4 | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4698 ± 0.0056 | 87.88 ± 0.11 | 0.7968 ± 0.0040 | 87.84 ± 0.15 |
| M4 | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4762 ± 0.0045 | 87.78 ± 0.44 | 0.8237 ± 0.0013 | 87.73 ± 0.42 |
| M6 | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4640 ± 0.0016 | 88.02 ± 1.13 | 0.7990 ± 0.0033 | 88.06 ± 1.04 |
| M6 | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4819 ± 0.0007 | 87.42 ± 0.63 | 0.8194 ± 0.0007 | 87.44 ± 0.56 |

### experiments/uniform_followup_v1/mosei

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| adapted_student | mosei | full | valid_mae | validation | 13 (1/1) | 0.5106 | 87.27 | 0.7575 | 87.34 |
| adapted_student | mosei | full | valid_mae | test | 13 (1/1) | 0.5284 | 85.64 | 0.7830 | 85.71 |
| ensemble_full | mosei | full | valid_mae | validation | 13 (1/1) | 0.4718 | 88.04 | 0.7921 | 88.08 |
| ensemble_full | mosei | full | valid_mae | test | 13 (1/1) | 0.4869 | 87.84 | 0.8172 | 87.84 |
| first_order_interaction | mosei | full | valid_mae | validation | 13 (1/1) | 0.4673 | 86.93 | 0.7970 | 87.07 |
| first_order_interaction | mosei | full | valid_mae | test | 13 (1/1) | 0.4857 | 86.43 | 0.8164 | 86.54 |
| subset7 | mosei | full | valid_mae | validation | 13 (1/1) | 0.4690 | 87.83 | 0.7891 | 87.76 |
| subset7 | mosei | full | valid_mae | test | 13 (1/1) | 0.4801 | 87.23 | 0.8170 | 87.15 |

### experiments/uniform_followup_v1/mosei_qiii_baselines_20260920

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| cmad_cafd | mosei | full | valid_mae | validation | 13 (1/1) | 0.4672 | 87.34 | 0.7986 | 87.42 |
| cmad_cafd | mosei | full | valid_mae | test | 13 (1/1) | 0.4845 | 86.57 | 0.8139 | 86.60 |
| ea_kd | mosei | full | valid_mae | validation | 13 (1/1) | 0.4794 | 87.07 | 0.7794 | 87.14 |
| ea_kd | mosei | full | valid_mae | test | 13 (1/1) | 0.4811 | 87.01 | 0.8123 | 87.08 |
| projector | mosei | full | valid_mae | validation | 13 (1/1) | 0.4633 | 87.48 | 0.7955 | 87.56 |
| projector | mosei | full | valid_mae | test | 13 (1/1) | 0.4772 | 86.71 | 0.8194 | 86.74 |

### experiments/uniform_main_v1/mosi

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| adapted_student | mosi | full | valid_mae | validation | 13 (1/1) | 0.7884 | 82.87 | 0.7720 | 82.77 |
| adapted_student | mosi | full | valid_mae | test | 13 (1/1) | 0.7972 | 82.47 | 0.7514 | 82.51 |
| cmad_cafd | mosi | full | valid_mae | validation | 13 (1/1) | 0.7446 | 81.48 | 0.7886 | 81.39 |
| cmad_cafd | mosi | full | valid_mae | test | 13 (1/1) | 0.7154 | 85.98 | 0.7919 | 86.02 |
| ea_kd | mosi | full | valid_mae | validation | 13 (1/1) | 0.7744 | 84.72 | 0.7773 | 84.71 |
| ea_kd | mosi | full | valid_mae | test | 13 (1/1) | 0.7518 | 85.52 | 0.7909 | 85.55 |
| ensemble_full | mosi | full | valid_mae | validation | 13 (1/1) | 0.7057 | 86.57 | 0.8128 | 86.54 |
| ensemble_full | mosi | full | valid_mae | test | 13 (1/1) | 0.7167 | 85.21 | 0.7980 | 85.28 |
| first_order_interaction | mosi | full | valid_mae | validation | 13 (1/1) | 0.7817 | 83.33 | 0.7703 | 82.94 |
| first_order_interaction | mosi | full | valid_mae | test | 13 (1/1) | 0.7732 | 83.38 | 0.7920 | 83.47 |
| full_kd | mosi | full | valid_mae | validation | 13 (1/1) | 0.7312 | 85.19 | 0.7965 | 85.19 |
| full_kd | mosi | full | valid_mae | test | 13 (1/1) | 0.7285 | 84.60 | 0.7961 | 84.64 |
| projector | mosi | full | valid_mae | validation | 13 (1/1) | 0.7326 | 83.80 | 0.8008 | 83.66 |
| projector | mosi | full | valid_mae | test | 13 (1/1) | 0.7272 | 84.15 | 0.7884 | 84.24 |
| subset7 | mosi | full | valid_mae | validation | 13 (1/1) | 0.7838 | 84.26 | 0.7833 | 84.11 |
| subset7 | mosi | full | valid_mae | test | 13 (1/1) | 0.7542 | 84.76 | 0.7979 | 84.83 |
| uniform_interaction | mosi | full | valid_mae | validation | 13 (1/1) | 0.7601 | 85.19 | 0.7924 | 85.01 |
| uniform_interaction | mosi | full | valid_mae | test | 13 (1/1) | 0.7225 | 85.37 | 0.8087 | 85.45 |

### experiments/video_source_v2

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| video_lora | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4671 ± 0.0038 | 88.02 ± 0.66 | 0.7901 ± 0.0075 | 88.01 ± 0.66 |
| video_lora | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4798 ± 0.0019 | 87.28 ± 0.61 | 0.8169 ± 0.0042 | 87.28 ± 0.57 |

### student

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| fullscale_ensemble_pair | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5087 ± 0.0014 | 85.47 ± 0.42 | 0.7445 ± 0.0036 | 85.49 ± 0.40 |
| fullscale_ensemble_pair | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5219 ± 0.0046 | 85.38 ± 0.23 | 0.7736 ± 0.0035 | 85.37 ± 0.24 |
| fullscale_full_kd | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5072 ± 0.0046 | 85.30 ± 0.60 | 0.7423 ± 0.0064 | 85.29 ± 0.49 |
| fullscale_full_kd | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5180 ± 0.0018 | 86.04 ± 0.10 | 0.7801 ± 0.0036 | 85.98 ± 0.16 |
| fullscale_pair_snr | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5085 ± 0.0013 | 85.30 ± 0.88 | 0.7408 ± 0.0069 | 85.38 ± 0.90 |
| fullscale_pair_snr | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5191 ± 0.0016 | 85.56 ± 0.42 | 0.7799 ± 0.0033 | 85.56 ± 0.39 |
| fullscale_reliability_utility_pair | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5055 ± 0.0023 | 85.14 ± 0.73 | 0.7451 ± 0.0027 | 85.13 ± 0.62 |
| fullscale_reliability_utility_pair | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5231 ± 0.0014 | 85.56 ± 0.11 | 0.7736 ± 0.0050 | 85.54 ± 0.12 |
| fullscale_selective50_interaction4 | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5061 ± 0.0047 | 85.33 ± 0.55 | 0.7462 ± 0.0030 | 85.23 ± 0.44 |
| fullscale_selective50_interaction4 | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5230 ± 0.0060 | 85.18 ± 0.53 | 0.7746 ± 0.0027 | 85.05 ± 0.45 |
| fullscale_student | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5323 ± 0.0080 | 84.79 ± 0.82 | 0.7180 ± 0.0073 | 84.28 ± 1.07 |
| fullscale_student | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5482 ± 0.0068 | 84.25 ± 0.92 | 0.7548 ± 0.0006 | 83.78 ± 1.17 |
| fullscale_subset4 | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5084 ± 0.0050 | 85.79 ± 0.38 | 0.7465 ± 0.0068 | 85.80 ± 0.40 |
| fullscale_subset4 | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5265 ± 0.0064 | 85.85 ± 0.10 | 0.7767 ± 0.0016 | 85.84 ± 0.15 |
| fullscale_utility_pair | mosei | full | valid_mae | validation | 13 (1/1) | 0.5199 | 84.35 | 0.7484 | 84.50 |
| fullscale_utility_pair | mosei | full | valid_mae | test | 13 (1/1) | 0.5226 | 86.08 | 0.7807 | 86.11 |
| polarity_e1 | mosei | full | valid_mae | validation | 13,42 (2/2) | 0.5145 ± 0.0043 | 85.19 ± 0.89 | 0.7371 ± 0.0014 | — |
| polarity_e1 | mosei | full | valid_mae | test | — (0/2) | — | — | — | — |
| polarity_e2 | mosei | full | valid_mae | validation | 42 (1/1) | 0.5127 | 84.21 | 0.7377 | — |
| polarity_e2 | mosei | full | valid_mae | test | — (0/1) | — | — | — | — |
| polarity_e2_detached | mosei | full | valid_mae | validation | 13 (1/1) | 0.5139 | 83.94 | 0.7482 | — |
| polarity_e2_detached | mosei | full | valid_mae | test | — (0/1) | — | — | — | — |
| stage_d_c0_task_only_lora | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5127 ± 0.0011 | 86.51 ± 1.47 | 0.7561 ± 0.0164 | 86.40 ± 1.30 |
| stage_d_c0_task_only_lora | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5249 ± 0.0114 | 85.64 ± 1.13 | 0.7827 ± 0.0121 | 85.59 ± 0.94 |
| stage_d_c2_ensemble_pair_lora | mosei | full | valid_mae | validation | 13,42 (2/2) | 0.4675 ± 0.0046 | 88.35 ± 0.84 | 0.7939 ± 0.0009 | 88.28 ± 0.86 |
| stage_d_c2_ensemble_pair_lora | mosei | full | valid_mae | test | 13,42 (2/2) | 0.4816 ± 0.0000 | 87.47 ± 0.56 | 0.8202 ± 0.0000 | 87.38 ± 0.59 |
| stage_d_cminus1_online_frozen_full_kd | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5039 ± 0.0019 | 85.91 ± 0.49 | 0.7495 ± 0.0061 | 85.81 ± 0.57 |
| stage_d_cminus1_online_frozen_full_kd | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5207 ± 0.0023 | 85.91 ± 0.17 | 0.7795 ± 0.0022 | 85.84 ± 0.13 |
| stage_d_cv2_c1_full_kd_lora_tempfix | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4662 ± 0.0002 | 87.60 ± 1.42 | 0.7942 ± 0.0016 | 87.64 ± 1.35 |
| stage_d_cv2_c1_full_kd_lora_tempfix | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4840 ± 0.0079 | 87.16 ± 0.79 | 0.8171 ± 0.0040 | 87.19 ± 0.69 |
| stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean | mosei | full | valid_mae | validation | 13,42 (2/2) | 0.4672 ± 0.0020 | 88.25 ± 0.20 | 0.7895 ± 0.0000 | 88.25 ± 0.26 |
| stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean | mosei | full | valid_mae | test | 13,42 (2/2) | 0.4823 ± 0.0056 | 87.22 ± 0.21 | 0.8181 ± 0.0041 | 87.21 ± 0.27 |
| stage_d_cv2_cminus1_online_frozen_full_kd_tempfix | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.5048 ± 0.0005 | 86.11 ± 0.74 | 0.7484 ± 0.0037 | 86.12 ± 0.71 |
| stage_d_cv2_cminus1_online_frozen_full_kd_tempfix | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.5246 ± 0.0030 | 85.62 ± 0.53 | 0.7744 ± 0.0043 | 85.60 ± 0.49 |
| stage_d_d1_full_kd_lora | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4728 ± 0.0057 | 87.39 ± 1.19 | 0.7907 ± 0.0026 | 87.39 ± 1.18 |
| stage_d_d1_full_kd_lora | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4884 ± 0.0063 | 87.13 ± 0.61 | 0.8109 ± 0.0053 | 87.14 ± 0.60 |
| stage_d_d2_ru_lora | mosei | full | valid_mae | validation | 13,2026,42 (3/3) | 0.4646 ± 0.0008 | 87.62 ± 0.14 | 0.7946 ± 0.0043 | 87.63 ± 0.16 |
| stage_d_d2_ru_lora | mosei | full | valid_mae | test | 13,2026,42 (3/3) | 0.4787 ± 0.0041 | 87.38 ± 0.25 | 0.8186 ± 0.0015 | 87.37 ± 0.19 |
| stage_d_d3_full_kd_hidden | mosei | full | valid_mae | validation | 13,42 (2/2) | 0.5114 ± 0.0055 | 85.47 ± 0.79 | 0.7426 ± 0.0055 | 85.28 ± 0.64 |
| stage_d_d3_full_kd_hidden | mosei | full | valid_mae | test | 13,42 (2/2) | 0.5295 ± 0.0044 | 85.51 ± 0.33 | 0.7712 ± 0.0026 | 85.37 ± 0.24 |
| a10_random_nonorthogonal_cseed100 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6701 | 80.95 | 0.6206 | 80.95 |
| a10_random_nonorthogonal_cseed100 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| a3_zscore_interaction4 | mosei | pilot | valid_mae | validation | 2026,2027,2028 (3/3) | 0.5905 ± 0.0222 | 83.33 ± 4.12 | 0.7269 ± 0.0385 | 83.26 ± 4.28 |
| a3_zscore_interaction4 | mosei | pilot | valid_mae | test | — (0/3) | — | — | — | — |
| a4_inverse_variance_interaction4 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6769 | 80.95 | 0.6299 | 80.95 |
| a4_inverse_variance_interaction4 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| a5_snr_interaction4 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6664 | 80.95 | 0.6735 | 80.82 |
| a5_snr_interaction4 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| a6_selective25_interaction4 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6349 | 80.95 | 0.6832 | 81.00 |
| a6_selective25_interaction4 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| a6_selective50_interaction4 | mosei | pilot | valid_mae | validation | 2026,2027,2028 (3/3) | 0.6351 ± 0.0127 | 78.57 ± 2.38 | 0.6739 ± 0.0165 | 78.60 ± 2.40 |
| a6_selective50_interaction4 | mosei | pilot | valid_mae | test | — (0/3) | — | — | — | — |
| a6_selective75_interaction4 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6362 | 83.33 | 0.6687 | 83.36 |
| a6_selective75_interaction4 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| a7_pair_raw | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6572 | 76.19 | 0.6441 | 76.03 |
| a7_pair_raw | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| a7_pair_snr | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6345 | 78.57 | 0.6942 | 78.51 |
| a7_pair_snr | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| a8_triple_raw | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6372 | 78.57 | 0.6549 | 78.61 |
| a8_triple_raw | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| a9_random_orthogonal_cseed100 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6828 | 73.81 | 0.6252 | 73.50 |
| a9_random_orthogonal_cseed100 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| a9_random_orthogonal_cseed200 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6321 | 83.33 | 0.6601 | 83.36 |
| a9_random_orthogonal_cseed200 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| a9_random_orthogonal_cseed300 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6486 | 83.33 | 0.6608 | 83.29 |
| a9_random_orthogonal_cseed300 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| baseline_benchmark500 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.7253 | 71.43 | 0.6001 | 70.40 |
| baseline_benchmark500 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| full_kd_benchmark500 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6202 | 85.71 | 0.7307 | 85.71 |
| full_kd_benchmark500 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| high_order_interaction_benchmark500 | mosei | pilot | valid_mae | validation | 2026,2027,2028 (3/3) | 0.6425 ± 0.0494 | 78.57 ± 4.76 | 0.6600 ± 0.0748 | 78.31 ± 5.08 |
| high_order_interaction_benchmark500 | mosei | pilot | valid_mae | test | — (0/3) | — | — | — | — |
| mobius_full_benchmark500 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6285 | 78.57 | 0.6910 | 78.61 |
| mobius_full_benchmark500 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| subset_value_4_benchmark500 | mosei | pilot | valid_mae | validation | 2026,2027,2028 (3/3) | 0.6029 ± 0.0244 | 76.98 ± 2.75 | 0.7135 ± 0.0168 | 76.95 ± 2.79 |
| subset_value_4_benchmark500 | mosei | pilot | valid_mae | test | — (0/3) | — | — | — | — |
| subset_value_benchmark500 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6306 | 80.95 | 0.6752 | 81.00 |
| subset_value_benchmark500 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2_inverse_variance | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6746 | 78.57 | 0.6380 | 78.32 |
| v2_inverse_variance | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2_pair_raw | mosei | pilot | valid_mae | validation | 13,42 (2/2) | 0.6339 ± 0.0138 | 84.52 ± 5.05 | 0.6677 ± 0.0139 | 84.53 ± 5.07 |
| v2_pair_raw | mosei | pilot | valid_mae | test | — (0/2) | — | — | — | — |
| v2_pair_snr | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6843 | 78.57 | 0.6154 | 78.61 |
| v2_pair_snr | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2_random_orthogonal_cseed100 | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6634 | 78.57 | 0.5965 | 78.61 |
| v2_random_orthogonal_cseed100 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2_raw_interaction4 | mosei | pilot | valid_mae | validation | 13,42 (2/2) | 0.6392 ± 0.0076 | 78.57 ± 3.37 | 0.6799 ± 0.0004 | 78.57 ± 3.37 |
| v2_raw_interaction4 | mosei | pilot | valid_mae | test | — (0/2) | — | — | — | — |
| v2_repro_pair_raw_gpu0 | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.7963 | 66.67 | 0.5230 | 64.78 |
| v2_repro_pair_raw_gpu0 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2_repro_pair_raw_gpu1 | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.7963 | 66.67 | 0.5230 | 64.78 |
| v2_repro_pair_raw_gpu1 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2_selective50 | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6257 | 80.95 | 0.6859 | 80.95 |
| v2_selective50 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2_snr_interaction4 | mosei | pilot | valid_mae | validation | 13,42 (2/2) | 0.6236 ± 0.0164 | 82.14 ± 5.05 | 0.6720 ± 0.0361 | 82.16 ± 5.02 |
| v2_snr_interaction4 | mosei | pilot | valid_mae | test | — (0/2) | — | — | — | — |
| v2_subset4 | mosei | pilot | valid_mae | validation | 13,42 (2/2) | 0.6307 ± 0.0403 | 79.76 ± 5.05 | 0.6877 ± 0.0523 | 79.69 ± 5.19 |
| v2_subset4 | mosei | pilot | valid_mae | test | — (0/2) | — | — | — | — |
| v2_triple_raw | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6823 | 80.95 | 0.6234 | 80.82 |
| v2_triple_raw | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2_zscore_interaction4 | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6934 | 76.19 | 0.6001 | 76.19 |
| v2_zscore_interaction4 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2d_inverse_variance_interaction4 | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6547 | 85.71 | 0.6425 | 85.75 |
| v2d_inverse_variance_interaction4 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2d_pair_raw | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6434 | 80.95 | 0.6497 | 81.00 |
| v2d_pair_raw | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2d_pair_snr | mosei | pilot | valid_mae | validation | 13,2026,42 (3/3) | 0.6221 ± 0.0068 | 80.95 ± 2.38 | 0.6797 ± 0.0152 | 80.89 ± 2.52 |
| v2d_pair_snr | mosei | pilot | valid_mae | test | — (0/3) | — | — | — | — |
| v2d_random_nonorthogonal | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6878 | 83.33 | 0.6475 | 83.36 |
| v2d_random_nonorthogonal | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2d_random_orthogonal | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.7153 | 78.57 | 0.5581 | 78.61 |
| v2d_random_orthogonal | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2d_raw_interaction4 | mosei | pilot | valid_mae | validation | 13,2026,42 (3/3) | 0.6423 ± 0.0276 | 76.19 ± 2.38 | 0.6775 ± 0.0223 | 75.94 ± 2.41 |
| v2d_raw_interaction4 | mosei | pilot | valid_mae | test | — (0/3) | — | — | — | — |
| v2d_selective50_interaction4 | mosei | pilot | valid_mae | validation | 13,2026,42 (3/3) | 0.6221 ± 0.0309 | 77.78 ± 2.75 | 0.6835 ± 0.0195 | 77.72 ± 2.80 |
| v2d_selective50_interaction4 | mosei | pilot | valid_mae | test | — (0/3) | — | — | — | — |
| v2d_snr_interaction4 | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6990 | 73.81 | 0.6202 | 73.49 |
| v2d_snr_interaction4 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2d_subset4 | mosei | pilot | valid_mae | validation | 13,2026,42 (3/3) | 0.6362 ± 0.0408 | 82.54 ± 1.37 | 0.6548 ± 0.0486 | 82.55 ± 1.34 |
| v2d_subset4 | mosei | pilot | valid_mae | test | — (0/3) | — | — | — | — |
| v2d_triple_raw | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6361 | 78.57 | 0.6839 | 78.51 |
| v2d_triple_raw | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| v2d_zscore_interaction4 | mosei | pilot | valid_mae | validation | 42 (1/1) | 0.6525 | 80.95 | 0.6374 | 81.00 |
| v2d_zscore_interaction4 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| stage_d_batch8_probe | mosei | smoke/preflight | valid_mae | validation | 13 (1/1) | 3.1667 | 38.46 | -0.2428 | 21.37 |
| stage_d_batch8_probe | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |
| stage_d_c0_task_only_lora_smoke | mosei | smoke/preflight | valid_mae | validation | 13 (1/1) | 2.8984 | 0.00 | -1.0000 | 0.00 |
| stage_d_c0_task_only_lora_smoke | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |
| stage_d_c2_ensemble_pair_lora_smoke | mosei | smoke/preflight | valid_mae | validation | 13 (1/1) | 3.3828 | 0.00 | -1.0000 | 0.00 |
| stage_d_c2_ensemble_pair_lora_smoke | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |
| stage_d_cminus1_online_frozen_full_kd_smoke | mosei | smoke/preflight | valid_mae | validation | 13 (1/1) | 2.9844 | 0.00 | -1.0000 | 0.00 |
| stage_d_cminus1_online_frozen_full_kd_smoke | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |
| stage_d_smoke_full_kd_lora | mosei | smoke/preflight | valid_mae | validation | 13 (1/1) | 0.5527 | 100.00 | -1.0000 | 100.00 |
| stage_d_smoke_full_kd_lora | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |
| stage_d_smoke_hidden_kd | mosei | smoke/preflight | valid_mae | validation | 13 (1/1) | 2.8333 | 61.54 | 0.2428 | 46.89 |
| stage_d_smoke_hidden_kd | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |
| stage_d_smoke_ru_lora | mosei | smoke/preflight | valid_mae | validation | 42 (1/1) | 2.5000 | 100.00 | — | — |
| stage_d_smoke_ru_lora | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |
| video_all_layers_preflight_20260911 | mosei | smoke/preflight | valid_mae | validation | 13 (1/1) | 2.8854 | 57.14 | — | 41.56 |
| video_all_layers_preflight_20260911 | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |

### student/tuning

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| interaction4_lambda0.25 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6366 | 78.57 | 0.6508 | 78.51 |
| interaction4_lambda0.25 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| interaction4_lambda0.5 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6429 | 85.71 | 0.6485 | 85.71 |
| interaction4_lambda0.5 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| interaction4_lambda2 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.7062 | 73.81 | 0.6155 | 73.50 |
| interaction4_lambda2 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| subset4_lambda0.25 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6492 | 83.33 | 0.6487 | 83.29 |
| subset4_lambda0.25 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| subset4_lambda0.5 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6155 | 85.71 | 0.6713 | 85.71 |
| subset4_lambda0.5 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |
| subset4_lambda2 | mosei | pilot | valid_mae | validation | 2026 (1/1) | 0.6282 | 85.71 | 0.6710 | 85.71 |
| subset4_lambda2 | mosei | pilot | valid_mae | test | — (0/1) | — | — | — | — |

### student/video_adaptation_smoke_v2

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| frozen_video | mosei | smoke/preflight | valid_mae | validation | 13 (1/1) | 1.8086 | 0.00 | -1.0000 | 0.00 |
| frozen_video | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |
| ta_only | mosei | smoke/preflight | valid_mae | validation | 13 (1/1) | 2.5000 | 100.00 | — | 100.00 |
| ta_only | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |
| video_lora | mosei | smoke/preflight | valid_mae | validation | 13 (1/1) | 3.4688 | 0.00 | — | 0.00 |
| video_lora | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |
| video_lora_resume | mosei | smoke/preflight | valid_mae | validation | 13 (1/1) | 3.4688 | 0.00 | — | 0.00 |
| video_lora_resume | mosei | smoke/preflight | valid_mae | test | — (0/1) | — | — | — | — |

### student/video_adaptation_v1

| 方法 | 数据集 | 范围 | 选模 | split | seeds（已有/该组） | MAE ↓ | Acc2 NZ ↑ (%) | Pearson ↑ | F1 NZ ↑ (%) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| frozen_video | mosei | full | valid_mae | validation | 13 (1/1) | 0.4717 | 87.69 | 0.7965 | 87.74 |
| frozen_video | mosei | full | valid_mae | test | 13 (1/1) | 0.4818 | 86.98 | 0.8200 | 86.99 |
| ta_only | mosei | full | valid_mae | validation | 13 (1/1) | 0.4690 | 88.60 | 0.7890 | 88.54 |
| ta_only | mosei | full | valid_mae | test | 13 (1/1) | 0.4833 | 87.53 | 0.8159 | 87.50 |
| video_lora | mosei | full | valid_mae | validation | 13 (1/1) | 0.4703 | 87.76 | 0.7922 | 87.73 |
| video_lora | mosei | full | valid_mae | test | 13 (1/1) | 0.4835 | 87.04 | 0.8143 | 86.99 |
