# RDID-MOSEI Stage A 实验结果

单 seed（2026）快速筛选；A1/A2 为既有固定基线。

| 方法 | MAE | Pearson | Acc-2 | F1 | Acc-7 | 高交互 MAE | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 subset4 | 0.5749 | 0.7289 | 0.7857 | 0.7851 | 0.6600 | 0.8849 | 14 |
| A2 raw interaction4 | 0.6379 | 0.7128 | 0.7381 | 0.7313 | 0.5600 | 0.8407 | 11 |
| A3 z-score interaction4 | 0.6161 | 0.6847 | 0.8571 | 0.8571 | 0.5600 | 0.8933 | 13 |
| A4 inverse-variance interaction4 | 0.6769 | 0.6299 | 0.8095 | 0.8095 | 0.5000 | 1.0520 | 15 |
| A5 SNR interaction4 | 0.6664 | 0.6735 | 0.8095 | 0.8082 | 0.4600 | 0.9275 | 6 |
| A6 selective top25 | 0.6349 | 0.6832 | 0.8095 | 0.8100 | 0.5200 | 0.8893 | 6 |
| A6 selective top50 | 0.6489 | 0.6766 | 0.7619 | 0.7619 | 0.5600 | 0.7965 | 7 |
| A6 selective top75 | 0.6362 | 0.6687 | 0.8333 | 0.8336 | 0.5000 | 0.8654 | 6 |
| A7 pair raw | 0.6572 | 0.6441 | 0.7619 | 0.7603 | 0.5400 | 0.9787 | 10 |
| A7 pair SNR | 0.6345 | 0.6942 | 0.7857 | 0.7851 | 0.4800 | 0.9019 | 28 |
| A8 triple raw | 0.6372 | 0.6549 | 0.7857 | 0.7861 | 0.5400 | 1.0884 | 8 |
| A9 orthogonal 100 | 0.6828 | 0.6252 | 0.7381 | 0.7350 | 0.5200 | 0.8980 | 5 |
| A9 orthogonal 200 | 0.6321 | 0.6601 | 0.8333 | 0.8336 | 0.5200 | 1.0198 | 13 |
| A9 orthogonal 300 | 0.6486 | 0.6608 | 0.8333 | 0.8329 | 0.5000 | 1.0848 | 10 |
| A10 nonorthogonal 100 | 0.6701 | 0.6206 | 0.8095 | 0.8095 | 0.4800 | 0.9188 | 10 |

## 三 seed 候选复验

| 方法 | MAE | Pearson | Acc-7 | 高交互 MAE |
|---|---:|---:|---:|---:|
| subset4 | 0.6029 ± 0.0244 | 0.7135 ± 0.0168 | 0.6000 ± 0.0529 | 0.8790 ± 0.1101 |
| z-score interaction4 | 0.5905 ± 0.0222 | 0.7269 ± 0.0385 | 0.5733 ± 0.0231 | 0.8287 ± 0.1131 |
| selective top50 | 0.6351 ± 0.0127 | 0.6739 ± 0.0165 | 0.5533 ± 0.0306 | 0.9383 ± 0.1513 |

## Stage A Gate

```json
{
  "condition_1_overall_mae": false,
  "condition_2_high_interaction_mae": true,
  "condition_3_interaction_distortion": "not_available_from_saved_tav_predictions",
  "condition_4_seed_variance": true,
  "three_seed_condition_1_zscore": true,
  "three_seed_condition_2_selective": false,
  "go": true
}
```
