# MOSI A 系列 official-test（仅报告）

测试清单：`/data/sjq/kd/dataset/cmu_mosi/manifests/official_test_windowed.jsonl`（688 windows / 686 utterances）。
测试指标在全部训练与 Validation 选择冻结之后计算，未参与任何选择。

|Run|checkpoint|epoch|valid MAE|test MAE|test Pearson|test Acc-2|test F1|
|---|---|---:|---:|---:|---:|---:|---:|
| A1_no_ce_lambda005 | valid_acc2_nonzero | 24 | 0.7931 | 0.6897 | 0.8076 | 0.8598 | 0.8593 |
| A1_no_ce_lambda005 | valid_mae | 29 | 0.7494 | 0.7013 | 0.8094 | 0.8643 | 0.8647 |
| A2_no_ce_lambda010 | valid_acc2_nonzero | 6 | 0.7833 | 0.7172 | 0.7940 | 0.8491 | 0.8484 |
| A2_no_ce_lambda010 | valid_mae | 6 | 0.7833 | 0.7172 | 0.7940 | 0.8491 | 0.8484 |
| A3_no_ce_lambda015 | valid_acc2_nonzero | 16 | 0.7982 | 0.7587 | 0.7926 | 0.8460 | 0.8454 |
| A3_no_ce_lambda015 | valid_mae | 12 | 0.7500 | 0.6875 | 0.8026 | 0.8613 | 0.8614 |
| A4_ce010_lambda010 | valid_acc2_nonzero | 26 | 0.7460 | 0.6872 | 0.8132 | 0.8567 | 0.8572 |
| A4_ce010_lambda010 | valid_mae | 26 | 0.7460 | 0.6872 | 0.8132 | 0.8567 | 0.8572 |
| A5_ce025_lambda010 | valid_acc2_nonzero | 16 | 0.8048 | 0.6957 | 0.8072 | 0.8582 | 0.8572 |
| A5_ce025_lambda010 | valid_mae | 13 | 0.7757 | 0.7343 | 0.7981 | 0.8430 | 0.8439 |
| A6_ce050_lambda010 | valid_acc2_nonzero | 5 | 0.8318 | 0.7866 | 0.7754 | 0.8323 | 0.8331 |
| A6_ce050_lambda010 | valid_mae | 32 | 0.7212 | 0.6979 | 0.7957 | 0.8445 | 0.8444 |
