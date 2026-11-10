# MOSI A 系列：CE × ICD 强度细化扫描报告

全部结果使用 seed13、与确认实验相同的冻结资产、冻结源码、训练协议和 Validation MAE checkpoint 选择。Official-test 仅在训练与 Validation 选择冻结之后用于报告，未参与任何选择（若尚未执行，则本报告只含 Validation）。

## 1. 本批结果

|Run|CE|ICD λ|Best epoch|MAE|Pearson|Acc-2|F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|A1_no_ce_lambda005|0.00|0.05|29|0.7494|0.7916|0.8380|0.8370|
|A2_no_ce_lambda010|0.00|0.10|6|0.7833|0.7875|0.8565|0.8570|
|A3_no_ce_lambda015|0.00|0.15|12|0.7500|0.7933|0.8333|0.8333|
|A4_ce010_lambda010|0.10|0.10|26|0.7460|0.7990|0.8657|0.8649|
|A5_ce025_lambda010|0.25|0.10|13|0.7757|0.7800|0.8241|0.8199|
|A6_ce050_lambda010|0.50|0.10|32|0.7212|0.8087|0.8380|0.8381|

## 2. CE=0 的 λ 扫描（含确认实验参考）

|Run|CE|ICD λ|Best epoch|MAE|Pearson|Acc-2|F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|F1_full_kd_no_ce (参考)|0.00|0.00|19|0.7548|0.7777|0.8333|0.8309|
|A1_no_ce_lambda005|0.00|0.05|29|0.7494|0.7916|0.8380|0.8370|
|A2_no_ce_lambda010|0.00|0.10|6|0.7833|0.7875|0.8565|0.8570|
|A3_no_ce_lambda015|0.00|0.15|12|0.7500|0.7933|0.8333|0.8333|
|F7_icd_no_ce_lambda025 (参考)|0.00|0.25|26|0.7464|0.7833|0.8519|0.8511|
|F5_icd_no_ce_lambda05 (参考)|0.00|0.50|15|0.7643|0.7862|0.8472|0.8469|
|F3_icd_no_ce_lambda1 (参考)|0.00|1.00|23|0.7697|0.7724|0.8611|0.8598|

F 系列数值来自 `icd_confirmatory_mosi_v1`，仅作为同一冻结协议下的上下文；它们不属于本批实验，不参与本批选择。

## 3. λ=0.10 的 CE 扫描

|Run|CE|ICD λ|Best epoch|MAE|Pearson|Acc-2|F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|A2_no_ce_lambda010|0.00|0.10|6|0.7833|0.7875|0.8565|0.8570|
|A4_ce010_lambda010|0.10|0.10|26|0.7460|0.7990|0.8657|0.8649|
|A5_ce025_lambda010|0.25|0.10|13|0.7757|0.7800|0.8241|0.8199|
|A6_ce050_lambda010|0.50|0.10|32|0.7212|0.8087|0.8380|0.8381|

## 4. 工程判断

- 本批最低 Validation MAE：**A6_ce050_lambda010** = 0.7212（best epoch 32，Pearson 0.8087）。
- 与最低 MAE 差不超过 0.005 的配置：A6_ce050_lambda010；单 seed 下视为难以区分。
- 判定阈值是工程筛选标准，不是统计显著性阈值。
- 相对 A1_no_ce_lambda005（λ=0.05），A2_no_ce_lambda010（λ=0.10）的 MAE 变化 +0.0339。
- 相对 A1_no_ce_lambda005（λ=0.05），A3_no_ce_lambda015（λ=0.15）的 MAE 变化 +0.0006。
- 相对 A2_no_ce_lambda010（c=0.00），A4_ce010_lambda010（c=0.10）的 MAE 变化 -0.0373。
- 相对 A2_no_ce_lambda010（c=0.00），A5_ce025_lambda010（c=0.25）的 MAE 变化 -0.0076。
- 相对 A2_no_ce_lambda010（c=0.00），A6_ce050_lambda010（c=0.50）的 MAE 变化 -0.0621。

## 5. 独立复核与审计

- 审计错误数：0。
- 逐样本预测独立重算 MAE、Pearson、Acc-2、Weighted F1，并核对 229 条 validation utterance、最佳轮次、三个保留 checkpoint、初始化哈希与参数数量。

## 6. Official-test（仅报告，未参与选择）

测试清单：`/data/sjq/kd/dataset/cmu_mosi/manifests/official_test_windowed.jsonl`（688 windows / 686 utterances）。测试指标在全部训练与 Validation 选择冻结之后计算。

|Run|checkpoint|epoch|valid MAE|test MAE|test Pearson|test Acc-2|test F1|
|---|---|---:|---:|---:|---:|---:|---:|
|A1_no_ce_lambda005|valid_acc2_nonzero|24|0.7931|0.6897|0.8076|0.8598|0.8593|
|A1_no_ce_lambda005|valid_mae|29|0.7494|0.7013|0.8094|0.8643|0.8647|
|A2_no_ce_lambda010|valid_acc2_nonzero|6|0.7833|0.7172|0.7940|0.8491|0.8484|
|A2_no_ce_lambda010|valid_mae|6|0.7833|0.7172|0.7940|0.8491|0.8484|
|A3_no_ce_lambda015|valid_acc2_nonzero|16|0.7982|0.7587|0.7926|0.8460|0.8454|
|A3_no_ce_lambda015|valid_mae|12|0.7500|0.6875|0.8026|0.8613|0.8614|
|A4_ce010_lambda010|valid_acc2_nonzero|26|0.7460|0.6872|0.8132|0.8567|0.8572|
|A4_ce010_lambda010|valid_mae|26|0.7460|0.6872|0.8132|0.8567|0.8572|
|A5_ce025_lambda010|valid_acc2_nonzero|16|0.8048|0.6957|0.8072|0.8582|0.8572|
|A5_ce025_lambda010|valid_mae|13|0.7757|0.7343|0.7981|0.8430|0.8439|
|A6_ce050_lambda010|valid_acc2_nonzero|5|0.8318|0.7866|0.7754|0.8323|0.8331|
|A6_ce050_lambda010|valid_mae|32|0.7212|0.6979|0.7957|0.8445|0.8444|

`valid_acc2_nonzero` checkpoint 仅作辅助留档；主报告仍以 `valid_mae` checkpoint 为准，测试集排名不作为任何配置选择的依据。

## 7. 范围声明

- MOSI Validation 结果来自本批 6 个 seed13 run；Official-test 仅在全部训练结束后用于报告，未参与训练、早停或 checkpoint 选择。
- 不包含 MOSEI 与额外种子；结论只能在本批与确认实验共享的冻结协议内陈述。
