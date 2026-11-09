# MOSI 第二阶段确认实验报告

全部结果使用 seed13，Validation MAE 选取同一个 checkpoint；Official-test 未参与训练、早停或任何配置选择，仅在全部训练结束后用于报告。所有资产保存在 Qi 服务器。

## 完整结果

|Run|CE|ICD λ|Scheduler|Best epoch|MAE|Pearson|Acc-2 nonzero|Weighted F1 nonzero|
|---|---:|---:|---|---:|---:|---:|---:|---:|
|F0_full_kd_ce|0.5|0.0|constant|13|0.7453|0.7950|0.8426|0.8421|
|F1_full_kd_no_ce|0.0|0.0|constant|19|0.7548|0.7777|0.8333|0.8309|
|F2_icd_ce_lambda1|0.5|1.0|constant|6|0.7768|0.7931|0.8333|0.8294|
|F3_icd_no_ce_lambda1|0.0|1.0|constant|23|0.7697|0.7724|0.8611|0.8598|
|F4_icd_ce_lambda05|0.5|0.5|constant|17|0.7583|0.7746|0.8519|0.8493|
|F5_icd_no_ce_lambda05|0.0|0.5|constant|15|0.7643|0.7862|0.8472|0.8469|
|F6_icd_ce_lambda025|0.5|0.25|constant|15|0.7417|0.7858|0.8611|0.8604|
|F7_icd_no_ce_lambda025|0.0|0.25|constant|26|0.7464|0.7833|0.8519|0.8511|
|S1_warmup_cosine|0.5|0.25|warmup_cosine|18|0.7636|0.7833|0.8102|0.8077|
|GP_icd_ce|0.5|0.25|constant|16|0.7366|0.8018|0.8704|0.8705|
|GP_icd_no_ce|0.0|0.25|constant|13|0.7431|0.8172|0.8472|0.8439|

## CE、ICD 与交互效应

正的 CE 效应表示移除真实标签 CE 改善 MAE；教师 KL 始终保留。
- λ=0: ΔCE=-0.0095。
- λ=0.25: ΔCE=-0.0046。
- λ=0.5: ΔCE=-0.0059。
- λ=1: ΔCE=+0.0071。
- CE=0, λ=0.25: ΔICD=+0.0085。
- CE=0, λ=0.5: ΔICD=-0.0094。
- CE=0, λ=1: ΔICD=-0.0149。
- CE=0.5, λ=0.25: ΔICD=+0.0036。
- CE=0.5, λ=0.5: ΔICD=-0.0130。
- CE=0.5, λ=1: ΔICD=-0.0315。
- λ=0.25: No-CE ICD 收益减去 CE ICD 收益=+0.0049。
- λ=0.5: No-CE ICD 收益减去 CE ICD 收益=+0.0036。
- λ=1: No-CE ICD 收益减去 CE ICD 收益=+0.0166。

## 开发阶段方法选择

原始最低 Validation MAE 为 GP_icd_ce（0.7366）。
依据 MAE 差不超过 0.005 时优先简单方案的工程规则，当前建议候选为 GP_icd_ce（0.7366）。
上述阈值不是统计显著性阈值；候选集合、匹配 Full KD 的至少 0.01 MAE 改善筛选和辅助指标变化见 method_selection.json。
固定 LR 的 S0 直接引用对应 Phase1 run；若调度预算扩展，则 S0/S1 均从相同初始化重跑，排除原 40 epoch cosine 结果。
GP 与同 CE、同 λ、同训练预算的普通 ICD 比较；仅梯度路由改变。GP 两种 CE 条件均保留，防止把 CE 的变化归因于投影。

## Official-test 最终评估（仅报告）

测试清单 `/data/sjq/kd/dataset/cmu_mosi/manifests/official_test_windowed.jsonl`（686 windows / 686 utterances）。下表 checkpoint 仍由 Validation MAE 选出；Acc-2 最优 checkpoint 及第一轮结果见 `official_test_summary.csv`。

|Run|checkpoint|epoch|valid MAE|test MAE|test Pearson|test Acc-2|
|---|---|---:|---:|---:|---:|---:|
|F7_icd_no_ce_lambda025|valid_mae|26|0.7464|0.6812|0.8101|0.8613|
|S1_warmup_cosine|valid_mae|18|0.7636|0.7004|0.8081|0.8613|
|F3_icd_no_ce_lambda1|valid_mae|23|0.7697|0.7036|0.8139|0.8567|
|F1_full_kd_no_ce|valid_mae|19|0.7548|0.7091|0.8152|0.8506|
|F6_icd_ce_lambda025|valid_mae|15|0.7417|0.7169|0.7905|0.8476|
|F0_full_kd_ce|valid_mae|13|0.7453|0.7174|0.7914|0.8430|
|GP_icd_ce|valid_mae|16|0.7366|0.7252|0.7940|0.8491|
|F5_icd_no_ce_lambda05|valid_mae|15|0.7643|0.7347|0.7889|0.8369|
|F4_icd_ce_lambda05|valid_mae|17|0.7583|0.7449|0.7882|0.8384|
|F2_icd_ce_lambda1|valid_mae|6|0.7768|0.7577|0.7999|0.8430|
|GP_icd_no_ce|valid_mae|13|0.7431|0.7746|0.8027|0.8384|

测试集排名不作为方法选择依据；单 seed、686 条测试样本上的差异仍需多种子确认。

## 机制、曲线及不确定性

七交互坐标、一/二/三阶重建误差与任务 MAE 分别见 interaction_analysis.json。分情感区间、交互强度和 source-video 误差均来自同一 MAE-selected checkpoint。
逐样本预测独立重算全部主要指标，并核对 229 utterances、10 source videos、初始化哈希、参数数量、完整逐轮 history、恰好三个保留 checkpoint 与 1e-5 选择阈值。
配对 source-video clustered bootstrap 使用 10,000 次重采样。只有 10 个 validation clusters，区间仅用于描述不确定性；不能解释成多种子确认。
梯度范数与方向只描述训练现象，不作为因果证明。训练目标因 loss 权重不同，其数值不能直接跨组比较。

## 后续阶段

Official-test 最终评估已按协议在全部训练结束后完成，仅用于报告；MOSEI 跨数据集验证与多种子确认尚未执行，本报告不宣称跨数据集普适性。

## 完成审计

审计发现 0 项错误；详情见 ../completion_audit.json。
