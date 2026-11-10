# MOSI A 系列：CE × ICD 强度细化扫描

**日期：** 2026-11-10
**数据集：** CMU-MOSI（官方 train/valid 划分，与第二阶段确认实验一致）
**主指标：** Validation MAE（越低越好）
**目的：** 在 `icd_confirmatory_mosi_v1`（F0–F7）基础上细化两个切片：
No-CE 条件下的低 ICD 强度（λ=0.05/0.10/0.15），以及 λ=0.10 条件下的 CE 权重
（c=0.10/0.25/0.50），用于确定最优 (c, λ) 组合。

---

## 1. 实验列表

| ID | 方法 | Task CE 权重 c | ICD 权重 λ | 输出目录 |
|---|---|---:|---:|---|
| A1 | uniform_interaction | 0.00 | 0.05 | runs/A1_no_ce_lambda005 |
| A2 | uniform_interaction | 0.00 | 0.10 | runs/A2_no_ce_lambda010 |
| A3 | uniform_interaction | 0.00 | 0.15 | runs/A3_no_ce_lambda015 |
| A4 | uniform_interaction | 0.10 | 0.10 | runs/A4_ce010_lambda010 |
| A5 | uniform_interaction | 0.25 | 0.10 | runs/A5_ce025_lambda010 |
| A6 | uniform_interaction | 0.50 | 0.10 | runs/A6_ce050_lambda010 |

A1–A3 构成 CE=0 的 λ 扫描；A2、A4、A5、A6 构成 λ=0.10 的 CE 扫描（A2 同时属于
两个切片）。全部 run 使用同一冻结学生、同一冻结三 Probe 教师资产、同一 seed、同一
数据划分。

## 2. 固定训练协议（与确认实验完全一致）

| 参数 | 设置 |
|---|---|
| Trainer | `project/scripts/train_icd_confirmatory.py`（SHA256 冻结） |
| 冻结源码 | `icd_confirmatory_mosi_v1/source/`（SHA256 冻结） |
| 资产 | 复制自 `icd_confirmatory_mosi_v1/assets/protocol.json`（SHA256 一致） |
| Seed | 13 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 1e-4，constant |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Max epochs | 40 |
| Early stopping patience | 8 |
| Checkpoint selection | Validation MAE（tie 1e-5） |
| Checkpoint retention | `best_valid_mae.pt`、`best_valid_acc2_nonzero.pt`、`last.pt` |
| Official-test | 本批不使用、不评估 |

损失定义与确认实验一致：

$$
\mathcal L = \mathcal L_{\text{task-reg}} + c\,\mathcal L_{\text{task-CE}}
+ \mathcal L_{\text{FullKD-reg}} + \mathcal L_{\text{FullKD-KL}}
+ \lambda\,\mathcal L_{\text{ICD}}.
$$

No-CE 仅移除真实标签 Task CE，教师 Full KD KL 始终保留。

## 3. 跨实验参考（仅用于解释，不参与本批选择）

来自 `icd_confirmatory_mosi_v1` 的已完成 run，仅作为同一冻结协议下的上下文：

| 参考 | c | λ | Valid MAE |
|---|---:|---:|---:|
| F1_full_kd_no_ce | 0.00 | 0.00 | 0.7548 |
| F7_icd_no_ce_lambda025 | 0.00 | 0.25 | 0.7464 |
| F5_icd_no_ce_lambda05 | 0.00 | 0.50 | 0.7643 |
| F3_icd_no_ce_lambda1 | 0.00 | 1.00 | 0.7697 |
| F0_full_kd_ce | 0.50 | 0.00 | 0.7453 |
| F6_icd_ce_lambda025 | 0.50 | 0.25 | 0.7417 |
| F4_icd_ce_lambda05 | 0.50 | 0.50 | 0.7583 |
| F2_icd_ce_lambda1 | 0.50 | 1.00 | 0.7768 |

跨实验数值只在初始化哈希、资产哈希、源码哈希一致时才可并列陈述；本批报告会显式
标注这些参考不属于 A 系列。

## 4. 判定与选择规则

- 主指标固定为 Validation MAE；不得用测试集或辅助指标替换。
- 同一批内 MAE 差不超过 0.005 视为单 seed 下难以区分，按更简单配置优先解释。
- 与匹配参考（同 c、同 λ）相比，MAE 改善至少 0.01 才认为有实质优势。
- 不根据本批结果修改训练协议；不做事后 λ 调整。

## 5. 执行方式

- 物理 GPU1，`CUDA_VISIBLE_DEVICES=1`，训练进程内使用逻辑 `cuda:0`。
- 2 个训练进程并行，单 worker 预留显存 38000 MiB，另加 8192 MiB 安全余量。
- 每个 run 独立输出目录，支持 `--resume`；存在的 `run_config.json` 必须与本次
  参数一致。
- 全部 6 个 run 完成后运行独立分析脚本，重新计算指标并写完成审计。

## 6. 本批不包含

- Official-test 评估（仅 Validation）。
- MOSEI 跨数据集实验。
- 额外随机种子。
- Scheduler、Gradient Projection 等机制改动。

---

## 7. 执行修订（2026-11-10）：追加 Official-test

- 用户在 A 系列训练进行中要求追加测试集评估。
- 追加内容：全部完成的 A run 的两个验证集选定 checkpoint
  （`best_valid_mae.pt`、`best_valid_acc2_nonzero.pt`）在官方 MOSI 测试清单
  （688 windows / 686 utterances）上评估，结果写入
  `outputs/experiments/official_test_a_series_v1/`，并汇总进 `analysis/report.md`。
- Official-test 仍未参与训练、早停、checkpoint 选择或任何超参数决定；
  `official_test_used_for_selection = false`。
- 本修订不改变第 2 节的训练协议、第 4 节的选择规则或第 1 节的 6 个 run 定义。
