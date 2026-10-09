# `tav_main_v1` 统一实验记录

> 文档状态：历史主实验完整记录。最终论文使用名称为 Interaction Coordinate Distillation (ICD)；`TAV` 和 M0/M1/M3/M4/M6 保留为实验阶段及内部方法编号。
>
> 临时运行整理：已验证的 `preflight` 包含 M0/M2/M3/M6 预检，以及 M6 checkpoint 连续训练恢复测试。早期恢复曾因 PyTorch 2.6 `weights_only` 默认值变化失败，`continuous_v2` 与 `resumed_v2` 随后完成。临时目录已由正式训练与本记录取代。

以下按“最终结果 → 阶段结果 → 历史进展 → 执行方案”的顺序完整保留原文。

## 原文快照：`TAV主实验结果汇总_20260917.md`

## TAV/P0 实验结果汇总（更新至 2026-09-18）

> 核对时间：2026-09-18T11:32:41+08:00（北京时间）。

### 1. 当前完成情况与正式三seed结果

**P0 的 15/15 个 train/valid run 和对应 15 个 valid-best checkpoint 的 MOSEI official-test 评测均已完成。** M0、M1、M3、M4、M6 均覆盖 seeds 13/42/2026；M4_seed42 和 M4_seed2026 的最佳 epoch 均为 11。测试结果只用于报告，不反向参与配置、方法或 checkpoint 选择。

以下正式聚合使用全部三种子完成 run。均值±样本标准差（ddof=1），本节主指标为 official-valid MAE。

| 模型 | 方法 | 完成 seeds | Valid MAE ↓（均值 ± 样本SD） |
|---|---|---:|---:|
| M0 | 冻结 TAV Student-only | 3/3 | 0.527314 ± 0.005825 |
| M1 | 冻结 TAV Full KD | 3/3 | 0.514162 ± 0.003197 |
| M3 | Full KD + T/A/V LoRA | 3/3 | 0.467097 ± 0.003826 |
| M4 | M3 + Uniform Interaction | 3/3 | 0.469803 ± 0.005565 |
| M6 | M3 + Reliability × Utility | 3/3 | 0.463968 ± 0.001598 |

#### 三 seed valid 辅助指标

各指标均来自同一个valid MAE最佳checkpoint；Acc-2/F1排除零标签，数值为比例。

| 模型 | Pearson ↑ | Acc-2 ↑ | Weighted F1 ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|
| M0 | 0.731520 ± 0.008172 | 0.852341 ± 0.017320 | 0.852290 ± 0.015816 | 0.538037 ± 0.004287 |
| M1 | 0.741258 ± 0.002165 | 0.842374 ± 0.007436 | 0.843132 ± 0.005073 | 0.546410 ± 0.005614 |
| M3 | 0.790061 ± 0.007510 | 0.880158 ± 0.006573 | 0.880114 ± 0.006603 | 0.578657 ± 0.009355 |
| M4 | 0.796798 ± 0.004035 | 0.878767 ± 0.001062 | 0.878384 ± 0.001503 | 0.568680 ± 0.005264 |
| M6 | 0.798981 ± 0.003321 | 0.880158 ± 0.011306 | 0.880625 ± 0.010428 | 0.574381 ± 0.004937 |

### 2. 全部15个run的最佳epoch与进度

全部 run 均已结束；最佳 checkpoint 统一由 valid MAE 选择。

| Run | 状态 | 已完成 epochs | 最佳 epoch | 最佳 valid MAE ↓ | 当前进度 |
|---|---|---:|---:|---:|---|
| M0_seed13 | 已完成 | 19 | 12 | 0.533328 | — |
| M0_seed42 | 已完成 | 14 | 7 | 0.526915 | — |
| M0_seed2026 | 已完成 | 16 | 9 | 0.521699 | — |
| M1_seed13 | 已完成 | 12 | 5 | 0.516635 | — |
| M1_seed42 | 已完成 | 18 | 11 | 0.510552 | — |
| M1_seed2026 | 已完成 | 13 | 6 | 0.515298 | — |
| M3_seed13 | 已完成 | 13 | 6 | 0.471261 | — |
| M3_seed42 | 已完成 | 22 | 15 | 0.466293 | — |
| M3_seed2026 | 已完成 | 17 | 10 | 0.463737 | — |
| M4_seed13 | 已完成 | 13 | 6 | 0.474555 | — |
| M4_seed42 | 已完成 | 18 | 11 | 0.463681 | — |
| M4_seed2026 | 已完成 | 18 | 11 | 0.471173 | — |
| M6_seed13 | 已完成 | 18 | 11 | 0.463839 | — |
| M6_seed42 | 已完成 | 15 | 8 | 0.462438 | — |
| M6_seed2026 | 已完成 | 12 | 5 | 0.465627 | — |

### 3. 核心比较与统计解释

- **M1 vs M0：**三个种子的 MAE 都改善，平均降低0.013152；三seed平均的条件区间为[-0.021008, -0.005164]。seed2026 单独的区间仍跨0。M1 的平均 Acc-2/F1 反而低于 M0，收益不能概括为所有指标均提升。
- **M3 vs M1：**三个种子均明显改善，平均降低0.047064；三seed平均及每个单seed区间均不跨0。当前最明确的收益来自给学生增加 T/A/V 适配能力。
- **M6 vs M3：**三seed平均降低0.003129（约0.67%），但仅2/3 seeds改善；seed2026反而高0.001890。三个单seed区间及三seed平均条件区间[-0.012249, +0.005558]均跨0，当前不能宣称RU稳定显著优于适配Full KD。M6平均Acc-2与M3相同，平均Acc-7略低，亦非所有指标占优。
- **M4 vs M3：**三 seed 平均 MAE 高 0.002705；仅 seed42 改善，三 seed 平均条件区间 `[-0.005851, +0.010799]` 跨 0。valid 上 Uniform Interaction 没有稳定优于 Full KD。
- **M6 vs M4：**三个种子的点估计均改善，平均降低 0.005835；但 seed42/2026 的单 seed 区间跨 0，三 seed平均条件区间上界约为 `+0.000003`，几乎贴零但仍包含零。可以报告方向一致和效应量，不能写成稳健显著优势。
- **主链结论：**valid 三 seed 平均点估计为 M6 < M3 < M4 < M1 < M0。最明确收益仍来自 T/A/V 适配；RU 相对 M3 的平均优势较小且区间跨 0，RU 相对 Uniform 的优势方向一致但统计边界接近零。

#### 逐seed及三seed平均的聚类bootstrap

Δ=candidate−baseline，负值为改善；1,871个valid utterance、300个原视频cluster、10,000次成对重采样、bootstrap seed2026。三seed平均区间沿用P0预定脚本：先对每个utterance的三个种子误差差值取平均，再按视频重采样；**条件于这三个已训练模型，不估计新的训练随机性**。区间未做多重比较校正，valid也用于选模。

| 比较 | 统计范围 | Δ MAE | 95% CI（未校正） |
|---|---|---:|---|
| M1 vs M0 | seed13 | -0.016693 | [-0.033156, -0.000419] |
| M1 vs M0 | seed42 | -0.016363 | [-0.030750, -0.001614] |
| M1 vs M0 | seed2026 | -0.006402 | [-0.019566, +0.006484] |
| M1 vs M0 | 三seed平均（条件区间） | -0.013152 | [-0.021008, -0.005164] |
| M3 vs M1 | seed13 | -0.045373 | [-0.063872, -0.027477] |
| M3 vs M1 | seed42 | -0.044259 | [-0.064588, -0.025719] |
| M3 vs M1 | seed2026 | -0.051560 | [-0.068172, -0.035935] |
| M3 vs M1 | 三seed平均（条件区间） | -0.047064 | [-0.062444, -0.032929] |
| M4 vs M3 | seed13 | +0.003293 | [-0.006595, +0.013395] |
| M4 vs M3 | seed42 | -0.002613 | [-0.012462, +0.007011] |
| M4 vs M3 | seed2026 | +0.007436 | [-0.006233, +0.021655] |
| M4 vs M3 | 三seed平均（条件区间） | +0.002705 | [-0.005851, +0.010799] |
| M6 vs M3 | seed13 | -0.007423 | [-0.019538, +0.004635] |
| M6 vs M3 | seed42 | -0.003855 | [-0.016794, +0.008908] |
| M6 vs M3 | seed2026 | +0.001890 | [-0.009618, +0.013394] |
| M6 vs M3 | 三seed平均（条件区间） | -0.003129 | [-0.012249, +0.005558] |
| M6 vs M4 | seed13 | -0.010716 | [-0.020785, -0.000997] |
| M6 vs M4 | seed42 | -0.001242 | [-0.011836, +0.009480] |
| M6 vs M4 | seed2026 | -0.005546 | [-0.015878, +0.004909] |
| M6 vs M4 | 三seed平均（条件区间） | -0.005835 | [-0.011697, +0.000003] |

该P0条件区间与新main_table_v1方案（逐seed区间、另报三seed点估计平均、不构造新的跨seed统一区间）的约定不同，后续不得混称。

### 4. MOSEI official-test 结果

15 个 run 均使用各自按 valid MAE 选出的 `best.pt`，在相同的 4,659 个 official-test utterance（4,687 个窗口）上评测。没有测试 `last.pt` 或逐 epoch checkpoint，也没有用 test 结果重新选择方法或轮次。下表为三 seed 均值±样本标准差；Acc-2/F1 排除真实标签为零的样本。

| 模型 | Test MAE ↓ | Pearson ↑ | Acc-2 ↑ | Weighted F1 ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|---:|
| M0 | 0.536696 ± 0.008767 | 0.767649 ± 0.005771 | 0.852412 ± 0.008165 | 0.852107 ± 0.007002 | 0.538814 ± 0.015761 |
| M1 | 0.527306 ± 0.002139 | 0.776850 ± 0.004184 | 0.848101 ± 0.005250 | 0.848309 ± 0.003787 | 0.543035 ± 0.002888 |
| M3 | 0.479825 ± 0.001906 | 0.816943 ± 0.004237 | 0.872776 ± 0.006062 | 0.872778 ± 0.005701 | 0.569793 ± 0.004943 |
| **M4** | **0.476242 ± 0.004540** | **0.823668 ± 0.001264** | **0.877821 ± 0.004429** | **0.877321 ± 0.004238** | **0.571367 ± 0.003611** |
| M6 | 0.481887 ± 0.000657 | 0.819403 ± 0.000701 | 0.874152 ± 0.006349 | 0.874376 ± 0.005601 | 0.568076 ± 0.003971 |

official test 上 M4 的五个汇总指标均最佳。M6 相对 M3 的平均 MAE 高 0.002062，相对 M4 高 0.005645；因此当前 test 结果不支持 R×U 优于 Uniform 或普通适配 Full KD。该观察不替代预注册的原视频聚类配对统计，亦不得据此回头筛选训练配置。

机器可读结果：`outputs/experiments/tav_main_v1/official_test_20260917/all_five_methods_summary.json`。目录中的 `M3_seed13/` 是正式前向前的 collator 冒烟失败记录；有效结果位于 `M3_seed13_retry1/`，汇总只读取后者。

### 5. 协议、运行与审计口径

- 统一在线原始T/A/V输入、teacher timing-v2；Full KD Probe seed2026，校准temperature=1.0114532709121704，KD temperature=2.0。
- batch8、accumulation1、lr1e-4、weight decay0.01、最多30 epochs、patience7，按valid MAE选模。M4/M6同一组三Probe均值interaction targets，仅权重不同。
- 15 个 run 的最佳 epoch/MAE 与 history、status、report 已交叉核对；official-test 清单、checkpoint 与源配置均保存 SHA-256。
- official test 已按用户明确授权完成一次性统一评测；结果只用于报告，不能用于配置或 checkpoint 选择。
- 历史 D/C/Video 协议结果保留在原报告中，不与本轮 TAV 协议混合排名；历史 LoRA(T+A) 也不能解释为 TA-only。
- M4_seed13 后期退化记录保留；另外两个 M4 种子均在 epoch11 取得最佳值，并于 epoch18 early stop。

### 6. 新主表、消融、MOSI 与逐 epoch 复训状态

本次读取了9月17日的方案、实现说明及 experiments.json，并检查 `outputs/experiments/main_table_v1/`。**MOSEI 和 MOSI 的资产目录均已存在，但未发现新主表训练的 status/history/report/predictions 产物。实现完成、dry-run 或资产准备不能列作实验成绩。**

| 部分 | 当前证据 | 汇总处理 |
|---|---|---|
| 固定学生主表A | 实现说明记录了Adapted Student-only、Full KD、Projector、EA-KD、RLD、SKD、Subset-7、Uniform、RU等入口；MOSEI protocol资产已存在 | 新协议尚无训练结果 |
| R-only / U-only / Ensemble Full / shuffled R×U / first-order等消融 | 已写入矩阵与实现说明 | 待执行，不填分数 |
| MOSI | 教师特征、三 Probe、交互目标、ensemble targets 和独立协议均已完成；计划含53次训练，39个run可直接启动、14个受选择门/原生资产阻塞 | 资产准备完成，尚未启动主表训练 |
| DLF / GsiT / DPDF-LQ | 矩阵中native_features_ready=false；实现说明记录缺标准特征及原生BERT资产 | 阻塞，不与固定学生成绩混列 |
| CMAD / Light-MER / CorrKD | 实现说明记录适配核查边界或官方来源限制，未发现对应训练结果 | 不列为已经复现的实验结果 |

编号映射须保持清楚：新主表中的 **Full KD 对应P0的M3**（已适配学生），不是冻结编码器的M1；**Adapted Student-only 不是M0**。Uniform/RU分别对应M4/M6的设计，但复用结果前仍需核对两个协议一致性，不能只改名当作已完成的新协议run。

为检查训练轨迹，已在独立目录 `outputs/experiments/tav_epoch_checkpoints_v1/` 启动 M0/M1/M3/M4/M6 各一次 seed13 复训。五个 run 当前均处于 epoch2，各已保存 epoch1 模型快照；每轮模型写入 `checkpoints/epoch_XXX.pt`，同时保留正常的 `best.pt/last.pt`。这些逐 epoch checkpoint 只用于分析，不进入 official-test 选择。

### 7. 后续与数据出处

等待五个逐 epoch 复训完成并审计训练轨迹；随后按9月17日方案开展独立主表与机制消融，重点检验 RU 相对适配 Full KD 的收益是否稳定，以及收益是否可由集成、七子集监督或单独 R/U 解释。MOSEI/MOSI 主表训练仍需显式启动，本轮状态更新没有自动启动这些任务。

本次结果源为 `outputs/experiments/tav_main_v1/students/*/{history,status,report}.json`、`outputs/experiments/tav_main_v1/summary.json`、`outputs/experiments/tav_main_v1/official_test_20260917/all_five_methods_summary.json`、MOSI `preparation_status.json` 及逐 epoch 复训 `queue_status.json`。历史机器可读 valid 汇总仍见[机器可读汇总](../../project/reports/tav_p0_results_20260917.json)。

相关文档：[总实验报告](../archive/reports/实验汇总报告.md)、[9月16日seed13快照](tav_main_v1.md)、[新主表方案](main_table_v1.md#experiment-plan)、[新实验实现说明](main_table_v1.md#implementation-notes)。

## 原文快照：`TAV主实验seed13结果汇总_20260916.md`

## TAV 主实验 seed13 最终结果（2026-09-16）

> 核对时间：2026-09-16T14:03:02+08:00（北京时间）。seed13 五个方法均已完成；其余种子进度为此时快照。

### 1. 结论与完成状态

**M0、M1、M3、M4、M6 的 seed13 全部完成，整条 seed13 技术审计通过。** 五个模型均按 valid MAE 选择最佳 checkpoint，并在连续 7 轮未改善后结束；所有 best epoch 已与 history.json、status.json 和最终 report.json 交叉核对。

本 seed 的最终 MAE 排序为 **M6 < M3 < M4 < M1 < M0**，满足预期的 **M6 < M3 < M1 < M0** 以及 **M6 < M4**。这支持继续按既定方案完成复种子，但还不能把单 seed 的结果写成三 seed 稳定结论。

M6 最佳值从此前快照的 epoch3 / 0.475440 更新为 **epoch11 / 0.463839**，最终训练到 epoch18 结束。

### 2. 最佳 epoch 与最终验证指标

MAE 越低越好，其余指标越高越好。所有指标均取最佳 MAE 所在 epoch；Acc-2 / Weighted F1 排除真实标签为零的样本，表内为比例。

| 模型 | 方法 | 完成 epochs | 最佳 epoch | Valid MAE ↓ | Pearson ↑ | Acc-2 ↑ | Weighted F1 ↑ | Acc-7 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | TAV Student-only | 19 | 12 | 0.533328 | 0.722086 | 0.832406 | 0.834069 | 0.533939 |
| M1 | TAV Full KD | 12 | 5 | 0.516635 | 0.740422 | 0.850487 | 0.848598 | 0.540887 |
| M3 | Full KD + T/A/V adaptation | 13 | 6 | 0.471261 | 0.786222 | 0.872740 | 0.872535 | 0.569214 |
| M4 | M3 + Uniform Interaction | 13 | 6 | 0.474555 | 0.792144 | 0.877608 | 0.876708 | 0.567076 |
| M6 | M3 + Reliability × Utility | 18 | 11 | 0.463839 | 0.797110 | 0.871349 | 0.872540 | 0.577231 |

M6 的 MAE、Pearson 和 Acc-7 在这五个 seed13 模型中最好；Acc-2 和 Weighted F1 仍是 M4 更高，因此不能称 M6 在所有指标上都最佳。

### 3. 核心比较与原视频聚类 bootstrap

Δ MAE = candidate − baseline，负值表示改善。对同一批 1,871 个 valid utterance 按 300 个原视频 cluster 成对重采样 10,000 次，固定 bootstrap seed2026。本次已刷新原先仅含 3 个完成 run 的 summary.json；以下五项均使用最终 checkpoint 的逐样本预测重新统计。

| 比较 | Δ MAE | 95% CI（未校正） | 解释 |
|---|---:|---|---|
| M1 vs M0 | -0.016693 | [-0.033156, -0.000419] | Full KD 有改善；区间不跨 0，接近边界 |
| M3 vs M1 | -0.045373 | [-0.063872, -0.027477] | T/A/V adaptation 有明显改善；区间不跨 0 |
| M4 vs M3 | +0.003293 | [-0.006595, +0.013395] | Uniform 点估计稍差；区间跨 0 |
| M6 vs M3 | -0.007423 | [-0.019538, +0.004635] | RU 点估计改善；区间跨 0 |
| M6 vs M4 | -0.010716 | [-0.020785, -0.000997] | RU 优于 Uniform；区间不跨 0 |

M6 相比 M3 降低 MAE **0.007423（约 1.58%）**，相比 M4 降低 **0.010716（约 2.26%）**。Uniform 未带来点估计收益，而选择性 RU 有改善，符合当前 motivation；不过 **M6 vs M3 的 95% CI 仍跨 0**，不能宣称这一比较已有稳健显著性。

所有区间均未作多重比较校正，且 valid 同时用于 checkpoint 选择。单 seed bootstrap 描述这些已训练模型面对原视频采样变化的不确定性，不替代训练随机性复验。仍需 seeds42/2026 的方向一致性及正式三 seed 汇总。

### 4. 同协议与技术核查

- 五个方法统一在线原始 T/A/V 输入；M0/M1 冻结编码器，M3/M4/M6 使用同样的 T/A/V LoRA。
- 教师为已修正视频时间信息的 teacher v2；Full KD Probe seed2026，校准 temperature=1.0114532709121704，KD temperature=2.0。
- M4/M6 使用同一套三 Probe 均值 interaction targets；区别为 Uniform 与 Reliability × Utility 权重。Utility 是训练集标签关联代理，不解释为因果 utility。
- official train/valid 原始 utterance 数分别为 16,326/1,871，窗口数为 16,399/1,879。
- batch size=8、gradient accumulation=1、learning rate=1e-4、weight decay=0.01、最多30 epochs、patience=7；按 valid MAE 选最佳 checkpoint。
- 各方法及完整 seed13 链审计均通过；计划记录的 81 个源码 SHA-256 本次核对无变化。审计按技术完整性放行，不以目标性能排序作门槛。
- official test 尚未使用。本次只做结果汇总与 CPU 上的 bootstrap，不改变训练配置。
- M4 后期退化记录保留：epoch13 MAE=0.614108，最后一个 batch 裁剪前梯度范数约4170.12，裁剪阈值1.0；最终仍选择epoch6。需观察其他种子是否重复出现。

### 5. 其他种子与自动补位

**P0 全部 15 runs：已完成 6 个、运行中 5 个、待运行 4 个。** 除五个 seed13 外，M1_seed2026 也已结束（最佳 epoch6，MAE=0.515298）。尚没有方法完成三个种子，暂不报告正式三 seed 均值。

| Run | 状态 | 进度 | 最佳 epoch | 最佳 valid MAE | GPU |
|---|---|---|---:|---:|---:|
| M0_seed42 | 运行中（暂时最佳） | epoch12，400/2050 steps | 7 | 0.526915 | 0 |
| M0_seed2026 | 运行中（暂时最佳） | epoch5，200/2050 steps | 4 | 0.533801 | 1 |
| M1_seed42 | 运行中（暂时最佳） | epoch18，900/2050 steps | 11 | 0.510552 | 0 |
| M1_seed2026 | 已完成 | 共13 epochs | 6 | 0.515298 | — |
| M3_seed42 | 运行中（暂时最佳） | epoch18，1/2050 steps | 15 | 0.466293 | 0 |
| M3_seed2026 | 运行中（暂时最佳） | epoch1，200/2050 steps | — | — | 1 |
| M4_seed42 | 待运行 | — | — | — | — |
| M4_seed2026 | 待运行 | — | — | — | — |
| M6_seed42 | 待运行 | — | — | — | — |
| M6_seed2026 | 待运行 | — | — | — | — |

GPU0 当前运行 M1_seed42、M3_seed42、M0_seed42；GPU1 运行 M0_seed2026、M3_seed2026。`rdid-tav-main-v1.service` 检查为 active，当前为双卡 3+2 个独立实验并行。M6_seed13 完成后，空出的 GPU1 槽位已自动补入 M3_seed2026。

所有方法的 seed13 门槛均已解除；M4_seed42、M4_seed2026、M6_seed42、M6_seed2026 按现有队列和资源预算等待自动补位。总上限仍为5，每卡上限3/2，冻结/适配方法预算16/34 GiB，每卡预留8 GiB；资源不足时允许少于5个并发。此次不调整队列优先级，不中断在训任务。

### 6. 下一步与复核入口

先完成主链全部三 seed，再汇总 M1 vs M0、M3 vs M1、M6 vs M3、M6 vs M4 的正式比较。随后执行 Reliability-only、Utility-only、R+U、R×U 和 frozen/adapted 等 P1 消融；视频使用诊断、MOSI、效率与 official test 继续后置。主链完成前不新增 KD 方法。

实时状态：

```bash
cd /home/wy/sjq/kd
watch -n 10 /home/wy/sjq/miniconda3/envs/kd/bin/python project/scripts/tav_distillation_status.py
```

来源：`outputs/experiments/tav_main_v1/students/*/history.json`、`status.json`、`report.json` 及 `predictions.jsonl`；队列 `status.json`、`seed13_chain_audit.json`、刷新后的 `summary.json`、`assets/protocol.json` 与 `plan.json`。

附：[机器可读最终结果与全队列快照](../../project/reports/tav_seed13_results_20260916.json)、[9月15日历史快照](tav_main_v1.md)、[执行方案](tav_main_v1.md)。上传内容仅含文档与聚合统计，不包含数据、权重或逐样本预测。

## 原文快照：`TAV主实验进展汇总_20260915.md`

## TAV 主实验进展汇总（2026-09-15）

> 本文保留9月15日历史快照。seed13 已全部完成，最新结果见 [9月16日最终汇总](tav_main_v1.md)。

> 采集时间：2026-09-15T19:37:42+08:00（北京时间）。这是固定时间快照，训练仍在继续。

### 1. 当前进度

**P0 共 15 runs：已完成 3 个，运行中 5 个，待运行 7 个。** 已完成的是 M1、M3、M4 的 seed13；M0、M6 的 seed13 仍在训练，同时运行 M1 的 seed42、seed2026 和 M3 的 seed42。

正式种子固定为 **13、42、2026**。每个方法的 seed13 完成且技术核查通过后，即可补该方法的另两个种子，无需等待整个 seed13 主链结束。截至本快照，没有方法完成全部三个种子，不能给出正式三 seed 均值与标准差。

### 2. 方法与实验口径

| 模型 | 配置 |
|---|---|
| M0 | TAV Student-only（编码器冻结） |
| M1 | TAV Full KD（编码器冻结） |
| M3 | TAV Full KD + T/A/V LoRA |
| M4 | M3 + Uniform Interaction |
| M6 | M3 + Reliability × Utility Interaction |

- 数据：official train 16,326 / valid 1,871 个原始 utterance；窗口数分别为 16,399 / 1,879。预测按原始 utterance 聚合。
- 输入：五个方法统一在线读取原始 T/A/V；历史 LoRA(T+A) 仍输入冻结视频特征，不能当作 TA-only。
- 教师：采用已修正视频时间信息的 teacher v2；Full KD 使用 Probe seed2026，校准 temperature 为 1.0114532709121704，KD temperature 为 2.0。
- 训练：batch size 8、gradient accumulation 1、learning rate 1e-4、weight decay 0.01；最多 30 epochs、early stopping patience 7，按 valid MAE 选择 checkpoint。
- M3/M4/M6 均适配 T/A/V LoRA；M4/M6 使用同一组三 Probe 均值 interaction targets，仅权重不同。Utility 使用训练集关联度代理，不作因果 utility 解释。
- 所有结果均为 official valid；official test 尚未使用。

### 3. 各实验进度与最佳 epoch

最佳值来自已经完成验证的 epoch；运行中的值是暂时最佳，不是最终结果。step 是训练进程最近一次写入状态文件的值，可能落后于实时进度。

| 模型 | seed | 状态 | 已完成 epochs | 当前训练进度 | 最佳 epoch | 最佳 valid MAE ↓ | GPU |
|---|---:|---|---:|---|---:|---:|---:|
| M0 | 13 | 运行中（暂时值） | 18 | epoch 19，900/2050 steps | 12 | 0.533328 | 0 |
| M0 | 42 | 待运行 | 0 | — | — | — | — |
| M0 | 2026 | 待运行 | 0 | — | — | — | — |
| M1 | 13 | 已完成 | 12 | — | 5 | 0.516635 | — |
| M1 | 42 | 运行中（暂时值） | 5 | epoch 6，1100/2050 steps | 3 | 0.535891 | 0 |
| M1 | 2026 | 运行中（暂时值） | 5 | epoch 6，800/2050 steps | 5 | 0.533765 | 1 |
| M3 | 13 | 已完成 | 13 | — | 6 | 0.471261 | — |
| M3 | 42 | 运行中（暂时值） | 5 | epoch 6，500/2050 steps | 4 | 0.470368 | 0 |
| M3 | 2026 | 待运行 | 0 | — | — | — | — |
| M4 | 13 | 已完成 | 13 | — | 6 | 0.474555 | — |
| M4 | 42 | 待运行 | 0 | — | — | — | — |
| M4 | 2026 | 待运行 | 0 | — | — | — | — |
| M6 | 13 | 运行中（暂时值） | 6 | epoch 7，800/2050 steps | 3 | 0.475440 | 1 |
| M6 | 42 | 待运行 | 0 | — | — | — | — |
| M6 | 2026 | 待运行 | 0 | — | — | — | — |

### 4. 最佳 MAE checkpoint 的其他指标

下列指标均取该 run 最佳 MAE 所在 epoch，没有分别选择各指标的最大值。Acc-2 与 F1 使用剔除零标签的口径；数值为比例。

| Run | 最佳 epoch | MAE ↓ | Pearson ↑ | Acc-2 ↑ | Weighted F1 ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| M0_seed13 | 12 | 0.533328 | 0.722086 | 0.832406 | 0.834069 | 0.533939 |
| M1_seed13 | 5 | 0.516635 | 0.740422 | 0.850487 | 0.848598 | 0.540887 |
| M1_seed42 | 3 | 0.535891 | 0.734962 | 0.845619 | 0.847931 | 0.540353 |
| M1_seed2026 | 5 | 0.533765 | 0.740315 | 0.861613 | 0.858660 | 0.535008 |
| M3_seed13 | 6 | 0.471261 | 0.786222 | 0.872740 | 0.872535 | 0.569214 |
| M3_seed42 | 4 | 0.470368 | 0.789021 | 0.884562 | 0.884509 | 0.580438 |
| M4_seed13 | 6 | 0.474555 | 0.792144 | 0.877608 | 0.876708 | 0.567076 |
| M6_seed13 | 3 | 0.475440 | 0.785844 | 0.879694 | 0.879500 | 0.570283 |

### 5. 当前可以支持的结论

seed13 暂时最佳值排序为 **M3 < M4 < M6 < M1 < M0**。M0、M6 尚未结束，排序仍可能变化。

| 比较（candidate − baseline） | seed13 Δ MAE | 状态与解释 |
|---|---:|---|
| M1 − M0 | -0.016693 | 暂时支持 Full KD 有效；M0 未结束 |
| M3 − M1 | -0.045373 | 两者已完成，适配学生有明显收益 |
| M4 − M3 | +0.003293 | 两者已完成，Uniform 点估计稍差 |
| M6 − M3 | +0.004178 | M6 未结束，当前暂未优于 M3 |
| M6 − M4 | +0.000885 | M6 未结束，当前暂未优于 Uniform |

**目前不能宣称 Reliability × Utility 已有效或论文主链已成立。** M3 的 seed42 暂时最佳 MAE 为 0.470368，与 seed13 的 0.471261 接近，是正向观察，仍需等待完成及第三个种子。

已完成 run 的现有原视频聚类 paired bootstrap（1,871 utterances、300 个原视频 cluster、10,000 次重采样）：

| 比较 | Δ MAE | 95% CI |
|---|---:|---|
| M3_vs_M1，seed13 | -0.045373 | [-0.063872, -0.027477] |
| M4_vs_M3，seed13 | +0.003293 | [-0.006595, +0.013395] |

M3 vs M1 的单 seed 区间不跨 0；M4 vs M3 的区间跨 0，不能据点估计认定 Uniform 明确有害。这些区间未作多重比较校正，且 valid 同时用于 checkpoint 选择，应结合三 seed 一致性解释。M1 vs M0、M6 vs M3、M6 vs M4 尚待相应 run 完成后统计；正式三 seed 比较均未完成。

### 6. 双卡并行与自动补位

宿主机检查：`rdid-tav-main-v1.service` 为 active，调度版本为 `p0-live-five-jobs-v5`。当前为独立实验之间的双卡与单卡多进程并行；每个训练 run 固定在一张卡上。

| GPU | 当前任务 | 已用 / 总显存（MiB） | 并发上限 |
|---|---|---:|---:|
| 0 | M0_seed13, M1_seed42, M3_seed42 | 52,601 / 85,651 | 3 |
| 1 | M1_seed2026, M6_seed13 | 40,700 / 85,651 | 2 |

- 总上限 5 个任务。冻结编码器方法按每任务 16 GiB、适配方法按 34 GiB 预算，每卡预留 8 GiB，并检查实测显存、主机内存和磁盘。
- 每完成一个任务，调度器会自动寻找已通过本方法 seed13 核查、且当前资源足够的任务补位；全部换成较重的适配任务时，不保证始终维持 5 个。
- 待运行：M0_seed42、M0_seed2026、M3_seed2026、M4_seed42、M4_seed2026、M6_seed42、M6_seed2026。其中 M0/M6 的新种子还需等待各自 seed13 完成并通过技术核查。
- 尚有空闲显存不等于应立即再加任务：当前已达用户授权的五任务上限，还需给峰值显存和验证阶段留余量。
- 最新五任务调度切换接管了原有训练 PID，没有中断 M0/M6。本次汇总也只读训练输出，不改变训练或调度配置。

### 7. 技术核查与需关注的问题

- M1、M3、M4 的 seed13 技术核查均通过；已核对计划记录的 81 个源码 SHA-256，当前无变化。M0/M6 的完成后核查及完整 seed13 链审计仍待运行。
- 核查关注输入路径、teacher v2 / temperature、LoRA 和 interaction target 一致性，不把预期性能排序作为放行门槛。
- M4_seed13 后期有不稳定迹象：epoch13 valid MAE 为 0.614108，最后一个 batch 的裁剪前梯度范数约 4170.12；代码裁剪阈值为 1.0。该记录是裁剪前值，不能解释成实际更新使用了这个范数。最终仍选用 epoch6 的 MAE 0.474555 checkpoint，后期退化应在复种子结果中继续观察。
- M0_seed13 当前已完成 epoch18，最佳在 epoch12，连续 6 轮未刷新；若 epoch19 仍未改善，按 patience=7 会触发早停，实际以训练输出为准。
- M4 不提升也继续 M6；主链完成前不新增 conflict gate、hidden KD、binary KD 或新的 weighting function。

### 8. 后续顺序

1. 自动完成 M0/M1/M3/M4/M6 × seeds13/42/2026，共 15 runs。
2. 汇总三 seed 均值、标准差及 M1 vs M0、M3 vs M1、M6 vs M3、M6 vs M4 的原视频聚类 bootstrap，明确区间及一致性。
3. P1：Uniform、Reliability-only、Utility-only、R+U、R×U 对照，以及适配与 interaction fidelity 分析。
4. P2：TA-only、frozen Video、Video LoRA、视频交换/帧扰动、MOSI、效率和最终 official test。

### 9. 实时查看与数据出处

在服务器项目目录下，可直接查看持续更新的队列与单任务状态：

```bash
cd /home/wy/sjq/kd
watch -n 10 cat outputs/experiments/tav_main_v1/status.json
## 单任务（按需要替换 run 名称）
cat outputs/experiments/tav_main_v1/students/M6_seed13/status.json
tail -n 40 outputs/experiments/tav_main_v1/logs/M6_seed13.log
systemctl --user status rdid-tav-main-v1.service --no-pager
nvidia-smi
```

本报告依据 `outputs/experiments/tav_main_v1/` 中的 `status.json`、各 run 的 `history.json` / `status.json` / `report.json`、`summary.json`、`seed13_method_audits/`、`assets/protocol.json` 和 `plan.json`。已完成 run 的最佳 epoch / MAE 已交叉核对历史与最终报告。

可下载的[机器可读汇总](../../project/reports/tav_p0_progress_20260915.json)保留全部精度、快照时间和聚合指标。上传范围为文档与聚合统计，不包含数据、模型权重、训练缓存或逐样本预测。

## 原文快照：`TAV主实验执行方案_20260914.md`

## TAV 主实验执行方案与项目核查

> 2026-09-16 15:39 六任务并行：按用户要求在 GPU1 同时补入 M6_seed2026，总上限调整为6，GPU0/GPU1各最多3个。保持冻结10 GiB、适配32 GiB预算及每卡8 GiB余量。目标组合为GPU0=M3_seed42+M0_seed42+M6_seed42，GPU1=M0_seed2026+M3_seed2026+M6_seed2026；M4两种子继续等待资源自动补位。原五个训练进程连续运行，调度测试17项通过。下方五任务记录为此前调度历史，以本条和 live_limits.json 为当前配置。

> 2026-09-16 15:34 显存准入校准：M1_seed42 完成后未补位的原因是旧预算把“一个冻结任务 + 两个适配任务 + 余量”计为16+34+34+8=92 GiB，超过单卡83.6 GiB。固定batch8配置下，已观测的进程总CUDA显存约8.3/29.7 GiB，分配峰值约5.91/26.24 GiB；现预算调整为冻结10 GiB、适配32 GiB，仍预留每卡8 GiB并检查实际可用显存，组合预算为82 GiB。保持总上限5、GPU0/1槽位3/2及M6优先顺序，四个在训进程连续运行。调度测试16项通过；变更证据保存在 outputs 下的 schedule_revisions。

> 2026-09-16 队列优先级调整：保持五个在训任务连续运行，后续自动补位顺序改为 **M6_seed42 → M6_seed2026 → M4_seed42 → M4_seed2026**。优先级由 `outputs/experiments/tav_main_v1/live_limits.json` 的 `pending_priority` 热更新；仍受每卡槽位和显存预算约束。此次仅替换调度器并接管原训练 PID，未重启训练。

> 2026-09-16 更新：M0/M1/M3/M4/M6 的 seed13 全部完成，最佳 MAE 分别为 0.533328 / 0.516635 / 0.471261 / 0.474555 / 0.463839。M6 最佳为 epoch11；M6 vs M3 的单 seed 95% CI 仍跨0。完整指标、bootstrap 与复种子进度见 [seed13 最终结果汇总](tav_main_v1.md)。

本文件落实 `docs/archive/plans/后续方案.md` 的 MOSEI 三模态蒸馏实验，按用户最新指令优先执行 M0/M1/M3/M4/M6 主链，记录实际代码口径、预注册选择和双卡调度。结果尚未产生时，不把预检或历史结果写入新主表。完整训练为多日任务，进度以状态文件为准。

2026-09-14 13:39（Asia/Shanghai）已启动并启用 `rdid-tav-main-v1.service`。启动前全项目 105 项测试通过；修复 checkpoint 中版本对象的序列化问题后，15 项定向测试和真实 M6 两轮连续/中断恢复对比通过（参数、预测逐值一致）。预检记录见 `outputs/experiments/tav_main_v1/preflight/validation.json`。首批为 GPU0 的 M0 seeds13/42、GPU1 的 M1 seeds13/42。

### 2026-09-14 最新执行优先级（替代此前 21-run 队列）

按最新要求，**seed13 也使用双卡、每卡最多 2 任务、总计最多 4 任务并行**。按 M0 → M1 → M3 → M4 → M6 的优先级派发，不要求前一模型训练结束才启动后一模型。逐模型完成后审计数据覆盖、teacher v2/temperature、同一 TAV input path、LoRA 参数/梯度和 interaction target 指纹；该模型技术审计通过即自动补它的 seeds42/2026。主表为 **5 × 3 = 15 runs**。seed13 和复制阶段均允许四任务并发；各模型的 seed13 完成且技术核查通过后，即可补该模型的 seeds42/2026，不再等待其他模型。

此前 M0/M1 seeds13/42 均未完成 epoch1，已停止并归档其日志、配置及目录；未伪称可恢复中途 step。新链从 seed13 M0 重新启动，模型和训练协议不变。原计划和调度源码保存在 `outputs/experiments/tav_main_v1/schedule_revisions/20260914_134554_seed13_first/`。

**技术审计不是性能 Gate**：M6<M3<M1<M0 和 M6<M4 是待验证假设，不是取消任务或筛选结果的条件。M4≈M3 或 M4>M3 均继续 M6；M6 失败也先分析，不继续叠加 conflict gate、hidden KD、binary KD 或新 weighting function。在五模型三 seed 主链结束前不新增 KD 方法。

P0 的四个重点 clustered bootstrap 为 M1 vs M0、M3 vs M1、M6 vs M3、M6 vs M4；另报告 M4 vs M3。审计产物是 `seed13_chain_audit.json`，其通过条件不含模型排序或改进阈值。

**P1 创新点消融**：Uniform / Reliability-only / Utility-only / R+U / R×U；冻结 vs adapted student；interaction fidelity 与 downstream MAE 的关系。Utility-only 已纳入计划，不能仅凭 Uniform/R/RU 的结果声称 R 与 U 互补。R+U 的尺度归一化和各消融权重应在执行该阶段前明确冻结，不在 P0 中新增实现或搜索。已有 M5 后移至 P1。

**P2 完整性实验**：TA-only、frozen Video（M2）、Video LoRA、cross-source swap、frame shuffle、repeat mid-frame、MOSI、独占设备效率和最终 official test。统一定位为 tri-modal utilization / diagnostic analysis；P0 所有任务使用 `--no-diagnostics`，视频诊断不插入主链。两张卡在 seed13 和复制阶段均并行多个主任务；主方法优先级仍高于 P1/P2。

2026-09-14 16:58 调度修订：用户要求 seed13 阶段立即启用双卡及卡内任务并行。旧串行调度已停止，M0 尚未完成首轮；旧记录归档后按相同训练协议重启。修订源码与原 plan 保存于 `schedule_revisions/20260914_165852_parallel_seed13/`。本次仅改变调度，batch、teacher、模型和损失均不变。

2026-09-15 调度更新：用户授权空位并行补新种子，改为逐模型审计后滚动补种子。M1/M3/M4 已完成的 seed13 可先扩展，M0/M6 无需提前结束。M0 从 epoch13 末、M6 从 epoch1 末恢复，未完成的当前 epoch 重算，训练代码与超参数不变。逐模型技术报告位于 `seed13_method_audits/{method}.json`，全部 seed13 完成后仍生成完整链审计。

2026-09-15 扩容更新：用户授权 **总计最多 5 个任务，GPU0 最多 3 个、GPU1 最多 2 个**，首个新增任务是 M3 seed42。新入口 `run_tav_live_queue.py` 接管已有训练 PID，不重启模型、不回退 epoch/step。轻量冻结模型每任务预算 16 GiB，adapted 模型预算 34 GiB，每卡余量 8 GiB；同时核查实际占用，预算不足则等待，因此全部为 LoRA 重任务时不承诺始终达到五并发。

完成一个任务后，调度器每 10 秒自动寻找已通过各自 seed13 核查且显存预算可容纳的待运行任务补位，无需人工启动。资源上限在 `outputs/experiments/tav_main_v1/live_limits.json` 中，每次循环重读；允许以后不重启进程调整到验证范围内的上限。修改并发上限不改变 batch、学习率或训练协议。

本次交接验证四个在训 PID 全部保留：M0 seed13=982875、M6 seed13=982894、M1 seed42=987819、M1 seed2026=989884。证据在 `schedule_revisions/20260915_111100_five_jobs_handover/handover.json`。源码、快照和 24 项测试记录保留在该修订目录。

**服务管理语义更新**：同名服务现在使用 `KillMode=process`。停止/重启服务只停止/替换调度器，已登记的训练进程继续运行，随后按 PID、启动时刻和 boot ID 接管；不是暂停所有训练的命令。运行登记在 `live_processes.json`。控制器遇到异常时停止补位并保留已有训练，状态明确写为 `controller_failed`，修复后恢复接管。若机器重启使原 PID 消失，则使用完整 epoch checkpoint 恢复。

### 项目核查结论

- 教师是冻结 Qwen3-Omni-30B-A3B-Instruct 的 Thinker 表示，加 MOSEI train 标签训练的共享 Probe；不是 zero-shot 评分教师。直接生成评分大量为零的问题不等于 Probe 输出为零。
- 新教师资产位于 `outputs/experiments/video_source_v2/teacher_features`，时间策略 `sampled_fps_v2`，18,278 个 train/valid 窗口 × 7 子集，127,946 项均完成且有限。
- 新 Probe seeds 为 2026/2027/2028；Full KD 固定 seed2026，不根据 valid 从三个 Probe 中挑选教师。对应校准温度 **1.0114532709121704**，蒸馏温度另为 **2.0**。
- 历史 `LoRATextAudioCachedVideoStudent` 实际输入在线 T/A 和缓存 V。因此 **LoRA(T+A) 表示适配范围，不等于 TA-only 输入**。冻结三模态和使用缓存 V 的历史实验不能直接统称为双模态机制实验；论文必须逐入口核对。真正 `ta_only` 才是双模态输入。
- 已完成的 timing-v2 全 Video LoRA 三 seed 为 0.467097 ± 0.003826 MAE，留作历史参考。M3 在新入口重跑，不混合历史 checkpoint 和新协议的结果。
- 旧训练器保留作历史复现；新主矩阵只从 `train_tav_distillation.py` 启动，显式拒绝 legacy 校准温度。

### 固定协议

数据只读取 MOSEI official train/valid：16,326 / 1,871 条 utterances，按原 source video 切分。窗口监督使用原 overlap aggregation weight，评估聚合回 utterance。official test 不参与输入指纹、训练、诊断或模型选择。

学生统一在线读取原始 T/A/V：Qwen3-0.6B-Base、WavLM-Base-Plus、VideoMAE-Base。Text 最多 256 tokens；Audio 为 16 kHz mono；Video 按原采样路径选 16 帧、同一 ImageProcessor。三路各压为 4 个 Q-Former tokens，再进入相同共享 Fusion Transformer 和分类/回归头。

各组均保留三模态输入。骨干 dropout 关闭，可训练 LoRA dropout=0.05。LoRA rank=8、alpha=16；适配全部 attention Q/K/V/output 投影，Text 28 层、Audio 12 层、Video 12 层。Video LoRA 共 48 个投影、589,824 个训练参数。M0/M1 的 T/A 零初始化 LoRA 冻结且 eval，仅用于保持公共参数初始化和计算入口一致；不训练编码器。

固定 seeds **13/42/2026**；physical batch=8、accumulation=1；AdamW lr=1e-4、weight_decay=.01；bf16 autocast；梯度裁剪 1.0；最多 30 epochs、patience=7；valid MAE 选模（改善阈值沿用原入口 1e-5）。不按中途性能取消某个 seed，也不使用 0.005 的硬筛选 Gate。

任务损失为 SmoothL1 + 0.5 CE；Full KD 为 SmoothL1 + 温度校准后的 T²KL，各权重 1。M0 完全不使用 KD 损失。

| ID | 输入 | 可训练骨干适配 | Full KD | Interaction |
|---|---|---|---|---|
| M0 | TAV | 无 | 无 | 无 |
| M1 | TAV | 无 | 有 | 无 |
| M2 | TAV | T+A LoRA，V 冻结 | 有 | 无 |
| M3 | TAV | T+A+V LoRA | 有 | 无 |
| M4 | TAV | 同 M3 | 有 | uniform |
| M5 | TAV | 同 M3 | 有 | reliability |
| M6 | TAV | 同 M3 | 有 | reliability × utility |

上表保留方法 ID 定义。**当前自动队列仅 M0/M1/M3/M4/M6，共 15 runs**；M5 为 P1，M2 为 P2，不自动插入 P0。

### 交互监督的具体选择

后续方案未指定的实现细节在查看新主矩阵结果前固定如下：

1. 坐标顺序为 T/A/V/TA/TV/AV/TAV，**七项全部监督**，包含 main effects 和 cross-modal terms；不是旧 pair-only 三项。七子集只重复运行小型 fusion，昂贵编码器每个 batch 只前向一次。
2. 每个 Probe 的 empty baseline 为其 official-train TAV utterance 均值。使用未移动的 subset scores 和该常数作 Möbius transform；Full KD 的回归目标不平移。学生 transform 的 baseline 固定为三个教师 baseline 的均值。
3. M4/M5/M6 均拟合同一组三 Probe 坐标均值，方差采用 ddof=1。三组差别仅在权重；Full KD 仍固定主 Probe seed2026。
4. Reliability = abs(mean) / sqrt(variance + 1e-4)，先逐样本除以七坐标均值，再 clip 到 [0.25,4]。
5. Utility 延续已有 RU 思路：按 train 的唯一 utterances 估计各坐标与标签的 absolute Pearson，floor=0.01 后均值归一化。**这是标签关联代理，不是已证明的因果/反事实效用，也不是学习式 cross-fit utility。**不使用 valid/test 标签估计权重。
6. M4 权重全 1；M5 使用 R；M6 使用 R×U，后两者最终逐样本七坐标均值归一化为 1。Interaction loss 是七项加权 SmoothL1 的均值，lambda_interaction=1。保持各方法损失尺度可比。
7. Möbius 变换只是可逆重参数化，不增加教师信息。是否改善下游任务由 M6 对 M3/M4/M5 的正式比较决定。

冻结文件 `outputs/experiments/tav_main_v1/assets/protocol.json` 保存实际 utility、empty baseline、所有教师预测/特征和清单 SHA-256。36,556 个实际音视频输入逐文件哈希另存 `student_input_manifest.jsonl`。

本轮不使用学生编码器特征缓存，故 `student_cache_manifest/checksum=null`，明确记录 online input policy 及 `student_input_manifest/checksum`，不能填写旧缓存指纹冒充本轮输入。每个 checkpoint/result 保存 teacher probe seed、温度、teacher feature manifest/checksum 和这些学生输入字段。

### 双卡与单卡并行

设备为两张 RTX 6000D，每卡 85,651 MiB。seed13 主链和复制阶段均采用独立实验任务并行，每卡最多 2 个训练进程，全机最多 4 个；同一 run 固定一个 seed、一张 GPU。改变并发数不会改变 batch 或梯度累积。

实际压力测试使用 8 个最长 30 秒 train 窗口，连续三次反向更新，两卡各两个进程同时运行：

| 方法 | 单进程峰值 allocated VRAM |
|---|---:|
| M0 | 5.75 GiB |
| M2 | 24.94 GiB |
| M3 | 24.95 GiB |
| M6 | 24.95 GiB |

M3/M6 同卡运行通过，活动 LoRA B 梯度有限且非零。这是压力预检，不是正式精度结果或独占设备吞吐 benchmark。历史全量 M3 入口最高约 26.24 GiB。

调度器按每进程 34 GiB、每卡额外 8 GiB 余量准入；同时检查实际显存、CPU 可用内存与磁盘。GPU 查询失败时等待，不视为空闲。不根据 OOM 自动降低 batch 或改超参数；失败停止本队列其余任务，保留最近完成 epoch 的 checkpoint，状态写为 failed。

不重建体积很大的学生特征缓存。开始时可用磁盘约 74 GiB，训练只保存最佳和最近 checkpoint（包含 optimizer/RNG），新任务准入要求仍有至少 15 GiB。中断前尚未完成第一轮的目录归档后从头跑；有 last.pt 则核对协议后恢复。正式训练前验证 M6 中断恢复与连续运行的参数及预测逐值一致。

### 结果、统计与诊断

每组保存 MAE、Pearson、Acc2、F1、Acc7；Acc2/F1 主口径去掉真值零样本，并保存含零版本；Acc7 使用回归分数取整裁剪。报告三 seed 均值和样本 SD。

P0 主比较 M1−M0、M3−M1、M6−M3、M6−M4；补 M4−M3。M2/M5 对应比较留待 P1/P2。MAE 使用至少 **10,000 次按原 video ID 的 paired clustered bootstrap**，逐 seed 和三 seed 平均 Δ 均保存 95% CI。后者条件于已训练的三个模型，不把 seed×utterance 当独立样本，也不宣称估计了全部训练随机性。CI 未作多重比较校正，结合主比较、seed 一致性、辅助指标和实际效应量解释。

P2 再对 M3/M6 的最佳 checkpoint 进行 5 次跨 source 视频交换、frame shuffle、repeat mid-frame；保存逐样本预测和视频聚类 ΔMAE。P0 禁止训练结束立即插入这些诊断。诊断不改变后续是否保留 Video。

15 组全部完成后自动生成 P0 主表；期间可手动调用汇总脚本更新已完成模型的结果。不以方法是否提升为完成条件。

### 后续阶段的依赖与设计

以下是主矩阵之后的阶段，**未放入当前 15-run P0 自动队列**：

- TA-only：同协议、同教师 TAV target、三 seed 的单独重训消融，仍只用于分析；不自动恢复旧 `video_source_v2` 的暂缓队列。
- 权重/utility 分析：按七坐标绘制 R、U、R×U 分布和 main/pair/triple 分组重建误差，对照 ΔMAE；不能把相关性作因果效用结论。
- 表示分析：对固定 train/valid 子样本比较冻结与适配 encoder 表示，保持样本、池化与教师层一致；结果只作诊断，不按 test 选择层。
- MOSI 已有 train/valid 清单（1,284/229 utterances，1,286/229 windows），但尚无同协议教师缓存/Probe。先用相同 timing-v2、七子集、共享 Probe 生成 MOSI 自有教师资产与 train-only 权重，然后 M0/M1/M3/M6 × 3 seeds 共 12 runs。MOSEI/MOSI 的任务 Probe 和校准须分别拟合；不得拿 MOSEI 标签或 utility 当 MOSI 统计。
- 效率：在训练队列结束、设备独占时，固定真实 utterance 长度分桶；分别测教师和 M0/M1/M3/M6 的总参数、训练参数、峰值显存、batch=1 p50/p95 latency、固定 batch throughput，并区分媒体解码和纯模型时间。预热后 CUDA synchronize，FLOPs/MACs 仅报告 profiler 实际覆盖的算子；并发训练耗时不作部署延迟。
- Robustness：固定 final checkpoint 做缺失 T/A/V、音频噪声和视频帧缺失等已声明扰动，在 valid 上报告正常−扰动差值；不以扰动结果反复筛选 test 配置。
- 最终 architecture、LoRA、lambda、teacher、temperature、weighting、epochs/lr、selection 规则全部锁定后，再为关键四方法统一做一次 official test；本轮服务没有 test 推理入口。

### 使用与产物

新入口：`project/scripts/train_tav_distillation.py`；调度：`project/scripts/run_tav_distillation.py`；汇总：`project/scripts/summarize_tav_distillation.py`。

后台服务名为 `rdid-tav-main-v1.service`。状态及日志位于独立目录 `outputs/experiments/tav_main_v1`；旧实验目录保留。

```bash
/home/wy/sjq/miniconda3/envs/kd/bin/python /home/wy/sjq/kd/project/scripts/tav_distillation_status.py
systemctl --user status rdid-tav-main-v1.service --no-pager
tail -F /home/wy/sjq/kd/outputs/experiments/tav_main_v1/pipeline.log
```

人工停止和恢复同一冻结队列：

```bash
systemctl --user stop rdid-tav-main-v1.service
systemctl --user restart rdid-tav-main-v1.service
```

阶段汇总为 `outputs/experiments/tav_main_v1/results.md` / `summary.json`，各 run 在 `students/M{0..6}_seed{13,42,2026}`。源码指纹和副本在 `plan.json` / `source_snapshot`。修改冻结源码或资产会阻止继续启动，不静默混合版本。
