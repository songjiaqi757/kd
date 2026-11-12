# Phase A 综合机制诊断与研究方向探索

生成时间：2026-11-12T05:43:15.077621+08:00。状态：已汇总所有终态任务；严格完成度见下表。

代码 commit `259673b02cedfad31c10a1e8632df382784efaa0`；只复用既有 Teacher/Student，optimizer steps=0；未使用 Test 选择模型、子集、权重或研究方向。分析口径与冻结执行协议见 protocol/plan.json及后续amendment。

| task | status | evidence |
| --- | --- | --- |
| A1 | 完成 | 四数据集逐样本 Acc-2 四象限、子集/阶数误差、source-video 汇总、固定规则案例 |
| A2 | 完成 | Teacher各Probe/ensemble与Student分开；Pearson/Spearman、原始/标准化谱、秩、条件数 |
| A3 | 完成（解释受限） | 127+empty组合、5外层/3内层 grouped CV、两utility、Shapley与去TAV敏感性；受Probe既有拟合及官方split重叠限制 |
| A4 | 完成 | Probe-head方差、ICC、差值分布和尺度；不解释为独立Teacher不确定性 |
| A5 | 受限 | 36/36 checkpoint完成；13个预定阶段缺少历史权重 |
| A6 | 完成（解释受限） | 归一化fidelity、两种精确reconstruction、grouped CV阶数utility及fidelity-task关联；无训练消融 |
| A7 | 完成（历史记录受限） | 数据指纹、source/parent重叠、Teacher target、Student seed、checkpoint/config与N2审计；历史缺失git信息保留缺失 |
| A8 | 完成 | 本报告与phaseB_decision.md；未启动Phase B训练 |

## 1. 四数据集 Teacher–Student 差距

以下仅列 seed13 的独立历史 run；不跨 campaign/target/预算拼接均值。Teacher列为 ensemble3 TAV，原始Full KD训练监督未必与该列相同。主表为原始 score MAE；SIMS官方clip口径另存 teacher_student_gap.csv。

| dataset | method | epoch | teacher_MAE | student_MAE | gap |
| --- | --- | --- | --- | --- | --- |
| mosi | full_kd | 19 | 0.4889 | 0.7330 | 0.2440 |
| mosi | subset7 | 11 | 0.4889 | 0.7838 | 0.2948 |
| mosi | uniform_interaction | 15 | 0.4889 | 0.7590 | 0.2700 |
| mosei | full_kd | 6 | 0.4339 | 0.4721 | 0.0382 |
| mosei | subset7 | 9 | 0.4339 | 0.4694 | 0.0355 |
| mosei | uniform_interaction | 6 | 0.4339 | 0.4750 | 0.0411 |
| sims | full_kd | 15 | 0.2700 | 0.4030 | 0.1331 |
| sims | subset7 | 16 | 0.2700 | 0.4074 | 0.1374 |
| sims | uniform_interaction | 18 | 0.2700 | 0.3989 | 0.1289 |
| sims_v2 | full_kd | 17 | 0.2502 | 0.3700 | 0.1198 |
| sims_v2 | subset7 | 16 | 0.2502 | 0.3659 | 0.1157 |
| sims_v2 | uniform_interaction | 20 | 0.2502 | 0.3607 | 0.1105 |

## 2. 四象限与误差模式

按 Acc-2 正确性划分，英语零标签排除后另记Q0，SIMS零包含。下面为ICD seed13 Validation的Q2；只定位最终预测差距，不认定某一坐标导致失败。全部Train/Validation、36模型、各象限标签强度及子集/阶数误差见 sample_quadrants/。

| dataset | n | fraction | teacher_MAE | student_MAE | label_abs_median |
| --- | --- | --- | --- | --- | --- |
| mosi | 19 | 0.0880 | 0.2988 | 1.8600 | 1.2000 |
| mosei | 82 | 0.0570 | 0.4144 | 0.9748 | 0.6667 |
| sims | 63 | 0.1382 | 0.3297 | 0.8185 | 0.6000 |
| sims_v2 | 73 | 0.1128 | 0.2861 | 0.8368 | 0.6000 |

## 3. 七子集冗余与几何

Teacher ensemble Validation原始协方差谱如下。相关性和低有效秩不能单独证明子集无条件价值；Möbius变换可逆，不产生新Teacher信息。标准化谱、Teacher各Probe和全部Student分别保存。

| dataset | space | effective_rank | stable_rank | condition_number |
| --- | --- | --- | --- | --- |
| mosi | prediction | 1.2640 | 1.0444 | 1079.3201 |
| mosi | interaction | 1.6059 | 1.1239 | 1070.2698 |
| mosei | prediction | 1.4680 | 1.0894 | 669.0047 |
| mosei | interaction | 2.0057 | 1.2540 | 406.2564 |
| sims | prediction | 1.5612 | 1.1149 | 626.8133 |
| sims | interaction | 1.8883 | 1.1939 | 613.0359 |
| sims_v2 | prediction | 1.4816 | 1.0966 | 807.1450 |
| sims_v2 | interaction | 1.6527 | 1.1368 | 948.1487 |

## 4. 条件增量效用与Shapley

Train内嵌套source-video grouped CV：外5折、内3折，Ridge alpha=[.01,.1,1,10,100]，所有组合相同预算；标准化仅由训练折拟合。两种utility严格分开（response=-MSE，task=-MAE）。TAV作为response输入有近乎平凡的泄露目标性质，因此另列六玩家去TAV敏感性。以下task Shapley去TAV表，stable_positive要求所有外折和novel-source Validation增量均为正，且novel Validation至少30个样本/5个source groups；这是候选筛查而非显著性检验。SIMS无novel Validation，SIMS v2仅1条，均不得升级为稳定排名。

| dataset | subset | CV_Shapley | fold_std | positive_fraction | valid_all | valid_novel | stable_positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mosi | t | 0.1730 | 0.0212 | 1.0000 | 0.1882 | 0.1882 | True |
| mosi | a | 0.2030 | 0.0214 | 1.0000 | 0.2085 | 0.2085 | True |
| mosi | v | 0.0142 | 0.0037 | 1.0000 | 0.0058 | 0.0058 | True |
| mosi | ta | 0.1933 | 0.0221 | 1.0000 | 0.1930 | 0.1930 | True |
| mosi | tv | 0.1711 | 0.0218 | 1.0000 | 0.1793 | 0.1793 | True |
| mosi | av | 0.1928 | 0.0225 | 1.0000 | 0.1879 | 0.1879 | True |
| mosei | t | 0.0712 | 0.0048 | 1.0000 | 0.0614 | 0.0614 | True |
| mosei | a | 0.0770 | 0.0049 | 1.0000 | 0.0673 | 0.0673 | True |
| mosei | v | 0.0182 | 0.0046 | 1.0000 | 0.0105 | 0.0105 | True |
| mosei | ta | 0.0828 | 0.0061 | 1.0000 | 0.0676 | 0.0676 | True |
| mosei | tv | 0.0810 | 0.0047 | 1.0000 | 0.0666 | 0.0666 | True |
| mosei | av | 0.0860 | 0.0058 | 1.0000 | 0.0783 | 0.0783 | True |
| sims | t | 0.0433 | 0.0036 | 1.0000 | 0.0380 | 缺失 | False |
| sims | a | 0.0578 | 0.0076 | 1.0000 | 0.0501 | 缺失 | False |
| sims | v | 0.0508 | 0.0117 | 1.0000 | 0.0525 | 缺失 | False |
| sims | ta | 0.0570 | 0.0046 | 1.0000 | 0.0499 | 缺失 | False |
| sims | tv | 0.0623 | 0.0119 | 1.0000 | 0.0593 | 缺失 | False |
| sims | av | 0.0761 | 0.0129 | 1.0000 | 0.0715 | 缺失 | False |
| sims_v2 | t | 0.0533 | 0.0059 | 1.0000 | 0.0422 | 0.0672 | False |
| sims_v2 | a | 0.0647 | 0.0052 | 1.0000 | 0.0505 | 0.0653 | False |
| sims_v2 | v | 0.0573 | 0.0050 | 1.0000 | 0.0545 | 0.0821 | False |
| sims_v2 | ta | 0.0644 | 0.0083 | 1.0000 | 0.0531 | 0.0727 | False |
| sims_v2 | tv | 0.0668 | 0.0064 | 1.0000 | 0.0629 | 0.0661 | False |
| sims_v2 | av | 0.0788 | 0.0064 | 1.0000 | 0.0725 | 0.0790 | False |

平均Shapley正值不等于给定所有其他子集后仍不可替代；完整上下文条件增量见full_context_marginal_summary.csv。是否删除某个子集必须结合这两类证据及真实训练消融。
## 5. Probe-head与残差稳定性

三Probe共用Teacher hidden cache；ICC(A,1)与方差是Probe-head敏感性，不能解释为独立Teacher epistemic uncertainty。下表比较最终TAV原始响应与三阶残差。

| dataset | space | mean_probe_variance | ICC | coordinate_std |
| --- | --- | --- | --- | --- |
| mosi | subset | 0.0193 | 0.9919 | 1.5448 |
| mosi | interaction | 0.0400 | 0.8924 | 0.5837 |
| mosei | subset | 0.0097 | 0.9888 | 0.9267 |
| mosei | interaction | 0.0324 | 0.8610 | 0.4583 |
| sims | subset | 0.0052 | 0.9843 | 0.5728 |
| sims | interaction | 0.0131 | 0.9119 | 0.3725 |
| sims_v2 | subset | 0.0040 | 0.9878 | 0.5709 |
| sims_v2 | interaction | 0.0054 | 0.9734 | 0.4484 |

## 6. Gradient compatibility

每checkpoint六个固定Train microbatch，eval/autograd、分参数组、无更新。原始目标保留其single-probe/ensemble差异；N2目标按历史No-CE+ensemble Full KD+0.30 ICD单独报告。以下ICD辅助项相对task全参数余弦；分组/范数比/中位数与负比例见 gradient_compatibility/。负余弦不是投影优化有效性的证据。

| dataset | method | epoch | group | mean_cosine | negative_fraction | norm_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| mosi | full_kd | 16 | all | 0.0975 | 0.5000 | 1.6084 |
| mosi | full_kd | 19 | all | 0.3230 | 0.3333 | 3.1783 |
| mosi | full_kd | 20 | all | 0.2521 | 0.3333 | 1.7032 |
| mosi | subset7 | 9 | all | 0.4793 | 0.1667 | 1.7015 |
| mosi | subset7 | 11 | all | 0.3494 | 0.1667 | 3.1093 |
| mosi | subset7 | 20 | all | 0.0914 | 0.5000 | 1.2615 |
| mosi | uniform_interaction | 13 | all | 0.1915 | 0.3333 | 1.0038 |
| mosi | uniform_interaction | 15 | all | 0.2592 | 0.5000 | 1.6322 |
| mosi | uniform_interaction | 20 | all | 0.1254 | 0.3333 | 1.6017 |
| mosei | full_kd | 5 | all | 0.1111 | 0.3333 | 0.7332 |
| mosei | full_kd | 6 | all | 0.2263 | 0.3333 | 0.4907 |
| mosei | full_kd | 20 | all | 0.0415 | 0.5000 | 0.6311 |
| mosei | subset7 | 5 | all | 0.5786 | 0.1667 | 0.5586 |
| mosei | subset7 | 9 | all | -0.0913 | 0.5000 | 0.5531 |
| mosei | subset7 | 20 | all | 0.0593 | 0.5000 | 0.4266 |
| mosei | uniform_interaction | 4 | all | 0.5473 | 0.1667 | 0.6189 |
| mosei | uniform_interaction | 6 | all | 0.8101 | 0.0000 | 0.1583 |
| mosei | uniform_interaction | 20 | all | 0.3358 | 0.3333 | 0.1634 |
| sims | full_kd | 4 | all | 0.0172 | 0.5000 | 0.9137 |
| sims | full_kd | 15 | all | -0.3385 | 0.6667 | 2.7912 |
| sims | full_kd | 20 | all | -0.2734 | 0.6667 | 1.5565 |
| sims | subset7 | 16 | all | -0.1986 | 0.5000 | 1.3045 |
| sims | subset7 | 17 | all | -0.3330 | 0.6667 | 0.5227 |
| sims | subset7 | 20 | all | -0.0783 | 0.5000 | 0.4410 |
| sims | uniform_interaction | 4 | all | -0.0536 | 0.5000 | 0.1884 |
| sims | uniform_interaction | 18 | all | -0.2801 | 0.6667 | 0.5129 |
| sims | uniform_interaction | 20 | all | -0.3105 | 0.8333 | 0.3000 |
| sims_v2 | full_kd | 1 | all | 0.0048 | 0.5000 | 0.9315 |
| sims_v2 | full_kd | 17 | all | 0.0107 | 0.3333 | 9.1764 |
| sims_v2 | full_kd | 20 | all | -0.1610 | 0.6667 | 7.6123 |
| sims_v2 | subset7 | 13 | all | -0.1249 | 0.6667 | 4.9692 |
| sims_v2 | subset7 | 16 | all | -0.1411 | 0.5000 | 7.2407 |
| sims_v2 | subset7 | 20 | all | -0.0354 | 0.5000 | 17.6226 |
| sims_v2 | uniform_interaction | 4 | all | -0.0460 | 0.5000 | 1.6793 |
| sims_v2 | uniform_interaction | 20 | all | -0.0160 | 0.3333 | 2.1013 |
| mosi | historical_N2 | 23 | all | 0.2585 | 0.3333 | 0.8373 |

在SIMS的ICD保留checkpoint中，epoch18/20全参数task–ICD平均cosine约为-0.280/-0.311，六个microbatch中负余弦比例为4/6和5/6；MOSI对应ICD保留checkpoint的平均cosine仍为正。这支持优先检查SIMS上的loss compatibility及参数组，而不支持“所有数据集普遍梯度冲突”的结论。样本数仍很小，且Early/Middle覆盖不完整，不能据此直接宣称PCGrad会改善性能。
历史权重缺失 13 个阶段，Early/Middle/Late覆盖不能视为完整；未通过重训补造权重。

## 7. 不同阶可吸收性与任务价值

各阶MAE、RMSE及相对Teacher Train坐标std归一化RMSE均保存；数学重建分别报告最终TAV与全部七响应。下面是novel-source Validation累计阶数离线utility；空输入是Train均值，full third-order包括全部七坐标。该分析器的增益不是实际Student删除阶数的消融收益。样本fidelity-task Pearson/Spearman及象限差异见 interaction_order_utility/。另补充所有保留checkpoint上六个固定Train windows的阶数fidelity/task配对改善（checkpoint_order_trajectory.csv、checkpoint_fidelity_task_association.csv），并列历史Validation MAE。该轨迹仅覆盖六个窗口、非均匀保留epoch且有checkpoint选择偏差，不能推断因果或完整训练动力学。

| dataset | utility | through_order | score | n |
| --- | --- | --- | --- | --- |
| mosi | response | 0 | -2.3777 | 229 |
| mosi | response | 1 | -0.0614 | 229 |
| mosi | response | 2 | -0.0243 | 229 |
| mosi | response | 3 | -0.0000 | 229 |
| mosi | task | 0 | -1.4188 | 229 |
| mosi | task | 1 | -0.4536 | 229 |
| mosi | task | 2 | -0.4552 | 229 |
| mosi | task | 3 | -0.4610 | 229 |
| mosei | response | 0 | -0.8609 | 1871 |
| mosei | response | 1 | -0.0433 | 1871 |
| mosei | response | 2 | -0.0146 | 1871 |
| mosei | response | 3 | -0.0000 | 1871 |
| mosei | task | 0 | -0.7778 | 1871 |
| mosei | task | 1 | -0.4408 | 1871 |
| mosei | task | 2 | -0.4260 | 1871 |
| mosei | task | 3 | -0.4269 | 1871 |
| sims_v2 | response | 0 | -0.8842 | 1 |
| sims_v2 | response | 1 | -0.0052 | 1 |
| sims_v2 | response | 2 | -0.0000 | 1 |
| sims_v2 | response | 3 | -0.0000 | 1 |
| sims_v2 | task | 0 | -0.6816 | 1 |
| sims_v2 | task | 1 | -0.3031 | 1 |
| sims_v2 | task | 2 | -0.2491 | 1 |
| sims_v2 | task | 3 | -0.2465 | 1 |

## 8. 数据与协议限制

| comparison | shared_parent_ids | sims_parent_fraction | sims_v2_parent_fraction | shared_source_video_ids | shared_parent_cross_split |
| --- | --- | --- | --- | --- | --- |
| SIMS_vs_SIMSv2_train_valid | 1824 | 1.0000 | 0.5414 | 60 | 0 |

官方Train/Validation源视频重叠另列source_video_overlap.csv；SIMS与v2不能描述为独立外部复现。原始single-Probe Full KD与ensemble额外目标混杂仍需统一target训练。历史N2是探索性seed13、epoch23、40epoch上限；不能充当20epoch确认结果。既有Probe可能用过Validation，因此外部Validation utility仍属于诊断；nested CV只保证离线analyzer的训练分组与调参隔离。

资产重核异常：0。缺失历史checkpoint、git记录或其他资产均保留缺失，不替代、不伪造。

## 9. 审稿意见证据范围

N7yq：A2/A3/A6可提供七子集冗余和条件增量的分析证据；自适应选择、双模态泛化及真实收益仍需新增训练。QhEZ：明确Möbius为可逆监督几何、residual contrast术语及Probe ensemble细节；几何独特性、统一target公平比较仍需训练。2Z9A：梯度/fidelity/数据协议可核验；多Seed确认性收益、配对差值置信区间和Matched-Spectrum几何控制不能由这些离线结果替代。

## 10. 研究决策

以下是直接由已计算数据形成的机制线索：

| dataset | residual_to_raw_TAV_probe_variance | raw_TAV_ICC | residual_TAV_ICC | third_order_task_cv_increment | third_order_positive_fold_fraction |
| --- | --- | --- | --- | --- | --- |
| mosi | 2.0661 | 0.9919 | 0.8924 | 0.0030 | 0.8000 |
| mosei | 3.3462 | 0.9888 | 0.8610 | 0.0054 | 1.0000 |
| sims | 2.5090 | 0.9843 | 0.9119 | 0.0046 | 0.8000 |
| sims_v2 | 1.3595 | 0.9878 | 0.9734 | 0.0034 | 1.0000 |

四数据集的原始七响应都呈低有效秩，但去TAV的六个子集task Shapley均在五个外折为正；这支持“存在冗余但条件上下文仍有价值”的检验假设，不支持仅凭相关性删除子集。三阶residual的Probe-head方差高于原始TAV且ICC较低，支持比较尺度归一化与匹配谱控制，尚不证明新几何会改善任务。第三阶的task utility增量应结合逐折符号与外部novel-source检查；完整response重建接近零误差来自Möbius可逆性，不能当作新发现或新信息。

见 phaseB_decision.md。Phase A没有预设新方法，不把离线Shapley当作训练权重收益，不把相关性当作因果贡献。下一阶段以统一target、20epoch和独立Student seeds进行验证；Phase B训练尚未授权启动。
