# Phase B 决策与最小可检验矩阵

当前Phase A状态：分析终态；存在资产/解释限制，不能宣称所有诊断完整。

## 证据驱动候选

| dataset | residual_to_raw_TAV_probe_variance | raw_TAV_ICC | residual_TAV_ICC | third_order_task_cv_increment | third_order_positive_fold_fraction |
| --- | --- | --- | --- | --- | --- |
| mosi | 2.0661 | 0.9919 | 0.8924 | 0.0030 | 0.8000 |
| mosei | 3.3462 | 0.9888 | 0.8610 | 0.0054 | 1.0000 |
| sims | 2.5090 | 0.9843 | 0.9119 | 0.0046 | 0.8000 |
| sims_v2 | 1.3595 | 0.9878 | 0.9734 | 0.0034 | 1.0000 |

具体可证伪假设H1：尺度敏感性导致部分ICD优化困难；若统一target/预算下normalization不优于diagonal与matched-spectrum控制，则不接受其独特机制。H2：高Shapley但完整上下文低增量意味着监督冗余；只有固定Train内选择且真实训练删减不损失任务，才接受subset reduction。H3：阶数fidelity与任务收益并不等价；只有真实阶数loss消融与离线方向一致才接受Order-aware机制。H4：MOSI较大Teacher–Student gap可能与loss balance有关；若FullKD-NoCE与ICD-NoCE改善相近，不能把NoCE收益归于ICD。

去TAV的task Shapley在所有外层fold及novel-source Validation均为正的坐标如下；如为空，则不支持升级Utility-aware主方法。正增量也只支持下一轮训练假设，不说明训练权重应直接等于该数值。

| dataset | subset | CV_Shapley | fold_std | valid_novel |
| --- | --- | --- | --- | --- |
| mosi | t | 0.1730 | 0.0212 | 0.1882 |
| mosi | a | 0.2030 | 0.0214 | 0.2085 |
| mosi | v | 0.0142 | 0.0037 | 0.0058 |
| mosi | ta | 0.1933 | 0.0221 | 0.1930 |
| mosi | tv | 0.1711 | 0.0218 | 0.1793 |
| mosi | av | 0.1928 | 0.0225 | 0.1879 |
| mosei | t | 0.0712 | 0.0048 | 0.0614 |
| mosei | a | 0.0770 | 0.0049 | 0.0673 |
| mosei | v | 0.0182 | 0.0046 | 0.0105 |
| mosei | ta | 0.0828 | 0.0061 | 0.0676 |
| mosei | tv | 0.0810 | 0.0047 | 0.0666 |
| mosei | av | 0.0860 | 0.0058 | 0.0783 |

1. Utility-aware：只在各折和novel-source检查稳定时提出候选；候选冻结后，用相同初始化、target、预算与uniform ICD配对训练。若排名不稳定，保留均匀基线。
2. Stable Geometry：联合Probe-head variance、ICC及各坐标scale检验，不因第三阶原始幅度大就判为噪声；比较统一尺度normalization与matched-spectrum/diagonal控制。不能解释为增加Teacher信息。
3. Order-aware/Absorbability-aware：只有归一化拟合难度高且task增量稳定时考虑absorbability；若高阶task价值小且跨折一致，再提出reduced-order。验证为实际loss消融，离线reconstruction不能代替。
4. Gradient-compatible：SIMS ICD在epoch18/20的task–ICD全参数均值均为负（约-0.280/-0.311，负比例4/6和5/6），可优先检验loss balance及参数组冲突；MOSI保留ICD checkpoints的均值为正，不提出普遍冲突。若配对训练的梯度投影/平衡不优于同等target的NoCE或尺度控制，则否定其额外机制。缺少Early权重时结论局限于保留阶段。
5. Fidelity不等于Task：同时报告response与task utility、样本fidelity-task关联及不同象限；若方向不一致，不按Teacher fidelity单独分配权重。

## 最小确认矩阵（待研究决策；未启动）

| 阶段 | 数据/预算 | 方法 | 要回答的问题 |
| --- | --- | --- | --- |
| B0 loss balance | MOSI，统一20epochs，Student13/42/2026，固定共同ensemble3 target/初始化 | Full KD、Subset7、ICD × CE有/无；NoCE ICD lambda0.30与CE ICD lambda0.30同权重配对，另保留历史lambda1参照 | No-CE收益是否独立于ICD；lambda/CE是否混杂；N2能否在20epoch确认 |
| B1 geometry | B0共同target/预算/seed，冻结同一CE状态与权重 | ICD、direct Subset7、Random Orthogonal、Diagonal、Matched-Spectrum | 改善是否由Möbius结构而非一般几何/尺度带来 |
| B2 一个机制候选 | 由Phase A最稳定证据选一个候选，MOSI先验冻结，3seeds | uniform ICD对一个新候选及相应单因素消融 | 新信号是否带来真实Student收益；不并行盲搜多个Adaptive方法 |
| B3 外部迁移/双模态 | MOSEI、SIMS/v2及双模态；相同配对seed和预算 | 只迁移B0/B2确认的最小方法组 | 跨数据收益与双模态适用性；SIMS/v2重叠分层披露 |

所有方法以Validation MAE选checkpoint，记录best epoch和epoch上限。Teacher Probe seeds不冒充Student seeds；不同campaign不拼接均值。方法和主要指标预先冻结后，Test仅作一次描述/确认审计，不反向选方法。B0共7种配置×3seeds=21次训练；B1复用已完成基线，仅新增3种几何控制×3seeds=9次训练。后续候选一次只增加一个机制。配对置信区间按source-video bootstrap，并同时报告每个Student seed的配对差值；不要以样本bootstrap代替训练seed不确定性。训练结束后再考虑方法升级。

## 未解决项

Early/Middle/Late缺失历史权重的阶段无法由Phase A补齐；历史run未记录的commit等字段保持缺失。独立Teacher不确定性、双模态训练、统一target多seed、20epoch N2复现和几何因果作用均需要Phase B新训练。当前不预先把N2升级为主方法。
