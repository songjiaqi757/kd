# GPU0 MOSI：六方法对照与文献启发实验

2026-10-08 冻结。用户要求 GPU1 继续原有六方法 MOSEI 队列，同时在 GPU0 运行 MOSI 对照，并试验文献调研方向及其他合理候选。两张卡独立调度。

共14种方法 × seeds13、42、2026 = 42个run。已检查原始ICD seed13、42的训练配置、源码与资产SHA256及权重，复用 `decision_icd_v1/mosi/icd_decision_none_seed{13,42}`。这些run的decision_mode为none，额外决策损失为0；诊断不改变更新或抽取随机数。因此本次新增40次完整训练，不重复这两个基线。

## 固定的六种方法

| 方法 | 内容 |
|---|---|
| uniform_interaction | 原始ICD，补齐seed2026 |
| uniform_interaction_shared_dropout | 七个子集共享融合dropout随机状态 |
| uniform_interaction_gradient_projected | 交互梯度与公共任务梯度冲突时投影 |
| uniform_interaction_coordinate_compatible | 按融合模块梯度兼容性调整坐标权重 |
| explicit_full_kd | 显式交互融合架构的FullKD对照 |
| explicit_uniform_interaction | 显式交互融合架构加ICD |

## 八种文献启发候选与必要对照

所有组均使用原始uniform_interaction学生架构，仅改训练更新或目标。以下名字是本地实验名，不代表已证实的新方法。

设 L0=监督任务+FullKD，L1为一阶ICD，L23为二三阶ICD，L1+L23保留原有七坐标均值缩放。

| 实验名 | 改动 | 对照目的 |
|---|---|---|
| order_routed | 编码器/池化更新L0+L1；融合/预测头更新L0+L1+L23 | 主假设：高阶蒸馏的参数更新位置 |
| scrambled_routed | 编码器仅接收固定索引0、3、6的ICD；融合接收全部ICD | 同样允许3个坐标，但不按阶数选择；未声称梯度范数完全匹配 |
| fusion_only | 编码器只更新L0；融合更新完整ICD目标 | 区分按阶数分配与简单削弱编码器辅助约束 |
| dgl_icd | 编码器更新三个单模态监督损失的均值；融合更新L0+全部ICD | 梯度解耦思路的本项目适配对照 |
| ensemble_source | FullKD回归和分类改为同一三探针ensemble来源；ICD目标不纠正 | 隔离教师来源一致性的收益 |
| target_uniform | 上述统一来源后，把完整预测残差的50%均匀分配到七坐标 | 标签混合与纠错的简单对照 |
| target_variance | 相同总残差按探针方差加平滑项分配到七坐标 | 检查结构化分配能否优于均匀分配 |
| late_decay | 前10轮原ICD系数不变，后10轮线性降到0；任务和FullKD保持 | 低成本检查持续ICD约束与后期退化的关系 |

`order_routed`等四个路由组使用独立autograd参数组更新，之后沿用原训练器的全局梯度裁剪和AdamW。编码器组含文本、音频、视频LoRA与三个池化模块；融合组含预测头。每100步及初始步记录各模块上公共任务与一阶/高阶ICD的梯度范数和余弦。这是诊断证据，不能单凭冲突比例宣称因果。

目标纠正先做简化pilot：

```
delta = 0.5 * (y - b - sum(mu))
uniform:  w_j = 1/7
variance: w_j ∝ variance_j + 0.1 * mean(variance) + 1e-6
mu_corrected_j = mu_j + w_j * delta
FullKD regression target = b + sum(mu_corrected)
```

两种纠正的FullKD标量目标完全一致，均为50%教师预测+50%训练标签；区别在残差如何分配到七坐标。FullKD分类在三组中均使用ensemble概率，未同时纠正类别分布。仅训练损失接触训练标签；不使用验证或测试标签构造目标。零方差时退化为均匀分配。

探针方差只是启发式不稳定性指标，不能视为校准误差协方差。本轮没有声称完成文献报告中的交叉拟合或完整协方差后验；先检查这个可控pilot是否有价值。`late_decay`也改变总ICD训练强度，因此不能仅凭收益声称时间安排本身是原因；若进入确认阶段应增加总强度匹配对照。

## 文献与新颖性界限

- [DGL，ICCV 2025](https://arxiv.org/html/2507.10213v1)：提供编码器与融合模块分离更新的依据。这里的dgl_icd使用本项目单模态子集、回归+分类任务和额外ICD，并非论文原架构/原协议复现。
- [Knowledge Distillation as Semiparametric Inference，ICLR 2021](https://arxiv.org/abs/2104.09732)：已研究教师误差、交叉拟合和损失修正。当前pilot不是它的理论算法，也不宣称普通教师纠错是新贡献。
- [On the Efficacy of Knowledge Distillation，ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html)：蒸馏效果与训练过程有关。late_decay是结合本项目退化现象提出的工程对照，不冒充该论文复现。
- [LSMI，ICML 2025](https://proceedings.mlr.press/v267/yang25aj.html) 已有按交互指导蒸馏；[DMIL，CVPR 2026](https://arxiv.org/abs/2606.11614) 已有交互分解与动态聚合。普通交互加权/门控不再单独作为创新点。

## 训练、选模与测评

- 20轮固定预算、batch8、AdamW lr1e-4、weight_decay0.01、原模型和LoRA范围不变。未自动扩大超参数搜索。
- 每次训练只保留 `best_valid_mae.pt`、`best_valid_acc2_nonzero.pt`、`last.pt`。`best.pt`是MAE权重软链接。原子写入时可能短暂出现临时文件，不保留逐轮权重。
- 两种选模各自完整报告MAE、Acc2NZ、Pearson、F1；不拼接两个checkpoint的指标。并列沿用原先改善阈值1e-5。
- 全部42个run都有两种选模的验证结果。匹配缓存存在时核验后复用；否则由独立GPU测评任务生成。逐样本预测、误差分层、配置和日志均保留。
- 原有六种固定方法共36次official-test测评，与GPU1原协议一致。八种新候选本轮不自动跑official test，先用验证结果筛选；方法、参数、主指标锁定后再确认。
- 预先将 valid_mae 下的三seed MAE作为候选主比较，要求平均MAE至少改善0.005、至少2/3种子改善且Acc2NZ平均下降不超过0.005；valid_acc2_nonzero 作为完整次级报告，不事后替换主标准。门槛只是开发筛选，不是显著性检验。目标纠正组还需与ensemble_source和target_uniform比较，路由组还需与fusion_only、scrambled_routed、dgl_icd比较。
- MOSI验证集较小，14方法筛选有选择偏差；不将验证最优或单seed胜出写成已证实泛化提升。

## 队列与结果位置

`outputs/experiments/mosi_research_v1/resource_queue/plan.json` 为119个调度任务：40次新训练、42个双checkpoint验证任务、36次测试测评、1次最终汇总。2个复用基线不占新训练任务；已完成测评标记自动跳过。

同目录 `registry.json`列出42个run及复用证据；`queue_status.json`为实时状态；`summary.md`/`summary.json`随验证完成更新，最终汇总任务补齐测试结果，JSON含单seed值和样本标准差；`scheduler.log`与`.logs/`记录运行过程。

GPU0沿用GPU1资源调度：每次训练34000MiB、测评8000MiB、额外6000MiB余量；每卡最多2个训练和2个测评任务，实际受显存预算约束。初始化未分配显存的任务也占预留预算，避免瞬时空闲导致超卖。每次启动后至少30秒再派发，失败则停止新派发并保留正在执行的任务。GPU1训练器与源码未修改。

显存实测后的修订：order_routed的峰值allocated达到44.02GiB，测评进程resident达到17524MiB。因此四种路由训练的预留上调为50000MiB，测评上调为20000MiB；其他训练仍为34000MiB，额外余量仍为6000MiB。两卡均已更新测评预算。通过`--adopt-running`校验PID启动时间和完整命令接管原训练进程，只替换调度器，未中断训练。旧预算计划保存在同目录`plan.before_memory_revision_20261008.json`。大显存路由任务可能只能和测评并行，不能同时容纳第二个完整训练任务。

在resource_queue目录创建 `STOP_AFTER_CURRENT` 可停止新派发。重启前检查残留子进程；不要同时启动第二个GPU0调度器。

验证记录：18项相关测试通过；order_routed、dgl_icd、target_variance在真实MOSI各8个训练/验证样本上跑过1轮，验证前向、梯度更新、两个最佳与last保存。小样本检查不代表模型效果，其结果与正式42个run分开保存。

## 2026-10-09 用户授权补齐测试

用户要求“需要测试但还没有测试的都测试一下”，因此将已冻结的八种研究候选全部加入双 checkpoint official-test 测评。保留原先先验证筛选的历史记录，不宣称这是盲测；未更改方法、参数或选中轮次。48 次新增测试已完成，registry.json 的 official_test_planned 已更新，原注册表保存在 registry.before_test_authorization_20261009.json。原训练队列的完成标记仍仅对应原 119 项任务；补测队列另存于 historical_test_completion_20261009。
