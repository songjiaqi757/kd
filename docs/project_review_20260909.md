**RDID-MSA 项目审计与修复建议 — 2026-09-09**

整体判断：项目已经形成数据准备、教师特征、Probe、学生缓存、训练和统计报告的完整研究链路；主要风险集中在实验定义一致性、缓存身份校验、LoRA 输入一致性和复现能力。当前证据支持 Full KD 有效、LoRA 路线值得继续；尚不足以证明 Reliability × Utility 稳定改善下游任务，也不足以将 Stage D 的全部收益单独归因于 LoRA。

本次阅读核心代码、训练/数据脚本、配置和阶段报告，核对本地完成运行的预测与报告，执行 CPU 复现。未修改训练实现、模型、数据集及原有实验结果，未启动全量训练或 official-test 推理。新增本报告及同名 JSON 证据。当前执行环境没有 git 命令，未开展 Git 历史或提交差异审计。

**已验证的基础情况**

- 使用项目环境运行 `python -m pytest project/tests -q`：44 passed，4 个 Transformer nested-tensor 性能提示，耗时 4.54 秒。
- 64 个源代码、Python 脚本和测试文件通过 AST 语法解析。
- official train/valid 窗口清单：train 为 16,399 个窗口、16,326 条 utterance、2,249 个源视频；valid 为 1,879 个窗口、1,871 条 utterance、300 个源视频。
- train 与 valid 的 video_id 交集为 0。此项证明当前清单的视频划分无交叉，不代表所有历史处理过程均已审计。
- 当前全量教师/学生缓存记录的 manifest SHA-256 均与所引用清单一致。未重算全部 18,278 个学生特征或 127,946 个教师特征。
- 对本次 9 组 MAE 比较涉及的已完成运行，检查 valid 样本 ID 唯一、集合一致、标签一致，并从预测文件重新计算 MAE、Pearson、Acc-2、Acc-7；与各自 report.json 的差异均小于 1e-12。
- WavLM 自定义 attention 在一个小型随机模型的 CPU float32、零 LoRA 增量测试中，与原实现输出最大差异为 1.49e-8。这项检查没有发现基础 attention 公式错误；尚不替代 GPU bf16 和全部梯度验证。

**优先级含义**

P1：可能改变训练目标、破坏结果可比性或导致静默数据错配，应在下一轮正式实验前处理。P2：影响统计解释、资源使用、断点恢复或复现，宜同步修复。以下区分已复现缺陷、静态可确认问题和需要实验验证的研究判断。

**1. P1 — 交互空集基线与研究方案不一致，SNR 权重实际受到影响**

证据：原方案 `docs/RDID-MSA_多模态情感交互蒸馏完整方案.md:810` 明确采用训练标签均值作为固定空集值，并要求教师/学生一致。`project/scripts/train_student_baseline.py:354`、`:715` 及 `project/scripts/build_teacher_interaction_reliability.py:51` 对原始预测直接调用 `mobius_transform(..., 0.0)`，没有先减去该均值。

按唯一训练 utterance 计算，均值为 **0.1462697559**，并非 0。对于 pair 坐标，改用方案基线相当于加上该常数；对于 triple 坐标，相当于减去该常数。

必须区分两种影响：

- 教师与学生使用同一常数时，未加权的交互差值、交互重建误差不变；常数平移也不会改变单个坐标跨样本的方差和 Pearson 相关。因此不能据此否定所有 raw interaction、E_pair 或 global utility 数值。
- SNR 使用 `abs(interaction_mean) / (sqrt(variance) + epsilon)`，依赖坐标绝对位置；符号一致性、交互强度分组以及 selective 阈值同样可能变化。因此 pair-SNR、selective、RU 的训练权重和相关诊断会改变。

本次 train-only、唯一 utterance 的敏感性检查中，四种高阶坐标合并后按中位数取 top-50%，**14.3728% 的“样本×坐标”入选状态发生变化**。生产脚本阈值按窗口计算，这个数字是明确的敏感性证据，不是训练重跑结果。

解决：在唯一训练 utterance 上计算并保存 `empty_baseline`，由训练、可靠性构建及分析脚本共用；明确选择“训练均值基线”或将零基线作为新的版本化实验定义。已有运行标记为 `baseline=0`，不得悄悄改写历史解释。重新生成受影响的 SNR/selection 统计后，只复验最终候选。

验收：使用 `V(S)=b+sum(w_m)` 的加性例子验证高阶项为 0；验证同基线平移不改变 raw KD 误差，同时能改变 SNR 权重；所有产物记录相同 baseline 和训练清单指纹。

**2. P1 — 全量训练使用了旧教师温度，校准元数据没有跟随教师目标切换**

证据：`train_student_baseline.py:83`、`train_student_lora.py:71`、`train_student_hidden_kd.py:50` 默认使用 **0.9259549975**。`run_stage_b_baselines.sh:34` 更将该数值直接写死。完成的 fullscale B1 报告和 D1 run_config 也记录了此温度。

然而实际教师 `outputs/probe/official_train_valid_seed2026/report.json` 的校准温度是 **1.0149979591**；另外两个全量 Probe 分别是 **1.0186201334** 和 **1.0033042431**。

影响：单教师分类 KD 没有使用对应全量 Probe 的校准温度；同样的 logits 会被变得更尖锐。与此同时，ensemble 二分类概率通过 `teacher_calibration_temperature()` 读取各教师真实报告，造成同一训练内不同分支的温度来源不统一。旧方法仍是一个可描述的 KD 配置，但不能称为使用了对应全量教师的校准分布。不能预先保证改正温度后 MAE 一定提高。

解决：默认从教师产物读取校准温度；CLI 默认设为 None，仅显式覆盖时生效，并记录覆盖原因。教师目标应携带 Probe、标签离散化、校准和特征版本指纹。修改后的正式候选和对照必须使用同一规则，旧实验单独保留。

验收：切换教师目标文件后自动使用新温度；缺失、非正或非有限温度时在训练前报错；核对实际 KD softmax 与报告元数据一致。

**3. P1 — LoRA 音频批处理改变 WavLM 特征，当前 D1 对照不只改变了可训练参数**

证据：`cache_student_encoder_features.py:70` 起逐样本运行冻结编码器；`train_student_lora.py:104` 起先将多条音频补零组成 batch，再运行在线编码器。本地 `model/WavLM-Base-Plus/config.json` 为 `feat_extract_norm="group"`。

WavLM 的 GroupNorm 位于卷积特征提取阶段，attention mask 不能撤销此前补零造成的归一化统计变化。[WavLM 官方文档](https://huggingface.co/docs/transformers/model_doc/wavlm)说明该选项使用第一层卷积的 group normalization；[PyTorch GroupNorm 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.modules.normalization.GroupNorm.html)说明其在训练和评估时均使用输入统计量。

本次使用本地预训练模型、CPU float32 和两条真实 train 音频复现：同一条 1.618625 秒音频，单独推理与和 4.581875 秒音频补零组批推理相比，有效 token 的隐藏表示最大绝对差 **1.30257**，相对 L2 差 **15.3157%**。合成短音频检查也出现差异。该数字是特征差异，不是 MAE 退化百分比。

影响：即便 LoRA 增量初始为 0，冻结缓存路径与在线路径也可能不同；同一 checkpoint 的输出还可能依赖 batch 长度组成。D1 相比 B1 的提升真实存在于当前预测文件，但“全部由 LoRA 引起”尚缺严格控制。

解决：优先采用逐样本、无补零的冻结音频前端，然后在前端输出上批处理可训练的 Transformer，并验证有效 token 一致性；或先用逐样本在线路径建立正确性参考。长度分桶只能降低变化，不能证明消除变化。为新实现建立使用相同前端、精度、组批与融合初始化的冻结对照，再比较 LoRA。

验收：同一实际音频单独运行、和长短音频组批运行时有效表示与最终预测应在预设误差内一致；验证零 adapter 的 online/cache 对齐、实际 LoRA 梯度及 GPU bf16 路径。不要直接切换归一化类型来“修复”，那会改变预训练模型。

**4. P1 — 学生缓存按行号复用，清单或模型变化后可静默错配**

证据：`cache_student_encoder_features.py:72` 使用 `items/{index:06d}.pt`，`:84` 只要文件存在就跳过计算；末尾重新写 index 和新的 manifest hash，恢复前不验证旧配置，也不核对特征内的样本身份。

本次在临时目录调用真实缓存入口、仅模拟模型加载进行复现：原缓存第 0 项代表 A，将清单改为 `[B,A]` 后重跑，B 被绑定到旧 A 特征，最终 `status` 仍为 `complete`。现有 `audit_student_feature_cache.py` 主要检查新 index、形状、dtype 和有限值；特征文件不包含样本身份，因而无法可靠识别这种语义错配。

`build_windowed_manifest.py:159` 也按 video/clip/window 序号命名媒体，改变窗口长度或重叠策略但复用 media-dir 时，旧文件可能直接复用为新时间区间。

本次没有发现当前大缓存已经错配；现有 manifest hash 匹配也不足以排除这种历史覆盖风险。

解决：恢复前验证 manifest 内容、模型 revision/hash、预处理配置、代码 schema；不匹配即拒绝复用。缓存键绑定 sample_id、媒体或时间窗口指纹和模型版本，特征内保存 sample_id/input fingerprint。窗口媒体目录或文件名也包含完整 window policy。使用临时文件+原子替换发布 index/config，并设置 writing/complete 状态。

验收：清单换序、文本修改、媒体替换、模型替换和窗口策略变化都不能复用旧特征；同配置正常恢复可跳过已校验项；损坏项能被识别后重算。

**5. P2 — 开启的梯度检查点在 LoRA train() 中被失效**

证据：`train_student_lora.py:239` 开启 gradient checkpointing，但 `student.py:410` 的 `train()` 随即将两个编码器设为 eval，仅恢复 LoRALinear 子模块的 train 状态。本地安装的 Transformers `modeling_layers.py:60` 要求 `gradient_checkpointing and self.training` 同时成立才调用 checkpoint 包装。

小型真实 Qwen3/WavLM 组合复现中，8 个带开启标志的模块全部 `training=False`，包含实际执行的两类 Transformer block。因此配置写着启用，但这些 block 不执行梯度重计算。

影响主要是显存和吞吐策略与配置不符，不意味着 LoRA 不会获得梯度。

解决：分离“关闭冻结骨干随机性”和“保持 checkpoint block 的训练状态”。可显式用非 reentrant checkpoint 包裹需要的 block，或使相应 block 保持 train 并分别关闭原始 dropout、LayerDrop、SpecAugment；保留 LoRA dropout。不能简单删除 eval 后任由所有数据增强开启，否则又改变实验条件。

验收：通过 block forward 计数或 profiler 确認反向时发生重计算；比较开启/关闭时的损失与梯度，并测量显存。

**6. P2 — 梯度累积不保持窗口权重，末尾不足一组时梯度尺度也不正确**

证据：`train_student_lora.py:273` 直接将每个已按当前 microbatch 权重和归一化的 loss 除以固定 accumulation。物理 batch=1 时，`w*loss/w` 将窗口权重完全抵消。末尾不足 accumulation 个 microbatch 仍除以完整 accumulation。

复现：两个样本的权重为 `[1,0.25]`，以相同回归输出计算，真实合并 batch 梯度是 `[0.16,0.12]`，两个单样本 microbatch 累积得到 `[0.10,0.30]`。

影响范围：目前核对的正式 D1/D2 配置使用 batch_size=8、gradient_accumulation=1，此缺陷不解释这些已完成运行。它会影响默认 batch=1、accumulation=8 的使用方式和后续显存降配实验。

解决：按整个逻辑 batch 的权重总和归一化损失分子，二分类项使用该逻辑 batch 非零标签权重总和；处理最后不足一组的真实分母。若希望每条 utterance 的训练贡献严格一致，还应明确父样本采样或固定总体归一化策略，避免随机 batch 自归一化带来的偏差。

验收：不等权样本、多窗口样本、最后不足一组时，关闭 dropout 后与等效大 batch 的梯度一致。

**7. P2 — 缺少完整训练恢复与运行身份校验，长实验容易重跑或混合产物**

证据：LoRA 训练只保存 best.pt 中的可训练参数与指标，不保存 optimizer、随机状态、DataLoader generator、early-stopping 状态；baseline 虽保存 last.pt 和 optimizer，却没有恢复入口，也没有完整随机状态。LoRA best.pt 采用直接写文件，非原子保存。多个入口对已有 output 使用 `exist_ok=True`，并覆盖 history/config；旧 report.json 不会在重启时被标为过期。队列 `run_stage_d_d2_2026_after_seed13.sh:8` 仅检查 report 文件存在。

影响：一次 D1/D2 需要数小时，中断后无法精确续训；同目录重启时，其他脚本可能将上一轮 report 误判为当前完成状态。LoRA checkpoint 只有增量参数，本身也不足以独立确定基础模型版本和预处理。

解决：统一 run_id/config hash、状态机和原子产物发布；已有目录默认拒绝覆盖，通过显式 resume 恢复；保存 optimizer、epoch、best/stale、Python/NumPy/torch CPU/CUDA RNG、DataLoader generator 等。检查 checkpoint 缺失键，只允许已知冻结参数缺失；队列验证 run_id、配置和 complete 状态。确需逐 batch 恢复时还应保存 sampler/cursor。

验收：同 seed 连续训练与按 epoch 中断后恢复的训练结果一致；旧报告不触发下游；部分写入或缺失 adapter 时明确失败。

**8. P2 — 显著性检验忽略源视频聚类；valid 多次参与选模限制了推断范围**

证据：`compare_student_runs.py:50` 和 polarity 检验按 utterance 独立抽样。当前 valid 的 1,871 条 utterance 只来自 300 个源视频，最多一个视频有 39 个窗口。校准、Probe early stopping、学生 early stopping、Stage A-D 方法筛选又多次使用 valid。

不能将所有 utterance 都视为完全独立。逐 seed bootstrap 只条件于已选定 checkpoint，未包含随机种子和模型选择不确定性；在校准集自身报告的 calibration-after 指标也不是独立泛化评估。这属于开发集选择偏差，不等于已经发现 test 泄漏。

本次新增按 video_id 有放回抽样的 paired cluster bootstrap，10,000 次，RNG seed=2026；抽中视频保留其全部 utterance，估计量仍是 utterance 平均 MAE。

| 对照 | seed | 候选减基线 MAE | 视频聚类 bootstrap 95% CI |
|---|---:|---:|---|
| B1 − B0 | 13 | -0.021140 | [-0.039907, -0.002385] |
| B1 − B0 | 42 | -0.035804 | [-0.050946, -0.020592] |
| B1 − B0 | 2026 | -0.018275 | [-0.031556, -0.005560] |
| RU − B1 | 13 | -0.006180 | [-0.021175, +0.008238] |
| RU − B1 | 42 | +0.003546 | [-0.009823, +0.016638] |
| RU − B1 | 2026 | -0.002542 | [-0.012738, +0.007853] |
| D1 − B1 | 13 | -0.041163 | [-0.060982, -0.022207] |
| D1 − B1 | 42 | -0.024665 | [-0.045627, -0.005469] |
| D2 − D1 | 42 | -0.015294 | [-0.025378, -0.004928] |

这些复算支持 B1 和已完成 D1 的系统表现改善，也保留 D2 seed42 的积极信号；RU 三个种子的区间仍均跨 0。聚类区间不必总比逐 utterance 区间宽，实际结果决定。这张表不能修复前述对照条件，也不能充当最终 test 显著性结论。

解决：保留 video_id 到预测产物，默认补充 cluster bootstrap；多个预设对照使用明确的多重比较规则，并同时报告逐 seed 结果及跨 seed 差值。后续方法开发可在 train 内按 video 分组交叉拟合/划分，用独立 fold 评估 sample-wise utility；最终配置和选择规则冻结后才做统一 test 评估。

**9. P2 — 产物对齐校验不完整，错误标签/重复 ID 可被静默接受**

证据：`load_teacher_targets()` 将重复 parent/subset 覆盖进字典；`attach_teacher_targets()` 主要校验 TAV split，对其他 subset 和 ensemble 的 split/label 一致性缺少完整检查；LoRA `load_rows():144` 用 cached 字段覆盖 raw manifest 字段，没有先比对标签、parent 和聚合权重。`compare_student_runs.py:41` 只使用 baseline 标签，未验证 candidate 标签一致。

本次选定已完成运行的标签/ID 检查没有发现错配，因此这是已确认的防线缺口，而非声称现有报告有错。

解决：定义统一的 manifest、teacher-target、feature-index、prediction schema；入口要求唯一主键、split/标签/parent/权重一致、七子集完整、数值有限，以及源产物指纹匹配；在读取目标或合并字典前拒绝冲突。

验收：重复行、相同 ID 不同标签、ensemble split 不同、替换清单同长度等情况均在训练或统计开始前失败。

**10. P2 — 配置、环境和测试依赖本机状态，可迁移复现不足**

证据：`workspace.yaml` 与多个 YAML 配置包含绝对路径，但训练脚本主要独立解析 argparse，没有统一读取 student_pilot.yaml 的配置链路。`environment/requirements.lock` 未显式列出直接使用的 `av`，torch 仅出现在注释；完整 pip_freeze 有这些包，因而不能简单说环境完全缺失。部分 tests 直接访问 `/home/wy/sjq/kd/dataset/...`，而数据被 Git 忽略。仓库没有统一打包/测试配置来区分单元测试和本机资产审计。

影响：本机 44 passed 不代表干净 checkout 能安装并通过测试；编辑 YAML 也不保证训练参数改变。不同工作目录下，相对路径与绝对路径入口的行为不一致。

解决：统一可校验的配置入口与 CLI 覆盖规则，持久化最终解析配置；补充 pyproject.toml、CPU 测试依赖与 GPU 环境安装说明，保留 CUDA 包索引和依赖锁定；将需要大数据/模型的检查标为 integration，并用小型 fixture 支持独立单元测试；所有路径从显式 workspace-root 解析。

验收：干净临时 checkout 不带真实数据/模型也能运行核心单测；integration 在明确配置资产后执行；变更 YAML 参数能在 resolved config 中直接观察。

**11. 研究设计 — 当前最需要补齐的是归因对照，而不是继续叠加损失**

Full KD 相比 Student-only 的已有三 seed MAE 约从 0.5323 降到 0.5072，基础蒸馏方向有证据。冻结 RU 平均仅改善约 0.0017，交互重建显著改善与任务收益之间仍缺连接。D1 相同两 seed 平均 0.4752 更值得优先核查，但需先处理音频路径混杂。

当前证据还有以下边界：

- `combined_loss()` 的真实标签监督只作用于 TAV；B1 请求的也是 TAV。B1 的缺失模态 token 没有得到直接训练，因此其七子集输出并不是一个受充分训练的缺失模态基线。RU 比 B1 更好重建交互，并不自动证明交互蒸馏是任务增益原因。
- global utility 实际是训练集上 `abs(corr(I,y))`，代表相关性，不是“某个样本加上该蒸馏项后的增量效用”；三个 Probe 都在同一训练集上拟合，Probe-seed 方差也不覆盖数据采样、教师模型和提示等其他不确定性。应准确命名其含义。
- D2 相比 D1 同时增加了多 Probe 的交互目标和额外损失；在 LoRA 条件下还缺 uniform ensemble-pair 对照，无法分离多教师、额外监督、可靠性及 utility 权重的作用。
- D1 相比冻结 B1 也不能回答“在 LoRA 学生上是否仍需要 KD”，因为缺少 Student-only + LoRA。
- 第一版 cosine hidden KD 失败，仅支持停止该预设配置；不能概括所有 hidden KD 均无效。Acc-2 分支按 binary NLL 选模，相比按 MAE 选模的基线存在选择准则变化，应按预设 safety gate 报告。

建议先冻结修复后的公共实现，然后按预算顺序完成最小矩阵：Student-only + LoRA、Full KD + LoRA、Uniform ensemble-pair + LoRA、RU + LoRA，保持输入、初始化规则、训练预算和选模指标一致。首轮只做工程对齐和一个 seed 的实现验证，不展开新的权重搜索；进入正式比较时再补齐预设 seeds。若要声称缺失模态鲁棒性，额外设置训练过的 modality-dropout/subset-task 对照，并在各 subset 上评估。

**执行顺序与停止条件**

1. 封存现有实验为原始实现版本；记录 baseline、温度、输入处理与已完成状态，避免边训练边改变共享代码导致混用。
2. 先修 P1：缓存身份校验、统一温度和空集基线、音频 online/cache 对齐；无需先重算所有历史实验。
3. 同步修复 checkpoint 恢复、梯度检查点和累积逻辑，并将复现失败场景纳入轻量测试。统一 schema 和 resolved config。
4. 用冻结/零 adapter 对照确认两条执行路径可比，检查 LoRA 梯度，按 batch_size=1/8 与重排样本复核预测一致性。
5. 在相同新版本下复验关键对照；先验证 Full KD + LoRA 的稳定收益，再判断 RU 是否带来独立增益。保留效果量阈值、跨 seed 方向一致性及视频聚类统计。
6. official valid 继续用于开发选择时明确标注。全部配置、候选和评估协议冻结后，才进行最终 test 实验。

验收重点是“同一输入产生可比特征、同一配置可恢复、每个目标能追溯来源、结论能排除明确混杂”，而不是单纯增加通过测试的数量。

**证据位置**

同目录 `project_review_20260909.json` 保存小型复现结果、真实训练音频差异、空集基线敏感性与 9 组 bootstrap 结果。CPU 复现使用项目现有环境，模型离线加载；缓存错配复现在独立临时目录并模拟模型加载。统计基于已有完成预测，不加载或重训学生 checkpoint。
