# `video_source_v2` 统一实验记录

> 文档状态：视频链路审计、修正、预检、训练与结果的完整记录。最终模型始终保留 T/A/V 三模态。
>
> 临时运行整理：`video_source_v2_preflight` 已完成教师配对诊断、特征提取、Probe 和学生预检；`student_smoke` 也已完成。二者均已被正式 `outputs/experiments/video_source_v2` 结果取代，临时目录已删除。后续脚本可重生成 `video_source_v2_results.generated.md`，不会覆盖本记录。2026-10-02 删除了来源为 `/tmp/pytest...` 的测试残留简表；真实结果保留在下方原文快照与 `outputs/experiments/video_source_v2/summary.json`。

以下按“最终结果 → 执行记录 → 上游审计”的逆时间顺序完整保留原文。

## 原文快照：`video_source_v2_results.md`

**Video source v2 — primary T/A/V LoRA results; controls deferred by user**

All student modes use the same corrected TAV Probe seed2026 targets and its recorded temperature.
TA-only is a diagnostic control; the final model requirement remains TAV. Official test was not evaluated.

| Mode | Seed | MAE | Pearson | Acc-2 |
|---|---:|---:|---:|---:|
| video_lora | 13 | 0.471261 | 0.786222 | 0.872740 |
| video_lora | 42 | 0.466293 | 0.785247 | 0.885257 |
| video_lora | 2026 | 0.463737 | 0.798715 | 0.882476 |

Three-seed MAE: **0.467097 ± 0.003826** (mean ± sample SD).

Mean MAE increase over five cross-source video shuffles:

| Seed | Mean delta MAE | Range |
|---:|---:|---:|
| 13 | +0.003674 | [+0.002017, +0.005497] |
| 42 | +0.003978 | [+0.000939, +0.005860] |
| 2026 | +0.004670 | [+0.003592, +0.006099] |

The consistent positive shuffle averages show video sensitivity, not net benefit from Video LoRA.
The same-protocol `frozen_video` and `ta_only` controls remain deferred, so no between-mode causal comparison is reported.

Machine-readable results: /home/wy/sjq/kd/outputs/experiments/video_source_v2/summary.json
Teacher paired diagnostic: /home/wy/sjq/kd/outputs/experiments/video_source_v2/teacher_pair/summary.json
Wall-clock efficiency is not an exclusive-device benchmark; two student jobs may decode media concurrently.

## 原文快照：`video_source_v2_execution.md`

**Video source v2：已授权的后台实验执行方案**

目标固定为保留 T/A/V 三模态，修正教师视频时间信息，并验证相同 attention 覆盖规则下的三模态 LoRA。TA-only 仅作诊断，不作为最终删模态方案。用户已要求停止 C2 且不安排重跑；旧 C2 checkpoint、配置和 history 保留在原目录。

**执行顺序**

1. **教师配对诊断**：固定 64 个不同源视频的 train 单窗口样本（22 负、20 中性、22 正），比较 legacy/corrected timing。相同像素、文本、权重和 prompt v1；对 V 记录 direct 原文与 hidden，对 TA/TAV 记录 hidden。TA 是不含视频的负对照，若其 hidden 在同样输入下变化则停止。共 192 个配对任务。分层样本零输出比例不能直接与原 benchmark500 的 91.6% 作总体比较。
2. **全量教师特征**：在修正时间策略下重新提取 official train/valid 七个子集，共 18,278 窗口、127,946 个任务。全部重新计算，形成独立版本，不替换旧缓存。完成后检查完整性、有限值和切分。
3. **共享 Probe**：固定 seeds2026/2027/2028，分别训练、验证选模和重新校准。学生固定使用 seed2026 的 TAV 目标及其新温度，不从三个 Probe 中挑最好者。
4. **真实资产学生小检查**：使用新目标运行 8 个 train/valid 父样本的全 12 层 Video LoRA，一轮，验证数据交接和 48 个视觉 LoRA B 投影的梯度。
5. **学生确认矩阵**：以下三组均运行 seeds13/42/2026，共 9 组，不按单 seed 性能提前取消另外两组。

2026-09-11 用户调整执行范围：本轮只运行 `video_lora` 的全部三个种子，两张 GPU 先分别运行 seed13、seed42，空闲后启动 seed2026。`frozen_video` 和 `ta_only` 六组对照暂缓，**不自动启动**。主实验及其视频扰动诊断完成后生成仅含主实验的报告，队列状态为 `paused / controls_deferred` 并退出，等待用户后续安排；普通服务重启不会解除暂缓。教师特征、Probe 与小检查仍为前置阶段；训练配置和数据保持原方案。调度变更前的方案和入口源码保存在 `outputs/experiments/video_source_v2/schedule_revisions/`。

2026-09-13 用户重新授权 C2 与 Video source v2 并行。C2 seed13 已完成，仅在物理 GPU1 从 epoch 1 重跑未完成的 seed42；停止时保存的 checkpoint 不含 optimizer/RNG，不能作严格续训，因此旧单 epoch 目录先归档再重跑。C2 仍使用原 corrected-temperature、empty-mean、uniform ensemble-pair 配置和旧冻结资产，不读取本轮新教师特征，也不改变 Video source v2 的数据、超参数或选模规则。两项并行仅影响墙钟耗时。

2026-09-14 执行状态：C2 seed42 与三个 `video_lora` seeds 均正常完成；Video source v2 按授权范围停在 `paused / controls_deferred`，`frozen_video` 与 `ta_only` 六组未自动启动。两张 GPU 已释放，official test 未评估。

| 模式 | 学生输入 | T/A LoRA | V | 教师目标 |
|---|---|---|---|---|
| frozen_video | TAV | 全部 attention 层 | 冻结 | 新 Probe TAV |
| video_lora | TAV | 全部 attention 层 | 全部 12 层 attention LoRA | 相同新 Probe TAV |
| ta_only | TA，诊断用 | 全部 attention 层 | 不输入 | 相同新 Probe TAV |

三者 rank=8、alpha=16、dropout=0.05，attention Q/K/V/output；Text 28 层、Audio 12 层、Video 12 层（T/A 架构本身的层数不同）。视频可训练参数 589,824。batch=8、梯度累积=1、AdamW lr=1e-4、weight_decay=.01、最多 30 epochs、patience=7、valid MAE 选模。各模态融合接口均 4 个 token。全局任务损失与 Full KD 权重保持原控制实验设置。

温度由 `--teacher-probe-report` 自动读取，要求 targets 与 report 来自同一 Probe 目录，且其特征标记为 sampled_fps_v2。TA-only 使用 TAV 教师属于显式教师特权信息控制，用于隔离学生 Video 输入的影响；它不代表教师从未使用视频。

每个完整 TAV 运行在最佳 checkpoint 上执行 5 次跨源视频置换、重复中间帧和打乱帧序；正常预测与各扰动结果均保存。最终生成逐 seed MAE/Pearson/Acc-2、均值/SD、视频聚类 paired bootstrap 和扰动比较。没有新增 ROI、多 clip、audio padding 修正、RU 或辅助损失，以保持本轮因素可解释。official test 不参与推理、训练和评估。

**后台与恢复**

- 服务：`rdid-video-source-v2.service`，用户 systemd 服务，随用户服务管理器启动。
- 用户 linger 已启用，退出登录/关闭会话后仍可运行。
- 教师阶段独占两张 GPU；Probe 依次训练；学生阶段最多同时两组，每张 GPU 一组，每次分配前确认空闲。
- C2 不属于本队列；其 seed42 已于 2026-09-13 作为独立服务固定到物理 GPU1 并行重跑。
- 代码、manifest 指纹及模型文件 size/mtime 身份保存在 plan.json。学生入口另保存其模型权重 SHA-256。不要在队列运行期间修改已冻结源码/资产。
- 教师配对任务逐项保存；全量特征按 completed.npy 续提；学生按 epoch 保存模型、optimizer 和 RNG 并恢复。第一轮尚未完成而中断的学生目录自动改名保留，从第一轮重跑。
- Probe 在中断后从该 seed 重新训练；已完成阶段通过 done 标记和产物核验后跳过。
- 普通脚本异常将状态写为 failed 并停止；其他在跑的同队列学生会停止，保留最后完成 epoch。修复原因后用下面的 restart 命令恢复。不会自动重复失败配置或自动降低 batch。
- 显存/驱动查询失败时等待，不把未知状态视为空闲。

**实时查看命令（任意目录可执行）**

推荐总览：

```bash
watch -n 15 '/home/wy/sjq/miniconda3/envs/kd/bin/python /home/wy/sjq/kd/project/scripts/video_source_v2_status.py'
```

连续查看总日志：

```bash
tail -F /home/wy/sjq/kd/outputs/experiments/video_source_v2/pipeline.log
```

查看服务：

```bash
systemctl --user status rdid-video-source-v2.service --no-pager
```

总览会打印当前详细日志路径。教师诊断详细日志示例：

```bash
tail -F /home/wy/sjq/kd/outputs/experiments/video_source_v2/logs/teacher_pair.log
```

异常或重启后手动恢复（服务配置也支持开机后自动启动）：

```bash
systemctl --user restart rdid-video-source-v2.service
```

**主要产物**

- 固定方案：`outputs/experiments/video_source_v2/plan.json`
- 实时状态：`outputs/experiments/video_source_v2/status.json`
- 配对诊断：`outputs/experiments/video_source_v2/teacher_pair/summary.json`
- 新教师特征：`outputs/experiments/video_source_v2/teacher_features/`
- 新 Probe：`outputs/experiments/video_source_v2/probes/seed{2026,2027,2028}/`
- 学生：`outputs/experiments/video_source_v2/students/{mode}_seed{13,42,2026}/`
- 最终汇总：`outputs/experiments/video_source_v2/summary.json`
- 最终可读报告：`docs/experiments/video_source_v2.md`（完成后生成）

本轮当 status.json 为 `status=paused`、`stage=controls_deferred` 时，三个主实验及诊断、报告均已完成，对照仍暂缓；完整九组矩阵才使用 `complete / complete`。关闭本次会话不会停止任务；用户可在结束后告知助手复核结果。进度以保存的完成数/epoch 为准。

**运行前验证**

全 12 层 Video LoRA 已用真实模型、batch=8 验证，8 train/8 valid 父样本的一轮峰值显存 10.64 GiB，48 个视觉 B 投影均有非零梯度。该显存是小检查值，不代表全量最长样本的峰值。另在双卡上完成一条真实样本的教师 V/TA/TAV 配对推理：V/TAV hidden 有变化，TA 完全一致；不是准确率改善结论。新流程的定向测试 33 项通过。端到端极小数据验证和服务注册结果记录在 `docs/audit/video_source_v2_preflight.json`。

## 原文快照：`video_resolution_review_20260914.md`

**Video 问题复核（2026-09-14）**

结论：上游 FPS 元数据错误和学生 Video LoRA 覆盖不对称已修复；教师直接评分大量为 0 的问题未解决。学生已利用视频，但收益幅度有限，尚无证据表明时序信息被充分利用。不能把“视频可以训练”写成“视频问题全部解决”。

本轮三个全 LoRA 种子于 2026-09-14 08:17 完成，状态为 paused / controls_deferred。冻结 Video 与 TA-only 对照仍按用户要求暂缓，本次复核不启动训练。以下均为 train/valid 证据，未评估 official test。

**教师直接评分**

64 条分层 train 样本的配对实验中，旧时间处理输出 0 为 61/64（95.3125%），修正后为 59/64（92.1875%）；双方解析失败均为 0。原 benchmark500 的 91.6% 来自不同样本集合，不能与 92.1875% 直接比较趋势。当前没有新的同一 benchmark500 全量 direct 复测证据。

因此，FPS 错误确实存在且已修复，但它不足以解释大量零输出；剩余根因未确定。直接生成评分与 hidden → Probe 是不同读出路径，当前学生使用后者，不把直接生成的零分当蒸馏目标。

**教师隐藏表示：同口径全量验证集、三个 Probe 种子均值**

| 指标 | 旧时间处理 | 修正后 |
|---|---:|---:|
| V MAE | 0.725676 | 0.721411 |
| V Pearson | 0.425259 | 0.430797 |
| TAV MAE | 0.437768 | 0.438675 |
| TAV Pearson | 0.836643 | 0.837201 |

V 只有小幅描述性改善，没有显著性结论，TAV MAE 还略升。新 T/A Pearson 均值分别为 0.804553 / 0.817987，V 仍明显弱于二者。此前 0.4723 属于 benchmark500 的小规模 Probe 验证，不能与本次全量验证直接比较。

修正后同一共享 Probe 的 TA / TAV 平均 MAE 为 0.455258 / 0.438675；每个 Probe 种子均为 TAV 更低。这支持视频在教师融合路径中提供补充信息，但不是独立训练的学生 TA 对照。

**学生与视频扰动诊断**

| Seed | 正常 MAE | Pearson | 换其他源视频 ΔMAE（五次均值） | 重复中间帧 ΔMAE | 打乱帧顺序 ΔMAE |
|---|---:|---:|---:|---:|---:|
| 13 | 0.471261 | 0.786222 | +0.003674 | +0.000324 | +0.001069 |
| 42 | 0.466293 | 0.785247 | +0.003978 | +0.001459 | +0.000391 |
| 2026 | 0.463737 | 0.798715 | +0.004670 | +0.003922 | -0.000065 |

三 seed MAE 均值 ± 样本标准差为 0.467097 ± 0.003826。各模型均注入 Video 全 12 层的 48 个投影，共 589,824 个可训练视觉 LoRA 参数；首步审计的全部 48 个 LoRA B 投影梯度均有限且非零。

15 次换视频诊断的 MAE 全部变差，提供了模型利用正确视频对应关系的证据；其中 7 次的单独视频聚类 bootstrap 95% 区间不跨 0，但这些结果共享样本且未校正多重比较，不能当作 15 个独立实验。扰动也会破坏跨模态对应关系，不能替代重新训练的对照。

打乱帧顺序的三个区间均跨 0；重复中间帧只有 seed2026 的区间不跨 0。因此，目前只可说对正确视频内容存在依赖，对帧顺序的依赖较弱；不能断言模型完全不使用运动，也不能宣称已充分学到表情动态。

两类学生对照仍暂缓，尚不能隔离全层 Video LoRA 的净收益。旧 C1 三 seed 平均 MAE 0.466175 也不能充当严格同条件对照：教师资产和训练路径不同，新三 seed 均值没有明显超过这一历史参考值。

**下一步建议**

保留三模态及统一 LoRA 规则。优先检查视频源文件与标签对应、有效人脸大小、裁剪/缩放和归一化、学生 16 帧采样对短暂表情的覆盖，并核验教师空间 patch 参数与模型处理链路。以上是待诊断方向，不是已确认的新故障。教师时间尺度修复不会自动改变学生固定 16 帧的输入方式。

随后在固定其他条件下，分别诊断视觉读出方式（当前最后有效输入 token hidden 与视觉 token pooling）或视频输入方式（全帧与人物区域、单段与多段采样），先做小规模、按源视频隔离的训练/验证，再决定是否扩大。不要同时改所有因素，也不要强迫 direct 输出非零作为成功标准。当前无需为使用三模态而先修好 direct 文本评分，但若要将其作为教师监督或论文结果，则必须单独验证修复。

**证据文件**

- `outputs/experiments/video_source_v2/teacher_pair/summary.json`
- `outputs/experiments/video_source_v2/probes/seed{2026,2027,2028}/report.json`
- `outputs/probe/official_train_valid_seed{2026,2027,2028}/report.json`
- `outputs/experiments/video_source_v2/summary.json`
- `outputs/experiments/video_source_v2/students/video_lora_seed{13,42,2026}/video_gradient_audit.json`
- `docs/archive/reports/实验汇总报告.md`：benchmark500 Probe 与历史 C1 结果的范围说明。

## 原文快照：`teacher_video_source_fix_20260911.md`

**教师 Video 上游时间信息问题：发现、修正与三模态后续约束**

用户明确要求最终保留 Text/Audio/Video 三模态，并采用对等的适配设置。TA-only 仅作为贡献诊断对照，不再作为删除 Video 的方案。本记录补充并调整 `video_followup_20260911.md` 的执行优先级：先修正和复验教师共同输入链路，再判断学生的视频表征与适配问题。

**1. 新发现：教师的视频采样 FPS 没有传入处理器**

本地环境为 Transformers 5.2.0。qwen_omni_utils 的采样默认目标 FPS 为 2，实际 FPS 随帧数取整有所变化；`return_video_kwargs=True` 会返回实际采样 FPS 和 `do_sample_frames=False`。旧生成评分脚本与隐藏特征提取脚本只接收 audio/images/videos，丢掉了这些参数。Qwen3OmniMoeProcessor 在未显式收到 FPS 时使用 1.0，并据此计算 `video_second_per_grid = temporal_patch_size / fps`。

本地 temporal_patch_size 为 2，所以旧路径总把每个时间块记为 2 秒；实际约 2 FPS 时应约为 1 秒。该字段进一步用于 Thinker 的视频时间位置编码，不是仅用于日志显示。processor 和 Thinker 的计算链路已通过本地源码核对，并对真实媒体运行位置编码计算复现。官方 Transformers 源码也给出了相同字段的计算逻辑：[处理器实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py)。本结论以已安装 5.2.0 和本地资产为准，不宣称上游所有版本均有同样行为。

**2. 真实片段复现**

对固定的前 3 个 train 窗口，在相同已采样像素上比较旧处理与显式传入实际 FPS 的处理。没有加载教师权重，没有生成评分，也没有访问 official test。三个窗口来自同一个源视频；它们用于证明处理链路问题，不代表全数据集故障比例。

| train sample | 实际采样 FPS | 旧秒数/时间块 | 修正秒数/时间块 | 最大位置 ID 差 |
|---|---:|---:|---:|---:|
| -3g5yACwYnA[0] | 1.906542 | 2.000000 | 1.049020 | 197.803925 |
| -3g5yACwYnA[1] | 1.851852 | 2.000000 | 1.080000 | 47.839996 |
| -3g5yACwYnA[2] | 1.949458 | 2.000000 | 1.025926 | 101.303711 |

每组的 pixel_values_videos 和 input_ids 均完全一致，实际计算的位置 ID 不同。之前对一个 valid 片段的独立检查也发现旧 2.0 秒对应实际约 1.019 秒，但上述可复算记录固定只取 train。

这证明存在上游时间尺度错误；**尚未证明它造成了 91.6% 的零输出，也尚未证明修正后的任务指标会提高。**

**3. 为什么使用 Probe 不能自动排除这个问题**

旧链路为：视频解码/采样 → processor 时间位置 → Thinker → 两条分支：直接生成分数；或最后一个有效输入 token 的 hidden → 共享 Probe → 学生蒸馏。

91.6% 的零分来自第一条分支，当前学生没有直接读取这些零分。Probe 训练使用真实 train 标签，特征来自生成之前的输入 hidden，也没有把生成的 `0` token 当作特征。但是，两条分支共享此前的输入处理，FPS 丢失可能影响二者。已有 V Probe 相关性与 TAV 优于 TA 的结果证明旧特征有信息，不证明其时间处理正确。旧全量教师特征应标记为 legacy timing，保留溯源，不能与新提取的含 V 特征悄悄拼接覆盖旧目录。

**4. 本次已落地的代码修正**

- 新增共享入口 `project/src/rdid_mosei/teacher_video.py`，保留采样器的实际 FPS，并禁止对已采样的视频重复抽帧。
- 在本地 Omni 处理器中 FPS 必须传单个 scalar，不能直接把工具返回的列表传进去。现有每次单窗口输入符合这一约束；多视频输入显式拒绝，避免错配。
- `run_teacher_benchmark.py` 与 `extract_teacher_probe_features.py` 均接入相同修正，并验证时间块秒数与实际 FPS 一致。
- 新默认策略为 `sampled_fps_v2`，新默认产物分别为 `teacher_benchmark500_windowed_timing_v2.jsonl` 和 `outputs/probe/features/benchmark500_timing_v2`。只用于保留历史行为的显式选项为 `--video-timing-policy legacy_v1`。
- 新策略拒绝向旧 timing 的输出文件/特征目录续写。生成结果和特征提取的 `video_preprocessing.jsonl` 记录实际 FPS、video grid 和时间块秒数。
- 保持原空间预处理不变，避免本次同时引入 patch 对齐、裁剪或分辨率变化。工具默认空间 patch 参数与模型配置也应另做检查，当前不把它当作已证实的性能根因。

验证：78 项测试通过；新生成/提取入口 dry-run 通过；两入口向旧产物目录写入的防混用检查通过；真实媒体 CPU 检查确认相同像素对应的时间位置已修正。**尚未重新运行 30B 教师或重训 Probe/Student，当前既有训练继续使用旧版本资产，不受本次入口修正影响。**

**5. 接下来具体怎么做**

1. 固定一份 train 内、覆盖多个源视频和正/负/中性的约 64 条诊断清单。在同一份原始媒体上做 legacy timing / corrected timing 配对，保持模型权重、prompt v1、像素、解码和生成参数不变。同时记录 V 的 direct 原文、hidden，以及 TA/TAV 的 hidden；T/TA 无视频路径应保持不变。只改时间元数据，避免把提示词改善误算为时间修正收益。
2. 对 direct 统计零值比例、MAE、Pearson和原文解析成功率；对 hidden 统计相同视频修正前后的变化及换视频响应。64 条主要用于定位，不足以证明全量准确率收益，也不在这些样本上同时训练/评估新的 Probe。
3. 在修正的 canonical 输入定义下，建立新的 full-train/valid 特征版本。重新提取 V/TV/AV/TAV 四个含视频子集；T/A/TA 只有在模型、prompt、manifest、预处理和输入指纹核验一致后才能复用。重训完整共享七子集 Probe，重新校准并导出新目标，不能沿用旧 Probe 的温度或仅替换少数分数。
4. 若时间修正后 direct 仍大量为 0，但新 Probe 的 V/TAV 有效，则把 direct 视为不适合该评分任务的读出通道，继续用一致的 Probe 监督三模态。禁止通过强迫非零或人工摊平分布冒充修复。若修正后的 hidden V 仍很弱，再做人物/局部时序输入与表征读出诊断。
5. 完成教师资产版本闭合后，再比较对等设置的三模态学生。保留 TA-only 仅用于测贡献；最终交付保持三模态。

**6. “三个模态设置一样”的具体定义**

旧实验只在 rank/alpha/dropout 上相同，覆盖层数并不对等：

| 模态 | 编码器总层数 | 旧 LoRA 覆盖 | 下一版建议覆盖规则 |
|---|---:|---:|---|
| Text | 28 | 全部 28 层 attention | 全部 attention 层 |
| Audio | 12 | 全部 12 层 attention | 全部 attention 层 |
| Video | 12 | 最后 2 层 attention | **全部 12 层 attention** |

保持三者均为 Q/K/V/output projection、rank=8、alpha=16、dropout=0.05、相同学习率/优化器规则和训练预算；融合接口各 4 个 token。Video 全 12 层对应约 589,824 个 LoRA 参数，原末两层只有 98,304 个。现有学生入口支持 `--video-layers 12`，但本次没有据此启动训练。

不同骨干层数、隐藏维和序列长度不同，因此相同适配规则不等于绝对相同参数量、相同帧/token 数或相同梯度大小。先建立覆盖比例相同的对照；若显存不足，整个对照矩阵统一 microbatch 与正确的梯度累积规则，不能只缩小 V 组有效 batch。T/A/V 全部可训练也不保证每个模态都有正收益，最终仍需视频置换和独立视觉读出验证有效信息，而非只看 adapter 梯度。

**产物与复算**

- 可复算诊断：`project/scripts/audit_teacher_video_timing.py`。
- 原始证据：`docs/audit/teacher_video_timing_audit_20260911.json`。
- 回归测试：`project/tests/test_teacher_video.py`。

```bash
cd /home/wy/sjq/kd
OMP_NUM_THREADS=2 /home/wy/sjq/miniconda3/envs/kd/bin/python \
  project/scripts/audit_teacher_video_timing.py
```

上述 CPU 复算不运行模型权重推理。接下来教师 A/B 诊断、全量新特征提取与三模态对等适配均是待执行步骤，不把本次代码修正写成“91.6% 问题已消失”。

## 原文快照：`video_followup_20260911.md`

**2026-09-11 新实验复核与视频问题下一步方案**

> 后续约束与新发现：用户明确要求最终保留三模态。进一步检查已复现教师视频采样 FPS 丢失、时间位置尺度错误，direct 与 hidden 入口均受影响；优先级更新为先修正并复验教师共同输入，再检验三模态对等适配。详见 [教师上游修正记录](video_source_v2.md)。下文关于 TA 最终精简方案的建议不再作为执行目标，TA-only 仅保留诊断用途。

本次直接复核完成运行的 report、逐样本预测、视频扰动预测、最佳 checkpoint 和配置指纹。统计覆盖 official valid 的 1,871 条 utterance、300 个原始视频；没有运行新的模型推理或读取 official test。新计算使用按 video ID 聚类的 10,000 次 paired bootstrap，随机种子 20260911，因此区间末位与原报告 seed2026 的区间略有不同。所有区间均以已选择的 checkpoint 为条件，不包含重新训练和反复验证选模的不确定性；子组分析仅用于提出假设。

**主要判断：T/A 适配的收益已经获得更强证据；V 的问题从“尚未适配”变成了“已适配、会影响预测，但没有稳定的净收益”。应停止重复扩展 Video v1，优先定位输入、表征和融合中的具体瓶颈。现在还不能把“最终删除 Video”写成已验证结论。**

**1. 新实验改变了什么**

| 实验 | 完成 seeds | valid MAE，均值 ± sample SD | 当前解释 |
|---|---|---:|---|
| C-v2：同路径冻结 T/A，Full KD，冻结缓存 V | 13/42/2026 | 0.504800 ± 0.000506 | 严格归因对照 |
| C0：T/A LoRA，task-only，冻结缓存 V | 13/42/2026 | 0.512738 ± 0.001057 | 仅适配不足以取得当前最佳效果 |
| C-v2 C1：T/A LoRA，Full KD，冻结缓存 V | 13/42/2026 | **0.466175 ± 0.000243** | 当前完成度最高的修正协议主基线 |
| legacy D1：T/A LoRA，Full KD，冻结缓存 V | 13/42/2026 | 约 0.4728 ± 0.0057 | 旧温度版本，保留历史对照 |
| legacy D2：T/A LoRA，RU KD，冻结缓存 V | 13/42/2026 | 约 **0.4646 ± 0.0008** | 性能信号较好，但尚不等于修正协议下 RU 的独立收益 |
| C-v2 C2：uniform ensemble-pair，T/A LoRA | 13 | 0.468594 | 比同 seed C1 差 0.002352；seed42 重跑中，不能提前作最终判定 |

C-v2 C1 比冻结对照平均改善 0.038625，三个 seed 分别改善 0.039108、0.038318、0.038448，逐 seed 聚类区间均不跨 0。相对 task-only 平均改善 0.046562。这支持“监督与可适配的 T/A 表征配合有效”。

legacy D2 比 legacy D1 平均改善 0.008244；逐 seed 仍只有 seed42 的区间不跨 0。此次把三个固定 seed 的逐样本误差差值先求平均再做视频聚类，区间为 [-0.015156, -0.001612]，只表示这三个既定模型的平均差，不能写成任意新 seed 都显著。D2 与 C-v2 C1 温度/监督协议不同，不能直接把两者的约 0.0016 差距归因于 RU。

核验时 C2 seed42 正在 GPU0 执行，GPU1 空闲。重启中断目录没有正式 report，不计入完成结果；Video v1 已完成并停止扩展。服务/GPU状态是本次检查快照，不保证之后不变。

**2. Video v1 的结论：LoRA 已经生效，但 Gate 失败**

三个模式的 T/A 都使用 LoRA；B 额外适配 VideoMAE 第 10/11 层的 Q/K/V 和 attention output，共 98,304 个参数。

| 模式，均为 seed13 | MAE ↓ | Pearson ↑ | Acc-2 ↑ | 最佳 epoch |
|---|---:|---:|---:|---:|
| A：TAV，冻结 V | 0.471689 | **0.796527** | 0.876912 | 12 |
| B：TAV，V 末两层 LoRA | 0.470305 | 0.792225 | 0.877608 | 8 |
| C：TA-only | **0.468983** | 0.788980 | **0.885953** | 8 |

- B−A：ΔMAE = **−0.001384**，本次区间 [-0.010538, +0.007853]。
- B−C：ΔMAE = **+0.001321**，本次区间 [-0.006488, +0.009199]。
- A−C：ΔMAE = +0.002706，区间 [-0.006871, +0.012685]。

B 未满足“比 A 至少改善 0.005 且胜过 C”的预设探索门槛；不补 v1 seeds42/2026 是正确的资源决策。CI 跨 0 也意味着不能证明三者等效，更不能证明所有 Video 方法都无效。

本次额外检查了最佳 checkpoint：8 个视觉 LoRA B 矩阵均非零，范数范围 1.2442–1.7235；首次反向传播的对应梯度均为有限非零值。A/B/C 共有输入文件的指纹一致，训练时记录的核心源码指纹与当前源码一致。因此“仍然冻结”“adapter 没有实际更新”已不是优先解释。这个检查不等于所有层的优化都已经充分。

**3. 新发现：不能把小 ΔMAE 直接解释成没有使用视频**

对同一个已训练模型，仅改变 valid 视频内容，T/A 与 batch 顺序保持不变：

| 模型与扰动 | ΔMAE，扰动−正常 | 95% 视频聚类 CI | 平均绝对预测变化 |
|---|---:|---|---:|
| 冻结 V：跨源换视频 | +0.000710 | [-0.002721, +0.004185] | 0.05621 |
| Video LoRA：跨源换视频 | **−0.001630** | [-0.006248, +0.002828] | **0.06730** |
| Video LoRA：重复中间帧 | −0.000996 | [-0.004515, +0.002354] | 0.03762 |
| Video LoRA：打乱帧序 | +0.000921 | [-0.001005, +0.002906] | 0.02151 |

Video LoRA 换视频时，21.49% 样本的预测改变量大于 0.1，90 分位改变量为 0.1460，非零标签样本中 3.13% 的预测跨越正负边界。误差改善的样本占 47.30%，恶化的占 46.93%，剩余为持平。把改善/恶化的误差量摊到全体样本，分别是 0.033026 和 0.031396，两者相减才是表中的平均改善 0.001630。

因此，当前证据支持“视频产生了影响，但有益影响与有害影响抵消”，不支持“视频计算图完全无效”。这也不能反过来证明模型学会了表情：影响可能来自人物身份、背景、亮度、姿态等。换视频同时破坏了身份、场景和语义配对，一次置换无法区分这些因素。

打乱时间顺序影响较弱，只能说在当前输入和模型下没有明显的时间顺序净收益；静态表情也可能包含有效信息。不要把“不依赖时间顺序”直接等同于“视频没有价值”。上述每项只有一个扰动随机实现，区间反映源视频抽样，不覆盖不同置换的随机性。

**4. 教师视频仍有信息，但并不是每个样本都有帮助**

现用 Probe seed2026 的 valid MAE：TA=0.445228、TAV=0.432316，加入视频平均改善 **0.012912**，聚类区间 [-0.021024, -0.004873]。三个 Probe 的 TA→TAV 平均改善约 0.018396。V-only 三 Probe 平均 Pearson 约 0.4253；早期 benchmark500 的 0.4723 不是全量 valid 数字。

探索性拆分发现，在教师加入视频有帮助的 981 条 valid 样本上，Student B 相对 C 平均改善约 0.01903；其余 890 条上平均恶化约 0.02375。教师/学生逐样本增益相关性只有 0.1722。这里使用了真实 valid 标签定义组别，且 B/C 的教师子集不同，所以这些数字仅提示异质性；不能用它们训练 gate、设定线上切换规则或宣称已经发现可部署的“有益视频识别器”。

Teacher direct V 的 91.6% 零输出仍是生成评分通道的历史问题，当前没有作为训练监督。本项目现有汇总还记录了提高教师视频采样率、修改提示词未带来改善；本次没有重新运行该历史尝试。暂时没有必要重新扩大 direct prompt 搜索，更不能把所有 0 当作可靠中性标签。Student 的 16 帧采样方案与 Teacher 的 FPS 设置是不同路径。

**5. 需要收紧的解释，以及待检验的原因**

| 项目 | 已确认事实 | 下一步意义 |
|---|---|---|
| TA-only 对照混合了两个变化 | C 使用 TA 教师，A/B 使用 TAV 教师 | B−A 可以检验增量视觉适配；B−C 比较的是完整方案，不能单独归因于学生 V 输入 |
| Video v1 仍是旧温度 | 0.9259549975；C-v2 为 1.0149979591 | 不直接跨版本比较 .468983 与 .466175，也不因温度不同否定 v1 内部 A/B；下一版统一修正温度 |
| 输入时间跨度不固定 | 整窗口只采 16 帧，约 39.6% 窗口超过 7.5 秒 | 邻帧间隔随片段长度增大，可能丢失短时变化；需按固定物理时间间隔做局部 clip 对照 |
| 空间裁切没有人物约束 | 短边缩放后中心裁剪 224 | 可能缩小或裁掉偏离中心的人脸；尚无全量人脸覆盖审计，不能宣布是根因 |
| 视觉适配范围有限 | 仅末两层 attention LoRA，其余 VideoMAE 冻结 | Gate 失败没有测到全部视频适配上限；先确认信息瓶颈再加容量 |
| 只监督融合输出 | 当前 loss 是融合任务损失 + Full KD，没有独立 V 任务头 | 梯度非零不保证学到独立视觉情感信号；可能存在模态优化不均衡，尚未实证定位 |
| 输入处理仍有共同限制 | 保留旧 WavLM padding 行为 | 同路径 A/B 比较仍有意义；部署前应修复并统一所有新对照，避免混用 old/new audio |

本地 VideoMAE 模型卡说明它是 Kinetics-400 上的自监督预训练模型，不能默认等同于表情识别器。VideoMAE 官方示例采用 16 帧、固定采样步长；这为检查当前“整段可变间距”的输入提供依据，但不证明改采样一定提高 MOSEI 指标。[官方采样代码](https://github.com/MCG-NJU/VideoMAE/blob/main/run_class_finetuning.py)

联合训练不同模态未必自动得到最佳泛化，已有研究讨论了不同模态的过拟合速度及优化差异。本项目“模态优化不均衡”仍然是假设，不据此直接上复杂梯度算法。[CVPR 2020 原论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html)

长片段提供了一条值得优先检查的线索：

| valid utterance 长度 | 数量 | 冻结 V MAE | Video LoRA MAE | TA-only MAE |
|---|---:|---:|---:|---:|
| ≤7.5 秒 | 1,038 | 0.467808 | 0.465932 | 0.461550 |
| 7.5–15 秒 | 709 | 0.465039 | 0.467788 | 0.474433 |
| >15 秒 | 124 | 0.542196 | 0.521295 | 0.500049 |

长片段组中 LoRA 比冻结 V 有改善，但仍比 TA 差约 0.02125。中长度组却有相反方向，因此不能据此采用“长视频直接禁用”规则。这是单 seed、事后分组、未作多重比较修正的诊断线索。

**6. 下一步按问题顺序解决，不再同时改多个因素**

**P0：收敛现有实验。** 保持 Video v1 的 no-go，不重复补种子或提高 rank。完成正在运行的 C2 seed42，按既定 Gate 决定后续 C2/C3，不把 legacy D2 的优势直接平移到 C-v2。保留 C-v2 C1 为完成度最高的修正协议基线；TA-only 是需要补齐同协议验证的精简候选，而不是已经证明最优的最终模型。

**P1：先区分“输入/视觉表征弱”和“融合没有利用好”。**

1. 对现有 A/B 最佳模型补做 5 个固定随机种子的跨源视频置换，保持 T/A、batch 和推理精度一致；先用不同种子的逐样本扰动误差求平均，再按源视频聚类。报告 ΔMAE、绝对预测变化、改善/恶化比例。选择典型成功/失败样本仅供可视化解释，不用于调验证规则。
2. 在 train 中预先按长度和来源分层选定约 200 条片段，检查原始帧与实际裁剪帧的人脸覆盖、分辨率、遮挡、人物数量、采样间隔与媒体时间对齐。固定样本名单与检查准则；质量指标计算不能用情感标签来决定保留与否。旧 25 条抽查没有普遍黑帧，不能替代这个审计。
3. 用已有冻结 VideoMAE 特征做轻量 V-only 读出：先固定 temporal/spatial mean pooling + 标准化 ridge，必要时增加一个固定容量的小 MLP。标准化、正则选择都只在 official train 内按 video 分组完成；最后只评估一次 official valid。教师 V hidden 可用相同类型读出作参照，但维度/模型规模不同，数值不是严格同容量因果对照。不要拿已在全部 train 上拟合的适配 checkpoint 构造所谓无泄漏 OOF 表征。

解释规则：冻结 V-only 读出很弱，优先处理输入/预训练表征；V-only 已有可靠信号而 TAV 没有净增益，优先检查融合和优化。单个线性 probe 失败不能证明特征没有任何信息。P1 比重新跑多组完整三模态训练更直接地回答瓶颈在哪里。

**P2：只在 P1 指向的环节上做一个小型受控改动。**

- 若人物覆盖差：在相同 16 帧上比较 center crop 与固定规则的人脸+上半身 ROI。按时间平滑人物框，检测失败退回旧裁剪并记录标记，禁止丢弃难样本。先冻结 VideoMAE 做读出比较，不同时改变时序和 LoRA。
- 若长片段时间采样过稀：保留原空间处理，增加固定方案的短 clip 表征。一个可评审起点是每段取 3 个均匀中心、每个约 2 秒/16 帧的局部 clip，短片段使用掩码避免把重复 clip 当成额外证据；clip 特征用简单均值池化。固定物理时间再映射到解码时间戳，不简单写死“每 4 帧”而忽略源 FPS。该方案约为 3 倍视觉前向预算，必须同时报告成本；另外保留 48 帧均匀采样分为 3 组的等帧数对照，区分局部连续性和仅增加帧数的收益。
- 若视觉读出有效但融合利用不佳：先冻结已训练 TA 主干，增加一个小视觉残差头，预测 `y_hat = y_hat_TA + r(TA_detached, V)`，残差输出层零初始化，先只训练残差头，用真实 train 标签监督，设置同结构“无 V 残差头”对照排除新增容量作用。通过后才允许适配 V 末端；不一开始叠加 RU、V-only KD、辅助损失、模态 dropout 和质量 gate。

这三个分支不同时启动。ROI/短 clip/残差结构目前都是建议，尚未实现或证明有效；沿用统一方案的 V1/V2/V3-v2/V4 顺序，以审计结果选择必要步骤。

**P3：候选通过低成本筛选后，才运行修正协议的视觉确认矩阵。**

| 组别 | 学生输入 | 学生适配 | 教师监督 | 要回答的问题 |
|---|---|---|---|---|
| TA-TAVteacher | TA | T/A LoRA | TAV Probe | 相同教师下，无 V 的学生基线；教师特权信息需明确披露 |
| TAV-frozen | TAV | T/A LoRA，V 冻结 | 同一 TAV Probe | 学生实际多一个 V 输入的收益 |
| TAV-candidate | TAV | T/A LoRA + 唯一筛选出的视觉改动 | 同一 TAV Probe | 候选视觉改动的额外收益 |
| TA-TAteacher | TA | T/A LoRA | TA Probe | 不依赖教师 V 的完整 TA 方案 |

前三组在输入实现、随机初始化、batch、优化器、训练预算、early-stop、融合容量等方面采用共同协议；TAV 的两组使用完全相同视频输入，若研究因素本身是 ROI/时序则只改变该因素。全部使用修正后的温度，任何 audio 修复在整个矩阵统一生效并版本化。不要拿 C-v2 缓存 V 的历史组直接替代新在线 V 的同路径对照。

先 seed13 筛选；晋级前冻结 seeds42/2026 的配置与门槛。候选进入最终主模型需同时胜过同教师 TA、冻结 V，并与实际 TA 方案比较；沿用原方案三 seed 平均 MAE 改善至少 0.005、方向一致、视频聚类 CI 不跨 0、跨源置换能造成稳定可解释退化的要求。重复帧和时间打乱可帮助判断收益是静态还是动态，不把时间敏感性作为所有视觉收益的必要条件。

若仍失败，停止扩张 Video 搜索，保留 TA 精简方案和冻结 V 基线作最终同协议选择。若有条件采用质量 gate，只能用训练集分组 OOF 的效用目标与推理可获得的质量特征训练；不能把上述 valid 的“教师有益组”标签喂给 gate。

**7. 本轮产物与执行边界**

- 本报告：`docs/experiments/video_source_v2.md`。
- 机器可读统计及输入 SHA-256：`docs/audit/video_followup_20260911.json`。
- 可重算脚本：`project/scripts/analyze_video_followup_20260911.py`。
- 旧正式结果：`docs/experiments/video_source_v2.md`、`outputs/student/video_adaptation_v1/summary.json`。

复算命令：

```bash
cd /home/wy/sjq/kd
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/wy/sjq/miniconda3/envs/kd/bin/python \
  project/scripts/analyze_video_followup_20260911.py
```

脚本检查预测 ID/标签一致、所有 valid 样本齐全、数值有限、重算 MAE 与原报告一致、三个模式的共有输入指纹一致，以及视觉 adapter 确已更新。它只写本报告配套 JSON，不修改模型、历史结果或训练服务。本次完成的是结果复核和方案制定；P1 新推理/读出训练以及 P2/P3 新实验均尚未执行。

## 原文快照：`video_adaptation_v1_results.md`

**Video adaptation results**

Official valid only. All numbers are conditional on validation-selected checkpoints.

| Mode | Seed | MAE | Pearson | Acc-2 |
|---|---:|---:|---:|---:|
| frozen_video | 13 | 0.471689 | 0.796527 | 0.876912 |
| video_lora | 13 | 0.470305 | 0.792225 | 0.877608 |
| ta_only | 13 | 0.468983 | 0.788980 | 0.885953 |

| Comparison | ΔMAE | Video-cluster 95% CI |
|---|---:|---|
| video_lora_minus_frozen_video_seed13 | -0.001384 | [-0.010452, +0.007687] |
| video_lora_minus_ta_only_seed13 | +0.001321 | [-0.006429, +0.009227] |

TA-only uses TA teacher targets. A/B use identical TAV teacher targets.
Single-seed promotion is an exploratory gate, not a significance claim.
Machine-readable results: /home/wy/sjq/kd/outputs/student/video_adaptation_v1/summary.json

## 原文快照：`video_review_20260909.md`

**Video 分支专项分析与改进方案 — 2026-09-09**

结论：video 分支确实值得优先检查。当前最有依据的判断是：视觉教师存在有用信号，而学生尚未稳定获得相应的增量收益；稀疏时间采样、空间裁剪、冻结的通用视觉编码器和融合监督都可能限制效果。尚未发现可以将全量视觉特征判为“读错/全部失效”的证据，也不建议直接删除视频或立即加大视觉 KD 权重。

本次基于现有 official train/valid 预测复算，抽查训练视频与预处理张量，未进行新训练或 official-test 评估。项目总审计报告及 JSON 已由 project/reports 移至本目录。

**1. 效果证据：教师能利用视频，学生的收益较弱**

全量教师 Probe 的 official-valid 三 seed（2026、2027、2028）平均值如下。各项同属 1,871 条 valid utterance，可在此表内比较。

| 输入子集 | MAE | Pearson | Acc-2，排除真值 0 |
|---|---:|---:|---:|
| T | 0.4813 | 0.8062 | 0.8904 |
| A | 0.4607 | 0.8171 | 0.8860 |
| V | 0.7257 | 0.4253 | 0.7186 |
| TA | 0.4562 | 0.8276 | 0.8924 |
| TV | 0.4605 | 0.8234 | 0.8917 |
| AV | 0.4444 | 0.8221 | 0.8776 |
| TAV | 0.4378 | 0.8366 | 0.8936 |

纯 V 明显弱于 T/A，但相关性不为零。更关键的是，TA→TAV 的平均 MAE 改善为 **0.01840**，不是没有增益。

在现有 B2 subset4 学生上，TA 和 TAV 都有子集蒸馏监督，可以做比 B1 缺失模态输出更有解释力的同模型比较：

| 学生 seed | TA MAE | TAV MAE | ΔMAE=TAV−TA | 按视频聚类 95% CI |
|---|---:|---:|---:|---|
| 13 | 0.51631 | 0.51405 | -0.00226 | [-0.00762, +0.00314] |
| 42 | 0.50564 | 0.50633 | +0.00069 | [-0.00579, +0.00735] |
| 2026 | 0.51356 | 0.50476 | -0.00880 | [-0.01843, +0.00111] |

B2 平均 TA MAE=0.51184，TAV MAE=0.50838，平均改善 **0.00346**。三个学生区间均跨 0；教师三个 seed 的对应区间则均低于 0，详见配套 JSON。Bootstrap 为按 300 个源视频抽样、10,000 次、RNG seed=2026。

这支持检查“教师视觉收益为何没有稳定传给学生”，但不是严格的因果归因：同模型 subset 比较还受到不同子集监督、缺失标记和共享参数影响，不能代替独立训练 TA-only 与 TAV 的对照。现有 D1/D2 LoRA 输出文件没有保存完整七子集预测，本次也没有据此推断其视觉贡献。

早期 benchmark500 上教师直接生成的 V 分数有 91.6% 为 0；那是生成评分通道的结果，不是当前 fullscale Probe，也不能与上表混为同一实验。现有主链路已使用 Probe，保持该选择合理。

**2. 输入不是简单的 RGB/归一化错误，但审计覆盖不足**

代码 `project/src/rdid_mosei/student_data.py:22` 用 PyAV 解码为 rgb24，随后交给 VideoMAE processor。本次检查一条实际 train 样本：输入为 uint8 RGB，processor 输出 `[1,16,3,224,224]`，执行一次 1/255 rescale 和 ImageNet mean/std 归一化，数值有限。该抽查没有发现 BGR、重复归一化或张量维度错误。

查看已有 `outputs/teacher_benchmark/video_diagnostic_contact_sheet.jpg`，可见人物及不同画面变化，不能将弱表现归结为全体黑帧；该图也不是全量质量证明。

现有 `audit_prepared_media.py` 检查媒体存在性、流数、时长和音频格式，却不检查人脸是否可见、空间裁剪是否保留人物、冻结帧、真实运动变化或口型/声音同步。总时长相同不等于内容逐时刻对齐。

建议增加：源视频与裁剪视频首/中/末时间戳对照、解码帧数与 PTS、黑帧比例、重复/冻结帧比例、人脸检测成功率及脸框在最终 crop 的保留比例。说话人检测与人脸跟踪应使用可观察信息，不用情感标签挑选画面。

**3. 时间采样是当前最具体、最值得验证的瓶颈**

目前不论窗口多长，均对整段视频取首尾包含的 16 帧。实际 train/valid 18,278 个窗口的时长：中位数 6.468 秒，P75=9.501 秒，P95=16.9585 秒，最大 30 秒。约 **39.6%** 超过 7.5 秒，约 **7.53%** 超过 15 秒。

相邻采样帧间隔近似 duration/15：6.5 秒片段约 0.43 秒，15 秒片段约 1 秒，30 秒片段约 2 秒。模型的相邻时间位置代表的真实时间跨度随样本变化，短暂表情变化或动作过程有可能被漏掉。此项是合理机制假设，尚未通过采样消融证明其贡献大小。

本地模型是 Kinetics-400 自监督预训练的 VideoMAE-Base。作者给出的 Kinetics 微调示例使用 16 帧、sampling_rate=4，并说明使用 dense sampling；这不等于 MOSEI 必须照搬该配置，但说明当前“所有长度都全程均匀 16 帧”并非已验证最合适的时间策略。[模型说明](https://huggingface.co/MCG-NJU/videomae-base)、[作者微调说明](https://github.com/MCG-NJU/VideoMAE/blob/main/FINETUNE.md)。

建议候选：每条 utterance 取 3 个覆盖不同时间位置的短 clip，每个 clip 16 帧、约 2–3 秒覆盖；明确以实际秒数/PTS 采样，而不是假定全体源视频帧率相同。每个 clip 独立经 VideoMAE，先压缩，再结合 clip 时间位置做聚合。更短 utterance 使用明确的补帧与有效性规则。先比较单 clip 和多 clip，记录计算、缓存成本；不要直接把 num_frames 改成 48 塞进现有 16 帧 checkpoint。

**4. 当前中心裁剪不能保证保留说话人面部**

实际 processor 为短边 resize 到 224，再 center crop 224×224。对 1280×720 视频，相当于保留原图中间约 720×720 区域，左右各约 280 像素被裁掉；即使脸保留下来，也可能占很少的视觉 patch。

视频包含背景、物体、字幕、姿态等信息，直接整帧处理并非错误。但当研究目标需要表情/目光细节时，中心裁剪缺少对说话人位置的保证。当前尚无全量脸框审计，不能声称已有多少比例人脸被裁掉。

建议先做保持同一模型和帧数的空间消融：当前 center crop 对照“人脸跟踪、适度扩大后的稳定 face/upper-body crop”。避免逐帧检测抖动；多人场景确定目标说话人；检测失败时回退原视图并记录质量。若局部视图有增益，再比较全局+局部双视图；不要一开始就丢弃全局手势和场景信息。

**5. 视频编码器始终冻结，任务适配不足是待验证原因**

当前 checkpoint 只有 Kinetics 自监督预训练，未针对情感任务微调。Stage A-C 的三编码器冻结，Stage D 只对 T/A 使用 LoRA，VideoMAE 继续读取旧缓存。这是代码可确认的事实；因此 D1 的整体改善并不证明视频分支得到改善。[模型用途说明](https://huggingface.co/MCG-NJU/videomae-base)也将下游微调作为主要用途。

建议先比较当前视频特征的简单线性/小型 Probe 与现有视觉 Q-Former 分支，确认信息是否存在。若输入/采样正常而视频表征仍明显偏弱，再尝试只训练视频末端 1–2 层或视觉 adapter/LoRA，并验证目标层和梯度。另一方向是增加经过验证的面部表情/动作特征作为受控对照；目前不需要立即下载更大视频模型。

**6. 文件与预处理版本已经存在可观察的不一致**

当前 `prepare_full_dataset.py` 写视频时有 `fps=25`；但本次随机抽取的 24 条 train 视频中，23 条实际 prepared FPS 不是 25，多数为 30，也包含约 29.97 和 23.976。当前脚本对已有文件会跳过，因此现存资产可能来自旧版本；本次未确认各文件历史生成版本。

不同 FPS 本身不证明视频不可用；问题在于不能仅根据当前脚本推定实际资产的生成方式，也不能以“文件存在”作为新采样版本有效的证明。

另外，定向检查的 `-NFrJFQijFE[0]` 抽样 16 个不同索引，却只有 10 张逐像素不同的帧。该样本有约 0.478 秒开头补帧，源内容也可能静止，不足以判断为解码错误。随机 24 条样本均为 16 张不同帧，未显示普遍完全重复现象。抽查结果不得当成全量问题发生率。

解决：媒体与缓存写入 preprocessing_version、源文件/窗口指纹、实际 FPS/PTS、采样索引及 crop 参数；任何一项变化均建立新版本，不能复用原来的 `items/000000.pt`。新旧版本分开保留后再比较。

**7. 如何用最少实验定位问题**

第一步，固定当前 checkpoint 做诊断：正常 video、随机置换 video、重复一张帧、打乱帧顺序，保持 text/audio 不变。跨源视频做多次置换，避免同视频相近帧造成伪“不敏感”。置零/缺失视频只能辅助观察，因为可能是未训练的输入分布。解释必须联合看：置换影响小提示模型对视频弱依赖；时间打乱影响小也可能是静态面部线索已足够，不单独证明时间模块坏了。

第二步，训练独立的 TA-only 对照，并和 TAV 使用一致的 T/A 编码器状态、训练预算、seed 和选择准则。如果验证视觉信息价值，TA-only 对照的教师目标也只能用 TA；若 TA-only 学生使用 TAV 教师，应单列为“无视觉推理、视觉教师蒸馏”实验。当前同模型 TA 输出不代替该对照。

第三步，按以下顺序做单因素候选：

| 优先顺序 | 变更 | 保持固定的内容 | 回答的问题 |
|---|---|---|---|
| V0 | 正常/置换/重复帧/乱序视频 | 当前 checkpoint、T/A、标签 | 模型依赖什么视觉信息？ |
| V1 | 稳定的人脸/上半身视图 | 16 帧策略、VideoMAE、训练预算 | 空间视图是否限制效果？ |
| V2 | 多个固定时间跨度短 clip | 已选视图、冻结视频编码器 | 时间采样能否改善任务？ |
| V3 | 视频末端轻量微调 | 已选输入、KD、融合结构 | 是否缺少任务适配？ |
| V4 | 简单视觉质量门控或调整辅助监督 | 已选视频表征 | 如何在不可靠视频下利用/抑制视觉？ |

V2 的输入量和计算会增加，必须额外记录显存、吞吐、缓存体积，必要时补充同等帧预算对照。先完成工程检查和单 seed pilot，再对预设候选补齐 seeds；效果量可沿用原方案的 MAE 0.005 建议目标，不能边看 valid 边不断改门槛。

最终判断应同时报告 TAV 对独立 TA-only 的 MAE/Acc-2、逐 seed 差值、按视频聚类置信区间以及时长/可见脸质量分组；各分组阈值在 train 固定。若可比条件下视觉始终不改善任务，可以保留 TA 作为部署或强基线，并诚实报告 video 的限制。

**建议当前优先做的组合：媒体与缓存版本审计 → 当前模型视觉扰动诊断 → 稳定局部视图 → 多短片段采样。** 在这些检查完成前，继续叠加 TV/AV 交互权重或直接给视频加 LoRA，容易把输入与表征问题混在一起。

证据：[视频复核 JSON](../audit/video_review_20260909.json) 保存全量已有预测的复算，[帧抽查 JSON](../audit/video_frame_spotcheck_20260909.json) 保存 25 条训练视频抽查。总审计见 [项目复核记录](../archive/reports/project_review_20260909.md) 和 [项目复核 JSON](../audit/project_review_20260909.json)。
