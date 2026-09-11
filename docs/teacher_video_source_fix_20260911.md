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
- 原始证据：`docs/teacher_video_timing_audit_20260911.json`。
- 回归测试：`project/tests/test_teacher_video.py`。

```bash
cd /home/wy/sjq/kd
OMP_NUM_THREADS=2 /home/wy/sjq/miniconda3/envs/kd/bin/python \
  project/scripts/audit_teacher_video_timing.py
```

上述 CPU 复算不运行模型权重推理。接下来教师 A/B 诊断、全量新特征提取与三模态对等适配均是待执行步骤，不把本次代码修正写成“91.6% 问题已消失”。
