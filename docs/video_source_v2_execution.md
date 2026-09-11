**Video source v2：已授权的后台实验执行方案**

目标固定为保留 T/A/V 三模态，修正教师视频时间信息，并验证相同 attention 覆盖规则下的三模态 LoRA。TA-only 仅作诊断，不作为最终删模态方案。用户已要求停止 C2 且不安排重跑；旧 C2 checkpoint、配置和 history 保留在原目录。

**执行顺序**

1. **教师配对诊断**：固定 64 个不同源视频的 train 单窗口样本（22 负、20 中性、22 正），比较 legacy/corrected timing。相同像素、文本、权重和 prompt v1；对 V 记录 direct 原文与 hidden，对 TA/TAV 记录 hidden。TA 是不含视频的负对照，若其 hidden 在同样输入下变化则停止。共 192 个配对任务。分层样本零输出比例不能直接与原 benchmark500 的 91.6% 作总体比较。
2. **全量教师特征**：在修正时间策略下重新提取 official train/valid 七个子集，共 18,278 窗口、127,946 个任务。全部重新计算，形成独立版本，不替换旧缓存。完成后检查完整性、有限值和切分。
3. **共享 Probe**：固定 seeds2026/2027/2028，分别训练、验证选模和重新校准。学生固定使用 seed2026 的 TAV 目标及其新温度，不从三个 Probe 中挑最好者。
4. **真实资产学生小检查**：使用新目标运行 8 个 train/valid 父样本的全 12 层 Video LoRA，一轮，验证数据交接和 48 个视觉 LoRA B 投影的梯度。
5. **学生确认矩阵**：以下三组均运行 seeds13/42/2026，共 9 组，不按单 seed 性能提前取消另外两组。

2026-09-11 用户调整执行范围：本轮只运行 `video_lora` 的全部三个种子，两张 GPU 先分别运行 seed13、seed42，空闲后启动 seed2026。`frozen_video` 和 `ta_only` 六组对照暂缓，**不自动启动**。主实验及其视频扰动诊断完成后生成仅含主实验的报告，队列状态为 `paused / controls_deferred` 并退出，等待用户后续安排；普通服务重启不会解除暂缓。教师特征、Probe 与小检查仍为前置阶段；训练配置和数据保持原方案。调度变更前的方案和入口源码保存在 `outputs/experiments/video_source_v2/schedule_revisions/`。

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
- C2 服务已停止，本队列没有 C2 重跑步骤。
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
- 最终可读报告：`docs/video_source_v2_results.md`（完成后生成）

本轮当 status.json 为 `status=paused`、`stage=controls_deferred` 时，三个主实验及诊断、报告均已完成，对照仍暂缓；完整九组矩阵才使用 `complete / complete`。关闭本次会话不会停止任务；用户可在结束后告知助手复核结果。进度以保存的完成数/epoch 为准。

**运行前验证**

全 12 层 Video LoRA 已用真实模型、batch=8 验证，8 train/8 valid 父样本的一轮峰值显存 10.64 GiB，48 个视觉 B 投影均有非零梯度。该显存是小检查值，不代表全量最长样本的峰值。另在双卡上完成一条真实样本的教师 V/TA/TAV 配对推理：V/TAV hidden 有变化，TA 完全一致；不是准确率改善结论。新流程的定向测试 33 项通过。端到端极小数据验证和服务注册结果记录在 `docs/video_source_v2_preflight.json`。
