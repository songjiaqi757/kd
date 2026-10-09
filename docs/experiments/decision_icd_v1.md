# Decision ICD：下一轮验证集初筛

2026-10-03 冻结。先在 MOSI 上做四组 × seeds 13、42，固定预算；验证集筛选后
再锁定候选、增加 seed2026，并在 MOSEI 复验。初筛脚本不读取、评测 official test。
现有 GPU 1 上的 equal_modality_extensions_v3 继续按原协议运行；新入口不修改其
训练源码或续训指纹。新队列在 GPU 0 串行运行，并检查显存、系统盘余量。

## 四组对照

| decision_mode | 目标 |
|---|---|
| none | 原始 Uniform ICD |
| full | ICD + TAV 决策值蒸馏 |
| subset | ICD + 七子集决策值蒸馏 |
| interaction | ICD + 七个决策交互坐标蒸馏 |

所有组使用相同的原始任务损失、Full KD、ICD、编码器、LoRA、fusion、优化器、
七子集 forward 和 dropout 行为。模型方法字段仍为 `uniform_interaction`，
`decision_mode` / `experiment_method` 明确标识额外损失，避免改动旧模型加载器。

教师目标固定为 `q_T(S) = sigmoid(mean_probe_score(S) / tau)`，不是
`mean(sigmoid(probe_score(S) / tau))`，也不宣称它是校准概率。
学生使用同一个回归输出构造 `q_S(S)`，没有额外 binary head。
interaction 组对 q 做 Möbius 变换，空集为 `sigmoid(b/tau)`，b 沿用冻结资产中的
`student_empty_baseline`。变换后的坐标解释为决策空间交互。

`tau=1.0`、`lambda_decision=1.0`，各组使用同样的逐样本 Smooth L1 与坐标均值。
坐标变换会改变实际梯度尺度：固定系数不等于梯度范数相同，不能据此宣称消除了尺度
混杂。每 100 个训练 batch 及每 epoch 第一个 batch 记录融合模块上的梯度方向、损失
大小；不得根据 test 结果做归一化或权重搜索。

固定 20 epoch、min_epochs=20、patience=20、batch size 8、AdamW lr=1e-4、
weight decay=0.01，保持原始 ICD lambda。所有种子均重新跑匹配预算的 baseline。

## 选择、诊断与阶段门槛

- 保留 `best_valid_mae.pt`、`best_valid_acc2_nonzero.pt` 和用于续训的 `last.pt`；
  `best.pt` 是 MAE 最优的软链接，不保存每轮 checkpoint。
- 使用原 trainer 的改善阈值 `1e-5`，并列保留早期 epoch。
- 两种策略各保存完整 validation predictions、指标和误差分层；每一行完整报告
  MAE/Corr/Acc2/F1，不能把两个 checkpoint 拼成一个模型的指标。
- 预固定分层：真实标签 0 单列；非零且 `|y|<=0.5` 为近零；`|y|>0.5` 为远零。
  分别报告近零反号、远零反号、符号正确但有强度误差的数量与 MAE 贡献。
- 梯度参考同时记录 supervised task 与 task+FullKD；交互、decision 梯度仅测量，
  不投影、不影响更新、不额外消耗随机数。
- 初筛主策略是 `valid_acc2_nonzero`：相对匹配 baseline，两 seed Acc2 均改善，
  平均改善至少 0.005（0.5 个百分点），平均 MAE 退化不超过 0.005。
  `valid_mae` 同样完整报告，不能用更有利的策略替换预设主策略。
- 这是进入确认阶段的门槛，不是显著性声明。验证集配对 source-video bootstrap
  2000 次；区间以已选择的 checkpoint 为条件，不校正选模偏差，不代表 seed 不确定性。
- 初筛完成后冻结候选再做三 seed 与跨数据集确认；失败时保留负结果，不自动扩大搜索。

## 运行与监控

```bash
cd /home/wy/sjq/kd
/home/wy/sjq/miniconda3/envs/kd/bin/python project/scripts/run_decision_icd_screen.py \
  --dataset mosi --device 0 \
  --output-root outputs/experiments/decision_icd_v1/mosi
```

队列状态：`outputs/experiments/decision_icd_v1/mosi/queue_status.json`；每组日志在
`.logs/`；完成一组即刷新 `summary.json` / `summary.md`。失败即停止队列，不掩盖错误。
同命令重启可跳过完成组并从 last.pt 续训未完成组；协议、输入指纹不一致会拒绝续训。
如需在当前 run 完成后停止，在 output-root 新建 `STOP_AFTER_CURRENT` 文件。

重新汇总：

```bash
/home/wy/sjq/miniconda3/envs/kd/bin/python project/scripts/summarize_decision_icd.py \
  --input-root outputs/experiments/decision_icd_v1/mosi \
  --output outputs/experiments/decision_icd_v1/mosi/summary.json
```

已有六个历史 run 的低成本诊断：
`project/reports/decision_icd_historical_validation.{json,md}`。其中没有缓存的 Acc2-best
逐样本预测明确标记缺失，不用其他 epoch 的预测冒充。新入口会直接解决这一缺口。

## 可恢复的磁盘整理

只有 sjq 同盘目录可用，因此采用 Zstandard 无损压缩，不能声称移动文件就能释放空间。
工具先盘点已完成 run 的旧 epoch，保护两种验证最优、原有 best/last、所有已发现的
checkpoint 软链接目标；缺少选模历史或未完成的 run 跳过。数据、教师资产、原始模型和
测试 JSON/CSV 全部保留。

压缩时流式计算 SHA256，再完整解压到校验流，确认原始字节数与哈希均一致后，才移除
原 `.pt`。压缩文件在 `outputs/checkpoint_archive_v1/`，原位置留下
`.pt.archive.json`，包含哈希、原路径、归档路径；该文件不是 torch checkpoint。
原有 checkpoint inventory 作为历史审计记录保留：其中部分路径现处于归档状态，
运行全 epoch sweep 前必须先恢复相应文件。

盘点（不改权重）：

```bash
/home/wy/sjq/miniconda3/envs/kd/bin/python project/scripts/manage_checkpoint_archive.py
```

压缩到系统盘约 28 GiB 可用后停止：

```bash
/home/wy/sjq/miniconda3/envs/kd/bin/python project/scripts/manage_checkpoint_archive.py \
  --apply --target-free-gib 28
```

恢复单个 checkpoint 到原路径：

```bash
/home/wy/sjq/miniconda3/envs/kd/bin/python project/scripts/manage_checkpoint_archive.py \
  --restore /absolute/path/to/checkpoints/epoch_001.pt.archive.json
```

恢复会再次校验 SHA256，不覆盖现存文件，要求原始大小加 1 GiB 空闲；压缩归档仍保留。
同盘归档可以恢复文件字节，但不防硬盘损坏。不要手动删除 `.zst` 或对应清单。
