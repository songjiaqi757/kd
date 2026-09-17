# KD 工作区整理记录（2026-09-17）

## 范围与原则

本次检查覆盖 `/home/wy/sjq/kd` 中的 Git 跟踪文件与本地大型产物。整理前约 404 GiB、151,107 个文件；Git 工作树无未提交改动。

判定原则是：

1. 保留原始数据、处理后数据、模型权重、特征缓存、最佳 checkpoint 和正式结果。
2. JSON 与 Markdown 报告分别承担机器可读证据与人类可读解释，不因同名而合并。
3. 历史快照保留决策语境，通过 `docs/README.md` 标记地位，不改动已被脚本或报告引用的稳定路径。
4. 只删除可再生成的缓存、无需续训的恢复 checkpoint，以及已有完整替代的中断运行。

整理前对 244 个 Git 跟踪文件计算 SHA-256，未发现字节级重复文件。大型 source/处理后数据和教师/学生产物即使存在相同媒体源，也承担溯源、窗口化、缓存或复现的不同职责，因此不按文件名或体积盲目去重。

## 已删除

- 122 个已完成实验的 `last.pt`。每个目录均已同时存在 `best.pt`、`report.json` 和 `predictions.jsonl`，且状态不是 `training`、`running` 或 `queued`。
- smoke/preflight 目录中的大型 `best.pt` / `last.pt`。其配置、历史、预测、指标和视频梯度审计均保留。
- 两个中断的 C2 seed42 目录：`interrupted_reboot_20260911_161314` 和 `stopped_before_gpu1_restart_20260913_124853`。同配置完整结果 `stage_d_cv2_c2_uniform_ensemble_pair_tempfix_emptymean_seed42` 已具备最佳权重、预测和报告。
- 项目、外部仓库与数据目录中的 `__pycache__` / `.pytest_cache`，以及四个 Hugging Face 模型目录中仅包含下载状态的 `.cache`。

合计删除 143 个冗余 checkpoint 文件和 2 个中断目录，释放 62,916,027,043 字节（约 58.6 GiB）。

## 明确保留

- 所有正式 `best.pt`、模型权重、教师缓存、编码器特征、预测和训练配置。
- 两个状态为 `training` 的 M4 `last.pt`，以及 4 个缺少完整评测证据的历史 `last.pt`；它们不满足安全删除条件。
- 所有历史方案、阶段报告和小型审计 JSON。
- 外部仓库源码与补丁，它们是方法适配和版本溯源依据。

## 结构收敛

- 根 `README.md` 收敛为稳定导航、目录边界和产物保留规则，不再重复维护逐日实验快照。
- 新增 `docs/README.md`，区分当前规范、最新结果、历史快照和专项审计。
- 修正调度快照 README 中 5 个失效的相对链接。

## 验证

- 整理前测试：153 passed，20 条既有 PyTorch warning。
- 整理后测试：153 passed，20 条同类既有 PyTorch warning。
- Markdown 相对链接缺失数：0。
- 仓库内 60 个 JSON 配置/报告均可解析。
- `git diff --check` 通过；整理后未残留 Python/pytest 缓存目录。
