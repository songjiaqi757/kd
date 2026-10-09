# 新实验统一在 MOSI 验证

2026-10-08 用户要求：现在所有新的实验都在 MOSI 上进行，以验证有效性。

同日后续更正：用户明确要求 GPU1 原有实验“还是继续，不要转到mosi”。因此，下面关于迁移或暂停旧 MOSEI 队列的安排已被撤销：原队列剩余12次训练继续使用 MOSEI，改由显存感知调度器执行训练与测评并行。没有启动 MOSI 迁移任务。此前关于后续新增研究方向优先 MOSI 的安排与既有 MOSEI 队列分开处理。实际执行计划见 `outputs/experiments/equal_modality_extensions_v3/mosei/resource_queue/plan.json`。

最新执行更新：用户要求 GPU0 同时跑MOSI对应六方法和文献启发候选。已冻结14方法×3seed，复用两个原始ICD基线，新增40次训练。详见 [GPU0 MOSI正式安排](mosi_research_v1_20261008.md)。下文“没有启动新方法训练”为此前历史记录，当前以最新协议和队列状态为准。

本要求覆盖此前新增实验的数据集安排，包括交互阶数梯度分配、教师目标修正，以及尚未开始的扩展方法。当前不自动安排 MOSEI 新训练或跨数据集复验。

- 新实验使用 MOSI，seeds 为13、42、2026；固定同一训练预算和对照协议。
- 用 validation 筛选方法和 checkpoint，冻结方案后进行 official test 确认；不使用逐轮 test 指标调方法。
- 分别报告 `best_valid_mae.pt` 和 `best_valid_acc2_nonzero.pt` 的完整指标，另外保留续训用 `last.pt`；`best.pt` 可为软链接。
- 先核对已有 MOSI baseline 的配置和资产，兼容者复用。已有 decision ICD 的两个 seed 结果仍标为两 seed；不得写成三 seed 完成，也不自动为未通过筛选的方法补跑。
- 先诊断，再实现最小改动和必要对照。当前没有因本次数据集安排而启动新方法训练。

旧队列处理：MOSEI `equal_modality_extensions_v3` 的派发父进程已单独 SIGSTOP，防止继续启动尚未开始的实验。已经启动的 `uniform_interaction_shared_dropout_seed2026` 及其当前运行包装进程继续执行；旧包装进程仍按原协议做完成后的测评。没有中止训练或改动模型源码。进程信息与恢复注意事项见 `project/reports/mosi_only_queue_transition_20261008.json`。不要自动恢复旧 MOSEI 派发队列。
