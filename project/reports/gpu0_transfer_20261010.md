# GPU0 接续两个 MOSEI 实验（2026-10-10）

全部正式队列已于 2026-10-14 结束，最终训练和测评结果见 [全部实验最终汇总](test_results_summary_20261014.md)。迁移的两个实验及四次测评均已完成，GPU0 队列已结束。

按用户要求，将 GPU1 队列中尚未启动的两个训练实验及其四次依赖测评转至空闲 GPU0：

- `uniform_interaction_coordinate_compatible_seed42`
- `uniform_interaction_coordinate_compatible_seed2026`

GPU0 历史补测队列已完成。两个实验保留原输出目录、数据、训练目标、seed、20 轮预算、batch size 8、学习率、双 best checkpoint 保存和 official-test 协议；命令仅将 `--device cuda:1` 改为 `--device cuda:0`。各实验完成后自动测评验证 MAE 最优及验证 Acc2 非零最优 checkpoint。

两个训练各预留 34000 MiB，测评各预留 20000 MiB，共享余量 6000 MiB，沿用原资源调度器及每次启动间隔至少 30 秒的约束。

GPU1 计划移除了这六个任务，避免两卡重复派发。只替换 GPU1 调度器父进程，通过 `--adopt-running` 接管原有训练 PID 748926、3290347；两个训练进程未重启。原计划与原状态已备份在 GPU1 队列目录，旧的更早调度器仍保持停止。

- [迁移记录与进程身份](gpu0_transfer_20261010.json)
- [GPU0 计划](../../outputs/experiments/equal_modality_extensions_v3/mosei/resource_queue_gpu0_20261010/plan.json)
- [GPU0 状态](../../outputs/experiments/equal_modality_extensions_v3/mosei/resource_queue_gpu0_20261010/queue_status.json)
- [GPU1 状态](../../outputs/experiments/equal_modality_extensions_v3/mosei/resource_queue/queue_status.json)

两个计划均通过调度器校验；核对所有任务恰好归属一个队列、依赖完整、GPU1 运行进程命令与身份一致，以及 GPU0 显存和磁盘预算足够。
