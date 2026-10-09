# GPU1 原有 MOSEI 队列接续与并行调度

用户于2026-10-08明确更正：原有 GPU1 队列继续，不转到 MOSI。没有启动 MOSI 迁移训练。

旧18次训练中已有5次完成、1次运行，原来尚未开始的12次由新调度器接续：

| 方法 | Seeds | 训练次数 |
|---|---|---:|
| uniform_interaction_gradient_projected | 13、42、2026 | 3 |
| uniform_interaction_coordinate_compatible | 13、42、2026 | 3 |
| explicit_full_kd | 13、42、2026 | 3 |
| explicit_uniform_interaction | 13、42、2026 | 3 |

数据、目标、训练预算保持原队列协议。每次训练20轮，保存 `best_valid_mae.pt`、`best_valid_acc2_nonzero.pt` 和 `last.pt`；`best.pt` 为 MAE 最优的软链接，不额外保存每轮权重。

原有 MOSEI shared dropout seed2026 及其测评包装进程继续运行，由新调度器预留显存。旧派发父进程保持停止，只为避免与接续队列重复派发；不应再 SIGCONT 旧父进程。当前已有实验没有被终止。

新队列把12次训练和24次测评分成独立任务。每次测评只使用对应训练已完成后的一个验证最优 checkpoint，沿用原有 official-test 协议。测试结果不能用来修改新研究方法或调参。

调度约束：

- 每10秒检查任务状态与GPU剩余显存；一次只启动一个任务，启动后至少30秒再启动下一个。
- 每次训练预留34000 MiB，每次测评8000 MiB，额外保留6000 MiB余量。已启动但尚未分配CUDA显存的进程也计入预算。
- 同日显存实测修订：测评进程resident高于8000MiB估计，测评预留已上调至20000MiB；训练仍34000MiB，额外余量仍6000MiB。只更换调度器并接管已有训练，未重启训练进程。旧预算计划保留为`plan.before_memory_revision_20261008.json`。
- 同时最多2个训练任务、2个测评任务，实际并行数还受显存预算约束；不保证四个任务能同时运行。已就绪测评优先，不能装下的任务等待。
- 外部正在执行的旧run及其测评包装进程也计入预算，直到退出；其他GPU进程的实际占用由 nvidia-smi 计入。
- 任一任务失败即停止继续派发，已启动任务继续完成，失败任务保留日志等待检查；不自动降低batch size或跳过失败。
- 重启通过任务完成标记跳过已完成任务。检测到上次调度器的子进程仍存活时拒绝重复启动；半成品训练只有存在last才能续训，测评半成品需人工检查，不自动覆盖。

执行文件：`project/scripts/run_gpu_resource_queue.py`。

冻结任务计划：`outputs/experiments/equal_modality_extensions_v3/mosei/resource_queue/plan.json`。

状态与日志：同目录 `queue_status.json`、`scheduler.log`、`.logs/`。状态文件区分 pending/running/complete/failed/blocked。计划含12次新训练及24次测评，不包含原包装进程负责的当前run和它的两次测评。

在上述目录创建 `STOP_AFTER_CURRENT` 可停止继续派发，已启动任务继续完成。不要直接启动第二个调度器；GPU级文件锁防止重复调度。

验证：显存初始化预留、已用显存去重、训练/测评并行、磁盘余量、依赖顺序、保存规则及测试授权，共6项调度测试通过；任务计划dry-run通过。
