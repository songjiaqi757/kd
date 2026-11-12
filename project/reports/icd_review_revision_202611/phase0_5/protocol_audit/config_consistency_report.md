# A7 数据与协议审计

当前代码：`259673b02cedfad31c10a1e8632df382784efaa0`；上游 Phase0 plan SHA256：`0aeb1486efd2ef49e0284f3b4417946afb7b941f765311d623e2207629c44c66`。36 个既有最佳 checkpoint、72 份 Train/Validation 预测已核验一致；运行配置及 checkpoint SHA256 单独保存。历史训练 commit 只在 run_config 实际记录时填写，未记录不由审计 commit 代替。Teacher Probe seed 与 Student seed 分列。原始 Full KD seed2026 与 ICD ensemble3 额外 target 的混杂仍需 Phase B 统一 target 控制。

N2：No-CE、lambda0.30、seed13、历史 best epoch23，配置预算40；不能称20-epoch复现。历史完整 loss 源码及配置已复制，target/probability规则保留。Test 只核对已有 metadata 字节指纹，不读取其预测或计算其任务效用。

SIMS/SIMS v2 的 parent/source-video 重叠见 cross_dataset_overlap.csv，不能视为完全独立的复现。官方 Train/Validation 的 source-video 重叠见 source_video_overlap.csv；grouped CV 按 source-video切分，外部 Validation 另提供 novel-source子集。Teacher/Probe 已有拟合过程不会因为离线CV而变成独立验证，全部离线 utility 是条件诊断。

资产复核异常数：0；完整清单见 asset_reverification.csv。
