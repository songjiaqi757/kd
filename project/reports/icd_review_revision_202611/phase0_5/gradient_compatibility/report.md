# A5 梯度诊断

已产出 36/36 个 checkpoint，每个六个固定 Train microbatch，batch=1，eval 模式、dropout 关闭。分别报告 Fusion、Text/Audio/Visual LoRA、Prediction Head、other 和全部参数。余弦零分母记缺失，不当作零或负数。原始 Full KD 使用原实验 seed2026 TAV target；Subset/ICD 使用原 ensemble3 target。N2 辅助项及总目标严格使用历史 ensemble3 TAV regression/probability、No-CE、0.30 ICD，并额外对 task_no_ce 计算余弦。原始与 N2 不混为统一 target 的受控训练。

阶段预先按 epoch1–6 / 7–13 / 14+ 定义；13 个 case-stage 缺少保留权重。已使用所有 seed13 保留的不同 epoch，而没有重训补造 Early。完整 Early/Middle/Late 覆盖为受限。六个 batch 的负余弦比例是诊断，不能证明梯度投影必然改善任务。

没有创建 optimizer，没有参数更新；checkpoint 哈希在加载前后核验，参数 version 未变化。
