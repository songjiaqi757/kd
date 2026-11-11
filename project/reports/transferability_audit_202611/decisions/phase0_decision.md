# Phase 0 决策

- mosi：ICD 验证集一/二/三阶归一化 MAE 为 0.6382 / 0.7173 / 0.9873。该误差衡量吸收，不直接代表坐标的任务效用。
- mosi：ICD seed13 的固定 Train 样本全参数 mean cosine(base, ICD) = 0.4066，负余弦比例 0.3333，raw gradient norm ratio = 0.9810；层组与坐标明细应优先用于定位，六个窗口不足以证明普遍训练冲突。
- mosei：ICD 验证集一/二/三阶归一化 MAE 为 0.3659 / 0.3935 / 0.4839。该误差衡量吸收，不直接代表坐标的任务效用。
- mosei：ICD seed13 的固定 Train 样本全参数 mean cosine(base, ICD) = 0.8100，负余弦比例 0.0000，raw gradient norm ratio = 0.1642；层组与坐标明细应优先用于定位，六个窗口不足以证明普遍训练冲突。
- sims：ICD 验证集一/二/三阶归一化 MAE 为 0.6912 / 0.6762 / 0.5984。该误差衡量吸收，不直接代表坐标的任务效用。
- sims：ICD seed13 的固定 Train 样本全参数 mean cosine(base, ICD) = -0.2570，负余弦比例 0.6667，raw gradient norm ratio = 0.9204；层组与坐标明细应优先用于定位，六个窗口不足以证明普遍训练冲突。
- sims_v2：ICD 验证集一/二/三阶归一化 MAE 为 0.6421 / 0.6078 / 0.5006。该误差衡量吸收，不直接代表坐标的任务效用。
- sims_v2：ICD seed13 的固定 Train 样本全参数 mean cosine(base, ICD) = 0.0667，负余弦比例 0.3333，raw gradient norm ratio = 0.7332；层组与坐标明细应优先用于定位，六个窗口不足以证明普遍训练冲突。

Phase 1–5 保持关闭。上述是描述性诊断，不是因果归因或自适应机制确认。
