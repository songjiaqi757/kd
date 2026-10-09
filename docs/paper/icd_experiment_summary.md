# ICD 实验结果总汇

> 最终方法名称：Interaction Coordinate Distillation (ICD)。`ICD/ICD.pdf` 是提交版本，本文件用于保存其结果依据与协议演变。
>
> 2026-09-24 快照采用 validation-MAE 选模，是当前正式口径；2026-09-23 快照包含较早的 test-sweep 诊断，只作为历史分支证据。

以下完整保留两次汇总，当前口径在前。

## 原文快照：`实验汇总_20260924.md`

## RDID-MSA 当前实验结果总汇（2026-09-24）

> 本文档是 2026-09-24 当前快照。主表只采用 **validation MAE 最低的 checkpoint**，
> official test 只评测该 checkpoint；不将 test-best 混入正式结果。所有已完成 epoch
> 的 checkpoint 仍全部保留。旧的 test-sweep 只作诊断，见
> [`实验汇总_20260923.md`](icd_experiment_summary.md)。
>
> 实时运行状态最后刷新：2026-09-24 16:18 CST。

### 1. 完成度

| 任务 | 完成情况 | 状态 |
|---|---:|---|
| MOSI 七方法 × 三 seed | 21/21 | 训练、valid-MAE 选模、official-test 全部完成 |
| MOSEI 七方法 × 三 seed | 12/21 完整 | 6 个训练中，3 个排队 |
| MOSEI Full KD / Uniform 三 seed | 6/6 | 复用已完成的 M3/M4 valid-best official-test |
| Official-valid interaction reconstruction | MOSEI + MOSI，图中六方法 | 完成；Ensemble 未纳入该诊断图 |
| Cross-modal fidelity / polarity outcome | MOSEI + MOSI | 完成，video-cluster bootstrap |
| Teacher 七子集 official-valid 推理 | MOSEI + MOSI | 完成 |
| 定性案例、效率与部署对比 | 主文所需项 | 完成 |

### 2. 指标口径

- `NZ`：排除真实标签为 0 的 Acc-2 / weighted F1。
- `HZ`：保留真实标签为 0 的 Acc-2 / weighted F1。
- 表中的 `±` 是三个 seed 的 sample standard deviation，不是置信区间。
- MOSEI official test 共 4,659 utterances（3,634 个 non-zero）；MOSI official test 共 686 utterances
  （656 个 non-zero）。

### 3. MOSI 七方法三 seed 正式结果

#### 3.1 Official-test 三 seed 均值 ± 标准差

| 方法 | MAE ↓ | Pearson ↑ | Acc-2 NZ ↑ | F1 NZ ↑ | Acc-2 HZ ↑ | F1 HZ ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full KD | 0.731630 ± 0.005597 | 0.784699 ± 0.011075 | 0.839939 ± 0.006986 | 0.840087 ± 0.007103 | 0.821672 ± 0.005891 | 0.821231 ± 0.006073 | **0.462585 ± 0.013466** |
| Ensemble Full KD | 0.729987 ± 0.012530 | 0.798586 ± 0.005622 | **0.851118 ± 0.000880** | **0.851740 ± 0.000936** | **0.838192 ± 0.004373** | **0.838347 ± 0.004674** | 0.455782 ± 0.016832 |
| Subset-7 KD | 0.757616 ± 0.037811 | 0.788780 ± 0.016624 | 0.845020 ± 0.016157 | 0.845775 ± 0.015873 | 0.834305 ± 0.015723 | 0.834609 ± 0.015532 | 0.434402 ± 0.018950 |
| First-order Interaction | 0.723460 ± 0.045025 | **0.799779 ± 0.013394** | 0.842480 ± 0.014962 | 0.842652 ± 0.014765 | 0.826531 ± 0.015219 | 0.826180 ± 0.015158 | 0.435374 ± 0.017269 |
| First+Second-order Interaction | 0.734492 ± 0.038039 | 0.789095 ± 0.019239 | 0.841972 ± 0.014806 | 0.842490 ± 0.014718 | 0.830418 ± 0.011872 | 0.830467 ± 0.011941 | 0.430029 ± 0.010204 |
| Random Orthogonal | **0.716921 ± 0.015365** | 0.797167 ± 0.008707 | 0.847053 ± 0.008932 | 0.847285 ± 0.008665 | 0.832362 ± 0.005256 | 0.832115 ± 0.005005 | 0.454810 ± 0.010512 |
| Uniform Interaction | 0.738779 ± 0.014125 | 0.789233 ± 0.016969 | 0.848577 ± 0.007520 | 0.849301 ± 0.007560 | 0.836735 ± 0.008867 | 0.837021 ± 0.008938 | 0.434888 ± 0.015723 |

目前 MOSI 的任务指标不支持“Uniform 全面最优”：Random Orthogonal 的平均 MAE 最低，
First-order 的 Pearson 最高，Ensemble Full KD 的两套 Acc-2/F1 最高，Full KD 的
Acc-7 最高。Uniform 的主要优势体现在 interaction reconstruction，而不是每项任务指标。

#### 3.2 MOSI 逐 seed 结果

| 方法 | Seed | Valid-best epoch | Test MAE | Pearson | Acc-2 NZ | F1 NZ | Acc-2 HZ | F1 HZ | Acc-7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full KD | 13 | 19 | 0.728486 | 0.796070 | 0.846037 | 0.846402 | 0.827988 | 0.827815 | 0.454810 |
| Full KD | 42 | 4 | 0.738093 | 0.773946 | 0.832317 | 0.832397 | 0.816327 | 0.815849 | 0.478134 |
| Full KD | 2026 | 19 | 0.728312 | 0.784081 | 0.841463 | 0.841463 | 0.820700 | 0.820029 | 0.454810 |
| Ensemble Full KD | 13 | 14 | 0.716657 | 0.797965 | 0.852134 | 0.852768 | 0.838192 | 0.838389 | 0.475219 |
| Ensemble Full KD | 42 | 15 | 0.731781 | 0.793300 | 0.850610 | 0.850936 | 0.833819 | 0.833652 | 0.446064 |
| Ensemble Full KD | 2026 | 8 | 0.741524 | 0.804492 | 0.850610 | 0.851516 | 0.842566 | 0.842999 | 0.446064 |
| Subset-7 KD | 13 | 11 | 0.754214 | 0.797890 | 0.847561 | 0.848335 | 0.841108 | 0.841471 | 0.434402 |
| Subset-7 KD | 42 | 19 | 0.721620 | 0.798857 | 0.859756 | 0.860212 | 0.845481 | 0.845528 | 0.453353 |
| Subset-7 KD | 2026 | 4 | 0.797013 | 0.769592 | 0.827744 | 0.828778 | 0.816327 | 0.816828 | 0.415452 |
| First-order Interaction | 13 | 14 | 0.773234 | 0.791980 | 0.833841 | 0.834730 | 0.819242 | 0.819618 | 0.415452 |
| First-order Interaction | 42 | 20 | 0.711578 | 0.792111 | 0.833841 | 0.833538 | 0.816327 | 0.815409 | 0.446064 |
| First-order Interaction | 2026 | 17 | 0.685567 | 0.815244 | 0.859756 | 0.859687 | 0.844023 | 0.843513 | 0.444606 |
| First+Second-order Interaction | 13 | 13 | 0.732626 | 0.798077 | 0.829268 | 0.830298 | 0.822157 | 0.822649 | 0.425656 |
| First+Second-order Interaction | 42 | 7 | 0.773430 | 0.767007 | 0.838415 | 0.838335 | 0.825073 | 0.824541 | 0.422741 |
| First+Second-order Interaction | 2026 | 11 | 0.697421 | 0.802200 | 0.858232 | 0.858839 | 0.844023 | 0.844213 | 0.441691 |
| Random Orthogonal | 13 | 16 | 0.719354 | 0.804637 | 0.853659 | 0.853978 | 0.836735 | 0.836570 | 0.451895 |
| Random Orthogonal | 42 | 13 | 0.700485 | 0.799259 | 0.850610 | 0.850379 | 0.833819 | 0.833075 | 0.446064 |
| Random Orthogonal | 2026 | 17 | 0.730924 | 0.787604 | 0.836890 | 0.837498 | 0.826531 | 0.826700 | 0.466472 |
| Uniform Interaction | 13 | 15 | 0.722491 | 0.808656 | 0.853659 | 0.854475 | 0.842566 | 0.842953 | 0.441691 |
| Uniform Interaction | 42 | 10 | 0.747678 | 0.781764 | 0.839939 | 0.840625 | 0.826531 | 0.826741 | 0.446064 |
| Uniform Interaction | 2026 | 5 | 0.746167 | 0.777280 | 0.852134 | 0.852805 | 0.841108 | 0.841369 | 0.416910 |

### 4. MOSEI 七方法三 seed 当前结果

#### 4.1 已完成三 seed 的方法

| 方法 | MAE ↓ | Pearson ↑ | Acc-2 NZ ↑ | F1 NZ ↑ | Acc-2 HZ ↑ | F1 HZ ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full KD | 0.479825 ± 0.001906 | 0.816943 ± 0.004237 | 0.872776 ± 0.006062 | 0.872778 ± 0.005701 | 0.818130 ± 0.015597 | 0.823853 ± 0.014060 | 0.569793 ± 0.004943 |
| Uniform Interaction | **0.476242 ± 0.004540** | **0.823668 ± 0.001264** | **0.877821 ± 0.004429** | **0.877321 ± 0.004238** | **0.830150 ± 0.011808** | **0.834578 ± 0.010913** | **0.571367 ± 0.003611** |

在已完成的 MOSEI 三 seed 对比中，Uniform 在表内全部七项 official-test 指标均值上
都优于 Full KD。其他五个方法尚未集齐三 seed，因此现在不做七方法最终排名。

#### 4.2 已完成的 12 个 valid-best official-test run

| 方法 | Seed | Valid-best epoch | Valid MAE | Test MAE | Pearson | Acc-2 NZ | F1 NZ | Acc-2 HZ | F1 HZ | Acc-7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full KD | 13 | 6 | 0.471261 | 0.481628 | 0.812610 | 0.868740 | 0.868710 | 0.816914 | 0.822557 | 0.575231 |
| Full KD | 42 | 15 | 0.466293 | 0.477830 | 0.817140 | 0.879747 | 0.879294 | 0.834299 | 0.838515 | 0.565572 |
| Full KD | 2026 | 10 | 0.463737 | 0.480018 | 0.821078 | 0.869840 | 0.870329 | 0.803177 | 0.810485 | 0.568577 |
| Ensemble Full KD | 13 | 11 | 0.471800 | 0.486895 | 0.817231 | 0.878371 | 0.878371 | 0.835158 | 0.839767 | 0.568362 |
| Subset-7 KD | 13 | 9 | 0.469039 | 0.480143 | 0.816999 | 0.872317 | 0.871513 | 0.822923 | 0.827392 | 0.569006 |
| Subset-7 KD | 2026 | 5 | 0.465798 | 0.480930 | 0.817778 | 0.878371 | 0.876572 | 0.862417 | 0.862960 | 0.574158 |
| First-order Interaction | 13 | 14 | 0.467276 | 0.485712 | 0.816425 | 0.864337 | 0.865356 | 0.794806 | 0.803078 | 0.567504 |
| First+Second-order Interaction | 13 | 14 | 0.465983 | 0.478789 | 0.819214 | 0.874243 | 0.874272 | 0.818845 | 0.824605 | 0.570079 |
| Random Orthogonal | 13 | 8 | 0.466510 | 0.478300 | 0.820199 | 0.870391 | 0.870720 | 0.810045 | 0.816675 | 0.571797 |
| Uniform Interaction | 13 | 6 | 0.474555 | 0.477360 | 0.822221 | 0.881948 | 0.880998 | 0.838807 | 0.842195 | 0.570079 |
| Uniform Interaction | 42 | 11 | 0.463681 | 0.471247 | 0.824225 | 0.873143 | 0.872686 | 0.816699 | 0.822076 | 0.575445 |
| Uniform Interaction | 2026 | 11 | 0.471173 | 0.480118 | 0.824557 | 0.878371 | 0.878278 | 0.834943 | 0.839463 | 0.568577 |

#### 4.3 正在训练/排队的 9 个 run

| Run | 快照进度 | 已完成 validation | 当前最佳 epoch | 当前最佳 Valid MAE |
|---|---|---:|---:|---:|
| Subset-7 seed42 | epoch 15 step 1/2050 | 14 | 11 | 0.465675 |
| Random Orthogonal seed42 | epoch 15 step 1/2050 | 14 | 13 | 0.470215 |
| Random Orthogonal seed2026 | epoch 15 step 200/2050 | 14 | 13 | 0.465514 |
| Ensemble Full KD seed42 | epoch 3 step 2000/2050 | 2 | 2 | 0.488292 |
| Ensemble Full KD seed2026 | epoch 3 step 2000/2050 | 2 | 1 | 0.507694 |
| First+Second-order seed42 | epoch 1 step 800/2050 | 0 | — | — |
| First-order seed42 | pending | 0 | — | — |
| First+Second-order seed2026 | pending | 0 | — | — |
| First-order seed2026 | pending | 0 | — | — |

以上只是运行快照；当前 best 会在后续 validation 结束后继续更新。早停规则为
`min_epochs=8, patience=7`，最多 20 epoch。

### 5. Interaction reconstruction（official validation，seed13）

#### 5.1 Aggregate interaction reconstruction MAE

| 数据集 | 方法 | First-order | Second-order | Third-order | 7-coordinate MAE ↓ |
|---|---|---:|---:|---:|---:|
| MOSEI | Full KD | 0.448510 | 0.459349 | 0.416637 | 0.448602 |
| MOSEI | Subset-7 | 0.299325 | 0.283996 | 0.250033 | 0.285714 |
| MOSEI | First-order | 0.295048 | 0.277916 | 0.242573 | 0.280209 |
| MOSEI | First+Second-order | **0.291299** | **0.216186** | 0.334293 | **0.265250** |
| MOSEI | Random Orthogonal | 0.303832 | 0.294232 | 0.262112 | 0.293758 |
| MOSEI | Uniform Interaction | 0.306983 | 0.235795 | **0.239812** | 0.266878 |
| MOSI | Full KD | 1.058190 | 1.236499 | 1.122787 | 1.143837 |
| MOSI | Subset-7 | 0.668164 | 0.619870 | **0.431790** | 0.613699 |
| MOSI | First-order | 0.650444 | 0.592830 | 0.458235 | 0.598294 |
| MOSI | First+Second-order | 0.650942 | **0.425969** | 0.846797 | 0.582504 |
| MOSI | Random Orthogonal | 0.634766 | 0.578104 | 0.439460 | 0.582581 |
| MOSI | Uniform Interaction | **0.626975** | 0.460138 | 0.507741 | **0.538440** |

MOSEI 上 First+Second-order 的 overall 最低，Uniform 的 third-order 最低；MOSI 上 Uniform
的 overall 和 first-order 最低。因此不应声称 Uniform 在两个数据集的每一阶都最好。

#### 5.2 Cross-modal fidelity gain 与 polarity outcome

定义 `E_cross=(e_TA+e_TV+e_AV+e_TAV)/4`，
`ΔE_cross=E_cross,Full−E_cross,Uniform`；正值表示 Uniform 更好。

| 数据集 | Spearman ρ（连续误差 gain） | Video-cluster 95% CI | Corrected − Harmed | Video-cluster 95% CI |
|---|---:|---:|---:|---:|
| MOSEI | +0.0118 | [-0.0333, +0.0589] | +0.067724 | [+0.009628, +0.125607] |
| MOSI | +0.0271 | [-0.0916, +0.1379] | +0.418720 | [+0.220063, +0.735306] |

- 连续 prediction-error gain 与 fidelity gain 没有显著单调关系。
- Corrected 组相对 Harmed 组有更大的 cross-modal fidelity gain，可表述为
  `associated with / tend to`，不能作因果表述。
- MOSI Corrected/Harmed 各只有 9 条（各6个 source-video cluster），必须标为小样本
  探索性复现。

### 6. Teacher 子集与定性证据

- MOSEI teacher subset prediction MAE：T 0.473170、A 0.454263、V 0.719782、TA 0.450449、
  TV 0.457895、AV 0.437054、TAV 0.433936。
- MOSI teacher subset prediction MAE：T 0.511995、A 0.474950、V 1.404599、TA 0.495273、
  TV 0.555220、AV 0.521385、TAV 0.488946。
- 两个主定性案例仅根据 teacher interaction statistics 选取，与 student error 无关：
  `130456[11]` 是强 first-order 模态冲突；`125676[7]` 是强 text–audio interaction。
- 精确选择规则：先在 teacher 的一阶坐标 $I_T,I_A,I_V$ 中筛出同时含正、负坐标的
  696 条 validation utterance，再按
  $s_{opp}=\min(\max_m[I_m]_+,\max_m[-I_m]_+)$ 降序排列；`130456[11]`
  排名 6/696（top 0.86%）。第二例在全部 1,871 条 validation utterance 中按
  $|I_{TA}|$ 降序排列；`125676[7]` 的 $|I_{TA}|=1.891960$，排名 66/1,871
  （top 3.53%，即预先采用的 top-5% 区间）。选择过程不读取任何 student prediction、
  task error 或 reconstruction error。

### 7. 效率与历史分支结论

- 2026-09-24 已在本机 `inspur-NF5280M6` 的同一块物理 GPU 1
  （`NVIDIA RTX 6000D`）依次完成 MOSEI Teacher→Student 部署效率实测。固定
  official-validation 的 90 条单窗口 TAV utterance，按 `<5 s`、`5–15 s`、`15–30 s`
  各抽 30 条，每条只计时 1 次；另用每档 1 条不重叠样本 warm-up，warm-up 不进入统计。
  所有模型均为 batch size 1，每个 model forward 前后均执行 `torch.cuda.synchronize()`。
  Teacher 和 Probe 使用 BF16，Student 使用 BF16 autocast inference。Teacher 口径是论文实际使用的 `Qwen3-Omni Thinker hidden →
  seed2026 Probe`，不是 direct text generation；Student 均为 seed13、validation-MAE
  选择的 checkpoint。

| Model | 部署参数 | 可训参数 | Peak GPU (GiB) | Model-only p50 / p95 (ms) | E2E p50 / p95 (ms) | E2E utt/s | 90条诊断 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-Omni Teacher + Probe | 31.720B | 0（部署冻结） | 67.526 | 435.32 / 2103.59 | 888.47 / 4449.77 | 0.735 | 0.548651 |
| Full KD Student | 819.309M | 42.650M | 2.091 | 94.36 / 126.75 | 482.97 / 1472.69 | 1.630 | 0.552137 |
| ICD Student | 819.309M | 42.650M | 2.091 | 95.36 / 126.48 | 486.46 / 1471.17 | 1.630 | 0.557631 |

- Teacher/Student 参数压缩比为 `38.72×`，batch-1 峰值显存比为 `32.29×`；Student
  model-only p50 快约 `4.6×`、p95 快约 `16.6×`，端到端吞吐约 `2.22×`。
- **Model-only latency 边界**：输入已经在 CPU 侧完成预处理；计时包括 host→device
  transfer、模型 forward 和标量回归输出，不包括媒体 I/O / decode、tokenization 与各模态
  preprocessing。
- **E2E throughput 边界**：包括 audio loading、video decoding / frame sampling、
  tokenization、多模态 preprocessing、host→device transfer 与 model forward；不包括
  DataLoader worker queue，也没有 multi-sample batching。吞吐定义为 `90 / E2E 总秒数`。
- **Peak GPU Mem. 定义**：warm-up 后重置统计，使用
  `torch.cuda.max_memory_allocated()`；包含已加载模型权重与 batch-1 inference tensor 的
  峰值 allocated memory，不包含 CUDA reserved/cache 与非 PyTorch allocation。
- 此前 Student 约 `5.425 GiB` 的记录也使用 allocated-memory 口径，但来自
  **batch size 8 的完整 official-test evaluator**；本表 `2.091 GiB` 是 batch size 1，
  因而两者不应直接比较，差异不构成冲突。
- Teacher 的路由 active-parameter architecture-equivalent 为 4.541B：按全部非 expert
  参数加每层 8/128 routed-expert 参数计算。这不是“每个多模态 token 实际触及参数数”
  的严格统一计数，论文必须连同定义报告，不能直接简写成精确 active params。
- Full KD 与 ICD（Uniform Interaction Distillation）的参数量、峰值显存和端到端吞吐相同；约 1 ms 的 model-only p50
  差异属于单次计时波动，不支持额外推理开销的结论。表中 90 条 MAE 仅用于同批效率
  诊断，不替代全量 official-validation 或 official-test 结果。
- MOSI seed13 上 Uniform 相对 Full KD：可训参数不变（42.650M），部署参数不变
  （819.309M），训练峰值约 25.03 GiB，test 峰值约 5.40 GiB；额外部署成本很小。
- **Training-time cost（MOSI seed13 实测）**：Full KD 和 ICD 的单卡训练墙钟分别为
  2,906.157 s（0.8073 GPU-h）与 2,900.673 s（0.8057 GPU-h），ICD/Full KD
  multiplier 为 **0.9981×**（−0.189%）。该数字来自现有完整训练 report，不含离线
  teacher-target 准备。
- **Offline target cache（MOSI 实测）**：30B teacher 对 train+valid 的七个非空子集只
  前向一次，生成 10,605 个 2,048 维 FP16 hidden-state cache；双卡 model-parallel
  extraction 用时 5,273.346 s（87.889 min，2.9296 GPU-h）。三个 Probe 随后并行训练
  45.635 s（约 0.0254 GPU-h），从 Probe 输出构造 interaction/ensemble targets 与
  student asset bundle 用时 7.786 s（CPU）。从 extraction 启动到冻结可复用 student
  targets 的总墙钟为 5,356.963 s（89.283 min），GPU 用量约 **2.955 GPU-h**。这些
  JSONL/NumPy 资产在所有 student 方法和 seeds 间复用；student 训练不再调用 30B
  teacher，也不在线重算 Möbius targets。
- P0 MOSEI：M4 Uniform official-test MAE `0.476242 ± 0.004540`，优于 M3 Full KD
  `0.479825 ± 0.001906`；M6 R×U 没有稳定优于 M3/M4。
- Stage A–C：interaction reconstruction 的显著改善并不自动转化为任务 MAE 改善。
- Stage D：LoRA 带来最稳定的性能增益；Hidden KD 退化，已按 gate 停止。
- Polarity KD：Acc-2 略升但 MAE 显著违反 safety gate，不再扩展。
- RDID-v2 开发实验中 U-only 最好（valid MAE 0.463781，当轮 test MAE 0.476687）；
  该批 test 已用于开发诊断，不是独立盲测确认。

### 8. 当前可支持的结论

1. 在 MOSEI 已完成的 Full KD vs Uniform 三 seed 对比中，Uniform 的七项平均
   official-test 指标都更好。
2. MOSI 七方法三 seed 中，Uniform 不是任务指标上的总体最优方法；Random
   Orthogonal 的平均 MAE 最低，Ensemble 的 polarity 指标最高。
3. Uniform 的一致优势主要是 interaction fidelity：MOSI overall 最优，MOSEI
   third-order 最优；但 MOSEI overall 略逊于 First+Second-order。
4. Polarity correction 与 cross-modal fidelity improvement 存在关联，但不能声称因果；MOSI
   的 corrected/harmed 组很小。
5. MOSEI 七方法的最终三 seed 排名必须等剩余 9 个 run 完成后再做。

### 9. 主要产物

- MOSI 七方法汇总：
  `outputs/experiments/multiseed_controls_v1/mosi_seven_controls_valid_mae/summary.json`
- MOSEI P0 三 seed official-test：
  `outputs/experiments/tav_main_v1/official_test_20260917/all_five_methods_summary.json`
- MOSEI Teacher→Student efficiency benchmark：
  `outputs/experiments/efficiency_benchmark_v1/mosei/summary.json`
- Cross-modal fidelity 分析：
  `project/reports/interaction_fidelity_task_gain_valid.json`
- 主图：
  `project/reports/paper_figures/fig_aggregate_interaction_reconstruction_both_flat80.pdf`
- Fidelity/polarity 行布局图：
  `project/reports/paper_figures/fig_interaction_fidelity_task_gain_valid_row_v2.pdf`
- 定性案例图：
  `project/reports/paper_figures/fig5_qualitative_interaction_distillation_flat80.pdf`
- 历史全量实验、旧分支与 test-sweep 诊断：
  [`实验汇总_20260923.md`](icd_experiment_summary.md)

## 原文快照：`实验汇总_20260923.md`

## RDID-MSA 实验汇总（2026-09-23）

> **最新正式矩阵：**2026-09-24 新增的 MOSI/MOSEI 七方法三 seed、完整
> NZ/HZ 指标和实时队列状态已汇总到
> [`实验汇总_20260924.md`](icd_experiment_summary.md)。本文保留为历史实验与分支证据总账。

> 核对时间：2026-09-23 19:03 +08:00。本文件只汇总实验协议、运行状态、数值结果、checkpoint 选择口径、统计结论和限制，不记录绘图与排版。

### 1. 口径与数据覆盖

- 数据集：CMU-MOSEI 和 CMU-MOSI。
- 主要固定学生对照：Adapted Student、Full KD、Subset-7 KD、Ensemble Full KD、First-order Interaction、First+Second-order Interaction、Uniform Interaction Distillation、Random Orthogonal，以及 Projector、EA-KD、CMAD-style CAFD。
- 最新 interaction evidence 均为 seed 13，在 official validation 上按 validation MAE 选 checkpoint。
- 逐 epoch test sweep 中的 test-best 仅是诊断/约定报告口径；它与 validation-selected 结果分开列出，不当作独立盲测证据。
- Interaction reconstruction 在所有方法上统一后验到同一 Anchored Möbius 坐标系 `[T,A,V,TA,TV,AV,TAV]`，不受训练时监督坐标或阶数影响。
- 全量七坐标重建 MAE 按 `(3×First + 3×Second + Third)/7` 计算，不是对三个阶数柱简单平均。

当前已重新扫描的全量记录：24,134 个实验产物、2,660 个 JSON 来源、200 个含训练历史的运行、3,360 条独立逐 epoch 记录、507 条报告内嵌 history、4,782 个标准指标块、224 条 Probe 子集指标和 512 个统计比较节点；JSON 解析失败为 0。逐运行、逐 epoch 和所有 test 记录见 [`全部实验数据总账_20260923.md`](../../outputs/experiment_catalog_20260923/README.md) 及 [`experiment_catalog_20260923`](../../outputs/experiment_catalog_20260923)。

### 2. 冻结的 MOSEI P0 主实验（三种子）

P0 的 15/15 个 train/valid run 和 15/15 个 valid-best official-test 测评均已完成。Official test 只用于报告，未用于该批 checkpoint 或超参选择。

| 方法 | Valid MAE ↓ | Official-test MAE ↓ |
|---|---:|---:|
| M0 冻结 TAV Student-only | 0.527314 ± 0.005825 | 0.536696 ± 0.008767 |
| M1 冻结 TAV Full KD | 0.514162 ± 0.003197 | 0.527306 ± 0.002139 |
| M3 Full KD + T/A/V LoRA | 0.467097 ± 0.003826 | 0.479825 ± 0.001906 |
| M4 M3 + Uniform Interaction | 0.469803 ± 0.005565 | **0.476242 ± 0.004540** |
| M6 M3 + Reliability × Utility | **0.463968 ± 0.001598** | 0.481887 ± 0.000657 |

结论：

- Valid 上 M6 最低，但相对 M3 的均值改善只有 0.003129，置信区间 `[-0.012249,+0.005558]` 跨 0。
- Test 上 M4 的汇总 MAE 最低；M6 比 M3 高 0.002062、比 M4 高 0.005645。
- 最稳定、幅度最大的收益仍来自 T/A/V 可适配表示；R×U 的泛化优势不成立。

### 3. MOSEI seed13 逐 epoch 实验

#### 3.1 Validation-selected 六方法（interaction evidence 主口径）

| 方法 | 完成 epoch | Valid 选中 epoch | Valid MAE ↓ | 该轮 Test MAE ↓ |
|---|---:|---:|---:|---:|
| Full KD | 20 | 6 | 0.471261 | 0.481628 |
| Subset-7 | 20 | 9 | 0.469039 | 0.480143 |
| First-order | 20 | 14 | 0.467276 | 0.485712 |
| First+Second-order | 20 | 14 | **0.465983** | 0.478789 |
| Random Orthogonal | 15 | 8 | 0.466510 | 0.478300 |
| Uniform Interaction | 20 | 6 | 0.474555 | **0.477360** |

`First+Second-order` 已训满 20 epoch，20/20 个 checkpoint 的 official-test 测评全部完成。Epoch 20 valid MAE 为 0.604817，valid-best 仍为 epoch 14。MOSEI interaction evidence 已按 validation-best 六个 checkpoint 重新生成，`provisional=false`。

#### 3.2 Test-best 诊断口径

| 方法 | Test-best epoch | Test MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Adapted Student | 3 | 0.528352 | 0.783018 | 0.856357 | 0.857108 | 0.541533 |
| Full KD | 6 | 0.481628 | 0.812610 | 0.868740 | 0.868710 | 0.575231 |
| Subset-7 | 9 | 0.480143 | 0.816999 | 0.872317 | 0.871513 | 0.569006 |
| Ensemble Full | 10 | 0.484757 | 0.812862 | 0.871491 | 0.871323 | 0.568362 |
| First-order | 8 | 0.482360 | 0.818290 | 0.876170 | 0.875918 | 0.575231 |
| First+Second-order | 14 | 0.478789 | 0.819214 | 0.874243 | 0.874272 | 0.570079 |
| Random Orthogonal | 14 | **0.475133** | **0.823634** | **0.880297** | **0.880121** | **0.575875** |
| Uniform Interaction | 6 | 0.477360 | 0.822221 | 0.881948 | 0.880998 | 0.570079 |
| Projector | 14 | 0.477205 | 0.819427 | 0.867089 | 0.867370 | 0.577592 |
| EA-KD | 9 | 0.479552 | 0.818063 | 0.876445 | 0.876321 | 0.572011 |
| CMAD-style CAFD | 13 | 0.481581 | 0.813789 | 0.871767 | 0.871747 | 0.572011 |

注意：Random Orthogonal 的 `0.475133` 是 test-best 数值，比 Uniform 的 `0.477360` 低 0.002227；但按 validation 选 checkpoint 时，Random 的 test MAE 为 0.478300，比 Uniform 的 0.477360 高 0.000940。两种口径不能混合。

### 4. MOSI seed13 完整对照

11 个方法的训练已完成。原九方法的保存轮次分别为 20/20/20/17/20/20/20/20/20；新增 First+Second-order 和 Random Orthogonal 均为 20 epoch。

#### 4.1 Test-best 口径

| 方法 | 已测 epoch | Test-best epoch | Test MAE ↓ | Pearson ↑ | Acc-2 ↑ | F1 ↑ | Acc-7 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adapted Student | 20 | 19 | 0.771375 | 0.765401 | 0.814024 | 0.813526 | 0.434402 |
| Full KD | 20 | 7 | 0.703903 | 0.804432 | 0.849085 | 0.848810 | 0.427114 |
| Projector | 20 | 10 | 0.692872 | 0.801082 | 0.853659 | 0.852899 | 0.465015 |
| EA-KD | 17 | 17 | 0.714829 | 0.795267 | 0.855183 | 0.854752 | 0.430029 |
| CMAD-style CAFD | 20 | 18 | 0.687814 | 0.805497 | **0.862805** | **0.863156** | **0.475219** |
| Subset-7 | 20 | 14 | 0.684607 | 0.805955 | 0.841463 | 0.839943 | **0.475219** |
| Ensemble Full | 20 | 18 | 0.684712 | 0.804083 | 0.844512 | 0.844003 | 0.462099 |
| First-order | 20 | 10 | 0.711129 | 0.789141 | 0.850610 | 0.849628 | 0.462099 |
| First+Second-order | 20 | 11 | 0.692794 | 0.799070 | 0.847561 | 0.847951 | 0.454810 |
| Random Orthogonal | 20 | 12 | 0.687693 | 0.810502 | 0.859756 | 0.860006 | 0.470845 |
| Uniform Interaction | 20 | 13 | **0.681703** | **0.816964** | 0.849085 | 0.847698 | 0.441691 |

Uniform 在 test-best 口径下 MAE 最低，比 Full KD、Subset-7、First-order、First+Second-order 和 Random Orthogonal 分别低 0.022200、0.002904、0.029426、0.011090 和 0.005990。这些都是单 seed 且 test 参与选轮的探索性结果。

#### 4.2 Validation-selected 六方法

| 方法 | Valid-best epoch | Valid MAE ↓ | 该轮 Test MAE ↓ |
|---|---:|---:|---:|
| Full KD | 19 | **0.731167** | 0.728304 |
| Subset-7 | 11 | 0.783804 | 0.753337 |
| First-order | 14 | 0.781684 | 0.773136 |
| First+Second-order | 13 | 0.764173 | 0.733550 |
| Random Orthogonal | 16 | 0.756575 | **0.719745** |
| Uniform Interaction | 15 | 0.760095 | 0.721709 |

这个表与 test-best 表回答不同问题：validation-selected 下 Random 的该轮 test MAE 比 Uniform 低 0.001964；但 Uniform 的 interaction reconstruction 明显更好，见下节。

### 5. Aggregate interaction reconstruction（official validation）

#### 5.1 MOSEI（1,871 utterances）

| 方法 | First-order MAE | Second-order MAE | Third-order MAE | 7-coordinate MAE ↓ |
|---|---:|---:|---:|---:|
| Full KD | 0.448510 | 0.459349 | 0.416637 | 0.448602 |
| Subset-7 | 0.299325 | 0.283996 | 0.250033 | 0.285714 |
| First-order | 0.295048 | 0.277916 | 0.242573 | 0.280209 |
| First+Second-order | **0.291299** | **0.216186** | 0.334293 | **0.265250** |
| Random Orthogonal | 0.303832 | 0.294232 | 0.262112 | 0.293758 |
| Uniform Interaction | 0.306983 | 0.235795 | **0.239812** | 0.266878 |

MOSEI 上 First+Second-order 的全七坐标 MAE 比 Uniform 低 0.001628；Uniform 的三阶误差最低。因此不应宣称 Uniform 在 MOSEI 每一阶或 overall 都最好。

#### 5.2 MOSI（229 utterances）

| 方法 | First-order MAE | Second-order MAE | Third-order MAE | 7-coordinate MAE ↓ |
|---|---:|---:|---:|---:|
| Full KD | 1.058190 | 1.236499 | 1.122787 | 1.143837 |
| Subset-7 | 0.668164 | 0.619870 | **0.431790** | 0.613699 |
| First-order | 0.650444 | 0.592830 | 0.458235 | 0.598294 |
| First+Second-order | 0.650942 | **0.425969** | 0.846797 | 0.582504 |
| Random Orthogonal | 0.634766 | 0.578104 | 0.439460 | 0.582581 |
| Uniform Interaction | **0.626975** | 0.460138 | 0.507741 | **0.538440** |

MOSI 上 Uniform 的 overall 七坐标重建 MAE 最低，但不是每一阶都最低：First+Second 的二阶最低，Subset-7 的三阶最低。

### 6. Interaction-strength 分层任务 MAE

强度定义为 teacher `TA/TV/AV/TAV` 绝对值均值，然后按 validation 样本分为三等份。

| 数据集 | 方法 | Low MAE | Medium MAE | High MAE |
|---|---|---:|---:|---:|
| MOSEI | Full KD | 0.410029 | 0.455779 | 0.547951 |
| MOSEI | Subset-7 | 0.411105 | 0.450923 | 0.545061 |
| MOSEI | First-order | **0.392296** | 0.457202 | 0.552314 |
| MOSEI | First+Second-order | 0.403604 | 0.450587 | **0.543734** |
| MOSEI | Random Orthogonal | 0.406724 | **0.445837** | 0.546936 |
| MOSEI | Uniform Interaction | 0.412887 | 0.456623 | 0.554125 |
| MOSI | Full KD | 0.770924 | **0.666431** | **0.756111** |
| MOSI | Subset-7 | 0.749691 | 0.776632 | 0.832133 |
| MOSI | First-order | 0.777899 | 0.781438 | 0.790453 |
| MOSI | First+Second-order | 0.761523 | 0.739492 | 0.791540 |
| MOSI | Random Orthogonal | 0.793738 | 0.686668 | 0.788831 |
| MOSI | Uniform Interaction | **0.742380** | 0.771083 | 0.774399 |

结论：重建误差较低不等于每个交互强度分层的任务 MAE 都最低。这是机制证据，不应扩大成“更好的交互重建必然提升每个样本的情感预测”。

#### 6.1 Cross-modal interaction fidelity 与 polarity correction

使用 seed13、按 validation MAE 选中的 Full KD 和 Uniform checkpoint。机制主变量只取真正的
跨模态坐标：`E_cross=(e_TA+e_TV+e_AV+e_TAV)/4`，并定义
`ΔE_cross=E_cross,Full−E_cross,Uniform`。正值表示 Uniform 恢复了更多 teacher
cross-modal interaction structure。七坐标 overall fidelity 只保留为辅助审计。

连续回归误差诊断仍未显示单调关系：

| 数据集 | Mean ΔE_cross | Mean ΔM | Spearman ρ | Asymptotic p | Video-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| MOSEI | +0.211872 | -0.003293 | +0.0118 | 0.6105 | [-0.0333,+0.0589] |
| MOSI | +0.736032 | -0.031203 | +0.0271 | 0.6828 | [-0.0916,+0.1379] |

主分析改为 Acc-2 non-zero polarity outcome 分组。排除真实标签为 0 的样本；真实标签 `>0`
为 positive，回归预测 `>=0` 为 predicted positive。下表纵轴量均为组内 mean `ΔE_cross`：

| 数据集 | Outcome | N / video clusters | Mean ΔE_cross | Cluster-bootstrap 95% CI |
|---|---|---:|---:|---:|
| MOSEI | Full wrong → Uniform correct | 54 / 44 | +0.128126 | [+0.090106,+0.171520] |
| MOSEI | Both correct | 1,208 / 273 | +0.248418 | [+0.231158,+0.266226] |
| MOSEI | Both wrong | 129 / 95 | +0.089356 | [+0.057728,+0.119436] |
| MOSEI | Full correct → Uniform wrong | 47 / 42 | +0.060401 | [+0.016308,+0.109611] |
| MOSI | Full wrong → Uniform correct | 9 / 6 | +0.612899 | [+0.419113,+0.830056] |
| MOSI | Both correct | 175 / 10 | +0.838131 | [+0.602569,+1.013536] |
| MOSI | Both wrong | 23 / 9 | +0.392014 | [+0.061889,+0.632542] |
| MOSI | Full correct → Uniform wrong | 9 / 6 | +0.194179 | [-0.173482,+0.499121] |

关键 corrected-minus-harmed contrast 为：MOSEI `+0.067724`，原视频聚类 95% CI
`[+0.009628,+0.125607]`；MOSI `+0.418720`，CI `[+0.220063,+0.735306]`。
因此在当前固定 seed13 validation 模型中，被 Uniform 纠正 polarity 的样本，相比被 Uniform
改错的样本，确实伴随更大的 cross-modal interaction fidelity improvement。该结果支持
`associated with / tend to`，不支持因果表述，也不表示 corrected 组在四组中绝对最高。

MOSEI Full/Uniform Acc-2 为 0.872740/0.877608，exact McNemar `p=0.5507`；MOSI 两者均为
0.851852，McNemar `p=1.0000`。所以本图提供的是错误转移的机制证据，不是 Acc-2 总体显著提升证据。
尤其 MOSI 只有 10 个原视频 cluster，corrected/harmed 各仅 9 条、6 个 cluster，必须作为探索性
复现看待。任务预测取自与 interaction reconstruction 相同的已保存七子集 forward。完整结果见
`project/reports/interaction_fidelity_task_gain_valid.{json,md}`，图见
`project/reports/paper_figures/fig_interaction_fidelity_task_gain_valid.{pdf,png}`，论文 caption 见同目录
`fig_interaction_fidelity_task_gain_valid_caption.md`。

### 7. Teacher subset behavior 与 interaction statistics

#### 7.1 MOSEI teacher（validation）

| 坐标 | Mean absolute magnitude | Median absolute magnitude | Signed mean |
|---|---:|---:|---:|
| T | 0.672182 | 0.515789 | +0.075995 |
| A | 0.646206 | 0.482218 | +0.059842 |
| V | 0.412124 | 0.328417 | -0.002002 |
| TA | 0.647186 | 0.495364 | -0.071545 |
| TV | 0.361832 | 0.310630 | -0.017770 |
| AV | 0.355143 | 0.321967 | -0.040594 |
| TAV | 0.375814 | 0.329519 | +0.047648 |

Teacher subset prediction MAE：T 0.473170、A 0.454263、V 0.719782、TA 0.450449、TV 0.457895、AV 0.437054、TAV 0.433936。视觉单模态显著弱于文本/音频和多模态子集。

#### 7.2 MOSI teacher（validation）

| 坐标 | Mean absolute magnitude | Median absolute magnitude | Signed mean |
|---|---:|---:|---:|
| T | 1.332315 | 1.368886 | +0.057556 |
| A | 1.375750 | 1.447827 | +0.052197 |
| V | 0.315391 | 0.251928 | +0.149093 |
| TA | 1.347384 | 1.358567 | -0.101262 |
| TV | 0.402050 | 0.334166 | -0.115043 |
| AV | 0.319282 | 0.269316 | -0.131739 |
| TAV | 0.463535 | 0.363643 | +0.131235 |

Teacher subset prediction MAE：T 0.511995、A 0.474950、V 1.404599、TA 0.495273、TV 0.555220、AV 0.521385、TAV 0.488946。MOSI 的 T/A/TA interaction 幅值明显大于视觉相关坐标。

### 8. Qualitative cases（teacher-only 选样）

两个主案例都仅使用 teacher interaction statistics 选择，与 student prediction error 无关，避免按 Uniform 表现挑样本。

#### 8.1 Conflicting first-order modality effects

- 样本 `130456[11]`，teacher first-order 为 T `+1.371`、A `+1.324`、V `-0.982`。
- 冲突强度在 696 个 mixed-sign validation 样本中排名 6（top 0.86%）。
- 7-coordinate reconstruction MAE：Full KD 0.841、Subset-7 0.646、First-order 0.763、Uniform **0.614**。
- GT `+0.667`；预测/绝对误差：Full KD `+1.922/1.255`、Subset-7 `+1.477/0.810`、First-order `+1.891/1.224`、Uniform `+0.633/0.034`。

#### 8.2 Strong text–audio interaction

- 样本 `125676[7]`，teacher `I_TA=+1.892`，在 1,871 个 validation 样本中排名 66（top 3.53%）。
- 7-coordinate reconstruction MAE：Full KD 0.544、Subset-7 0.568、First-order 0.536、Uniform **0.276**。
- GT `-1.333`；预测/绝对误差：Full KD `-1.406/0.073`、Subset-7 `-1.211/0.122`、First-order `-1.156/0.177`、Uniform `-1.320/0.013`。

案例只支持 qualitative evidence：Uniform 在这些例子中获得更低的整体交互重建误差，不代表每个坐标都最接近 teacher，也不能从单个样本推导整体因果结论。

### 9. 效率实验（MOSI seed13）

| 方法 | 训练分钟 | 训练峰值 GiB | 可训参数 M | 部署参数 M | Test 模型时间 s | Test 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|
| Full KD | 48.436 | 25.032 | 42.650 | 819.309 | 14.861 | 5.404 |
| Projector | 54.931 | 25.049 | 43.701 | 819.309 | 25.307 | 5.408 |
| Subset-7 | 40.277 | 25.033 | 42.650 | 819.309 | 14.980 | 5.404 |
| First-order | 50.157 | 25.033 | 42.650 | 819.309 | 15.092 | 5.404 |
| Uniform Interaction | 48.345 | 25.033 | 42.650 | 819.309 | 15.063 | 5.404 |

Uniform 相对 Full KD：训练时间 -0.189%、可训参数 0%、训练显存 +0.005%、部署参数 0%、推理时间 +1.363%、推理显存 0%。结论是 Uniform 基本不增加参数和显存成本，本次推理时间差异很小。

### 10. 历史机制实验与阶段结论

#### 10.1 benchmark500 学生蒸馏消融

seed2026 的 50 条 validation 小样本结果：Student-only 0.7253、Full KD 0.6202、Subset-7 value KD 0.6306、Möbius-7 KD 0.6285、subset4 **0.5749**、interaction4 0.6379。Full KD 相对 Student-only 的 MAE 差 -0.1051，95% CI `[-0.2059,-0.0055]`。Subset-value 与 Möbius 参数化本身不自动创造额外监督信息。

#### 10.2 Stage A v2 确定性复验

- 19 次确定性运行完成；同 seed 跨 GPU 的逐 epoch loss、指标和梯度范数一致。
- 三种子 benchmark500：SNR pair-only 与 selective-top50 的 MAE 均为 **0.6221**，subset4 为 0.6362，raw interaction4 为 0.6423。
- 但 validation 只有 50 条，结果仅用于候选筛选。

#### 10.3 Official-valid full-scale subset4 vs SNR pair-only

| 方法 | MAE | Pearson | Acc-2 | F1 | Acc-7 | E_pair |
|---|---:|---:|---:|---:|---:|---:|
| subset4 | **0.5084 ± 0.0050** | **0.7465 ± 0.0068** | **0.8579 ± 0.0038** | **0.8580 ± 0.0040** | **0.5505 ± 0.0082** | 0.3905 ± 0.0171 |
| SNR pair-only | 0.5085 ± 0.0013 | 0.7408 ± 0.0069 | 0.8530 ± 0.0088 | 0.8538 ± 0.0090 | 0.5491 ± 0.0051 | **0.2519 ± 0.0066** |

SNR pair-only 明显改善 pair interaction reconstruction，但总体任务 MAE 没有稳定改善；三个 seed 的 MAE bootstrap CI 均跨 0，full-scale 下游 gate 为 No-Go。

#### 10.4 Stage B official-valid 主基线

| 方法 | MAE | Pearson | Acc-2 | F1 | Acc-7 |
|---|---:|---:|---:|---:|---:|
| Student-only | 0.5323 ± 0.0080 | 0.7180 ± 0.0073 | 0.8479 ± 0.0082 | 0.8428 ± 0.0107 | 0.5396 ± 0.0062 |
| Full KD | **0.5072 ± 0.0046** | 0.7423 ± 0.0064 | 0.8530 ± 0.0060 | 0.8529 ± 0.0049 | **0.5507 ± 0.0043** |
| subset4 | 0.5084 ± 0.0050 | **0.7465 ± 0.0068** | **0.8579 ± 0.0038** | **0.8580 ± 0.0040** | 0.5505 ± 0.0082 |

Full KD 和 subset4 均在 3/3 seeds 的 MAE 上显著优于 Student-only；subset4 没有证明能在 Full KD 上继续稳定改善 MAE。

#### 10.5 Stage C interaction utility

| 方法 | MAE | Pearson | Acc-7 | E_pair |
|---|---:|---:|---:|---:|
| Full KD | 0.5072 ± 0.0046 | 0.7423 ± 0.0064 | 0.5507 ± 0.0043 | 0.7202 ± 0.1559 |
| Uniform ensemble-pair | 0.5087 ± 0.0014 | 0.7445 ± 0.0036 | 0.5519 ± 0.0064 | **0.2425 ± 0.0019** |
| R×U | **0.5055 ± 0.0023** | 0.7451 ± 0.0027 | **0.5583 ± 0.0031** | 0.2432 ± 0.0054 |
| selective-top50 | 0.5061 ± 0.0047 | **0.7462 ± 0.0030** | 0.5557 ± 0.0086 | 0.2638 ± 0.0109 |

R×U 的平均 MAE 比 Full KD 低 0.0017，未达到预注册 0.005 效果量，三个逐 seed CI 均跨 0；因此不支持“R×U 稳定优于 Full KD”。

#### 10.6 Stage D LoRA 与 Hidden KD

| 方法 | MAE | Pearson | Acc-2 | F1 | Acc-7 |
|---|---:|---:|---:|---:|---:|
| Full KD | 0.5072 ± 0.0046 | 0.7423 ± 0.0064 | 0.8530 ± 0.0060 | 0.8529 ± 0.0049 | 0.5507 ± 0.0043 |
| Full KD + LoRA(T+A) | 0.4728 ± 0.0057 | 0.7907 ± 0.0026 | 0.8739 ± 0.0119 | 0.8739 ± 0.0118 | 0.5744 ± 0.0059 |
| R×U + LoRA(T+A) | **0.4646 ± 0.0008** | **0.7946 ± 0.0043** | **0.8762 ± 0.0014** | **0.8763 ± 0.0016** | **0.5797 ± 0.0096** |

LoRA 改善在 3/3 seeds 方向一致。R×U+LoRA 相对 Full-KD+LoRA 平均改善 0.008244，但只有 seed42 的单 seed 聚类 bootstrap CI 不跨 0。Hidden KD 两种子 MAE 为 0.5114 ± 0.0055，比同 seed Full KD 退化 0.0033，分支按 gate 停止。

#### 10.7 Polarity KD

Binary subset4 相对 subset4 的 Acc-2 在 2/2 seeds 上升约 0.56–0.70 个百分点，但 bootstrap CI 均跨 0、McNemar `p>0.45`，同时 MAE 平均退化约 0.025，超过 0.005 safety gate，因此不再扩展该分支。

#### 10.8 RDID-v2 MOSEI 开发实验

| 方法 | 状态/已测轮次 | Valid-best epoch | Valid MAE | 该轮 Test MAE |
|---|---:|---:|---:|---:|
| Soft-RU α=0.25 | 完成/21 | 14 | 0.469490 | 0.482976 |
| Soft-RU α=0.50 | 完成/20 | 13 | 0.469485 | 0.479558 |
| Soft-RU α=0.75 | 完成/14 | 7 | 0.466370 | 0.489116 |
| R-only | 完成/24 | 17 | 0.468907 | 0.482318 |
| U-only | 完成/21 | 14 | **0.463781** | **0.476687** |
| Amplitude×U | 暂停/15 | 14（暂定） | 0.467881 | 0.477220 |

该批 test 已用于开发性诊断，不是盲测确认。Amplitude×U 仍是未完整训练快照。

### 11. 当前未完成与受限项

- MOSEI First+Second-order 已完成 20/20 epoch 训练和 20/20 official-test 测评；Random Orthogonal 已完成 15/15。六方法 validation-selected interaction evidence 和自动 finalization 已完成。
- MOSI First+Second-order 和 Random Orthogonal 的 20/20 epoch 训练与 20/20 test sweep 均已完成。
- `main_table_v1` 原定两数据集各 53 次训练的完整多种子最低充分集尚未全部执行；当前大量新比较仍是 seed13。
- DLF/DPDF-LQ 原生系统因标准 `aligned_50.pkl` 和原生 `bert-base-uncased` 资产不齐未启动；GsiT 仍为可选项。
- 不能从单 seed、test-best 或两个 qualitative case 推导多种子稳定优势。
- Full KD 与 Uniform 的连续 task-error gain 和 cross-modal fidelity gain 没有稳健的单调关系；但 polarity corrected 组的 `ΔE_cross` 高于 polarity harmed 组，两个数据集的原视频聚类区间均高于 0。该分组结果仍是固定 seed13 的关联性机制证据。

### 12. 当前可支持的总结

1. Full KD 相对 Student-only 的改善在早期小样本和 official-valid 三种子上都得到支持。
2. 开放 T/A 表示适配带来的改善幅度最稳定，明显大于多数 interaction weighting 变体。
3. Uniform Interaction 在冻结 P0 的 MOSEI official-test 三种子汇总上获得最低 MAE，在 MOSI seed13 test-best 上也获得最低 MAE；后者是 test-selected 探索性结果。
4. Uniform 在 MOSI 的七坐标 interaction reconstruction 最好；MOSEI 上 First+Second-order 以约 0.0016 的很小差距优于 Uniform。
5. First+Second-order 直接改善了二阶坐标重建，但 MOSI 三阶重建误差明显增大，说明监督阶数的改善不会自动外推到未监督高阶项。
6. Random Orthogonal 在部分 test-best 指标上很强，但其 interaction reconstruction 没有稳定优于有语义的 Uniform 坐标，这为“特定交互基并非任意旋转基”提供对照证据。
7. 当前最安全的论文表述是：Uniform 在不显著增加参数或显存开销的情况下改善 teacher interaction structure 的恢复；连续 fidelity gain 与回归误差改善没有单调关系，但被 Uniform 纠正 polarity 的样本相较反向改错样本往往具有更大的 cross-modal fidelity gain。后者是关联而非因果，且 MOSI 样本很少。

### 13. 主要机器可读来源

- 全量目录：`outputs/experiment_catalog_20260923/`
- MOSEI/MOSI interaction evidence：`project/reports/interaction_evidence_v1/*_interaction_evidence_valid.json`
- Fidelity–task gain：`project/reports/interaction_fidelity_task_gain_valid.{json,md}`、`project/reports/paper_figures/fig_interaction_fidelity_task_gain_valid.{pdf,png}` 及对应 caption
- Teacher statistics：`project/reports/interaction_evidence_v1/*_teacher_structure_valid.json`
- MOSEI 新增对照：`outputs/experiments/interaction_evidence_v1/mosei/`
- MOSI 新增对照：`outputs/experiments/interaction_evidence_v1/mosi/`
- MOSI 九方法：`outputs/experiments/uniform_main_v1/mosi/`
- MOSEI 后续对照：`outputs/experiments/uniform_followup_v1/mosei/`
- MOSEI P0：`outputs/experiments/tav_main_v1/`
- Stage A–D 统计：`project/reports/stage_a_v2_results.json`、`stage_b_baselines_three_seed.json`、`stage_c_multiseed_results.json`、`stage_d_final_significance.json`
- RDID-v2：`outputs/experiments/rdid_v2_mosei/`
