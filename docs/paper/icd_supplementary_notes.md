# ICD 补充说明

> 汇总 qualitative selection、training-time cost、Teacher Probe 与 ensemble control 的论文就绪口径。

## 原文快照：`补充说明_定性选择_训练成本_Probe_20260925.md`

## Qualitative selection、training-time cost 与 Teacher Probe 补充说明

本文档整理可直接写入论文/回复审稿人的精确口径。所有数字来自冻结资产、运行日志和
训练 report；没有用估算值替代缺失实验。

### 1. Qualitative example selection

案例只使用 CMU-MOSEI official-validation 的 teacher interaction coordinates 选择，
不读取任何 student prediction、task error 或 reconstruction error。

- **First-order conflict.** 在 $I_T,I_A,I_V$ 同时含正、负坐标的 696 条 utterance 中，定义
  $$s_{\mathrm{opp}}=\min\{\max_{m\in\{T,A,V\}}[I_m]_+,\;
  \max_{m\in\{T,A,V\}}[-I_m]_+\}.$$
  按 $s_{\mathrm{opp}}$ 降序取高排名案例。`130456[11]` 的分数为 0.981857，排名
  6/696（top 0.862%）。
- **Text--audio interaction.** 在全部 1,871 条 validation utterance 中按
  $|I_{TA}|$ 降序排列，并从预先指定的 top-5% 区间中展示案例。`125676[7]` 的
  $|I_{TA}|=1.891960$，排名 66/1,871（top 3.528%）。

Paper-ready English:

> We selected qualitative examples using teacher-side interaction statistics only, before
> inspecting any student prediction or error. For first-order conflict, we retained validation
> utterances with mixed-sign $I_T,I_A,I_V$ and ranked them by
> $s_{\rm opp}=\min(\max_m[I_m]_+,\max_m[-I_m]_+)$. The displayed case ranks 6th among
> 696 eligible utterances (top 0.86%). For text--audio interaction, we ranked all 1,871
> validation utterances by $|I_{TA}|$ and selected a case from the pre-specified top-5% region;
> the displayed case has $|I_{TA}|=1.892$ and ranks 66th (top 3.53%).

### 2. Training-time cost and offline reuse

#### 2.1 MOSI measured profile

| Stage | Hardware/accounting | Wall-clock | GPU-hours |
|---|---|---:|---:|
| Teacher seven-subset feature extraction | two-GPU model parallel | 5,273.346 s (87.889 min) | 2.9296 |
| Three Probe training/calibration | three concurrent processes on two GPUs | 45.635 s | about 0.0254 |
| Cached Probe outputs → interactions/ensemble/student assets | CPU | 7.786 s | 0 |
| Full target-preparation pipeline | extraction start → frozen student assets | 5,356.963 s (89.283 min) | about 2.955 |
| Full KD student training, seed13 | one GPU | 2,906.157 s | 0.8073 |
| ICD student training, seed13 | one GPU | 2,900.673 s | 0.8057 |

Thus the measured ICD/Full-KD student-training wall-clock multiplier is **0.9981×**
($-0.189\%$). The upstream target-preparation cost is offline and amortized across methods and
student seeds.

The teacher runs once over every train/validation window and each of the seven non-empty modality
subsets. It stores the last valid input-token Thinker state as a 2,048-dimensional FP16 NumPy
cache, with an index and a resumable completion mask. All three Probes read this same cache.
Their per-parent/subset regression outputs and calibrated class probabilities are persisted as
JSONL. A deterministic CPU step then computes anchored Möbius coordinates, the three-Probe mean
and variance, and the ensemble TAV probabilities. Student jobs join these frozen assets once and
never invoke the 30B teacher or recompute interaction targets online.

Paper-ready English:

> Teacher supervision is generated offline and reused. We run the frozen teacher once for each
> train/validation utterance and non-empty modality subset, cache the resulting 2,048-D FP16
> hidden states, and train all Probes from this shared cache. Probe predictions, calibrated
> probabilities, and Möbius interaction targets are then materialized as immutable JSONL/NumPy
> assets and reused by every student method and seed. On CMU-MOSI, the complete target pipeline
> took 89.28 min and approximately 2.96 GPU-hours; the cache-to-target CPU transformation took
> 7.79 s. Student training took 0.807 GPU-hours for Full KD and 0.806 GPU-hours for ICD, giving an
> ICD/Full-KD wall-clock multiplier of 0.998×. No teacher forward pass is performed during student
> training.

### 3. Teacher Probe specification

The frozen Qwen3-Omni Thinker supplies the last non-padding input-token hidden state. The Probe is

$$\mathrm{LN}(2048)\rightarrow\mathrm{Linear}(2048,512)\rightarrow\mathrm{GELU}
\rightarrow\mathrm{Dropout}(0.1),$$

followed by a seven-class linear head and a scalar regression head
$3\tanh(\mathrm{Linear}(512,1))$. It is trained only on the official training split. The official
validation split is used for checkpoint selection, early stopping, and post-hoc temperature
calibration; official test is excluded.

- Probe seeds: 2026, 2027, 2028.
- Optimizer: AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 128.
- Loss: weighted Smooth-L1 regression plus $0.5$ times seven-class cross-entropy. Subset sampling
  mass is 0.1 for each of T/A/V/TA/TV/AV and 0.4 for TAV; window weights sum to one within each
  parent/subset.
- Maximum 200 epochs; patience 25; an improvement must exceed $10^{-5}$.
- For each seed, $\phi_s^\star$ is the epoch with minimum official-validation joint loss. There is
  no best-of-three seed selection: seed2026 is fixed in advance for the single-Probe TAV target,
  while seeds 2026/2027/2028 are all used in the ensemble interaction target.
- $\tau_{\rm cal}$ is fitted after checkpoint selection by minimizing official-validation
  seven-class NLL with scalar temperature scaling (LBFGS, constrained to $[0.05,20]$).

| Dataset | seed2026 | seed2027 | seed2028 | best epochs |
|---|---:|---:|---:|---|
| MOSEI $\tau_{\rm cal}$ | 1.011453 | 1.066849 | 1.008141 | 6 / 5 / 8 |
| MOSI $\tau_{\rm cal}$ | 1.191218 | 1.205872 | 1.180255 | 12 / 10 / 12 |

Paper-ready English:

> Each Probe maps the frozen teacher's last valid input-token state through LayerNorm, a
> 2048-to-512 linear layer, GELU, and dropout (0.1), followed by separate seven-class and bounded
> scalar-regression heads. Probes are trained on the official training split only. For each of
> seeds 2026/2027/2028, $\phi_s^\star$ is selected by minimum joint loss on official validation,
> with patience 25; no Probe seed is selected by downstream student performance. After checkpoint
> selection, a scalar $\tau_{\rm cal}$ is fitted on validation NLL. The resulting temperatures are
> 1.011/1.067/1.008 on MOSEI and 1.191/1.206/1.180 on MOSI.

### 4. Ensemble asymmetry and controls

The supervision paths are not identical and should be stated explicitly:

| Method | Base TAV target | Additional target |
|---|---|---|
| Full KD | fixed seed2026 Probe | none |
| Ensemble Full KD | mean of three calibrated Probes | none |
| ICD | fixed seed2026 TAV KD | seven-subset interaction target from the same three-Probe ensemble |

The existing **Full KD versus Ensemble Full KD** comparison is the direct single-Probe versus
ensemble-Probe control for ordinary full-modality KD. On MOSI, their three-seed test MAEs are
0.731630 and 0.729987, whereas ICD is 0.738779; hence the MOSI results do not support an ICD task
gain attributable to a better ensemble target. On MOSEI seed13, Full KD / Ensemble Full KD / ICD
test MAEs are 0.481628 / 0.486895 / 0.477360: the ensemble-only control is worse than single-Probe
Full KD while ICD is better than both, which argues against a generic ensemble-only explanation
for that seed.

This evidence controls the ordinary full-modality ensemble explanation, but it is not a complete
$2\times2$ factorial isolation of the interaction target source. If a reviewer requires that
stronger claim, the missing cell is **ICD-single**: recompute the seven interaction coordinates
from seed2026 alone and train it under exactly the same student seeds and checkpoint policy as
ensemble ICD. Until that run is complete, the paper should say that Ensemble Full KD weakens the
ensemble-only alternative, not that all ensemble effects have been completely isolated.

### 5. Evidence paths

- Qualitative selection audit: `project/scripts/plot_qualitative_interaction_distillation.py`
- Teacher feature/probe plan: `outputs/experiments/main_table_v1/mosi/preparation_plan.json`
- Probe implementation: `project/src/rdid_mosei/probe.py` and
  `project/scripts/train_teacher_probe_v2.py`
- MOSI target-preparation logs: `outputs/experiments/main_table_v1/mosi/logs/`
- Student timing: `outputs/experiments/uniform_main_v1/mosi/summary/efficiency.json`
- Single/ensemble control results: `docs/paper/icd_experiment_summary.md`
