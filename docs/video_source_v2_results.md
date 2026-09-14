**Video source v2 — primary T/A/V LoRA results; controls deferred by user**

All student modes use the same corrected TAV Probe seed2026 targets and its recorded temperature.
TA-only is a diagnostic control; the final model requirement remains TAV. Official test was not evaluated.

| Mode | Seed | MAE | Pearson | Acc-2 |
|---|---:|---:|---:|---:|
| video_lora | 13 | 0.471261 | 0.786222 | 0.872740 |
| video_lora | 42 | 0.466293 | 0.785247 | 0.885257 |
| video_lora | 2026 | 0.463737 | 0.798715 | 0.882476 |

Three-seed MAE: **0.467097 ± 0.003826** (mean ± sample SD).

Mean MAE increase over five cross-source video shuffles:

| Seed | Mean delta MAE | Range |
|---:|---:|---:|
| 13 | +0.003674 | [+0.002017, +0.005497] |
| 42 | +0.003978 | [+0.000939, +0.005860] |
| 2026 | +0.004670 | [+0.003592, +0.006099] |

The consistent positive shuffle averages show video sensitivity, not net benefit from Video LoRA.
The same-protocol `frozen_video` and `ta_only` controls remain deferred, so no between-mode causal comparison is reported.

Machine-readable results: /home/wy/sjq/kd/outputs/experiments/video_source_v2/summary.json
Teacher paired diagnostic: /home/wy/sjq/kd/outputs/experiments/video_source_v2/teacher_pair/summary.json
Wall-clock efficiency is not an exclusive-device benchmark; two student jobs may decode media concurrently.
