**Video adaptation results**

Official valid only. All numbers are conditional on validation-selected checkpoints.

| Mode | Seed | MAE | Pearson | Acc-2 |
|---|---:|---:|---:|---:|
| frozen_video | 13 | 0.471689 | 0.796527 | 0.876912 |
| video_lora | 13 | 0.470305 | 0.792225 | 0.877608 |
| ta_only | 13 | 0.468983 | 0.788980 | 0.885953 |

| Comparison | ΔMAE | Video-cluster 95% CI |
|---|---:|---|
| video_lora_minus_frozen_video_seed13 | -0.001384 | [-0.010452, +0.007687] |
| video_lora_minus_ta_only_seed13 | +0.001321 | [-0.006429, +0.009227] |

TA-only uses TA teacher targets. A/B use identical TAV teacher targets.
Single-seed promotion is an exploratory gate, not a significance claim.
Machine-readable results: /home/wy/sjq/kd/outputs/student/video_adaptation_v1/summary.json
