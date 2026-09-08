# Stage C seed-13 screening

范围：official train/valid；official test 未使用。Gate 基线为 B1 Full KD。

| 方法 | MAE | ΔMAE | Pearson | ΔPearson | ΔE_pair | High-I ΔMAE | High-U ΔMAE | High-C ΔMAE | Bootstrap 95% CI | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C1_uniform_ensemble_pair | 0.507612 | -0.004769 | 0.748125 | +0.012347 | -0.525205 | -0.024123 | -0.012196 | -0.007467 | [-0.017344, +0.007421] | YES |
| C2_snr_pair | 0.509045 | -0.003336 | 0.747858 | +0.012080 | -0.509298 | -0.009672 | -0.013562 | -0.014932 | [-0.014891, +0.008027] | YES |
| C3_utility_pair | 0.519933 | +0.007552 | 0.748412 | +0.012635 | -0.487577 | -0.002504 | -0.001647 | -0.008448 | [-0.002254, +0.017197] | NO |
| C4_reliability_utility_pair | 0.506201 | -0.006180 | 0.743379 | +0.007602 | -0.519948 | -0.016556 | -0.006174 | -0.001957 | [-0.019265, +0.007164] | YES |
| selective50_interaction4 | 0.501967 | -0.010414 | 0.748657 | +0.012880 | -0.499892 | -0.021209 | -0.011899 | -0.017124 | [-0.020988, -0.000188] | YES |

## Gate details

- `C1_uniform_ensemble_pair`: G1_mae_at_least_0.003_better=True, G2_pearson_not_down_more_than_0.003=True, G3_not_fidelity_only=True, G4_high_subgroup_at_least_0.003_better=True
- `C2_snr_pair`: G1_mae_at_least_0.003_better=True, G2_pearson_not_down_more_than_0.003=True, G3_not_fidelity_only=True, G4_high_subgroup_at_least_0.003_better=True
- `C3_utility_pair`: G1_mae_at_least_0.003_better=False, G2_pearson_not_down_more_than_0.003=True, G3_not_fidelity_only=False, G4_high_subgroup_at_least_0.003_better=True
- `C4_reliability_utility_pair`: G1_mae_at_least_0.003_better=True, G2_pearson_not_down_more_than_0.003=True, G3_not_fidelity_only=True, G4_high_subgroup_at_least_0.003_better=True
- `selective50_interaction4`: G1_mae_at_least_0.003_better=True, G2_pearson_not_down_more_than_0.003=True, G3_not_fidelity_only=True, G4_high_subgroup_at_least_0.003_better=True

只有 Promote=YES 的方法可补 seed 42/2026；该规则不使用 official-test 信息。
