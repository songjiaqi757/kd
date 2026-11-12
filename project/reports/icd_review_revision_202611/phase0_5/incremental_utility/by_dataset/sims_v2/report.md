# sims_v2: A3

Source-video grouped nested CV: outer 5 folds, inner 3 folds, alpha grid [0.01, 0.1, 1.0, 10.0, 100.0]. Standardization/intercept fit on training fold only. Train and valid never mixed. Empty input predicts training-fold mean. Response utility=-MSE; task utility=-MAE. All 127 nonempty combinations plus empty baseline use identical ridge budget. Scores report each outer fold; fold std is descriptive, not a population confidence interval.

TAV input trivially exposes the response target; exclusion-of-TAV six-player sensitivity is reported separately. Shapley describes analyzer predictive utility, not causal effects or Student training gains. Official Validation may share source videos with Train; valid_novel_source excludes these videos. No novel subset means that external independence check is unavailable. Shared pretrained teacher/probe fitting makes this diagnostic, not independent end-to-end generalization evidence. No adaptive weights are applied.

