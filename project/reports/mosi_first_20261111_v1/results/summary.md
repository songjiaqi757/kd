# MOSI First — seed 13



Status: complete. Jobs: {'pending': 0, 'running': 0, 'complete': 72, 'failed': 0, 'skipped': 0}.

Frozen protocol: up to 30 epochs, validation-MAE early stopping patience 7, batch size 8, LR 1e-4, weight decay 0.01.

Only seed 13. Three fixed teacher prediction caches. MAE and Acc2 best are selected using validation only; last is retained for resume.

Official-test results are exploratory reporting and do not select methods, matrices, hyperparameters or checkpoints. No cross-seed conclusions are available.

| Method | Seed | Checkpoint policy | Epoch | MAE | Acc2 nonzero | F1 nonzero | Pearson |
|---|---:|---|---:|---:|---:|---:|---:|
| A_full_no_ce | 13 | last | 15 | 0.728041 | 0.836890 | 0.837447 | 0.783365 |
| A_full_no_ce | 13 | valid_acc2_nonzero | 5 | 0.752225 | 0.847561 | 0.847951 | 0.789825 |
| A_full_no_ce | 13 | valid_mae | 7 | 0.725958 | 0.823171 | 0.823486 | 0.778319 |
| B_N2_icd030 | 13 | last | 18 | 0.697316 | 0.849085 | 0.848352 | 0.803846 |
| B_N2_icd030 | 13 | valid_acc2_nonzero | 7 | 0.749947 | 0.846037 | 0.845669 | 0.795368 |
| B_N2_icd030 | 13 | valid_mae | 10 | 0.729619 | 0.842988 | 0.843874 | 0.790386 |
| C_full_ce | 13 | last | 23 | 0.723193 | 0.824695 | 0.823605 | 0.785458 |
| C_full_ce | 13 | valid_acc2_nonzero | 19 | 0.716762 | 0.838415 | 0.838703 | 0.793811 |
| C_full_ce | 13 | valid_mae | 16 | 0.710000 | 0.830793 | 0.829857 | 0.789607 |
| D_icd030_ce | 13 | last | 22 | 0.750388 | 0.847561 | 0.847152 | 0.792331 |
| D_icd030_ce | 13 | valid_acc2_nonzero | 21 | 0.725623 | 0.836890 | 0.835517 | 0.790126 |
| D_icd030_ce | 13 | valid_mae | 15 | 0.698488 | 0.846037 | 0.845388 | 0.800212 |
| E4_icd050_ce | 13 | last | 20 | 0.708819 | 0.847561 | 0.848270 | 0.798666 |
| E4_icd050_ce | 13 | valid_acc2_nonzero | 12 | 0.718975 | 0.847561 | 0.848472 | 0.801268 |
| E4_icd050_ce | 13 | valid_mae | 13 | 0.727433 | 0.853659 | 0.854533 | 0.801343 |
| F7_icd025 | 13 | last | 13 | 0.846339 | 0.839939 | 0.840873 | 0.797860 |
| F7_icd025 | 13 | valid_acc2_nonzero | 6 | 0.754231 | 0.836890 | 0.837078 | 0.782969 |
| F7_icd025 | 13 | valid_mae | 6 | 0.754231 | 0.836890 | 0.837078 | 0.782969 |
| diagonal_mse | 13 | last | 20 | 0.712007 | 0.846037 | 0.846771 | 0.814705 |
| diagonal_mse | 13 | valid_acc2_nonzero | 10 | 0.688624 | 0.850610 | 0.849177 | 0.812066 |
| diagonal_mse | 13 | valid_mae | 13 | 0.693131 | 0.841463 | 0.841746 | 0.816907 |
| icd_mse | 13 | last | 21 | 0.724013 | 0.861280 | 0.861378 | 0.801280 |
| icd_mse | 13 | valid_acc2_nonzero | 3 | 0.769655 | 0.842988 | 0.842225 | 0.784049 |
| icd_mse | 13 | valid_mae | 14 | 0.703819 | 0.856707 | 0.856636 | 0.799056 |
| matched0_huber | 13 | last | 18 | 0.737725 | 0.846037 | 0.846458 | 0.790697 |
| matched0_huber | 13 | valid_acc2_nonzero | 14 | 0.689698 | 0.855183 | 0.853496 | 0.812418 |
| matched0_huber | 13 | valid_mae | 11 | 0.723566 | 0.850610 | 0.851045 | 0.795126 |
| matched0_mse | 13 | last | 17 | 0.763126 | 0.856707 | 0.855765 | 0.791727 |
| matched0_mse | 13 | valid_acc2_nonzero | 5 | 0.770967 | 0.850610 | 0.849409 | 0.802955 |
| matched0_mse | 13 | valid_mae | 10 | 0.732516 | 0.864329 | 0.864599 | 0.805034 |
| matched1_huber | 13 | last | 16 | 0.777801 | 0.847561 | 0.848431 | 0.798933 |
| matched1_huber | 13 | valid_acc2_nonzero | 3 | 0.750379 | 0.830793 | 0.829620 | 0.779515 |
| matched1_huber | 13 | valid_mae | 9 | 0.720339 | 0.846037 | 0.846145 | 0.789489 |
| matched2_huber | 13 | last | 16 | 0.702108 | 0.852134 | 0.851511 | 0.808730 |
| matched2_huber | 13 | valid_acc2_nonzero | 11 | 0.702876 | 0.849085 | 0.847577 | 0.799680 |
| matched2_huber | 13 | valid_mae | 9 | 0.719352 | 0.847561 | 0.846870 | 0.790950 |
| normalized_icd | 13 | last | 24 | 0.685065 | 0.859756 | 0.859756 | 0.807878 |
| normalized_icd | 13 | valid_acc2_nonzero | 12 | 0.709888 | 0.858232 | 0.858266 | 0.805310 |
| normalized_icd | 13 | valid_mae | 17 | 0.717783 | 0.856707 | 0.857262 | 0.799462 |
| normalized_subset | 13 | last | 10 | 0.723496 | 0.820122 | 0.817124 | 0.785290 |
| normalized_subset | 13 | valid_acc2_nonzero | 4 | 0.765423 | 0.842988 | 0.842011 | 0.791942 |
| normalized_subset | 13 | valid_mae | 3 | 0.819896 | 0.824695 | 0.825723 | 0.765609 |
| orthogonal_huber | 13 | last | 17 | 0.728382 | 0.844512 | 0.843599 | 0.794955 |
| orthogonal_huber | 13 | valid_acc2_nonzero | 11 | 0.744905 | 0.849085 | 0.849691 | 0.801029 |
| orthogonal_huber | 13 | valid_mae | 10 | 0.755600 | 0.844512 | 0.845198 | 0.793328 |
| subset7 | 13 | last | 26 | 0.721463 | 0.847561 | 0.846666 | 0.802731 |
| subset7 | 13 | valid_acc2_nonzero | 8 | 0.723196 | 0.852134 | 0.851415 | 0.802825 |
| subset7 | 13 | valid_mae | 19 | 0.712829 | 0.855183 | 0.855803 | 0.807634 |
| weighted_huber | 13 | last | 22 | 0.714505 | 0.842988 | 0.842520 | 0.794671 |
| weighted_huber | 13 | valid_acc2_nonzero | 22 | 0.714505 | 0.842988 | 0.842520 | 0.794671 |
| weighted_huber | 13 | valid_mae | 15 | 0.753622 | 0.847561 | 0.848270 | 0.793114 |
| weighted_mse | 13 | last | 27 | 0.773399 | 0.841463 | 0.842411 | 0.796357 |
| weighted_mse | 13 | valid_acc2_nonzero | 8 | 0.725319 | 0.844512 | 0.844512 | 0.792570 |
| weighted_mse | 13 | valid_mae | 20 | 0.715401 | 0.850610 | 0.851443 | 0.797490 |

Missing original Probe checkpoints are documented in teacher_manifest.json. Cached predictions have been verified against all three prediction files.
