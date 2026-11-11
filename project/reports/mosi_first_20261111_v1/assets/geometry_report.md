# MOSI offline geometry report



Train utterances: 1284; rank: 7; condition number: 11.91607978.

Subset order: T, A, V, TA, TV, AV, TAV. Common anchor equals the mean of the three train-TAV probe means.

Three-probe raw scores and temperature-softmax probabilities were reconstructed and verified against the cached ensemble.

Normalization uses unique training utterances only; sample standard deviation; floor 0.05.

Exact matrices, covariance, correlations, quantiles and reliability statistics: geometry.json and matched_matrices.json.

Original Probe weights are missing; checkpoint-dependent work is skipped. No missing weights are downloaded or retrained.

Fixed-batch base/auxiliary and per-coordinate gradient diagnostics are recorded before every training epoch.

Historical MOSI test comparisons and this mechanism screen are exploratory. All methods and representative matrices are locked before new test evaluation.
