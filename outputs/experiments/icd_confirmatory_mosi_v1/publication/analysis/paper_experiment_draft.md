# ICLR 实验文字草稿

We conducted a validation-only 2 × 4 factorial study on CMU-MOSI at seed 13, varying the ground-truth classification weight c ∈ {0, 0.5} and interaction distillation weight λ ∈ {0, 0.25, 0.5, 1}. All runs share three teacher probes, the student initialization, video-disjoint train/validation split, and validation-MAE checkpoint selection. Removing task CE retains teacher KL supervision.
The raw minimum validation MAE was 0.7366 (GP_icd_ce). This single-seed development result does not establish cross-dataset or multi-seed robustness. Paired source-video bootstrap analysis describes uncertainty across the ten validation videos.

The main and ablation tables are available as per_run.csv. All auxiliary metrics refer to each run's same MAE-selected checkpoint. Gradient geometry and interaction reconstruction errors provide descriptive mechanism evidence and are not treated as causal proof.
