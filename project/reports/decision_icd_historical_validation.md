# Historical validation error audit

All strata use fixed |GT| <= 0.5; neutral is separate.

| Run | MAE epoch | Acc2 epoch | Cached epoch | Near-zero wrong | Away wrong |
|---|---:|---:|---:|---:|---:|
| full_kd_seed13 | 19 | 16 | 19 | 12 | 20 |
| uniform_interaction_seed13 | 15 | 13 | 15 | 10 | 22 |
| full_kd_seed2026 | 19 | 15 | 19 | 14 | 21 |
| uniform_interaction_seed2026 | 5 | 12 | 5 | 15 | 21 |
| M3_seed13 | 6 | 5 | 6 | 95 | 88 |
| M4_seed13 | 6 | 4 | 6 | 97 | 79 |

Error counts refer to the cached prediction epoch only. Missing selected-epoch predictions are not inferred; the new trainer saves them for both policies.
