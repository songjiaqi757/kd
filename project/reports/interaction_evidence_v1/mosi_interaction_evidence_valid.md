# MOSI interaction evidence (seed13, official valid)

## Interaction reconstruction error

| Method | I_T | I_A | I_V | I_TA | I_TV | I_AV | I_TAV | Avg. ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full KD | 0.541540 | 1.595234 | 1.037797 | 1.579069 | 1.038385 | 1.092043 | 1.122787 | 1.143837 |
| Subset-7 KD | 0.570370 | 1.142789 | 0.291334 | 1.145404 | 0.383912 | 0.330292 | 0.431790 | 0.613699 |
| First-order Interaction | 0.561546 | 1.112398 | 0.277389 | 1.044928 | 0.383412 | 0.350150 | 0.458235 | 0.598294 |
| Uniform Interaction | 0.542659 | 1.046699 | 0.291568 | 0.694448 | 0.378472 | 0.307495 | 0.507741 | 0.538440 |
| First + Second-order Interaction | 0.554788 | 1.123156 | 0.274881 | 0.609354 | 0.365934 | 0.302619 | 0.846797 | 0.582504 |
| Random Orthogonal | 0.544379 | 1.058037 | 0.301881 | 1.027417 | 0.383736 | 0.323158 | 0.439460 | 0.582581 |

## Error by interaction order

| Method | First-order ↓ | Second-order ↓ | Third-order ↓ |
|---|---:|---:|---:|
| Full KD | 1.058190 | 1.236499 | 1.122787 |
| Subset-7 KD | 0.668164 | 0.619870 | 0.431790 |
| First-order Interaction | 0.650444 | 0.592830 | 0.458235 |
| Uniform Interaction | 0.626975 | 0.460138 | 0.507741 |
| First + Second-order Interaction | 0.650942 | 0.425969 | 0.846797 |
| Random Orthogonal | 0.634766 | 0.578104 | 0.439460 |

## Interaction-strength stratified MAE

| Method | Low ↓ | Medium ↓ | High ↓ |
|---|---:|---:|---:|
| Full KD | 0.770924 | 0.666431 | 0.756111 |
| Subset-7 KD | 0.749691 | 0.776632 | 0.832133 |
| First-order Interaction | 0.777899 | 0.781438 | 0.790453 |
| Uniform Interaction | 0.742380 | 0.771083 | 0.774399 |
| First + Second-order Interaction | 0.761523 | 0.739492 | 0.791540 |
| Random Orthogonal | 0.793738 | 0.686668 | 0.788831 |
