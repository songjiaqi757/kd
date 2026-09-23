# MOSEI interaction evidence (seed13, official valid)

## Interaction reconstruction error

| Method | I_T | I_A | I_V | I_TA | I_TV | I_AV | I_TAV | Avg. ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full KD | 0.279338 | 0.764366 | 0.301827 | 0.726456 | 0.262268 | 0.389324 | 0.416637 | 0.448602 |
| Subset-7 KD | 0.252851 | 0.431388 | 0.213736 | 0.413137 | 0.223670 | 0.215182 | 0.250033 | 0.285714 |
| First-order Interaction | 0.243411 | 0.427440 | 0.217134 | 0.429950 | 0.213756 | 0.196306 | 0.230882 | 0.279840 |
| Uniform Interaction | 0.246642 | 0.470811 | 0.203495 | 0.303493 | 0.218297 | 0.185594 | 0.239812 | 0.266878 |
| First + Second-order Interaction | 0.242868 | 0.432113 | 0.198916 | 0.267796 | 0.200775 | 0.179986 | 0.334293 | 0.265250 |
| Random Orthogonal | 0.253903 | 0.421446 | 0.206891 | 0.393409 | 0.217934 | 0.203136 | 0.246114 | 0.277548 |

## Error by interaction order

| Method | First-order ↓ | Second-order ↓ | Third-order ↓ |
|---|---:|---:|---:|
| Full KD | 0.448510 | 0.459349 | 0.416637 |
| Subset-7 KD | 0.299325 | 0.283996 | 0.250033 |
| First-order Interaction | 0.295995 | 0.280004 | 0.230882 |
| Uniform Interaction | 0.306983 | 0.235795 | 0.239812 |
| First + Second-order Interaction | 0.291299 | 0.216186 | 0.334293 |
| Random Orthogonal | 0.294080 | 0.271493 | 0.246114 |

## Interaction-strength stratified MAE

| Method | Low ↓ | Medium ↓ | High ↓ |
|---|---:|---:|---:|
| Full KD | 0.410029 | 0.455779 | 0.547951 |
| Subset-7 KD | 0.411105 | 0.450923 | 0.545061 |
| First-order Interaction | 0.417458 | 0.456505 | 0.536310 |
| Uniform Interaction | 0.412887 | 0.456623 | 0.554125 |
| First + Second-order Interaction | 0.403604 | 0.450587 | 0.543734 |
| Random Orthogonal | 0.402349 | 0.451619 | 0.552131 |
