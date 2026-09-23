# Selected Interaction Case Studies

## Protocol

- Dataset/split: CMU-MOSEI official validation, 1,871 utterances.
- Run: seed 13; the four currently available methods use their reported
  test-MAE-selected checkpoints.
- Coordinates: anchored Möbius order `[T, A, V, TA, TV, AV, TAV]`.
- Final qualitative cases are selected without using any student prediction or
  reconstruction error.
- Conflicting-first-order-modality-effects criterion: among samples whose teacher first-order
  coordinates `[T, A, V]` contain both signs, define opposition strength as
  `min(max positive magnitude, max negative magnitude)`. `130456[11]` has score
  0.982 and ranks 6/696 (top 0.86%) among mixed-sign validation samples.
- Strong text-audio criterion: rank all validation samples by absolute teacher
  `TA` interaction. `125676[7]` has `|I_TA| = 1.892` and ranks 66/1,871
  (top 3.53%), satisfying the top-5% threshold.

These teacher-only rules make case eligibility independent of student outcomes
and avoid selecting examples solely because they make the proposed method look
favorable. Student results are displayed only after the cases are fixed. The
final aggregate analysis should add First+Second and Random Orthogonal after
their checkpoints become available.

## Recommended main-text cases

### 1. `125676[7]`: strong TA interaction and nuanced negative wording

> I mean it had a plot but it was sort of like what's the point

Target sentiment: **-1.333**. The teacher has strong negative unimodal effects
but a large positive TA correction:

| Profile | T | A | V | TA | TV | AV | TAV | Interaction MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher | -1.745 | -1.933 | -0.681 | 1.892 | 0.683 | 0.754 | -0.770 | — |
| Full KD | -1.597 | -0.638 | -0.462 | 0.654 | 0.454 | 0.427 | -0.419 | 0.544 |
| Subset-7 | -1.159 | -0.823 | -0.554 | 0.769 | 0.445 | 0.335 | -0.398 | 0.568 |
| First-order | -0.964 | -0.800 | -0.769 | 0.589 | 0.433 | 0.753 | -0.573 | 0.536 |
| Uniform | -1.425 | -1.152 | -0.816 | 1.370 | 0.566 | 0.714 | -0.753 | **0.276** |

| Method | Prediction | Absolute task error ↓ |
|---|---:|---:|
| Full KD | -1.406 | 0.073 |
| Subset-7 | -1.211 | 0.122 |
| First-order | -1.156 | 0.177 |
| Uniform | **-1.320** | **0.013** |

Why it works: the utterance is not a simple negative keyword example. The
speaker concedes that the movie has a plot, then dismisses its purpose. Uniform
recovers the teacher's large positive TA correction and the signs of every
higher-order coordinate while retaining the correct negative prediction.

### 2. `130456[11]`: conflicting first-order modality effects

> I really like how it's done because, if you watch this movie five times you
> will still not understand everything about it

Target sentiment: **0.667**. Teacher unimodal responses conflict: T and A are
positive, whereas V is negative. Higher-order corrections include negative TA
and TAV but positive TV and AV.

| Profile | T | A | V | TA | TV | AV | TAV | Interaction MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher | 1.371 | 1.324 | -0.982 | -1.126 | 0.781 | 1.024 | -0.931 | — |
| Full KD | 1.747 | -0.317 | -0.278 | 0.285 | 0.309 | 0.335 | -0.335 | 0.841 |
| Subset-7 | 1.403 | -0.071 | -0.663 | -0.062 | 0.609 | 0.250 | -0.164 | 0.646 |
| First-order | 1.794 | -0.091 | -0.448 | 0.029 | 0.339 | 0.339 | -0.245 | 0.763 |
| Uniform | 0.919 | -0.390 | -0.796 | -0.411 | **0.773** | 0.548 | -0.185 | **0.614** |

| Method | Prediction | Absolute task error ↓ |
|---|---:|---:|
| Full KD | 1.922 | 1.255 |
| Subset-7 | 1.477 | 0.810 |
| First-order | 1.891 | 1.224 |
| Uniform | **0.633** | **0.034** |

Why it works: the sentence mixes explicit praise with a criticism about the
movie being difficult to understand. Uniform nearly exactly reconstructs the
teacher TV coordinate (0.773 versus 0.781) and avoids the strong positive
over-prediction made by the other methods.

### 3. `4dV3slGSc60[8]`: third-order sign recovery

> It seemed to encompass everything about Business Administration I needed and
> professor you know was right on the ball.

Target sentiment: **1.667**. The teacher TAV coordinate is positive, whereas
Full KD reconstructs it with the wrong sign.

| Profile | TA | TV | AV | TAV | Interaction MAE ↓ |
|---|---:|---:|---:|---:|---:|
| Teacher | -1.529 | -0.596 | -0.735 | 0.768 | — |
| Full KD | 0.335 | -0.227 | 0.152 | -0.160 | 0.978 |
| Subset-7 | -0.080 | -0.114 | -0.132 | 0.194 | 0.791 |
| First-order | 0.069 | -0.089 | 0.018 | 0.053 | 0.870 |
| Uniform | **-0.946** | **-0.223** | **-0.123** | **0.787** | **0.576** |

Full KD and Uniform both predict **1.656** (absolute task error 0.010), but only
Uniform recovers the teacher's third-order sign and magnitude. This is therefore
a mechanism case rather than a performance-win case: identical final accuracy
can conceal substantially different internal interaction structure.

## Recommended appendix failure case

### 4. `70280[8]`: comparative-language failure

> ...in general I think that is a much better movie than the movie I have just
> seen in theaters

Target sentiment: **0.667**. All four students predict the wrong polarity:

| Method | Prediction | Absolute task error ↓ | Interaction MAE ↓ |
|---|---:|---:|---:|
| Full KD | -0.781 | 1.448 | 1.015 |
| Subset-7 | -0.463 | **1.130** | 1.083 |
| First-order | -1.344 | 2.010 | 0.854 |
| Uniform | -0.766 | 1.432 | **0.793** |

The teacher profile itself is dominated by negative unimodal coordinates and a
large negative TAV term (`-1.566`), even though the gold label is mildly
positive. Uniform improves interaction reconstruction but cannot resolve the
scope of the comparison: praise is directed at *Rear Window*, while the current
movie is criticized. This case is useful for stating the method's limitation:
matching teacher interactions cannot correct a teacher structure that is
misaligned with the annotation's semantic scope.

## Recommendation

Use cases 1 and 2 in the main paper. Use case 3 if space permits as direct
third-order mechanism evidence. Put case 4 in the appendix as a limitation.
