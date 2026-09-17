# CMAD / CAFD adaptation audit

Audit date: 2026-09-17. Frozen source and commit are recorded in `SOURCES.json`.

## Released method

The official release is a complete missing-modality system, not a standalone generic feature
loss. Its student randomly receives one of seven modality-presence combinations. The training
objective combines:

- task L1 regression and teacher-output regression distillation;
- CAFD's student/teacher hidden-representation MSE, cross-sample cosine-similarity matrices,
  bidirectional KL, and diagonal/off-diagonal correlation term;
- MAR weights estimated from the previous epoch's student-versus-teacher ground-truth error
  gap for each modality combination.

The public command covers MOSEI and uses a BERT/Perceiver teacher and student with the same
architecture. Although its task head is continuous regression, the published code selects and
prints sign-based accuracy/F1 and evaluates the test loader whenever validation F1 improves.
It therefore cannot be run unchanged under RDID-MSA's valid-MAE/one-shot-test protocol.

## Decision

- Do not put the paper's missing-modality result directly in the complete-TAV MAE main table.
- Porting only `compute_weighted_mse_loss` must be called `CAFD component adaptation`; it is
  not a reproduction of CMAD because it omits random modality dropping and MAR.
- A faithful CMAD run belongs to a separately preregistered missing-modality extension and
  must first remove per-epoch test access, select one checkpoint using validation only, audit
  sample IDs/splits, and report all seven missing-modality conditions.
- The minimum paper set includes a clean `CMAD-style CAFD component adaptation` in the
  fixed-student trainer. It retains the official weighted feature-MSE and cross-sample
  correlation construction, adds a necessary 512→2048 student projection, fixes `tau=0.2`,
  and tunes only its loss coefficient. It deliberately excludes modality dropping and MAR.
