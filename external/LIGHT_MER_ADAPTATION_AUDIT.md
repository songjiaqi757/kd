# Light-MER / SWD-H adaptation audit

Audit date: 2026-09-17. Frozen source and commit are recorded in `SOURCES.json`.

## Finding

The released Stage-1 method is not a drop-in fixed-student baseline for the current
continuous-regression table. It distils a generative Qwen3-8B multimodal teacher into a
Qwen3-0.6B generative student. Its main SWD-H signal aligns projected final-layer hidden
states at answer-token positions with sliced Wasserstein distance. The released template
uses 200 projections, `p=1`, an answer-token mask, and a 5,000-step weight ramp.

The training template consumes MER-Caption+ and produces free-form answer tokens. MOSI and
MOSEI are present in the inference/evaluation pipeline, but this is not the same training
objective, output space, checkpoint protocol, or MAE regression head used by RDID-MSA.
Running its reported system also requires the released teacher/student checkpoints,
Qwen3-8B, CLIP-Large/Base, Chinese-HuBERT-Large/Base, and the repository's processed media.
Those assets are not present in this workspace.

## Decision

- Do not insert the complete Light-MER result into the fixed-student MAE table.
- Do not call pooled fused-feature Wasserstein matching a complete Light-MER reproduction:
  it removes the answer-token geometry that defines the released SWD-H mechanism.
- If a later experiment ports only SWD-H, label it `SWD-H component adaptation`, freeze the
  token/feature correspondence and projection budget, and give it the same validation-only
  selection and three-seed budget as other adapted KD methods.
- Keep full Light-MER as related work unless a separate generative-system comparison with
  compatible datasets, outputs, metrics, and checkpoints is preregistered.

CorrKD was also rechecked for this revision. An official paper is available, but no author
repository was identified in the paper page, authors' public project listings, or targeted
GitHub search. It therefore remains a code-availability audit item rather than a runnable
experiment; this statement is time-scoped to the audit date.
