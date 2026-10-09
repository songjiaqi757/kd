# Equal-modality ICD method extensions

This experiment family tests method changes while holding the three modality
paths fixed. Every row uses the same text, audio, and video models, preprocessing,
LoRA scope and rank, pooling dimensions, optimizer, batch size, epoch budget,
teacher assets, and seed set. New rules are shared across modality identities;
no row receives a hand-tuned text-, audio-, or video-specific weight.

## Rows

- `uniform_interaction`: unchanged ICD reference.
- `uniform_interaction_shared_dropout`: reuses one fusion-dropout realization
  for all seven subset evaluations of an example.
- `uniform_interaction_gradient_projected`: projects the interaction auxiliary
  gradient only when it conflicts with the common task plus full-KD gradient.
- `uniform_interaction_coordinate_compatible`: weights the seven coordinates by
  their gradient cosine with the common task plus full-KD objective, measured on
  the shared fusion module. Weights are detached and normalized to mean one.
- `explicit_full_kd`: explicit shared unary/pair/triple interaction fusion with
  ordinary Full KD. This is the architecture control.
- `explicit_uniform_interaction`: the same explicit fusion trained with ICD.

The runner retains both `best_valid_mae.pt` and
`best_valid_acc2_nonzero.pt`, with `best.pt` pointing to the MAE checkpoint.
Each policy reports MAE, Acc-2 and weighted F1 from its own single checkpoint.
Ties retain the earlier checkpoint (the trainer requires improvement > 1e-5).
The legacy runner below automatically evaluates official test after each run;
use the validation-only [decision ICD screening](decision_icd_v1.md) runner for
new method development. Do not use these legacy test results to choose a new
candidate or hyperparameters.

## Run

```bash
cd /home/wy/sjq/kd
project/scripts/run_equal_modality_method_extensions.sh \
  0 \
  outputs/experiments/main_table_v1/mosei/assets/protocol.json \
  dataset/cmu_mosei/manifests/official_test_windowed.jsonl \
  outputs/experiments/equal_modality_extensions_v1/mosei \
  all
```

Use a different free CUDA device or run a single seed by replacing `all` with
`13`, `42`, or `2026`. The runner refuses to overwrite an existing run.

After all 18 runs complete:

```bash
/home/wy/sjq/miniconda3/envs/kd/bin/python \
  project/scripts/summarize_equal_modality_method_extensions.py \
  --input-root outputs/experiments/equal_modality_extensions_v1/mosei \
  --output project/reports/equal_modality_extensions_v1_mosei.json
```

The summary includes every seed, mean and sample standard deviation, and paired
candidate-minus-reference deltas. Statistical significance should still use the
existing source-video clustered paired bootstrap before a method claim is made.
