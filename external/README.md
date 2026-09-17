# External official implementations

`original/` contains shallow clones of the official repositories required by the
experiment design or its domain-related adaptation audit. The directory is intentionally ignored by the parent repository;
`SOURCES.json` is the versioned source/commit record.

The checked-out DLF, GsiT, and DPDF-LQ trees contain local protocol adaptations. Their
architectures and losses are unchanged. The adaptations only:

- remove all test-set evaluation from training epochs and hyperparameter tuning;
- select checkpoints by validation MAE;
- separate final-test access behind an explicit flag;
- redirect checkpoints, logs, reports, and per-sample predictions into the RDID-MSA run directory.

The exact diffs are versioned under `patches/`; their SHA-256 values match the corresponding
`local_protocol_diff_sha256` entries in `SOURCES.json`. After a fresh clone at the pinned
commit, replay them with:

```bash
/home/wy/sjq/miniconda3/envs/kd/bin/git -C external/original/DLF apply ../../patches/DLF-valid-only.patch
/home/wy/sjq/miniconda3/envs/kd/bin/git -C external/original/GsiT apply ../../patches/GsiT-valid-only.patch
/home/wy/sjq/miniconda3/envs/kd/bin/git -C external/original/DPDF-LQ apply ../../patches/DPDF-LQ-valid-only.patch
```

Modified upstream files:

- `DLF/run.py`, `DLF/trains/singleTask/DLF.py`;
- `GsiT/src/MMSA-GsiT/run.py`, `GsiT/src/MMSA-GsiT/trains/custom/GSIT.py`;
- `DPDF-LQ/train.py`.

The fixed-student KD methods are clean adaptations in
`project/src/rdid_mosei/main_table_kd.py`; upstream code is used as the frozen formula
reference and is not imported into the RDID student trainer. Repositories without a
license file are therefore not copied into versioned project code.

To verify the pinned checkout and local adaptations:

```bash
/home/wy/sjq/miniconda3/envs/kd/bin/git -C external/original/EA-KD rev-parse HEAD
/home/wy/sjq/miniconda3/envs/kd/bin/git -C external/original/DLF diff --check
/home/wy/sjq/miniconda3/envs/kd/bin/git -C external/original/GsiT diff --check
/home/wy/sjq/miniconda3/envs/kd/bin/git -C external/original/DPDF-LQ diff --check
```
