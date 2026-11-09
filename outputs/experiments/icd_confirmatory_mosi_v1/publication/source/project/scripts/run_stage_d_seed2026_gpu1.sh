#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/wy/sjq/miniconda3/envs/kd/bin/python
SCRIPT=project/scripts/train_student_lora.py
COMMON=(--device cuda:0 --seed 2026 --batch-size 8 --gradient-accumulation 1 --epochs 30 --patience 7 --num-workers 2)

case "${1:-}" in
  d1)
    CUDA_VISIBLE_DEVICES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 "$PYTHON" "$SCRIPT" \
      --output outputs/student/stage_d_d1_full_kd_lora_seed2026 \
      --method full_kd "${COMMON[@]}"
    ;;
  d2)
    CUDA_VISIBLE_DEVICES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 "$PYTHON" "$SCRIPT" \
      --output outputs/student/stage_d_d2_ru_lora_seed2026 \
      --method reliability_utility_pair \
      --teacher-targets outputs/probe/official_train_valid_seed2026/predictions.jsonl \
      --teacher-targets-ensemble \
        outputs/probe/official_train_valid_seed2026/predictions.jsonl \
        outputs/probe/official_train_valid_seed2027/predictions.jsonl \
        outputs/probe/official_train_valid_seed2028/predictions.jsonl \
      "${COMMON[@]}"
    ;;
  *) echo "usage: $0 {d1|d2}" >&2; exit 2 ;;
esac
