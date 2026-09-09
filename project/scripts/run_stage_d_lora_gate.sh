#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/wy/sjq/miniconda3/envs/kd/bin/python
SCRIPT=project/scripts/train_student_lora.py

run_d1() {
  local gpu="$1"
  local seed="$2"
  local output="outputs/student/stage_d_d1_full_kd_lora_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_DISABLE_PROGRESS_BARS=1 "$PYTHON" "$SCRIPT" \
    --output "$output" \
    --device cuda:0 \
    --method full_kd \
    --seed "$seed" \
    --batch-size 8 \
    --gradient-accumulation 1 \
    --epochs 30 \
    --patience 7 \
    --num-workers 2
}

run_d2() {
  local gpu="$1"
  local seed="$2"
  local output="outputs/student/stage_d_d2_ru_lora_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_DISABLE_PROGRESS_BARS=1 "$PYTHON" "$SCRIPT" \
    --output "$output" \
    --device cuda:0 \
    --method reliability_utility_pair \
    --teacher-targets outputs/probe/official_train_valid_seed2026/predictions.jsonl \
    --teacher-targets-ensemble \
      outputs/probe/official_train_valid_seed2026/predictions.jsonl \
      outputs/probe/official_train_valid_seed2027/predictions.jsonl \
      outputs/probe/official_train_valid_seed2028/predictions.jsonl \
    --seed "$seed" \
    --batch-size 8 \
    --gradient-accumulation 1 \
    --epochs 30 \
    --patience 7 \
    --num-workers 2
}

case "${1:-}" in
  d1-seed13) run_d1 0 13 ;;
  d1-seed42) run_d1 1 42 ;;
  d2-seed13) run_d2 1 13 ;;
  d2-seed42) run_d2 1 42 ;;
  *) echo "usage: $0 {d1-seed13|d1-seed42|d2-seed13|d2-seed42}" >&2; exit 2 ;;
esac
