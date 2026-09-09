#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/wy/sjq/miniconda3/envs/kd/bin/python
SCRIPT=project/scripts/train_student_hidden_kd.py

for seed in 13 42; do
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SCRIPT" \
    --output "outputs/student/stage_d_d3_full_kd_hidden_seed${seed}" \
    --device cuda:0 \
    --seed "$seed" \
    --batch-size 8 \
    --epochs 30 \
    --patience 7 \
    --num-workers 2
done
