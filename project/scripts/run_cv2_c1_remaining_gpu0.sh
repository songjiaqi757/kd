#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/wy/sjq/miniconda3/envs/kd/bin/python
SCRIPT=/home/wy/sjq/kd/project/scripts/train_student_lora.py
ROOT=/home/wy/sjq/kd
LOG_DIR="$ROOT/outputs/logs/cv2_c1_tempfix"
mkdir -p "$LOG_DIR"
cd "$ROOT"

for seed in 42 2026; do
  name="stage_d_cv2_c1_full_kd_lora_tempfix_seed${seed}"
  output="$ROOT/outputs/student/$name"
  if [[ -f "$output/report.json" ]]; then
    echo "skip complete $name"
    continue
  fi
  "$PYTHON" "$SCRIPT" \
    --output "$output" \
    --device cuda:0 \
    --method full_kd \
    --teacher-targets "$ROOT/outputs/probe/official_train_valid_seed2026/predictions.jsonl" \
    --teacher-calibration-temperature 1.0149979591 \
    --seed "$seed" \
    --batch-size 8 \
    --gradient-accumulation 1 \
    --epochs 30 \
    --patience 7 \
    --num-workers 2 \
    >>"$LOG_DIR/${name}.log" 2>&1
done
