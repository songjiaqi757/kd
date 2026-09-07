#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {subset4|pair_snr} CUDA_DEVICE" >&2
  exit 2
fi

method="$1"
device="$2"
root="/home/wy/sjq/kd"
python_bin="/home/wy/sjq/miniconda3/envs/kd/bin/python"
trainer="$root/project/scripts/train_student_baseline.py"
features="$root/outputs/student/features/official_train_valid"
teacher="$root/outputs/probe/official_train_valid_seed2026/predictions.jsonl"

cd "$root"
for seed in 13 2026; do
  if [[ "$method" == "subset4" ]]; then
    "$python_bin" "$trainer" \
      --features "$features" \
      --output "$root/outputs/student/fullscale_subset4_seed${seed}" \
      --device "cuda:${device}" \
      --seed "$seed" \
      --method subset_value_4 \
      --teacher-targets "$teacher" \
      --lambda-coordinate 0
  elif [[ "$method" == "pair_snr" ]]; then
    "$python_bin" "$trainer" \
      --features "$features" \
      --output "$root/outputs/student/fullscale_pair_snr_seed${seed}" \
      --device "cuda:${device}" \
      --seed "$seed" \
      --method pair_snr \
      --teacher-targets "$teacher" \
      --teacher-targets-ensemble \
        "$root/outputs/probe/official_train_valid_seed2026/predictions.jsonl" \
        "$root/outputs/probe/official_train_valid_seed2027/predictions.jsonl" \
        "$root/outputs/probe/official_train_valid_seed2028/predictions.jsonl" \
      --lambda-subset 0
  else
    echo "unknown method: $method" >&2
    exit 2
  fi
done
