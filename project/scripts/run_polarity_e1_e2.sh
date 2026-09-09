#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CUDA_DEVICE {13|42}" >&2
  exit 2
fi

device="$1"
seed="$2"
case "$seed" in
  13|42) ;;
  *) echo "pilot seed must be 13 or 42" >&2; exit 2 ;;
esac

root="/home/wy/sjq/kd"
python_bin="/home/wy/sjq/miniconda3/envs/kd/bin/python"
trainer="$root/project/scripts/train_student_baseline.py"
features="$root/outputs/student/features/official_train_valid"
teacher="$root/outputs/probe/official_train_valid_seed2026/predictions.jsonl"

cd "$root"
for specification in "e1:binary_full_kd" "e2:binary_subset4"; do
  name="${specification%%:*}"
  method="${specification##*:}"
  output="$root/outputs/student/polarity_${name}_seed${seed}"
  if [[ -f "$output/report.json" ]]; then
    echo "Skipping completed run: $output"
    continue
  fi
  if [[ -d "$output" ]]; then
    echo "Refusing to overwrite incomplete run directory: $output" >&2
    exit 3
  fi
  "$python_bin" "$trainer" \
    --features "$features" \
    --teacher-targets "$teacher" \
    --output "$output" \
    --device "cuda:${device}" \
    --seed "$seed" \
    --method "$method" \
    --alpha-ce 0.5 \
    --alpha-binary 1 \
    --lambda-full 1 \
    --lambda-kd-regression 1 \
    --lambda-kd-classification 1 \
    --lambda-binary-full 1 \
    --lambda-binary-subset 1 \
    --kd-temperature 2 \
    --teacher-calibration-temperature 0.9259549975395203
done
