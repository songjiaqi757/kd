#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {student|full_kd} CUDA_DEVICE" >&2
  exit 2
fi

method="$1"
device="$2"
root="/home/wy/sjq/kd"
python_bin="/home/wy/sjq/miniconda3/envs/kd/bin/python"
trainer="$root/project/scripts/train_student_baseline.py"
features="$root/outputs/student/features/official_train_valid"
teacher="$root/outputs/probe/official_train_valid_seed2026/predictions.jsonl"

case "$method" in
  student|full_kd) ;;
  *)
    echo "unknown method: $method" >&2
    exit 2
    ;;
esac

cd "$root"
for seed in 13 42 2026; do
  command=(
    "$python_bin" "$trainer"
    --features "$features"
    --output "$root/outputs/student/fullscale_${method}_seed${seed}"
    --device "cuda:${device}"
    --seed "$seed"
    --method "$method"
    --teacher-calibration-temperature 0.9259549975395203
  )
  if [[ "$method" == "full_kd" ]]; then
    command+=(
      --teacher-targets "$teacher"
      --lambda-full 1
      --lambda-kd-regression 1
      --lambda-kd-classification 1
      --kd-temperature 2
    )
  fi
  "${command[@]}"
done
