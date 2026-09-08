#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {utility_queue|reliability_queue} CUDA_DEVICE" >&2
  exit 2
fi

queue="$1"
device="$2"
root="/home/wy/sjq/kd"
python_bin="/home/wy/sjq/miniconda3/envs/kd/bin/python"
trainer="$root/project/scripts/train_student_baseline.py"
features="$root/outputs/student/features/official_train_valid"
teacher="$root/outputs/probe/official_train_valid_seed2026/predictions.jsonl"
ensemble=(
  "$root/outputs/probe/official_train_valid_seed2026/predictions.jsonl"
  "$root/outputs/probe/official_train_valid_seed2027/predictions.jsonl"
  "$root/outputs/probe/official_train_valid_seed2028/predictions.jsonl"
)

case "$queue" in
  utility_queue)
    methods=(ensemble_pair utility_pair)
    ;;
  reliability_queue)
    methods=(reliability_utility_pair selective_interaction4)
    ;;
  *)
    echo "unknown queue: $queue" >&2
    exit 2
    ;;
esac

cd "$root"
for method in "${methods[@]}"; do
  case "$method" in
    ensemble_pair) output="fullscale_ensemble_pair_seed13" ;;
    utility_pair) output="fullscale_utility_pair_seed13" ;;
    reliability_utility_pair) output="fullscale_reliability_utility_pair_seed13" ;;
    selective_interaction4) output="fullscale_selective50_interaction4_seed13" ;;
  esac
  "$python_bin" "$trainer" \
    --features "$features" \
    --output "$root/outputs/student/$output" \
    --device "cuda:${device}" \
    --seed 13 \
    --method "$method" \
    --teacher-targets "$teacher" \
    --teacher-targets-ensemble "${ensemble[@]}" \
    --teacher-calibration-temperature 0.9259549975395203 \
    --lambda-full 1 \
    --lambda-subset 0 \
    --lambda-coordinate 1 \
    --lambda-kd-regression 1 \
    --lambda-kd-classification 1 \
    --kd-temperature 2 \
    --selective-keep-fraction 0.5
done
