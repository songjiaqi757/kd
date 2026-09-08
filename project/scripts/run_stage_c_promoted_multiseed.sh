#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 {primary|uniform} CUDA_DEVICE {MAX_WAIT_MINUTES|force}" >&2
  exit 2
fi

queue="$1"
device="$2"
wait_policy="$3"
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
  primary) methods=(selective_interaction4 reliability_utility_pair) ;;
  uniform) methods=(ensemble_pair) ;;
  *) echo "unknown queue: $queue" >&2; exit 2 ;;
esac

if [[ "$wait_policy" == "force" ]]; then
  echo "Starting $queue queue immediately on GPU $device by explicit user request"
else
  max_wait_minutes="$wait_policy"
  if ! [[ "$max_wait_minutes" =~ ^[0-9]+$ ]]; then
    echo "MAX_WAIT_MINUTES must be a non-negative integer or force" >&2
    exit 2
  fi
  for ((minute = 0; minute <= max_wait_minutes; minute++)); do
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$device" | tr -d ' ')
    if [[ "$used_mib" =~ ^[0-9]+$ ]] && ((used_mib < 1000)); then
      echo "GPU $device available after $minute minute(s); starting $queue queue"
      break
    fi
    if ((minute == max_wait_minutes)); then
      echo "GPU $device remained occupied for $max_wait_minutes minutes; exiting without training" >&2
      exit 75
    fi
    sleep 60
  done
fi

cd "$root"
for method in "${methods[@]}"; do
  for seed in 42 2026; do
    case "$method" in
      ensemble_pair) output="fullscale_ensemble_pair_seed${seed}" ;;
      reliability_utility_pair) output="fullscale_reliability_utility_pair_seed${seed}" ;;
      selective_interaction4) output="fullscale_selective50_interaction4_seed${seed}" ;;
    esac
    "$python_bin" "$trainer" \
      --features "$features" \
      --output "$root/outputs/student/$output" \
      --device "cuda:${device}" \
      --seed "$seed" \
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
done
