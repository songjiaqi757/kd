#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 CUDA_DEVICE ASSETS TEST_MANIFEST OUTPUT_ROOT [SEED|all]" >&2
  exit 2
fi

device="$1"
assets="$2"
test_manifest="$3"
output_root="$4"
seed_selector="${5:-all}"
root="/home/wy/sjq/kd"
python_bin="/home/wy/sjq/miniconda3/envs/kd/bin/python"
trainer="$root/project/scripts/train_main_table_kd.py"
evaluator="$root/project/scripts/evaluate_main_table_test.py"
max_parallel="${MAX_PARALLEL:-2}"
min_free_mib="${MIN_FREE_MIB:-28000}"
launch_settle_seconds="${LAUNCH_SETTLE_SECONDS:-120}"
checkpoint_disk_mib_per_job="${CHECKPOINT_DISK_MIB_PER_JOB:-3800}"
disk_reserve_mib="${DISK_RESERVE_MIB:-5000}"

for value in "$max_parallel" "$min_free_mib" "$launch_settle_seconds" \
             "$checkpoint_disk_mib_per_job" "$disk_reserve_mib"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "parallel controls must be non-negative integers" >&2; exit 2; }
done
(( max_parallel >= 1 )) || { echo "MAX_PARALLEL must be at least 1" >&2; exit 2; }

methods=(
  uniform_interaction
  uniform_interaction_shared_dropout
  uniform_interaction_gradient_projected
  uniform_interaction_coordinate_compatible
  explicit_full_kd
  explicit_uniform_interaction
)
if [[ "$seed_selector" == "all" ]]; then
  seeds=(13 42 2026)
else
  case "$seed_selector" in
    13|42|2026) seeds=("$seed_selector") ;;
    *) echo "seed must be 13, 42, 2026, or all" >&2; exit 2 ;;
  esac
fi

run_one() {
  local method="$1"
  local seed="$2"
  local output="$output_root/${method}_seed${seed}"
  local status=""
  if [[ -f "$output/status.json" ]]; then
    status="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$output/status.json")"
  fi
  local resume_args=()
  if [[ -f "$output/run_config.json" && "$status" != "complete" ]]; then
    resume_args=(--resume)
  elif [[ -f "$output/run_config.json" && "$status" == "complete" ]]; then
    echo "training already complete: $output"
  fi
  if [[ "$status" != "complete" ]]; then
    "$python_bin" "$trainer" \
      --method "$method" \
      --assets "$assets" \
      --output "$output" \
      --device "cuda:${device}" \
      --seed "$seed" \
      --epochs 20 \
      --min-epochs 20 \
      --patience 20 \
      --batch-size 8 \
      --learning-rate 0.0001 \
      --weight-decay 0.01 \
      --checkpoint-selection valid_mae \
      --retain-best-mae-and-acc2 \
      "${resume_args[@]}"
  fi
  for checkpoint in best_valid_mae.pt best_valid_acc2_nonzero.pt best.pt last.pt; do
    [[ -e "$output/$checkpoint" ]] || { echo "missing retained checkpoint: $output/$checkpoint" >&2; exit 1; }
  done
    for policy in valid_mae valid_acc2_nonzero; do
      local test_output="$output/test/$policy"
      if [[ -f "$test_output/status.json" ]] && \
         [[ "$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$test_output/status.json")" == "complete" ]]; then
        echo "test already complete: $test_output"
        continue
      fi
      "$python_bin" "$evaluator" \
        --run "$output" \
        --output "$test_output" \
        --test-manifest "$test_manifest" \
        --device "cuda:${device}" \
        --batch-size 8 \
        --checkpoint-selection "$policy" \
        --allow-official-test
    done
}

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= max_parallel )); do
    wait -n || true
  done
  while :; do
    local active free_mib disk_free_mib disk_required_mib
    active="$(jobs -rp | wc -l)"
    free_mib="$(nvidia-smi -i "$device" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    disk_free_mib="$(df -Pm "$output_root" | awk 'NR==2 {print $4}')"
    disk_required_mib=$((disk_reserve_mib + checkpoint_disk_mib_per_job * (active + 1)))
    if (( free_mib >= min_free_mib && disk_free_mib >= disk_required_mib )); then
      return
    fi
    echo "waiting for resources: gpu${device}_free=${free_mib}/${min_free_mib} MiB disk_free=${disk_free_mib}/${disk_required_mib} MiB"
    sleep 60
  done
}

mkdir -p "$output_root/.logs"
for method in "${methods[@]}"; do
  for seed in "${seeds[@]}"; do
    wait_for_slot
    log="$output_root/.logs/${method}_seed${seed}.log"
    echo "launching $method seed=$seed on cuda:$device; log=$log"
    run_one "$method" "$seed" >"$log" 2>&1 &
    sleep "$launch_settle_seconds"
  done
done
wait
