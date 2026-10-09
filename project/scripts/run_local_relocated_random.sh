#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SEED PHYSICAL_GPU" >&2
  exit 2
fi

seed="$1"
physical_gpu="$2"
if [[ "$seed" != "42" && "$seed" != "2026" ]]; then
  echo "unsupported seed: $seed" >&2
  exit 2
fi
if [[ "$physical_gpu" != "0" && "$physical_gpu" != "1" ]]; then
  echo "unsupported physical GPU: $physical_gpu" >&2
  exit 2
fi

project_root="/home/wy/sjq/kd"
python_bin="/home/wy/sjq/miniconda3/envs/kd/bin/python"
base="$project_root/outputs/experiments/multiseed_controls_v1/mosei"
run="$base/students/random_orthogonal_seed${seed}"
train_log="$base/logs/random_orthogonal_seed${seed}.local_resume.train.log"
job_log="$base/logs/random_orthogonal_seed${seed}.local_resume.job.log"

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

cd "$project_root"

"$python_bin" -u project/scripts/train_main_table_all_epochs_relocated.py \
  --method random_orthogonal \
  --assets outputs/experiments/main_table_v1/mosei/assets/protocol.json \
  --output "$run" \
  --seed "$seed" \
  --device cuda:0 \
  --epochs 20 \
  --min-epochs 8 \
  --patience 7 \
  --batch-size 8 \
  --coordinate-seed 20260922 \
  --checkpoint-selection valid_mae \
  --resume >>"$train_log" 2>&1

# The standard job runner observes the completed training run, skips training,
# and performs the one-shot validation-MAE-selected official-test evaluation.
"$python_bin" -u project/scripts/run_multiseed_control_job.py \
  --dataset mosei \
  --method random_orthogonal \
  --seed "$seed" \
  --base "$base" \
  --device cuda:0 >>"$job_log" 2>&1
