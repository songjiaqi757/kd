#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_DIR="/home/wy/sjq/kd/model/Qwen3-Omni-30B-A3B-Instruct"
HF_BIN="${HF_BIN:-/home/wy/sjq/.local/bin/hf}"

export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export HF_HUB_DISABLE_XET=1

mkdir -p "${MODEL_DIR}"

# Serialize large transfers because the local proxy failed under concurrency.
# Hugging Face keeps partial files, so rerunning is safe after interruption.
for shard_number in $(seq 1 15); do
    shard=$(printf 'model-%05d-of-00015.safetensors' "${shard_number}")
    echo "[$(date '+%F %T')] downloading ${shard}"
    "${HF_BIN}" download "${REPO_ID}" "${shard}" --local-dir "${MODEL_DIR}"
done

echo "[$(date '+%F %T')] downloading tokenizer and configuration files"
"${HF_BIN}" download "${REPO_ID}" --exclude 'model-*.safetensors' --local-dir "${MODEL_DIR}"
echo "[$(date '+%F %T')] model download complete"
