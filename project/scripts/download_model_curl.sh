#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_DIR="/home/wy/sjq/kd/model/Qwen3-Omni-30B-A3B-Instruct"
BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main"
HF_BIN="${HF_BIN:-/home/wy/sjq/.local/bin/hf}"

mkdir -p "${MODEL_DIR}"

for shard_number in $(seq 1 15); do
    shard=$(printf 'model-%05d-of-00015.safetensors' "${shard_number}")
    final_path="${MODEL_DIR}/${shard}"
    part_path="${final_path}.part"

    if [[ -s "${final_path}" ]]; then
        echo "[$(date '+%F %T')] already complete: ${shard}"
        continue
    fi

    while true; do
        before_bytes=$(stat -c %s "${part_path}" 2>/dev/null || echo 0)
        echo "[$(date '+%F %T')] downloading/resuming ${shard} from ${before_bytes} bytes"

        # Do not use curl's internal --retry here. A fresh curl invocation must
        # recalculate the current .part length after every broken connection.
        if curl \
            --fail \
            --http1.1 \
            --location \
            --continue-at - \
            --connect-timeout 30 \
            --max-time 600 \
            --speed-limit 1024 \
            --speed-time 120 \
            --silent \
            --show-error \
            --output "${part_path}" \
            "${BASE_URL}/${shard}"; then
            break
        fi

        after_bytes=$(stat -c %s "${part_path}" 2>/dev/null || echo 0)
        echo "[$(date '+%F %T')] interrupted: ${before_bytes} -> ${after_bytes} bytes; retrying in 5s"
        sleep 5
    done

    mv "${part_path}" "${final_path}"
    echo "[$(date '+%F %T')] complete: ${shard} ($(stat -c %s "${final_path}") bytes)"
done

echo "[$(date '+%F %T')] all weight shards complete"

export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
while ! "${HF_BIN}" download "${REPO_ID}" --exclude 'model-*.safetensors' --local-dir "${MODEL_DIR}"; do
    echo "[$(date '+%F %T')] metadata download interrupted; retrying in 10s"
    sleep 10
done
echo "[$(date '+%F %T')] tokenizer and configuration files complete"
