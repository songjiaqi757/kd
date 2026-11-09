#!/usr/bin/env bash
set -euo pipefail

while systemctl --user is-active --quiet rdid-stage-d-gpu1-queue; do
  sleep 60
done

test -f outputs/student/stage_d_d2_ru_lora_seed13/report.json
exec project/scripts/run_stage_d_seed2026_gpu1.sh d2
