#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/wy/sjq/miniconda3/envs/kd/bin/python
SCRIPT=/home/wy/sjq/kd/project/scripts/train_student_lora.py
ROOT=/home/wy/sjq/kd
LOG_DIR="$ROOT/outputs/logs/stage_d_attribution"
ENSEMBLE=(
  "$ROOT/outputs/probe/official_train_valid_seed2026/predictions.jsonl"
  "$ROOT/outputs/probe/official_train_valid_seed2027/predictions.jsonl"
  "$ROOT/outputs/probe/official_train_valid_seed2028/predictions.jsonl"
)
mkdir -p "$LOG_DIR"
cd "$ROOT"

run_one() {
  local method="$1"
  local seed="$2"
  local name output
  if [[ "$method" == student ]]; then
    name="stage_d_c0_task_only_lora_seed${seed}"
  else
    name="stage_d_c2_ensemble_pair_lora_seed${seed}"
  fi
  output="$ROOT/outputs/student/$name"
  if [[ -f "$output/report.json" ]]; then
    echo "skip complete $name"
    return
  fi
  local command=(
    "$PYTHON" "$SCRIPT"
    --output "$output"
    --device cuda:0
    --method "$method"
    --teacher-targets "$ROOT/outputs/probe/official_train_valid_seed2026/predictions.jsonl"
    --seed "$seed"
    --batch-size 8
    --gradient-accumulation 1
    --epochs 30
    --patience 7
    --num-workers 2
  )
  if [[ "$method" == ensemble_pair ]]; then
    command+=(--teacher-targets-ensemble "${ENSEMBLE[@]}")
  fi
  "${command[@]}" >>"$LOG_DIR/${name}.log" 2>&1
}

# Complete the fixed two-seed screening matrix without stacking GPU processes.
run_one student 13
run_one ensemble_pair 13
run_one student 42
run_one ensemble_pair 42

# Promote C2 only if it improves on same-seed D1 by >=0.005 for both pilot seeds.
if "$PYTHON" -c '
import json
from pathlib import Path
r=Path("/home/wy/sjq/kd/outputs/student")
ok=True
for s in (13,42):
    d1=json.loads((r/f"stage_d_d1_full_kd_lora_seed{s}/report.json").read_text())["valid_metrics"]["mae"]
    c2=json.loads((r/f"stage_d_c2_ensemble_pair_lora_seed{s}/report.json").read_text())["valid_metrics"]["mae"]
    ok &= d1-c2 >= .005
raise SystemExit(0 if ok else 1)
'; then
  run_one ensemble_pair 2026
fi

"$PYTHON" -c '
import json
from pathlib import Path
r=Path("/home/wy/sjq/kd/outputs/student")
out={"status":"complete","pilot_seeds":[13,42],"promotion_rule":"C2 beats same-seed D1 by >=0.005 for 2/2 seeds","official_test_evaluated":False,"runs":{}}
for p in sorted(r.glob("stage_d_c[02]_*_lora_seed*/report.json")):
    out["runs"][p.parent.name]=json.loads(p.read_text())
(r/"stage_d_attribution_queue_summary.json").write_text(json.dumps(out,indent=2)+"\n")
'
