#!/usr/bin/env bash
set -euo pipefail

# Heavy online-LoRA jobs share GPU 1 sequentially. Existing cached-feature jobs
# can continue concurrently because their memory footprint is small.
project/scripts/run_stage_d_lora_gate.sh d2-seed42
project/scripts/run_stage_d_lora_gate.sh d2-seed13
