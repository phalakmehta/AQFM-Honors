#!/usr/bin/env bash
# AQFM v2, two seed-0 tests to improve the metrics (2026-11-05). Each changes ONE thing from the step-9
# recipe (scripts/pipeline_v2_delhi.sh: city preset, data/delhi, 12,000 steps, batch 8, dropout 0.3,
# weight decay 0.05, patience 3, --compile). Compared with runs/v2_s0_do03 on the 2024 VALIDATION score.
#   A. longer training: 24,000 steps (cosine schedule over 24k); patience 24 so early stopping cannot cut
#      the schedule short (best.pt is still the best validation step)
#   B. medium width: d_model 256, 8 heads, d_ff 512 (4,572,219 parameters; v2 1,172,271, v1 23,801,736)
# Then the 2025 test for both, scored like every other result.
# Run from anywhere:  nohup setsid bash scripts/pipeline_v2_size_steps.sh > runs/v2_size_steps_pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }

train() {  # out_dir, extra args...
  local out=$1; shift
  log "train $out: $*"
  $PY -u -m aqfm.train --phase main --preset city --data data/delhi --out "$out" --device cuda --batch-size 8 \
      --weight-decay 0.05 --compile --dropout 0.3 --seed 0 "$@" > "$out.txt" 2>&1
  tail -1 "$out.txt"
  grep -q "^done" "$out.txt" || { log "$out did not finish - stopping"; exit 1; }
}

evaluate() {  # out_json, log_file, checkpoint
  log "evaluate -> $1: $3"
  $PY -u -m aqfm.evaluate --data data/delhi --ckpt "$3" --out "$1" --device cuda --seed 0 > "$2" 2>&1
  tail -2 "$2"
}

train runs/v2_s0_do03_24k --steps 24000 --patience 24
train runs/v2m256_s0_do03 --steps 12000 --patience 3 --d-model 256 --n-heads 8 --d-ff 512
evaluate results/v2_s0_24k_results.json runs/eval_v2_s0_24k.txt runs/v2_s0_do03_24k/best.pt
evaluate results/v2m256_s0_results.json runs/eval_v2m256_s0.txt runs/v2m256_s0_do03/best.pt
log "pipeline finished"
