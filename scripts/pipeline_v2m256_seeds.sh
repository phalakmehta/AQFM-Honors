#!/usr/bin/env bash
# AQFM v2 medium width (d_model 256, 8 heads, d_ff 512; 4,572,219 parameters): seeds 1 and 2 with the exact
# settings of seed 0 (scripts/pipeline_v2_size_steps.sh, run B), which won on the 2024 validation score
# (1.7268 vs 1.7372 for v2 and 1.7299 for v1). Then the 2025 test: each seed, mean ± std, pooled 3-seed ensemble.
# Run from anywhere:  nohup setsid bash scripts/pipeline_v2m256_seeds.sh > runs/v2m256_seeds_pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }

train() {  # out_dir, seed
  log "train $1: seed $2"
  $PY -u -m aqfm.train --phase main --preset city --data data/delhi --out "$1" --device cuda --batch-size 8 \
      --weight-decay 0.05 --compile --dropout 0.3 --seed "$2" --steps 12000 --patience 3 \
      --d-model 256 --n-heads 8 --d-ff 512 > "$1.txt" 2>&1
  tail -1 "$1.txt"
  grep -q "^done" "$1.txt" || { log "$1 did not finish - stopping"; exit 1; }
}

evaluate() {  # out_json, log_file, checkpoints...
  local out=$1 lg=$2; shift 2
  log "evaluate -> $out: $*"
  $PY -u -m aqfm.evaluate --data data/delhi --ckpt "$@" --out "$out" --device cuda --seed 0 > "$lg" 2>&1
  tail -2 "$lg"
}

[ -f results/v2m256_s0_results.json ] || { log "seed 0 result missing - run pipeline_v2_size_steps.sh first"; exit 1; }
for s in 1 2; do train runs/v2m256_s${s}_do03 $s; done
for s in 1 2; do
  evaluate results/v2m256_s${s}_results.json runs/eval_v2m256_s${s}.txt runs/v2m256_s${s}_do03/best.pt
done
$PY scripts/seed_stats.py results/v2m256_seeds_stats.json \
    results/v2m256_s0_results.json results/v2m256_s1_results.json results/v2m256_s2_results.json
evaluate results/v2m256_ens3_results.json runs/eval_v2m256_ens3.txt \
    runs/v2m256_s0_do03/best.pt runs/v2m256_s1_do03/best.pt runs/v2m256_s2_do03/best.pt
log "pipeline finished"
