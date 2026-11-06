#!/usr/bin/env bash
# AQFM v2 medium width (d_model 256, 8 heads, d_ff 512; 4,572,219 parameters) WITHOUT future weather
# (--no-future-met): seeds 0, 1, 2 with the settings of scripts/pipeline_v2m256_seeds.sh. Compares with
# v2 / v1 without future weather and with PatchTST, E-STGCN and XGBoost (no future met).
# Then the 2025 test: each seed, mean ± std, pooled 3-seed ensemble.
# Run from anywhere:  nohup setsid bash scripts/pipeline_v2m256_nofm.sh > runs/v2m256_nofm_pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }

train() {  # out_dir, seed
  log "train $1: seed $2"
  $PY -u -m aqfm.train --phase main --preset city --data data/delhi --out "$1" --device cuda --batch-size 8 \
      --weight-decay 0.05 --compile --dropout 0.3 --seed "$2" --steps 12000 --patience 3 \
      --d-model 256 --n-heads 8 --d-ff 512 --no-future-met > "$1.txt" 2>&1
  tail -1 "$1.txt"
  grep -q "^done" "$1.txt" || { log "$1 did not finish - stopping"; exit 1; }
}

evaluate() {  # out_json, log_file, checkpoints...
  local out=$1 lg=$2; shift 2
  log "evaluate -> $out: $*"
  $PY -u -m aqfm.evaluate --data data/delhi --ckpt "$@" --out "$out" --device cuda --seed 0 > "$lg" 2>&1
  tail -2 "$lg"
}

for s in 0 1 2; do train runs/v2m256_nofm_s${s}_do03 $s; done
for s in 0 1 2; do
  evaluate results/v2m256_nofm_s${s}_results.json runs/eval_v2m256_nofm_s${s}.txt runs/v2m256_nofm_s${s}_do03/best.pt
done
$PY scripts/seed_stats.py results/v2m256_nofm_seeds_stats.json \
    results/v2m256_nofm_s0_results.json results/v2m256_nofm_s1_results.json results/v2m256_nofm_s2_results.json
evaluate results/v2m256_nofm_ens3_results.json runs/eval_v2m256_nofm_ens3.txt \
    runs/v2m256_nofm_s0_do03/best.pt runs/v2m256_nofm_s1_do03/best.pt runs/v2m256_nofm_s2_do03/best.pt
log "pipeline finished"
