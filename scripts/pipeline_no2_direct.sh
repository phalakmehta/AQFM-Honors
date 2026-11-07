#!/usr/bin/env bash
# NO2 forecast directly (train.py --no2-direct; user, 2026-11-07). Cause: NO2 = NOx x (1 - NO share) can never be
# above NOx, and the NO share is valid for only 58 % of the training NO2 readings (docs/V2_PROGRESS.md step 12).
# The 6 v2-M paper models again with --no2-direct (with and without future weather, seeds 0/1/2), each seed and the
# 3-seed ensembles scored on the 2025 test for every pollutant (ensemble forecasts saved), then the same-point
# comparisons against the old v2-M, XGBoost and persistence (stage SAVE files).                        ~4 h GPU
# Run from anywhere:  nohup setsid bash scripts/pipeline_no2_direct.sh > runs/no2_direct.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }
grep -q -- "--no2-direct" aqfm/train.py || { log "aqfm/train.py has no --no2-direct"; exit 1; }
git diff --quiet HEAD -- aqfm tests baselines && [ -z "$(git ls-files --others --exclude-standard aqfm tests baselines)" ] \
    || { log "uncommitted code under aqfm/ tests/ baselines/: commit first, so every run records a clean commit"; exit 1; }
F=results/forecasts
for f in $F/delhi_v2m256_ens3.npz $F/delhi_v2m256_nofm_ens3.npz $F/delhi_xgboost_fm.npz $F/delhi_xgboost.npz \
         $F/delhi_classical.npz; do
  [ -f $f ] || { log "$f missing: run STAGE=SAVE of scripts/pipeline_paper_extras.sh first"; exit 1; }
done
ALL="--targets PM2.5 PM10 Ozone NOx NO2"
WIDE="--d-model 256 --n-heads 8 --d-ff 512"

train() {  # out_dir, extra args...
  local out=$1; shift
  log "train $out: $*"
  $PY -u -m aqfm.train --phase main --preset city --data data/delhi --out "$out" --device cuda --batch-size 8 \
      --weight-decay 0.05 --compile --dropout 0.3 --steps 12000 --patience 3 "$@" > "$out.txt" 2>&1
  tail -1 "$out.txt"
  grep -q "^done" "$out.txt" || { log "$out did not finish - stopping"; exit 1; }
}
evaluate() {  # out json, log, extra args, checkpoints...
  local out=$1 lg=$2 extra=$3; shift 3
  log "evaluate $out"
  $PY -u -m aqfm.evaluate --data data/delhi $ALL $extra --ckpt "$@" --out $out --device cuda --seed 0 > $lg 2>&1
  tail -4 $lg
}
stats() {  # out, target, then NAME FILE pairs
  local out=$1 tgt=$2; shift 2
  local a=()
  while [ $# -gt 0 ]; do a+=(--model "$1" "$2"); shift 2; done
  $PY scripts/forecast_stats.py --target $tgt "${a[@]}" --out results/paper/$out.json | tail -5
}

for v in "" "_nofm"; do
  extra="$WIDE --no2-direct"; [ -n "$v" ] && extra="$extra --no-future-met"
  for s in 0 1 2; do
    train runs/v2m256_no2d${v}_s${s}_do03 --seed $s $extra
  done
  for s in 0 1 2; do
    evaluate results/v2m256_no2d${v}_s${s}_results.json runs/eval_v2m256_no2d${v}_s${s}.txt "" \
        runs/v2m256_no2d${v}_s${s}_do03/best.pt
  done
  $PY scripts/seed_stats.py results/v2m256_no2d${v}_seeds_stats.json \
      results/v2m256_no2d${v}_s0_results.json results/v2m256_no2d${v}_s1_results.json results/v2m256_no2d${v}_s2_results.json
  evaluate results/v2m256_no2d${v}_ens3_results.json runs/eval_v2m256_no2d${v}_ens3.txt \
      "--save-forecasts $F/delhi_v2m256_no2d${v}_ens3.npz" \
      runs/v2m256_no2d${v}_s0_do03/best.pt runs/v2m256_no2d${v}_s1_do03/best.pt runs/v2m256_no2d${v}_s2_do03/best.pt
done

for c in PM2.5 PM10 Ozone NOx NO2; do
  stats delhi_${c}_with_weather_no2d $c "AQFM v2-M, NO2 direct" $F/delhi_v2m256_no2d_ens3.npz \
      "AQFM v2-M" $F/delhi_v2m256_ens3.npz "XGBoost + future ERA5" $F/delhi_xgboost_fm.npz:xgboost \
      Persistence $F/delhi_classical.npz:persistence
  stats delhi_${c}_no_weather_no2d $c "AQFM v2-M, NO2 direct" $F/delhi_v2m256_no2d_nofm_ens3.npz \
      "AQFM v2-M" $F/delhi_v2m256_nofm_ens3.npz XGBoost $F/delhi_xgboost.npz:xgboost \
      Persistence $F/delhi_classical.npz:persistence
done
log "pipeline finished"
