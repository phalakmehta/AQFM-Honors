#!/usr/bin/env bash
# Seasonal-aware model variant (v2-M-S) training and evaluation pipeline.
# Architecture: AQFMConfig.city() with width 256 (d_model=256, n_heads=8, d_ff=512),
# --season flag enabled, and --no2-direct.
# Trains 3 seeds with future weather (seeds 0, 1, 2) -> runs/v2m256s_s{0,1,2}_do03
# Evaluates individual seeds -> results/v2m256s_s{0,1,2}_results.json
# Aggregates seed stats -> results/v2m256s_seeds_stats.json
# Evaluates 3-seed ensemble -> results/v2m256s_ens3_results.json
# Saves ensemble forecasts -> results/forecasts/delhi_v2m256s_ens3.npz
#
# Also trains 3 seeds without future weather (--no-future-met) -> runs/v2m256s_nofm_s{0,1,2}_do03
# Evaluates individual seeds -> results/v2m256s_nofm_s{0,1,2}_results.json
# Aggregates seed stats -> results/v2m256s_nofm_seeds_stats.json
# Evaluates 3-seed ensemble -> results/v2m256s_nofm_ens3_results.json
# Saves ensemble forecasts -> results/forecasts/delhi_v2m256s_nofm_ens3.npz
#
# Runs comparisons against base v2-M (delhi_v2m256_no2d_ens3.npz), XGBoost, and Persistence.
# Runs P2 (extreme_value_analysis.py), P3 (station_error_analysis.py), P4 (missing_value_impact.py).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }

export TRITON_CACHE_DIR=/tmp/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" results/forecasts results/paper

git diff --quiet HEAD -- aqfm tests baselines && [ -z "$(git ls-files --others --exclude-standard aqfm tests baselines)" ] \
    || { log "uncommitted code under aqfm/ tests/ baselines/: commit first, so every run records a clean commit"; exit 1; }

F=results/forecasts
ALL="--targets PM2.5 PM10 Ozone NOx NO2"
WIDE="--d-model 256 --n-heads 8 --d-ff 512"

train() {
  local out=$1; shift
  if [ -f "$out/best.pt" ] && [ -f "$out.txt" ] && grep -q "^done" "$out.txt"; then
    log "skip $out: already finished"
    return 0
  fi
  log "train $out: $*"
  $PY -u -m aqfm.train --phase main --preset city --data data/delhi --out "$out" --device cuda --batch-size 8 \
      --weight-decay 0.05 --compile --dropout 0.3 --steps 12000 --patience 3 "$@" > "$out.txt" 2>&1
  tail -1 "$out.txt"
  grep -q "^done" "$out.txt" || { log "$out did not finish - stopping"; exit 1; }
}

evaluate() {
  local out=$1 lg=$2 extra=$3; shift 3
  if [ -f "$out" ]; then
    log "skip evaluate $out: already exists"
    return 0
  fi
  log "evaluate $out"
  $PY -u -m aqfm.evaluate --data data/delhi $ALL $extra --ckpt "$@" --out $out --device cuda --seed 0 > $lg 2>&1
  tail -4 $lg
}

stats() {
  local out=$1 tgt=$2; shift 2
  local a=()
  while [ $# -gt 0 ]; do a+=(--model "$1" "$2"); shift 2; done
  $PY scripts/forecast_stats.py --target $tgt "${a[@]}" --out results/paper/$out.json | tail -5
}

for v in "" "_nofm"; do
  extra="$WIDE --no2-direct --season"
  [ -n "$v" ] && extra="$extra --no-future-met"

  for s in 0 1 2; do
    train runs/v2m256s${v}_s${s}_do03 --seed $s $extra
  done

  for s in 0 1 2; do
    evaluate results/v2m256s${v}_s${s}_results.json runs/eval_v2m256s${v}_s${s}.txt "" \
        runs/v2m256s${v}_s${s}_do03/best.pt
  done

  if [ ! -f "results/v2m256s${v}_seeds_stats.json" ]; then
    $PY scripts/seed_stats.py results/v2m256s${v}_seeds_stats.json \
        results/v2m256s${v}_s0_results.json results/v2m256s${v}_s1_results.json results/v2m256s${v}_s2_results.json
  fi

  evaluate results/v2m256s${v}_ens3_results.json runs/eval_v2m256s${v}_ens3.txt \
      "--save-forecasts $F/delhi_v2m256s${v}_ens3.npz" \
      runs/v2m256s${v}_s0_do03/best.pt runs/v2m256s${v}_s1_do03/best.pt runs/v2m256s${v}_s2_do03/best.pt
done

log "Running forecast stats comparisons..."
for c in PM2.5 PM10 Ozone NOx NO2; do
  stats delhi_${c}_with_weather_seasonal $c \
      "AQFM v2-M-S (Seasonal)" $F/delhi_v2m256s_ens3.npz \
      "AQFM v2-M" $F/delhi_v2m256_no2d_ens3.npz \
      "XGBoost + future ERA5" $F/delhi_xgboost_fm.npz:xgboost \
      Persistence $F/delhi_classical.npz:persistence

  stats delhi_${c}_no_weather_seasonal $c \
      "AQFM v2-M-S (Seasonal)" $F/delhi_v2m256s_nofm_ens3.npz \
      "AQFM v2-M" $F/delhi_v2m256_no2d_nofm_ens3.npz \
      XGBoost $F/delhi_xgboost.npz:xgboost \
      Persistence $F/delhi_classical.npz:persistence
done

log "Running extreme value analysis (Priority 2)..."
$PY scripts/extreme_value_analysis.py \
    --forecast $F/delhi_v2m256s_ens3.npz $F/delhi_v2m256_no2d_ens3.npz \
    --out results/paper/delhi_v2m256s_extreme.json \
    --report results/paper/delhi_v2m256s_extreme.md

log "Running spatial station error analysis (Priority 3)..."
$PY scripts/station_error_analysis.py \
    --forecast $F/delhi_v2m256s_ens3.npz \
    --stations data_pipeline/stations/delhi.csv \
    --facts data_pipeline/stations/delhi_facts.csv \
    --target PM2.5 \
    --out results/paper/delhi_v2m256s_station_error.json \
    --out-csv results/paper/delhi_v2m256s_station_error_map.csv \
    --report results/paper/delhi_v2m256s_station_error.md

log "Running missing value impact analysis (Priority 4)..."
$PY scripts/missing_value_impact.py \
    --forecast $F/delhi_v2m256s_ens3.npz \
    --data data/delhi \
    --target PM2.5 \
    --checkpoint runs/v2m256s_s0_do03/best.pt \
    --device cuda \
    --out results/paper/delhi_v2m256s_missing_impact.json \
    --report results/paper/delhi_v2m256s_missing_impact.md

log "pipeline finished"
