#!/usr/bin/env bash
# Step 10, test 1: zero-shot transfer. The Delhi-trained v2 and v2-M (3-seed ensembles, with and without future
# weather) forecast Mumbai with no Mumbai training at all. Mumbai data = data/mumbai (hourly 2023-2024, built by
# data_pipeline/convert_hourly_city.py); test = 2024-07-01 .. end of 2024 (2023 = training and 2024 H1 = validation
# for the fine-tuning runs). Mumbai's ERA5 has no 2 m temperature, humidity or sunlight: models with future weather
# get those three as missing, which the scaling turns into 0 (= the Delhi training mean).
# Run from anywhere:  nohup setsid bash scripts/pipeline_mumbai_zeroshot.sh > runs/mumbai_zeroshot_pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }

evaluate() {  # out_json, log_file, checkpoints...
  local out=$1 lg=$2; shift 2
  log "evaluate -> $out: $*"
  $PY -u -m aqfm.evaluate --data data/mumbai --test-start 2024-07-01 --ckpt "$@" --out "$out" \
      --device cuda --seed 0 > "$lg" 2>&1
  tail -2 "$lg"
}
ens() { echo runs/$1_s0_do03/best.pt runs/$1_s1_do03/best.pt runs/$1_s2_do03/best.pt; }

[ -f data/mumbai/values.npy ] || { log "data/mumbai missing - build it first"; exit 1; }
for m in v2m256_nofm v2_nofm v2m256 v2; do
  evaluate results/mumbai_zeroshot_${m}_ens3_results.json runs/eval_mumbai_zeroshot_${m}_ens3.txt $(ens $m)
done
log "pipeline finished"
