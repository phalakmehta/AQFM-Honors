#!/usr/bin/env bash
# Step 10 follow-up (user: "yea okay", 2026-11-06): Mumbai fine-tuning with NOTHING frozen, 6 and 12 months, seeds 0-2.
# Same recipe as the frozen-backbone runs of scripts/pipeline_mumbai_transfer.sh (lr 3e-5, up to 6,000 steps,
# validation every 50, patience 8; start from the Delhi v2 model without future weather of the same seed).
# Question: with 6+ months, from scratch beats the frozen-backbone fine-tune on PM2.5 - does the gap close when every
# weight trains? Selection stays on validation (2024 H1): compare the best validation scores with
# runs/mumbai_ft_v2_backbone_m{6,12}_s*. Test results are scored like the others, plus the same-point comparison.
# Run from anywhere:  nohup setsid bash scripts/pipeline_mumbai_unfrozen.sh > runs/mumbai_unfrozen.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }
SPLIT="--val-start 2024-01-01 --test-start 2024-07-01"
FT="--phase finetune --steps 6000 --warmup 100 --val-every 50 --patience 8 --lr 3e-5"
ALL="--targets PM2.5 PM10 Ozone NOx NO2"
mkdir -p results/paper results/forecasts

evaluate() {  # out_json, log_file, extra args, checkpoints...
  local out=$1 lg=$2 extra=$3; shift 3
  log "evaluate -> $out: $*"
  $PY -u -m aqfm.evaluate --data data/mumbai $SPLIT $extra --ckpt "$@" --out "$out" --device cuda --seed 0 > "$lg" 2>&1
  tail -2 "$lg"
}

for s in 0 1 2; do
  for n in 6 12; do
    out=runs/mumbai_ft_v2_none_m${n}_s$s
    if [ -e "$out" ]; then log "$out exists - stopping (move it and $out.txt aside first)"; exit 1; fi
    log "train $out"
    $PY -u -m aqfm.train --data data/mumbai $SPLIT --out "$out" --device cuda --batch-size 8 --weight-decay 0.05 \
        --compile $FT --init runs/v2_nofm_s${s}_do03/best.pt --freeze none --train-months $n --seed $s > "$out.txt" 2>&1
    tail -1 "$out.txt"
    grep -q "^done" "$out.txt" || { log "$out did not finish - stopping"; exit 1; }
  done
done

F=results/forecasts
for n in 6 12; do
  k=mumbai_ft_v2_none_m$n
  for s in 0 1 2; do
    evaluate results/${k}_s${s}_results.json runs/eval_${k}_s$s.txt "" runs/${k}_s$s/best.pt
  done
  $PY scripts/seed_stats.py results/${k}_seeds_stats.json results/${k}_s{0,1,2}_results.json
  evaluate results/${k}_ens3_results.json runs/eval_${k}_ens3.txt "" runs/${k}_s{0,1,2}/best.pt
  evaluate results/paper/mumbai_ftnone_m${n}_ens3.json runs/eval_paper_mumbai_ftnone_m${n}_ens3.txt \
      "$ALL --save-forecasts $F/mumbai_ftnone_m${n}_ens3.npz" runs/${k}_s{0,1,2}/best.pt
  for c in PM2.5 PM10 NOx Ozone; do
    $PY scripts/forecast_stats.py --target $c --model "AQFM fine-tuned, nothing frozen" $F/mumbai_ftnone_m${n}_ens3.npz \
        --model "AQFM fine-tuned, backbone frozen" $F/mumbai_ft_m${n}_ens3.npz \
        --model "AQFM from scratch" $F/mumbai_scratch_m${n}_ens3.npz \
        --model "XGBoost (same months)" $F/mumbai_xgboost_m$n.npz:xgboost \
        --model Persistence $F/mumbai_classical.npz:persistence \
        --out results/paper/mumbai_${c}_m${n}_unfrozen.json | tail -5
  done
done
log "pipeline finished (unfrozen)"
