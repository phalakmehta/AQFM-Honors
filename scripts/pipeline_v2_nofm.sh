#!/usr/bin/env bash
# AQFM v2 without future weather (2026-11-04): the v2 counterpart of v1's no-future-met
# ablation (results/aqfm_do3_nofm_*). Same settings as scripts/pipeline_v2_delhi.sh, with the
# dropout chosen there on validation (0.3), plus --no-future-met: the model is not given the ERA5
# weather for the forecast hours (the stations' past readings, weather included, are still used).
#   1. seeds 0, 1, 2
#   2. 2025 test: each seed, mean ± std, pooled 3-seed ensemble
#   3. results/v2_delhi_summary.md regenerated (adds the no-future-weather rows)
# Run from anywhere:  nohup setsid bash scripts/pipeline_v2_nofm.sh > runs/v2_nofm_pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }
DATA=data/delhi DO=0.3 DT=do03
[ -f $DATA/values.npy ] || { log "$DATA missing - build it first (docs/V2_PROGRESS.md, step 2)"; exit 1; }

train() {  # out_dir, extra args...
  local out=$1; shift
  log "train $out: $*"
  $PY -u -m aqfm.train --phase main --preset city --data $DATA --out "$out" --device cuda --batch-size 8 \
      --weight-decay 0.05 --patience 3 --compile --steps 12000 --no-future-met "$@" > "$out.txt" 2>&1
  tail -1 "$out.txt"
  grep -q "^done" "$out.txt" || { log "$out did not finish - stopping"; exit 1; }
}

evaluate() {  # out_json, log_file, checkpoints...
  local out=$1 lg=$2; shift 2
  log "evaluate -> $out: $*"
  $PY -u -m aqfm.evaluate --data $DATA --ckpt "$@" --out "$out" --device cuda --seed 0 > "$lg" 2>&1
  tail -2 "$lg"
}

# 1. seeds 0, 1, 2
for s in 0 1 2; do train runs/v2_nofm_s${s}_$DT --dropout $DO --seed $s; done
CK="runs/v2_nofm_s0_$DT/best.pt runs/v2_nofm_s1_$DT/best.pt runs/v2_nofm_s2_$DT/best.pt"

# 2. 2025 test: single seeds, mean ± std, pooled ensemble
for s in 0 1 2; do
  evaluate results/v2_nofm_s${s}_results.json runs/eval_v2_nofm_s${s}.txt runs/v2_nofm_s${s}_$DT/best.pt
done
$PY scripts/seed_stats.py results/v2_nofm_seeds_stats.json \
    results/v2_nofm_s0_results.json results/v2_nofm_s1_results.json results/v2_nofm_s2_results.json
evaluate results/v2_nofm_ens3_results.json runs/eval_v2_nofm_ens3.txt $CK

# 3. summary
$PY scripts/v2_summary.py v2 > results/v2_delhi_summary.md
log "pipeline finished"
