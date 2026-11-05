#!/usr/bin/env bash
# AQFM v2, build-prompt step 9: full Delhi training with the city preset (1.17M parameters) on
# data/delhi, with v1's settings so the two compare fairly (scripts/pipeline_2026-10-14.sh):
# 12,000 steps, batch 8, warmup 2,000 (default), 512 validation windows (default), weight decay
# 0.05, patience 3, --compile. Selection uses the 2024 VALIDATION score only.
#   1. dropout 0.15 (guide) vs 0.3 (v1), seed 0
#   2. seeds 1, 2 with the chosen dropout; all 37 stations, no hold-out (compares with v1)
#   3. 2025 test: each seed, mean ± std, pooled 3-seed ensemble
#   4. one separate run with the 4-station hold-out: "new station that has its own history",
#      "virtual sensing (no readings at the station)", and the all-station model on the same 4
#   5. errors by distance to the nearest neighbour (0-5, 5-15, > 15 km): v2 and v1 ensembles
#   6. results/v2_delhi_summary.md
# Run from anywhere:  nohup setsid bash scripts/pipeline_v2_delhi.sh > runs/v2_pipeline.log 2>&1 &
# DRY=1: a few steps on data/preprocessed into runs/v2dry_*, results/v2dry_* (tests the chain only).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }

if [ "${DRY:-0}" = 1 ]; then
  TAG=v2dry DATA=data/preprocessed HOLD=4,7,32,35
  TR="--steps 30 --warmup 10 --val-every 15 --val-windows 16"
  EV="--limit-origins 2 --samples 20"
else
  TAG=v2 DATA=data/delhi HOLD=delhi:4
  TR="--steps 12000"
  EV=""
  [ -f $DATA/values.npy ] || { log "$DATA missing - build it first (docs/V2_PROGRESS.md, step 2)"; exit 1; }
  $PY - <<'EOF' || { log "data/delhi station order differs from data/preprocessed - stopping"; exit 1; }
import json, sys
a, b = (json.load(open(f"data/{d}/preprocessing_report.json"))["station_order"] for d in ("delhi", "preprocessed"))
sys.exit(0 if a == b else 1)
EOF
fi

pick() {  # print the checkpoint with the lowest validation selection score (same formula as train.py)
  $PY - "$@" <<'EOF'
import sys, torch
best = None
for p in sys.argv[1:]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    v, c = ck["val"], ck["cfg"]
    s = c["w_nll"] * v["nll_forecast"] + c["w_crps"] * v["crps_forecast"]
    print(f"  candidate {p}: step {ck['step']} dropout {c['dropout']} score {s:.4f}", file=sys.stderr)
    if best is None or s < best[0]:
        best = (s, p)
print(best[1])
EOF
}

train() {  # out_dir, extra args...
  local out=$1; shift
  log "train $out: $*"
  $PY -u -m aqfm.train --phase main --preset city --data $DATA --out "$out" --device cuda --batch-size 8 \
      --weight-decay 0.05 --patience 3 --compile $TR "$@" > "$out.txt" 2>&1
  tail -1 "$out.txt"
  grep -q "^done" "$out.txt" || { log "$out did not finish - stopping"; exit 1; }
}

evaluate() {  # out_json, log_file, data, extra args (before --ckpt), checkpoints...
  local out=$1 lg=$2 data=$3 extra=$4; shift 4
  log "evaluate -> $out: $extra $*"
  $PY -u -m aqfm.evaluate --data $data $extra --ckpt "$@" --out "$out" --device cuda --seed 0 $EV > "$lg" 2>&1
  tail -2 "$lg"
}

# 1. dropout, seed 0
train runs/${TAG}_s0_do015 --dropout 0.15 --seed 0
train runs/${TAG}_s0_do03 --dropout 0.3 --seed 0
WIN=$(pick runs/${TAG}_s0_do015/best.pt runs/${TAG}_s0_do03/best.pt)
case $WIN in
  runs/${TAG}_s0_do015/*) DT=do015 DO=0.15 ;;
  runs/${TAG}_s0_do03/*)  DT=do03  DO=0.3 ;;
esac
log "dropout chosen on validation: $DO ($WIN)"

# 2. seeds 1 and 2
for s in 1 2; do train runs/${TAG}_s${s}_$DT --dropout $DO --seed $s; done
CK="$WIN runs/${TAG}_s1_$DT/best.pt runs/${TAG}_s2_$DT/best.pt"

# 3. 2025 test: single seeds, mean ± std, pooled ensemble
evaluate results/${TAG}_s0_results.json runs/eval_${TAG}_s0.txt $DATA "" $WIN
for s in 1 2; do
  evaluate results/${TAG}_s${s}_results.json runs/eval_${TAG}_s${s}.txt $DATA "" runs/${TAG}_s${s}_$DT/best.pt
done
$PY scripts/seed_stats.py results/${TAG}_seeds_stats.json \
    results/${TAG}_s0_results.json results/${TAG}_s1_results.json results/${TAG}_s2_results.json
evaluate results/${TAG}_ens3_results.json runs/eval_${TAG}_ens3.txt $DATA "" $CK

# 4. unseen stations: separate run with the 4-station hold-out
train runs/${TAG}_holdout4_s0_$DT --dropout $DO --seed 0 --holdout $HOLD
HK=runs/${TAG}_holdout4_s0_$DT/best.pt
HELD=$($PY -c "import torch; print(','.join(map(str, torch.load('$HK', map_location='cpu', weights_only=False)['held_out_stations'])))")
log "hold-out stations: $HELD"
evaluate results/${TAG}_holdout4_newstation_results.json runs/eval_${TAG}_holdout4_newstation.txt $DATA \
    "--stations held_out" $HK
evaluate results/${TAG}_holdout4_virtual_results.json runs/eval_${TAG}_holdout4_virtual.txt $DATA \
    "--stations held_out --virtual-sensing" $HK
evaluate results/${TAG}_s0_on_holdout4_results.json runs/eval_${TAG}_s0_on_holdout4.txt $DATA \
    "--stations $HELD --no-imputation" $WIN

# 5. by distance to the nearest neighbour (bands from this data's coordinates; same station order as v1)
BANDS=$($PY - $DATA <<'EOF'
import sys, numpy as np
sys.path.insert(0, ".")
from aqfm.data import haversine_km
c = np.load(f"{sys.argv[1]}/coords.npy").astype(np.float64)
d = haversine_km(c, c) + np.eye(len(c)) * 1e9
nn = d.min(1)
for name, lo, hi in (("0_5", 0, 5), ("5_15", 5, 15), ("15p", 15, np.inf)):
    idx = np.flatnonzero((nn >= lo) & (nn < hi))
    if len(idx):
        print(f"{name}:{','.join(map(str, idx))}")
EOF
)
V1CK="runs/main_s0_do3/best.pt runs/main_s1_do3/best.pt runs/main_s2_do3/best.pt"
for b in $BANDS; do
  name=${b%%:*} idx=${b#*:}
  log "nearest-neighbour band $name km: stations $idx"
  evaluate results/${TAG}_ens3_nnband_${name}_results.json runs/eval_${TAG}_ens3_nnband_${name}.txt $DATA \
      "--stations $idx --no-imputation" $CK
  evaluate results/${TAG}_v1ens3_nnband_${name}_results.json runs/eval_${TAG}_v1ens3_nnband_${name}.txt \
      data/preprocessed "--stations $idx --no-imputation" $V1CK
done

# 6. summary
$PY scripts/v2_summary.py $TAG > results/${TAG}_delhi_summary.md
log "pipeline finished"
