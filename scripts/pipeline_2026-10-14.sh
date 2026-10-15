#!/usr/bin/env bash
# Unattended chain (2026-10-14): dropout sweep -> seeds 1,2 -> evaluations + 3-seed ensemble
# -> no-future-met ablation. Selection uses the 2024 VALIDATION score only; 2025 test
# results never choose anything.
# Record of the exact chain that produced results/aqfm_do3_* (only change since: seed_stats.py moved to scripts/).
set -euo pipefail
cd /home/phalak/aqfm/AQFM-Honors
PY=../.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
[ -f data/preprocessed/values.npy ] || { log "data/preprocessed missing - stopping"; exit 1; }

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

train() {  # out_dir, extra args... (same schedule as runs/main_s0_anchor, patience 3)
  local out=$1; shift
  log "train $out: $*"
  $PY -u -m aqfm.train --phase main --out "$out" --device cuda --batch-size 8 --steps 12000 \
      --weight-decay 0.05 --patience 3 --compile "$@" > "$out.txt" 2>&1
  tail -1 "$out.txt"
  grep -q "^done" "$out.txt" || { log "$out did not finish - stopping"; exit 1; }
}

evaluate() {  # out_json, log_file, checkpoints...
  local out=$1 lg=$2; shift 2
  log "evaluate -> $out: $*"
  $PY -u -m aqfm.evaluate --ckpt "$@" --out "$out" --device cuda --seed 0 > "$lg" 2>&1
  tail -3 "$lg"
}

# 1. dropout sweep, seed 0 (0.1 = runs/main_s0_anchor, already trained)
train runs/main_s0_do2 --dropout 0.2 --seed 0
train runs/main_s0_do3 --dropout 0.3 --seed 0
WIN=$(pick runs/main_s0_anchor/best.pt runs/main_s0_do2/best.pt runs/main_s0_do3/best.pt)
case $WIN in
  runs/main_s0_anchor/*) TAG=anchor DO=0.1 ;;
  runs/main_s0_do2/*)    TAG=do2    DO=0.2 ;;
  runs/main_s0_do3/*)    TAG=do3    DO=0.3 ;;
esac
log "final setup chosen on validation: dropout $DO ($WIN)"

# 2. seeds 1 and 2 of the final setup
for s in 1 2; do train runs/main_s${s}_$TAG --dropout $DO --seed $s; done

# 3. 2025 test: each seed alone (mean +- std), then the pooled 3-seed ensemble
[ $TAG = anchor ] || evaluate results/aqfm_${TAG}_s0_results.json runs/eval_${TAG}_s0.txt "$WIN"
for s in 1 2; do
  evaluate results/aqfm_${TAG}_s${s}_results.json runs/eval_${TAG}_s${s}.txt runs/main_s${s}_$TAG/best.pt
done
$PY scripts/seed_stats.py results/aqfm_${TAG}_seeds_stats.json \
    results/aqfm_${TAG}_s0_results.json results/aqfm_${TAG}_s1_results.json results/aqfm_${TAG}_s2_results.json
evaluate results/aqfm_${TAG}_ens3_results.json runs/eval_${TAG}_ens3.txt \
    "$WIN" runs/main_s1_$TAG/best.pt runs/main_s2_$TAG/best.pt

# 4. no-future-met ablation of the final setup (compare with plain XGB)
train runs/main_s0_${TAG}_nofm --no-future-met --dropout $DO --seed 0
evaluate results/aqfm_${TAG}_nofm_s0_results.json runs/eval_${TAG}_nofm_s0.txt runs/main_s0_${TAG}_nofm/best.pt

log "pipeline finished"
