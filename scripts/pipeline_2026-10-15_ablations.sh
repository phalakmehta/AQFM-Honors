#!/usr/bin/env bash
# Stage E ablations of the final setup, seed 0 each, plus no-future-met seeds 1 and 2 (2026-10-15).
# Same recipe as runs/main_s0_do3 (scripts/pipeline_2026-10-14.sh) with ONE switch changed per run.
# Nothing is chosen on 2025: each run keeps its best 2024-validation checkpoint and is tested once.
# Compare each ablation with results/aqfm_do3_s0_results.json (same seed). A failed run is logged
# and skipped; the chain carries on with the next one.
# E7 (no latest-value anchor) is not re-run: run 1 (runs/main_s0) vs the dropout-0.1 anchor run
# (runs/main_s0_anchor) differ only in the anchor and the planned schedule length (PROJECT_CONTEXT §6).
set -uo pipefail
cd /home/phalak/aqfm/AQFM-Honors
PY=../.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
[ -f data/preprocessed/values.npy ] || { log "data/preprocessed missing - stopping"; exit 1; }
# every run must record a real commit: wait until the code and this script are committed
CODE=(aqfm tests 'baselines/*.py' scripts/pipeline_2026-10-15_ablations.sh)
if [ -n "$(git status --porcelain -- "${CODE[@]}")" ]; then
  log "waiting for the code to be committed: $(git status --porcelain -- "${CODE[@]}" | tr '\n' ' ')"
  while [ -n "$(git status --porcelain -- "${CODE[@]}")" ]; do sleep 30; done
fi
log "code at commit $(git rev-parse --short HEAD), starting"
FAILED=()

train() {  # out_dir, extra args...  (final recipe + the extra switches)
  local out=$1; shift
  log "train $out: $*"
  $PY -u -m aqfm.train --phase main --out "$out" --device cuda --batch-size 8 --steps 12000 \
      --weight-decay 0.05 --patience 3 --dropout 0.3 --compile "$@" > "$out.txt" 2>&1
  tail -1 "$out.txt"
  grep -q "^done" "$out.txt" || { log "$out did not finish"; FAILED+=("$out"); return 1; }
}

evaluate() {  # out_json, log_file, checkpoints...
  local out=$1 lg=$2; shift 2
  log "evaluate -> $out: $*"
  $PY -u -m aqfm.evaluate --ckpt "$@" --out "$out" --device cuda --seed 0 > "$lg" 2>&1 \
    || { log "evaluation $out failed"; FAILED+=("$out"); return 1; }
  tail -3 "$lg"
}

ablation() {  # tag, switches...: train seed 0, then test it on 2025
  local tag=$1; shift
  train runs/main_s0_do3_$tag --seed 0 "$@" &&
    evaluate results/aqfm_do3_${tag}_s0_results.json runs/eval_do3_${tag}_s0.txt runs/main_s0_do3_$tag/best.pt
}

ablation nospatial   --no-spatial              # E3: no cross-station attention
ablation nowind      --spatial-bias distance   # E3: spatial attention by distance only (no wind)
ablation pm25direct  --pm25-direct             # E4: PM2.5 predicted directly, not PM10 x fraction
ablation gapfillonly --random-masking          # E5: trained to fill gaps only, no forecast targets
ablation noblh       --no-blh                  # E2: no boundary-layer height

# no-future-met seeds 1 and 2 (seed 0 = runs/main_s0_do3_nofm), then mean ± std and the 3-seed ensemble
for s in 1 2; do
  train runs/main_s${s}_do3_nofm --seed $s --no-future-met &&
    evaluate results/aqfm_do3_nofm_s${s}_results.json runs/eval_do3_nofm_s${s}.txt runs/main_s${s}_do3_nofm/best.pt
done
if [ -f results/aqfm_do3_nofm_s1_results.json ] && [ -f results/aqfm_do3_nofm_s2_results.json ]; then
  $PY scripts/seed_stats.py results/aqfm_do3_nofm_seeds_stats.json results/aqfm_do3_nofm_s0_results.json \
      results/aqfm_do3_nofm_s1_results.json results/aqfm_do3_nofm_s2_results.json
  evaluate results/aqfm_do3_nofm_ens3_results.json runs/eval_do3_nofm_ens3.txt \
      runs/main_s0_do3_nofm/best.pt runs/main_s1_do3_nofm/best.pt runs/main_s2_do3_nofm/best.pt
fi

log "pipeline finished; failed: ${FAILED[*]:-none}"
