#!/usr/bin/env bash
# Stage F1 (2026-10-16): PatchTST on the exact 2025 protocol, with the same selection procedure AQFM had
# (scripts/pipeline_2026-10-14.sh): dropout 0.1 / 0.2 / 0.3 on seed 0, the lowest 2024-validation score
# wins, then seeds 1 and 2 of the winner. Only the winner's three seeds are tested on 2025, each once;
# then mean ± std over seeds and the 3-seed ensemble (mean forecast). Nothing is chosen on 2025.
# A failed run is logged and the chain carries on.
# Second attempt: every run trains its whole 40k-step schedule (--patience 0) and keeps its best validation
# checkpoint. In the first attempt (20k steps, patience 3; runs patchtst_do*_s*) validation kept falling
# slowly to the end and patience 3 stopped 2 of 5 runs early on a noisy flat stretch, which also decided
# the dropout choice. AQFM keeps patience 3: its validation rises after the peak.
set -uo pipefail
cd /home/phalak/aqfm/AQFM-Honors
PY=../.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
[ -f data/preprocessed/values.npy ] || { log "data/preprocessed missing - stopping"; exit 1; }
# every run must record a real commit: wait until the code and this script are committed
CODE=(aqfm tests 'baselines/*.py' 'deep_baselines/*.py' 'deep_baselines/*.sh' deep_baselines/tests deep_baselines/.gitignore)
if [ -n "$(git status --porcelain -- "${CODE[@]}")" ]; then
  log "waiting for the code to be committed: $(git status --porcelain -- "${CODE[@]}" | tr '\n' ' ')"
  while [ -n "$(git status --porcelain -- "${CODE[@]}")" ]; do sleep 30; done
fi
log "code at commit $(git rev-parse --short HEAD), starting"
OMP_NUM_THREADS=4 $PY -m pytest deep_baselines/tests -q 2>&1 | tail -1
mkdir -p deep_baselines/runs
RUNS=deep_baselines/runs
P=patchtst40k                     # run-name prefix of this attempt
SCHED=(--steps 40000 --patience 0)
R=baselines/results
FAILED=()

train() {  # out_dir, args...
  local out=$1; shift
  log "train $out: $*"
  $PY -u deep_baselines/patchtst.py train --out "$out" "$@" > "$out.txt" 2>&1
  tail -1 "$out.txt"
  grep -q "^done" "$out.txt" || { log "$out did not finish"; FAILED+=("$out"); return 1; }
}

evaluate() {  # results_file, log_file, checkpoints...
  local res=$1 lg=$2; shift 2
  log "evaluate -> $res: $*"
  $PY -u deep_baselines/patchtst.py evaluate --ckpt "$@" --out "$res" > "$lg" 2>&1 \
    || { log "evaluation $res failed"; FAILED+=("$res"); return 1; }
  tail -3 "$lg"
}

# 1. dropout sweep, seed 0 (as AQFM: 0.1 / 0.2 / 0.3)
for d in 1 2 3; do train $RUNS/${P}_do${d}_s0 --seed 0 --dropout 0.$d "${SCHED[@]}"; done

# 2. pick the dropout on 2024 validation only: lowest best validation L1
TAG=$(P=$P $PY - <<'EOF'
import json, os
P = os.environ["P"]
best = {}
for d in (1, 2, 3):
    log = f"deep_baselines/runs/{P}_do{d}_s0/log.jsonl"
    if os.path.exists(log):
        vals = [json.loads(l)["val"]["l1"] for l in open(log) if '"val"' in l]
        if vals:
            best[f"do{d}"] = min(vals)
print(" ".join(f"{k}={v:.5f}" for k, v in best.items()), file=open(f"deep_baselines/runs/{P}_dropout_choice.txt", "w"))
print(min(best, key=best.get) if best else "")
EOF
)
log "validation: $(cat $RUNS/${P}_dropout_choice.txt) -> chosen: ${TAG:-none}"
[ -n "$TAG" ] || { log "no sweep run finished - stopping"; exit 1; }
echo "chosen: $TAG" >> $RUNS/${P}_dropout_choice.txt

# 3. seeds 1 and 2 of the chosen dropout
for s in 1 2; do train $RUNS/${P}_${TAG}_s$s --seed $s --dropout 0.${TAG#do} "${SCHED[@]}"; done

# 4. test the chosen setting's three seeds on 2025, then mean ± std and the ensemble
for s in 0 1 2; do
  [ -f $RUNS/${P}_${TAG}_s$s/best.pt ] &&
    evaluate $R/patchtst_s${s}_results.json $RUNS/eval_${P}_s$s.txt $RUNS/${P}_${TAG}_s$s/best.pt
done
if [ -f $R/patchtst_s0_results.json ] && [ -f $R/patchtst_s1_results.json ] && [ -f $R/patchtst_s2_results.json ]; then
  $PY deep_baselines/seed_stats.py $R/patchtst_seeds_stats.json \
      $R/patchtst_s0_results.json $R/patchtst_s1_results.json $R/patchtst_s2_results.json || FAILED+=(seed_stats)
  evaluate $R/patchtst_ens3_results.json $RUNS/eval_${P}_ens3.txt \
      $RUNS/${P}_${TAG}_s0/best.pt $RUNS/${P}_${TAG}_s1/best.pt $RUNS/${P}_${TAG}_s2/best.pt
fi

log "pipeline finished; failed: ${FAILED[*]:-none}"
