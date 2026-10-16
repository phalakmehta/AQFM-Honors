#!/usr/bin/env bash
# Stage F4: E-STGCN (our PyTorch port) on the exact 2025 protocol, with the selection procedure AQFM and
# PatchTST had: 3 values of one hyperparameter on seed 0 (here the learning rate around theirs, 2e-4),
# the lowest 2024-validation score wins, then seeds 1 and 2 of the winner. One model per pollutant (as
# published), each selected on its own. Every run trains its whole schedule and keeps its best validation
# checkpoint (as PatchTST). Only the winners are tested on 2025, each once; then mean ± std over seeds and
# the 3-seed ensemble (mean forecast). Nothing is chosen on 2025. A failed run is logged and the chain
# carries on.
set -uo pipefail
cd /home/phalak/aqfm/AQFM-Honors
PY=../.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
[ -f data/preprocessed/values.npy ] || { log "data/preprocessed missing - stopping"; exit 1; }
# every run must record a real commit: wait until the code and this script are committed
CODE=(aqfm tests 'baselines/*.py' 'deep_baselines/*.py' 'deep_baselines/*.sh' deep_baselines/tests
      deep_baselines/estgcn/tests deep_baselines/.gitignore deep_baselines/estgcn/.gitignore)
if [ -n "$(git status --porcelain -- "${CODE[@]}")" ]; then
  log "waiting for the code to be committed: $(git status --porcelain -- "${CODE[@]}" | tr '\n' ' ')"
  while [ -n "$(git status --porcelain -- "${CODE[@]}")" ]; do sleep 30; done
fi
log "code at commit $(git rev-parse --short HEAD), starting"
OMP_NUM_THREADS=4 $PY -m pytest deep_baselines/estgcn/tests -q 2>&1 | tail -1
RUNS=deep_baselines/estgcn/runs
R=baselines/results
STEPS=20000
mkdir -p $RUNS
FAILED=()

train() {  # out_dir, args...
  local out=$1; shift
  log "train $out: $*"
  $PY -u deep_baselines/estgcn/estgcn.py train --out "$out" --steps $STEPS "$@" > "$out.txt" 2>&1
  tail -1 "$out.txt"
  grep -q "^done" "$out.txt" || { log "$out did not finish"; FAILED+=("$out"); return 1; }
}

evaluate() {  # results_file, log_file, checkpoints...
  local res=$1 lg=$2; shift 2
  log "evaluate -> $res: $*"
  $PY -u deep_baselines/estgcn/estgcn.py evaluate --ckpt "$@" --out "$res" > "$lg" 2>&1 \
    || { log "evaluation $res failed"; FAILED+=("$res"); return 1; }
  tail -3 "$lg"
}

declare -A TAG
for pol in pm25 pm10; do
  target=$([ $pol = pm25 ] && echo PM2.5 || echo PM10)
  # 1. learning-rate sweep, seed 0: 1e-4, 2e-4 (theirs), 4e-4
  for l in 1 2 4; do train $RUNS/${pol}_lr${l}_s0 --target $target --seed 0 --lr ${l}e-4; done
  # 2. pick on 2024 validation only: lowest best validation MSE
  TAG[$pol]=$(POL=$pol $PY - <<'EOF'
import json, os
pol = os.environ["POL"]
best = {}
for l in (1, 2, 4):
    log = f"deep_baselines/estgcn/runs/{pol}_lr{l}_s0/log.jsonl"
    if os.path.exists(log):
        vals = [json.loads(x)["val"]["mse"] for x in open(log) if '"val"' in x]
        if vals:
            best[f"lr{l}"] = min(vals)
print(" ".join(f"{k}={v:.5f}" for k, v in best.items()), file=open(f"deep_baselines/estgcn/runs/{pol}_lr_choice.txt", "w"))
print(min(best, key=best.get) if best else "")
EOF
)
  log "$pol validation: $(cat $RUNS/${pol}_lr_choice.txt) -> chosen: ${TAG[$pol]:-none}"
  [ -n "${TAG[$pol]}" ] || { log "no $pol sweep run finished - stopping"; exit 1; }
  echo "chosen: ${TAG[$pol]}" >> $RUNS/${pol}_lr_choice.txt
  # 3. seeds 1 and 2 of the chosen learning rate
  for s in 1 2; do train $RUNS/${pol}_${TAG[$pol]}_s$s --target $target --seed $s --lr ${TAG[$pol]#lr}e-4; done
done

# 4. test each seed (PM2.5 and PM10 model of that seed) on 2025, then mean ± std and the ensemble
ck() { echo $RUNS/pm25_${TAG[pm25]}_s$1/best.pt $RUNS/pm10_${TAG[pm10]}_s$1/best.pt; }
for s in 0 1 2; do
  evaluate $R/estgcn_s${s}_results.json $RUNS/eval_estgcn_s$s.txt $(ck $s)
done
if [ -f $R/estgcn_s0_results.json ] && [ -f $R/estgcn_s1_results.json ] && [ -f $R/estgcn_s2_results.json ]; then
  $PY deep_baselines/seed_stats.py $R/estgcn_seeds_stats.json \
      $R/estgcn_s0_results.json $R/estgcn_s1_results.json $R/estgcn_s2_results.json || FAILED+=(seed_stats)
  evaluate $R/estgcn_ens3_results.json $RUNS/eval_estgcn_ens3.txt $(ck 0) $(ck 1) $(ck 2)
fi

log "pipeline finished; failed: ${FAILED[*]:-none}"
