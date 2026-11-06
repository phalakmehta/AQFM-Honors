#!/usr/bin/env bash
# Step 10, Delhi -> Mumbai transfer (docs/AQFM_V2_BUILD_PROMPT.md section 5, step 10; docs/V2_PROGRESS.md step 10).
# Mumbai protocol: data/mumbai (hourly 2023-2024); training/fine-tuning data = the last N months of 2023
# (--train-months N), validation = 2024-01-01 .. 06-30 (picks every setting), test = 2024-07-01 .. 12-31.
# Models without future weather (Mumbai's ERA5 lacks temperature, humidity and sunlight). Fine-tuning starts from the
# Delhi checkpoint of the same seed and keeps its Delhi scaling (A8, C10).
#   STAGE=A  pick the fine-tune setup, seed 0, 3 months: v2-M and v2 x freeze none / encoder / backbone (validation only)
#   STAGE=B  seed 0 curve: fine-tune vs from scratch with 1, 3, 6, 12 months (needs SIZE and FREEZE from stage A)
#   STAGE=C  seeds 1, 2 of the curve; mean +- std; 3-seed ensembles
#   STAGE=D  joint Delhi + Mumbai (12 months) from scratch, seeds 0-2
# Run from anywhere:  STAGE=A nohup setsid bash scripts/pipeline_mumbai_transfer.sh > runs/mumbai_transfer_A.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }
SPLIT="--val-start 2024-01-01 --test-start 2024-07-01"
# Fine-tune recipe from stage A2 (validation only): lr 3e-5 best (1.9196 at step 550; Delhi model untouched 1.9637;
# lr 1e-4 1.9207 at step 150; 1e-5 1.9209 at step 1250). Stage A (lr 1e-4, validation every 250) chose v2 + backbone.
# Budget 6,000 steps (was 1,500 until 2026-11-06 18:10): the 12-month fine-tune peaked at the 1,500-step limit
# (validation 1.8983 at its last step); with 6,000 it reached 1.8782 at step 4,400 (runs/mumbai_ft_v2_backbone_m12_steps6000_s0).
# The 1,500-step runs are kept as runs/mumbai_ft1500_*.
FT="--phase finetune --steps 6000 --warmup 100 --val-every 50 --patience 8 --lr 3e-5"
SCRATCH="--steps 12000 --val-every 250 --patience 8"     # finer than Delhi's 1000/3: small data can overfit early
FT_A="--phase finetune --steps 3000 --warmup 300 --val-every 250 --patience 4"   # stage A as run (lr: finetune default 1e-4)
WIDE="--d-model 256 --n-heads 8 --d-ff 512"

train() {  # out_dir, extra args... (skipped when that run already finished: resume after a stop)
  local out=$1; shift
  if [ -f "$out/best.pt" ] && grep -q "^done" "$out.txt" 2>/dev/null; then log "skip $out (finished before)"; return; fi
  log "train $out: $*"
  $PY -u -m aqfm.train --data data/mumbai $SPLIT --out "$out" --device cuda --batch-size 8 \
      --weight-decay 0.05 --compile "$@" > "$out.txt" 2>&1
  tail -1 "$out.txt"
  grep -q "^done" "$out.txt" || { log "$out did not finish - stopping"; exit 1; }
}
evaluate() {  # out_json, log_file, checkpoints... (skipped when the result exists)
  local out=$1 lg=$2; shift 2
  if [ -f "$out" ]; then log "skip $out (exists)"; return; fi
  log "evaluate -> $out: $*"
  $PY -u -m aqfm.evaluate --data data/mumbai $SPLIT --ckpt "$@" --out "$out" --device cuda --seed 0 > "$lg" 2>&1
  tail -2 "$lg"
}
init() { [ "$1" = v2m ] && echo runs/v2m256_nofm_s$2_do03/best.pt || echo runs/v2_nofm_s$2_do03/best.pt; }
scratch_size() { [ "$1" = v2m ] && echo "$WIDE" || echo ""; }

case "${STAGE:?set STAGE=A|B|C|D}" in
A)
  for size in v2m v2; do
    for fr in none encoder backbone; do
      train runs/mumbai_stageA_ft_${size}_${fr}_m3_s0 $FT_A --init "$(init $size 0)" --freeze $fr --train-months 3 --seed 0
    done
  done
  $PY - <<'EOF'
import glob, re
for f in sorted(glob.glob("runs/mumbai_stageA_ft_*_m3_s0.txt")):
    m = re.search(r"best val total=([\d.]+)", open(f).read())
    print(f"{f}: best validation {m.group(1) if m else '?'}")
EOF
  ;;
A2)
  # Stage A peaked at its first validation (step 250) in every run: calibrate on validation. Reference = the Delhi
  # model itself (50 steps at lr 1e-12 changes nothing); then learning rates with validation every 50 steps.
  train runs/mumbai_ft_v2_backbone_m3_lr0_s0 --phase finetune --steps 50 --warmup 1 --val-every 50 --patience 4 \
      --lr 1e-12 --init "$(init v2 0)" --freeze backbone --train-months 3 --seed 0
  for lr in 1e-4 3e-5 1e-5; do
    train runs/mumbai_ft_v2_backbone_m3_lr${lr}_s0 --phase finetune --steps 1500 --warmup 100 --val-every 50 \
        --patience 8 --lr $lr --init "$(init v2 0)" --freeze backbone --train-months 3 --seed 0
  done
  ;;
B|C)
  : "${SIZE:?set SIZE=v2m|v2 from stage A}" "${FREEZE:?set FREEZE=none|encoder|backbone from stage A}"
  seeds=$([ "$STAGE" = B ] && echo 0 || echo "1 2")
  for s in $seeds; do
    for n in 1 3 6 12; do
      train runs/mumbai_ft_${SIZE}_${FREEZE}_m${n}_s$s $FT --init "$(init $SIZE $s)" --freeze $FREEZE --train-months $n --seed $s
      train runs/mumbai_scratch_${SIZE}_m${n}_s$s --phase main --preset city $(scratch_size $SIZE) --no-future-met \
          --dropout 0.3 $SCRATCH --train-months $n --seed $s
    done
  done
  for s in $seeds; do
    for n in 1 3 6 12; do
      for kind in ft_${SIZE}_${FREEZE} scratch_${SIZE}; do
        evaluate results/mumbai_${kind}_m${n}_s${s}_results.json runs/eval_mumbai_${kind}_m${n}_s${s}.txt \
            runs/mumbai_${kind}_m${n}_s${s}/best.pt
      done
    done
  done
  if [ "$STAGE" = C ]; then
    for n in 1 3 6 12; do
      for kind in ft_${SIZE}_${FREEZE} scratch_${SIZE}; do
        $PY scripts/seed_stats.py results/mumbai_${kind}_m${n}_seeds_stats.json \
            results/mumbai_${kind}_m${n}_s{0,1,2}_results.json
        evaluate results/mumbai_${kind}_m${n}_ens3_results.json runs/eval_mumbai_${kind}_m${n}_ens3.txt \
            runs/mumbai_${kind}_m${n}_s{0,1,2}/best.pt
      done
    done
  fi
  ;;
D)
  : "${SIZE:?set SIZE=v2m|v2}"
  for s in 0 1 2; do
    log "train runs/joint_delhi_mumbai_${SIZE}_s$s"
    $PY -u -m aqfm.train --data data/delhi data/mumbai $SPLIT --out runs/joint_delhi_mumbai_${SIZE}_s$s --device cuda \
        --batch-size 8 --weight-decay 0.05 --compile --phase main --preset city $(scratch_size $SIZE) --no-future-met \
        --dropout 0.3 --steps 12000 --patience 3 --train-months 12 --seed $s > runs/joint_delhi_mumbai_${SIZE}_s$s.txt 2>&1
    grep -q "^done" runs/joint_delhi_mumbai_${SIZE}_s$s.txt || { log "joint s$s did not finish - stopping"; exit 1; }
  done
  ck="runs/joint_delhi_mumbai_${SIZE}_s0/best.pt runs/joint_delhi_mumbai_${SIZE}_s1/best.pt runs/joint_delhi_mumbai_${SIZE}_s2/best.pt"
  evaluate results/mumbai_joint_${SIZE}_ens3_results.json runs/eval_mumbai_joint_${SIZE}_ens3.txt $ck
  log "evaluate Delhi's usual 2025 test with the joint model (data/delhi alone: same windows as every Delhi result)"
  $PY -u -m aqfm.evaluate --data data/delhi --ckpt $ck \
      --out results/delhi_joint_${SIZE}_ens3_results.json --device cuda --seed 0 > runs/eval_delhi_joint_${SIZE}_ens3.txt 2>&1
  tail -2 runs/eval_delhi_joint_${SIZE}_ens3.txt
  ;;
esac
log "pipeline finished (stage $STAGE)"
