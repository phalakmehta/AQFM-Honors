#!/usr/bin/env bash
# Experiments the paper still needs (plan: HANDOFF.md section 8 and RUNBOOK.md). Needs the evaluation options of
# 2026-11-06 (--targets, --save-forecasts, --test-end, --met-noise) and train.py --v1-spatial.
#   STAGE=SAVE   Delhi: the main ensembles again with every forecast saved and Ozone/NOx/NO2 scored; the baselines
#                (classical, XGBoost, XGBoost + future ERA5) on data/delhi with forecasts saved; same-point
#                comparisons with significance (scripts/forecast_stats.py).                          ~2.5 h GPU + CPU
#   STAGE=ABL    Ablations on v2-M (seed 0): no BLH, no anchor, gap-filling only.                 ~1.5 h GPU
#                (No spatial ablations: the spatial part is there so the model works for any city's station
#                layout, not to improve accuracy - user, 2026-11-07.)
#   STAGE=FOLD   A second test year: v2-M trained on 2018-2022, picked on 2023, tested on 2024 (3 seeds, with and
#                without future weather) + the baselines on the same split.                          ~4 h GPU
#   STAGE=MUMBAI after the Mumbai transfer chain: final Mumbai models + baselines with forecasts saved; same-point
#                comparisons per data amount (fine-tuned vs scratch vs XGBoost vs zero-shot vs persistence). ~1.5 h
#   STAGE=NOISE  Realistic weather: v2-M (with future weather) ensemble with the future weather perturbed,
#                noise sd 0.25 / 0.5 / 1.0 (scaled units) x sqrt(lead in days).                     ~45 min GPU
# Run from anywhere:  STAGE=SAVE nohup setsid bash scripts/pipeline_paper_extras.sh > runs/paper_SAVE.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }
grep -q -- "--save-forecasts" aqfm/evaluate.py || { log "aqfm/evaluate.py has no --save-forecasts: apply the 2026-11-06 patch first"; exit 1; }
mkdir -p results/paper results/forecasts
ALL="--targets PM2.5 PM10 Ozone NOx NO2"
WIDE="--d-model 256 --n-heads 8 --d-ff 512"
ens() { echo runs/$1_s0_$2/best.pt runs/$1_s1_$2/best.pt runs/$1_s2_$2/best.pt; }

evaluate() {  # name, data, extra args, checkpoints...
  local name=$1 data=$2 extra=$3; shift 3
  log "evaluate $name"
  $PY -u -m aqfm.evaluate --data $data $extra --ckpt "$@" --out results/paper/$name.json \
      --save-forecasts results/forecasts/$name.npz --device cuda --seed 0 > runs/eval_paper_$name.txt 2>&1
  tail -6 runs/eval_paper_$name.txt
}
train() {  # out_dir, extra args...
  local out=$1; shift
  log "train $out: $*"
  $PY -u -m aqfm.train --phase main --preset city --data data/delhi --out "$out" --device cuda --batch-size 8 \
      --weight-decay 0.05 --compile --dropout 0.3 --steps 12000 --patience 3 "$@" > "$out.txt" 2>&1
  tail -1 "$out.txt"
  grep -q "^done" "$out.txt" || { log "$out did not finish - stopping"; exit 1; }
}
stats() {  # out, target, then NAME FILE pairs
  local out=$1 tgt=$2; shift 2
  local a=()
  while [ $# -gt 0 ]; do a+=(--model "$1" "$2"); shift 2; done
  $PY scripts/forecast_stats.py --target $tgt "${a[@]}" --out results/paper/$out.json | tail -5
}
baselines() {  # tag, data, extra args for the split
  local tag=$1 data=$2 extra=$3
  log "baselines $tag"
  nice $PY scripts/city_baselines.py --data $data $extra $ALL --save-forecasts results/forecasts/${tag}_classical.npz \
      --out results/paper/${tag}_classical.json > runs/paper_${tag}_classical.txt 2>&1
  for fm in "" "--future-met"; do
    local n=${tag}_xgboost${fm:+_fm}
    nice $PY scripts/city_xgboost.py --data $data $extra $fm $ALL --save-forecasts results/forecasts/$n.npz \
        --out results/paper/$n.json > runs/paper_$n.txt 2>&1
  done
}

case "${STAGE:?set STAGE=SAVE|ABL|FOLD|MUMBAI|NOISE}" in
SAVE)
  evaluate delhi_v2m256_ens3 data/delhi "$ALL" $(ens v2m256 do03)
  evaluate delhi_v2m256_nofm_ens3 data/delhi "$ALL" $(ens v2m256_nofm do03)
  evaluate delhi_v2_ens3 data/delhi "$ALL" $(ens v2 do03)
  evaluate delhi_v2_nofm_ens3 data/delhi "$ALL" $(ens v2_nofm do03)
  evaluate delhi_v1_ens3 data/preprocessed "$ALL" $(ens main do3)
  baselines delhi data/delhi "--test-start 2025-01-01"
  F=results/forecasts
  for c in PM2.5 PM10 Ozone NOx NO2; do
    stats delhi_${c}_with_weather $c "AQFM v2-M" $F/delhi_v2m256_ens3.npz "AQFM v2" $F/delhi_v2_ens3.npz \
        "XGBoost + future ERA5" $F/delhi_xgboost_fm.npz:xgboost Persistence $F/delhi_classical.npz:persistence
    stats delhi_${c}_no_weather $c "AQFM v2-M" $F/delhi_v2m256_nofm_ens3.npz "AQFM v2" $F/delhi_v2_nofm_ens3.npz \
        XGBoost $F/delhi_xgboost.npz:xgboost Persistence $F/delhi_classical.npz:persistence
  done
  ;;
ABL)
  # Run on 2026-11-06 before the spatial ablations were dropped, with this same loop: "nospatial --no-spatial"
  # (results/paper/delhi_v2m256_abl_nospatial_s0.json).
  for ab in "noblh --no-blh" "nolast --no-last-obs" "gapfillonly --random-masking"; do
    set -- $ab
    train runs/v2m256_abl_$1_s0 --seed 0 $WIDE "${@:2}"
    evaluate delhi_v2m256_abl_$1_s0 data/delhi "" runs/v2m256_abl_$1_s0/best.pt
  done
  ;;
FOLD)
  F24="--val-year 2023 --test-year 2024"
  for s in 0 1 2; do
    train runs/v2m256_fold24_s${s}_do03 --seed $s $WIDE $F24
    train runs/v2m256_fold24_nofm_s${s}_do03 --seed $s $WIDE $F24 --no-future-met
  done
  evaluate delhi_fold24_v2m256_ens3 data/delhi "$F24 --test-end 2025-01-01" $(ens v2m256_fold24 do03)
  evaluate delhi_fold24_v2m256_nofm_ens3 data/delhi "$F24 --test-end 2025-01-01" $(ens v2m256_fold24_nofm do03)
  baselines delhi_fold24 data/delhi "--test-start 2024-01-01 --test-end 2025-01-01"
  F=results/forecasts
  stats delhi_fold24_PM2.5_with_weather PM2.5 "AQFM v2-M" $F/delhi_fold24_v2m256_ens3.npz \
      "XGBoost + future ERA5" $F/delhi_fold24_xgboost_fm.npz:xgboost Persistence $F/delhi_fold24_classical.npz:persistence
  stats delhi_fold24_PM2.5_no_weather PM2.5 "AQFM v2-M" $F/delhi_fold24_v2m256_nofm_ens3.npz \
      XGBoost $F/delhi_fold24_xgboost.npz:xgboost Persistence $F/delhi_fold24_classical.npz:persistence
  ;;
MUMBAI)
  # after scripts/pipeline_mumbai_transfer.sh stages B-D: the final Mumbai models with forecasts saved (Mumbai has no
  # NO readings, so its NO2 comes from the Delhi-learned NO share: report with that caveat)
  M="--val-start 2024-01-01 --test-start 2024-07-01"
  evaluate mumbai_zeroshot_v2_nofm_ens3 data/mumbai "$M $ALL" $(ens v2_nofm do03)
  for n in 1 3 6 12; do
    evaluate mumbai_ft_m${n}_ens3 data/mumbai "$M $ALL" runs/mumbai_ft_v2_backbone_m${n}_s{0,1,2}/best.pt
    evaluate mumbai_scratch_m${n}_ens3 data/mumbai "$M $ALL" runs/mumbai_scratch_v2_m${n}_s{0,1,2}/best.pt
  done
  evaluate mumbai_joint_ens3 data/mumbai "$M $ALL" runs/joint_delhi_mumbai_v2_s{0,1,2}/best.pt
  nice $PY scripts/city_baselines.py --data data/mumbai --test-start 2024-07-01 $ALL \
      --save-forecasts results/forecasts/mumbai_classical.npz --out results/paper/mumbai_classical.json > runs/paper_mumbai_classical.txt 2>&1
  for n in 1 3 6 12; do
    case $n in 1) from=2023-12-01;; 3) from=2023-10-01;; 6) from=2023-07-01;; 12) from=2023-01-01;; esac
    nice $PY scripts/city_xgboost.py --data data/mumbai --hourly --test-start 2024-07-01 --train-from $from \
        --train-to 2024-01-01 $ALL --save-forecasts results/forecasts/mumbai_xgboost_m$n.npz \
        --out results/paper/mumbai_xgboost_m$n.json > runs/paper_mumbai_xgboost_m$n.txt 2>&1
  done
  F=results/forecasts
  for c in PM2.5 PM10 Ozone NOx NO2; do
    for n in 1 3 6 12; do
      stats mumbai_${c}_m$n $c "AQFM fine-tuned" $F/mumbai_ft_m${n}_ens3.npz "AQFM from scratch" $F/mumbai_scratch_m${n}_ens3.npz \
          "XGBoost (same months)" $F/mumbai_xgboost_m$n.npz:xgboost "AQFM zero-shot" $F/mumbai_zeroshot_v2_nofm_ens3.npz \
          Persistence $F/mumbai_classical.npz:persistence
    done
  done
  ;;
NOISE)
  for sd in 0.25 0.5 1.0; do
    evaluate delhi_v2m256_ens3_metnoise$sd data/delhi "--met-noise $sd --no-imputation" $(ens v2m256 do03)
  done
  ;;
esac
log "pipeline finished (stage $STAGE)"
