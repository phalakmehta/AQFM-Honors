# Plan: O3 and NO2 results for every model (not started; written 2026-10-16)

Goal: the same 2025 comparison we have for PM2.5 and PM10, for ozone (O3) and NO2, for every model,
with the same test points, seeds and selection rules. Estimated total: about 6 h of runs.

## What already exists

- **AQFM** already predicts both: ozone is one of its 8 outputs, and NO2 is rebuilt from two outputs
  (NOx × (1 − NO fraction), `aqfm/physical.py: reconstruct_no_no2`). So AQFM needs **no retraining**,
  only scoring of the existing checkpoints.
- All baselines (classical, XGBoost, TTM, PatchTST, E-STGCN) were run for PM2.5 and PM10 only, so they
  need new runs.
- Test data is there: 2025 coverage O3 0.897, NO2 0.931; scored points at 24 h: O3 10,645, NO2 10,822.

## What to build (new folder `o3_no2/`; no existing code changes)

| File | What it does |
|---|---|
| `eval_aqfm.py` | Scores existing AQFM checkpoints for O3 and NO2. Copies the forecast loop of `aqfm/evaluate.py` with the same seed and order, so its PM2.5/PM10 numbers must equal the registered results: a built-in check (`--check-against <registered result>`) |
| `run_baselines.py` | Runs the existing `baselines/run_baselines_{classical,xgboost,ttm}.py` unchanged, with the target channels switched to O3 (5) and NO2 (2) before import; writes new result files |
| `patchtst.py`, `estgcn.py` | Thin wrappers around `deep_baselines/` with the channels switched. E-STGCN thresholds: NO2 80 µg/m³ (as in their paper), O3 100 µg/m³ (NAAQS 8-h; their paper has no ozone) |
| `seed_stats.py` | Mean ± std over seeds for any pollutant (the existing ones print PM only) |
| `make_tables.py` | Writes `results/TABLES_O3_NO2.md` (numbers are quoted only from there) |
| `pipeline_o3no2.sh` | The chain (waits for a commit, runs the tests, then everything below) |
| `tests/` | Wrappers pick the right channels (run in subprocesses); NO2 rebuild formula; on synthetic data `eval_aqfm.py` reproduces `aqfm/evaluate.py`'s PM numbers exactly; tiny end-to-end runs |

## Runs

| Step | Runs | Time (estimate) |
|---|---|---|
| AQFM: final setup (3 seeds + ensemble) and no future weather (3 seeds + ensemble) | 8 evaluations, no training | ~1.5 h GPU |
| Persistence, seasonal naive, climatology | 1 | ~10 min CPU |
| XGBoost and XGBoost + future ERA5 | 2 × 8 models | ~1–2 h CPU (in parallel with the GPU steps) |
| IBM TTM zero-shot | 1 (one-time model download from Hugging Face; reachable, checked 2026-10-16) | ~15–60 min |
| PatchTST | dropout 0.1 / 0.2 / 0.3 on seed 0, then seeds 1, 2 of the winner, whole 40k schedule | ~1 h GPU |
| E-STGCN | per pollutant: learning rate 1e-4 / 2e-4 / 4e-4 on seed 0, then seeds 1, 2 (10 runs) | ~3 h GPU |
| **Total** | | **~6 h wall-clock** |

## Decisions (already made)

- **Metrics:** MAE and RMSE at 1 / 6 / 24 / 72 h for every model; AQFM also CRPS and 90 % / 95 % coverage.
  No AQI bands (our code has them for PM2.5 only). Possible later: MAE above the standard (NO2 > 80, O3 > 100).
- **XGBoost's extra pollutant feature:** the existing code uses PM2.5 at t − 1 for any target other than
  PM2.5; kept as is (unchanged code).
- **Result files:** `results/aqfm_*_o3no2_results.json`, `baselines/results/*_o3no2_results.json`; all
  registered in `scripts/manifest.json`; models (PatchTST, E-STGCN) to Drive like the others.

## Caveats to report

- AQFM's NO2 assumes NO + NO2 = NOx (in ppb). At 8 stations the sensors break this (ratio 0.66–0.71, mostly
  IITM stations and Rohini), so AQFM's NO2 may look worse there: report those stations separately.
- `check_submission.py` checks only `results/TABLES.md` for freshness; `TABLES_O3_NO2.md` needs a manual
  regenerate-and-compare (or a one-line addition to the check, if allowed).

## Before starting

1. `git push origin main`; GPU free; data in `data/preprocessed/`.
2. Build the folder above, run its tests, show the diff, get the go.
3. Launch with `nohup setsid`; while it runs, do not create or edit code files under `aqfm/`, `baselines/`,
   `deep_baselines/` or `o3_no2/` (they would mark the runs as uncommitted code).
