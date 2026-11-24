# Submission checklist

Everything the paper needs, where it lives, and whether it is done. Update this after every finished step.
**One command checks it all:** `python scripts/check_submission.py` (quick, ~1 min) — must print `ALL OK`.
Every result's command, models and checksums: `scripts/manifest.json`. Every number: `results/TABLES.md` (generated).

Where: **GitHub** = github.com/phalakmehta/AQFM-Honors · **Drive** = Google Drive (`drive_upload_2026-10-14/` and
`Foundation Model(Mainak Sir)/`) · **machine** = the GPU machine only (`/home/phalak/aqfm/`).

Last updated: 2026-11-07 (v2 section).

## Code

| Item | Where | Status |
|---|---|---|
| Model `aqfm/`, tests (46 pass), data pipeline, baselines | GitHub | ✅ |
| `requirements.txt` (pins) + `scripts/environment_gpu.txt` (exact GPU environment) | GitHub | ✅ |
| `scripts/check_submission.py`, `manifest.json`, `make_tables.py`, `seed_stats.py`, `pipeline_2026-10-14.sh`, `pipeline_2026-10-15_ablations.sh` | GitHub | ✅ |
| `aqfm/evaluate.py` stores its options (`res["args"]`); `.gitignore` tracks `runs/` logs | GitHub | ✅ |
| Deep baselines `deep_baselines/` (own folder; imports `aqfm/`, `baselines/`, changes neither): PatchTST, 8 tests (`python -m pytest deep_baselines/tests -q`; not run by `check_submission.py`) | machine | ✅ `de1bffb`, `889e82d`; results ⬜ commit, ⬜ push |
| E-STGCN re-implementation `deep_baselines/estgcn/` (own folder, 10 tests; `python -m pytest deep_baselines/estgcn/tests -q`) | machine | ✅ `16ed23b`; ⬜ push |

## Data

| Item | Where | Status |
|---|---|---|
| Raw CPCB CSVs (`yearly_data.zip`) and raw ERA5 (`ERA 5/`) | Drive `Foundation Model(Mainak Sir)/` | ✅ |
| `data/preprocessed/` (the arrays every model used; sha256 in the manifest) | Drive `preproccesed/` + machine | ✅ |
| `data/stations.csv` | GitHub | ✅ |

## Models (best checkpoint on 2024 validation; sha256 in the manifest and `models/SHA256SUMS.txt`)

| Item | Where | Status |
|---|---|---|
| Final setup seeds 0/1/2, no-future-met seed 0 (`aqfm_do3_*.pt`) | Drive `models/` + machine | ✅ |
| History: anchor dropout 0.1, run 1, run 2 (`aqfm_anchor_do1_s0`, `aqfm_run1_s0`, `aqfm_run2_wd_anneal_s0`) | machine + `drive_upload_2026-10-14/models` | ⬜ upload to Drive |
| Run 2 main phase (only needed to retrain run 2's anneal) | machine | ✅ (not uploaded on purpose) |
| Stage E ablations (`aqfm_do3_{nospatial,nowind,pm25direct,gapfillonly,noblh}_s0`), no-future-met seeds 1/2 | machine + `drive_upload_2026-10-14/models` | ✅ trained; ⬜ upload to Drive |
| PatchTST seeds 0/1/2 (`patchtst_s{0,1,2}`) | machine + `drive_upload_2026-10-14/models` | ✅ trained; ⬜ upload to Drive |
| E-STGCN PM2.5 and PM10 models, seeds 0/1/2 (`estgcn_{pm25,pm10}_s{0,1,2}`) | machine + `drive_upload_2026-10-14/models` | ✅ trained; ⬜ upload to Drive |

## Results

| Item | Where | Status |
|---|---|---|
| Baselines: classical, XGBoost, XGBoost + ERA5, TTM (`baselines/results/`) | GitHub | ✅ |
| AQFM final setup: 3 seeds, mean ± std, ensemble, no-future-met seed 0 (`results/aqfm_do3_*`) | GitHub + Drive | ✅ |
| AQFM history: run 1, run 2, anchor dropout 0.1 | GitHub + Drive | ✅ |
| Stage E: 5 ablations, no-future-met seeds 1/2 + mean ± std + ensemble (`results/aqfm_do3_*`), `results/aqfm_ablations_summary.md` | machine + Drive folder | ✅ committed, ⬜ push and upload |
| PatchTST: seeds 0/1/2, mean ± std, ensemble (`baselines/results/patchtst_*`), `results/patchtst_summary.md` (like-for-like comparison) | GitHub (⬜ push) + Drive folder | ✅, ⬜ upload |
| E-STGCN: seeds 0/1/2, mean ± std, ensemble (`baselines/results/estgcn_*`), `results/estgcn_summary.md` (all no-weather models in one table) | machine + Drive folder | ✅, ⬜ commit, ⬜ upload |
| `results/TABLES.md` (all tables, generated) | GitHub + Drive | ✅ (PatchTST and E-STGCN rows added 2026-10-16, ⬜ commit) |
| Training and evaluation logs (`runs/*/config.json`, `log.jsonl`, `runs/*.txt`; `deep_baselines/runs/`, `deep_baselines/estgcn/runs/`) | GitHub + Drive `run_logs/` | ✅ (E-STGCN logs ⬜ commit) |

## AQFM v2 (city preset, 1.17M parameters; branch `aqfm-v2`; log: `docs/V2_PROGRESS.md`)

| Item | Where | Status |
|---|---|---|
| v2 code: spatial-scale changes A1–A9, fixes B1, B3–B7, any-city data steps, width flags, date splits, hourly-array converter; 86 tests pass (`tests/` 68 + `deep_baselines/` 18) | GitHub branch `aqfm-v2` | ✅ committed and pushed (`a00d49d`); `main` has it up to `f104c7c` (PR #1) |
| `data/delhi/` (built from the raw inputs, `docs/V2_PROGRESS.md` step 2; sha256 in the manifest) | machine | ✅ (rebuildable, not on Drive) |
| v2 models: 3 seeds, dropout 0.15 run, 4-station hold-out run, 3 seeds without future weather (`aqfm_v2_*`); v2-M (width 256, 4.57M) 3 seeds with and 3 without future weather (`aqfm_v2m_*`); 24k-step test (`aqfm_v2_24k_s0`) | machine + `drive_upload_2026-11-05_v2/models` (68 files with steps 10–12, `SHA256SUMS.txt` checked) | ✅ in the manifest; ⬜ upload to Drive |
| v2 results: 3 seeds, mean ± std, ensemble, unseen stations, distance bands, without future weather (`results/v2_*`), `results/v2_delhi_summary.md`; v2-M and 24k (`results/v2m256_*`, `results/v2_s0_24k_results.json`) | GitHub branch `aqfm-v2` | ✅ committed, pushed, in the manifest and `results/TABLES.md` |
| Step 10: transfer to Mumbai (`data/mumbai` from the only Mumbai data, an hourly 2023–2024 array; `docs/V2_PROGRESS.md` step 10) | machine + GitHub | ✅ all stages, same-point comparisons, in the manifest and `results/TABLES.md` |
| Step 11: Delhi same-point comparisons with significance (ozone, NOx, NO2 too), v2-M ablations, second test year (2024), noisy future weather, Mumbai fine-tune with nothing frozen (`docs/V2_PROGRESS.md` step 11; `results/paper/`) | machine + GitHub | ✅ committed and pushed 2026-11-07; models in the manifest and the Drive folder |
| Step 12: NO2 forecast directly (`--no2-direct`): the 6 v2-M models retrained, per-seed and ensemble results, same-point comparisons (`docs/V2_PROGRESS.md` step 12; `results/v2m256_no2d_*`, `results/paper/delhi_*_no2d.json`) | machine + GitHub + Drive folder | ✅ committed and pushed 2026-11-07; in the manifest, `results/TABLES.md` and the Drive folder (68 files). These are the paper's main v2-M models (decision in step 12) |
| Step 13: seasonal-naive baseline used future data at 72 h; fixed in `baselines/run_baselines_classical.py` and `scripts/city_baselines.py`, all classical outputs rerun, `tests/test_baselines.py` (`docs/V2_PROGRESS.md` step 13) | GitHub | ✅ committed and pushed 2026-11-07; manifest `code` of the two classical entries updated |
| Step 14: XGBoost with the latest readings (`scripts/city_xgboost.py --lags-from-last`, off by default): Delhi 2025, Delhi 2024 (PM2.5), Mumbai 1/3/6/12 months; same-point files `results/paper/*_xgblast.json` (`docs/V2_PROGRESS.md` step 14). The paper's main XGBoost baseline | GitHub | ✅ committed and pushed 2026-11-07; `results/paper/` is not in the manifest (as steps 10–12); forecasts `results/forecasts/*_last.npz` are on the machine only |
| Step 15: gap filling against stronger baselines (`scripts/gapfill_eval.py`, 4/12/24/48 h, line, line + cycle, nearest stations, XGBoost gap filler) and one improvement attempt (`scripts/gapfill_anchored.py`, developed on 2024 validation); `results/paper/*gapfill*.json` (`docs/V2_PROGRESS.md` step 15). The paper only mentions gap filling | GitHub | ✅ committed and pushed 2026-11-08 (1ce6cfb, 6d251d0); forecasts `results/forecasts/*gapfill*.npz` on the machine only |
| Step 16: deep-learning baselines on the same forecasts with tests: PatchTST, E-STGCN and TTM re-run from their saved models with `--save-forecasts` (identical to the published results; `baselines/results/*_rerun.json`, in the manifest), comparisons `results/paper/delhi_{PM2.5,PM10}_no_weather_deep.json` (`docs/V2_PROGRESS.md` step 16) | GitHub | ✅ committed and pushed 2026-11-08 (6d251d0 + manifest commit); forecasts `results/forecasts/delhi_{patchtst_ens3,estgcn_ens3,ttm}.npz` on the machine only |

## Experiments the paper needs (roadmap: `PROJECT_CONTEXT.md` §6)

| Item | Status |
|---|---|
| Baselines on the rebuilt data (Stage B) | ✅ |
| AQFM final setup, 3 seeds + ensemble (Stages C, D) | ✅ |
| E1 oracle gap (no future met): seeds 0, 1, 2 + ensemble | ✅ |
| E7 no anchor (run 1 vs anchor dropout 0.1; not re-run, see `PROJECT_CONTEXT.md` §6) | ✅ |
| E2 no BLH · E3 no spatial attention, no wind · E4 PM2.5 predicted directly · E5 gap-filling only (seed 0 each) | ✅ (`results/aqfm_ablations_summary.md`) |
| F1 PatchTST on our 2025 protocol (same treatment as AQFM; `results/patchtst_summary.md`) | ✅ |
| F4 E-STGCN, our re-implementation on our protocol (same treatment; `results/estgcn_summary.md`) | ✅ |
| F5 O3 and NO2 for every model (plan: `docs/plan_o3_no2.md`, ~6 h) | ⬜ |
| G write-up: literature review, methods deviations (§7.6), figures, slides with real numbers | ⬜ |
| H open risks: COVID year, station types (discuss or test); second test year done (v2-M, step 11) | ⬜ |

## Final checks (in this order, just before submission)

| Item | Status |
|---|---|
| Everything committed and pushed; Drive folder uploaded | ⬜ |
| `python scripts/check_submission.py` prints `ALL OK` | ⬜ (all OK except "not pushed" after the E-STGCN commit; on branch `aqfm-v2` the check compares with GitHub `main`, so it fails until v2 is merged into `main`) |
| `python scripts/check_submission.py --reproduce` (re-runs every result, ~7.7 h; LAST) | ⬜ |
| Every number in the paper copied from `results/TABLES.md` | ⬜ |
