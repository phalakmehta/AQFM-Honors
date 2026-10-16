# Submission checklist

Everything the paper needs, where it lives, and whether it is done. Update this after every finished step.
**One command checks it all:** `python scripts/check_submission.py` (quick, ~1 min) — must print `ALL OK`.
Every result's command, models and checksums: `scripts/manifest.json`. Every number: `results/TABLES.md` (generated).

Where: **GitHub** = github.com/phalakmehta/AQFM-Honors · **Drive** = Google Drive (`drive_upload_2026-10-14/` and
`Foundation Model(Mainak Sir)/`) · **machine** = the GPU machine only (`/home/phalak/aqfm/`).

Last updated: 2026-10-16, 15:45.

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
| G write-up: literature review, methods deviations (§7.6), figures, slides with real numbers | ⬜ |
| H open risks: COVID year, single test year, station types (discuss or test) | ⬜ |

## Final checks (in this order, just before submission)

| Item | Status |
|---|---|
| Everything committed and pushed; Drive folder uploaded | ⬜ |
| `python scripts/check_submission.py` prints `ALL OK` | ⬜ (all OK except "not pushed" after the E-STGCN commit) |
| `python scripts/check_submission.py --reproduce` (re-runs every result, ~7.7 h; LAST) | ⬜ |
| Every number in the paper copied from `results/TABLES.md` | ⬜ |
