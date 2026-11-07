# AQFM runbook (for any assistant or person continuing the work)

Written 2026-11-06. Read this first, then `HANDOFF.md` (state, results, paper to-do in §8), then the end of
`docs/V2_PROGRESS.md` (the detailed log). Every step below gives the command, the time, what it writes and how to
check it.

## 1. Rules (the user's; do not break them)

1. **Git:**
   - Commit or push only when the user asks, each time.
   - Author and committer: `Phalak Mehta <phalakmehta246@gmail.com>`.
   - **No AI tool may appear as a contributor:** no `Co-Authored-By` lines, no "generated with" lines.
   - `CLAUDE.md` is never committed (it is listed in `.git/info/exclude`).
   - Work on branch `aqfm-v2`.
   - Push from VS Code (Source Control → Sync), where the GitHub login lives.
2. **Old results stay reproducible.** Never modify:
   - `files_AqFM/`;
   - `data/preprocessed/` and `data/stations.csv`;
   - `baselines/*.py`, `baselines/results/`;
   - existing `results/*.json` or `runs/*`.

   New work writes new files.
3. **Selection only on validation.** Delhi: train 2018–2023, validation 2024, test 2025. Mumbai: train 2023, validation 2024-01..06, test 2024-07..12. Never tune anything on a test period. Report all seeds.
4. **Every result is registered in `scripts/manifest.json`** with its exact command. Numbers in docs come from generated files, never typed from memory.
5. **While a training or evaluation chain runs, do not create or edit `.py` files under `aqfm/`, `tests/`, `baselines/` or `deep_baselines/`.** Each run records the git state of those folders, and an edited file marks the run "dirty", which means it cannot be registered. Editing `scripts/*.sh`, `scripts/*.py` and `*.md` is safe.
6. **`train.py` appends to `runs/<name>/log.jsonl`.** Before rerunning a run that was interrupted, move its folder (and `runs/<name>.txt`) aside.
7. **Ask the user before anything over about an hour, and before any download.** Give the time estimate.
8. **Write for the user:** short and plain.

## 2. Environment

- **Paths:** repo `/home/phalak/aqfm/AQFM-Honors`; Python `../.venv/bin/python` from the repo root; GPU: RTX 4000 Ada.
- **Tests:**
  ```
  ../.venv/bin/python -m pytest -q tests deep_baselines
  ```
  Expect 91 passed. About 40 s. Don't run them while training (or use `OMP_NUM_THREADS=2 nice`).
- **Chains run detached:**
  ```
  STAGE=X nohup setsid bash scripts/<script>.sh > runs/<log> 2>&1 &
  ```
  Follow with `tail runs/<log>`. A finished chain prints `pipeline finished`; a failed run prints `did not finish - stopping`.
- **Is the GPU busy?** `nvidia-smi`, and `pgrep -af "aqfm.train|aqfm.evaluate"`.
- **Typical times:**

  | Run | Time |
  |---|---|
  | v2 training (12k steps) | ~17 min |
  | v2-M training | ~30 min |
  | Mumbai fine-tune | 2–5 min |
  | Delhi 3-seed ensemble evaluation | ~14 min (single model ~5) |
  | Mumbai ensemble evaluation | ~5 min |

## 3. State (updated 2026-11-07, after the step 11 commit)

- **Committed on `aqfm-v2`:**
  - `a239e03` (pushed): Mumbai stages A–D and the evaluation patch;
  - `a1a9e23` (pushed): stage MUMBAI, manifest, tables;
  - `c07d55a` (pushed): stage SAVE results, the dropped spatial ablations, the unfrozen Mumbai script, docs;
  - the next commit (pushed 2026-11-07): step 11 results, manifest, tables and docs (`git log -1 -- RUNBOOK.md`).
  - `main` is still at PR #1 (`6cdade1`); everything since `f104c7c` is only on `aqfm-v2`.
- **All paper experiments are done (2026-11-07 06:01).**
  - Chains: `/home/phalak/aqfm/chains/overnight_2026-11-06.sh` ran SAVE and the first ablation (log
    `runs/paper_overnight_chain.log`). It was stopped by hand when the user dropped the spatial ablations.
    `/home/phalak/aqfm/chains/resume_2026-11-07.sh` then ran the rest of ABL, FOLD, NOISE and the unfrozen Mumbai
    fine-tune (log `runs/paper_resume_chain.log`).
  - All checks passed: the logs end with `pipeline finished`, the AQFM results are not dirty, SAVE's PM numbers are
    unchanged, and every training ended with `done`.
  - Write-up, tables and findings: `docs/V2_PROGRESS.md` step 11.
- **Spatial ablations dropped (user, 2026-11-07 00:20):** "the purpose of spatial scale is to scale the model to
  different cities, not to improve the results".
  - The distance-bias run was stopped mid-training and moved to `/home/phalak/aqfm/cancelled_runs/2026-11-07/` (not
    used). `--v1-spatial` was not run.
  - The no-spatial run that had already finished is kept, for reference only.
- **Registered after the runs:**
  - Manifest: 86 models and 126 results. The step 11 models and the `results/mumbai_ft_v2_none_*` files were added
    by `/home/phalak/aqfm/pending_patches/add_step11_manifest.py`, which checks each command with the scripts' own
    parsers and was mutation-tested. `results/paper/` is not registered, as before.
  - `results/TABLES.md` regenerated. `scripts/make_tables.py` now labels the FOLD runs "validation 2023, test 2024"
    and has the nothing-frozen Mumbai rows.
  - `scripts/check_submission.py` passes everything except the git checks.
  - Tests: 91 pass (`check_submission.py` runs only `tests/`: 73).
- **`results/forecasts/` is not in git** (about 1 GB per stage; `.gitignore`). It is regenerated by the stage that made
  it. `results/paper/` (the comparisons) is tracked.
- **Drive:** `/home/phalak/aqfm/drive_upload_2026-11-05_v2/models/` has 62 models, including the 16 of step 11 (ABL,
  FOLD, Mumbai nothing frozen). README and `SHA256SUMS.txt` updated; all OK. The user uploads that folder.

## 4. Queue, in order

### 4.1 Next

1. **NO2 fix (user, 2026-11-07: "try to fix it").**
   - Cause: NO2 is rebuilt from NOx and the NO share. The NO share is valid for only 58% of training NO2 readings
     (at 10 stations for at most 15%), and at some stations NO2 rose above NOx in 2025. Rebuilt NO2 cannot exceed
     NOx, so it is under-forecast there.
   - Fix: `aqfm/train.py --no2-direct` (NO2 is its own output in place of the NO share) and `aqfm/evaluate.py` reads
     it. New test in `tests/test_aqfm.py`; 92 tests pass. Committed (user chose "NO2 runs after trial").
   - Trial (v2-M with weather, seed 0; `/home/phalak/aqfm/explore/`): better on validation 2024 for NO2 at every
     lead, other pollutants about equal (`docs/V2_PROGRESS.md` step 12). Adopted.
   - **Running (started 2026-11-07 ~11:50, ~4 h):** `scripts/pipeline_no2_direct.sh`, log `runs/no2_direct.log`. It
     trains `runs/v2m256_no2d{,_nofm}_s{0,1,2}_do03`, writes `results/v2m256_no2d*_results.json`, seed stats,
     ensembles (forecasts saved) and `results/paper/delhi_*_no2d.json`. Check: the log ends with `pipeline
     finished`; every training ends with `done`; the results record a clean commit.
   - Then: register the 6 models and their results in the manifest, add rows to `scripts/make_tables.py`, copy the
     models to the Drive folder, update the report (`/home/phalak/aqfm/reports/`).
2. **Paper decisions for the user** (findings in `docs/V2_PROGRESS.md` step 11):
   - Virtual sensing: **dropped** (user, 2026-11-07). It is not part of the paper's claims.
   - Mumbai: validation picks "nothing frozen" for 6 and 12 months. Make it the headline fine-tune for those amounts,
     with backbone frozen kept for 1–3 months (nothing frozen was not run there)?
   - Mumbai PM2.5: validation and test disagree on fine-tuned vs from scratch. Report it as it is.
3. **Optional:** FOLD with weather, seed 2, peaked at its last step (12,000). A longer run would tell whether that
   budget mattered (~30 min GPU; ask first).

### 4.2 Paper experiments (`scripts/pipeline_paper_extras.sh`; each stage is separate; ask the user before starting)

| Stage | What | Time | Main outputs |
|---|---|---|---|
| `MUMBAI` | Final Mumbai models + baselines with every forecast saved; same-point comparisons with significance per data amount | ~1.5 h | `results/paper/mumbai_*`, `results/forecasts/mumbai_*.npz` |
| `SAVE` | Delhi: main ensembles with forecasts saved and Ozone/NOx/NO2 scored; baselines on `data/delhi`; same-point comparisons | ~2.5 h | `results/paper/delhi_*` |
| `ABL` | v2-M ablations (seed 0): no BLH, no anchor, gap-filling only (no spatial ablations, §3) | ~1.5 h | `results/paper/delhi_v2m256_abl_*` |
| `FOLD` | Second test year: train 2018–22, pick on 2023, test 2024 | ~4 h | `results/paper/delhi_fold24_*` |
| `NOISE` | Realistic weather: future weather with noise (sd 0.25 / 0.5 / 1.0 × √lead-days) | ~45 min | `results/paper/delhi_v2m256_ens3_metnoise*` |

Command: `STAGE=MUMBAI nohup setsid bash scripts/pipeline_paper_extras.sh > runs/paper_MUMBAI.log 2>&1 &`.

**Checks after each stage:**
- the log ends with `pipeline finished (stage X)`;
- every `results/paper/*.json` has `"dirty": false` in its `code`;
- `forecast_stats` prints, for each lead, the number of shared points and a p-value for each comparison.

Then:
- **Before the first real use,** check that `scripts/city_xgboost.py` without `--hourly` gives Delhi's published XGBoost numbers:
  ```
  python scripts/city_xgboost.py --data data/preprocessed --out /tmp/x.json
  ```
  Compare it with `baselines/results/xgboost_results.json`.
- **Register the new results in the manifest:** follow the existing entries. The registration helper pattern (text insertion that keeps old entries identical, with assertions on each result's `args` and `code`) is described in `docs/V2_PROGRESS.md` (housekeeping, 2026-11-05).
- **Regenerate `results/TABLES.md`** (`python scripts/make_tables.py`). First make it put Mumbai runs and `results/paper/*` in their own sections; its validation table currently lists every `runs/*/log.jsonl`.

### 4.3 Later (decide with the user)

- **O3/NO2/NOx for PatchTST and E-STGCN:** `docs/plan_o3_no2.md`. AQFM, XGBoost and the classical baselines are covered by stage `SAVE` (`--targets`).
- ~~Virtual-sensing fix~~ dropped (user, 2026-11-07).
- **Stage-H robustness:** training without 2020 (`--train-months` does not do this; it needs a new option).
- **The final `python scripts/check_submission.py --reproduce`:** last, over 10 h; ask first.

## 5. Things learned the hard way

- **Mumbai data is hourly, on the hh:45 step.** Anything that reads "1 h before" must read the latest hourly value (`scripts/city_xgboost.py --hourly`). The original XGBoost gives no valid forecast on hourly data.
- **The classical-baseline rule** counts a point only where all three methods have a forecast. On Mumbai that is about 5 % fewer points than AQFM, so compare on the same points with `scripts/forecast_stats.py`.
- **Fine-tuning on Mumbai peaks early** (stage A2: lr 3e-5, best around step 550). Validate often (every 50 steps), or the best point is missed.
- **One run's log said `dirty: True`:** that happens when a code file changes between commits. Rerun after committing; the numbers are identical (checked for Mumbai zero-shot: 4 × 93 numbers equal).
- **XGBoost numbers move by up to 0.7 % between software setups.** The published v1 XGBoost files were made before this
  machine's xgboost 3.0.5; re-run here, the original script and `scripts/city_xgboost.py` agree exactly with each other
  but not with the published files. Compare AQFM with XGBoost runs made on the same machine (stage SAVE).
- **Never overwrite a `.sh` file that a running chain is executing.** Bash reads scripts bit by bit as it runs. Use `sed -i`, or write a temp file and rename it; both leave the running copy alone.
- **Give every run a unique folder name.** Two stages writing the same `runs/<name>` mix their logs. The stage-A and stage-B collision of 2026-11-06 is described in the progress log.
- **Don't run CPU-heavy jobs (XGBoost `n_jobs=-1`, tests) next to training.** The load reached 53 on 24 cores and training slowed about 6×.
- **Check every training budget on validation.** If the best step equals the last step, the limit is binding; give it more steps.
- **Delhi checkpoints scale weather with Delhi's statistics,** and that is kept for other cities on purpose (A8). Mumbai's ERA5 has no temperature, humidity or sunlight, so the with-weather models get those as "Delhi average". Use the no-weather models for Mumbai.
