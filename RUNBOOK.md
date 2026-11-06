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

## 3. State (updated 2026-11-06 21:30)

- **Done and pushed (commit `a239e03` on `aqfm-v2`):** Mumbai stages A–D (results `results/mumbai_*`,
  `results/delhi_joint_v2_ens3_results.json`; tables and findings in `docs/V2_PROGRESS.md` step 10), the evaluation
  patch (`--targets`, `--save-forecasts`, `--test-end`, `--met-noise`, `--v1-spatial`; 91 tests pass), the paper
  scripts.
- **Done (21:51):** stage `MUMBAI` (`results/paper/mumbai_*`, `results/forecasts/mumbai_*`); checks and findings in `docs/V2_PROGRESS.md` step 10; manifest registration done (70 models, 116 results); `results/TABLES.md` regenerated; `scripts/check_submission.py` passes everything except the git checks.
- **Then, unattended (user: "yea start", 2026-11-06 21:37):** `/home/phalak/aqfm/chains/overnight_2026-11-06.sh`,
  log `runs/paper_overnight_chain.log`. After MUMBAI it runs the Delhi XGBoost reproduction check alone (output
  `/home/phalak/aqfm/checks/xgb_delhi_repro.*`; the log says IDENTICAL or DIFFERENT), then stages SAVE → ABL → FOLD →
  NOISE (§4.2; logs `runs/paper_<STAGE>.log`). It stops at the first stage that does not finish. Expected end: about
  noon on 2026-11-07. To resume after a stop: rerun the failed stage's command from §4.2, then the later ones (stages do
  not skip finished work, so do not rerun a finished stage).
- **Progress (checked 2026-11-07 00:15):** the XGBoost check line says DIFFERENT, as expected (the 0.1–0.7 % gap to the
  published file, §5; the original script gives the same numbers as `city_xgboost.py`). Stage SAVE finished 23:45 and
  passed §4.1 checks 1–2: the five ensembles have 82/82 PM numbers equal to the earlier files, commit `a1a9e23`, not
  dirty. Stage ABL started 23:45; its first ablation (no spatial attention) finished 00:13.
- **Spatial ablations dropped (user, 2026-11-07 00:20):** "the purpose of spatial scale is to scale the model to
  different cities, not to improve the results". The chain was stopped at 00:22 during the distance-bias run (moved to
  `/home/phalak/aqfm/cancelled_runs/2026-11-07/`, not used); `--v1-spatial` was not run. The no-spatial result that
  was already made is kept (`results/paper/delhi_v2m256_abl_nospatial_s0.json`). `scripts/pipeline_paper_extras.sh`
  stage ABL now has only noblh, nolast and gapfillonly.
- **Running now:** `/home/phalak/aqfm/chains/resume_2026-11-07.sh`, log `runs/paper_resume_chain.log`: the rest of
  ABL (log `runs/paper_ABL_resumed.log`; `runs/paper_ABL.log` has the no-spatial part), then FOLD and NOISE
  (`runs/paper_<STAGE>.log`), stopping at the first stage that does not finish; then the unfrozen Mumbai fine-tune
  either way (`runs/mumbai_unfrozen.log`). Expected end: about 08:00 on 2026-11-07.
- **Second commit of 2026-11-06** (after `a239e03`): the manifest, tables, stage MUMBAI results and docs.
- **`results/forecasts/` is not in git** (about 1 GB per stage; `.gitignore`). It is regenerated by the stage that made
  it. `results/paper/` (the comparisons) is tracked.
- **Drive:** the 31 Mumbai and joint models are copied to `/home/phalak/aqfm/drive_upload_2026-11-05_v2/models/`
  (README and `SHA256SUMS.txt` updated; 46 files, all OK). The user uploads that folder.

## 4. Queue, in order

### 4.1 When the overnight chain has finished (`runs/paper_overnight_chain.log` ends with `overnight chain finished`)

(The steps after stage MUMBAI were done on 2026-11-06; see `docs/V2_PROGRESS.md` step 10.)

1. **Check each stage:** `tail -3 runs/paper_{SAVE,ABL_resumed,FOLD,NOISE}.log` ends with `pipeline finished (stage X)`.
   Every `results/paper/delhi_*.json` from `aqfm.evaluate` has `"dirty": false` and, in `code`, commit `a1a9e23`
   (HEAD when the stages started; its code is the same as `a239e03`) or the commit of 2026-11-07 ~00:40 (docs,
   scripts and SAVE results only; same code).
   The XGBoost check line in the chain log says `IDENTICAL`.
2. **SAVE: the PM numbers must not move.** The PM2.5/PM10 numbers in `results/paper/delhi_{v2m256,v2m256_nofm,v2,v2_nofm}_ens3.json`
   must equal `results/{v2m256,v2m256_nofm,v2,v2_nofm}_ens3_results.json`, and `delhi_v1_ens3.json` must equal
   `results/aqfm_do3_ens3_results.json`. The Mumbai files passed this check.
3. **Write up** in `docs/V2_PROGRESS.md` (a new step 11), with tables generated from the files:
   - SAVE: same-point tables with p-values, Delhi PM2.5/PM10/NOx/ozone/NO2;
   - ABL: each ablation (no BLH, no anchor, gap-filling only) against the full v2-M seed 0 (`results/v2m256_s0_results.json`);
   - FOLD: the 2024 test year;
   - NOISE: the error as the weather noise grows.
4. **Register** the new trained models (ABL, FOLD) in the manifest and copy them to the Drive folder, following
   `/home/phalak/aqfm/pending_patches/add_mumbai_manifest.py`. `results/paper/` is not checked by `check_submission.py`.
5. **Commit and push, only if the user agrees.**

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
- **Virtual-sensing fix,** or report it as a limitation.
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
