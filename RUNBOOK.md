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
  Expect 86 passed; 91 once the 2026-11-06 patch is in. About 40 s. Don't run them while training (or use `OMP_NUM_THREADS=2 nice`).
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

## 3. State at the time of writing

- **Running (relaunched 18:07 on 2026-11-06, with 6,000-step fine-tunes; see `docs/V2_PROGRESS.md` step 10):** `scripts/pipeline_mumbai_transfer.sh`, stages B → C → D. Settings: SIZE=v2, FREEZE=backbone. Logs: `runs/mumbai_transfer_{B,C,D}.log`. The script skips finished runs, so after a stop or power cut, rerun the same command.
  - It is finished when `runs/mumbai_transfer_D.log` ends with `pipeline finished (stage D)`.
  - Outputs:
    - `results/mumbai_{ft_v2_backbone,scratch_v2}_m{1,3,6,12}_{s0,s1,s2,seeds_stats,ens3}_results.json`;
    - `results/mumbai_joint_v2_ens3_results.json` and `results/delhi_joint_v2_ens3_results.json`.
- **Pending code patch (tested; 91 tests pass; mutation-checked):** `/home/phalak/aqfm/pending_patches/2026-11-06_eval_options.patch`. It adds:
  - `aqfm/evaluate.py`: `--targets` (Ozone, NOx, NO2), `--save-forecasts`, `--test-end`, `--met-noise`;
  - `aqfm/train.py`: `--v1-spatial`;
  - `tests/test_eval_options.py`.

  Every option is off by default, and the PM numbers are unchanged.
- **Uncommitted (commit when the user asks):**
  - `scripts/manifest.json` (v2-M and 24k registered), `SUBMISSION.md`, `HANDOFF.md`, `docs/V2_PROGRESS.md`, this file;
  - `scripts/city_xgboost.py`, `scripts/city_baselines.py` (new options), `scripts/forecast_stats.py`, `scripts/pipeline_paper_extras.sh`, `scripts/pipeline_mumbai_transfer.sh` (stage A2 + recipe);
  - the Mumbai results and run logs.

## 4. Queue, in order

### 4.1 When the Mumbai chain has finished

1. **Check it:**
   ```
   grep -c "^done" runs/mumbai_transfer_*.log; tail -3 runs/mumbai_transfer_D.log
   ```
   Expect `pipeline finished (stage D)`, and no `did not finish`.
2. **Apply the patch and test:**
   ```
   git apply /home/phalak/aqfm/pending_patches/2026-11-06_eval_options.patch
   ../.venv/bin/python -m pytest -q tests deep_baselines        # expect 91 passed
   ```
   If the patch was already applied, `git apply` refuses; check with `grep -c save-forecasts aqfm/evaluate.py`.
3. **Commit and push, only if the user agrees.** Do this before the next chain, so its runs record a clean commit.
4. **Table of the Mumbai results:** fine-tuned vs from scratch vs XGBoost per data amount. Write it into `docs/V2_PROGRESS.md`, step 10. The XGBoost numbers per amount are in `results/mumbai_xgboost_m{1,3,6,12}_results.json`.

### 4.2 Paper experiments (`scripts/pipeline_paper_extras.sh`; each stage is separate; ask the user before starting)

| Stage | What | Time | Main outputs |
|---|---|---|---|
| `MUMBAI` | Final Mumbai models + baselines with every forecast saved; same-point comparisons with significance per data amount | ~1.5 h | `results/paper/mumbai_*`, `results/forecasts/mumbai_*.npz` |
| `SAVE` | Delhi: main ensembles with forecasts saved and Ozone/NOx/NO2 scored; baselines on `data/delhi`; same-point comparisons | ~2.5 h | `results/paper/delhi_*` |
| `ABL` | v2-M ablations (seed 0) + v2-M with the v2 spatial changes off (3 seeds) | ~5 h | `results/paper/delhi_v2m256_abl_*`, `delhi_v2m256_v1spatial_ens3` |
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
- **Never overwrite a `.sh` file that a running chain is executing.** Bash reads scripts bit by bit as it runs. Use `sed -i`, or write a temp file and rename it; both leave the running copy alone.
- **Give every run a unique folder name.** Two stages writing the same `runs/<name>` mix their logs. The stage-A and stage-B collision of 2026-11-06 is described in the progress log.
- **Don't run CPU-heavy jobs (XGBoost `n_jobs=-1`, tests) next to training.** The load reached 53 on 24 cores and training slowed about 6×.
- **Check every training budget on validation.** If the best step equals the last step, the limit is binding; give it more steps.
- **Delhi checkpoints scale weather with Delhi's statistics,** and that is kept for other cities on purpose (A8). Mumbai's ERA5 has no temperature, humidity or sunlight, so the with-weather models get those as "Delhi average". Use the no-weather models for Mumbai.
