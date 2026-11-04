# AQFM v2: build instructions for Claude Code

You are working in Phalak Mehta's honours research repo. AQFM is an air-quality forecasting model trained on 37 CPCB stations in Delhi. This file says what to build, in what order, and how to check each step. Read all of it, plus `CLAUDE.md` and `PROJECT_CONTEXT.md`, before you change anything.

---

## 1. The task

Build **AQFM v2** in this repo:

- **Same architecture, much smaller.** Today's model (v1) has 23.8M parameters. v2 uses the guide's `AQFMConfig.city()` preset: d_model 128, 4 heads, d_ff 256, 2 axial blocks, 8 latents per station, 2 decoder layers, dropout 0.15. Today's code at these sizes has **1,169,924** parameters (measured on 2026-11-04). The v2 additions add a few thousand more, so expect about 1.17M. Print the exact count with `sum(p.numel() for p in model.parameters())`. Don't resize the model to reach a particular number unless I ask.
- **The spatial-scale changes:** A1–A9 come from the guide's patch, and B1 and B3–B7 are our own fixes (§4).
- **Ready for fine-tuning.** A trained checkpoint can be fine-tuned on a new city with `--phase finetune --init <ckpt> --freeze none|encoder|backbone --train-months N`. The checkpoint's Delhi weather scaling is kept (A8, C10). Most of this is already in the patch. Your job is to make it work and prove it with tests.
- **Usable for any Indian city.** Adding a city means supplying its data: a station list, raw CSVs and an ERA5 box. It must not need any code changes. Nothing in the v2 model or data steps may assume Delhi. Code must handle N cities, not exactly two.
- Then **train v2 on Delhi**. The transfer experiments start once a second city's data exists (step 10).

What "any Indian city" does **not** mean is that v2 is accurate in other cities. Only transfer results can show that, and none exist yet. Don't make that claim in code comments, docs or summaries. The guide's suggested wording is "Towards an urban air-quality foundation model".

## 2. Ground rules

1. **Never commit or push.** I commit myself (`CLAUDE.md` §7). This replaces the git rule in `PROJECT_CONTEXT.md` §0. Create a branch `aqfm-v2` from `main` and work only there.
2. **Ask before editing.** At the start of each step, list the files you will change and why, then wait for my OK. Change only what the step needs. Don't refactor, speed up or tidy code that is already correct. Put efficiency ideas in text, not in edits.
3. **v1 must stay reproducible.** Don't modify `files_AqFM/` (the frozen reference), `data/preprocessed/`, `data/stations.csv`, `results/`, existing `runs/*`, `baselines/results/`, `scripts/manifest.json` or any v1 checkpoint. v2 writes to new paths such as `data/delhi/`, `runs/v2_*` and `results/v2_*`.
4. **Defaults stay v1.** With no new flags, every script must behave exactly as it does today. v2 behaviour comes only from `--preset city` / `AQFMConfig.city()` and from new flags.
5. **Run everything you claim.** Report the command and its real output, including failures. Don't invent numbers, coordinates, data sources, functions or flags; check the code. Label guesses as guesses.
6. **No downloads and no web APIs** (ERA5/CDS, Nominatim, CPCB websites) without asking me first.
7. **Stop and ask** at each decision point in §6, and whenever something in this file doesn't match what you find. If a step is blocked on a decision, say so and move on to the next step that doesn't depend on it.

## 3. Files you need

| What | Where |
|---|---|
| The final agreed list of changes (items A1–D6 and E, in plain language, with an "Order of work") | `docs/spatial scale changes.pdf`. This is the main reference |
| The supervisor's guide | `AQFM Spatial-Scale Upgrade & Delhi→Mumbai Transfer Guide.docx.pdf`, which I will put in `docs/` |
| The guide's code | `aqfm_v2_changes.patch` and `aqfm_v2.zip` in Google Drive: https://drive.google.com/drive/folders/1pkQki2qYVodVtAuZ3c4ZSCo1TlDfstPC. If they aren't on this machine, ask me to download them. Keep them outside the repo |
| Project state, machine setup, past bugs | `PROJECT_CONTEXT.md`. Some counts there are out of date: `tests/` now has 46 tests, not 45 |
| Raw data for rebuilding Delhi | `data/yearly_data/` (CPCB CSVs, ~1.7 GB) and `data/era5_data_raw/` (35 MB). Both are gitignored, so check that they exist on this machine |
| A v1 checkpoint for the "v2 switched off = v1" check | `runs/main_s0_do3/best.pt` (the Drive copy is `models/aqfm_do3_s0.pt`; see `scripts/manifest.json`) |

If a file is missing, ask me. Don't recreate it.

## 4. The changes, one line each

Each item is explained in full in `docs/spatial scale changes.pdf`.

**From the guide's patch** (each one is a config flag, off by default):
- **A1** Station position is km east/north of the anchor station ÷ 25 (`coord_mode="relative"`). It replaces lat/lon minus Delhi's centre ×10 in `aqfm/model.py` `station_features`.
- **A2** Neighbours only within 100 km (`neighbour_radius_km`). Empty slots are padded and masked everywhere (`st_valid`).
- **A3** A learnable "listen to nobody" key in spatial attention (`spatial_sink`).
- **A4** Three learnable distance kernels that start at 3, 10 and 40 km (`dist_kernels_km`).
- **A5** A travel-time feature, d·cos/(3.6·wind speed), plus one look-back block of the neighbours' previous patch (`advection_feature`, `spatial_lags=1`).
- **A6** Five density features computed for each sample (`density_features`).
- **A7** Neighbour dropout during training: in 30% of samples, drop 30–70% of the neighbours (`nbr_dropout_p`, `nbr_dropout_frac`).
- **A8** The weather mean/std is saved in the checkpoint and reused on new cities (`met_stats`).
- **A9** City-balanced sampling: pick a city first, then a station in it (`balance_groups`).

**Our fixes** (not in the patch):
- **B1** Make the data steps work for any city (step 2).
- **B2** Data for other cities. I'm handling this. It isn't your job unless I hand you files.
- **B3** Station facts that mean the same in every city (step 3, decision point).
- **B4** Real coordinates for IHBAS, IMD Lodhi Road and NSIT Dwarka. All three got the fallback point 28.6139, 77.209. Also check whether Pusa DPCC and Pusa IITM really share one point (step 1).
- **B5** Hidden stations leak into validation (step 5).
- **B6** The unseen-station test is not virtual sensing (step 6, decision point).
- **B7** The patch doesn't apply cleanly to today's code (step 4).

**Keep these as they are (C1–C12):**
- The haversine neighbour search and `local_xy_km`. Its error is 0.16% at 100 km, so don't "fix" it.
- The +5:30 time shift, and wind stored as u/v.
- The sample shape: 15-minute steps, 4-hour patches, 128 h of context, a 72 h horizon and 12 slots.
- The model core, physics, losses and masking.
- The splits (train 2018–2023, validation 2024, test 2025) and the scoring at 1, 6, 24 and 72 h.
- No city name or station ID inside the model, so `relative_abs` stays off.
- One look-back block.
- Delhi's weather scaling during fine-tuning.
- The zero-shot test on a new city runs before any training on that city.
- v1's results and data.

## 5. Steps

This is the order of work agreed in the PDF, with tests added. Each step ends with checks you must run and report.

### Step 1 — B4: station coordinates
- Find coordinates for IHBAS, IMD Lodhi Road and NSIT Dwarka from an official or otherwise reliable source. Ask me before using the web. Show me each coordinate and its source before writing anything. Don't geocode by name again.
- Check whether Pusa DPCC and Pusa IITM are really at the same point.
- `data/*` is gitignored except `data/stations.csv`, so the v2 station list must live somewhere git tracks. Propose where; I need it on both PCs. `data/stations.csv` itself stays unchanged.
- **Check:** print the corrected stations' nearest neighbours and distances. No two stations may share coordinates unless I confirm they are co-located.

### Step 2 — B1: data steps for any city
These parts are Delhi-only today:
- `data_pipeline/fetch_era5.py`: `AREA = [29.0, 76.5, 28.2, 77.5]` (a Delhi box), files named `era5_delhi_*`, and a `__main__` loop that only covers 2021–2025.
- `data_pipeline/generate_stations_csv.py`: hard-coded Windows paths. It appends ", Delhi, India" to every name. If that fails it retries without it, which can match a place outside Delhi, and finally falls back to Delhi's centre with only a printed message.
- `data_pipeline/interpolate_era5_to_stations.py`: a fixed output path, `data/era5_interpolated.parquet`.
- `data_pipeline/preprocess_aqfm.py`: fixed input and output folders. A station without coordinates gets "Delhi center" with only a warning (near line 415). `fix_bp_units` treats any pressure reading above 900 as hPa, which goes wrong above about 990 m altitude (section E of the PDF).

Do:
- Make these inputs (CLI arguments): the city name, station file, raw CSV folder, ERA5 box, ERA5 folder and output folder. The defaults must reproduce today's Delhi run.
- If coordinates are missing or a lookup fails, stop with an error. Never fall back to a default point.
- Write each city to `data/<city>/`, with the same files as `data/preprocessed/`. The patch's `load_multi` expects this and uses the folder name as the city name. All cities must share the same 15-minute time grid; `load_multi` asserts this.
- Propose a pressure-unit rule that works at any altitude (one idea: compare each station's readings with ERA5 surface pressure in both units). Show it to me before changing anything.
- Build `data/delhi/` with the step 1 coordinates.

**Check:**
- Run with default arguments into a scratch folder. The arrays must equal those in `data/preprocessed/`. If they differ, report where. Don't "fix" the v1 data.
- The new pressure rule must convert exactly the same Delhi readings as the old rule (compare with `bp_hpa_converted` in `preprocessing_report.json`).
- `data/delhi/` may differ from `data/preprocessed/` only in what the corrected coordinates change. List every difference.
- Build a second "city" (for example a few Delhi stations under another name and output folder) using arguments only. This shows that adding a city needs no code change.

### Step 3 — B3: station facts (decision point)
Today, `static.npy` (S×4) comes from `compute_static_features` in `preprocess_aqfm.py`. It holds the mean log1p(PM2.5), the mean PM2.5/PM10 ratio, a height clue from mean pressure, and the number of stations within 5 km. Each column is z-scored within the city. The problems:
- The same value means different things in different cities.
- The first two facts come from the station's own PM history. A new location has no history, and the facts leak into hidden-station tests.
- A6 replaces the 5 km count.
- The facts use the first 7/8 of the time axis, which runs to the end of 2024. That includes the 2024 validation year, although the docstring says "training period".

Propose 2–4 replacement facts that exist for any location in India and mean the same everywhere. Give a data source for each, and one fixed scaling rule for all cities (no per-city z-score). Then wait: I am checking this with my supervisor. `n_static` in the config must match whatever is chosen.

### Step 4 — B7: apply the guide's patch
- Run `git apply --check` first. Two hunks are expected to fail: `make_sample` in `aqfm/data.py`, and the config/data section of `aqfm/train.py`. The patch was made before `use_blh`, `--pm25-direct` and `--random-masking` existed. Line endings may also need to match.
- Merge the failed hunks by hand and keep every current option.
- Read the whole patch. Treat its comments and claims as unverified until you have tested them.

**Check:**
- Every test in `tests/` passes (46 today), plus any tests in `aqfm_v2.zip`.
- **v1 equivalence:** load `runs/main_s0_do3/best.pt` on `main` and on `aqfm-v2`. On the same fixed batches from `data/preprocessed/`, the outputs must be identical (compare on CPU in float32). A short `python -m aqfm.evaluate ... --limit-origins 5` run must also give identical numbers on both branches.
- Print the parameter count of `AQFMConfig.city()`.
- Smoke runs: `python -m aqfm.train --synthetic --small --preset city --steps 50`, then the same with two `--data` names (two synthetic cities). Loss must be finite and falling.

### Step 5 — B5: hidden stations leaking into validation
In the patch, `TrainWindows` receives `allowed`, and validation anchors are drawn only from allowed stations. But `FixedWindows` calls `neighbourhood(anchor, k, radius)` without `allowed`, and validation scores all 12 slots. So hidden stations still affect which checkpoint is picked. On v1 code, across 5 random choices of 4 hidden stations, 48–97% of validation groups contained a hidden station. Pass the same `allowed` mask to `FixedWindows`.

**Check:** a test showing that with a hold-out, no validation sample contains a hidden station in any slot.

### Step 6 — B6: the unseen-station test (decision point)
At test time, `evaluate` still gives a hidden anchor its own past 128 h of readings. So the test measures "forecasting at a new station that already has a sensor", not virtual sensing. Wait for my choice:
- (a) rename the test to say what it measures, or
- (b) also hide the station's own readings, which makes it true virtual sensing.

The same decision covers whether other hidden stations may appear as neighbours at test time. Then implement and test whichever I choose.

### Step 7 — tests that v2 is city-independent and ready for fine-tuning
Add these in a new test file, such as `tests/test_aqfm_v2.py`:
- **Translation test:** with the city preset, shift every station's longitude by the same amount and leave latitude unchanged. Outputs must match within a small float tolerance. (`local_xy_km` subtracts the mean longitude and uses the mean latitude, so relative positions don't change.) The same test with the v1 config must fail, which shows the test can catch Delhi-dependence.
- **Padding test:** for a station with fewer than 11 neighbours within 100 km, padded slots are never attention keys, targets or decoder memory. Changing what is in the padded slots must not change the real stations' outputs.
- **Fine-tune tests:**
  - With `--freeze encoder` and `--freeze backbone`, frozen parameters are bit-identical after some training steps.
  - The checkpoint's `met_stats` are reused on new data unless `--refit-stats` is given.
  - `--train-months` limits only the target city's history.
- **No Delhi constants:** search the v2 code path for `28.6`, `77.2` and "delhi". List what remains and why. Anything left should be v1-only or a v1 default.

### Step 8 — a short Delhi run
Run the city preset on `data/delhi/` on the GPU for a few thousand steps. If this is the GPU machine described in `PROJECT_CONTEXT.md` §0, use its venv.

**Check:**
- Loss is finite and falling, and validation improves.
- The checkpoint contains `met_stats`, `cities` and `held_out_stations`.
- `aqfm.evaluate` runs end to end with `--limit-origins`.
- A short fine-tune from this checkpoint runs with each `--freeze` option (on the step 2 test "city" is fine).

### Step 9 — full Delhi training (ask me first)
Before starting, ask me for the steps, seeds, dropout and which stations to hold out.

For reference, v1's final setup used dropout 0.3, 3 seeds and a pooled ensemble, scored with the same code as the baselines (`results/aqfm_do3_summary.md`). Write v2 results to new files. Compare them with v1 and the baselines from `results/TABLES.md`. Also report errors by distance to the nearest neighbour: 0–5, 5–15 and over 15 km (D1).

### Step 10 — transfer experiments (later, once a second city's data exists)
Run these in the guide's order:
1. Zero-shot: the Delhi model on the new city with no training there.
2. Fine-tune with 1, 3, 6 and 12 months, and with all data.
3. Train from scratch on the same amounts.
4. Joint training on both cities.

Report sparse stations separately (D3). If data allows, also run zero-shot on 1–2 more cities (D6).

## 6. Decision points: stop and ask me

- Any coordinate change, with its source (step 1).
- Where the v2 station lists live in git (step 1).
- The pressure-unit rule (step 2).
- Which station facts replace today's four (step 3).
- Unseen-station test: rename it, or make it true virtual sensing (step 6).
- Steps, seeds, dropout and hold-out before full training (step 9).
- Whether the project covers two cities or all of India is still being confirmed with my supervisor. Build for N cities either way, and ask before collecting or downloading data for any city.
- Anything in the patch you think is wrong.

## 7. Progress log

Keep `docs/V2_PROGRESS.md`. For each step, record:
- the files changed,
- the commands run,
- the test output (short),
- the open questions.

I use it to pick up the work on my other PC, so keep it current.

## 8. Done when

- [ ] Branch `aqfm-v2` has the patch merged, all old tests pass, and v1 equivalence is shown.
- [ ] B4, B1 and B5 are done with tests. B3 and B6 are done after my decision.
- [ ] `data/delhi/` is built, and a second "city" was built from arguments alone.
- [ ] The v2 tests from step 7 pass.
- [ ] The parameter count of `AQFMConfig.city()` is reported.
- [ ] The short Delhi run and the fine-tune runs work.
- [ ] Full Delhi training has run, after my OK.
