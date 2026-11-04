# AQFM v2: progress log

Instructions: `docs/AQFM_V2_BUILD_PROMPT.md`. Branch `aqfm-v2` (from `main` at bd396a8). Nothing is committed by Claude.
Commands are run from the repo root with `../.venv/bin/python` (GPU machine, `PROJECT_CONTEXT.md` §0).

## Step 0: setup (2026-11-04)

**Inputs, kept outside the repo** in `/home/phalak/aqfm/v2_inputs/` (downloaded with gdown from the Drive folder
`1pkQki2qYVodVtAuZ3c4ZSCo1TlDfstPC`; gdown ran from a separate scratch venv, so the project venv is unchanged):

| File | sha256 |
|---|---|
| `aqfm_v2_changes.patch` | `6e0f5a0faac3c37e732b71251947364e0e10d2ab6cb622faf17aa07c0703a38c` |
| `aqfm_v2.zip` | `a20ec706efdc11acc2966a5f75f062c670f906c46c0bca24ade424507e0a8277` |
| `AQFM Spatial-Scale Upgrade & Delhi→Mumbai Transfer Guide.docx` | `96dceda3a0e19a0ad8d8989fc8efd748ae5b03fb6e660fa547bfb574373da343` |

`aqfm_v2.zip` holds only an `aqfm/` package (11 files); it has **no tests**.

**CLAUDE.md:** restored from git history (`git show ac2d2f8^:CLAUDE.md > CLAUDE.md`), listed in `.git/info/exclude`, never committed.

**Raw data** (arrived 2026-11-04, step 2): `data/yearly_data/`, `data/era5_data_raw/`, `data/era5_interpolated.parquet`.

**Baseline tests on `main` (bd396a8):**
```
../.venv/bin/python -m pytest tests -q
46 passed, 1 warning in 10.00s
../.venv/bin/python -m pytest deep_baselines/tests deep_baselines/estgcn/tests -q
18 passed in 16.60s
```

**Order agreed for now:** steps 1, 4, 5, 7, 8 (step 8 on `data/preprocessed/`); step 2 later.

## Step 1: B4 station coordinates ✅ (2026-11-04)

**Decision (user):** CPCB coordinates for **all** stations; Pusa DPCC and Pusa IMD at CPCB's shared point.

**Files changed:** new `data_pipeline/stations/delhi.csv` (38 rows, same order and `station_id`s as `data/stations.csv`;
columns `station_id, lat, lon, source, cpcb_name, cpcb_code`; written by script from the parsed CPCB rows, not by hand).
`data/stations.csv` unchanged.

**Pusa:** no third independent source found (aqicn's Pusa page has no address or coordinates; search found none).
Recorded disagreement: CPCB has DPCC (#1563) and IMD (#107) both at 28.6396, 77.1463; the preprint has Pusa IMD at
28.61, 77.10 (5.59 km away) and Pusa DPCC at 28.64, 77.15 (agrees with CPCB).

**CPCB vs preprint, all 38 rows** (scratchpad `step1/vs_preprint.py`): only Pusa IMD differs by > 2 km (5.59 km).
The other 37 agree within 0.62 km (2-decimal rounding alone allows up to 0.74 km). The preprint also matches CPCB where
CPCB disagrees with our old geocodes (North Campus, Najafgarh, Mundka), so it was probably taken from CPCB too: it is a
consistency check, not an independent source.

**Check** (`../.venv/bin/python <scratchpad>/step1/nbrs.py data_pipeline/stations/delhi.csv ihbas imd_lodhi nsit pusa,`):
```
37 stations
pairs with identical coordinates: [('pusa/dpcc', 'pusa/iitm')]
nearest-neighbour km: min 0.00; 11th neighbour: median 9.39, max 24.73; max pair 39.4
ihbas:          vivek_vihar 1.6, sonia_vihar 6.1, patparganj 6.6, chandni_chowk 7.8, ito 8.4, ... lodhi_road/iitm 13.0
imd_lodhi_road: lodhi_road/iitm 0.7, jln_stadium 1.4, mdc_stadium 2.4, nehru_nagar 3.5, ito 4.3, ... chandni_chowk 7.2
nsit_dwarka:    dwarka_sec8 5.7, mundka 9.4, igi_t3 9.8, najafgarh 10.6, pusa/dpcc 11.6, ... rohini 16.2
pusa (both):    each other 0.0, shadipur 1.3, north_campus 2.3, punjabi_bagh 4.1, ... lodhi_road/iitm 9.3
```

### Step 1 analysis (before the decision)

**Source found:** CPCB, *List of CAAQM Stations (All India)*, https://app.cpcbccr.com/ccr_docs/caaqms_list_All_India.pdf
(sha256 `72bd8763…5513`; PDF dated 2026-07-07; says it is a snapshot of CPCB's CCR station directory as published on
data.gov.in; 4 decimals ≈ 11 m). Cross-check at 2 decimals: supplement of egusphere-2024-1811 (Copernicus preprint), table of sites.

| Our station | CPCB row (code) | CPCB lat, lon | Preprint (2 dp) | Today |
|---|---|---|---|---|
| `ihbas,_dilshad_garden,_delhi_-_cpcb` | IHBAS, Dilshad Garden, Delhi - CPCB (#114) | 28.6812, 77.3025 | 28.68, 77.30 | 28.6139, 77.209 (fallback), 11.8 km off |
| `imd_lodhi_road,_delhi_-_iitm` | Lodhi Road, Delhi - IMD (#109) | 28.5918, 77.2273 | 28.59, 77.23 | fallback, 3.0 km off |
| `nsit_dwarka,_delhi_-_cpcb` | NSIT Dwarka, Delhi - CPCB (#115) | 28.6091, 77.0325 | 28.61, 77.03 | fallback, 17.2 km off |

- Which Lodhi Road is which (from our own data): `imd_lodhi_road` has PM2.5 from 2018-01-01 = the IMD station (#109);
  `lodhi_road,_delhi_-_iitm` starts 2020-11-13, the same day as `chandni_chowk,_delhi_-_iitm` = the IITM station (#5395).
  The 2019 CPCB list has only one Lodhi Road (IMD). Our `_iitm` files are CPCB's "- IMD" stations except these two.
- **Pusa:** CPCB lists DPCC (#1563) and IMD (#107) at the same point, 28.6396, 77.1463 (ours: 28.6363, 77.1576, 1.2 km away).
  The preprint has Pusa IMD at 28.61, 77.10. Hourly PM2.5 correlation DPCC vs IMD 2019–2023: r = 0.79, below DPCC pairs
  within 3 km (0.88–0.95) but IMD/IITM stations correlate weakly with all neighbours (0.61–0.67), so the data cannot decide.
- **Other stations also differ from CPCB by 2–7 km** (name geocodes): Najafgarh 6.97, North Campus 6.42, Mundka 4.50,
  Bawana 3.16, Rohini 3.11, Narela 3.08, IGI T3 2.83, Aya Nagar 2.34, Wazirpur 2.24 km; 26 others 0.05–1.5 km.
  Indirect check (Spearman of PM2.5 correlation vs distance, all pairs): today +0.097, CPCB −0.023. Clear gain for the
  three fallback stations; for the other nine the per-station differences are small and mixed.
- Check run (scratchpad `step1/nbrs.py`): fixing only the 3 → no identical pairs except Pusa; 11th-neighbour median
  9.25 km (was 8.30), max 21.48. All 37 from CPCB → only Pusa identical; median 9.39, max 24.73, widest pair 39.4 km.
  Fixing only the 3 puts `imd_lodhi_road` 0.1 km from `lodhi_road/iitm` (mixed sources); with all-CPCB it is 0.7 km.
- Proposed location of the v2 list: `data_pipeline/stations/delhi.csv` (tracked without a `.gitignore` change;
  columns `station_id, lat, lon` + provenance `source, cpcb_name, cpcb_code`; both pipeline scripts read by column name).

## Step 4: B7 patch merged ✅ (2026-11-04)

**Files changed:** `aqfm/config.py`, `layers.py`, `model.py`, `evaluate.py` (patch hunks as they are);
`aqfm/data.py`, `aqfm/train.py` (patch + the two failed hunks merged by hand + fixes below); new `tests/test_aqfm_v2.py`.
Applied with `git apply --reject --directory=aqfm /home/phalak/aqfm/v2_inputs/aqfm_v2_changes.patch`; the `.rej` files were
merged by hand and deleted. Every current option is kept (`--no-blh` in `make_sample` and train.py, `--pm25-direct`, `--random-masking`).

**Fixes on top of the patch (user-approved):**
1. `TrainWindows`: one anchor pool and no `--train-months` → v1 draw order (t0, then anchor). Patch order changed every v1 window.
2. train.py validation recipe: the v1 lines are kept verbatim (deep_baselines execs them); with a hold-out a second
   block redraws the pairs from the seen stations only (same rng seed 12345).
3. `parse_holdout` has no seed: per city, `default_rng(crc32(city name)).permutation(city's stations)[:n]`. Same stations
   for every training seed, whatever other cities are loaded, and nested (`delhi:4` ⊇ `delhi:3`).
   On the 37-station order of `data/preprocessed/`: **`delhi:4` → indices [4, 7, 32, 35] = Chandni Chowk (IITM), DTU (CPCB),
   Sirifort (CPCB), Vivek Vihar (DPCC)**; `delhi:3` → [4, 7, 35]. (Applies to `data/delhi/` if step 2 keeps that order — to check.)
4. `--train-months N` (now an int) = exactly the last N calendar months before validation, windows included
   (`train_months_start`; 1 → Dec 2023, 12 → all of 2023). A target-city station is also never a *neighbour* in an
   earlier window (matters only if two cities are within 100 km). The anchor is never dropped by that filter.
5. `--train-months` now also works with `--no-balance` (looked up by the anchor's city, not the sampling pool).
6. With a hold-out, everything fitted from data uses the seen stations only: RevIN fallback stats and met stats
   (`AQData(..., stats_stations=allowed)`) and the **outage fit** (`fit_outages`, the three v1 lines moved into a helper).
   The outage fit: for each station, a step is "out" when fewer than half of the 9 CPCB pollutant channels are observed;
   runs of ≥ 16 steps (one 4-h patch) on the training split are collected and a LogNormal is fitted to their lengths
   (v1: μ = 3.932, σ = 1.030, n = 3033, median 12.8 h). Pre-training draws its simulated station outages (block masks) from it.
   **Not covered:** `static.npy` (made by `preprocess_aqfm.py`: z-scored across all stations, and features 0–1 come from
   each station's own PM history). It cannot be re-fitted exactly here (z-scores skip zero placeholders; raw values are
   not stored). B3 replaces these facts with a fixed scaling rule, which removes the issue.

**Checks:**
```
../.venv/bin/python -m pytest tests -q                                    52 passed (46 old + 6 new)
../.venv/bin/python -m pytest deep_baselines/tests deep_baselines/estgcn/tests -q   18 passed
```
- v1 equivalence (`<scratchpad>/step4/equiv.py`, run on `git archive main aqfm baselines` and on the branch; CPU float32):
  runs/main_s0_do3/best.pt on 16 fixed validation windows → **545 of 545 arrays bit-identical** (batches, logits, mu,
  log_sigma, RevIN mean/std, every loss part, 24 pre-training TrainWindows samples, RevIN fallback stats, met scaling).
  The only new arrays are `st_valid` (all True).
- `python -m aqfm.evaluate --ckpt runs/main_s0_do3/best.pt --device cpu --seed 0 --limit-origins 5` on both trees:
  **every number identical** (149 shared values; the 4 that "differ" are NaN on both sides: PM2.5 24 h Good and
  Satisfactory bands have n = 0 in 5 January origins). PM2.5 MAE 25.8005 / 48.3413 / 59.0886 / 67.4413 on both.
  The branch adds keys (`cities`, `stations`, new cfg/args) and warns "checkpoint has no met_stats (pre-v2)" (expected).
- Outage fit helper on data/preprocessed = the v1 line exactly: (3.9324581259871136, 1.0298001703439896, 3033),
  the values stored in runs/main_s0_do3/config.json.
- **Parameters:** `AQFMConfig.city()` = **1,172,271**; with every v2 switch off = 1,169,924 (the prompt's number), so the
  v2 parts add 2,347. Also: v1 23,801,736; `city(share_axial_blocks=True)` 907,051; `city(d192, 6 heads, d_ff 384, 3 blocks)` 3,188,155 (guide: 23.80M / 1.17M / 0.91M / 3.19M).
- Smoke runs (CPU, 50 steps, `--out` in the scratchpad): `--synthetic --small --preset city --data x`: train total
  2.523 → 1.940, val 1.9117 → 1.8849; `--data x y` (cities `['city0','city1']`): 2.274 → 1.913, val 1.9148 → 1.8862. Finite, falling.
- Mutation check of the new tests (each bug put back in a scratch copy): v2 draw order for v1, `--no-balance` lookup,
  no neighbour filter, stats ignore hold-out, outage fit ignores hold-out, old 30.44-day months, shared-rng hold-out →
  **each caught by a test**. (The `--no-balance` one was first missed: the neighbour filter could drop the anchor; fixed.)

**Noted, not changed:** the all-True decoder memory mask (`st_valid`) is bit-identical on CPU; GPU bf16 not compared (v1
GPU training was not bit-reproducible run to run anyway). Results JSONs gain keys. Fine-tuning keeps the checkpoint's
outage parameters (no outage fit outside `--phase main`).

## Step 5: B5 hidden stations out of validation ✅ (2026-11-04)

**Files changed:** `aqfm/data.py` (`FixedWindows(..., allowed=None)` passes the mask to `neighbourhood`),
`aqfm/train.py` (validation `FixedWindows` gets `allowed` when there is a hold-out, else `None` = v1),
`tests/test_aqfm_v2.py` (+1 test).

**Test** `test_holdout_stations_never_in_validation_or_training`: runs `train.main()` (synthetic, `--holdout 1,6`) and
records the stations of every validation and training sample: no hidden station in any slot (anchor, neighbour or
padding); `held_out_stations` saved. Fails when the fix is reverted (checked in a scratch copy).

**Measured on the real 37 stations** (`delhi:4` hold-out [4, 7, 32, 35], city radius 100 km, 512 validation windows,
v1 coordinates): patch as delivered → **97.7 %** of validation groups contained a hidden station (9.1 % of all slots);
with the fix → **0.0 %**.

```
../.venv/bin/python -m pytest tests -q          53 passed
deep_baselines tests                             18 passed
```

## Step 7: city-independence and fine-tune tests ✅ (2026-11-04)

**Files changed:** `tests/test_aqfm_v2.py` only (+5 tests; 12 in the file). No code change in this step.

| Test | What it shows |
|---|---|
| `test_city_preset_is_translation_invariant_and_v1_is_not` | `AQFMConfig.city()`: every longitude shifted by −4.4° or +10° → logits/mu/log_sigma equal within 1e-4. `AQFMConfig()` (v1): differ by > 1e-2 |
| `test_padded_slots_change_nothing` | anchor with 4 neighbours within 100 km (7 padded slots): padded slots have no observed context, no forecast or imputation targets, invalid queries; random values in their x_ctx, static, met_ctx, met_hz, y_hz change neither the real stations' outputs nor the loss (atol 1e-6) |
| `test_finetune_freeze_keeps_frozen_weights_bit_identical` | `--freeze encoder` and `--freeze backbone`: every frozen parameter bit-identical after fine-tuning; some trainable ones moved |
| `test_finetune_keeps_checkpoint_stats_unless_refit` | fine-tune on other data keeps the checkpoint's `met_stats` and RevIN fallback stats; `--refit-stats` refits both; checkpoint stores `cities`, `init` |
| `test_train_months_targets_the_last_city` | `--data x y z --train-months 1` limits city 2 only (with step 4's two `--train-months` tests) |

Mutation check (scratch copy): city preset back to `coord_mode="delhi"`, no decoder memory mask, padded queries valid,
freeze freezing nothing, fine-tune always refitting stats → **each caught**.

**Freeze shares (city preset, measured):** none 1,172,271 (100 %); encoder 632,367 (53.9 %); backbone 76,079 (6.5 %).
Guide: ~54 % and ~7 %.

**Found, not changed (patch behaviour):** `local_xy_km` centres the flat map on the mean latitude of *all* slots,
padded ones included. Padded slots always carry the anchor's coordinates, so the result is deterministic, but the centre
moves towards the anchor when slots are padded. Effect if padded coordinates were moved: max |Δmu| of real stations
1.3e-4 (0.5° north), 8.1e-4 (3°). Fix would be to pass `st_valid` into `local_xy_km`; not done (decision for you).

**No Delhi constants** (`grep -n -i -E "28\.6|77\.2|delhi" aqfm/*.py`), what remains and why:
- `model.py:29-31,60` `station_features` (lat/lon − (28.6, 77.2), ×10): used only when `coord_mode == "delhi"`, the v1 default (kept so v1 reproduces).
- `config.py:111-122` the v1 default `coord_mode="delhi"` and its comment; `config.py:151` docstring.
- `data.py:335-336` `synthetic_raw` places toy stations around (28.6, 77.2): test/synthetic data only.
- Comments/docstrings only: `data.py:44-45,209`, `physical.py:6`, `layers.py:143,164`, `train.py:8-16,102-105`.
- The city-preset path has none (the translation test proves the model does not depend on absolute position).
- `data_pipeline/` is still Delhi-only (`fetch_era5.py` AREA and file names, `generate_stations_csv.py` ", Delhi, India"
  + 28.6139/77.209 fallback, `preprocess_aqfm.py:415-417` "Delhi center" fallback): step 2, waiting for the raw data.

```
../.venv/bin/python -m pytest tests -q          58 passed
deep_baselines tests                             18 passed
v1 probe (equiv.py, branch vs main)              545 of 545 arrays bit-identical
```

## Step 8: short Delhi run ✅ (2026-11-04)

**Files changed:** none in code. New run folder `runs/v2_short_delhi/` (config.json, log.jsonl; `*.pt` git-ignored) and
console log `runs/v2_short_delhi.txt`. Fine-tune / eval outputs only in the scratchpad (`step8/`).

**Run** (GPU, data/preprocessed = v1 coordinates; hold-out = the `delhi:4` stations passed as indices because the city
is named `preprocessed` here; warmup shortened from the default 2000 to fit 3000 steps):
```
../.venv/bin/python -m aqfm.train --phase main --preset city --data data/preprocessed --holdout 4,7,32,35 \
    --out runs/v2_short_delhi --device cuda --batch-size 8 --steps 3000 --warmup 300 --val-every 500 --compile --seed 0
```
- 1.17M parameters, bf16, `--compile` works with the v2 layers: **0.10 s/step, 5.1 min** in total (v1: 0.38 s/step).
- Outage fit on the 33 seen stations: μ 3.940, σ 1.023, n 2567 (all 37: 3.932 / 1.030 / 3033). Met scaling AT 24.81 ± 8.17 °C.
- Train total, mean per 500 steps: 2.091, 1.926, 1.895, 1.874, 1.794, 1.814 — all finite. Validation score
  (NLL + 0.5 CRPS) improved at every check: 1.8672, 1.8221, 1.8068, 1.7987, 1.7890, **1.7878** (step 3000).
- `best.pt` contains `met_stats`, `cities` (`['preprocessed']`), `held_out_stations` ([4, 7, 32, 35]), `init`, `code`.
- `aqfm.evaluate --limit-origins 5` runs end to end: `--stations all` (37 anchors) and `--stations held_out` (4 anchors).
  Zero-shot path on a scratch city (`--data <scratchpad>/cities/minicity`, the first 10 stations' arrays sliced out of
  data/preprocessed) prints "transfer evaluation" and uses the checkpoint's met stats.
- Fine-tune from `best.pt` on that scratch city, 300 steps each (`--phase finetune --freeze none|encoder|backbone`;
  encoder also `--train-months 3` → steps [201504, 210336) = Oct–Dec 2023, 92 days exactly):

  | freeze | trainable | frozen tensors bit-identical | trainable tensors moved | met_stats / RevIN kept |
  |---|---|---|---|---|
  | none | 1.172M | – (0 frozen) | 177 / 177 | yes / yes |
  | encoder | 0.632M | 69 / 69 | 108 / 108 | yes / yes |
  | backbone | 0.076M | 150 / 150 | 27 / 27 | yes / yes |

  `--phase anneal` from the encoder fine-tune also runs (it trains all parameters: `--freeze` defaults to none).
  The Delhi hold-out is correctly not carried over to a different city.

**Observation to check before step 9 (not a failed check):** on the 5-origin evaluation, PM2.5 1 h MAE is 50.8 against
33.9 for persistence on the same points (v1 final model: 25.8); at 6 h / 24 h it is far better than persistence (71.9 vs 97.5,
71.7 vs 100.1). The anchor is on (`use_last_obs` True, gate weights learned, mean |w| 0.0067 vs v1's 0.0031). Likely
undertraining (3000 steps), but v1 had the same short-lead failure before the anchor, so a validation short-lead
diagnostic against persistence should be part of step 9. The validation curve is behind v1's at equal steps
(1.7878 vs 1.7624 at 3000), on a different validation set (hidden stations excluded), with a 20× smaller model.

## Decisions received 2026-11-04 (second message)

- Leave both open items as they are: `local_xy_km` centring on padded slots, and `static.npy` in hold-out fits (goes with B3).
- B6: do both (rename + true virtual sensing). B3: replace the four facts with location-only facts from free global
  datasets (downloads allowed). B2 code now, checks when the raw data arrives. Step 9 settings fixed (see step 9).
- From now on: no waiting between steps; stop only on a failed check or a build-prompt mismatch. Never commit or push.

## Step 6: B6 unseen-station tests ✅ (2026-11-04)

**Files changed:** `aqfm/evaluate.py`, `tests/test_aqfm_v2.py` (+2 tests).
- `--stations held_out` is now reported as **"new station that has its own history"** (`res["test"]`; help text says so).
- New `--virtual-sensing`: **"virtual sensing (no readings at the station)"** — the anchor's 10 pollutant channels and the
  two fractions built from them are hidden for the whole 128 h context (`_FixedBlock(0, context)`, the mask the outage
  test already used). The outage-imputation test is skipped in this mode (nothing left to impute against).
- **My choice (not specified):** with `--stations held_out`, the *other* hidden stations are not neighbours either, so each
  hidden station is tested next to stations seen in training only (`res["neighbours_exclude_held_out"] = True`).
- **Still visible to a "virtual" station:** its own weather channels (AT, RH, wind, SR, BP, rain: CPCB sensor readings
  gap-filled with ERA5) and, until B3 lands, its `static.npy` facts, two of which come from its own PM history.
  A place without a sensor would have ERA5 weather only.

**Tests:** `test_virtual_sensing_never_sees_the_anchors_own_pollutants` (end-to-end `evaluate()`: scrambling the anchor's
past pollutant readings leaves the virtual-sensing result unchanged, and changes the normal one);
`test_held_out_tests_use_only_seen_neighbours_and_honest_names`. Mutation check (scratch copy incl. `baselines/`;
control copy passes): option ignored → caught; hidden stations as neighbours → caught.

**Smoke run** on `runs/v2_short_delhi/best.pt`, `--limit-origins 5`, 4 hidden stations (numbers only show the paths run):
new station with own history PM2.5 MAE 40.2 / 63.0 / 80.2 / 67.5 (1/6/24/72 h); virtual sensing 105.8 / 62.2 / 106.8 / 90.0.
Real numbers come from the step 9 hold-out run.

```
../.venv/bin/python -m pytest tests -q          60 passed
```

## Step 3: B3 station facts ✅ facts computed (2026-11-04); used when `data/delhi/` is built (step 2)

**Decision (user):** replace the four facts with location-only facts from free global datasets: elevation, built-up and
green share within 1 km, distance to the coast; one fixed scaling rule; easy to swap; `n_static` stays 4.

**Files changed:** new `data_pipeline/station_facts.py` (fact registry `FACTS`, `DEFAULT_FACTS`, `scaled_static`),
new `data_pipeline/stations/delhi_facts.csv` (38 stations × 4 raw facts), new `data_pipeline/stations/SOURCES.md`
(sources, licences, required notices, file hashes, definitions), `tests/test_aqfm_v2.py` (+1 test).

| Fact | Definition | Fixed scaling (all cities) | Source / licence |
|---|---|---|---|
| elevation | median Copernicus DEM GLO-30 height within 1 km (m) | m / 1000 | Copernicus WorldDEM-30 licence (free; notice required) |
| built-up share | share of ESA WorldCover 2021 pixels within 1 km = Built-up | as is | CC BY 4.0 |
| green share | share within 1 km = tree, shrub, grass, herbaceous wetland, mangrove (not cropland) | as is | CC BY 4.0 |
| distance to coast | great-circle km to the nearest Natural Earth 1:10m coastline segment | log10(1 + km) / 3 | public domain |

Licences checked on the licence texts themselves (Copernicus PDF Art. 4–6; ESA WorldCover repo; Natural Earth terms). The
Copernicus notice and liability sentence are in `SOURCES.md` because the repo is public and the CSV holds derived values.

**Tools:** rasterio is not in the project venv, so facts are computed in a separate venv outside the repo,
`/home/phalak/aqfm/.venv_geo` (rasterio 1.5.2 / GDAL 3.12.2, numpy 2.5.4, pandas 3.1.0, affine 3.0.1). The project
venv is unchanged; `preprocess_aqfm.py` only reads the CSV. Command:
`../.venv_geo/bin/python data_pipeline/station_facts.py --stations data_pipeline/stations/delhi.csv --out data_pipeline/stations/delhi_facts.csv` (2 s).

**Delhi values** (38 stations): elevation 207.7–262.9 m (ridge sites highest: Aya Nagar, Karni Singh range); built-up
0.04 (Najafgarh) – 0.95 (Chandni Chowk); green 0.04 (Chandni Chowk) – 0.84 (Karni Singh range); coast 802–837 km.
Checks: coast distance vs brute-force nearest vertex (ITO 823.8 vs 829.0 km; Colaba, Mumbai 0.71; Chennai Marina 0.22;
Pune 87.3); capped at the exact nearest-vertex distance (no Delhi value changed). A Mumbai elevation lookup stops with
"no tile ... download it first" (no silent default).

**Test** `test_station_facts_fixed_scaling_and_size`: `len(DEFAULT_FACTS) == n_static == 4`; a station's scaled facts are
identical whether computed alone or with all 38 (no per-city statistic); the scaling rules give 1000 m → 1.0, 999 km → 1.0;
a missing station or NaN fact is an error. 61 tests pass.

**Note:** with these facts Delhi's stations differ little in elevation and coast distance (both nearly constant within
Delhi); they matter for transfer (Mumbai: ~0 km from the coast, ~10 m elevation), which is their purpose.

## Step 2: B1 data steps for any city — code ✅; real-data checks 1, 3, 4 ✅, 2 ❌ (2026-11-04)

**Files changed:** `data_pipeline/preprocess_aqfm.py`, `interpolate_era5_to_stations.py`, `fetch_era5.py`,
`fix_era5_zips.py`, `generate_stations_csv.py`; new `tests/test_pipeline_v2.py` (5 tests). Every new option defaults to
the v1 behaviour, except that a missing coordinate is now an error (the build prompt's rule).
- `preprocess_aqfm.py`: `--stations --raw-dir --era5 --out` (defaults: `data/stations.csv`, `data/yearly_data`,
  `data/era5_interpolated.parquet`, `data/preprocessed`; `ROOT` is the repo, not `c:\Users\phala\Foundation Model`).
  Station without coordinates → `ValueError` (was "Delhi center" with a warning). `--bp-rule threshold|era5`
  (default threshold = v1). `--static legacy|facts --facts <csv>` (default legacy = v1). Report keys `bp_rule` and
  `static_facts` appear only with the non-default options, so a default run writes the v1 report.
- `interpolate_era5_to_stations.py`: `--stations --era5-dir --out` (v1 defaults). A station outside the ERA5 grid now
  stops the run (`OutsideERA5Box`); before, it got NaN weather silently. Other per-month errors still only print (v1).
- `fetch_era5.py`: `--city --area N W S E --start --end --out-dir` (defaults: Delhi box, `era5_delhi_*`, 2021-01..2025-12,
  `data/era5_data_raw`); `fetch_month(year, month, out_dir)` still works as before. Not run (no CDS download).
- `fix_era5_zips.py`: optional folder argument. `generate_stations_csv.py`: `--raw-dir --region --out` (out required,
  so it can never overwrite `data/stations.csv`); a name not found stops with an error (no retry without the region, no
  default point). Not run (Nominatim is a web API; v2 uses CPCB coordinates anyway).

**Pressure-unit rule (`--bp-rule era5`, proposal now implemented as an option; v1 rule stays the default):** a BP
reading is hPa when it is closer to ERA5's surface pressure in hPa than to ERA5's surface pressure in mmHg, at that
station and time (`bp_hpa_mask_era5`; runs per station inside `merge_era5`, before gap-filling, like v1's rule runs
before the merge). ERA5 follows altitude, so it works where "> 900" fails. Where ERA5 is missing at a step, the
station's median ERA5 pressure is used; a station with no ERA5 at all is an error. Real-data check: it converts 6,290 readings
more than v1's 5,410,316 (below, check 2).

**Left as is (Delhi-safe, but not for every altitude — your call):** `PHYSICAL_BOUNDS` drops BP < 600 mmHg (true
pressure is below that above ~2,400 m, e.g. Shimla) and AT < −15 °C (Ladakh winters). Not changed, so Delhi stays identical.

**Checks run now (synthetic):**
- `tests/test_pipeline_v2.py`: a fake city (3 stations: lowland hPa, lowland mmHg, 1,500 m hPa; CPCB-format CSVs + ERA5
  parquet) built from arguments only; v1 rule converts the lowland hPa readings and wrongly keeps 845 hPa at 1,500 m;
  the era5 rule converts the same lowland readings (identical arrays) and the hill readings too; rule correct at sea
  level, 1,500 m and 3,500 m; facts become the static input; two output folders stack with `load_multi`; missing
  coordinates and stations outside the ERA5 box stop the run; geocoder stops on a missing name. Mutation check: each
  reinstated v1 behaviour (">900" in the era5 rule, Delhi-centre fallback, no box check, facts ignored) is caught.
- **Defaults = v1 on the same fake city:** `main`'s own `preprocess_aqfm.py` (from `git archive main`, path constants
  pointed at the fake city) vs the branch with default options → values, observed, timestamps_h, coords, static,
  era5_met **all identical**, report identical (`<scratchpad>/step2/v1_vs_branch.py`).

```
../.venv/bin/python -m pytest tests -q          66 passed
deep_baselines tests                             18 passed
```

### Step 2 real-data checks (2026-11-04) — 1, 3, 4 ✅; 2 ❌ (pressure rule) → decision: v1 rule for Delhi

**Raw data:** Drive folder `16TvrpmukdnnUqp8vzQOdqJsiUrFtg1-H` (3 zips + `SHA256SUMS.txt`), downloaded with gdown into
`/home/phalak/aqfm/raw_inputs/` (outside the repo); all three sha256 match `SHA256SUMS.txt`:
`aqfm_yearly_data_2018_2025.zip` a58035da…8842 (291 CSVs, 1.35 GB unpacked; 35/36/36/37/36/37/37/37 stations in
2018…2025), `aqfm_era5_data_raw.zip` 4fbe30a3…8a8a (100 NetCDFs, 2017-01..2025-12, none zip-in-nc),
`aqfm_era5_interpolated.zip` 704ba6e1…70a3 (the parquet + a `stations.csv` that equals the repo's except CRLF line
endings; the repo copy was kept). Moved into `data/` (git-ignored). Alipur has no CSVs (37 of 38 stations, as in v1).
Commands, logs and the analysis scripts: `<scratchpad>/step2real/`.

1. ✅ **Defaults reproduce v1.** `preprocess_aqfm.py --out <scratch>/default` (15 s): values, observed, static, coords,
   timestamps_h, era5_met **bit-identical** to `data/preprocessed/` (dtype, shape, bytes); report identical.
   `interpolate_era5_to_stations.py --out <scratch>/era5_default.parquet` (11 s): pyarrow `Table.equals` the v1 parquet
   (same schema, same row order, 11,990,862 rows).
2. ❌ **Pressure rule.** `--bp-rule era5` converts **5,416,606** readings, not 5,410,316 (build prompt §step 2: "must convert
   exactly the same Delhi readings"). Every reading v1 converts, the era5 rule also converts (v1-only: 0); the era5 rule
   converts **6,290 more**, all raw values 849–900 at 19 stations (3,250 at Ashok Vihar 2018-02..2023-06, 1,656 at
   Sonia Vihar, 406 Vivek Vihar, 277 Sri Aurobindo Marg, …). They are impossible in both units: as mmHg they are
   122–175 mmHg above ERA5 (station medians ~737 mmHg), as hPa 67–124 hPa below ERA5 (~985 hPa). v1 keeps them as
   850–900 mmHg (inside the 600–1100 mmHg bound, so not dropped); the era5 rule makes them 637–675 mmHg. Both are wrong.
   Of 5,930,665 observed BP readings that is 0.11%. A tolerance rule cannot reproduce v1 exactly either: v1 itself
   converts readings from 96 hPa below to 128 hPa above ERA5 (raw 900–1113), which overlap the 6,290
   (`bp_analysis.py`: tolerance 30–70 hPa never gives v1's set). Only side effect in a legacy-static build: static
   column 2 (the height clue from mean BP) changes; `data/delhi/` uses the facts, so not there.
3. ✅ (built with the era5 rule, pending 2) **`data/delhi/`**: Delhi interpolation (11 s) →
   `data/delhi_era5_interpolated.parquet`; preprocess with `--bp-rule era5 --static facts` (15 s). Station order and
   channel orders equal `data/preprocessed/`'s. Differences from `data/preprocessed/`, all expected:
   - `coords` (CPCB), `static` (facts: elevation/1000 0.208–0.263, built 0.041–0.952, green 0.038–0.836,
     coast 0.968–0.974); report keys `bp_rule`, `static_facts`, `bp_hpa_converted`.
   - `era5_met`: all 8 channels at all 37 stations (weather at the new points). Mean |diff| AT 0.03 °C, RH 0.22 %,
     U/V 0.03/0.02 m/s, BP 0.02 mmHg, BLH 4.4 m; largest at NSIT Dwarka (moved 17 km: AT 0.23 °C, RH 1.6 %, BLH 29 m).
   - `values`, station weather: differences **only at the ERA5-filled gaps** (checked against the raw-CSV gap mask:
     AT 4,446,323 of 4,446,914 gaps, RH, WS, WD, SR, TOT_RF likewise, 0 at real readings), plus the 6,290 BP readings
     of check 2; BLH channel everywhere. **All pollutant channels identical; `observed` identical; timestamps identical.**
   - Note: the final `observed` marks ERA5-filled weather as observed (`observed[...] = 1.0` in `merge_era5`), so the
     fill positions can only be recovered from the raw CSVs.
**Decision (user, 2026-11-04 14:35): `data/delhi/` uses v1's threshold rule** (rebuilt without `--bp-rule era5`). The
ERA5 rule stays an option for cities above ~990 m (not needed for Delhi or Mumbai). Rebuilt `data/delhi/`: converts
exactly 5,410,316 readings; vs the era5-rule build only those 6,290 BP values differ (observed, era5_met, static,
coords, timestamps identical); vs `data/preprocessed/` BP differs only at the 4,383,112 ERA5-filled gaps; report
differs from v1 only in `static_facts`. The era5-rule build is kept at `<scratchpad>/step2real/delhi_era5rule/`.

4. ✅ **Second city from arguments alone:** a 4-row station file (Alipur, Bawana, Mundka, Narela) → its own ERA5
   parquet → `--out <scratch>/city2/northdelhi --bp-rule era5 --static facts` (3 stations with data). `load_multi(
   [data/delhi, northdelhi])` → groups `['delhi', 'northdelhi']`, 40 stations, sizes [37, 3]; the 3 stations' values,
   observed, era5_met, static, coords equal their rows in `data/delhi/` exactly.

## Step 9: full Delhi training — launched 2026-11-04 14:36

`nohup setsid bash scripts/pipeline_v2_delhi.sh > runs/v2_pipeline.log 2>&1 &` on the rebuilt `data/delhi/` (v1 BP rule).
Runs record `code: bd396a8 + uncommitted changes` (branch not committed).

### Step 9 results (chain 14:36–16:41, no failures) — full tables: `results/v2_delhi_summary.md`

Generated from the results files; its v1 and baseline rows equal `results/TABLES.md`. Trainings: 17.2 / 17.2 / 14.5 /
17.1 / 17.1 min (seed 1 early-stopped at 10k). Dropout: 0.3 (1.7372) vs 0.15 (1.7380) — a tie, like v1's sweep.
Validation score per seed: v2 1.7372 / 1.7494 / 1.7370 (1.7412 ± 0.0071) vs v1 1.7299 / 1.7290 / 1.7255 (1.7281 ± 0.0023);
v2's best steps 9k / 7k / 11k (v1: 5k / 5k / 6k).

PM2.5 MAE, 2025 test (µg/m³), 1 / 6 / 24 / 72 h, Severe 24 h:
- v2 ensemble 16.37 / 28.29 / 33.90 / 37.44, 88.3 — v1 ensemble 15.71 / 27.54 / 32.04 / 36.51, 89.0.
- v2 single 16.59 ± 0.25 / 28.77 ± 0.66 / 34.48 ± 0.24 / 38.27 ± 0.54, 89.2 ± 0.2 — v1 16.08 ± 0.17 / 28.13 ± 0.08 /
  33.22 ± 0.76 / 37.70 ± 1.21, 91.5 ± 0.6.
- So v2 (1.17M, 5% of v1's parameters) is 0.5–1.9 µg/m³ (2–6%) behind v1 at every lead, better on Severe days; PM10
  likewise (ensemble 24 h 61.25 vs 58.78). v2 still beats XGBoost + future ERA5 at 1 h, 6 h and Severe; behind it at
  24 h (33.90 vs 33.65) and 72 h (37.44 vs 36.45). 1 h: v2 ensemble 16.37 vs persistence 15.89 (v1 single seeds are
  also above persistence at 1 h; only v1's ensemble is below it).
- Probabilistic, PM2.5 24 h (ensembles): CRPS 24.80 vs 23.66; 90 % coverage 0.899 vs 0.914; 95 % 0.942 vs 0.951.

Unseen stations (hold-out run, stations 4, 7, 32, 35; other hidden stations are not neighbours):
- **New station that has its own history**: 15.71 / 26.70 / 35.86 / 40.36 vs the same stations seen in training
  (all-station model, seed 0) 15.62 / 25.98 / 34.28 / 39.74 → +0.1 to +1.6. Works.
- **Virtual sensing (no readings at the station)**: 60.59 / 56.09 / 62.17 / 60.95 — fails. References on exactly the
  same scoring points (point counts equal; `<scratchpad>/step9/vs_reference.py`, not a results file): mean of the 3
  nearest seen stations' latest readings 42.48 / 47.29 / 58.52 / 65.13; nearest seen station's actual value at the
  target time (not a forecast) 46.23 / 40.67 / 42.72 / 42.91. Likely cause (from the code, not tested): with no visible
  readings, `MaskedRevIN` falls back to the global training mean/std for that station, and training never shows an
  anchor with an empty context, so the model has no learned way to take the level from neighbours. Fixing it needs a
  training change (whole-station context dropout, neighbour-based normalisation) — a design decision.

Nearest-neighbour bands (ensembles; v1 on `data/preprocessed/`, same stations): 0–5 km (30 stations) v2 15.80 / 27.69 /
33.13 / 36.75 vs v1 15.23 / 27.28 / 31.33 / 35.80; 5–15 km (7) v2 18.78 / 30.49 / 37.30 / 39.66 vs v1 17.85 / 28.69 /
35.40 / 38.86. Both models are ~2–4 µg/m³ worse at the more isolated stations; the v1–v2 gap is similar in both bands.

**Settings (user, 2026-11-04):** v1's: 12,000 steps, batch 8, warmup 2,000, 512 validation windows (+ v1's weight
decay 0.05, patience 3, `--compile`, validation every 1,000 steps); dropout 0.15 vs 0.3 on seed 0 by validation score;
then seeds 1, 2 and the pooled ensemble; main models on all 37 stations, no hold-out; one separate run with the
4-station hold-out (`delhi:4` → [4, 7, 32, 35]) for the two unseen-station tests; same scoring code; comparison with v1
and the baselines; nearest-neighbour distance bands.

**Files:** new `scripts/pipeline_v2_delhi.sh` (the chain; refuses to start unless `data/delhi/` exists and has the same
station order as `data/preprocessed/`), new `scripts/v2_summary.py` (writes `results/v2_delhi_summary.md` from the result
files only; v1 and baseline rows come from the same files as `results/TABLES.md`).
Outputs: `runs/v2_s{0,1,2}_do*`, `runs/v2_holdout4_s0_do*`, `results/v2_*_results.json`, `results/v2_seeds_stats.json`,
`results/v2_delhi_summary.md`. Also evaluates the reference: the all-station model on the 4 hidden stations, and the v1
ensemble per distance band (on `data/preprocessed/`, same stations).
Launch: `nohup setsid bash scripts/pipeline_v2_delhi.sh > runs/v2_pipeline.log 2>&1 &`. Estimate ~3 h (5 trainings
× ~25 min at 0.1 s/step + evaluations).

**Dry run** (`DRY=1`: 30 steps, 2 origins, on `data/preprocessed/`): the whole chain ran end to end in 2 min, exit 0
(dropout pick, seeds, evaluations, ensemble, seed stats, hold-out run and both unseen-station tests, reference, bands,
summary). The summary's v1 and baseline rows equal the published ones (v1 ensemble PM2.5 15.71 / 27.54 / 32.04 / 36.51,
Severe 89.0; XGBoost + future ERA5 17.99 / 28.48 / 33.65 / 36.45, Severe 93.4). Dry outputs deleted afterwards.

**Distance bands (D1)** with the CPCB coordinates: 30 stations have a neighbour within 5 km, 7 within 5–15 km (Aya Nagar,
Bawana, Mundka, Najafgarh, Narela, NSIT Dwarka, Sonia Vihar), **none beyond 15 km** — Delhi has no sparse station, so the
> 15 km band is empty (the chain skips it). With v1's coordinates: 28 / 9 / 0.

### v2 without future weather — launched 2026-11-04 21:06 (user: "Yea train that also")

All step 9 v2 checkpoints (and v1's) have `use_future_met = True`: they are given the ERA5 weather for the forecast
hours, like XGBoost + future ERA5. This run is v2's counterpart of v1's no-future-met rows (`results/aqfm_do3_nofm_*`):
same settings, dropout 0.3 (chosen in step 9), `--no-future-met`, seeds 0, 1, 2, each seed + mean ± std + 3-seed
ensemble. New `scripts/pipeline_v2_nofm.sh` (log `runs/v2_nofm_pipeline.log`); `scripts/v2_summary.py` now also prints
the no-future-weather rows (v2 and v1) and a line saying which rows get future weather. Test before launch (scratchpad,
30 steps, 2 origins): checkpoint `use_future_met = False`, evaluation ran, the summary's v1 no-future-weather rows equal
`results/TABLES.md` (ensemble 15.77 / 27.97 / 38.41 / 43.20). Estimate ~1 h.

Seed 0 took 25.3 min (step 9 runs: ~17): a CPU test run started alongside it (stopped after 15 min) slowed it.
Code record: the v2 code was committed on `aqfm-v2` during this chain (user, 2026-11-04). Runs and evaluations started
before that record `bd396a8 + uncommitted changes` (all of step 9, nofm seeds 0 and 1); later ones record the new commit.
The code under `aqfm/`, `data_pipeline/` and `tests/` is the same in both (last edited 12:07, before step 9).

## Status against §8 "Done when"

- [x] Branch `aqfm-v2` has the patch merged, all old tests pass, v1 equivalence shown (step 4).
- [x] B4, B1 (code; real-data checks pending), B5 done with tests; B3 (facts) and B6 (both tests) done after your decisions.
- [x] `data/delhi/` and a second city from arguments alone: built and checked (step 2); `data/delhi/` uses v1's BP rule
  (decision after check 2).
- [x] The v2 tests pass (66 tests in `tests/`: 46 v1 + 15 in `test_aqfm_v2.py` + 5 in `test_pipeline_v2.py`).
- [x] Parameter count of `AQFMConfig.city()`: 1,172,271.
- [x] The short Delhi run and the fine-tune runs work (step 8).
- [x] Full Delhi training: done 2026-11-04 16:41 (`results/v2_delhi_summary.md`; step 9 results above).

**Changed / new files on `aqfm-v2` (nothing committed):** `aqfm/{config,data,evaluate,layers,model,train}.py`;
`data_pipeline/{preprocess_aqfm,interpolate_era5_to_stations,fetch_era5,fix_era5_zips,generate_stations_csv}.py`;
new `data_pipeline/station_facts.py`, `data_pipeline/stations/{delhi.csv,delhi_facts.csv,SOURCES.md}`,
`tests/test_aqfm_v2.py`, `tests/test_pipeline_v2.py`, `scripts/pipeline_v2_delhi.sh`, `scripts/v2_summary.py`,
`runs/v2_short_delhi/` + `runs/v2_short_delhi.txt`, `docs/V2_PROGRESS.md`. Outside the repo: `/home/phalak/aqfm/v2_inputs/`,
`/home/phalak/aqfm/.venv_geo`, `data/global/` (git-ignored downloads).

### Step 4 prep (before the go)

- The patch is `diff -ruN aqfm_orig/ aqfm/`, so it needs `git apply --directory=aqfm`. All files are LF here.
- `git apply --check`: config.py, evaluate.py, layers.py, model.py apply (small offsets). Fail: `data.py` hunk at
  `make_sample` (no `use_blh` block in the patch) and `train.py` config/data section (no `--pm25-direct` / `--random-masking`).
- `aqfm_v2.zip` has no tests.
- Found while reading (to handle in the merge, or to decide):
  1. TrainWindows draws anchor before t0 → with no v2 flags every training window differs from v1 (0/1000 equal).
     Breaks "defaults stay v1". Fix: keep the v1 draw order when there is one city and no `--train-months`.
  2. The new validation lines break `deep_baselines` test `test_validation_pairs_match_aqfm` (it execs train.py's
     recipe lines; `allowed` is undefined there). The pairs themselves are identical (`rng.choice(arange(S)) == rng.integers(S)`).
     Fix: keep the v1 lines; draw hold-out pairs in a separate branch.
  3. `parse_holdout` seeds the hold-out with the training seed → seeds 0/1/2 would hide different stations.
  4. `--train-months N` lets window *starts* cover N months, so the data seen is N months + one window (8.3 days).
  5. `--train-months` has no effect with `--no-balance` (group lookup key is `None`).
  6. RevIN fallback stats, met stats and the outage fit still use hold-out stations' training-period data (small leak).

## Open questions

- Virtual sensing fails (step 9, ~60 MAE vs 42–65 for neighbour references): fix in training (whole-station
  context dropout + neighbour-based normalisation) or report as a limitation — your call.
- Data quality, not changed: 6,290 Delhi BP readings (raw 849–900) are impossible in both units and are kept as v1 kept
  them; a plausibility filter would be a separate change for v1 and v2 alike.
- Left as is by decision (2026-11-04): `local_xy_km` centring on padded slots; `static.npy` in hold-out fits (gone with B3).
- My choice, reversible: in both unseen-station tests the other hidden stations are not neighbours.
- For later cities (not Delhi): BP < 600 mmHg and AT < −15 °C are dropped as implausible in `preprocess_aqfm.py`
  (wrong above ~2,400 m / in Ladakh winters); the station-facts choice can still change before step 9.
- Step 10 protocol: anneal after a frozen fine-tune unfreezes everything unless `--freeze` is passed again.
