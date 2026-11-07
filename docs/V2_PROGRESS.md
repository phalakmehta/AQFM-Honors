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
v2's best steps 9k / 7k / 10k (v1: 5k / 5k / 6k; corrected 2026-11-04 from "11k", checked in `runs/v2_s2_do03.txt`).

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

### v2 without future weather — chain 21:06–22:20, no failures (user: "Yea train that also")

Validation score per seed 1.8115 / 1.8092 / 1.7943 (best steps 6k / 9k / 11k; with future weather 1.7372 / 1.7494 /
1.7370). PM2.5 MAE, 2025 test (µg/m³), 1 / 6 / 24 / 72 h, Severe 24 h (all 13,394 windows, `virtual_sensing` False):
- v2 ensemble 16.35 / 29.08 / 38.43 / 42.40, 102.7 — v1 no-future-weather ensemble 15.77 / 27.97 / 38.41 / 43.20, 104.9.
- v2 single 16.65 ± 0.67 / 30.02 ± 1.56 / 39.47 ± 1.30 / 43.66 ± 0.89, 104.2 ± 3.8 — v1 16.00 ± 0.42 / 28.33 ± 0.10 /
  38.98 ± 0.24 / 43.77 ± 1.04, 105.7 ± 1.2 (v2's seeds vary more).
- So without future weather v2 is behind v1 at 1 h (+0.58) and 6 h (+1.11), level at 24 h (+0.02), ahead at 72 h
  (−0.80) and on Severe days (−2.2). PM10 ensemble 29.20 / 48.30 / 72.83 / 81.22 vs v1 28.46 / 46.02 / 72.49 / 81.98.
- Future weather is worth 4.5 (24 h) and 5.0 (72 h) to v2 (33.90 / 37.44 with it) vs 6.4 and 6.7 to v1: most of v2's
  gap to v1 in step 9 is in how much it gains from future weather (observation; cause not tested).
- Against the baselines without weather (`deep_baselines/README.md`: PatchTST and E-STGCN see no weather): v2 beats
  XGBoost (no future met) 18.17 / 30.06 / 38.63 / 43.74 at every lead and E-STGCN 16.37 / 30.91 / 38.64 / 44.62 at
  6–72 h; PatchTST 16.26 / 28.04 / 37.75 / 43.31 is better than v2 (and v1) at 1, 6 and 24 h, v2 better at 72 h.
  Persistence 15.89 is better at 1 h.
- PM2.5 24 h (ensembles): CRPS 28.11 vs v1 27.96; 90 % coverage 0.905 vs 0.912; 95 % 0.949 vs 0.953.
Tests after the chain (GPU free): `pytest tests` 66 passed; deep_baselines tests 18 passed.

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

### Housekeeping (2026-11-05, user: "Do housekeeping")

- Commits (as the user, no Claude lines): `277a666` no-future-weather results + log; `39ea48b` v2 in
  `scripts/manifest.json` (7 `data/delhi/` files, 8 models, 17 results; old entries unchanged, checked by parsing)
  and the 9 v2 training runs in `results/TABLES.md` (`python scripts/make_tables.py`); then this commit.
- Model `code` in the manifest: `8be6172` for the runs that record `bd396a8 + uncommitted changes` (`git diff
  8be6172 HEAD -- aqfm data_pipeline tests` is empty; code files last edited 12:07, training from 14:36), `a356a6c`
  for nofm seed 2. The checkpoints keep their true `train_code`.
- Step-9 results re-scored under the clean commit `39ea48b`: `python scripts/check_submission.py --skip-tests
  --out reproduce_out/2026-11-05_v2step9 --reproduce <the 12 step-9 results>` → all 12 **IDENTICAL** (351–410 numbers
  each, 41 min: 01:01–01:42). In the 11 evaluation files only `code` was replaced (now `39ea48b`, clean); `seeds_stats` has no code
  field and is unchanged. Manifest `code` for those 11: `39ea48b`. `results/v2_delhi_summary.md` and `TABLES.md`
  regenerate unchanged.
- v2 checkpoints copied for Drive: `/home/phalak/aqfm/drive_upload_2026-11-05_v2/models/` (8 files, 37 MB,
  `SHA256SUMS.txt` = manifest; README); manifest `drive` names added. `SUBMISSION.md`: v2 section.
- `python scripts/check_submission.py` (full, with tests): everything OK, 66 tests pass; left: uncommitted files
  (before this commit) and "GitHub main is at this commit" (the check compares with `main`; v2 is on `aqfm-v2`).
- Not pushed (this PC has no GitHub login); the user pushes `aqfm-v2` from VS Code.

### Improving the metrics: longer training and medium width — launched 2026-11-05 15:08 (user: "run 1 and 2")

Why: v2 underfits (train loss ≈ validation score, 1.73 vs 1.74; dropout 0.15 and 0.3 tie), its validation score
improves until 9–10k and flattens as the cosine schedule ends at 12k; v1 overfits (best at 5k, worse by 8k).
New flags `--d-model/--n-heads/--d-ff` in `aqfm/train.py` (default: the preset's; 74 tests pass; committed `f104c7c`
on the user's OK). `scripts/pipeline_v2_size_steps.sh`, log `runs/v2_size_steps_pipeline.log`, seed 0, one change each:
- A `runs/v2_s0_do03_24k`: 24,000 steps (patience 24, so early stopping cannot cut the schedule short).
- B `runs/v2m256_s0_do03`: d_model 256, 8 heads, d_ff 512 = 4,572,219 parameters; 12,000 steps, patience 3.
- Compared with `runs/v2_s0_do03` (validation 1.7372) on the 2024 validation score; the winner gets seeds 1, 2.
  Then 2025 test: `results/v2_s0_24k_results.json`, `results/v2m256_s0_results.json`.
- If the PC loses power: move the interrupted run's folder aside first (`train.py` appends to `log.jsonl`, so a rerun
  into the same folder would mix two logs), then rerun the script; a finished run A can be skipped by editing the script.

**Results (chain 15:08–16:20, no failures; all records `f104c7c`, clean).** A took 35.3 min, B 29.6 min.

| seed 0 | params | best step | validation | PM2.5 1h | 6h | 24h | 72h | Severe 24h | CRPS 24h | cov90 |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 (12k) `runs/v2_s0_do03` | 1.17M | 9000 | 1.7372 | 16.36 | 28.01 | 34.27 | 38.84 | 89.3 | 25.07 | 0.894 |
| A: 24k steps | 1.17M | 16000 | 1.7291 | 16.58 | 29.49 | 35.05 | 38.50 | 90.2 | 25.66 | 0.874 |
| B: width 256 | 4.57M | 10000 | **1.7268** | 16.11 | 28.64 | 33.44 | 37.88 | 91.3 | 24.51 | 0.897 |
| v1 `runs/main_s0_do3` | 23.8M | 5000 | 1.7299 | 16.25 | 28.06 | 33.97 | 38.74 | 91.0 | 25.00 | 0.906 |

PM10 (1/6/24/72 h): v2 29.60 / 45.99 / 62.96 / 70.58; A 29.49 / 48.00 / 63.35 / 69.37; B 29.09 / 47.36 / 60.89 / 67.82;
v1 29.80 / 47.01 / 62.87 / 69.67.
- Winner on validation: **B** (also below v1 seed 0 on validation). A's validation gain did not carry to the test
  (worse at 1–24 h); its validation curve is noisy (1.730 at 11k, 1.750 at 14k, 1.729 at 16k).
- One seed only. v2's seed-to-seed std (PM2.5): 1h 0.25, 6h 0.66, 24h 0.24, 72h 0.54, so B's 24 h gain (−0.83) is
  > 3 std, its 72 h gain (−0.96) ~2 std, its 6 h loss (+0.63) ~1 std. Seeds 1 and 2 of B decide (~1 h).
- Seeds 1, 2 of B launched 2026-11-05 16:34 (user: "Yea Ok"): `scripts/pipeline_v2m256_seeds.sh`, log
  `runs/v2m256_seeds_pipeline.log` → `runs/v2m256_s{1,2}_do03`, `results/v2m256_{s1,s2,ens3}_results.json`,
  `results/v2m256_seeds_stats.json`. Code record `f104c7c` (clean).

**3-seed results (chain 16:34–17:57, no failures; training and evaluation records `f104c7c`, clean).** Validation:
v2m256 1.7268 / 1.7248 / 1.7286 (best steps 10k / 12k / 11k); v1 1.7299 / 1.7290 / 1.7255; v2 1.7372 / 1.7494 / 1.7370.

| PM2.5 MAE | 1h | 6h | 24h | 72h | Severe 24h | CRPS 24h | cov90 / cov95 |
|---|---|---|---|---|---|---|---|
| v2m256 ensemble (4.57M) | 15.99 | 27.95 | 32.26 | 36.95 | 87.9 | 23.63 | 0.894 / 0.939 |
| v2 ensemble (1.17M) | 16.37 | 28.29 | 33.90 | 37.44 | 88.3 | 24.80 | 0.899 / 0.942 |
| v1 ensemble (23.8M) | 15.71 | 27.54 | 32.04 | 36.51 | 89.0 | 23.66 | 0.914 / 0.951 |
| v2m256 single, mean ± std | 16.14 ± 0.09 | 28.27 ± 0.32 | 32.76 ± 0.81 | 37.36 ± 0.46 | 88.4 ± 2.8 | | |
| v1 single, mean ± std | 16.08 ± 0.17 | 28.13 ± 0.08 | 33.22 ± 0.76 | 37.70 ± 1.21 | 91.5 ± 0.6 | | |

PM10 ensembles: v2m256 28.94 / 45.79 / 58.74 / 65.83; v1 28.88 / 45.50 / 58.78 / 64.93; v2 29.42 / 46.33 / 61.25 / 67.18.
PM10 single mean: v2m256 29.03 / 46.30 / 59.65 / 66.66; v1 29.41 / 46.54 / 61.05 / 67.51.
- v2m256 vs v2 ensemble: better at every lead (PM2.5 −1.2 to −4.9%, PM10 −1.2 to −4.1%).
- v2m256 vs v1 ensemble: PM2.5 +0.7 to +1.8% (was +2.6 to +5.8% for v2), Severe −1.3% (best of the three),
  CRPS equal; PM10 within ±0.6% at 1–24 h, +1.4% at 72 h. Single-model mean is below v1's at 24 h, 72 h and Severe
  (PM2.5) and at every lead (PM10). 19% of v1's parameters.
- Coverage slightly under the nominal 90/95% (0.894 / 0.939), as for v2.
- Not run for v2m256: the unseen-station / virtual-sensing hold-out, distance bands.

### v2m256 without future weather — chain 2026-11-05 18:21–20:18, no failures (user: "Yea do without future also")

`scripts/pipeline_v2m256_nofm.sh` (seeds 0–2, `--no-future-met`, same settings as the v2m256 seeds), log
`runs/v2m256_nofm_pipeline.log` → `runs/v2m256_nofm_s{0,1,2}_do03`, `results/v2m256_nofm_{s0,s1,s2,ens3}_results.json`,
`results/v2m256_nofm_seeds_stats.json`. Evaluation records `f104c7c` (clean). Training 29.0–29.4 min each;
validation 1.7928 / 1.7898 / 1.7865 (best steps 10k / 11k / 11k); v1 nofm 1.7986 / 1.8015 / 1.8035.

| PM2.5 MAE, no future weather | 1h | 6h | 24h | 72h | Severe 24h | CRPS 24h | cov90 / cov95 |
|---|---|---|---|---|---|---|---|
| v2m256 ensemble (4.57M) | 15.92 | **27.67** | 38.09 | **41.54** | 105.1 | 27.84 | 0.910 / 0.951 |
| v2 ensemble (1.17M) | 16.35 | 29.08 | 38.43 | 42.40 | 102.7 | 28.11 | 0.905 / 0.949 |
| v1 ensemble (23.8M) | **15.77** | 27.97 | 38.41 | 43.20 | 104.9 | 27.96 | 0.912 / 0.953 |
| PatchTST, 3 seeds | 16.26 | 28.04 | **37.75** | 43.31 | 110.6 | | |
| E-STGCN, 3 seeds | 16.37 | 30.91 | 38.64 | 44.62 | **98.7** | | |
| v2m256 single, mean ± std | 16.00 ± 0.17 | 28.01 ± 0.50 | 38.80 ± 1.13 | 42.62 ± 0.60 | 106.1 ± 3.8 | | |

PM10 ensemble: v2m256 28.66 / 45.87 / 72.27 / 79.33; v1 28.46 / 46.02 / 72.49 / 81.98; v2 29.20 / 48.30 / 72.83 / 81.22.
- v2m256 vs v1 without future weather: better at 6 h (−0.30), 24 h (−0.32), 72 h (−1.66, −3.8%); +0.15 at 1 h;
  Severe tie (+0.2); CRPS slightly lower (27.84 vs 27.96). Best AQFM model without future weather.
- vs PatchTST: better at 1, 6, 72 h; PatchTST still lowest at 24 h (37.75 vs 38.09). E-STGCN still best on Severe.
- Results PDF updated (outside repo: `reports/aqfm_v2/AQFM_v2_results.pdf`, 4 pages; old one kept as
  `AQFM_v2_results_2026-11-04.pdf`). The 8-page report was not regenerated (its text is from 4 Nov).

## Step 10: Mumbai (2026-11-06) — stages A–D and the same-point comparison done

**Data (user: "this is the only data we have of bombay, so whatever u want to do, do on this data"; and "our model
has to adapt to every format").** Drive folder `1nmzzrQp5JUzagS9pQFAPbLqjJIynlXQR`, copied read-only (notebooks never
run) to `/home/phalak/aqfm/raw_inputs/mumbai_drive_2026-11-06/`. Used: `mumbai_all_data_2324_28locs.npy`
(sha256 `bb355573…3330`), float64 (17,544 h, 28 stations, 15 columns) = hourly 2023-01-01 .. 2024-12-31, built by a
PM2.5-GNN Colab notebook from `*_combined_cleaned.csv`: PM2.5, PM10, NO2, O3, AT, RH, WS, WD, NOx (CPCB) + ERA5 BLH,
K-index, u10, v10, sp, tp. Found:
- **Gaps filled**: 0 % missing anywhere; 16.3 % of PM2.5 hours are the 2nd+ value of a run of >= 3 identical hours
  (13.7 % in runs >= 24 h) — copied forward. Detectable, so hidden again (only the first value of a run kept).
- **ERA5 columns are UTC**: BLH peaks at row-hour 8, station AT / O3 at 13 / 14 (IST). Shifted +5:30 (C3); after the
  shift BLH peaks at 13:00 IST.
- **Station order** = `Mumbai_28_loc.csv` order: on PM10 deviations from the city median, the co-located partner is
  the best match in 4 of 6 co-located pairs (chance 1/27 each); PM10 similarity falls with distance (Spearman −0.25).
- **Coordinates**: 19 stations are in CPCB's list (same PDF as Delhi, sha256 `72bd8763…5513`) and match exactly,
  except **site_5397 (Khindipada-Bhandup West): file 19.0992, 72.5534 (in the sea), CPCB 19.1653, 72.9221** (39.4 km)
  → CPCB used. 9 stations (site_5807 … site_5964) are not in CPCB's list → file coordinates (all on land in Mumbai).
  `data_pipeline/stations/mumbai.csv`. Geometry: nearest neighbour median ~1.6 km; one station > 15 km (Vasai West,
  17.5 km).
- **Broken met sensors** (median > 5 robust SD from the other stations, automatic rule): AT at site_5104 (38.2 °C) and
  site_5403 (1.7 °C); RH at site_5399 (27 %), site_5400 (2.4 %), site_5403 (14.5 %); WS (+WD) at site_5113 (25.6 m/s),
  site_5403 (8.0), site_5413 (5.1). Dropped for the whole record (wind then comes from ERA5, as Delhi's gap-fill).
- Missing compared with Delhi: 15-min resolution, NO, SO2, NH3, CO, Benzene, station SR/BP/rain; ERA5 t2m, d2m, ssrd
  (so future AT, RH, SR are missing for the with-weather models); 2018–2022 and 2025.
- Station facts (`data_pipeline/stations/mumbai_facts.csv`, tiles N18/N19 E072 DEM + WorldCover N18E072 downloaded
  under the B3 permission; URLs and sha256 in `data_pipeline/stations/SOURCES.md`): elevation 1–69 m, coast 0.02–5.5 km (Delhi ~800 km).

**New code (uncommitted):** `data_pipeline/convert_hourly_city.py` (any city given as a processed hourly array +
column map; hourly value on the hh:45 step = the step the evaluation scores, other 3 steps unobserved; flat runs,
PHYSICAL_BOUNDS, met-outlier rule; ERA5 shift + 15-min interpolation + `_era5_derive` + gap-fill on hourly steps;
BLH at every step); `aqfm/train.py` + `aqfm/evaluate.py`: optional `--val-start` / `--test-start` dates (defaults =
the old Jan-1 years, unchanged); `scripts/city_baselines.py` (classical baselines of `baselines/run_baselines_classical.py`
for any folder/test start; that file is unchanged); `scripts/pipeline_mumbai_zeroshot.sh`; 2 tests in
`tests/test_pipeline_v2.py` (converter: placement, flat runs, broken-sensor drop, ERA5 shift/interpolation, gap-fill
hourly only, BLH; two cities stack; date splits). 86 tests pass. Mutation check: no ERA5 shift, no flat-run rule,
readings on hh:00 → each caught (a 4th mutation was a no-op, so it shows nothing).

Built: `data/mumbai/` (28 × 18 × 70,176 steps, 2023-01-01 .. 2024-12-31 23:45 IST; 0.7 s). Kept per channel after
the rules: PM2.5 83.7 %, PM10 84.7 %, NO2 84.5 %, O3 86.2 %, NOx 85.1 %, AT 80.3 %, RH 66.2 %, WS 72.5 %, WD 75.1 %.

**Mumbai protocol (my choice; the data has no 2025):** training / fine-tuning data = 2023 (`--train-months` N = the
last N months of 2023), validation = 2024-01-01 .. 06-30, test = 2024-07-01 .. 12-31 (`--test-start 2024-07-01`;
182 origins × 28 stations). Delhi stays 2018–2023 / 2024 / 2025.

**Classical baselines** (`scripts/city_baselines.py`, `results/mumbai_classical_results.json`), PM2.5 MAE 1/6/24/72 h:
persistence 4.37 / 9.85 / 11.87 / 16.83; seasonal naive 12.04 / 10.84 / 11.89 / 12.15; climatology 19.90 / 17.74 /
19.48 / 19.67 (n ≈ 3,880). Mumbai's H2-2024 PM2.5 is far lower than Delhi's (monsoon), so compare skill relative
to persistence, not absolute MAE across cities.

**Zero-shot (Delhi models, no Mumbai training; `scripts/pipeline_mumbai_zeroshot.sh`, log
`runs/mumbai_zeroshot_pipeline.log`; 3-seed ensembles; 5,068 windows).** First run 15:11–15:27 (3 of 4 recorded
uncommitted code); rerun 15:27–15:46 under `a00d49d` (clean): all 4 × 93 numbers identical. PM2.5 MAE 1/6/24/72 h:

| | 1 h | 6 h | 24 h | 72 h | CRPS 24 h | cov90 24 h |
|---|---|---|---|---|---|---|
| v2-M, no future weather | 6.16 | 10.17 | 14.52 | 18.37 | 10.60 | 0.880 |
| v2, no future weather | 6.24 | 11.19 | 15.22 | 18.77 | 11.31 | 0.903 |
| v2-M, with future weather (AT, RH, SR missing) | 9.66 | 14.10 | 22.37 | 23.68 | 17.30 | 0.583 |
| v2, with future weather (AT, RH, SR missing) | 6.52 | 10.26 | 15.43 | 17.51 | 10.97 | 0.858 |
| persistence (n 3,888 vs AQFM 4,103) | 4.37 | 9.85 | 11.87 | 16.83 | | |
| seasonal naive | 12.04 | 10.84 | 11.89 | 12.15 | | |

- **Zero-shot does not beat persistence in Mumbai** (and loses to seasonal naive at 24/72 h). With-weather models get
  3 of 8 future-weather inputs as "Delhi average" and suffer (v2-M badly: coverage 0.58).
- 24 h outage imputation: AQFM 10.6–13.8 vs linear interpolation 9.56 (in Delhi AQFM beat linear).
- Point sets differ: the classical baselines count a point only where all three have a forecast (Delhi's rule), so
  they score ~5 % fewer points than AQFM. Before any paper number: score both on the same points.

**XGBoost on Mumbai** (`scripts/city_xgboost.py --hourly`: the code, features and settings of
`baselines/run_baselines_xgboost.py`, unchanged file; every lookup reads the latest hourly reading at or before the
step — without that, the most recent lag (hh:00) is never observed and no forecast is valid). Same points as AQFM
(n 4,103 at 24 h). PM2.5 MAE 1/6/24/72 h: trained on everything before the test (2023 + 2024 H1, Delhi's rule)
6.62 / 8.86 / 12.15 / 16.22; on the last N months of 2023 only: 1 month 13.51 / 29.27 / 27.16 / 30.02;
3 months 7.50 / 12.16 / 16.09 / 22.58; 6 months 7.06 / 9.61 / 12.69 / 16.74; 12 months 6.88 / 9.04 / 12.30 / 16.72
(`results/mumbai_xgboost[_mN]_results.json`, log `runs/mumbai_xgboost.txt`). Not yet checked that the script
reproduces Delhi's published XGBoost numbers without `--hourly` (~20 min CPU; do when the GPU chain is done).

**Fine-tune setup (validation = 2024 H1 only).** `scripts/pipeline_mumbai_transfer.sh`.
- Stage A (16:13; lr 1e-4, 3,000 steps, validation every 250, 3 months, seed 0), best validation score:
  v2 + freeze backbone **1.9213**, v2 none 1.9231, v2-M backbone 1.9228, v2 encoder 1.9252, v2-M encoder 1.9470,
  v2-M none 1.9551 → v2 (1.17M) + frozen backbone (6.5 % trainable). Margin ≤ 0.002 over the next two: a close call.
  Every run peaked at its first validation (step 250).
- So the chain (stages B–D, launched 16:14) was stopped after 1 minute (partial run moved to the scratchpad) and
  stage A2 run (16:15–16:28): the Delhi model untouched scores **1.9637**; lr 1e-4 → 1.9207 at step 150 then flat;
  **lr 3e-5 → 1.9196 at step 550**; lr 1e-5 → 1.9209 at step 1,250 (still creeping). Fine-tuning helps (−0.044).
- Recipe now: fine-tune lr 3e-5, 1,500 steps, warmup 100, validation every 50, patience 8; from scratch: the Delhi
  recipe but validation every 250 and patience 8 (small data may overfit early). Size and freeze stay from stage A.
- **Stages B → C → D relaunched 16:29** (SIZE=v2 FREEZE=backbone; logs `runs/mumbai_transfer_{B,C,D}.log`; ~7 h):
  B = seed 0 fine-tune vs scratch at 1/3/6/12 months; C = seeds 1, 2 + mean ± std + ensembles; D = joint Delhi +
  Mumbai from scratch, 3 seeds, scored on Mumbai 2024 H2 and on Delhi's usual 2025 test. Runs record `a00d49d` (clean).

**Stage B (1,500-step fine-tunes) finished 17:29; seed-0 validation:** fine-tuned / from scratch: 1 month 1.9322 /
2.0431; 3 months 1.9196 / 1.9790; 6 months 1.9120 / 1.9462; 12 months 1.8983 / 1.8907. Best fine-tune steps 200, 550,
650 and **1,500 = the limit** for 12 months; scratch best steps 500–2,750 (limit 12,000, not binding).
- **Budget check (validation only; launched 17:19, before any 12-month AQFM test result existed):** the 12-month
  fine-tune with 6,000 steps reaches **1.8782 at step 4,400** (`runs/mumbai_ft_v2_backbone_m12_steps6000_s0`;
  46 min because the machine was overloaded, see below) — better than both the capped run and from scratch.
- So the chain was stopped at 18:06 (stage C was at its 2nd fine-tune) and **fine-tuning now gets 6,000 steps for
  every data amount** (early stopping ends small-data runs earlier). The 1,500-step runs and their seed-0 test
  results are kept, renamed `runs/mumbai_ft1500_*` and `results/mumbai_ft1500_*` (logs `runs/mumbai_transfer_B_ft1500.log`,
  `runs/mumbai_transfer_C_ft1500_stopped.log`); from-scratch runs are unaffected and reused.
- The script now skips runs that already finished and results that exist (resume after a stop or power cut).
  **Relaunched B → C → D at 18:07.**
- **Folder collision found and fixed:** stage A and stage B both wrote `runs/mumbai_ft_v2_backbone_m3_s0`; stage B
  appended to stage A's `log.jsonl` and replaced its config, checkpoint and console log (stage B's own results were
  fine). Split back: the first 31 log lines (stage A: steps 1–1,250, 5 validations, best 1.9213, as printed in
  `runs/mumbai_transfer_A.log`) are now `runs/mumbai_stageA_ft_v2_backbone_m3_s0/` (with a README); all stage-A runs
  are renamed `runs/mumbai_stageA_*` and the script writes stage A there.
- **Machine overload:** the low-priority XGBoost check (`n_jobs=-1`) plus two training processes pushed the load to
  ~53 on 24 cores and slowed training ~6× (0.6 s/step, GPU 1 %). The check was stopped (rerun later, alone).

**Stages B + C finished 20:22 (6,000-step fine-tunes; no failures; every run and result records `a00d49d`, clean).**
Test = Mumbai 2024-07-01 .. 12-31 (181 origins, 5,068 windows). Tables generated from `results/mumbai_{ft_v2_backbone,
scratch_v2}_m{1,3,6,12}_{ens3_results,seeds_stats}.json`, `results/mumbai_xgboost_m*_results.json`,
`results/mumbai_classical_results.json` and the runs' `log.jsonl` (generator in the session scratchpad; the
validation scores are checked against each run's printed `best val total`).
PM2.5 MAE (3-seed ensemble; ± = sd of the 3 single seeds), 1 / 6 / 24 / 72 h:

| months | method | 1 h | 6 h | 24 h | 72 h | CRPS 24 h | cov90 24 h |
|---|---|---|---|---|---|---|---|
| 1 | fine-tuned (Delhi v2, frozen backbone) | 5.86 ± 0.27 | 10.38 ± 0.50 | 12.28 ± 0.58 | 15.11 ± 0.45 | 9.22 | 0.928 |
| 1 | from scratch (v2) | 10.89 ± 0.48 | 10.51 ± 0.14 | 12.41 ± 0.36 | 14.08 ± 0.32 | 9.25 | 0.929 |
| 1 | XGBoost (same months) | 13.51 | 29.27 | 27.16 | 30.02 | | |
| 3 | fine-tuned (Delhi v2, frozen backbone) | 5.99 ± 0.19 | 10.03 ± 0.21 | 12.05 ± 0.24 | 14.80 ± 0.13 | 9.06 | 0.934 |
| 3 | from scratch (v2) | 6.85 ± 0.63 | 10.42 ± 0.11 | 12.80 ± 0.82 | 15.75 ± 0.82 | 9.46 | 0.908 |
| 3 | XGBoost (same months) | 7.50 | 12.16 | 16.09 | 22.58 | | |
| 6 | fine-tuned (Delhi v2, frozen backbone) | 6.17 ± 0.16 | 10.15 ± 0.31 | 12.26 ± 0.24 | 15.01 ± 0.32 | 9.24 | 0.925 |
| 6 | from scratch (v2) | 5.76 ± 0.48 | 9.08 ± 0.18 | 11.55 ± 0.25 | 14.24 ± 0.22 | 8.64 | 0.921 |
| 6 | XGBoost (same months) | 7.06 | 9.61 | 12.69 | 16.74 | | |
| 12 | fine-tuned (Delhi v2, frozen backbone) | 6.46 ± 0.28 | 10.25 ± 0.36 | 12.33 ± 0.19 | 14.95 ± 0.25 | 9.27 | 0.920 |
| 12 | from scratch (v2) | 5.39 ± 0.23 | 8.79 ± 0.15 | 11.28 ± 0.17 | 14.21 ± 0.36 | 9.64 | 0.928 |
| 12 | XGBoost (same months) | 6.88 | 9.04 | 12.30 | 16.72 | | |
| 0 | zero-shot (Delhi v2, no future weather) | 6.24 | 11.19 | 15.22 | 18.77 | | |
| 0 | persistence (~5 % fewer points) | 4.37 | 9.85 | 11.87 | 16.83 | | |
| 0 | seasonal naive (~5 % fewer points) | 12.04 | 10.84 | 11.89 | 12.15 | | |

PM10 MAE (3-seed ensemble; ± = sd of the 3 single seeds), 1 / 6 / 24 / 72 h:

| months | method | 1 h | 6 h | 24 h | 72 h | CRPS 24 h | cov90 24 h |
|---|---|---|---|---|---|---|---|
| 1 | fine-tuned (Delhi v2, frozen backbone) | 9.90 ± 0.13 | 21.03 ± 0.42 | 24.59 ± 0.52 | 30.28 ± 0.45 | 18.34 | 0.904 |
| 1 | from scratch (v2) | 21.55 ± 0.83 | 22.20 ± 0.09 | 26.62 ± 0.63 | 30.13 ± 0.47 | 19.73 | 0.888 |
| 1 | XGBoost (same months) | 26.19 | 53.83 | 48.03 | 60.49 | | |
| 3 | fine-tuned (Delhi v2, frozen backbone) | 9.87 ± 0.12 | 20.58 ± 0.46 | 24.25 ± 0.39 | 29.79 ± 0.27 | 18.11 | 0.905 |
| 3 | from scratch (v2) | 12.71 ± 1.49 | 21.95 ± 0.26 | 26.63 ± 1.48 | 31.89 ± 1.44 | 19.85 | 0.840 |
| 3 | XGBoost (same months) | 18.60 | 27.18 | 36.82 | 50.78 | | |
| 6 | fine-tuned (Delhi v2, frozen backbone) | 9.89 ± 0.13 | 20.35 ± 0.51 | 24.32 ± 0.45 | 29.96 ± 0.33 | 18.17 | 0.895 |
| 6 | from scratch (v2) | 10.97 ± 0.75 | 20.63 ± 0.07 | 25.57 ± 0.73 | 31.03 ± 0.82 | 19.14 | 0.871 |
| 6 | XGBoost (same months) | 16.15 | 22.57 | 28.65 | 35.91 | | |
| 12 | fine-tuned (Delhi v2, frozen backbone) | 9.99 ± 0.10 | 20.32 ± 0.27 | 24.24 ± 0.30 | 29.75 ± 0.28 | 18.08 | 0.891 |
| 12 | from scratch (v2) | 10.33 ± 0.25 | 20.10 ± 0.26 | 24.86 ± 0.51 | 30.56 ± 1.00 | 19.28 | 0.878 |
| 12 | XGBoost (same months) | 15.94 | 22.67 | 27.37 | 34.46 | | |
| 0 | zero-shot (Delhi v2, no future weather) | 9.88 | 21.69 | 28.24 | 34.93 | | |
| 0 | persistence (~5 % fewer points) | 10.45 | 25.05 | 26.99 | 35.73 | | |
| 0 | seasonal naive (~5 % fewer points) | 25.57 | 25.05 | 27.09 | 27.51 | | |

Best validation score (nll + 0.5 crps, 2024 H1) and best step / last step, seeds 0 / 1 / 2:

| months | fine-tuned | from scratch |
|---|---|---|
| 1 | 1.9322 (200/600); 1.9163 (600/1000); 1.8962 (650/1050) | 2.0431 (500/2500); 2.0273 (750/2750); 2.0403 (500/2500) |
| 3 | 1.9197 (400/800); 1.8965 (1900/2300); 1.9033 (650/1050) | 1.9790 (1250/3250); 1.9837 (2000/4000); 1.9889 (1750/3750) |
| 6 | 1.9117 (650/1050); 1.9007 (1500/1900); 1.8993 (2350/2750) | 1.9462 (2500/4500); 1.9279 (2250/4250); 1.9260 (1750/3750) |
| 12 | 1.8782 (4450/4850); 1.8795 (4450/4850); 1.8808 (3400/3800) | 1.8907 (2750/4750); 1.8766 (3750/5750); 1.8792 (2750/4750) |

- **No training budget was binding:** every run ended by early stopping (fine-tune best step ≤ 4,450 of 6,000;
  from scratch ≤ 3,750 of 12,000).
- **Delhi pretraining helps most with little Mumbai data.** With 1 month, PM2.5 at 1 h: fine-tuned 5.86 vs from
  scratch 10.89; PM10 is better fine-tuned at every lead with 1–3 months. XGBoost trained on the same 1–3 months is
  far worse (PM2.5 24 h: 27.16 / 16.09).
- **PM2.5 with 6+ months: from scratch is better at every lead** (12 months, 24 h: 11.28 vs 12.33). The fine-tuned
  model stays at ~12.0–12.3 (24 h) whatever the amount, while from scratch improves 12.41 → 11.28. Likely cause
  (inference, not tested): only 6.5 % of the weights train (frozen backbone, chosen in stage A on 3 months only).
  On validation the 12-month fine-tune was slightly better (seed mean 1.8795 vs 1.8822): the validation score covers
  every forecast target and the validation half-year (Jan–Jun) differs from the test half (Jul–Dec, monsoon).
- **PM10: fine-tuned beats from scratch at 24 h and 72 h for every amount** (12 months, 24 h: 24.24 vs 24.86).
- **Against the simple methods (not yet on the same points):** persistence is best for PM2.5 at 1 h (4.37 vs best
  AQFM 5.39); at 24 h AQFM is ahead (PM2.5 11.28 vs 11.87, PM10 24.24 vs 26.99); at 72 h seasonal naive beats every
  AQFM (PM2.5 12.15 vs 14.08; PM10 27.51 vs 29.75). 90 % interval coverage at 24 h: 0.84–0.93.
- Next: the same-point comparison with significance (`scripts/pipeline_paper_extras.sh` stage MUMBAI). A possible
  follow-up (not run; decide with the user): fine-tuning with nothing frozen at 6 and 12 months.

**Stage D finished 21:23 (no failures; runs and results record `a00d49d`, clean): one v2 model (no future weather)
trained from scratch on Delhi 2018–2023 + Mumbai 2023 together**, picked on both validation periods (Delhi 2024 +
Mumbai 2024 H1; best scores 1.7829 at step 9,000; 1.7888 at step 12,000; 1.7881 at step 12,000 of 12,000 — not comparable with the Mumbai-only scores).

PM2.5 MAE, 3-seed ensembles, all without future weather:

| test | model | 1 h | 6 h | 24 h | 72 h | CRPS 24 h | cov90 24 h |
|---|---|---|---|---|---|---|---|
| Mumbai | one model for both cities (joint) | 5.67 | 8.79 | 11.93 | 14.71 | 9.90 | 0.922 |
| Mumbai | fine-tuned from Delhi, 12 months | 6.46 | 10.25 | 12.33 | 14.95 | 9.27 | 0.920 |
| Mumbai | from scratch, 12 months | 5.39 | 8.79 | 11.28 | 14.21 | 9.64 | 0.928 |
| Delhi 2025 | one model for both cities (joint) | 16.08 | 29.13 | 39.43 | 42.75 | 28.62 | 0.908 |
| Delhi 2025 | v2 trained on Delhi only | 16.35 | 29.08 | 38.43 | 42.40 | 28.11 | 0.905 |
| Delhi 2025 | v2-M trained on Delhi only | 15.92 | 27.67 | 38.09 | 41.54 | 27.84 | 0.910 |

PM10 MAE, 3-seed ensembles, all without future weather:

| test | model | 1 h | 6 h | 24 h | 72 h | CRPS 24 h | cov90 24 h |
|---|---|---|---|---|---|---|---|
| Mumbai | one model for both cities (joint) | 10.03 | 19.41 | 24.78 | 30.30 | 18.48 | 0.875 |
| Mumbai | fine-tuned from Delhi, 12 months | 9.99 | 20.32 | 24.24 | 29.75 | 18.08 | 0.891 |
| Mumbai | from scratch, 12 months | 10.33 | 20.10 | 24.86 | 30.56 | 19.28 | 0.878 |
| Delhi 2025 | one model for both cities (joint) | 29.18 | 48.78 | 73.89 | 82.41 | 53.17 | 0.888 |
| Delhi 2025 | v2 trained on Delhi only | 29.20 | 48.30 | 72.83 | 81.22 | 52.52 | 0.886 |
| Delhi 2025 | v2-M trained on Delhi only | 28.66 | 45.87 | 72.27 | 79.33 | 51.94 | 0.895 |

- On Mumbai the joint model lands between the 12-month fine-tune and from scratch for PM2.5 (24 h 11.93 vs 12.33 /
  11.28) and is the best at 6 h for PM10 (19.41).
- On Delhi it costs a little against the Delhi-only v2 (PM2.5 24 h 39.43 vs 38.43, +2.6 %): adding Mumbai did not
  help Delhi.
- **Budget caveat:** seeds 1 and 2 were best at the last step (12,000), so with twice the data the Delhi schedule may
  be short for the joint model. The Delhi-only v2 runs peaked at steps 6,000–11,000. Option (not run): the joint
  model with 24,000 steps, ~1.75 h GPU (3 seeds).

**Commit `a239e03` (2026-11-06 21:25, pushed; user: "Yes to both"):** stages A–D results and logs, the evaluation
patch (91 tests pass), the paper scripts and docs. Then `scripts/pipeline_paper_extras.sh` stage MUMBAI launched
21:25 (log `runs/paper_MUMBAI.log`), recording `a239e03`.

**Stage MUMBAI finished 21:51 (no failures; all 10 AQFM files record `a239e03`, clean).** Checks: the rerun with the new
evaluation options gives every PM2.5 / PM10 number of the earlier Mumbai results exactly (7 files × 42 numbers);
the committed `scripts/city_xgboost.py` reproduces `results/mumbai_xgboost_m{1,3,6,12}_results.json` exactly (4 × 42),
and `scripts/city_baselines.py` reproduces `results/mumbai_classical_results.json` exactly (3 × 42). Same-point tables
(every method scored on the points where all of them have a forecast; generated from the files named):
PM2.5 MAE on the same points (`results/paper/mumbai_PM2.5_m<N>.json`). Each other method is marked against fine-tuned: `*` = different at p < 0.05 (Diebold–Mariano on daily means, Newey–West lag 7); **bold** = best in the row group:

| months | method | 1 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|---|
| 1 | fine-tuned | 5.86 | 10.38 | 12.23 | 15.04 |
| 1 | from scratch | 10.89* | 10.51 | 12.35 | **13.94*** |
| 1 | XGBoost (same months) | 13.51* | 29.27* | 27.11* | 29.94* |
| 1 | zero-shot | 6.24* | 11.19* | 15.14* | 18.63* |
| 1 | persistence | **4.40*** | **9.73** | **11.55*** | 16.32 |
| 3 | fine-tuned | 5.99 | 10.03 | 12.00 | **14.73** |
| 3 | from scratch | 6.85 | 10.42 | 12.74 | 15.63 |
| 3 | XGBoost (same months) | 7.50* | 12.16* | 16.03* | 22.47* |
| 3 | zero-shot | 6.24* | 11.19* | 15.14* | 18.63* |
| 3 | persistence | **4.40*** | **9.73** | **11.55** | 16.32 |
| 6 | fine-tuned | 6.17 | 10.15 | 12.21 | 14.94 |
| 6 | from scratch | 5.76 | **9.08*** | **11.49** | **14.09*** |
| 6 | XGBoost (same months) | 7.06 | 9.61 | 12.62 | 16.62 |
| 6 | zero-shot | 6.24 | 11.19* | 15.14* | 18.63* |
| 6 | persistence | **4.40*** | 9.73 | 11.55* | 16.32 |
| 12 | fine-tuned | 6.46 | 10.25 | 12.28 | 14.88 |
| 12 | from scratch | 5.39* | **8.79*** | **11.22*** | **14.08*** |
| 12 | XGBoost (same months) | 6.88 | 9.04* | 12.24 | 16.62 |
| 12 | zero-shot | 6.24* | 11.19* | 15.14* | 18.63* |
| 12 | persistence | **4.40*** | 9.73 | 11.55* | 16.32 |

Points per lead (1/6/24/72 h): 4,093 / 4,046 / 4,097 / 4,094 (days: 181).

PM10 MAE on the same points (`results/paper/mumbai_PM10_m<N>.json`). Each other method is marked against fine-tuned: `*` = different at p < 0.05 (Diebold–Mariano on daily means, Newey–West lag 7); **bold** = best in the row group:

| months | method | 1 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|---|
| 1 | fine-tuned | 9.90 | **21.03** | **24.51** | 30.12 |
| 1 | from scratch | 21.55* | 22.20 | 26.55* | **29.97** |
| 1 | XGBoost (same months) | 26.19* | 53.83* | 47.97* | 60.45* |
| 1 | zero-shot | **9.88** | 21.69 | 28.04* | 34.48* |
| 1 | persistence | 10.31 | 24.71* | 26.57* | 35.23* |
| 3 | fine-tuned | **9.87** | **20.58** | **24.18** | **29.68** |
| 3 | from scratch | 12.71* | 21.95 | 26.55* | 31.74* |
| 3 | XGBoost (same months) | 18.60* | 27.18* | 36.75* | 50.73* |
| 3 | zero-shot | 9.88 | 21.69 | 28.04* | 34.48* |
| 3 | persistence | 10.31* | 24.71* | 26.57* | 35.23* |
| 6 | fine-tuned | 9.89 | **20.35** | **24.24** | **29.81** |
| 6 | from scratch | 10.97* | 20.63 | 25.48 | 30.84 |
| 6 | XGBoost (same months) | 16.15* | 22.57* | 28.58* | 35.83* |
| 6 | zero-shot | **9.88** | 21.69* | 28.04* | 34.48* |
| 6 | persistence | 10.31* | 24.71* | 26.57* | 35.23* |
| 12 | fine-tuned | 9.99 | 20.32 | **24.14** | **29.57** |
| 12 | from scratch | 10.33 | **20.10** | 24.79 | 30.41 |
| 12 | XGBoost (same months) | 15.94* | 22.67* | 27.26* | 34.36* |
| 12 | zero-shot | **9.88** | 21.69 | 28.04* | 34.48* |
| 12 | persistence | 10.31 | 24.71* | 26.57* | 35.23* |

Points per lead (1/6/24/72 h): 4,067 / 4,034 / 4,079 / 4,074 (days: 181).

NOx MAE on the same points (`results/paper/mumbai_NOx_m<N>.json`). Each other method is marked against fine-tuned: `*` = different at p < 0.05 (Diebold–Mariano on daily means, Newey–West lag 7); **bold** = best in the row group:

| months | method | 1 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|---|
| 1 | fine-tuned | 4.58 | 7.12 | **8.53** | **10.17** |
| 1 | from scratch | 7.62* | **7.04** | 9.62* | 10.59 |
| 1 | XGBoost (same months) | 9.23* | 13.63* | 10.81* | 20.64* |
| 1 | zero-shot | **4.52** | 7.24* | 9.04* | 10.82* |
| 1 | persistence | 4.52 | 8.92* | 9.26* | 11.88* |
| 3 | fine-tuned | 4.53 | **6.90** | **8.62** | **10.23** |
| 3 | from scratch | 4.96* | 7.05 | 9.15* | 10.55* |
| 3 | XGBoost (same months) | 7.55* | 10.30* | 12.73* | 19.98* |
| 3 | zero-shot | **4.52** | 7.24* | 9.04 | 10.82 |
| 3 | persistence | 4.52 | 8.92* | 9.26* | 11.88* |
| 6 | fine-tuned | **4.48** | **6.81** | **8.60** | **10.22** |
| 6 | from scratch | 4.58 | 6.83 | 9.20* | 10.60 |
| 6 | XGBoost (same months) | 7.06* | 8.63* | 10.07* | 12.89* |
| 6 | zero-shot | 4.52 | 7.24* | 9.04* | 10.82 |
| 6 | persistence | 4.52 | 8.92* | 9.26* | 11.88* |
| 12 | fine-tuned | **4.47** | **6.90** | **8.61** | **10.23** |
| 12 | from scratch | 4.64* | 7.22* | 8.77 | 10.41* |
| 12 | XGBoost (same months) | 6.81* | 8.25* | 9.69* | 12.17* |
| 12 | zero-shot | 4.52 | 7.24* | 9.04 | 10.82 |
| 12 | persistence | 4.52 | 8.92* | 9.26* | 11.88* |

Points per lead (1/6/24/72 h): 4,201 / 4,148 / 4,200 / 4,186 (days: 181).

Ozone MAE on the same points (`results/paper/mumbai_Ozone_m<N>.json`). Each other method is marked against fine-tuned: `*` = different at p < 0.05 (Diebold–Mariano on daily means, Newey–West lag 7); **bold** = best in the row group:

| months | method | 1 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|---|
| 1 | fine-tuned | 3.67 | **6.04** | 5.94 | 6.65 |
| 1 | from scratch | 5.89* | 6.88* | 7.65* | 8.22* |
| 1 | XGBoost (same months) | 5.91* | 9.48* | 8.77* | 9.19* |
| 1 | zero-shot | 3.72* | 6.05 | **5.89** | **6.56*** |
| 1 | persistence | **3.55** | 6.75* | 6.98* | 8.53* |
| 3 | fine-tuned | 3.61 | **6.02** | 5.94 | 6.67 |
| 3 | from scratch | 4.05* | 6.10* | 6.17* | 7.06* |
| 3 | XGBoost (same months) | 5.05* | 7.34* | 7.01* | 8.65* |
| 3 | zero-shot | 3.72* | 6.05 | **5.89** | **6.56*** |
| 3 | persistence | **3.55** | 6.75* | 6.98* | 8.53* |
| 6 | fine-tuned | **3.55** | 5.99 | 5.92 | 6.63 |
| 6 | from scratch | 3.71* | **5.90** | 5.99 | 6.72 |
| 6 | XGBoost (same months) | 4.89* | 6.70* | 6.61* | 7.56* |
| 6 | zero-shot | 3.72* | 6.05 | **5.89** | **6.56** |
| 6 | persistence | 3.55 | 6.75* | 6.98* | 8.53* |
| 12 | fine-tuned | **3.51** | 6.01 | 5.96 | 6.63 |
| 12 | from scratch | 3.58 | **5.90** | 6.05 | 6.95* |
| 12 | XGBoost (same months) | 4.75* | 6.50* | 6.38* | 7.43* |
| 12 | zero-shot | 3.72* | 6.05 | **5.89** | **6.56** |
| 12 | persistence | 3.55 | 6.75* | 6.98* | 8.53* |

Points per lead (1/6/24/72 h): 4,217 / 4,174 / 4,217 / 4,213 (days: 181).

**Best AQFM against the simple methods, PM2.5** (`results/paper/mumbai_PM2.5_best_vs_simple.json`; same command with
`--model "AQFM from scratch 12 months" results/forecasts/mumbai_scratch_m12_ens3.npz --model "AQFM fine-tuned 3 months"
results/forecasts/mumbai_ft_m3_ens3.npz --model Persistence results/forecasts/mumbai_classical.npz:persistence --model
"Seasonal naive" results/forecasts/mumbai_classical.npz:snaive`; PM10 in `..._PM10_best_vs_simple.json`):
1 h persistence better (4.40 vs 5.39, p = 0.002); 6 h AQFM 8.78 vs persistence 9.71 (p = 0.23) and seasonal naive
10.58 (p = 0.001); 24 h a tie (11.21 vs 11.55, p = 0.51; seasonal naive 11.57, p = 0.49); 72 h seasonal naive better
(11.82 vs 14.06, p = 0.008). PM10: AQFM beats persistence at 6 h and 72 h (p ≤ 0.011) and seasonal naive at 1–24 h;
seasonal naive is better at 72 h (27.09 vs 30.35, p = 0.043).

**What this says (2024 H2 = monsoon and post-monsoon):**
- **Delhi pretraining transfers for NOx, ozone and PM10 at every data amount.** A Delhi-pretrained model (fine-tuned,
  or for ozone at 24–72 h the Delhi model with no Mumbai training) is best or tied at almost every lead. For NOx the
  fine-tuned model still beats from scratch with 12 months (significant at 1, 6 and 72 h).
- **PM2.5 is the exception:** fine-tuning helps with 1–3 months only. From scratch is significantly better with 12
  months at every lead, and with 6 months at 6 h and 72 h.
- **XGBoost on the same months is worse than fine-tuned AQFM in every cell for PM10, NOx and ozone and for PM2.5 with
  1–3 months** (all significant). For PM2.5 with 6–12 months it is close (better at 6 h: 9.04 vs 10.25 with
  12 months, p < 0.05).
- **Mumbai PM2.5 is hard to beat:** the best AQFM ties persistence and seasonal naive at 24 h, loses to persistence at
  1 h and to seasonal naive at 72 h.
- **Mumbai NO2 is not usable from AQFM:** 1 h MAE 15.6–17.8 against persistence 3.79 and XGBoost 5.2–7.0
  (`results/paper/mumbai_NO2_m<N>.json`). Cause: the model never reads NO2 directly; it reads NOx and the NO share,
  which needs NO, and Mumbai has no NO. So Mumbai's NO2 readings never reach the model, and the NO2 output uses a
  share the model never saw in Mumbai. Mumbai's NO2/NOx (ppb) ratio is lower and much more variable than Delhi's
  (median 0.39 vs 0.53; 7 % of hours above 1 vs 2 %). Report NO2 for Delhi only, or as a limitation. A possible fix
  (not done; needs retraining the Mumbai models): when NO is missing, take the share from NO2 and NOx.

**Delhi XGBoost reproduction check (overnight chain, 2026-11-06 21:52–21:58).** `scripts/city_xgboost.py --data
data/preprocessed` vs the published `baselines/results/xgboost_results.json`: same points (every `n` equal) but 28 of 42
numbers differ by 0.1–0.7 % (PM2.5 MAE 1/6/24/72 h 18.30 / 30.03 / 38.49 / 43.78 vs 18.17 / 30.06 / 38.63 / 43.74;
output `/home/phalak/aqfm/checks/xgb_delhi_repro.json`). The unchanged `baselines/run_baselines_xgboost.py` re-run today
(`scripts/check_submission.py --skip-tests --reproduce baselines/results/xgboost_results.json --out
/home/phalak/aqfm/checks/repro_xgb_v1`, 16 min) gives the same new numbers: **all 42 identical to `city_xgboost.py`**.
So the new script is a faithful copy; the published XGBoost file came from another software state (it was committed
2026-10-12 01:02; this venv's xgboost 3.0.5 was installed 2026-10-12 17:02). Consequences: `check_submission.py
--reproduce` reports the two v1 XGBoost files as DIFFERENT (expected, by up to 0.7 %); for the paper, compare
AQFM with the XGBoost runs made on this machine (stage SAVE, `results/paper/delhi_xgboost*.json`), which also have
their forecasts saved for the same-point tests.

## Step 11: paper experiments (SAVE, ABL, FOLD, NOISE) and the unfrozen Mumbai fine-tune — done 2026-11-07

`scripts/pipeline_paper_extras.sh` stages SAVE → ABL → FOLD → NOISE, then `scripts/pipeline_mumbai_unfrozen.sh`.
Chains: `/home/phalak/aqfm/chains/overnight_2026-11-06.sh` (SAVE and the first ablation; log
`runs/paper_overnight_chain.log`), then `/home/phalak/aqfm/chains/resume_2026-11-07.sh` (log
`runs/paper_resume_chain.log`). Times: SAVE 21:58–23:45; ABL 23:45–00:22 and 00:23–01:45; FOLD 01:45–05:07;
NOISE 05:07–05:29; unfrozen Mumbai 05:29–06:01.

**Checks (RUNBOOK §4.1):**
- Every stage log ends with `pipeline finished`.
- **SAVE:** the five ensembles (`v2m256`, `v2m256_nofm`, `v2`, `v2_nofm`, v1 `do3`) have 82/82 PM2.5/PM10 numbers
  equal to the earlier `results/*_ens3_results.json`.
- **Code state:** every AQFM result records a clean code state. SAVE and the no-spatial run record `a1a9e23`; the
  later 18 record `c07d55a` (docs, scripts and SAVE results only; the code is the same). The baseline and
  comparison files don't record a code state, as before.
- **Training:** all 15 new trainings ended with `done`. One budget is binding: FOLD with weather, seed 2, peaked at
  its last step (12,000). The Mumbai runs stopped early, best at steps 250–1,700.
- **XGBoost reproduction check:** the chain logged DIFFERENT, as expected (the 0.1–0.7 % gap to the published file;
  see the step 10 note). The XGBoost rows below were all run on this machine.
- **Spatial ablations dropped (user, 2026-11-07 00:20):** the spatial part is there so the model works for any
  city's station layout, not to improve accuracy. Stopped during the distance-bias run (moved out of the repo to
  `/home/phalak/aqfm/cancelled_runs/2026-11-07/`, not used); `--v1-spatial` not run. The no-spatial-attention run
  that had finished is kept, for reference only. A model without spatial attention also runs on any station
  layout (each station on its own), so these results do not show that the spatial part is what makes the model
  portable. Only the full model was tested on Mumbai.

Tables generated from the files named (MAE; same points unless said; p = Diebold–Mariano on daily means,
Newey–West lag 7; the first model of each table is the reference).

### SAVE: Delhi 2025 test, same points (MAE; in brackets: difference to the first model, Diebold–Mariano p)

**PM2.5, with future ERA5 weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **15.99** | **27.95** | **32.25** | **36.94** |
| AQFM v2 | 16.37 (+0.38, p=0.001) | 28.29 (+0.33, p=0.216) | 33.90 (+1.65, p=0.016) | 37.44 (+0.49, p=0.611) |
| XGBoost + future ERA5 | 17.97 (+1.98, p=0.000) | 28.62 (+0.66, p=0.250) | 33.43 (+1.17, p=0.146) | 36.53 (-0.41, p=0.820) |
| Persistence | 15.90 (-0.09, p=0.658) | 35.58 (+7.63, p=0.000) | 43.84 (+11.59, p=0.000) | 54.62 (+17.68, p=0.000) |
| points | 12210 | 11998 | 10306 | 10297 |

**PM2.5, no future weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **15.92** | **27.67** | **38.09** | **41.54** |
| AQFM v2 | 16.35 (+0.43, p=0.003) | 29.08 (+1.41, p=0.000) | 38.42 (+0.33, p=0.279) | 42.39 (+0.85, p=0.162) |
| XGBoost | 18.23 (+2.31, p=0.000) | 30.01 (+2.34, p=0.000) | 38.39 (+0.31, p=0.767) | 43.79 (+2.25, p=0.191) |
| Persistence | 15.90 (-0.02, p=0.723) | 35.58 (+7.91, p=0.000) | 43.84 (+5.76, p=0.000) | 54.62 (+13.08, p=0.000) |
| points | 12210 | 11998 | 10306 | 10297 |

**PM10, with future ERA5 weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **28.94** | **45.79** | **58.73** | **65.84** |
| AQFM v2 | 29.42 (+0.49, p=0.628) | 46.33 (+0.54, p=0.381) | 61.24 (+2.51, p=0.012) | 67.20 (+1.35, p=0.323) |
| XGBoost + future ERA5 | 34.32 (+5.38, p=0.000) | 47.18 (+1.39, p=0.225) | 61.90 (+3.17, p=0.008) | 67.18 (+1.34, p=0.602) |
| Persistence | 31.28 (+2.34, p=0.270) | 71.50 (+25.72, p=0.000) | 84.63 (+25.89, p=0.000) | 106.48 (+40.64, p=0.000) |
| points | 12528 | 12340 | 10630 | 10624 |

**PM10, no future weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **28.66** | **45.86** | **72.28** | **79.36** |
| AQFM v2 | 29.20 (+0.54, p=0.018) | 48.30 (+2.44, p=0.000) | 72.83 (+0.55, p=0.306) | 81.23 (+1.87, p=0.051) |
| XGBoost | 34.63 (+5.97, p=0.000) | 49.63 (+3.77, p=0.000) | 73.54 (+1.26, p=0.405) | 83.16 (+3.80, p=0.076) |
| Persistence | 31.28 (+2.62, p=0.266) | 71.50 (+25.64, p=0.000) | 84.63 (+12.35, p=0.000) | 106.48 (+27.13, p=0.000) |
| points | 12528 | 12340 | 10630 | 10624 |

**Ozone, with future ERA5 weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **5.07** | **6.50** | **8.87** | **9.61** |
| AQFM v2 | 5.15 (+0.08, p=0.105) | 6.61 (+0.11, p=0.001) | 8.88 (+0.01, p=0.994) | 9.71 (+0.11, p=0.465) |
| XGBoost + future ERA5 | 6.18 (+1.11, p=0.000) | 7.09 (+0.59, p=0.000) | 9.20 (+0.33, p=0.038) | 10.01 (+0.41, p=0.014) |
| Persistence | 4.98 (-0.09, p=0.235) | 7.88 (+1.39, p=0.000) | 11.61 (+2.74, p=0.000) | 13.81 (+4.20, p=0.000) |
| points | 12197 | 12159 | 10642 | 10640 |

**Ozone, no future weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **5.21** | **7.04** | **9.69** | **10.58** |
| AQFM v2 | 5.62 (+0.41, p=0.000) | 7.24 (+0.20, p=0.000) | 9.72 (+0.03, p=0.379) | 10.63 (+0.05, p=0.006) |
| XGBoost | 6.29 (+1.08, p=0.000) | 7.46 (+0.43, p=0.008) | 10.10 (+0.41, p=0.006) | 11.18 (+0.60, p=0.000) |
| Persistence | 4.98 (-0.23, p=0.085) | 7.88 (+0.85, p=0.001) | 11.61 (+1.92, p=0.000) | 13.81 (+3.23, p=0.000) |
| points | 12197 | 12159 | 10642 | 10640 |

**NOx, with future ERA5 weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **11.94** | **15.86** | **24.92** | **27.69** |
| AQFM v2 | 12.10 (+0.17, p=0.007) | 16.20 (+0.33, p=0.016) | 24.95 (+0.03, p=0.949) | 26.74 (-0.95, p=0.048) |
| XGBoost + future ERA5 | 15.08 (+3.15, p=0.000) | 17.24 (+1.37, p=0.000) | 26.55 (+1.63, p=0.001) | 29.33 (+1.64, p=0.024) |
| Persistence | 12.55 (+0.62, p=0.000) | 25.45 (+9.59, p=0.000) | 36.63 (+11.72, p=0.000) | 44.74 (+17.05, p=0.000) |
| points | 12328 | 12423 | 10862 | 10854 |

**NOx, no future weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **11.95** | **17.25** | **29.94** | **32.77** |
| AQFM v2 | 11.99 (+0.03, p=0.484) | 17.59 (+0.34, p=0.002) | 30.27 (+0.33, p=0.167) | 33.57 (+0.80, p=0.040) |
| XGBoost | 15.54 (+3.58, p=0.000) | 19.26 (+2.00, p=0.000) | 32.37 (+2.43, p=0.000) | 37.05 (+4.28, p=0.000) |
| Persistence | 12.55 (+0.60, p=0.000) | 25.45 (+8.20, p=0.000) | 36.63 (+6.70, p=0.000) | 44.74 (+11.97, p=0.000) |
| points | 12328 | 12423 | 10862 | 10854 |

**NO2, with future ERA5 weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **9.91** | **10.84** | **15.21** | **16.93** |
| AQFM v2 | 9.63 (-0.28, p=0.000) | 10.79 (-0.05, p=0.286) | 14.87 (-0.34, p=0.002) | 16.53 (-0.40, p=0.041) |
| XGBoost + future ERA5 | 9.03 (-0.88, p=0.000) | 9.90 (-0.94, p=0.000) | 13.14 (-2.07, p=0.000) | 15.06 (-1.87, p=0.000) |
| Persistence | 8.52 (-1.40, p=0.000) | 16.60 (+5.76, p=0.000) | 15.86 (+0.65, p=0.166) | 19.62 (+2.69, p=0.000) |
| points | 12577 | 12607 | 10819 | 10813 |

**NO2, no future weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **9.79** | **11.13** | **15.92** | **17.93** |
| AQFM v2 | 9.61 (-0.19, p=0.000) | 11.27 (+0.15, p=0.033) | 15.96 (+0.04, p=0.748) | 18.24 (+0.32, p=0.059) |
| XGBoost | 9.11 (-0.68, p=0.004) | 10.21 (-0.92, p=0.000) | 14.30 (-1.62, p=0.000) | 16.75 (-1.17, p=0.001) |
| Persistence | 8.52 (-1.28, p=0.000) | 16.60 (+5.47, p=0.000) | 15.86 (-0.06, p=0.900) | 19.62 (+1.69, p=0.005) |
| points | 12577 | 12607 | 10819 | 10813 |

### ABL: v2-M seed 0, one change each (Delhi 2025 test, MAE)

| Run | val score | PM2.5 1h | PM2.5 6h | PM2.5 24h | PM2.5 72h | PM10 1h | PM10 6h | PM10 24h | PM10 72h |
|---|---|---|---|---|---|---|---|---|---|
| full v2-M (seed 0) | 1.7268 | 16.11 | 28.64 | 33.44 | 37.88 | 29.09 | 47.36 | 60.89 | 67.82 |
| no spatial attention (kept; spatial ablations dropped) | 1.7290 | 16.10 | 28.72 | 32.71 | 37.72 | 29.21 | 46.59 | 60.03 | 68.15 |
| no BLH | 1.7368 | 16.01 | 29.08 | 33.20 | 37.29 | 29.20 | 47.52 | 60.60 | 67.73 |
| no anchor (last observation) | 1.8129 | 33.05 | 33.75 | 35.15 | 39.42 | 60.93 | 54.78 | 64.28 | 70.02 |
| gap-filling training only | 1.9040 | 32.33 | 34.23 | 36.25 | 38.23 | 56.78 | 55.64 | 68.46 | 72.21 |
| seed spread of full v2-M (std of 3 seeds) | 0.0019 | 0.09 | 0.32 | 0.81 | 0.46 | 0.13 | 0.92 | 1.33 | 1.08 |

### FOLD: second test year (train 2018–22, pick on 2023, test 2024), same points

**PM2.5 2024, with future ERA5 weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **16.66** | **30.89** | **37.25** | **42.11** |
| XGBoost + future ERA5 | 20.16 (+3.49, p=0.000) | 31.69 (+0.80, p=0.096) | 36.99 (-0.26, p=0.752) | 40.48 (-1.63, p=0.322) |
| Persistence | 16.94 (+0.27, p=0.533) | 40.13 (+9.24, p=0.000) | 47.15 (+9.89, p=0.000) | 55.78 (+13.67, p=0.000) |
| points | 12188 | 12036 | 10258 | 10254 |

**PM2.5 2024, no future weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M | **16.72** | **31.71** | **41.91** | **47.95** |
| XGBoost | 20.35 (+3.63, p=0.000) | 33.05 (+1.34, p=0.031) | 40.87 (-1.04, p=0.226) | 45.70 (-2.25, p=0.193) |
| Persistence | 16.94 (+0.22, p=0.682) | 40.13 (+8.42, p=0.000) | 47.15 (+5.24, p=0.000) | 55.78 (+7.83, p=0.000) |
| points | 12188 | 12036 | 10258 | 10254 |

**PM10 2024 (each model on its own points; MAE)**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M, with weather (3-seed ensemble) | 32.74 | 51.67 | 69.29 | 76.85 |
| AQFM v2-M, no future weather | 32.76 | 52.71 | 80.87 | 87.67 |
| XGBoost + future ERA5 | 39.37 | 54.02 | 70.37 | 75.72 |
| XGBoost | 39.73 | 55.41 | 79.45 | 88.44 |
| Persistence | 36.34 | 80.12 | 94.16 | 106.66 |

### NOISE: v2-M with future weather, the future weather perturbed (Delhi 2025 test, 3-seed ensemble, MAE)

| Future weather | PM2.5 1h | PM2.5 6h | PM2.5 24h | PM2.5 72h | PM10 1h | PM10 6h | PM10 24h | PM10 72h |
|---|---|---|---|---|---|---|---|---|
| ERA5 as is | 15.99 | 27.95 | 32.26 | 36.95 | 28.94 | 45.79 | 58.74 | 65.83 |
| noise sd 0.25 × √(lead days) | 16.00 | 28.11 | 32.76 | 38.31 | 28.97 | 45.96 | 60.28 | 70.14 |
| noise sd 0.5 × √(lead days) | 16.04 | 28.77 | 35.11 | 45.03 | 29.11 | 47.23 | 66.18 | 85.10 |
| noise sd 1.0 × √(lead days) | 16.55 | 33.05 | 45.46 | 60.01 | 30.31 | 55.09 | 86.48 | 109.22 |
| none (v2-M trained without future weather) | 15.92 | 27.67 | 38.09 | 41.54 | 28.66 | 45.87 | 72.27 | 79.33 |

### Mumbai: fine-tuning with nothing frozen vs backbone frozen

**Validation (2024-01..06), best score per seed (nll + 0.5 × crps; lower is better)**

| Months | backbone frozen s0 / s1 / s2 (mean) | nothing frozen s0 / s1 / s2 (mean) |
|---|---|---|
| 6 | 1.9117 / 1.9007 / 1.8993 (1.9039) | 1.9034 / 1.9045 / 1.8869 (1.8983) |
| 12 | 1.8782 / 1.8795 / 1.8808 (1.8795) | 1.8568 / 1.8668 / 1.8585 (1.8607) |

**Mumbai PM2.5, 6 months of training data (test 2024-07..12)**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM fine-tuned, nothing frozen | **6.02** | **9.52** | **11.91** | **14.58** |
| AQFM fine-tuned, backbone frozen | 6.17 (+0.16, p=0.067) | 10.15 (+0.63, p=0.000) | 12.21 (+0.30, p=0.042) | 14.94 (+0.36, p=0.095) |
| AQFM from scratch | 5.76 (-0.26, p=0.475) | 9.08 (-0.44, p=0.016) | 11.49 (-0.43, p=0.223) | 14.09 (-0.49, p=0.111) |
| XGBoost (same months) | 7.06 (+1.05, p=0.015) | 9.61 (+0.08, p=0.949) | 12.62 (+0.70, p=0.236) | 16.62 (+2.04, p=0.148) |
| Persistence | 4.40 (-1.62, p=0.000) | 9.73 (+0.21, p=0.884) | 11.55 (-0.36, p=0.252) | 16.32 (+1.74, p=0.122) |
| points | 4093 | 4046 | 4097 | 4094 |

**Mumbai PM10, 6 months of training data (test 2024-07..12)**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM fine-tuned, nothing frozen | **9.92** | **20.17** | **24.26** | **29.55** |
| AQFM fine-tuned, backbone frozen | 9.89 (-0.04, p=0.650) | 20.35 (+0.19, p=0.554) | 24.24 (-0.02, p=0.906) | 29.81 (+0.26, p=0.412) |
| AQFM from scratch | 10.97 (+1.04, p=0.020) | 20.63 (+0.46, p=0.131) | 25.48 (+1.22, p=0.022) | 30.84 (+1.30, p=0.023) |
| XGBoost (same months) | 16.15 (+6.22, p=0.000) | 22.57 (+2.40, p=0.000) | 28.58 (+4.32, p=0.000) | 35.83 (+6.28, p=0.000) |
| Persistence | 10.31 (+0.39, p=0.184) | 24.71 (+4.54, p=0.013) | 26.57 (+2.31, p=0.002) | 35.23 (+5.68, p=0.000) |
| points | 4067 | 4034 | 4079 | 4074 |

**Mumbai NOx, 6 months of training data (test 2024-07..12)**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM fine-tuned, nothing frozen | **4.48** | **7.06** | **8.57** | **10.16** |
| AQFM fine-tuned, backbone frozen | 4.48 (+0.01, p=0.782) | 6.81 (-0.25, p=0.070) | 8.60 (+0.03, p=0.545) | 10.22 (+0.07, p=0.196) |
| AQFM from scratch | 4.58 (+0.10, p=0.140) | 6.83 (-0.23, p=0.143) | 9.20 (+0.63, p=0.006) | 10.60 (+0.44, p=0.046) |
| XGBoost (same months) | 7.06 (+2.58, p=0.000) | 8.63 (+1.57, p=0.000) | 10.07 (+1.50, p=0.000) | 12.89 (+2.73, p=0.000) |
| Persistence | 4.52 (+0.05, p=0.217) | 8.92 (+1.86, p=0.000) | 9.26 (+0.69, p=0.000) | 11.88 (+1.72, p=0.001) |
| points | 4201 | 4148 | 4200 | 4186 |

**Mumbai Ozone, 6 months of training data (test 2024-07..12)**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM fine-tuned, nothing frozen | **3.57** | **5.94** | **5.90** | **6.62** |
| AQFM fine-tuned, backbone frozen | 3.55 (-0.02, p=0.472) | 5.99 (+0.04, p=0.187) | 5.92 (+0.02, p=0.219) | 6.63 (+0.01, p=0.487) |
| AQFM from scratch | 3.71 (+0.14, p=0.000) | 5.90 (-0.05, p=0.463) | 5.99 (+0.09, p=0.104) | 6.72 (+0.10, p=0.030) |
| XGBoost (same months) | 4.89 (+1.32, p=0.000) | 6.70 (+0.76, p=0.000) | 6.61 (+0.71, p=0.000) | 7.56 (+0.94, p=0.000) |
| Persistence | 3.55 (-0.02, p=0.913) | 6.75 (+0.81, p=0.014) | 6.98 (+1.08, p=0.000) | 8.53 (+1.91, p=0.000) |
| points | 4217 | 4174 | 4217 | 4213 |

**Mumbai PM2.5, 12 months of training data (test 2024-07..12)**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM fine-tuned, nothing frozen | **5.85** | **9.24** | **11.77** | **14.42** |
| AQFM fine-tuned, backbone frozen | 6.46 (+0.60, p=0.000) | 10.25 (+1.01, p=0.000) | 12.28 (+0.52, p=0.000) | 14.88 (+0.46, p=0.008) |
| AQFM from scratch | 5.39 (-0.46, p=0.006) | 8.79 (-0.46, p=0.004) | 11.22 (-0.55, p=0.028) | 14.08 (-0.34, p=0.292) |
| XGBoost (same months) | 6.88 (+1.02, p=0.026) | 9.04 (-0.20, p=0.449) | 12.24 (+0.47, p=0.436) | 16.62 (+2.20, p=0.107) |
| Persistence | 4.40 (-1.45, p=0.000) | 9.73 (+0.49, p=0.602) | 11.55 (-0.22, p=0.412) | 16.32 (+1.90, p=0.081) |
| points | 4093 | 4046 | 4097 | 4094 |

**Mumbai PM10, 12 months of training data (test 2024-07..12)**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM fine-tuned, nothing frozen | **9.92** | **19.77** | **24.21** | **29.59** |
| AQFM fine-tuned, backbone frozen | 9.99 (+0.08, p=0.638) | 20.32 (+0.54, p=0.038) | 24.14 (-0.07, p=0.728) | 29.57 (-0.02, p=0.883) |
| AQFM from scratch | 10.33 (+0.41, p=0.020) | 20.10 (+0.33, p=0.184) | 24.79 (+0.58, p=0.181) | 30.41 (+0.82, p=0.221) |
| XGBoost (same months) | 15.94 (+6.03, p=0.000) | 22.67 (+2.90, p=0.000) | 27.26 (+3.05, p=0.000) | 34.36 (+4.77, p=0.001) |
| Persistence | 10.31 (+0.40, p=0.180) | 24.71 (+4.93, p=0.006) | 26.57 (+2.36, p=0.002) | 35.23 (+5.64, p=0.000) |
| points | 4067 | 4034 | 4079 | 4074 |

**Mumbai NOx, 12 months of training data (test 2024-07..12)**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM fine-tuned, nothing frozen | **4.56** | **7.09** | **8.55** | **10.14** |
| AQFM fine-tuned, backbone frozen | 4.47 (-0.08, p=0.057) | 6.90 (-0.18, p=0.080) | 8.61 (+0.07, p=0.185) | 10.23 (+0.09, p=0.252) |
| AQFM from scratch | 4.64 (+0.08, p=0.131) | 7.22 (+0.13, p=0.236) | 8.77 (+0.23, p=0.040) | 10.41 (+0.28, p=0.078) |
| XGBoost (same months) | 6.81 (+2.25, p=0.000) | 8.25 (+1.16, p=0.000) | 9.69 (+1.14, p=0.000) | 12.17 (+2.03, p=0.000) |
| Persistence | 4.52 (-0.03, p=0.446) | 8.92 (+1.83, p=0.000) | 9.26 (+0.71, p=0.000) | 11.88 (+1.74, p=0.000) |
| points | 4201 | 4148 | 4200 | 4186 |

**Mumbai Ozone, 12 months of training data (test 2024-07..12)**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM fine-tuned, nothing frozen | **3.54** | **5.92** | **5.94** | **6.70** |
| AQFM fine-tuned, backbone frozen | 3.51 (-0.02, p=0.612) | 6.01 (+0.09, p=0.074) | 5.96 (+0.02, p=0.534) | 6.63 (-0.07, p=0.058) |
| AQFM from scratch | 3.58 (+0.05, p=0.039) | 5.90 (-0.02, p=0.747) | 6.05 (+0.12, p=0.057) | 6.95 (+0.25, p=0.004) |
| XGBoost (same months) | 4.75 (+1.22, p=0.000) | 6.50 (+0.59, p=0.001) | 6.38 (+0.44, p=0.000) | 7.43 (+0.73, p=0.000) |
| Persistence | 3.55 (+0.02, p=0.696) | 6.75 (+0.83, p=0.008) | 6.98 (+1.04, p=0.000) | 8.53 (+1.83, p=0.000) |
| points | 4217 | 4174 | 4217 | 4213 |

**What this says:**
- **Delhi PM2.5/PM10, against XGBoost run on the same points:**
  - AQFM v2-M is clearly better at 1 h (p < 0.001, with and without future weather);
  - at 6 h it is better without future weather (p < 0.001);
  - at 24 h and 72 h most differences are not significant. The exception is PM10 at 24 h with future weather,
    where AQFM is better (p = 0.008).
  - Persistence ties AQFM at 1 h (PM2.5 p ≥ 0.65, PM10 p ≥ 0.26).
- **The second test year (2024) gives the same picture for PM2.5:**
  - AQFM is better than XGBoost at 1 h (p < 0.001), and at 6 h without weather (p = 0.031);
  - at 24–72 h XGBoost is lower, but not significantly (p ≥ 0.19);
  - persistence ties at 1 h.
  - Every model's PM2.5 error is higher in 2024 than in 2025. Both the test year and the training data (one year less)
    differ, so the two cannot be separated.
- **Ozone and NOx: AQFM v2-M beats XGBoost at every lead, with and without future weather** (p ≤ 0.038). For NOx it
  also beats persistence at every lead, including 1 h.
- **NO2 is a weak point in Delhi too:**
  - XGBoost beats AQFM at every lead (p ≤ 0.004), and persistence wins at 1 h;
  - AQFM beats persistence only at 6 h and 72 h.
  - Untested guess at the cause: the model forecasts NOx and the NO share and rebuilds NO2 from them (the Mumbai
    NO2 note in step 10), so NO2 carries the errors of both.
  - Report NO2 as a limitation, or fix the representation (not done).
- **Ablations (one seed; compare with the seed spread in the last row):**
  - The anchor (last observation) and forecast training matter most. Without them the 1 h error doubles
    (PM2.5 33.05 and 32.33 vs 16.11), and validation is far worse.
  - Without BLH, validation is worse by 0.0100 (about 5 × the seed spread). The test differences are mixed in sign
    and within 1.5 × the seed spread.
  - No spatial attention: validation +0.0022 (about one seed spread); the test is within the seed spread.
- **Future weather that is not perfect (NOISE):**
  - The 1 h forecast changes little (PM2.5 at most +0.56, at sd 1.0).
  - With noise sd 0.25, the with-weather model still beats the no-weather model at 24 h and 72 h.
  - At sd 0.5 it is still better at 24 h (PM2.5 35.11 vs 38.09) but worse at 72 h (45.03 vs 41.54).
  - At sd 1.0 it is worse from 6 h on.
  - So the gain from future weather at 72 h needs a good weather forecast. Which sd matches a real weather forecast
    is unknown here: that needs archived forecasts, which this machine doesn't have. The noise is in the model's
    scaled units: one unit = one standard deviation of that variable's hourly values over the training years
    (after a log for some variables).
- **Mumbai, nothing frozen vs backbone frozen:**
  - **Validation picks nothing frozen** at 12 months (every seed; mean 1.8607 vs 1.8795) and at 6 months on average
    (1.8983 vs 1.9039; seed 1 is the other way). Both fine-tunes score better on validation than from scratch
    (1.8822 at 12 months, 1.9334 at 6 months).
  - With 1–3 months, nothing frozen was not run with this recipe. Stage A, at 3 months and lr 1e-4, had it
    0.002 behind backbone frozen.
  - **Test PM2.5: nothing frozen beats backbone frozen** (12 months: every lead, p ≤ 0.008; 6 months: 6 h and 24 h,
    p ≤ 0.042). It closes about 40–50 % of the gap to from scratch at 24 h.
  - From scratch is still better on PM2.5 with 12 months at 1, 6 and 24 h (p ≤ 0.028), and with 6 months at 6 h
    (p = 0.016).
  - So for PM2.5, validation (all pollutants, NLL and CRPS, 2024 H1) and test (MAE, 2024 H2) disagree on fine-tuned
    vs from scratch. Report it as it is.
  - **PM10, NOx and ozone:** nothing frozen ≈ backbone frozen. The only significant difference is PM10 at 12 months,
    6 h, where nothing frozen is better (p = 0.038). From scratch is never significantly better than nothing frozen,
    and is worse at several leads. XGBoost is worse in every cell (p ≤ 0.001).
  - PM2.5 against persistence: persistence still wins at 1 h; tie at 6–72 h.

**New trained models to register (manifest, Drive):** ABL `runs/v2m256_abl_{nospatial,noblh,nolast,gapfillonly}_s0`;
FOLD `runs/v2m256_fold24{,_nofm}_s{0,1,2}_do03`; Mumbai `runs/mumbai_ft_v2_none_m{6,12}_s{0,1,2}`.

## Step 12: NO2 forecast directly — done 2026-11-07

**Why (user, 2026-11-07: "about no2 try to fix it").** In stage SAVE, XGBoost beat AQFM v2-M on Delhi NO2 at every
lead, and persistence won at 1 h. Cause, from the data and the saved forecasts:
- AQFM forecast NOx and the NO share of NOx (`NOX_FRAC`) and rebuilt NO2 = NOx × (1 − share). The share is only
  defined where NO + NO2 = NOx (in ppb) within 10 %. In 2018–2023 that holds for 57.6 % of the NO2 readings, and at
  10 of the 37 stations for at most 15 % of them.
- At three of those stations the NO2/NOx ratio (median, ppb) was about 0.3 in 2018–2024 and 0.69–0.76 in 2025, with
  NO2 above NOx in 22–29 % of 2025 readings. Rebuilt NO2 can never exceed NOx, so v2-M under-forecast NO2 there (mean
  error at 24 h −31 and −22 µg/m³ at the two worst stations).
- Same forecasts as XGBoost + future ERA5, NO2 MAE 1/6/24/72 h: at those 10 stations AQFM 14.67/13.54/19.21/21.07 vs
  XGBoost 9.24/10.08/14.75/17.51; at the other 27 AQFM 8.19/9.88/13.64/15.32 vs XGBoost 8.95/9.84/12.51/14.11.
  (From `results/forecasts/delhi_v2m256_ens3.npz` and `delhi_xgboost_fm.npz`; the diagnosis looked at the 2025 test,
  the choice below is made on validation.)

**Fix:** `aqfm/train.py --no2-direct` puts NO2 (log1p µg/m³) in place of `NOX_FRAC` among the targets, the same way
`--pm25-direct` treats PM2.5. NOx stays a direct target; NO is no longer forecast (never scored). `aqfm/evaluate.py`
reads NO2 from its own slot when the model has one. New test `test_no2_direct_is_its_own_output` (fails without
either change); 92 tests pass.

**Trial (one model, uncommitted code, so not registered; outside the repo):** v2-M with future weather, seed 0, otherwise as
`runs/v2m256_s0_do03` (`/home/phalak/aqfm/explore/no2_try.sh`; best at step 10,000 of 12,000; 29 min). Both models
scored on validation 2024 (`/home/phalak/aqfm/explore/eval_val24.py`: `aqfm.evaluate` with origins from 2024-01-01 and
scoring stopped at 2025-01-01; it only relaxes the check that the scored period is the checkpoint's own test period;
normalisation period unchanged) and on the 2025 test. MAE, old (NO2 rebuilt) → new (NO2 direct):

| Period | Pollutant | 1 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|---|
| validation 2024 | NO2 | 10.50 → 7.08 | 12.37 → 9.90 | 16.00 → 12.90 | 17.80 → 14.94 |
| validation 2024 | NOx | 12.55 → 12.48 | 17.72 → 17.57 | 28.52 → 28.56 | 30.71 → 31.27 |
| validation 2024 | Ozone | 5.28 → 5.28 | 6.90 → 6.86 | 8.71 → 8.78 | 9.56 → 9.65 |
| validation 2024 | PM2.5 | 16.78 → 16.50 | 31.40 → 31.33 | 38.94 → 38.40 | 42.12 → 42.17 |
| validation 2024 | PM10 | 32.99 → 32.63 | 53.36 → 53.27 | 71.22 → 69.96 | 76.29 → 75.43 |
| test 2025 | NO2 | 10.09 → 7.72 | 10.98 → 9.38 | 15.38 → 12.91 | 17.13 → 14.83 |
| test 2025 | NOx | 12.01 → 11.98 | 16.05 → 15.98 | 25.72 → 25.64 | 28.35 → 28.66 |
| test 2025 | Ozone | 5.10 → 5.16 | 6.56 → 6.55 | 8.86 → 8.94 | 9.60 → 9.69 |
| test 2025 | PM2.5 | 16.11 → 15.91 | 28.64 → 28.64 | 33.44 → 32.50 | 37.88 → 36.97 |
| test 2025 | PM10 | 29.09 → 28.93 | 47.36 → 47.05 | 60.89 → 58.89 | 67.82 → 65.72 |

**Decision (on validation):** NO2 is better at every lead (by 2.5–3.4 µg/m³); NOx, ozone, PM2.5 and PM10 change by
little (largest worsening: NOx 72 h +0.56). So the 6 v2-M paper models are retrained with `--no2-direct`
(`scripts/pipeline_no2_direct.sh`, log `runs/no2_direct.log`, ~4 h; the user chose "NO2 runs after trial" on
2026-11-07), then the same-point comparisons are redone (`results/paper/delhi_*_no2d.json`).

**Results** (`scripts/pipeline_no2_direct.sh`, 11:44–15:25; tables below generated by
`/home/phalak/aqfm/pending_patches/step12_tables.py` from the files named).

Checks: the log ends with `pipeline finished`; all 6 trainings end with `done`; every checkpoint and result records
commit d34609e, clean; ensembles over 362 origins, as the old v2-M ensembles. Two runs were best at their last step
(`v2m256_no2d_s1`, `v2m256_no2d_nofm_s2`), as the old `v2m256_s1` was: the 12,000-step budget binds for them; not
tested further. Seed 0 with weather stopped by patience at step 8,000 (best 5,000). The trial, with the same seed and
recipe, beat its step-5,000 score at step 8,000 by 0.0001 and ran to 12,000: GPU runs are not bit-identical. This seed
is the weakest of the six (PM2.5 1 h MAE 17.17 against 16.06 and 15.86), so the with-weather seed spread is wider than
the old one (PM2.5 1 h ± 0.71 against ± 0.09; `results/TABLES.md`). The validation scores are not comparable with the
old v2-M runs': NO2 replaces the NO share among the scored outputs.

**Training** (each run: best selection score on 2024 validation, its step, last validation; `runs/<run>/log.jsonl`)

| Run | best score | best step | last validation |
|---|---|---|---|
| v2m256_no2d_s0_do03 | 1.7495 | 5000 | 8000 |
| v2m256_no2d_s1_do03 | 1.7350 | 12000 | 12000 |
| v2m256_no2d_s2_do03 | 1.7358 | 11000 | 12000 |
| v2m256_no2d_nofm_s0_do03 | 1.8035 | 10000 | 12000 |
| v2m256_no2d_nofm_s1_do03 | 1.8016 | 11000 | 12000 |
| v2m256_no2d_nofm_s2_do03 | 1.8007 | 12000 | 12000 |

**NO2, same points, Delhi 2025 test** (MAE; in brackets: difference to the first model, Diebold–Mariano p; `results/paper/delhi_NO2_*_no2d.json`)

**NO2, with future ERA5 weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M, NO2 direct | **7.73** | **9.31** | **12.87** | **14.73** |
| AQFM v2-M | 9.91 (+2.18, p=0.000) | 10.84 (+1.53, p=0.000) | 15.21 (+2.33, p=0.000) | 16.93 (+2.20, p=0.000) |
| XGBoost + future ERA5 | 9.03 (+1.30, p=0.000) | 9.90 (+0.59, p=0.000) | 13.14 (+0.26, p=0.144) | 15.06 (+0.33, p=0.129) |
| Persistence | 8.52 (+0.78, p=0.000) | 16.60 (+7.29, p=0.000) | 15.86 (+2.98, p=0.000) | 19.62 (+4.89, p=0.000) |
| points | 12577 | 12607 | 10819 | 10813 |

**NO2, no future weather**

| Model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| AQFM v2-M, NO2 direct | **7.68** | **9.57** | **14.23** | **16.52** |
| AQFM v2-M | 9.79 (+2.11, p=0.000) | 11.13 (+1.56, p=0.000) | 15.92 (+1.68, p=0.000) | 17.93 (+1.41, p=0.000) |
| XGBoost | 9.11 (+1.43, p=0.000) | 10.21 (+0.63, p=0.000) | 14.30 (+0.07, p=0.725) | 16.75 (+0.23, p=0.315) |
| Persistence | 8.52 (+0.83, p=0.000) | 16.60 (+7.03, p=0.000) | 15.86 (+1.62, p=0.000) | 19.62 (+3.10, p=0.000) |
| points | 12577 | 12607 | 10819 | 10813 |

**Other pollutants: NO2-direct ensemble against the old v2-M ensemble, same points** (old − new MAE, µg/m³; positive = NO2 direct better; Diebold–Mariano p)

| Pollutant | weather | 1h | 6h | 24h | 72h |
|---|---|---|---|---|---|
| PM2.5 | with | +0.19 (p=0.142) | +0.16 (p=0.496) | +0.08 (p=0.712) | +0.61 (p=0.089) |
| PM2.5 | without | +0.07 (p=0.224) | -0.14 (p=0.552) | +0.41 (p=0.385) | -0.61 (p=0.474) |
| PM10 | with | -0.06 (p=0.579) | +0.05 (p=0.765) | -0.30 (p=0.306) | +0.33 (p=0.533) |
| PM10 | without | -0.07 (p=0.217) | -0.51 (p=0.064) | +0.01 (p=0.960) | -1.27 (p=0.221) |
| Ozone | with | -0.02 (p=0.292) | -0.01 (p=0.515) | -0.01 (p=0.625) | -0.03 (p=0.372) |
| Ozone | without | +0.10 (p=0.000) | -0.03 (p=0.147) | -0.03 (p=0.117) | -0.04 (p=0.010) |
| NOx | with | +0.02 (p=0.569) | -0.01 (p=0.901) | -0.22 (p=0.320) | +0.15 (p=0.602) |
| NOx | without | -0.06 (p=0.092) | -0.12 (p=0.213) | -0.05 (p=0.791) | -0.20 (p=0.554) |

**NO2 by station group, with future weather** (MAE on the points both models forecast; from `results/forecasts/delhi_v2m256_no2d_ens3.npz`, `delhi_v2m256_ens3.npz`, `delhi_xgboost_fm.npz`)

| Stations | model | 1h | 6h | 24h | 72h |
|---|---|---|---|---|---|
| 10 without a usable NO share | AQFM v2-M, NO2 direct | 7.97 | 9.33 | 14.57 | 17.21 |
| 10 without a usable NO share | AQFM v2-M (old) | 14.67 | 13.54 | 19.21 | 21.07 |
| 10 without a usable NO share | XGBoost + future ERA5 | 9.24 | 10.08 | 14.75 | 17.51 |
| the other 27 | AQFM v2-M, NO2 direct | 7.65 | 9.30 | 12.21 | 13.76 |
| the other 27 | AQFM v2-M (old) | 8.19 | 9.88 | 13.64 | 15.32 |
| the other 27 | XGBoost + future ERA5 | 8.95 | 9.84 | 12.51 | 14.11 |

**Findings.**
- NO2 is better than the old v2-M at every lead, with and without future weather (by 1.4–2.3 µg/m³, all p < 0.001).
- NO2 now beats XGBoost at every lead: significantly at 1 h and 6 h (p < 0.001); at 24 h and 72 h by 0.07–0.33,
  not significant (p = 0.13–0.73). It beats persistence at 1 h too (7.73 against 8.52 with weather), which the old
  model did not.
- At the 10 stations without a usable NO share, the error falls most (24 h: 19.21 → 14.57) and is now level with
  XGBoost (14.75); at the other 27 it falls too (24 h: 13.64 → 12.21; XGBoost 12.51).
- PM2.5, PM10, ozone and NOx: no difference with p < 0.05 except ozone without future weather (1 h 0.10 better,
  72 h 0.04 worse).

**Decision (2026-11-07): the NO2-direct v2-M models are the paper's main v2-M, for all five pollutants** (rows "AQFM
v2-M, NO2 predicted directly" in `results/TABLES.md`; same-point tests in `results/paper/delhi_*_no2d.json`).
- The fix was adopted on 2024 validation before these 6 models were scored on the test (decision above). Keeping the
  old model for some pollutants and the new one for NO2 after seeing the 2025 test would be choosing on the test.
- One model then gives every reported number. On the 2025 test no pollutant is significantly worse, except ozone
  without future weather at 72 h (by 0.04 µg/m³).
- NO2 no longer depends on NO readings, so the design also suits cities that do not measure NO, such as Mumbai (not
  tested).
- The old v2-M models stay registered. The v2-M ablations, the second test year, the noisy-weather test and the Mumbai
  zero-shot test (step 11) were run with them, and they are the fair partner for the size comparison with v1 and v2
  (same outputs). Those results compare the recipe with itself, so they stand; the paper says they were run before NO2
  was forecast directly.
- Single models: seed 0 with weather stopped early (above), so the spread over seeds is wider than the old model's.
  The 3-seed ensemble stays the headline, as before.

Registered: `scripts/manifest.json` (6 models `aqfm_v2m_no2d{,_nofm}_s{0,1,2}`, 10 results; 92 models, 136 results;
`/home/phalak/aqfm/pending_patches/add_step12_manifest.py`, which checks each command against the recorded arguments
and fails on a changed seed or target list). The ensemble commands are registered without `--save-forecasts`, which
only writes `results/forecasts/*.npz`, so `--reproduce` does not overwrite the saved forecasts. `scripts/make_tables.py`:
4 rows and the "NO2 predicted directly" label; `results/TABLES.md` regenerated. Drive folder: the 6 models, README and
`models/SHA256SUMS.txt` (68 models, all OK, same sha256 as the manifest). `check_submission.py`: all OK except
uncommitted files and GitHub main (aqfm-v2 not merged). Tests: 92 pass (`tests`, `deep_baselines/tests`,
`deep_baselines/estgcn/tests`).

## Status against §8 "Done when"

- [x] Branch `aqfm-v2` has the patch merged, all old tests pass, v1 equivalence shown (step 4).
- [x] B4, B1 (code; real-data checks pending), B5 done with tests; B3 (facts) and B6 (both tests) done after your decisions.
- [x] `data/delhi/` and a second city from arguments alone: built and checked (step 2); `data/delhi/` uses v1's BP rule
  (decision after check 2).
- [x] The v2 tests pass (66 tests in `tests/`: 46 v1 + 15 in `test_aqfm_v2.py` + 5 in `test_pipeline_v2.py`).
- [x] Parameter count of `AQFMConfig.city()`: 1,172,271.
- [x] The short Delhi run and the fine-tune runs work (step 8).
- [x] Full Delhi training: done 2026-11-04 16:41 (`results/v2_delhi_summary.md`; step 9 results above).

**Changed / new files on `aqfm-v2` (committed 2026-11-04 in 8be6172 and a356a6c, results and manifest 2026-11-05;
not pushed):** `aqfm/{config,data,evaluate,layers,model,train}.py`;
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
