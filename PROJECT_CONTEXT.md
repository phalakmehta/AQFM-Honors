# AQFM: Project Context for a New Session

> Read this first. It tells you what the project is, what state each part is in, which bugs were found and fixed on 2026-10-08, and what remains to be done (**§6 is the roadmap**).
> Older docs (`_archive_old/project_handoff.md`, `_archive_old/preprocessing_result_aqfm`, the presentation PDF) contain **numbers and claims that are now stale**. Where they disagree with this file, this file wins.

## Status at a glance (last updated 2026-10-08, 19:35 IST)

| Area | State |
|---|---|
| Data pipeline code | ✅ fixed (3 critical data bugs, §3) |
| ERA5 download 2017-12→2020-12 | ✅ complete, 37/37 months |
| `data/preprocessed/` arrays on disk | ✅ **rebuilt and verified** — (37, 18, 280512) + `era5_met.npy`, all gates passed |
| Baselines | ✅ **re-run on the rebuilt data** — see §4. New bar: **XGBoost PM2.5 @24 h = 38.63** |
| Model `aqfm/` | ✅ built, 31 tests pass, mutation-checked, **runs end-to-end on the real rebuilt data** — but never *trained* |
| Training / evaluation runs | ⬜ not started; needs a GPU (this machine is CPU-only) |
| Paper / comparisons | ⬜ not started (§6 F, G; §9, §10) |

**Are we on track?** Yes for everything that does not need a GPU: the data is rebuilt and all verification gates pass, and the baselines are honest. **The only remaining blocker is GPU access** — AQFM has still never been trained, so whether it works is unknown. Next milestone: §6 Stage C.

---

## 0. Ground rules from the user (see also `CLAUDE.md`)

- **Never commit or push to git.** The user does all git operations; Claude must not appear as a contributor.
- **Do not edit `files_AqFM/` without explicit permission.** It is the original reference implementation and is kept frozen. **The model being developed is the new package `aqfm/`** (§7), built on 2026-10-08 with the user's go-ahead.
- This is a research project aimed at publication. Correctness and honesty over speed: every leakage path needs a test, and every claim needs a measurement.
- Verify everything by running it. Don't fabricate numbers. Ask when something is ambiguous.
- Machine: Windows 11, Python 3.12.5, 16.5 GB RAM, **torch 2.10 CPU-only (no CUDA GPU)**, numpy 2.0.2, pandas 3.0.5, xgboost 3.0.5, granite-tsfm 0.3.9, xarray, pyarrow, pytest 8.3, cdsapi (with `~/.cdsapirc` configured). The full model (23.79M params) takes **~15 s per sample (forward + backward) on this CPU** (measured), so real training needs a GPU (Colab/Kaggle/university). CPU is fine for tests and `--small` runs.

---

## 1. What the project is

Honours project by Phalak Mehta: **"Air Quality Forecasting using Foundation Models: a spatio-temporal deep-learning approach for multi-pollutant forecasting in Delhi."**

- **Data:** CPCB (Central Pollution Control Board) ground stations in Delhi NCR, 15-minute cadence, 2018–2025, **37 stations** (38 are listed in `data/stations.csv`; `alipur` has no CSVs). Plus ERA5 reanalysis weather (incl. boundary-layer height, BLH) interpolated to the station locations.
- **Model (AQFM):** channel-independent patch tokens (one token = 1 station × 1 channel × 16 steps, i.e. 4 h). Masked RevIN normalisation; an axial encoder (time → channel → space attention) with a wind-aware spatial bias; a Perceiver-style pooler giving 16 latents per station; a **query-based decoder** (query = target channel + station + lead patch + time); a mixture-of-3-Gaussians head in normalised log1p space; NLL + CRPS loss (order-statistic estimator). Physical reparameterisation: PM2.5 = PM10 × σ(f_pm) and NO = NOx × σ(f_nox).
- **Training plan:** Delhi-only, two phases. (1) Mixed masking (tail 50% / block 35% / channel 15%). (2) Anneal with tail-only masking at lower LR. National pre-training is out of scope for v1 (`docs/aqfm_implementation_plan.md`). The plan also says not to call it a "foundation model" in write-ups until cross-region transfer is shown.
- **Goal:** beat the XGBoost baseline on PM2.5 at the 24 h horizon on the held-out 2025 test year, and do well in the **Severe AQI band (PM2.5 > 250)**. Also report episode recall/precision, calibration (PIT, interval coverage), and CRPS by lead time.

Key design references: `files_AqFM/aqfm_design.tex` (full spec), `docs/aqfm_implementation_plan.md` (Delhi-only v1 plan), `presentation/aqfm honors final presentation.pdf` (what was presented).

---

## 2. Repository map

| Path | What it is | Status |
|---|---|---|
| `data/yearly_data/<year>/<station>.csv` | Raw CPCB CSVs (gitignored). Columns: `Station, Timestamp, PM2.5 (µg/m³), PM10, NO (µg/m³), NO2 (µg/m³), NOx (ppb), NH3, SO2, CO (mg/m³), Ozone, Benzene, …, AT, RH, WS, WD, RF, TOT-RF, SR, BP (mmHg), VWS`. Timestamps are **local time (IST)**. | raw input |
| `data/stations.csv` | station_id, lat, lon (38 rows) | ok |
| `data/era5_data_raw/` | 100 monthly ERA5 NetCDFs, 2017-01..2025-12 (35 MB). Nothing else lives here any more | complete |
| `data_pipeline/fetch_era5.py` | CDS API download, one NetCDF per month → `data/era5_data_raw/` | ok. Its `__main__` loop only covers 2021–2025; see §3 D1 |
| `data_pipeline/fix_era5_zips.py` | Unpacks months the CDS returned as zip-in-.nc | ok |
| `data_pipeline/interpolate_era5_to_stations.py` | Bilinear to stations, hourly → 15 min → `data/era5_interpolated.parquet` | **fixed** (§3 D1) |
| `data_pipeline/generate_stations_csv.py` | Geocodes station names (Nominatim) → `data/stations.csv` | ok (one-off; coordinates are geocoded, not official) |
| `data_pipeline/preprocess_aqfm.py` | CPCB + ERA5 → `data/preprocessed/{values,observed}.npy (37,18,280512)` (17 CPCB channels + **BLH** as channel 17), **`era5_met.npy (37,8,280512)`** (pure ERA5, IST: AT, RH, U, V, SR, BP, TOT_RF, BLH; NaN = missing), `timestamps_h.npy`, `coords.npy`, `static.npy (37,4)`, `preprocessing_report.json` | **fixed** (§3 D2, D3), BLH + era5_met added. Re-run and verified 2026-10-08 (§6 Stage A) |
| `aqfm/` | **The model (new).** See §7 | built and tested, not yet trained |
| `tests/test_aqfm.py` | 31 tests for `aqfm/` (`python -m pytest tests -q`) | all pass |
| `baselines/baseline_utils.py` | Shared split (test = 2025), horizons {1h:4, 6h:24, 24h:96, 72h:288 steps}, MAE/RMSE, AQI bands, episode detection | ok |
| `baselines/run_baselines_classical.py` | Persistence, seasonal-naive, climatology | ok |
| `baselines/run_baselines_xgboost.py` | XGBoost with lagged features, one model per (target, horizon) | **fixed** (§3 B1) |
| `baselines/run_baselines_ttm.py` | IBM Granite TTM r2, zero-shot, hourly | ok (see caveat in §3 B3) |
| `baselines/results/*.json` | Baseline metrics | regenerated (see §4) |
| `files_AqFM/` | Original reference model + design `.tex/.pdf` + README | **frozen**; superseded by `aqfm/`; its bugs are listed in §5 (all fixed in `aqfm/`) |
| `_archive_old/` | **Superseded material, moved here 2026-10-11.** `project_handoff.md` (replaced by this file), `preprocessing_result_aqfm` (describes the old 17-channel output), `Zeroshot forecast/` (the leaky TTM experiment behind the slides' "32.6 MAE"), `era5_data_2021_2025.zip`, `baseline_results_OLD_invalid/` (pre-fix baseline JSONs; only the XGBoost one differs from the current results), `preprocessing_plan_aqfm` (old plain-language plan, describes 38 stations × 19 channels), and `cpcb_raw_intermediates/` (the `preprocess2017..2020/` folders + zip that were misfiled inside `data/era5_data_raw/`; already processed into `data/yearly_data/`). Gitignored. Safe to delete to free ~371 MB | archive |
| `docs/` | `PROJECT_DOCUMENTATION.md` (full plain-language write-up, basis for the slides), `DATA_REPORT.md`, `BOTTLENECKS_AND_IMPROVEMENTS.md`, `aqfm_implementation_plan.md`, `baselines/` (baseline plan, report, testing guide) | docs |
| `docs/figures/` | EDA figures | ok |
| `benchmark_gpu.py` | Measures s/step and peak VRAM for batch sizes on the current device (run from repo root) | ok |
| `presentation/` | `build_pptx_js.js` (generates the deck; `npm install` here for `pptxgenjs`), `aqfm honors final presentation.pdf` | stale numbers. The TTM figure path now points at `_archive_old/Zeroshot forecast/`; that figure is from the leaky experiment, so regenerate it when the slides are rebuilt (§6 G2) |

Run baselines **from inside `baselines/`**: paths are relative (`../data/preprocessed`, `results/`).

### Evaluation protocol (all baselines; AQFM must use the same one)
- Test origins: one per day at **00:00 IST** in 2025, `np.arange(test_start, test_end - 288, 96)`.
- Origin `t` predicts index `t + h - 1`, using information up to `t - 1`. So "1h" means the value at 00:45, and "24h" and "72h" both mean 23:45.
- Metrics are on raw µg/m³ at 15-min points, only where the target is observed. AQI bands (PM2.5, from y_true): Good 0–30, Satisfactory 30–60, Moderate 60–90, Poor 90–120, Very Poor 120–250, Severe ≥ 250.
- The classical baselines also require their own prediction to exist (walk-back ≤ 7 days), so their n is slightly smaller than XGBoost's.

---

## 3. Bugs found and FIXED (2026-10-08)

### D1. ERA5 for Apr 2017 → Dec 2020 was fabricated by interpolation (critical)
- Only 2017-01..03 and 2021-01..2025-12 had been downloaded. `interpolate_era5_to_stations.py` used `resample('15min').interpolate('linear')`, which drew a **straight line across the 3.75-year hole**. The per-step slope was exactly constant (verified): 12.5 °C at noon in June 2019, ssrd = 0 (no sunlight) and tp = 0 for 2018–2020.
- That fake data then "gap-filled" CPCB met channels (AT, RH, WS, WD, SR, BP, TOT_RF) for 2018–2020 with `observed = 1`, and was part of the "100% met coverage" claim. For ~40% of midday hours, SR was 0 at the affected stations.
- **Fix:** downloaded 2017-12 → 2020-12 with the existing `fetch_month()`. `interpolate_era5_to_stations.py` now uses `interpolate(..., limit=3, limit_area='inside')`, so only the 15-min slots inside an hour are filled and any future missing month stays NaN.

### D2. ERA5 is UTC, CPCB is IST: 5 h 30 min misalignment (critical)
- Verified: ERA5 solar radiation peaks at 07 UTC, while CPCB SR peaks at 12–13 h and CPCB AT at 13–14 h local time. Every ERA5-filled met value was 5.5 h out of phase.
- **Fix:** `preprocess_aqfm.py::merge_era5` now shifts ERA5 `time` by +5 h 30 min before aligning. The 2017-12 download is needed so that 2018-01-01 00:00 IST (= 2017-12-31 18:30 UTC) is covered.

### D3. BP channel had mixed units (significant)
- `BP (mmHg)` is reported in **hPa (~980)** by about half the stations (per-station medians were either ~740 or ~981). ERA5 fills were in mmHg, so within a single station the series jumped 980 ↔ 735. This also broke static feature 2 (the elevation proxy).
- **Fix:** new `fix_bp_units()` in `preprocess_aqfm.py`. Any CPCB BP reading > 900 is converted with ×0.750062. The ranges don't overlap in Delhi, so the rule is per value and also handles stations that change units mid-record.

### B1. XGBoost baseline used future data (critical: it was the headline benchmark)
- PM2.5 predictions were clipped to the **observed PM10 at the target time**: `values[s, 1, origin + h - 1]`. That is ground truth from the future. The old "PM2.5 24h MAE 36.15" was computed with this leak and with the corrupted met data from D1/D2.
- **Fix:** removed the clip. A legitimate version would need a PM10 *forecast*, not the observed value.

### D4. preprocess_aqfm.py ran out of memory on the rebuilt parquet (fixed)
- `merge_era5` did `pd.read_parquet()` on the whole 730 MB / 12M-row file and then added 10 derived float64 columns, needing several GB. On this 16 GB machine it died with `ArrowMemoryError: malloc of size 767415168 failed`.
- **Fix:** it now reads **one station at a time** with a pyarrow row filter (~324k rows each) and builds derived met as float32. Verified on a real day: 0 NaN, SR peaks at hour 12 IST, BLH 12–3206 m, AT 26.4–42.1 °C for 1 June.

### B2. Stale numbers in docs
- `baseline_evaluation_report.md`'s AQI table did not match `classical_results.json` (e.g. Persistence Severe 161.46 / n = 672 in the report vs 119.25 / n = 1248 in the JSON). The slides quote TTM "PM2.5 MAE 32.6 @96h" from the old leaky `Zeroshot forecast/run_ttm.py`. The report was regenerated (see §4); the slides are still stale.

### B3. Caveat, not fixed: TTM is not strictly apples-to-apples
- TTM runs on **hourly means** and is scored against hourly-mean truth. Every other baseline is scored on 15-min point values (which are noisier), so TTM gets a slightly easier target. Keep this in mind when comparing, or re-score TTM against 15-min points.

---

## 4. Baseline numbers on the REBUILT data (2026-10-08, honest)

Test year 2025, daily origins, metrics in µg/m³ (see the protocol in §2). Produced by `baselines/results/*.json`; the pre-fix versions are kept in `_archive_old/baseline_results_OLD_invalid/` for reference only.

### MAE (µg/m³)

| Model | Target | 1h | 6h | 24h | 72h |
|---|---|---|---|---|---|
| Persistence | **PM2.5** | 15.89 | 35.53 | 43.78 | 54.54 |
| Persistence | **PM10** | 31.28 | 71.47 | 84.64 | 106.47 |
| Seasonal Naive | **PM2.5** | 43.17 | 39.87 | 43.60 | 43.84 |
| Seasonal Naive | **PM10** | 81.21 | 66.60 | 84.18 | 84.55 |
| Climatology | **PM2.5** | 44.83 | 43.30 | 43.79 | 44.04 |
| Climatology | **PM10** | 83.18 | 70.78 | 82.76 | 82.97 |
| XGBoost | **PM2.5** | 18.17 | 30.06 | 38.63 | 43.74 |
| XGBoost | **PM10** | 34.71 | 49.44 | 73.65 | 83.46 |
| IBM TTM (zero-shot) | **PM2.5** | 20.23 | 32.43 | 37.70 | 41.32 |
| IBM TTM (zero-shot) | **PM10** | 37.55 | 52.80 | 71.74 | 79.25 |

### RMSE (µg/m³)

| Model | Target | 1h | 6h | 24h | 72h |
|---|---|---|---|---|---|
| Persistence | **PM2.5** | 29.85 | 60.85 | 75.85 | 92.13 |
| Persistence | **PM10** | 52.39 | 106.03 | 124.50 | 150.84 |
| Seasonal Naive | **PM2.5** | 75.31 | 65.36 | 74.98 | 75.52 |
| Seasonal Naive | **PM10** | 120.76 | 100.64 | 122.72 | 123.57 |
| Climatology | **PM2.5** | 69.19 | 63.32 | 68.52 | 68.92 |
| Climatology | **PM10** | 111.93 | 96.10 | 110.71 | 110.97 |
| XGBoost | **PM2.5** | 35.26 | 48.67 | 66.00 | 72.64 |
| XGBoost | **PM10** | 58.46 | 75.39 | 105.49 | 116.23 |
| IBM TTM (zero-shot) | **PM2.5** | 37.07 | 52.61 | 64.23 | 69.47 |
| IBM TTM (zero-shot) | **PM10** | 58.63 | 76.76 | 102.88 | 111.73 |

### PM2.5 at 24 h, MAE by AQI band

| Model | Good | Satisfactory | Moderate | Poor | Very Poor | Severe |
|---|---|---|---|---|---|---|
| **n** | 1553 | 2640 | 1597 | 1002 | 2246 | 1248 |
| Persistence | 16.7 | 20.4 | 28.3 | 41.2 | 60.2 | 119.3 |
| Seasonal Naive | 17.1 | 20.2 | 28.4 | 41.2 | 60.3 | 117.4 |
| Climatology | 30.5 | 26.0 | 34.0 | 42.1 | 47.1 | 105.8 |
| XGBoost | 21.8 | 17.5 | 23.2 | 35.1 | 47.6 | 110.5 |
| IBM TTM (zero-shot) | 18.6 | 17.1 | 23.0 | 35.0 | 49.8 | 112.4 |

### The number to beat

**XGBoost, PM2.5 @ 24 h = 38.63 MAE** (was 36.15 before the leakage fix). In the
Severe band (>250 µg/m³) the best baseline is Climatology at 105.8, with XGBoost
at 110.5 — every method is poor exactly where it matters most, which is the
opening AQFM has.

### What changed vs the old (invalid) numbers, and why

| Baseline | Change | Why |
|---|---|---|
| Persistence / Seasonal naive / Climatology | **identical** | They read only PM2.5/PM10 history, which was never corrupted. Expected and a good consistency check |
| XGBoost | **PM2.5 worse by +0.09 / +2.04 / +2.48 / +2.97** at 1/6/24/72 h | The removed leak (clipping predictions with the *observed future* PM10) was worth ~2.5 MAE at 24 h. PM10 is essentially unchanged (±0.5), as expected since only the PM2.5 path was leaking |
| IBM TTM | **identical** | See below |

> [!IMPORTANT]
> **TTM is channel-independent, so it cannot use weather at all.** Verified
> directly: scrambling the AT/RH/WS/WD inputs changes TTM's PM2.5/PM10 forecast
> by *exactly* 0.0, while its own met-channel forecasts do change (max Δ 4.14).
> Feeding it 6 channels is therefore pointless — it is effectively a univariate
> PM forecaster, which is why the ERA5 fixes left it untouched. Worth stating in
> the paper: it is a real limitation of the TTM baseline and part of the
> motivation for cross-channel attention in AQFM.

> [!NOTE]
> TTM is still scored on **hourly means** while every other baseline is scored on
> 15-min points (§3 B3), so its numbers are not strictly comparable.

---

## 5. Bugs in the frozen `files_AqFM/` (all fixed in the new `aqfm/`)

`files_AqFM/` was left untouched at the user's request. Its 10 tests pass but catch none of the problems below; each was reproduced with a script on synthetic data. The "Proposed fix" column is what `aqfm/` now does (see §7).

| # | Severity | Problem | Evidence | Proposed fix |
|---|---|---|---|---|
| M1 | **critical** | `data.py` applies `log1p` to pollutants **before** `augment_with_ratios`. So `pm_frac = logit(log1p(PM2.5)/log1p(PM10))`, but `physical.py` reconstructs PM2.5 = expm1(PM10) · σ(frac) in linear space. Wrong PM2.5. | true ratio 0.594 vs σ(pm_frac) 0.901 | call `augment_with_ratios` on the raw arrays first, then log1p channels 0–8 |
| M2 | **critical** | `AirQualityWindows.__init__` writes log1p **into the caller's array** (`np.asarray` does not copy float32). A second dataset built from the same array gets a double log1p; a read-only mmap crashes. | caller's PM2.5 81.8 → 4.42 after construction | fixed by the same reordering as M1 (`augment_with_ratios` returns new arrays) |
| M3 | **critical** | Pretraining masks run over context+horizon, but the encoder only sees `x_ctx` and the loss only reads horizon targets. **Tail:** lead patch 1 is scored in only 3.9% of samples vs 99.9% for patch 18, so short leads are barely trained. **Block/channel:** the context targets (~11% of cells) are never scored, so imputation is never learned, and only a few horizon cells get a loss. | measured per-lead coverage | user chose the **minimal fix**: always use the full horizon as the target; block/channel regimes corrupt the context input only; tail = plain forecasting. (Alternative: add decoder queries over context patches for real imputation.) |
| M4 | **critical** | PM2.5 is **not a decoder target** (`target_index = (PM10, pm_frac, NOx, nox_frac, O3, SO2, CO, NH3)`), and `train.step` never passes `physical_fn` to `total_loss`. So PM2.5, the headline metric, is never trained or scored directly. | read `train.py` | in `step`, denorm samples → reshape (n,B,S,Ct,H,P) → PM10 slot 0 and frac slot 1 → `PhysicalReconstruction` → energy score against log1p PM2.5 from `x_full[:, :, 0]` / `m_tgt[:, :, 0]` |
| M5 | high | Package not importable as documented: the tests do `from aqfm import ...`, but the folder is `files_AqFM` with no `__init__.py`. | — | add `__init__.py` (exports AQFM, AQFMConfig, AirQualityWindows, MaskSampler, synthetic_dataset) and either rename the folder to `aqfm` or change the imports. Tests were run from a scratch copy named `aqfm`. |
| M6 | high (perf) | `self.values[sel][..., sl]` copies the **whole time axis** of 12 stations (~256 MB) for every sample. | 20 ms per 32 MB measured | `self.values[sel, :, sl]` |
| M7 | medium | `train.evaluate()` pools MAE in log1p space across all 8 target slots (including logit fractions) and pools episode detection across channels. Not comparable to the baselines. | — | write a separate eval script that uses the §2 protocol, in µg/m³, for PM2.5 (reconstructed) and PM10 |
| M8 | medium | NO is in µg/m³ but NOx is in **ppb**. `nox_frac = NO/NOx` mixes units, so the NO2 = (1−f)·NOx reconstruction is not physical. | CSV headers | convert NO to ppb (×24.45/30.006) before taking the ratio; or drop the NOx reparam claim |
| M9 | medium | Wind direction is averaged per patch arithmetically (350° and 10° → 180°). Channel dropout can zero WS/WD before the wind bias is computed. | `model.encode` | circular mean via u/v components; take wind from unmasked met |
| M10 | medium | Fitted outage LogNormal (μ = 0.73, σ = 0.83, median 0.5 h) is dominated by 1–2 step glitches, so block masks are tiny. The design intends multi-hour station-wide outages. The config still has the defaults (3.0, 1.1). | `preprocessing_report.json` | decide: fit on gaps ≥ 1 patch, or keep defaults; then set the config explicitly |
| M11 | medium | `test_no_future_leakage` perturbs `x_full`, but the encoder reads `x_ctx`, so the test can never fail. | read test | perturb the dataset's horizon values before windowing and assert `x_ctx` and the encoder output are unchanged |
| M12 | low | Neighbourhood anchor is random and ignores index `i` → evaluation is not deterministic. One RNG is shared across DataLoader workers. | — | seed per index for eval |

Minor, left alone: static features are normalised including zero placeholders; `train_end = 7/8·T` approximates the end of 2024 (still before 2025, so no leak); the outage fit includes 2025 masks; `TOT_RF` semantics (CPCB per-15-min vs ERA5 per-hour accumulation) are unverified.

The design gaps (future met, BLH, imputation, continuous lead) are also closed in `aqfm/`.

---

## 6. TO DO — the full remaining roadmap

Status key: ⬜ not started · 🔄 in progress · ✅ done. Do the stages in order; each depends on the one before. **Nothing below has produced a real-data number yet.**

### Stage A — Data rebuild ✅ DONE (2026-10-08 19:05)

Verified: 37/37 months; no zip-in-.nc; 2018–2020 ERA5 has ~20k distinct values/year
(not 1, so no longer fabricated); `values.npy` (37, 18, 280512); `era5_met.npy`
(37, 8, 280512) with **0 NaN**; **5,410,316** BP readings converted hPa→mmHg
(station medians now all 733–742); **SR peaks at hour 12 IST** (was 7 = UTC);
BLH coverage 100%; PM2.5 coverage 86.1%.

- ✅ **A1. ERA5 download, 2017-12 → 2020-12** (37 months). Started 2026-10-08 14:08 IST; 8 months done by 15:11 (~8 min/month on the CDS queue, so ~4 h total). It runs `fetch_month()` from `data_pipeline/fetch_era5.py` in a loop.
  - Check progress: `ls data/era5_data_raw/*.nc | wc -l` (71 at 8 months; should reach **100** when all 37 are in).
  - If it died, just re-run it — `fetch_month` skips files that already exist:
    ```bash
    python -c "import sys; sys.path.insert(0,'data_pipeline'); from fetch_era5 import fetch_month; [fetch_month(y,m,'data/era5_data_raw') for y,m in [(2017,12)]+[(y,m) for y in (2018,2019,2020) for m in range(1,13)]]"
    ```
- ✅ **A2. `python data_pipeline/fix_era5_zips.py`** — the CDS now returns zip-in-`.nc`; every new file needs unpacking. **Verify:** `python -c "import zipfile,glob; print([f for f in glob.glob('data/era5_data_raw/*.nc') if zipfile.is_zipfile(f)])"` prints `[]`.
- ✅ **A3. `python data_pipeline/interpolate_era5_to_stations.py`** → rebuilds `data/era5_interpolated.parquet` (~730 MB, several minutes). The script is fixed (`limit=3, limit_area='inside'`), so a missing month now stays NaN instead of becoming a straight line.
  - **Verify (this is the D1 regression check):** per-year distinct values must be similar across 2018–2025. The old broken file had `ssrd` and `tp` with **1** distinct value for 2018, 2019 and 2020, versus ~20,000 for 2021+.
    ```bash
    python -c "
    import pyarrow.parquet as pq
    e = pq.read_table('data/era5_interpolated.parquet', filters=[('station_id','==','ashok_vihar,_delhi_-_dpcc_cleaned')]).to_pandas().set_index('time')
    print(e.groupby(e.index.year)[['ssrd','t2m','u10','blh']].nunique())"
    ```
- ✅ **A4. `python data_pipeline/preprocess_aqfm.py`** → rebuilds `data/preprocessed/` (~2.5 min). Now also writes **`era5_met.npy`** and a **BLH** channel.
  - **Verify:** `values.npy` is `(37, 18, 280512)`; `era5_met.npy` is `(37, 8, 280512)`; in `preprocessing_report.json`, `era5_met_nan_frac` ≈ 0 for every variable, `bp_hpa_converted` > 0, and BLH coverage ≈ 100%.
  - **Verify IST (D2) and BP (D3)** — solar radiation must peak at hour 12–13, and every station's BP median must be ~730–745:
    ```bash
    python -c "
    import numpy as np
    v=np.load('data/preprocessed/values.npy',mmap_mode='r'); o=np.load('data/preprocessed/observed.npy',mmap_mode='r')
    ts=np.load('data/preprocessed/timestamps_h.npy'); h=(ts%24).astype(int)
    sr=np.asarray(v[5,13]); ok=np.asarray(o[5,13])>0
    print('SR peak hour:', int(np.argmax([sr[ok&(h==k)].mean() for k in range(24)])))
    bp=np.asarray(v[:,14,::97]); ob=np.asarray(o[:,14,::97])>0
    print('BP medians:', sorted({int(np.median(bp[s][ob[s]])) for s in range(37)}))"
    ```
  - The old `preprocessing_result_aqfm` write-up is then stale: rewrite it from the new report.

### Stage B — Honest baselines ✅ DONE (2026-10-08 19:27)

- ✅ **B1. Re-ran all three baselines** on the rebuilt data (from inside `baselines/`): `run_baselines_classical.py`, `run_baselines_xgboost.py` (leakage fix applied → **expect the PM2.5 24 h MAE to be worse than the old 36.15**; that is the point), `run_baselines_ttm.py`.
- ✅ **B2. Filled in §4** of this file with the new numbers, and **regenerate** the tables in `docs/baselines/baseline_evaluation_report.md` directly from the JSONs (the current AQI table does not match them — §3 B2).
- ✅ **B3. Done (2026-10-11):** TTM is also scored on 15-min points, as entry `IBM TTM (zero-shot, 15-min scoring)` in `ttm_results.json`: PM2.5 MAE 20.95 / 32.13 / 39.16 / 42.75 at 1/6/24/72 h. The original hourly-mean entry `IBM TTM (zero-shot)` (§4) is unchanged. Also added: `run_baselines_xgboost.py --future-met` (XGBoost + ERA5 at the target time) → PM2.5 MAE 17.99 / 28.48 / **33.65** / 36.45; this is the fair bar for AQFM when it uses future ERA5 (see `docs/BOTTLENECKS_AND_IMPROVEMENTS.md` §3.1).

### Stage C — Train AQFM (needs a GPU)

- ⬜ **C1. Get GPU access.** This laptop is CPU-only and measured ~15 s per sample forward+backward, i.e. roughly a year of compute for a 100k-step run — not viable. Options: Colab/Kaggle (free tier, needs checkpointing to Drive because sessions are cut off), or a university cluster. Upload `aqfm/`, `tests/`, `data/preprocessed/` (~1.5 GB) and `baselines/`.
- ✅ **C0. Real-data integration verified (2026-10-08).** `AQData` builds from the rebuilt arrays in 10 s → (37, 20, 280512), **zero NaN** in both inputs and met; a forward+backward pass on a real batch gives finite losses (nll 2.06, crps 0.88, physical 0.53). Derived-channel coverage: **PM_FRAC 81.6%**, **NOX_FRAC 54.8%** (lower because the NO+NO₂=NOx identity filter rejects stations that violate it — expected, and better than silently fitting nonsense).
- ✅ **C1a. Outage distribution fitted on the real training split:** μ=3.932, σ=1.030 from **3,033** station-wide outages ≥ 1 patch → **median 12.8 h**. This is what `train.py --phase main` now sets automatically. (The old pipeline's 0.5 h figure was dominated by 1–2-step glitches and would have made block masking useless.)
- ⬜ **C2. Sanity run on real data** before the long one:
  `python -m aqfm.train --small --steps 500 --val-every 100 --out runs/sanity`
  **Verify:** the outage LogNormal fit prints a sensible median (hours, not minutes); the loss falls; `nll_impute`/`crps_impute` are non-zero; no NaN.
- ⬜ **C3. Phase 1 (mixed-masking pre-training).**
  `python -m aqfm.train --phase main --out runs/main --batch-size 8 --steps 100000 --grad-checkpoint --device cuda`
  Tune `--batch-size` to the card (start at 4 if it OOMs; `--grad-checkpoint` trades ~30% speed for ~4× less activation memory). Step count is a guess: stop on the validation plateau, which is what `--patience` does.
- ⬜ **C4. Phase 2 (forecast anneal).**
  `python -m aqfm.train --phase anneal --init runs/main/best.pt --out runs/anneal --steps 20000 --device cuda`
- ⬜ **C5. Watch for trouble** in `runs/*/log.jsonl`: validation rising while training falls (overfitting → more dropout/fewer steps); `gnorm` spiking (lower LR); `physical` term dominating (lower `w_physical`); imputation loss stuck (check that the block masks are actually large).

### Stage D — Evaluate

- ⬜ **D1.** `python -m aqfm.evaluate --ckpt runs/anneal/best.pt --out results/aqfm_results.json --device cuda`
  Produces: MAE/RMSE at 1/6/24/72 h for PM2.5 and PM10, AQI-band MAE at 24 h, episode precision/recall/F1 at 24 h and 48 h, CRPS by lead hour, PIT histograms, 90/95% coverage, and 24 h-outage imputation vs linear interpolation.
- ⬜ **D2. The three numbers that decide the paper:** (a) PM2.5 24 h MAE vs XGBoost, (b) **Severe-band (>250) MAE and episode recall**, (c) whether 90% intervals actually cover ~90%.
- ⬜ **D3. Repeat with ≥3 seeds** (`--seed`) and report mean ± std. A single seed is not evidence.

### Stage E — Ablations and the oracle gap (only once Stage D is in)

Each is a separate training run, so budget GPU time. All switches already exist (§7.5).
- ⬜ **E1. Oracle gap** — `--no-future-met`. **Mandatory for honesty:** ERA5 is reanalysis, not a forecast, so the main result overstates operational skill. Reviewers will ask.
- ⬜ **E2. No BLH** — the design doc expects this to be the largest single-feature effect. Needs a config/preprocessing switch to drop channel 17 (not yet implemented).
- ⬜ **E3. Spatial ablations** — `--no-spatial` (no cross-station attention) and `--spatial-bias distance` (drops the wind asymmetry).
- ⬜ **E4. No physical reparameterisation** — `target_channels` with `PM2.5` directly instead of `PM10 + PM_FRAC`. Count the PM2.5 > PM10 violations that appear.
- ⬜ **E5. Random-only masking** — `forecast_targets_always=False`, `p_tail_mask=0`.
- ⬜ **E6. Not implemented, build only if a reviewer needs it:** Design A (one decoder per pollutant) and a point head + MSE.

### Stage F — Competing published models (needed for a strong paper)

Run their public code on **our exact 2025 protocol**; do not copy numbers from their papers (different splits, stations and horizons). See §9.
- ⬜ **F1. PatchTST / iTransformer** (per-station, no spatial attention) — isolates the value of spatial attention.
- ⬜ **F2. AirFormer** (AAAI 2023, code public) — closest architecture: attention + uncertainty.
- ⬜ **F3. AirPhyNet** (ICLR 2024, code public) — closest physics claim.
- ⬜ **F4. E-STGCN** (arXiv 2411.12258) — reports results on **37 Delhi stations**, most likely the same network, so it is the closest direct competitor.

### Stage G — Write-up

- ⬜ **G1. Literature review** before any novelty claim (§9 is a starting list, not a review).
- ⬜ **G2. Update** `baseline_evaluation_report.md`, the slides (`presentation/build_pptx_js.js`) and `aqfm_design.tex` with real numbers; remove every stale number (§3 B2).
- ⬜ **G3. Document the deviations from the design doc (§7.6) and the deliberate deviations from the old code** in the methods section.
- ⬜ **G4. Naming:** do **not** call it a "foundation model" without cross-region transfer — `docs/aqfm_implementation_plan.md` says so explicitly. "Spatio-temporal multi-pollutant forecasting model" is defensible.
- ⬜ **G5. Target venue** — decide from the results (§10).

### Stage H — Open scientific risks to address in the paper

- ⬜ **H1. COVID (2020)** is inside the training window and emissions were abnormal. Run a train-without-2020 comparison, or at least discuss it.
- ⬜ **H2. One test year** (2025) may measure drift rather than skill. The design doc asks for rolling-origin backtests inside the training period.
- ⬜ **H3. Station representativeness** — a kerbside and a background monitor are different measurement processes. The design doc wants a site-type static covariate; we do not have that metadata yet.
- ⬜ **H4. Known minor gaps** (§5 "Minor, left alone"): `TOT_RF` accumulation semantics unverified; static features normalised including zero placeholders; in the physical CRPS, PM10 and the fraction are sampled independently.

---

## 7. The model: `aqfm/` (built 2026-10-08)

Built to `files_AqFM/aqfm_design.tex`, with the user's four decisions: a new package (old one frozen); known-future ERA5 met in the decoder behind a switch; full imputation (trained and scored separately); BLH as an input channel. **Status: implemented, 31 tests passing, CLI smoke runs pass (main + anneal + evaluate). NOT yet trained on real data.**

### 7.1 Files
| File | Contents |
|---|---|
| `config.py` | `AQFMConfig` (reference size = 23.79M params, measured), `.small()`; **all channel bookkeeping** (`RAW_CHANNELS` 18, `INPUT_CHANNELS` 20, `CH`, `LOG_CHANNELS`, `OUTAGE_CHANNELS`, `ERA5_MET_CHANNELS`) |
| `physical.py` | `derive_inputs` (raw 18 ch → 20 model channels), `reconstruct_pm25`, `reconstruct_no_no2` |
| `masking.py` | `MaskSampler` (tail/block/channel, context only), `enforce_derived_visibility`, `fit_block_lognormal` |
| `layers.py` | `MaskedRevIN`, `TimeFeatures`, `LeadFeatures`, `MaskedMHA`, `SpatialBias` (+ `wind_geometry`), `AxialBlock` |
| `model.py` | `PatchTokenizer`, `StationPooler`, `QueryDecoder`, `MixtureHead`, `AQFM` |
| `losses.py` | `mixture_nll`, `crps_mixture` (exact closed form), `sample_crps` (order-statistic), `total_loss` |
| `data.py` | `AQData` (transformed arrays + train-only stats), `make_sample`, `TrainWindows`, `FixedWindows`, `build_queries`, `synthetic_raw` |
| `train.py` | CLI: `--phase main|anneal`, date splits, outage fit, AdamW + cosine, bf16 on CUDA, validation, `best.pt`/`last.pt`, early stopping, ablation flags |
| `evaluate.py` | CLI: the 2025 test protocol, using `baseline_utils` metric functions directly, plus extras (below) |

### 7.2 Data flow
- **Inputs (20 channels):** 9 pollutants, AT, RH, **U, V** (WS/WD converted, because degrees can't be averaged or normalised), SR, BP, TOT_RF, Benzene, **BLH**, **PM_FRAC** = logit(PM2.5/PM10), **NOX_FRAC** = logit(NO_ppb/NOx_ppb). Ratios are computed in physical units before log1p. NOX_FRAC is observed only where NO_ppb + NO2_ppb is within 10% of NOx (re-checked 2026-10-12 with `docs/data_report_stats.py`: 26 of 36 stations have a median (NO+NO2)/NOx within ±1%, but 8 stations, 7 IITM ones and Rohini, sit at 0.66–0.71). log1p is applied to pollutants, TOT_RF, Benzene and BLH.
- **Splits:** train 2018–2023 (window horizons end before 2024-01-01), validation 2024 (fixed 512 random forecast windows, seed 12345), test 2025. Train-only statistics: RevIN fallback, ERA5 standardisation, outage LogNormal (station-wide pollutant gaps ≥ 1 patch).
- **Sample** = an anchor station plus its 11 nearest neighbours (anchor is station 0); context 512 steps; horizon 288 steps.

### 7.3 Architecture
1. **MaskedRevIN** per (sample, station, channel) from *visible context only*. Fallback when fewer than 16 visible steps: global train stats. The std is floored at 0.1 × the global std, so a stuck sensor can't blow the targets up.
2. **Tokens** (B,S,C,32,D) = Linear([values; mask] of a 16-step patch) + channel embedding + station MLP(coords, static) + Fourier time (6 log bands from 3 h to 1 yr, plus day/week/year cycles, float64) + **log(1 + hours since this series' previous observed patch)**. Fully unobserved patches are excluded from every key set (an all-invalid row is left unmasked, so there's no NaN).
3. **Encoder:** 4 pre-norm axial blocks (time → channel → space → FFN). Spatial attention gets a per-head **wind-aware bias** MLP(log(1+d), cos and sin of the angle between bearing(i→j) and the wind's FROM direction at i, wind speed, cos·speed, exp(−d/25 km)). Wind comes from the masked patch mean of U/V (circular-safe).
4. **StationPooler:** Perceiver cross-attention to 16 latents per station (latents carry station identity).
5. **Queries:** target-slot embedding + task embedding (forecast or impute) + **continuous signed lead** (hours from the origin; negative for imputation) + station MLP + Fourier time + (optional) Linear(**future ERA5 met patch mean**). Forecast queries form a dense grid (12 stations × 8 slots × 18 patches = 1728). Imputation queries cover all 8 slots at ≤48 sampled hidden (station, patch) positions.
6. **Decoder:** 4 pre-norm layers of {query self-attention (padding-masked), cross-attention to latents, FFN}.
7. **Head:** a 3-component Gaussian mixture per 15-min step, in normalised transformed space.
8. **Targets (8 slots):** PM10, PM_FRAC, NOx, NOX_FRAC, Ozone, SO2, CO, NH3. PM2.5 = PM10·σ(frac); NO and NO2 come from NOx and the fraction (in ppb, converted back to µg/m³).

### 7.4 Training objective
- **Loss:** NLL + 0.5 × **exact mixture CRPS** (closed form, Grimit et al. 2006, so no sampling noise), plus 0.25 × **sample CRPS** (order-statistic, 32 samples) of the reconstructed PM2.5, NO and NO2 in log1p space. Optional episode weighting (`episode_alpha`, default 0). Forecast and imputation losses are logged separately.
- **Phase 1** (main, `pretrain=True`): every sample has the full horizon as its forecasting target (all 18 leads are trained equally). The context is corrupted by tail (50%, none), block (35%, 1–4 station-wide outages of the CPCB channels, LogNormal lengths) or channel (15%, a random pollutant subset over a random window). Met is never hidden (it is ERA5-backed at inference). A fraction is hidden whenever any of its constituents is (this prevents the PM10 × frac → PM2.5 leak). Hidden-but-observed cells become imputation targets.
- **Phase 2** (anneal): pure forecasting, lr 5e-5.
- Checkpoints are selected on the validation NLL + CRPS (the deterministic part).

### 7.5 Evaluation (`evaluate.py`) and ablation switches
- The same origins, horizons, metric functions and AQI bands as the baselines (it imports `baseline_utils`). The point forecast is the median of 200 samples. Extras: CRPS by lead hour, PIT histograms, 90/95% coverage, PM2.5 > 250 episode precision/recall/F1 at 24 h and 48 h, and **imputation of a 24 h station-wide outage** scored against linear interpolation on the same cells.
- Switches (CLI flags / config): `--no-future-met` (oracle gap), `--spatial-bias distance|none`, `--no-spatial`, `forecast_targets_always=False` + `p_tail_mask=0` (random-only masking), `target_channels=(..."PM2.5"...)` without fractions (no physical reparam). Not implemented: Design A decoder and the point-head + MSE ablation.

### 7.6 Deviations from the design doc (deliberate; mention them in the paper)
- RevIN fallback uses global train stats, not the identity; the std floor is 0.1 × global.
- The main CRPS term is the exact closed form; the order-statistic estimator is used only for the reconstructed quantities.
- Wind is represented as U/V; NOx fractions are in ppb with an identity filter.
- The gap feature is per series (time since that series was last observed), not the gap between patch timestamps (which is always 4 h on a dense grid).
- The tail regime adds no corruption; the forecast horizon is a target in every regime (the design's "mask the last k patches" left short leads almost untrained in the old code).
- Station outages hide pollutant channels only, not met.
- Known simplification: in the physical CRPS, PM10 and the fraction are sampled independently per query (their correlation is not modelled).
- Samples are capped at log1p = 20 (about 4.9e8 ug/m3) before back-transformation, to prevent float32 overflow.

### 7.7 Verification done
- 31 pytest tests: horizon-perturbation invariance (bit-identical encoder), RevIN visible-only statistics, mask partition and derived-visibility rules, block outages are station-wide, every lead is a target, imputation targets equal the hidden truth, alignment with the baseline protocol, CRPS (order-statistic == pairwise exactly; closed form == Monte Carlo; known single-Gaussian value), NLL == `torch.distributions`, no gradient from unobserved targets, gap feature, wind geometry (upwind ⇒ cos = 1), station-permutation equivariance, gradient checkpointing gives identical gradients, all ablation configs run, a fixed batch can be overfitted, the outage fit recovers its parameters, and the evaluation pipeline runs end to end.
- **Mutation check:** 6 deliberately injected bugs (no derived-visibility rule, ratio in the wrong space, RevIN using hidden cells, context shifted into the horizon, CRPS sign, wind FROM/TO flipped) are each caught by a test.
- Not yet verified: anything on real data (losses, speed and memory on a GPU, skill).

---

## 8. How to reproduce the whole pipeline

```bash
# from repo root
python data_pipeline/fetch_era5.py              # main loop covers 2021-2025; 2017-12..2020-12 fetched separately via fetch_month()
python data_pipeline/fix_era5_zips.py           # CDS returns zip-in-.nc; run after every download
python data_pipeline/interpolate_era5_to_stations.py   # -> data/era5_interpolated.parquet
python data_pipeline/preprocess_aqfm.py                    # -> data/preprocessed/  (ROOT is hard-coded to c:\Users\phala\Foundation Model)
cd baselines
python run_baselines_classical.py
python run_baselines_xgboost.py
python run_baselines_xgboost.py --future-met # same + ERA5 weather at the target time -> xgboost_future_met_results.json
python run_baselines_ttm.py                  # needs granite-tsfm; model cached in ~/.cache/huggingface
cd ..
python -m pytest tests -q                    # 31 tests, ~30 s on CPU
python -m aqfm.train --phase main   --out runs/main   --grad-checkpoint --device cuda
python -m aqfm.train --phase anneal --init runs/main/best.pt --out runs/anneal --device cuda
python -m aqfm.evaluate --ckpt runs/anneal/best.pt --out results/aqfm_results.json --device cuda
```

---

## 9. Related work to compare against (verified to exist, 2026-10-08)

Details below come from abstracts and listings; **read each paper's data and protocol section before citing or comparing**. Numbers in their papers are not comparable to ours (different splits, stations, horizons) — re-run their code on our protocol instead (Stage F).

| Work | Venue | Why it matters here |
|---|---|---|
| **E-STGCN** — graph conv + LSTM + generalized Pareto for extremes, [arXiv 2411.12258](https://www.arxiv.org/pdf/2411.12258) | 2024 | Uses **37 Delhi stations** — almost certainly the same CPCB network. Closest direct competitor; also targets extreme events |
| **AirFormer** — two-stage transformer (deterministic + stochastic latent), [arXiv](https://arxiv.org/pdf/2211.15979), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26676), code `yoshall/airformer` | AAAI 2023 | Closest architecture: attention over space/time **with uncertainty**, 1,085 stations in China, 72 h horizon like ours |
| **AirPhyNet** — diffusion/advection differential equations in a graph net, [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c52cfe4455c441c691663a2ff5b0b2b1-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2402.03784v2), code `kethmih/AirPhyNet` | ICLR 2024 | Closest **physics** claim; also evaluates sparse-data and sudden-change settings |
| **STGCN variants for Delhi PM2.5**, [listing](https://ideas.repec.org/a/eee/tefoso/v208y2024ics0040162524004827.html) | TF&SC 2024 | Delhi-specific graph baseline |
| **STEEP** — co-occurrence graph + DCRNN, [IDA](https://journals.sagepub.com/doi/10.3233/IDA-240028) | 2025 | Delhi usage unconfirmed — check before citing as Delhi work |
| **BiGRU-1DCNN hybrid**, Delhi 2018–2023, [summary](https://communities.springernature.com/amp/posts/spatio-temporal-forecasting-using-a-hybrid-bigru-1dcnn-model-for-pm2-5-concentrations-in-delhi-india-2018-2023-across-multiple-monitoring-stations) | — | Non-graph Delhi deep-learning baseline |
| **PatchTST / iTransformer** | 2023 | Channel-independent, per-station: isolates the value of spatial attention (the design doc asks for this) |
| **Aurora** — 1.3 B-param atmosphere foundation model that also does global air pollution, [MSR](https://microsoft.com/en-us/research/publication/aurora-a-foundation-model-of-the-atmosphere) | Nature, May 2025 | **Cite for positioning only, do not compete.** Also the reason to be careful with the term "foundation model" (G4) |

---

## 10. Publication strategy (what to aim for depending on the result)

| Outcome | Realistic venue |
|---|---|
| Clearly beats XGBoost **and** the deep baselines, with ablations | Top ML conference (AAAI / KDD / ICLR) or a top journal |
| Comparable average error but better **Severe-band** skill, calibration, or imputation | Domain journals (*Atmospheric Environment*, *Environmental Modelling & Software*, *Science of the Total Environment*) — check each one's scope |
| Worse overall, honestly analysed | Workshop (e.g. Climate Change AI at NeurIPS/ICLR), applied journal, or a benchmark/dataset paper |

Contributions that stand even without the best average MAE:
1. **A clean, reproducible Delhi benchmark** — the data pitfalls found here (fabricated interpolated ERA5, UTC/IST mismatch, mixed BP units, NO/NOx unit mismatch, PM10 leakage in a common GBM setup) are a contribution in themselves.
2. **Severe-episode skill** (pooled MAE is dominated by clean summer months).
3. **Calibrated predictive intervals** — most baselines emit a point value.
4. **Imputation during real station outages**, scored against interpolation.
5. **Interpretability** — which upwind stations the model attends to under different wind regimes; the design doc calls this publishable on its own.
6. **Ablations** quantifying BLH, wind-aware attention, physical constraints and future met.

**Do not:** tune on 2025, report only the best seed, or drop a baseline that wins. Reviewers check, and it sinks papers.
