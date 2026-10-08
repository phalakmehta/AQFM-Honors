# AQFM: Project Context for a New Session

> Read this first. It tells you what the project is, what state each part is in, which bugs were found and fixed on 2026-10-08, which known bugs remain, and what to do next.
> Older docs (`project_handoff.md`, `preprocessing_result_aqfm`, `baseline_evaluation_docs/baseline_evaluation_report.md`, the presentation PDF) contain **numbers and claims that are now stale**. Where they disagree with this file, this file wins.

---

## 0. Ground rules from the user (see also `CLAUDE.md`)

- **Never commit or push to git.** The user does all git operations; Claude must not appear as a contributor.
- **Do not edit `files_AqFM/` without explicit permission.** It is the original reference implementation and is kept frozen. **The model being developed is the new package `aqfm/`** (§8), built on 2026-10-08 with the user's go-ahead.
- This is a research project aimed at publication. Correctness and honesty over speed: every leakage path needs a test, and every claim needs a measurement.
- Verify everything by running it. Don't fabricate numbers. Ask when something is ambiguous.
- Machine: Windows 11, Python 3.12.5, 16.5 GB RAM, **torch 2.10 CPU-only (no CUDA GPU)**, numpy 2.0.2, pandas 3.0.5, xgboost 3.0.5, granite-tsfm 0.3.9, xarray, pyarrow, pytest 8.3, cdsapi (with `~/.cdsapirc` configured). The full model (23.79M params) takes **~15 s per sample (forward + backward) on this CPU** (measured), so real training needs a GPU (Colab/Kaggle/university). CPU is fine for tests and `--small` runs.

---

## 1. What the project is

Honours project by Phalak Mehta: **"Air Quality Forecasting using Foundation Models: a spatio-temporal deep-learning approach for multi-pollutant forecasting in Delhi."**

- **Data:** CPCB (Central Pollution Control Board) ground stations in Delhi NCR, 15-minute cadence, 2018–2025, **37 stations** (38 are listed in `stations.csv`; `alipur` has no CSVs). Plus ERA5 reanalysis weather (incl. boundary-layer height, BLH) interpolated to the station locations.
- **Model (AQFM):** channel-independent patch tokens (one token = 1 station × 1 channel × 16 steps, i.e. 4 h). Masked RevIN normalisation; an axial encoder (time → channel → space attention) with a wind-aware spatial bias; a Perceiver-style pooler giving 16 latents per station; a **query-based decoder** (query = target channel + station + lead patch + time); a mixture-of-3-Gaussians head in normalised log1p space; NLL + CRPS loss (order-statistic estimator). Physical reparameterisation: PM2.5 = PM10 × σ(f_pm) and NO = NOx × σ(f_nox).
- **Training plan:** Delhi-only, two phases. (1) Mixed masking (tail 50% / block 35% / channel 15%). (2) Anneal with tail-only masking at lower LR. National pre-training is out of scope for v1 (`aqfm_implementation_plan.md`). The plan also says not to call it a "foundation model" in write-ups until cross-region transfer is shown.
- **Goal:** beat the XGBoost baseline on PM2.5 at the 24 h horizon on the held-out 2025 test year, and do well in the **Severe AQI band (PM2.5 > 250)**. Also report episode recall/precision, calibration (PIT, interval coverage), and CRPS by lead time.

Key design references: `files_AqFM/aqfm_design.tex` (full spec), `aqfm_implementation_plan.md` (Delhi-only v1 plan), `aqfm honors final presentation.pdf` (what was presented).

---

## 2. Repository map

| Path | What it is | Status |
|---|---|---|
| `yearly_data/<year>/<station>.csv` | Raw CPCB CSVs (gitignored). Columns: `Station, Timestamp, PM2.5 (µg/m³), PM10, NO (µg/m³), NO2 (µg/m³), NOx (ppb), NH3, SO2, CO (mg/m³), Ozone, Benzene, …, AT, RH, WS, WD, RF, TOT-RF, SR, BP (mmHg), VWS`. Timestamps are **local time (IST)**. | raw input |
| `stations.csv` | station_id, lat, lon (38 rows) | ok |
| `era5script/fetch_era5.py` | CDS API download, one NetCDF per month → `era5_data_raw/` | ok. Its `__main__` loop only covers 2021–2025; see §3 D1 |
| `era5script/fix_era5_zips.py` | Unpacks months the CDS returned as zip-in-.nc | ok |
| `era5script/interpolate_era5_to_stations.py` | Bilinear to stations, hourly → 15 min → `era5_interpolated.parquet` | **fixed** (§3 D1) |
| `preprocess_aqfm.py` | CPCB + ERA5 → `preprocessed/{values,observed}.npy (37,18,280512)` (17 CPCB channels + **BLH** as channel 17), **`era5_met.npy (37,8,280512)`** (pure ERA5, IST: AT, RH, U, V, SR, BP, TOT_RF, BLH; NaN = missing), `timestamps_h.npy`, `coords.npy`, `static.npy (37,4)`, `preprocessing_report.json` | **fixed** (§3 D2, D3), BLH + era5_met added. **Must be re-run** once the ERA5 download completes; the arrays on disk are still the old 17-channel ones |
| `aqfm/` | **The model (new).** See §8 | built and tested, not yet trained |
| `tests/test_aqfm.py` | 31 tests for `aqfm/` (`python -m pytest tests -q`) | all pass |
| `baseline_codes/baseline_utils.py` | Shared split (test = 2025), horizons {1h:4, 6h:24, 24h:96, 72h:288 steps}, MAE/RMSE, AQI bands, episode detection | ok |
| `baseline_codes/run_baselines_classical.py` | Persistence, seasonal-naive, climatology | ok |
| `baseline_codes/run_baselines_xgboost.py` | XGBoost with lagged features, one model per (target, horizon) | **fixed** (§3 B1) |
| `baseline_codes/run_baselines_ttm.py` | IBM Granite TTM r2, zero-shot, hourly | ok (see caveat in §3 B3) |
| `baseline_results/*.json` | Baseline metrics | regenerated (see §4) |
| `files_AqFM/` | Original reference model + design `.tex/.pdf` + README | **frozen**; superseded by `aqfm/`; its bugs are listed in §5 (all fixed in `aqfm/`) |
| `Zeroshot forecast/` | Old, leaky TTM experiment (source of the "32.6 MAE" on the slides) | superseded by `run_baselines_ttm.py` |
| `plots/` | EDA figures | ok |
| `build_pptx_js.js` | Generates the presentation | stale numbers |

Run baselines **from inside `baseline_codes/`**: paths are relative (`../preprocessed`, `../baseline_results`).

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

### B2. Stale numbers in docs
- `baseline_evaluation_report.md`'s AQI table did not match `classical_results.json` (e.g. Persistence Severe 161.46 / n = 672 in the report vs 119.25 / n = 1248 in the JSON). The slides quote TTM "PM2.5 MAE 32.6 @96h" from the old leaky `Zeroshot forecast/run_ttm.py`. The report was regenerated (see §4); the slides are still stale.

### B3. Caveat, not fixed: TTM is not strictly apples-to-apples
- TTM runs on **hourly means** and is scored against hourly-mean truth. Every other baseline is scored on 15-min point values (which are noisier), so TTM gets a slightly easier target. Keep this in mind when comparing, or re-score TTM against 15-min points.

---

## 4. Current baseline numbers (after fixes)

RESULTS_PLACEHOLDER

---

## 5. Bugs in the frozen `files_AqFM/` (all fixed in the new `aqfm/`)

`files_AqFM/` was left untouched at the user's request. Its 10 tests pass but catch none of the problems below; each was reproduced with a script on synthetic data. The "Proposed fix" column is what `aqfm/` now does (see §8).

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

## 6. What to do next (in order)

1. **Finish the data rebuild** (in progress on 2026-10-08): wait for the ERA5 download (2017-12 → 2020-12; progress is the number of `era5_delhi_201[7-9]_*.nc` / `2020_*.nc` files in `era5_data_raw/`), then `python era5script/fix_era5_zips.py`, `python era5script/interpolate_era5_to_stations.py`, then `python preprocess_aqfm.py`. **Verify:** the per-year unique-value counts of ERA5 variables are similar across 2018–2025 (no straight line); SR peaks at 12–13 h IST; per-station BP medians are all ~730–745 mmHg; `era5_met_nan_frac` ≈ 0; `values.npy` has 18 channels.
2. Re-run the baselines (classical, XGBoost, TTM) on the rebuilt data and fill in §4. Regenerate the AQI table in `baseline_evaluation_report.md` from the JSONs.
3. Train on a GPU. First a quick `--small` run on real data to check the losses fall, then:
   `python -m aqfm.train --phase main --out runs/main --batch-size 8 --steps 100000 --grad-checkpoint` (tune batch size and steps to the GPU), then
   `python -m aqfm.train --phase anneal --init runs/main/best.pt --out runs/anneal --steps 20000`.
   Watch `runs/*/log.jsonl`: train and val curves, `crps_impute`, `physical`.
4. `python -m aqfm.evaluate --ckpt runs/anneal/best.pt --out results/aqfm_results.json`. Compare against `baseline_results/*.json` (same protocol and the same metric code).
5. Run the oracle-gap experiment (`--no-future-met`) and the ablations (§8.5) **only after** the main model beats XGBoost. Repeat with ≥3 seeds and report mean ± std.
6. Write-up: update the report, slides and `aqfm_design.tex` with honest numbers. Report the oracle gap (ERA5 is reanalysis, not a real forecast). Do a literature review to position the work (e.g. AirFormer, AAAI 2023, and other spatio-temporal transformers for air quality) before making novelty claims.

---

## 8. The model: `aqfm/` (built 2026-10-08)

Built to `files_AqFM/aqfm_design.tex`, with the user's four decisions: a new package (old one frozen); known-future ERA5 met in the decoder behind a switch; full imputation (trained and scored separately); BLH as an input channel. **Status: implemented, 31 tests passing, CLI smoke runs pass (main + anneal + evaluate). NOT yet trained on real data.**

### 8.1 Files
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

### 8.2 Data flow
- **Inputs (20 channels):** 9 pollutants, AT, RH, **U, V** (WS/WD converted, because degrees can't be averaged or normalised), SR, BP, TOT_RF, Benzene, **BLH**, **PM_FRAC** = logit(PM2.5/PM10), **NOX_FRAC** = logit(NO_ppb/NOx_ppb). Ratios are computed in physical units before log1p. NOX_FRAC is observed only where NO_ppb + NO2_ppb is within 10% of NOx (verified: median ratio 0.999–1.001 at most stations; one station sits at 0.667). log1p is applied to pollutants, TOT_RF, Benzene and BLH.
- **Splits:** train 2018–2023 (window horizons end before 2024-01-01), validation 2024 (fixed 512 random forecast windows, seed 12345), test 2025. Train-only statistics: RevIN fallback, ERA5 standardisation, outage LogNormal (station-wide pollutant gaps ≥ 1 patch).
- **Sample** = an anchor station plus its 11 nearest neighbours (anchor is station 0); context 512 steps; horizon 288 steps.

### 8.3 Architecture
1. **MaskedRevIN** per (sample, station, channel) from *visible context only*. Fallback when fewer than 16 visible steps: global train stats. The std is floored at 0.1 × the global std, so a stuck sensor can't blow the targets up.
2. **Tokens** (B,S,C,32,D) = Linear([values; mask] of a 16-step patch) + channel embedding + station MLP(coords, static) + Fourier time (6 log bands from 3 h to 1 yr, plus day/week/year cycles, float64) + **log(1 + hours since this series' previous observed patch)**. Fully unobserved patches are excluded from every key set (an all-invalid row is left unmasked, so there's no NaN).
3. **Encoder:** 4 pre-norm axial blocks (time → channel → space → FFN). Spatial attention gets a per-head **wind-aware bias** MLP(log(1+d), cos and sin of the angle between bearing(i→j) and the wind's FROM direction at i, wind speed, cos·speed, exp(−d/25 km)). Wind comes from the masked patch mean of U/V (circular-safe).
4. **StationPooler:** Perceiver cross-attention to 16 latents per station (latents carry station identity).
5. **Queries:** target-slot embedding + task embedding (forecast or impute) + **continuous signed lead** (hours from the origin; negative for imputation) + station MLP + Fourier time + (optional) Linear(**future ERA5 met patch mean**). Forecast queries form a dense grid (12 stations × 8 slots × 18 patches = 1728). Imputation queries cover all 8 slots at ≤48 sampled hidden (station, patch) positions.
6. **Decoder:** 4 pre-norm layers of {query self-attention (padding-masked), cross-attention to latents, FFN}.
7. **Head:** a 3-component Gaussian mixture per 15-min step, in normalised transformed space.
8. **Targets (8 slots):** PM10, PM_FRAC, NOx, NOX_FRAC, Ozone, SO2, CO, NH3. PM2.5 = PM10·σ(frac); NO and NO2 come from NOx and the fraction (in ppb, converted back to µg/m³).

### 8.4 Training objective
- **Loss:** NLL + 0.5 × **exact mixture CRPS** (closed form, Grimit et al. 2006, so no sampling noise), plus 0.25 × **sample CRPS** (order-statistic, 32 samples) of the reconstructed PM2.5, NO and NO2 in log1p space. Optional episode weighting (`episode_alpha`, default 0). Forecast and imputation losses are logged separately.
- **Phase 1** (main, `pretrain=True`): every sample has the full horizon as its forecasting target (all 18 leads are trained equally). The context is corrupted by tail (50%, none), block (35%, 1–4 station-wide outages of the CPCB channels, LogNormal lengths) or channel (15%, a random pollutant subset over a random window). Met is never hidden (it is ERA5-backed at inference). A fraction is hidden whenever any of its constituents is (this prevents the PM10 × frac → PM2.5 leak). Hidden-but-observed cells become imputation targets.
- **Phase 2** (anneal): pure forecasting, lr 5e-5.
- Checkpoints are selected on the validation NLL + CRPS (the deterministic part).

### 8.5 Evaluation (`evaluate.py`) and ablation switches
- The same origins, horizons, metric functions and AQI bands as the baselines (it imports `baseline_utils`). The point forecast is the median of 200 samples. Extras: CRPS by lead hour, PIT histograms, 90/95% coverage, PM2.5 > 250 episode precision/recall/F1 at 24 h and 48 h, and **imputation of a 24 h station-wide outage** scored against linear interpolation on the same cells.
- Switches (CLI flags / config): `--no-future-met` (oracle gap), `--spatial-bias distance|none`, `--no-spatial`, `forecast_targets_always=False` + `p_tail_mask=0` (random-only masking), `target_channels=(..."PM2.5"...)` without fractions (no physical reparam). Not implemented: Design A decoder and the point-head + MSE ablation.

### 8.6 Deviations from the design doc (deliberate; mention them in the paper)
- RevIN fallback uses global train stats, not the identity; the std floor is 0.1 × global.
- The main CRPS term is the exact closed form; the order-statistic estimator is used only for the reconstructed quantities.
- Wind is represented as U/V; NOx fractions are in ppb with an identity filter.
- The gap feature is per series (time since that series was last observed), not the gap between patch timestamps (which is always 4 h on a dense grid).
- The tail regime adds no corruption; the forecast horizon is a target in every regime (the design's "mask the last k patches" left short leads almost untrained in the old code).
- Station outages hide pollutant channels only, not met.
- Known simplification: in the physical CRPS, PM10 and the fraction are sampled independently per query (their correlation is not modelled).

### 8.7 Verification done
- 31 pytest tests: horizon-perturbation invariance (bit-identical encoder), RevIN visible-only statistics, mask partition and derived-visibility rules, block outages are station-wide, every lead is a target, imputation targets equal the hidden truth, alignment with the baseline protocol, CRPS (order-statistic == pairwise exactly; closed form == Monte Carlo; known single-Gaussian value), NLL == `torch.distributions`, no gradient from unobserved targets, gap feature, wind geometry (upwind ⇒ cos = 1), station-permutation equivariance, gradient checkpointing gives identical gradients, all ablation configs run, a fixed batch can be overfitted, the outage fit recovers its parameters, and the evaluation pipeline runs end to end.
- **Mutation check:** 6 deliberately injected bugs (no derived-visibility rule, ratio in the wrong space, RevIN using hidden cells, context shifted into the horizon, CRPS sign, wind FROM/TO flipped) are each caught by a test.
- Not yet verified: anything on real data (losses, speed and memory on a GPU, skill).

---

## 7. How to reproduce the data pipeline

```bash
# from repo root
python era5script/fetch_era5.py              # main loop covers 2021-2025; 2017-12..2020-12 were fetched separately via fetch_month()
python era5script/fix_era5_zips.py           # only if CDS returned zips
python era5script/interpolate_era5_to_stations.py   # -> era5_interpolated.parquet
python preprocess_aqfm.py                    # -> preprocessed/  (ROOT is hard-coded to c:\Users\phala\Foundation Model)
cd baseline_codes
python run_baselines_classical.py
python run_baselines_xgboost.py
python run_baselines_ttm.py                  # needs granite-tsfm; model cached in ~/.cache/huggingface
```
