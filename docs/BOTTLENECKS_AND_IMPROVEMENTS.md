# AQFM: Bottlenecks and How to Improve Them

**Document date:** 11 October 2026 (status updated 12 October 2026)
**Scope:** everything that currently slows the project down, limits the quality of its results, or could make a reviewer reject the conclusions. Each item gives the evidence, its impact, and a concrete fix.

**How to read the evidence:** ✅ *measured* = run and checked on 11 Oct 2026 (or on the stated date). 📖 *from code* = read directly in the source, with file and line. ❓ *unverified* = a reasoned expectation that still needs to be checked; the check is given.

---

## Priority overview

| # | Bottleneck | Area | Priority | Effort |
|---|---|---|---|---|
| 1 | No GPU, so AQFM has never been trained | Compute | **P0, blocks everything** | access, not code |
| 2 | 3 stations have fallback coordinates | Data | **P0, silent error** | small |
| 3 | ✅ Fixed: station-density static feature was empty (loop bug) | Data | done | done |
| 4 | AQFM gets future weather; XGBoost did not (✅ XGBoost + future ERA5 added: 33.65 at 24 h; AQFM oracle gap still to run) | Methodology | **P0, fairness** | medium |
| 5 | Evaluation computes 12 stations but scores 1 | Compute | P1 | small–medium |
| 6 | Every attention call carries a mask | Compute | P1 | small |
| 7 | Data upload to Colab: ~1.8 GB, hard-coded paths | Workflow | P1 | small |
| 8 | Single test year; COVID year in training | Methodology | P1 | medium |
| 9 | ✅ Fixed: reproducibility (pinned versions, stats script, TTM pin, evaluation seed) | Reproducibility | done | done |
| 10 | Weather channels mix station and reanalysis data | Data | P2 | medium |
| 11 | Analyser ceilings (1000 µg/m³): measured, negligible for test metrics | Data | P3 | document only |
| 12 | ✅ Fixed: TTM now also scored on 15-min points (39.16 at 24 h) | Methodology | done | done |
| 13 | Slow pure-Python loops in pipeline and baselines | Compute | P2 | small |
| 14 | ERA5 download is sequential with a blind retry loop | Workflow | P3 | small |
| 15 | Presentation builder depends on files outside the repo | Workflow | P3 | small |

---

## 1. Compute bottlenecks

### 1.1 No GPU: the model has never been trained (P0)

- **Evidence ✅ measured (11 Oct 2026, `python benchmark_gpu.py --batch 1 2 --iters 2`):** on this laptop (CPU only, 16 GB RAM), one training step of the full 23.79 M-parameter model at **batch size 1 takes 16.5 s**. The benchmark projects **458 hours for 100k steps** at batch 1, and the planned batch size is 8. A run at **batch size 2 crashed Python with a segmentation fault**; the cause was not investigated (memory exhaustion is a guess).
- **Impact:** Stages C–H of the roadmap (training, evaluation, ablations, competitor comparisons, paper) cannot start. There are **no AQFM accuracy numbers** yet.
- **Fix:**
  1. Get a GPU: Kaggle (free weekly GPU quota), Google Colab, or a university cluster. Free tiers cut sessions off, so save `last.pt` regularly (already saved every validation) and copy `runs/` to persistent storage.
  2. On the GPU, run `python benchmark_gpu.py` **first**. It prints seconds per step, peak VRAM and projected run time for batch sizes 2–16, so you can choose `--batch-size` and `--steps` from measurements.
  3. Run the 500-step sanity check before the long run: `python -m aqfm.train --small --steps 500 --val-every 100 --out runs/sanity`.
  4. If time is tight, train a medium configuration (e.g. `d_model=256`, 3 encoder blocks) first. A trained smaller model is worth more than an untrained large one.

### 1.2 Evaluation does 12× more work than it needs (P1)

- **Evidence 📖 `aqfm/evaluate.py`:** every test forecast builds a 12-station neighbourhood and decodes all 1,728 queries, but only the anchor station's queries are scored (`sel_q = torch.arange(Ct * H)`). With 362 origins × 37 stations = **13,394 windows ✅**, about 11/12 of the decoder output and sampling is thrown away. The imputation evaluation repeats the pattern and also loops over results in Python, one example at a time.
- **Impact:** evaluation is likely to take hours, and it is repeated for every seed (≥ 3) and every ablation (≥ 5).
- **Fix:** measure first. If evaluation on the GPU is fast enough, change nothing.
  - **Not valid (corrected 12 Oct 2026):** an earlier version of this document suggested decoding only the anchor station's queries. That would **change the forecasts**: the decoder's queries attend to each other (query self-attention), so removing the other 11 stations' queries changes the anchor's output compared with how the model was trained.
  - **Valid option (a protocol change, so it needs a decision):** score *every* station of each neighbourhood rather than only the anchor, so fewer neighbourhoods are needed per origin. This is in-distribution, because training scores all 12 stations, but it changes which neighbourhood forecasts each station, so it must be documented in the methods.

### 1.3 Every attention call carries a mask, which may block the fastest GPU kernel (P1)

- **Evidence 📖 `aqfm/layers.py: MaskedMHA.forward`:** a float `attn_mask` is built whenever `key_valid` is passed, and **every** call passes it, including the decoder's query self-attention, where all forecast queries are valid.
- **Expectation ❓ unverified:** PyTorch's FlashAttention backend does not accept an arbitrary additive mask, so these calls probably fall back to the memory-efficient or plain kernel. The decoder's query self-attention is the largest attention in the model: 1,728 × 1,728 ≈ 3 M scores per head, 8 heads, 4 layers, per example.
- **How to check on the GPU:** run one forward pass inside `with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION]):` (both names are importable from `torch.nn.attention` in the installed torch 2.10). With only Flash allowed, a call that cannot use it should fail with a "no available kernel" error instead of silently falling back.
- **Fix:**
  - Pass `key_valid=None` when every key is valid (always true for forecast-only batches) so the call can use Flash.
  - Use bf16 (already on by default on CUDA).
  - If the query self-attention dominates: restrict it to queries of the **same station** (8 slots × 18 leads = 144 per group). This keeps the PM10 ↔ PM-fraction interaction the design relies on, at about 1/12 of the cost. This changes the model, so treat it as an ablation.

### 1.4 Slow pure-Python loops (P2)

All of these run once per pipeline execution, so they slow iteration but do not block it. Measured full run times on this laptop (12 Oct 2026): classical baselines 66 s, XGBoost ~5 min, XGBoost + future ERA5 ~6 min, TTM ~5.5 min. These are suggestions only and **have not been applied**: the code is correct as it is.

| Location | Loop | Fix |
|---|---|---|
| `data_pipeline/preprocess_aqfm.py: fit_outage_distribution` | Python loop over 37 × 280,512 steps; **its result is not used**, because `aqfm/train.py` refits on the training split | Delete it, or reuse the vectorised `aqfm.masking.fit_block_lognormal`. Mark the field in the report as superseded. |
| `baselines/run_baselines_ttm.py: forward_fill_nans`, `fit_scaler` | Python triple loops over stations × channels × ~70k hours | `pandas.DataFrame.ffill()` and `np.nanmean` / `np.nanstd` |
| `baselines/run_baselines_classical.py: build_climatology_dict` | 37 × 24 × 12 boolean scans of the full training period | One `np.bincount` / `groupby` over (station, hour, month) |

### 1.5 Memory headroom (P2)

- **Evidence 📖 `aqfm/data.py: AQData`, `aqfm/physical.py: derive_inputs`:** the whole dataset is held in RAM after transformation: inputs float32 (37 × 20 × 280,512) ≈ 830 MB, mask uint8 ≈ 210 MB, ERA5 ≈ 330 MB. While it is being built, a float32 mask copy (≈ 830 MB) also exists. This fits in 16 GB and on typical cloud machines.
- **Watch out:** with `--workers > 0` on **Windows** (spawn start method), each DataLoader worker receives a pickled copy of the dataset. Use workers on Linux (Colab/Kaggle), or keep `--workers 0` on Windows. Data loading is not the bottleneck (~4 ms per sample, per `benchmark_gpu.py`'s note).

---

## 2. Data bottlenecks

### 2.1 Three stations sit at a fallback coordinate (P0)

- **Evidence ✅ measured:** `IHBAS Dilshad Garden`, `IMD Lodhi Road` and `NSIT Dwarka` all have exactly (28.6139, 77.209) in `data/stations.csv`. That is the default `generate_stations_csv.py` writes when Nominatim fails.
- **Impact:** the model's **spatial attention, wind-aware bias and neighbourhood selection** all depend on coordinates. These three stations look co-located (distance 0, so the bearing is undefined) and get the wrong neighbours. Their ERA5 weather is also interpolated at the wrong point (limited effect at ERA5's ~25 km resolution). The spatial-attention ablation, which is a headline claim, would be measured on partly wrong geometry.
- **Fix:** take official coordinates from CPCB's station list (or the CPCB / DPCC station pages), replace the three rows, then re-run `interpolate_era5_to_stations.py` → `preprocess_aqfm.py` → baselines. The baselines do not use coordinates directly; only XGBoost's weather features change, through the re-interpolated ERA5 fill, so their results should barely move. Check this rather than assume it. Also review whether `Lodhi Road (IITM)` and `IMD Lodhi Road (IITM)` are separate sites or the same one, and confirm the shared Pusa coordinate.

### 2.2 ✅ FIXED (11 Oct 2026): station-density feature was empty, a one-line bug

- **Evidence ✅ measured + 📖 `data_pipeline/preprocess_aqfm.py:488`:** the density loop iterated `for i in range(S)` but assigned `static[si, 3]`, where `si` was left over from the previous loop. In `static.npy`, column 3 was **0 for 36 stations and 4 for the last one**, and it was never standardised.
- **Fix applied:** `static[si, 3]` → `static[i, 3]`; preprocessing re-run. Only `static.npy` changed; every other output file is identical. Tests pass. If the coordinates are corrected later (§2.1), re-run preprocessing again, because density depends on them.

### 2.3 Weather channels mix two sources without telling the model (P2)

- **Evidence 📖 `merge_era5`:** where a station's own weather sensor is missing, the ERA5 value is written into the same channel with `observed = 1`. Native sensor coverage is only 57–69% for AT, RH, WS, WD, SR and BP ✅.
- **Impact:** a local reading (e.g. an urban heat island in AT) and a 25 km reanalysis value look identical to the model, which adds noise and a distribution shift.
- **Fix:** keep a "source" flag (1 = sensor, 0 = ERA5) as an extra mask channel, or drop the CPCB weather channels from the encoder and rely on `era5_met.npy` (always available and consistent). Compare the two as an ablation.

### 2.4 Analyser ceilings truncate extreme values (measured: negligible for the test metrics)

- **Evidence ✅ measured:** PM2.5 and PM10 top out at exactly 1000 µg/m³ (168 and 1,827 readings); NOx, NO, NO2, NH3 and Ozone at ~500; SO2 at 200.
- **Impact ✅ measured (11 Oct 2026):** in the 2025 test year only 7 of 88,081 severe PM2.5 readings (and 149 of 334,347 severe PM10 readings) are at the cap, and **none** of the 1,252 severe PM2.5 points scored at 24 h. The effect on reported metrics is negligible.
- **Fix:** none needed. One sentence in the paper's limitations is enough.

### 2.5 Unused data that could help (opportunity)

- **2026 CSVs (35 stations, partial year)** are on disk but unused. Once complete enough, they give a **second, fully untouched test period**, which directly addresses the single-test-year risk (§3.2).
- **2017** (36 stations) is excluded only because ERA5 alignment starts in Dec 2017. It could extend training by a year.

---

## 3. Methodology bottlenecks (what a reviewer will question)

### 3.1 AQFM sees future weather; the baselines do not (P0, fairness)

- **Evidence 📖:** AQFM's decoder receives **future ERA5 weather** for every forecast patch (`use_future_met=True` by default). XGBoost uses weather only at t − 1 and t − 96 (`baselines/run_baselines_xgboost.py`, `WEATHER_CHANNELS`), and TTM uses none at all.
- **Impact:** ERA5 is a **reanalysis**, meaning the *true* past weather, not a forecast. Any AQFM gain over XGBoost would mix "better model" with "perfect knowledge of tomorrow's weather". Reviewers will ask.
- **✅ Done (11 Oct 2026): XGBoost + future ERA5.** `python run_baselines_xgboost.py --future-met` adds the 8 ERA5 variables at the target time as features → `baselines/results/xgboost_future_met_results.json`. A plain run is unchanged.

  | PM2.5 MAE (µg/m³) | 1 h | 6 h | 24 h | 72 h | Severe band, 24 h |
  |---|---|---|---|---|---|
  | XGBoost | 18.17 | 30.06 | 38.63 | 43.74 | 110.5 |
  | **XGBoost + future ERA5** | **17.99** | **28.48** | **33.65** | **36.45** | **93.4** |

  Knowing the true future weather is worth ~5 µg/m³ at 24 h and ~7 at 72 h. **So if AQFM is run with future ERA5 (its default), the fair bar is 33.65, not 38.63.**
- **Still needed:**
  1. Always report the **oracle gap**: train and evaluate AQFM with `--no-future-met` (switch already built; roadmap item E1). Compare it with plain XGBoost (38.63), and AQFM with future met against XGBoost + future ERA5 (33.65).
  2. For an operational claim, the future weather should come from a real **forecast** product (e.g. ECMWF/GFS forecasts archived at issue time), not reanalysis.

### 3.2 One test year, and an abnormal training year (P1)

- **Evidence:** test = 2025 only. 2020 mean PM2.5 was 96.0 vs 112.3 in 2018 ✅, with COVID-19 lockdown emissions inside the training data.
- **Impact:** a single year can reflect that year's weather (drift) rather than skill. Lockdown data may teach patterns that do not recur.
- **Fix:** rolling-origin back-tests (e.g. train ≤ 2021 → test 2022; ≤ 2022 → 2023; ≤ 2023 → 2024), a 2026 test once available (§2.5), a train-without-2020 comparison, and ≥ 3 seeds with mean ± std.

### 3.3 ✅ DONE (11 Oct 2026): TTM is scored on an easier target

- **Evidence 📖 `baselines/run_baselines_ttm.py`:** TTM forecasts **hourly means** and was scored only against hourly-mean truth; every other model is scored on 15-minute points.
- **Fix applied:** the same TTM forecasts are now *also* scored at the 15-minute point every other baseline uses. `ttm_results.json` keeps the original entry `IBM TTM (zero-shot)` unchanged and adds `IBM TTM (zero-shot, 15-min scoring)`:

  | PM2.5 MAE (µg/m³) | 1 h | 6 h | 24 h | 72 h |
  |---|---|---|---|---|
  | TTM, hourly-mean scoring (original) | 20.23 | 32.43 | 37.70 | 41.32 |
  | **TTM, 15-min scoring (comparable with the others)** | **20.95** | **32.13** | **39.16** | **42.75** |

  On the comparable scoring, TTM (39.16) is slightly *worse* than XGBoost (38.63) at 24 h. The model weights are now pinned to the exact Hugging Face commit used (`TTM_REVISION`).

### 3.4 Missing stronger baselines (P1 for publication)

- XGBoost has no spatial features (neighbour stations' recent PM). A spatial XGBoost is cheap to build and is the natural "is attention needed?" control.
- The published competitors (PatchTST/iTransformer, AirFormer, AirPhyNet, E-STGCN, the last of which uses 37 Delhi stations) have not been run on this protocol yet (roadmap Stage F).

---

## 4. Workflow and reproducibility bottlenecks

### 4.1 Moving to a GPU machine is harder than it needs to be (P1)

- **Evidence 📖:**
  - `data_pipeline/preprocess_aqfm.py` hard-codes `ROOT = r"c:\Users\phala\Foundation Model"`; `generate_stations_csv.py` and `presentation/build_pptx_js.js` also use absolute Windows paths.
  - `observed.npy` is stored as **float32** (747 MB ✅) although it only contains 0 and 1.
  - Baseline scripts must be run from inside `baselines/` (paths are relative to the current directory).
- **Fix:**
  - Derive paths from the script location, e.g. `ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`, or accept `--root`.
  - Save `observed.npy` as `uint8`: **747 MB → ~187 MB**, so the upload drops from ~1.8 GB to ~1.3 GB. `aqfm` already converts the mask on load; check `baselines/` (they compare `> 0.5`, which works for uint8).
  - Use `os.path.dirname(__file__)` in `baseline_utils.py` so the baselines run from any directory.

### 4.2 ✅ FIXED (12 Oct 2026): reproducibility of every reported result

- **Problems found:** no pinned package versions; the data-report statistics were computed by a script outside the repository; the TTM weights were loaded as "latest" from Hugging Face; `aqfm/evaluate.py` sampled without a seed (two runs gave different results, verified); the documented ERA5 download command fetched only 2021–2025.
- **Fixes applied:**
  - `requirements.txt` with the exact versions used.
  - `docs/data_report_stats.py` recomputes every number in `DATA_REPORT.md` → `docs/data_report_stats.json`.
  - TTM pinned to commit `d6a79570…` (`TTM_REVISION`).
  - `python -m aqfm.evaluate --seed 0` (default 0, saved in the results); verified identical results for the same seed.
  - The full reproduction recipe, including the 2017-12 → 2020-12 ERA5 download, is in `PROJECT_DOCUMENTATION.md` §13.
- **Verified (12 Oct 2026, this laptop):** a fresh re-run reproduced all four baseline result files **byte-for-byte**; preprocessing reproduced identical arrays on every re-run; the statistics script reproduced every number in `DATA_REPORT.md`.
- **Remaining limits:** reproduction on a different machine or OS has not been tested. GPU training is not bit-exact in PyTorch, so report AQFM as mean ± std over ≥ 3 seeds. `data/stations.csv` (geocoded) must be kept as a fixed input.

### 4.3 Experiment tracking is minimal (P2)

- Training writes `runs/<name>/log.jsonl`, `config.json`, `best.pt` and `last.pt`, which is enough for one run. With 3 seeds × (main + 5–6 ablations), comparing runs by hand becomes error-prone.
- **Fix:** a small script that collects every `runs/*/config.json` + `results/*.json` into one table. TensorBoard or Weights & Biases is optional. Also add `runs/` and `results/` to `.gitignore`, since checkpoints are roughly 100 MB each.

### 4.4 ERA5 download (P3)

- **Evidence 📖 `data_pipeline/fetch_era5.py`:** one month per request, strictly sequential (~8 min per month on the CDS queue, per the 8 Oct 2026 log). The retry loop retries **any** exception up to 1,000 times with 60 s waits, so a permanent error (bad credentials, invalid request) hangs for up to ~16 hours instead of failing. The `__main__` loop covers only 2021–2025.
- **Fix:** retry only network and queue errors and fail fast on others. Submit a few months concurrently (❓ check the CDS's current per-user request limit first). Make the year range a command-line argument.

### 4.5 Presentation builder (P3)

- **Evidence 📖 `presentation/build_pptx_js.js`:** EDA images and two diagrams are loaded from `C:/Users/phala/.gemini/antigravity-ide/brain/...`, outside the repository. The TTM figure comes from the archived leaky experiment, and the slide numbers are stale.
- **Fix:** point `PLOTS` at `docs/figures/` (the same images; file sizes match), copy the two diagram JPGs into `presentation/assets/`, regenerate the correlation figure (its weather rows predate the ERA5 fix) and the TTM figure, and update every number from `baselines/results/*.json`.

---

## 5. Suggested order of work

1. **Done (11–12 Oct 2026):** density bug (§2.2), XGBoost + future ERA5 (§3.1), TTM 15-min scoring (§3.3), reproducibility fixes (§4.2).
2. **Before the final paper runs:** correct the three station coordinates (§2.1), then re-run interpolation, preprocessing, tests and baselines.
3. **On the GPU:** `benchmark_gpu.py` → sanity run → main → anneal → evaluate (`--seed`) → `--no-future-met` → ≥ 3 seeds → ablations.
4. **For the paper:** rolling back-tests or the 2026 test, COVID check, ceiling and coordinate limitations stated, competitors run on the same protocol.
