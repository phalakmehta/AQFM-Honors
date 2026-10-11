# Baseline Evaluation - Detailed Implementation Plan

> **Purpose**: Obtain accurate baseline results for all models listed in the AQFM presentation (Slide 19).
> **Status**: PLAN ONLY - no code executed yet.
> **Created**: 2026-09-30

---

## 1. What We Need to Produce

From the presentation's "Baselines We Must Beat" section, we need results for **6 baselines**:

| # | Baseline | Type | Status |
|---|----------|------|--------|
| 1 | **Persistence** | Classical | Not done |
| 2 | **Seasonal Naive** | Classical | Not done |
| 3 | **Climatology** | Classical | Not done |
| 4 | **XGBoost** | ML (gradient-boosted trees) | Not done |
| 5 | **IBM TTM** | Foundation model (zero-shot) | **Redo** - old results unreliable |
| 6 | **PatchTST / iTransformer** | Deep learning time-series | Not done |

### Metrics to Report
- **MAE** (Mean Absolute Error) in ug/m3 - primary metric
- **RMSE** (Root Mean Squared Error) in ug/m3 - secondary
- **Stratified by AQI band** for PM2.5 at 24h horizon
- Do **NOT** use MAPE (explodes when true ~ 0, as proven by old TTM run)

### Target Channels
- **PM2.5** (channel index 0 in preprocessed data)
- **PM10** (channel index 1 in preprocessed data)

### Forecast Horizons

| Label | Steps (15-min) | Hours |
|-------|----------------|-------|
| 1h    | 4              | 1     |
| 6h    | 24             | 6     |
| 24h   | 96             | 24    |
| 72h   | 288            | 72    |

---

## 2. Data Assets Available

### 2a. Preprocessed Arrays (PRIMARY - use for all baselines)

Location: `preprocessed/`

| File | Shape | Description |
|------|-------|-------------|
| `values.npy` | `(37, 17, 280512)` float32 | All measurements. Missing = 0.0 |
| `observed.npy` | `(37, 17, 280512)` float32 | Binary mask. 1 = real, 0 = missing |
| `timestamps_h.npy` | `(280512,)` float64 | Hours since 2018-01-01 00:00 |
| `coords.npy` | `(37, 2)` float32 | Lat/lon per station |
| `static.npy` | `(37, 4)` float32 | Static features per station |

**Channel order** (from `preprocessing_report.json`):

```
0: PM2.5    1: PM10     2: NO2     3: NOx     4: NO
5: Ozone    6: SO2      7: NH3     8: CO      9: AT
10: RH      11: WS      12: WD     13: SR     14: BP
15: TOT_RF  16: Benzene
```

**Time range**: 2018-01-01 00:00 to 2025-12-31 23:45 (280,512 steps at 15-min).
Step = 0.25 hours. Steps per day = 96. Steps per hour = 4.

### 2b. Raw CSVs

Location: `yearly_data/{year}/{station}_cleaned.csv`
Columns: `Station, Timestamp, PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, Benzene, ..., AT, RH, WS, WD, TOT-RF, SR, BP, ...`

### 2c. Stations

37 stations across Delhi NCR. Full list in `preprocessing_report.json` -> `station_order`.

---

## 3. Test/Train Split Convention

**ALL baselines MUST use the same split for fair comparison.**

- **Train**: 2018-01-01 to 2024-12-31 (7 years)
- **Test**: 2025-01-01 to 2025-12-31 (1 year)
- The exact index where 2025 starts:

```python
hours_to_2025 = (datetime(2025,1,1) - datetime(2018,1,1)).total_seconds() / 3600
test_start = np.searchsorted(timestamps_h, hours_to_2025)
# Expected: test_start ~ 245,568 (7 years x 365.25 x 96)
```

- **No validation split needed** for classical baselines (Persistence, Seasonal Naive, Climatology)
- For XGBoost: train on 2018-2024, no separate val (use built-in early stopping with 10% of train held out)
- For TTM: zero-shot, so no train split needed
- For PatchTST/iTransformer: train on 2018-2023, validate on 2024, test on 2025

---

## 4. Evaluation Protocol

For each baseline, at each horizon:

1. **Select evaluation origins**: Every day (every 96 steps) within the test period. Total origins ~ 365.
2. **At each origin `t`**:
   - The baseline produces a forecast for `t + horizon_steps`
   - Record the ground-truth value at `t + horizon_steps`
   - Record whether the ground truth is observed (`observed[s, ch, t+h] == 1`)
3. **Compute metrics only where ground truth is observed** (mask-aware)
4. **Average across all 37 stations** - every station contributes equally
5. For PM2.5 at 24h, also stratify by AQI band

### AQI Bands for PM2.5 (ug/m3)

| Band | Range |
|------|-------|
| Good | 0 - 30 |
| Satisfactory | 30 - 60 |
| Moderate | 60 - 90 |
| Poor | 90 - 120 |
| Very Poor | 120 - 250 |
| Severe | 250+ |

---

## 5. Baseline Implementations - Detailed Specs

### BASELINE 1: Persistence

**Logic**: Forecast at horizon `h` = last observed value before the forecast origin.

```
forecast(station s, channel c, origin t, horizon h) = values[s, c, t-1]
  (if observed[s,c,t-1] == 1, else walk back up to 7 days to find last observed)
```

**Properties**:
- Trivial to implement, instant to run
- Should be the floor - if any model is worse than this, it is broken
- Gets worse at longer horizons because it cannot capture diurnal cycle

**Implementation Steps**:
1. Load `values.npy` (mmap), `observed.npy` (mmap), `timestamps_h.npy`
2. Compute test_start index
3. For each origin (daily), for each station, for each target channel:
   - Find last observed value before origin (walk back, max 7 days)
   - Record prediction = that value
   - Record truth = value at origin + horizon
4. Compute MAE/RMSE only where truth is observed

**Estimated runtime**: < 5 minutes (pure numpy, no model)

---

### BASELINE 2: Seasonal Naive

**Logic**: Forecast = value at the same time-of-day, one day ago (24h = 96 steps back from the target time).

```
forecast(s, c, origin t, horizon h) = values[s, c, (t + h) - 96]
  (fallback: walk back in 24h increments up to 7 days if that timestep is missing)
```

**Properties**:
- Captures the diurnal cycle for free (strong for 24h horizon)
- At 1h horizon, this predicts "what happened 23h ago" - NOT ideal
- Correct interpretation: we predict the target time using the same hour yesterday

**Implementation Steps**:
Same structure as Persistence but prediction = value 96 steps before target time.

**Estimated runtime**: < 5 minutes

---

### BASELINE 3: Climatology

**Logic**: Forecast = historical average for this (station, channel, hour-of-day, month) computed from training data only.

```
clim[s, c, hour, month] = mean(values[s, c, t] where observed[s,c,t]==1
                                AND t is in training period
                                AND hour_of_day(t) == hour
                                AND month(t) == month)

forecast(s, c, origin t, horizon h) = clim[s, c, hour_of_day(t+h), month(t+h)]
```

**Properties**:
- Strong at very long horizons (72h) because it captures seasonal + diurnal climatology
- Weak at short horizons because it ignores recent conditions
- Independent of horizon (same prediction regardless of lead time)

**Implementation Steps**:
1. Convert `timestamps_h` to `(hour_of_day, month)` using:
   ```python
   base = np.datetime64("2018-01-01T00:00")
   datetimes = base + (timestamps_h * 3600).astype("timedelta64[s]")
   hours = pd.DatetimeIndex(datetimes).hour  # 0-23
   months = pd.DatetimeIndex(datetimes).month  # 1-12
   ```
2. Build lookup table `clim_mean[37, 2, 24, 12]` from training data
3. For each test origin, look up clim_mean for the target timestamp's hour and month

**Estimated runtime**: ~5 minutes (building the lookup is the slow part)

---

### BASELINE 4: XGBoost (Gradient-Boosted Trees)

**Logic**: Train an XGBoost regressor per target channel (PM2.5 and PM10) with lagged features. Separate model per horizon.

**Feature Vector** (for predicting station `s`, channel `c`, at origin `t`, horizon `h`):

| Feature Group | Features | Count |
|---------------|----------|-------|
| Target lags | `values[s, c, t-k]` for k in [1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h] (= steps [4,8,12,24,48,96,192,288]) | 8 |
| Weather at t-1 | `values[s, w, t-1]` for w in [AT, RH, WS, WD, BP] | 5 |
| Weather at t-96 | `values[s, w, t-96]` (24h ago) | 5 |
| Other pollutant | PM10 if target is PM2.5, vice versa, at t-1 | 1 |
| Time features | hour_of_day(t+h), day_of_week(t+h), month(t+h), day_of_year(t+h) | 4 |
| **Total** | | **23** |

- Only use training samples where the target at origin-1 AND target value are observed
- Set missing lag features to 0 (XGBoost handles this natively)

**XGBoost Hyperparameters**:
```python
xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    tree_method="hist",  # fast on CPU
    random_state=42,
    n_jobs=-1,
)
```

**Training Data Construction**:
- Use ALL stations pooled together (one model for all stations, not per-station)
- Subsample training period: use every 4th timestep to reduce training set size
- Expected training samples: ~37 stations x ~61,000 subsampled steps x ~80% observed ~ 1.8M samples
- This is large but `tree_method="hist"` handles it efficiently
- Total models to train: 2 channels x 4 horizons = **8 models**

**Estimated runtime**: ~30-60 minutes total

---

### BASELINE 5: IBM TTM (Zero-Shot) - REDO

#### Problems with the Old Implementation (`Zeroshot forecast/run_ttm.py`)

1. **Data processing mismatch**: The old script loads raw CSVs, resamples to 1h freq, applies forward-fill + interpolation, then `dropna()`. This creates a DIFFERENT dataset than the preprocessed arrays. We need consistency with other baselines.

2. **Random test sampling**: Line 175 uses `torch.randperm(len(dataset))[:1000]` - random 1000 samples from the test set, not systematic daily origins. Results are not reproducible (no seed set on randperm).

3. **Scaling mismatch**: Uses `StandardScaler` fitted across ALL stations pooled via `partial_fit`. The scaler inverse_transform on model output may not correctly reverse if the model's internal predictions are channel-independent.

4. **Only 6 input channels**: The old script uses only `[PM2.5, PM10, AT, RH, WS, WD]` - misses crucial features.

5. **No per-horizon evaluation**: Only evaluates at 96h (full forecast). Need 1h/6h/24h/72h breakdowns.

6. **Metric at aggregate level only**: Reports one MAE across all 96 timesteps pooled. Does not stratify by AQI band.

7. **The MAPE of 141,300% is a red flag**: Indicates near-zero true values in the test set from forward-fill contamination.

#### Correct TTM Implementation Plan

**Data Preparation**:
- Use the preprocessed `values.npy` and `observed.npy` as the canonical data source
- TTM expects input shape `(batch, context_length, num_channels)`
- Resample from 15-min to 1h (average 4 consecutive observed steps) to match TTM's expected resolution
  - After resampling: `(37, 17, 70128)` steps at 1h
  - Context = 512 hours ~ 21.3 days
  - Prediction = 96 hours = 4 days (for longest horizon)
  - The 1h/6h/24h/72h horizons map to steps 1/6/24/72 in the 1h-resampled data

**Missing Data Handling**:
- For resampled values: average only observed values in each 1h window
- Set resampled value to NaN if no observation exists in that hour
- For TTM input: forward-fill NaN gaps (TTM cannot handle NaN inputs)
- Track which 1h slots were genuinely observed vs imputed

**Model Setup** (confirmed working via API test):
```python
from tsfm_public import TinyTimeMixerForPrediction

model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2",
    context_length=512,     # 512 hours
    prediction_length=96,   # up to 96 hours
    num_input_channels=6,   # PM2.5, PM10, AT, RH, WS, WD
)
# Output field: outputs.prediction_outputs  (shape: batch, pred_len, channels)
```

**Installed packages**: `granite-tsfm 0.3.9`, `transformers 5.16.1`.
CUDA is NOT available - all inference runs on CPU.

**Evaluation Protocol**:
- Test period: 2025 (same as other baselines)
- Origins: every day (every 24 steps in 1h-resampled data)
- For each origin, get 512h context -> predict 96h future
- Extract predictions at steps [1, 6, 24, 72] for the 4 horizons
- Inverse-scale predictions back to ug/m3
- Compare against ground truth only where observed

**Scaling**:
- Fit StandardScaler on training period (2018-2024) per-channel
- Transform both context and ground-truth using same scaler
- Inverse-transform predictions before computing metrics

**Key Corrections from Old Script**:
- Use same data pipeline as other baselines (preprocessed arrays)
- Deterministic daily test origins (not random sampling)
- Per-horizon metrics (1h, 6h, 24h, 72h)
- AQI-stratified reporting for PM2.5
- Only compute MAE/RMSE where ground truth was genuinely observed

**Estimated runtime**: ~20-40 minutes on CPU

---

### BASELINE 6: PatchTST / iTransformer

#### Package Situation

- **PatchTST**: Available via HuggingFace `transformers` (already installed v5.16.1) as `PatchTSTForPrediction`
- **iTransformer**: NOT pre-installed. Options: `pip install neuralforecast` OR manual implementation

#### PatchTST Implementation
```python
from transformers import PatchTSTConfig, PatchTSTForPrediction

config = PatchTSTConfig(
    num_input_channels=6,
    context_length=512,
    prediction_length=96,
    patch_length=24,
    stride=24,
    d_model=128,
    num_attention_heads=4,
    num_hidden_layers=3,
    dropout=0.1,
)
model = PatchTSTForPrediction(config)
```

**Training Required**: Unlike TTM zero-shot, PatchTST and iTransformer need training.
- Train on 2018-2023, Validate on 2024, Test on 2025
- ~30-50 epochs with early stopping (patience=5)
- Use hourly-resampled data to reduce sequence length 4x

**iTransformer Key Idea**: Treats each variate (channel) as a token instead of each time patch. Standard transformer attention across variates.

**CPU Warning**: PatchTST/iTransformer training on CPU will be slow (hours). Consider:
1. Using smaller model configs (d_model=64, 2 layers)
2. Training on hourly data (not 15-min)
3. Fewer epochs with aggressive early stopping
4. Google Colab with GPU as fallback

---

## 6. Implementation Order

Execute in this order - each phase is independently runnable and produces results.

### Phase 1: Classical Baselines (Quick Wins)
- **Script**: `run_baselines_classical.py`
- **Implements**: Persistence, Seasonal Naive, Climatology
- **Runtime**: ~10 minutes
- **Dependencies**: numpy, pandas only

### Phase 2: XGBoost
- **Script**: `run_baselines_xgboost.py`
- **Implements**: XGBoost with lagged features
- **Runtime**: ~30-60 minutes
- **Dependencies**: numpy, pandas, xgboost, scikit-learn

### Phase 3: IBM TTM (Redo)
- **Script**: `run_baselines_ttm.py`
- **Implements**: Zero-shot TTM with correct evaluation protocol
- **Runtime**: ~20-40 minutes on CPU
- **Dependencies**: numpy, pandas, torch, granite-tsfm, transformers

### Phase 4: PatchTST / iTransformer
- **Script**: `run_baselines_dl.py`
- **Implements**: PatchTST (from HuggingFace), iTransformer
- **Runtime**: ~2-6 hours on CPU (training required)
- **Dependencies**: numpy, pandas, torch, transformers, (neuralforecast for iTransformer)

---

## 7. Output Format

Each phase saves results to `baseline_results/`:

```
baseline_results/
|-- classical_results.json
|-- xgboost_results.json
|-- ttm_results.json
|-- dl_results.json
|-- summary_table.md
```

### JSON Schema

```json
{
  "baseline_name": {
    "PM2.5": {
      "1h":  {"MAE": 12.3, "RMSE": 18.5, "n": 13505},
      "6h":  {"MAE": 15.1, "RMSE": 22.3, "n": 13505},
      "24h": {"MAE": 25.2, "RMSE": 38.1, "n": 13505},
      "72h": {"MAE": 35.8, "RMSE": 52.4, "n": 13505},
      "24h_aqi": {
        "Good":         {"MAE": 8.1,  "RMSE": 10.2, "n": 3200},
        "Satisfactory": {"MAE": 12.3, "RMSE": 15.1, "n": 2800},
        "Moderate":     {"MAE": 18.5, "RMSE": 22.3, "n": 2100},
        "Poor":         {"MAE": 25.2, "RMSE": 31.4, "n": 1800},
        "Very Poor":    {"MAE": 38.1, "RMSE": 48.5, "n": 2400},
        "Severe":       {"MAE": 65.3, "RMSE": 85.2, "n": 1205}
      }
    },
    "PM10": { "1h": {}, "6h": {}, "24h": {}, "72h": {} }
  }
}
```

(Numbers above are placeholder examples, NOT actual results.)

---

## 8. Shared Utility Code

All scripts should import from a common utility module `baseline_utils.py`:

```python
import numpy as np
import pandas as pd
from datetime import datetime

PREPROCESSED = "preprocessed"
CHANNEL_NAMES = ["PM2.5","PM10","NO2","NOx","NO","Ozone","SO2","NH3","CO",
                 "AT","RH","WS","WD","SR","BP","TOT_RF","Benzene"]
TARGET_CHANNELS = [0, 1]       # PM2.5, PM10
TARGET_NAMES = ["PM2.5", "PM10"]
HORIZONS = {"1h": 4, "6h": 24, "24h": 96, "72h": 288}
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 96
START_DATE = datetime(2018, 1, 1)

AQI_BANDS = {
    "Good": (0, 30), "Satisfactory": (30, 60), "Moderate": (60, 90),
    "Poor": (90, 120), "Very Poor": (120, 250), "Severe": (250, float("inf")),
}

def load_data():
    values = np.load(f"{PREPROCESSED}/values.npy", mmap_mode="r")
    observed = np.load(f"{PREPROCESSED}/observed.npy", mmap_mode="r")
    timestamps_h = np.load(f"{PREPROCESSED}/timestamps_h.npy")
    return values, observed, timestamps_h

def get_test_split(timestamps_h):
    hours_to_2025 = (datetime(2025,1,1) - START_DATE).total_seconds() / 3600
    test_start = int(np.searchsorted(timestamps_h, hours_to_2025))
    return test_start, len(timestamps_h)

def timestamps_to_dt_features(timestamps_h):
    base = np.datetime64("2018-01-01T00:00")
    dts = base + (timestamps_h * 3600).astype("timedelta64[s]")
    idx = pd.DatetimeIndex(dts)
    return {"hour": idx.hour.values, "month": idx.month.values,
            "dow": idx.dayofweek.values, "doy": idx.dayofyear.values}

def compute_metrics(y_true, y_pred, mask):
    valid = mask > 0.5
    if valid.sum() == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "n": 0}
    err = y_pred[valid] - y_true[valid]
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "n": int(valid.sum())}

def aqi_stratified(y_true, y_pred, mask):
    valid = mask > 0.5
    out = {}
    for name, (lo, hi) in AQI_BANDS.items():
        sel = valid & (y_true >= lo) & (y_true < hi)
        n = int(sel.sum())
        if n == 0:
            out[name] = {"MAE": float("nan"), "RMSE": float("nan"), "n": 0}
        else:
            e = y_pred[sel] - y_true[sel]
            out[name] = {"MAE": float(np.mean(np.abs(e))),
                         "RMSE": float(np.sqrt(np.mean(e**2))), "n": n}
    return out
```

---

## 9. Verification Checklist

After running each baseline, verify:

- [ ] Total test samples `n` is consistent across baselines (~37 x 365 x obs_rate ~ 12,000-13,500)
- [ ] Persistence MAE at 1h should be small (< 15 ug/m3 for PM2.5)
- [ ] Persistence MAE at 72h should be large (> 30 ug/m3)
- [ ] Seasonal Naive at 24h should be competitive (captures diurnal cycle)
- [ ] Climatology should be roughly constant across horizons (does not depend on lead time)
- [ ] XGBoost should beat all classical baselines at short horizons (1h, 6h)
- [ ] XGBoost at 72h might lose to Climatology (feature staleness)
- [ ] All MAE values should be positive and in physically reasonable range (5-100 ug/m3)
- [ ] AQI-stratified MAE should increase with band severity (Severe has highest absolute error)
- [ ] PM10 MAE > PM2.5 MAE at every horizon (PM10 has higher variance)

---

## 10. Known Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| CPU-only, no GPU | TTM is <1M params (CPU fine); PatchTST/iTransformer use small configs |
| Memory pressure (700MB .npy) | Use `mmap_mode="r"` for numpy loads; build features lazily |
| XGBoost training set too large | Subsample every 4th step (~1.8M samples, manageable) |
| TTM model download | Already cached from previous run |
| PatchTST training too slow on CPU | d_model=64, 2 layers; hourly data; fewer epochs |
| Missing data at test origins | Walk back to find last observed; skip if no recent data |

---

## 11. File Structure After Completion

```
Foundation Model/
|-- baseline_utils.py              # Shared utilities
|-- run_baselines_classical.py     # Phase 1: Persistence, Seasonal Naive, Climatology
|-- run_baselines_xgboost.py       # Phase 2: XGBoost
|-- run_baselines_ttm.py           # Phase 3: IBM TTM (corrected)
|-- run_baselines_dl.py            # Phase 4: PatchTST, iTransformer
|-- baseline_results/
|   |-- classical_results.json
|   |-- xgboost_results.json
|   |-- ttm_results.json
|   |-- dl_results.json
|   |-- summary_table.md
|-- run_baselines.py               # (TO DELETE - old combined script)
|-- Zeroshot forecast/
    |-- run_ttm.py                 # (OLD - superseded by run_baselines_ttm.py)
```
