# Air Quality Baseline Evaluation Report

**Regenerated:** 2026-10-08 — directly from `baselines/results/*.json`.
**Targets:** PM2.5 and PM10 (µg/m³) · **Test period:** 1 Jan – 31 Dec 2025, daily forecast origins.

> [!WARNING]
> This supersedes the earlier version of this report. The previous numbers were
> invalid for two reasons: (1) the XGBoost PM2.5 forecasts were clipped using the
> *observed future* PM10, which is data leakage, and (2) the meteorological inputs
> for 2018–2020 were fabricated by interpolating across a 3.75-year gap in the
> ERA5 download, and ERA5 was misaligned with CPCB by 5 h 30 min (UTC vs IST).
> Both are fixed. The old files are kept in `_archive_old/baseline_results_OLD_invalid/`.

## Protocol

Origins are every day at 00:00 IST in 2025. An origin `t` predicts index `t + h − 1`
using only information up to `t − 1`. Metrics are computed on raw µg/m³ at 15-minute
points, and only where the target was actually observed. Since pooled error is
dominated by clean summer months, PM2.5 at 24 h is also stratified by CPCB AQI band.

---

## 1. Forecast skill — MAE (µg/m³)

| Model | Target | 1h | 6h | 24h | 72h |
|:---|:---|---:|---:|---:|---:|
| **Persistence** *(predicts t = last observed)* | **PM2.5** | 15.89 | 35.53 | 43.78 | 54.54 |
| **Persistence** *(predicts t = last observed)* | **PM10** | 31.28 | 71.47 | 84.64 | 106.47 |
| **Seasonal Naive** *(predicts t = same hour yesterday)* | **PM2.5** | 43.17 | 39.87 | 43.60 | 43.84 |
| **Seasonal Naive** *(predicts t = same hour yesterday)* | **PM10** | 81.21 | 66.60 | 84.18 | 84.55 |
| **Climatology** *(station × hour × month mean)* | **PM2.5** | 44.83 | 43.30 | 43.79 | 44.04 |
| **Climatology** *(station × hour × month mean)* | **PM10** | 83.18 | 70.78 | 82.76 | 82.97 |
| **XGBoost** *(lagged features, no leakage)* | **PM2.5** | 18.17 | 30.06 | 38.63 | 43.74 |
| **XGBoost** *(lagged features, no leakage)* | **PM10** | 34.71 | 49.44 | 73.65 | 83.46 |
| **IBM TTM (zero-shot)** *(hourly means; channel-independent)* | **PM2.5** | 20.23 | 32.43 | 37.70 | 41.32 |
| **IBM TTM (zero-shot)** *(hourly means; channel-independent)* | **PM10** | 37.55 | 52.80 | 71.74 | 79.25 |

## 2. Forecast skill — RMSE (µg/m³)

| Model | Target | 1h | 6h | 24h | 72h |
|:---|:---|---:|---:|---:|---:|
| **Persistence** | **PM2.5** | 29.85 | 60.85 | 75.85 | 92.13 |
| **Persistence** | **PM10** | 52.39 | 106.03 | 124.50 | 150.84 |
| **Seasonal Naive** | **PM2.5** | 75.31 | 65.36 | 74.98 | 75.52 |
| **Seasonal Naive** | **PM10** | 120.76 | 100.64 | 122.72 | 123.57 |
| **Climatology** | **PM2.5** | 69.19 | 63.32 | 68.52 | 68.92 |
| **Climatology** | **PM10** | 111.93 | 96.10 | 110.71 | 110.97 |
| **XGBoost** | **PM2.5** | 35.26 | 48.67 | 66.00 | 72.64 |
| **XGBoost** | **PM10** | 58.46 | 75.39 | 105.49 | 116.23 |
| **IBM TTM (zero-shot)** | **PM2.5** | 37.07 | 52.61 | 64.23 | 69.47 |
| **IBM TTM (zero-shot)** | **PM10** | 58.63 | 76.76 | 102.88 | 111.73 |

## 3. PM2.5 at the 24 h horizon, MAE by AQI band

| Model | Good | Satisfactory | Moderate | Poor | Very Poor | Severe |
|:---|---:|---:|---:|---:|---:|---:|
| **Sample size (n)** | 1553 | 2640 | 1597 | 1002 | 2246 | 1248 |
| **Persistence** | 16.7 | 20.4 | 28.3 | 41.2 | 60.2 | 119.3 |
| **Seasonal Naive** | 17.1 | 20.2 | 28.4 | 41.2 | 60.3 | 117.4 |
| **Climatology** | 30.5 | 26.0 | 34.0 | 42.1 | 47.1 | 105.8 |
| **XGBoost** | 21.8 | 17.5 | 23.2 | 35.1 | 47.6 | 110.5 |
| **IBM TTM (zero-shot)** | 18.6 | 17.1 | 23.0 | 35.0 | 49.8 | 112.4 |

---

## 4. What these numbers say

**The bar to beat is XGBoost at 38.63 MAE (PM2.5, 24 h).** Before the leakage fix
it appeared to be 36.15; roughly 2.5 µg/m³ of that was the model being handed the
future PM10 it was supposed to be forecasting around.

**Persistence is very strong at 1 h** (15.89) and remains the floor any short-range
model must clear. By 72 h it collapses to 54.54, while Climatology (44.04) becomes
the honest long-horizon competitor — at that range, knowing the season beats knowing
the present.

**Every method fails in the Severe band.** With PM2.5 above 250 µg/m³, the best
result is Climatology at 105.8 MAE and XGBoost at 110.5. These are precisely the
episodes a forecast exists to warn about, and the pooled MAE hides the failure
completely: XGBoost looks like a 38.63 model overall and a 110-off model when it
matters. This gap is the main opening for AQFM.

**The IBM TTM baseline cannot use weather.** Verified directly: scrambling its
AT/RH/WS/WD inputs changes its PM2.5/PM10 forecast by exactly 0.0, because the
TTM backbone is channel-independent. It is therefore effectively a univariate
forecaster, and its numbers were unaffected by the meteorological corrections.
It is also scored on hourly means rather than 15-minute points, so it is not
strictly comparable to the rest of the table.

**Consistency check:** the three classical baselines are bit-identical to their
pre-fix values. That is expected and reassuring — they read only PM2.5/PM10
history, which was never corrupted; only the meteorological channels were.
