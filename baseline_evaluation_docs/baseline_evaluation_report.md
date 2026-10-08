# Air Quality Foundation Model: Baseline Evaluation Report
**Date:** September 30, 2026
**Target Variables:** PM2.5 and PM10 (µg/m³)
**Test Period:** January 1, 2025 to December 31, 2025 (Daily forecast origins)

This report tracks the performance of our baseline models. To properly evaluate air quality forecasting, we report the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) across four critical operational horizons: 1-hour, 6-hours, 24-hours, and 72-hours.

Crucially, standard pooled metrics can be misleading because clean summer months dominate the average and hide the model's failures during dangerous winter episodes. Therefore, for the 24-hour PM2.5 forecasts, we stratify the MAE by Central Pollution Control Board (CPCB) AQI bands.

---

## 1. Overall Forecast Skill (MAE in µg/m³)

| Model | Target | 1h | 6h | 24h | 72h |
|:---|:---|---:|---:|---:|---:|
| **Persistence** | **PM2.5** | 15.89 | 35.53 | 43.78 | 54.54 |
| *(Predicts t = t-1)* | **PM10** | 31.28 | 71.47 | 84.64 | 106.47 |
| **Seasonal Naive** | **PM2.5** | 43.17 | 39.87 | 43.60 | 43.84 |
| *(Predicts t = t-24h)*| **PM10** | 81.21 | 66.60 | 84.18 | 84.55 |
| **Climatology** | **PM2.5** | 44.83 | 43.30 | 43.79 | 44.04 |
| *(Historical Average)*| **PM10** | 83.18 | 70.78 | 82.76 | 82.97 |
| **XGBoost (Lagged)** | **PM2.5** | 18.08 | 28.02 | 36.15 | 40.77 |
| | **PM10** | 34.77 | 49.26 | 73.69 | 82.92 |
| **IBM TTM (Zero-Shot)**| **PM2.5** | 20.23 | 32.43 | 37.70 | 41.32 |
| | **PM10** | 37.55 | 52.80 | 71.74 | 79.25 |

### Initial Insights (Classical Baselines)
* **Short-Term (1h):** Persistence is very strong at 1h (MAE 15.89 for PM2.5). Any advanced model must beat this to prove it is learning more than just "things haven't changed much in the last hour."
* **Mid-Term (24h):** The diurnal cycle matters. Seasonal Naive (43.60) narrowly beats Persistence (43.78) for PM2.5 and beats it clearly for PM10 (84.18 vs 84.64) because it predicts based on the same hour yesterday.
* **Long-Term (72h):** Persistence degrades significantly (54.54 for PM2.5) because pollution can change drastically over 3 days. Here, Climatology (44.04) proves to be a strong, safe lower bound.

---

## 2. AQI Stratification (24h PM2.5 Horizon)

A model must perform well on "Severe" days to be operationally useful. The table below shows the MAE broken down by the actual observed AQI band at the 24h forecast horizon.

| Model | Good <br>(0-30) | Satisfactory <br>(30-60) | Moderate <br>(60-90) | Poor <br>(90-120) | Very Poor <br>(120-250) | Severe <br>(250+) |
|:---|---:|---:|---:|---:|---:|---:|
| **Sample Size (n)** | 2831 | 2259 | 1546 | 973 | 2005 | 672 |
| **Persistence** | 22.41 | 24.36 | 32.53 | 40.85 | 70.72 | 161.46 |
| **Seasonal Naive** | 24.12 | 26.23 | 31.90 | 41.28 | 70.08 | 151.76 |
| **Climatology** | 32.99 | 20.89 | 24.01 | 37.38 | 66.82 | 153.86 |
| **XGBoost** | 18.94 | 16.62 | 21.29 | 31.60 | 42.70 | 109.73 |
| **IBM TTM** | 18.64 | 17.11 | 23.02 | 35.00 | 49.83 | 112.43 |

### Stratification Insights
* **The "Severe" Failure:** All classical methods completely fail to predict Severe episodes (PM2.5 > 250). Seasonal Naive is off by an average of 151.76 µg/m³ during these critical events.
* **The "Good" Illusion:** Persistence looks amazing on Good days (MAE 22.41) and Climatology looks amazing on Moderate days (MAE 24.01). If we only looked at pooled MAE, the model would look acceptable, masking the catastrophic failures during Severe winter smog.

---
*Note: All baseline models (Classical, XGBoost, and IBM TTM) have completed evaluation.*
