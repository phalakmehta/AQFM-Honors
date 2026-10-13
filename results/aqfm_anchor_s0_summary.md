# AQFM with the latest-value anchor: seed 0 results (2026-10-13)

**Change:** each decoder query gets its series' latest visible value (RevIN-normalised), the
age of that value and a has-value flag, both as a query feature and as a gated residual on the
mixture means (`use_last_obs`, ablation flag `--no-last-obs`). Before this, the Perceiver
pooler hid "now" from the decoder, and short leads lost badly to persistence.

**Run:** `runs/main_s0_anchor` (main phase, 12k steps, batch 8, wd 0.05, seed 0, 117.8 min).
**Selected checkpoint:** step 5000, chosen on 2024 validation only
(`runs/main_s0_anchor/best.pt`, identical to `best_step5000.pt`).
**Raw results:** `results/aqfm_anchor_s0_results.json` (2025 test, 362 origins, 13,394 windows,
median of 200 samples, sampling seed 0).

## Validation (2024): selection score nll + 0.5·crps (lower is better)

| step | 1000 | 2000 | 3000 | 4000 | **5000** | 6000 | 7000 | 8000 | 9000 | 10000 | 11000 | 12000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor | 1.8191 | 1.7754 | 1.7618 | 1.7499 | **1.7299** | 1.7445 | 1.7453 | 1.7875 | 1.7722 | 1.7850 | 1.7973 | 1.8039 |

Best previous: run 1 1.8185 (step 5000), run 2 (wd 0.1) 1.8221. The run still overfits after ~5k steps.

## Short leads, PM2.5 MAE on 2024 validation (weekly origins, anchor station)

| lead | run 1 | anchor, step 1000 | **anchor, step 5000** | persistence |
|---|---|---|---|---|
| 15 min | 37.2 | 12.91 | **10.76** | 10.35 |
| 1 h | 37.6 | 19.31 | **16.28** | 17.41 |
| 2 h | – | 23.73 | **20.79** | 23.44 |
| 4 h | – | 28.04 | **22.63** | 30.95 |
| 6 h | – | 29.59 | **24.82** | 34.10 |
| 24 h | – | 34.76 | **32.33** | 37.67 |
| 72 h | – | 42.70 | **40.93** | 50.40 |

corr(lead-0 median, last obs) = 0.991 (run 1: 0.83; truth: 0.961).

## 2025 test: MAE (µg/m³)

### PM2.5

| model | 1 h | 6 h | 24 h | 72 h | Severe band, 24 h |
|---|---|---|---|---|---|
| **AQFM + anchor** | **16.06** | 28.79 | 34.45 | 39.60 | **91.07** |
| AQFM run 1 | 33.61 | 33.75 | 35.85 | 38.83 | 100.80 |
| XGBoost + future ERA5 | 17.99 | **28.48** | **33.65** | **36.45** | 93.45 |
| XGBoost | 18.17 | 30.06 | 38.63 | 43.74 | 110.50 |
| IBM TTM (zero-shot) | 20.23 | 32.43 | 37.70 | 41.32 | 112.43 |
| Persistence | 15.89 | 35.53 | 43.78 | 54.54 | 119.25 |
| Seasonal naive | 43.17 | 39.87 | 43.60 | 43.84 | 117.41 |
| Climatology | 44.83 | 43.30 | 43.79 | 44.04 | 105.85 |

### PM10

| model | 1 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|
| **AQFM + anchor** | **29.37** | 48.25 | 63.56 | 70.99 |
| AQFM run 1 | 62.83 | 55.96 | 65.72 | 69.53 |
| XGBoost + future ERA5 | 34.24 | **47.22** | **62.07** | **67.16** |
| XGBoost | 34.71 | 49.44 | 73.65 | 83.46 |
| IBM TTM (zero-shot) | 37.55 | 52.80 | 71.74 | 79.25 |
| Persistence | 31.28 | 71.47 | 84.64 | 106.47 |

## 2025 test: probabilistic and other metrics (AQFM + anchor; run 1 in brackets)

| PM2.5 | 1 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|
| CRPS | 12.18 | 20.90 | 25.36 | 28.98 |
| 90% coverage | 0.928 | 0.914 | 0.907 (0.887) | 0.898 |
| 95% coverage | 0.963 | 0.954 | 0.949 (0.932) | 0.944 |

- PM10 coverage at 24 h: 0.883 (90%), 0.939 (95%).
- PM2.5 > 250 episodes at 24 h: precision 0.658 (0.685), recall 0.751 (0.632), F1 0.701 (0.657).
  At 48 h: F1 0.687 (0.621).
- 24 h station-outage imputation, MAE: PM2.5 31.13 vs linear 36.29 (run 1: 30.67);
  PM10 57.15 vs linear 66.66 (run 1: 57.14).

## Reading

- The anchor fixes short leads. PM2.5 is best at 1 h among the learned models (16.06 vs
  XGB + ERA5 17.99), with persistence still marginally ahead (15.89). PM10 is best of all at
  1 h (29.37 vs persistence 31.28).
- AQFM is best in the PM2.5 Severe band at 24 h (91.07 vs 93.45) and is well calibrated.
- It still trails XGB + ERA5 at 24 h and 72 h (PM2.5 34.45 vs 33.65 and 39.60 vs 36.45).
  This is consistent with the overfitting after ~5k steps.

## Caveats

- Single seed. The paper needs 3 seeds (mean ± std) for the final configuration.
- Future ERA5 met is used as perfect prognosis; the `--no-future-met` oracle-gap run is still to do.
- The 2025 evaluation ran concurrently with the last training steps on the same GPU. This affects
  speed only: the evaluated checkpoint is byte-identical to the final `best.pt`.
