# AQFM final setup: dropout sweep, 3 seeds, ensemble, no-future-met (2026-10-14)

**Final setup:** AQFM with the latest-value anchor, **dropout 0.3**, main phase only
(12k-step cosine schedule, warmup 2k, lr 3e-4, batch 8, weight decay 0.05), early stopping with
patience 3 on the 2024 validation score (nll + 0.5·crps, forecast part), best checkpoint kept.
Every choice below was made on **2024 validation only**; 2025 test numbers never selected anything.

**Code added for this (uncommitted, 37 tests pass):**
- `train.py --dropout P` (main phase only) and `--compile` (torch.compile with inductor
  `fallback_random=True`, so dropout draws match eager mode; 0.49 → 0.38 s/step; 300-step losses
  match eager to the 4th digit, the same as eager-vs-eager GPU nondeterminism).
- `evaluate.py --ckpt A B C`: equal-weight ensemble. Each model contributes ceil(200/3) = 67
  samples to one pooled predictive distribution; all metrics come from the pool. A single
  checkpoint gives bit-identical results to the previous code.

**Raw results:** `results/aqfm_do3_s{0,1,2}_results.json` (single seeds),
`results/aqfm_do3_seeds_stats.json` (mean, std with ddof=1, values),
`results/aqfm_do3_ens3_results.json` (ensemble), `results/aqfm_do3_nofm_s0_results.json`
(no future met). 2025 test: 362 origins, 13,394 windows, median of 200 samples, sampling seed 0.
**Runs:** `runs/main_s0_do2`, `runs/main_s0_do3`, `runs/main_s1_do3`, `runs/main_s2_do3`,
`runs/main_s0_do3_nofm`; chain script `runs/pipeline.sh`, log `runs/pipeline_dropout.log`.

## Validation (2024): selection score nll + 0.5·crps (lower is better)

| run | dropout | seed | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main_s0_anchor (2026-10-13) | 0.1 | 0 | 1.8191 | 1.7754 | 1.7618 | 1.7499 | **1.7299** | 1.7445 | 1.7453 | 1.7875 | 1.7722 | 1.7299 @ 5k |
| main_s0_do2 | 0.2 | 0 | 1.8186 | 1.7756 | 1.7614 | 1.7515 | **1.7306** | 1.7431 | 1.7472 | 1.7804 | – | 1.7306 @ 5k |
| main_s0_do3 | 0.3 | 0 | 1.8192 | 1.7747 | 1.7624 | 1.7495 | **1.7299** | 1.7425 | 1.7476 | 1.7718 | – | 1.7299 @ 5k |
| main_s1_do3 | 0.3 | 1 | 1.8200 | 1.7796 | 1.7657 | 1.7342 | **1.7290** | 1.7349 | 1.7412 | 1.7554 | – | 1.7290 @ 5k |
| main_s2_do3 | 0.3 | 2 | 1.8114 | 1.7673 | 1.7535 | 1.7384 | 1.7376 | **1.7255** | 1.7498 | 1.7454 | 1.7518 | 1.7255 @ 6k |
| main_s0_do3_nofm | 0.3 | 0 | 1.8665 | 1.8341 | 1.8158 | 1.8098 | 1.8004 | **1.7986** | 1.8001 | 1.8125 | 1.8051 | 1.7986 @ 6k |

- **Dropout makes no measurable difference.** At full precision the three seed-0 bests are
  1.7299366 (0.1), 1.7306133 (0.2) and 1.7299053 (0.3); 0.3 was picked by the pre-agreed
  lowest-score rule, by 0.00003. Together with run 2 (weight decay 0.1, 1.8229 vs 1.8185), this
  says the post-peak degradation is not a lack of regularisation.
- **Seed spread is ~20× the dropout effect:** at 4k steps the dropout-0.3 seeds score 1.7495,
  1.7342 and 1.7384. Final setup over 3 seeds: **1.7281 ± 0.0023**.
- The best step is not fixed: seeds 0 and 1 peak at 5k, seed 2 and no-future-met at 6k.

## 2025 test: PM2.5 MAE (µg/m³)

| model | 1 h | 6 h | 24 h | 72 h | Severe band, 24 h |
|---|---|---|---|---|---|
| **AQFM ensemble (3 seeds)** | **15.71** | **27.54** | **32.04** | 36.51 | **89.0** |
| AQFM single model, mean ± std (3 seeds) | 16.08 ± 0.17 | 28.13 ± 0.08 | 33.22 ± 0.76 | 37.70 ± 1.21 | 91.5 ± 0.6 |
| ↳ seed 0 / 1 / 2 | 16.25 / 16.09 / 15.91 | 28.06 / 28.11 / 28.22 | 33.97 / 32.46 / 33.24 | 38.74 / 36.37 / 38.00 | |
| AQFM dropout 0.1, seed 0 (2026-10-13) | 16.06 | 28.79 | 34.45 | 39.60 | 91.1 |
| XGBoost + future ERA5 | 17.99 | 28.48 | 33.65 | **36.45** | 93.4 |
| *AQFM no future met, seed 0* | *16.47* | *28.45* | *38.77* | *42.84* | *107.1* |
| *XGBoost (no future met)* | *18.17* | *30.06* | *38.63* | *43.74* | *110.5* |
| IBM TTM (zero-shot) | 20.23 | 32.43 | 37.70 | 41.32 | 112.4 |
| Persistence | 15.89 | 35.53 | 43.78 | 54.54 | 119.3 |
| Climatology | 44.83 | 43.30 | 43.79 | 44.04 | 105.8 |

PM2.5 RMSE: ensemble 34.11 / 46.42 / 55.32 / 63.08; single-model mean 34.57 / 47.13 / 57.04 / 64.64;
XGBoost + future ERA5 35.14 / 46.19 / 57.54 / 62.07; no-future-met 34.81 / 48.13 / 64.23 / 71.46;
XGBoost 35.26 / 48.67 / 66.00 / 72.64.

## 2025 test: PM10 MAE (µg/m³)

| model | 1 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|
| **AQFM ensemble (3 seeds)** | 28.88 | **45.50** | **58.78** | **64.93** |
| AQFM single model, mean ± std | 29.41 ± 0.34 | 46.54 ± 0.41 | 61.05 ± 1.59 | 67.51 ± 1.99 |
| XGBoost + future ERA5 | 34.24 | 47.22 | 62.07 | 67.16 |
| *AQFM no future met, seed 0* | ***28.72*** | *47.56* | *73.32* | *81.80* |
| *XGBoost (no future met)* | *34.71* | *49.44* | *73.65* | *83.46* |
| IBM TTM (zero-shot) | 37.55 | 52.80 | 71.74 | 79.25 |
| Persistence | 31.28 | 71.47 | 84.64 | 106.47 |

## PM2.5 at 24 h, MAE by AQI band

| model | Good | Satisfactory | Moderate | Poor | Very Poor | Severe |
|---|---|---|---|---|---|---|
| AQFM ensemble | **12.7** | **13.4** | **22.1** | **31.0** | 43.1 | **89.0** |
| AQFM single, mean ± std | 13.1 ± 1.4 | 14.0 ± 0.7 | 22.6 ± 0.9 | 31.6 ± 0.7 | 45.6 ± 2.3 | 91.5 ± 0.6 |
| XGBoost + future ERA5 | 17.8 | 15.2 | **22.1** | 32.6 | **41.7** | 93.4 |
| AQFM no future met, seed 0 | 16.9 | 17.8 | 24.8 | 36.2 | 51.7 | 107.1 |
| XGBoost | 21.8 | 17.5 | 23.2 | 35.1 | 47.6 | 110.5 |

## Probabilistic and other metrics

| PM2.5 | 1 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|
| CRPS, ensemble | 12.00 | 19.93 | 23.66 | 26.64 |
| CRPS, single mean ± std | 12.21 ± 0.11 | 20.37 ± 0.08 | 24.40 ± 0.62 | 27.47 ± 0.96 |
| CRPS, no future met | 12.50 | 20.81 | 28.05 | 31.45 |
| 90% coverage, ensemble / single mean / no-fm | 0.938 / 0.932 / 0.939 | 0.928 / 0.915 / 0.911 | 0.914 / 0.896 / 0.912 | 0.906 / 0.889 / 0.920 |
| 95% coverage, ensemble / single mean / no-fm | 0.968 / 0.963 / 0.968 | 0.961 / 0.955 / 0.952 | 0.951 / 0.939 / 0.956 | 0.949 / 0.937 / 0.958 |

- PM10 90% coverage at 24 h / 72 h: ensemble 0.897 / 0.889, single mean 0.871 / 0.863
  (single models are slightly over-confident for PM10 at long leads; pooling fixes most of it).
- **PM2.5 > 250 episodes** (precision / recall / F1):
  24 h: ensemble 0.736 / 0.734 / **0.735**; single 0.714 ± 0.029 / 0.729 ± 0.014 / 0.721 ± 0.011;
  no-fm 0.640 / 0.613 / 0.626. 48 h: ensemble F1 0.699; single 0.684 ± 0.010; no-fm 0.560.
- **24 h station-outage imputation, MAE:** PM2.5 ensemble 29.49, single 30.69 ± 1.52,
  no-fm 32.93, linear interpolation 36.29. PM10 ensemble 53.56, single 55.75 ± 2.90,
  no-fm 59.86, linear 66.66.

## Value of future weather (oracle gap)

PM2.5 MAE increase when future ERA5 is removed (same model, seed 0, dropout 0.3):

| | 1 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|
| AQFM: no-fm − with met | +0.22 | +0.39 | +4.80 | +4.10 |
| XGBoost: plain − future ERA5 | +0.18 | +1.58 | +4.98 | +7.30 |

## Reading

- **The 3-seed ensemble beats XGBoost + future ERA5 on PM2.5 at 1, 6 and 24 h** (32.04 vs 33.65
  at 24 h) and **ties at 72 h** (36.51 vs 36.45 MAE; RMSE 63.08 vs 62.07, slightly worse). It is
  also below persistence at 1 h (15.71 vs 15.89). **For PM10 it beats every baseline at every
  lead time** (at 1 h the no-future-met AQFM is marginally lower still: 28.72 vs 28.88).
- **Single models:** better than XGBoost + ERA5 at 1 h and 6 h; at 24 h the gap (33.22 vs 33.65)
  is smaller than the seed std (0.76), so it is a tie; at 72 h single models are behind
  (37.70 ± 1.21 vs 36.45).
- **Severe band (> 250):** ensemble 89.0 vs 93.4 for XGBoost + ERA5, the best of all models.
  XGBoost + ERA5 is better in the Very Poor band (41.7 vs 43.1).
- **Calibration:** the ensemble's intervals are close to nominal at every lead
  (90%: 0.906–0.938; 95%: 0.949–0.968).
- **Without future met**, AQFM ties plain XGBoost at 24 h (38.77 vs 38.63, one seed) and is
  better at 1, 6 and 72 h (42.84 vs 43.74) and in the Severe band (107.1 vs 110.5). XGBoost gains
  more from future ERA5 at 72 h (7.3 vs 4.1), which suggests AQFM uses long-lead weather less
  effectively than XGBoost does; a possible direction for improvement (one seed, so tentative).

## Caveats for the paper

- One test year (2025). Differences smaller than the seed std (PM2.5: 0.17 / 0.08 / 0.76 / 1.21)
  are not evidence.
- The ensemble is a single number (no spread); the no-future-met result is one seed.
- The ensemble pools 3 networks against one XGBoost model, so it is not a strictly like-for-like
  comparison. Report the single-model mean ± std alongside it, and say so.
- ERA5 is reanalysis, so every "with future met" result assumes a perfect weather forecast.
  The no-future-met rows are the operational lower bound.
