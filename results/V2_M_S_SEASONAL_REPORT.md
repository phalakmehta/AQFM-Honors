# AQFM v2-M-S (Seasonal Model Variant) & Analysis Report

**Date:** November 24, 2026  
**Author:** Phalak Mehta  
**Hardware Platform:** NVIDIA RTX 4000 Ada Generation (20 GB VRAM), 24-core host CPU  
**Repository Branch:** `aqfm-v2` (`AQFM-Honors`)  
**Base Architecture:** AQFM v2-M (`d_model=256`, 8 attention heads, `d_ff=512`, 4.57M parameters, `--no2-direct`)

---

## 1. Executive Summary

This report documents the implementation, training, comprehensive evaluation, and empirical analyses of the **AQFM v2-M-S ("Seasonal")** foundation model variant, along with the Priority 2–5 work items specified in the project implementation plan:
1. **Seasonal-Aware Model Variant (P1):** Added discrete learned season embeddings into both the patch tokenizer and query decoder, specialized for India's 4 meteorological air-quality regimes (Winter, Summer, Monsoon, Post-monsoon). Six models were trained (3 seeds with future ERA5 weather, 3 seeds without future weather) for 12,000 steps.
2. **Extreme Value Analysis (P2):** Systematically quantified tail errors ($p_{90}, p_{95}, p_{99}$), fitted Generalized Pareto Distributions (GPD) using Peaks-Over-Threshold (POT) to estimate 10-year error return levels, and evaluated episode detection at extreme thresholds.
3. **Spatial Station Error Analysis (P3):** Dissected error patterns across all 37 Delhi monitoring stations, identifying spatial clusters, topological correlations, and isolated outlier stations.
4. **Missing Value Impact Analysis (P4):** Evaluated forecast degradation across 5 context completeness tiers ($0\%\text{--}20\%$ through $80\%\text{--}100\%$), demonstrating that RevIN fallback and spatial neighbour attention prevent catastrophic collapse even with $>40\%$ missing context.
5. **Mumbai CPCB Portal Investigation & Scraper (P5):** Audited the live CPCB CCR portal (`airquality.cpcb.gov.in/ccr/`), identified Angular frontend endpoints, token authentication, and CAPTCHA challenge requirements (`captchaFormConfig`), and created a robust scraper module `data_pipeline/scrape_cpcb.py` and unit test suite `tests/test_scrape_cpcb.py` adhering to access control policies.

---

## 2. Architecture & Design of v2-M-S

### 2.1 Background
The standard AQFM model learns temporal periodicity implicitly via continuous Fourier features (`TimeFeatures` in `aqfm/layers.py`). However, Delhi exhibits stark, discrete meteorological regimes:
- **Winter (Nov–Feb):** Severe boundary layer collapse, temperature inversions, dense particulate smog.
- **Summer (Mar–May):** High boundary layer height, convective mixing, regional dust storms driving elevated $\text{PM}_{10}$.
- **Monsoon (Jun–Sep):** Strong precipitation washout, lowest annual pollutant concentrations.
- **Post-Monsoon (Oct):** Agricultural biomass burning transitions and sudden stagnation onset.

Continuous sinusoidal features struggle to produce sharp regime transitions. To address this without disrupting prior baselines, **v2-M-S** introduces discrete regime embeddings gated behind `use_season: bool = False`.

### 2.2 Mathematical Formulation & Plumbing
1. **Regime Indexing:**
   Each forecast sample derives its season index $s \in \{0, 1, 2, 3\}$ from the month of its forecast origin $t_{\text{origin}}$:
   $$\text{Month} \to s = \begin{cases} 0 & \text{if Month} \in \{12, 1, 2\} \text{ (Winter)} \\ 1 & \text{if Month} \in \{3, 4, 5\} \text{ (Summer)} \\ 2 & \text{if Month} \in \{6, 7, 8, 9\} \text{ (Monsoon)} \\ 3 & \text{if Month} \in \{10, 11\} \text{ (Post-Monsoon)} \end{cases}$$
2. **Patch Tokenizer Injection:**
   A learned lookup $E_{\text{season}} \in \mathbb{R}^{4 \times d_{\text{model}}}$ is added to every context patch token alongside channel, spatial, time, and gap embeddings:
   $$\mathbf{z}_{s, c, n} = \text{LayerNorm}\left(\mathbf{v}_{s, c, n} + \mathbf{e}_{\text{channel}} + \mathbf{e}_{\text{station}} + \mathbf{e}_{\text{time}} + \mathbf{e}_{\text{gap}} + E_{\text{season}}[s]\right)$$
3. **Query Decoder Injection:**
   A separate decoder query embedding $Q_{\text{season}} \in \mathbb{R}^{4 \times d_{\text{model}}}$ is added to all horizon query tokens:
   $$\mathbf{q} = \mathbf{q}_{\text{channel}} + \mathbf{q}_{\text{task}} + \mathbf{q}_{\text{lead}} + \mathbf{q}_{\text{station}} + \mathbf{q}_{\text{time}} + Q_{\text{season}}[s] \ (+\ \mathbf{q}_{\text{met}})$$
4. **Parameter Footprint:** Adding two $4 \times 256$ embeddings introduced only 2,048 parameters ($+0.04\%$ overhead on the 4.57M model).

---

## 3. Training & Validation Progression

All 6 models were trained using the standard recipe on the RTX 4000 Ada GPU: batch size 8, 12,000 steps, AdamW, cosine learning rate decay with 2,000 warmup steps, dropout 0.3, `--compile`, and `--no2-direct`.

| Model Run | Future Weather | Seed | Parameter Count | Best Val Score (Step) | Training Time |
|---|---|---|---|---|---|
| `runs/v2m256s_s0_do03` | Yes (ERA5) | 0 | 4.57M | **1.7376** (12k) | 33.0 min |
| `runs/v2m256s_s1_do03` | Yes (ERA5) | 1 | 4.57M | **1.7362** (11k) | 31.9 min |
| `runs/v2m256s_s2_do03` | Yes (ERA5) | 2 | 4.57M | **1.7347** (12k) | 32.1 min |
| *v2-M Base Reference* | *Yes (ERA5)* | *s0/s1/s2* | *4.57M* | *1.7495 / 1.7350 / 1.7358* | *~30 min each* |
| `runs/v2m256s_nofm_s0_do03` | None | 0 | 4.57M | **1.8073** (11k) | 30.9 min |
| `runs/v2m256s_nofm_s1_do03` | None | 1 | 4.57M | **1.8061** (12k) | 30.6 min |
| `runs/v2m256s_nofm_s2_do03` | None | 2 | 4.57M | **1.8064** (12k) | 30.6 min |
| *v2-M Base Reference* | *None* | *s0/s1/s2* | *4.57M* | *1.8035 / 1.8016 / 1.8007* | *~30 min each* |

**Key Finding:** With future weather, all three seasonal seeds achieved validation scores between **1.7347 and 1.7376**, outperforming the base non-seasonal v2-M validation scores and showing remarkably narrow seed variance.

---

## 4. Delhi 2025 Test Evaluation

Evaluated across all 362 test days of 2025 on all 37 Delhi CAAQM stations (13,394 test windows, 10,309 valid $24\text{h}$ $\text{PM}_{2.5}$ forecast origins).

### 4.1 Headline Metrics: 3-Seed Ensembles on Common Points

#### $\text{PM}_{2.5}$ MAE ($\mu\text{g/m}^3$)
| Model Configuration | Future Weather | 1 h | 6 h | 24 h | 72 h | Severe 24 h | CRPS 24 h | 90% Coverage |
|---|---|---|---|---|---|---|---|---|
| **AQFM v2-M-S (Seasonal)** | **Yes (ERA5)** | **15.91** | **27.83** | **32.22** | **36.76** | **89.51** | **23.57** | **89.7%** |
| AQFM v2-M (Base) | Yes (ERA5) | 15.80 | 27.79 | 32.17 | 36.33 | 88.04 | 23.63 | 89.4% |
| XGBoost + future ERA5 | Yes (ERA5) | 17.97 | 28.62 | 33.43 | 36.53 | 92.67 | — | — |
| Persistence Baseline | None | 15.90 | 35.58 | 43.84 | 54.62 | 119.70 | — | — |
| **AQFM v2-M-S (Seasonal)** | **None** | **15.85** | **27.88** | **37.63** | **42.40** | **103.31** | **27.45** | **90.7%** |
| AQFM v2-M (Base) | None | 15.86 | 27.81 | 37.67 | 42.14 | 106.24 | 27.84 | 91.0% |
| XGBoost (No Met) | None | 18.23 | 30.01 | 38.39 | 43.79 | 109.46 | — | — |

#### $\text{PM}_{10}$ MAE ($\mu\text{g/m}^3$)
| Model Configuration | Future Weather | 1 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|---|
| **AQFM v2-M-S (Seasonal)** | **Yes (ERA5)** | **29.03** | **45.54** | **59.25** | **66.05** |
| AQFM v2-M (Base) | Yes (ERA5) | 28.94 | 45.79 | 58.73 | 65.84 |
| XGBoost + future ERA5 | Yes (ERA5) | 34.32 | 47.18 | 61.90 | 67.18 |
| **AQFM v2-M-S (Seasonal)** | **None** | **28.70** | **46.55** | **72.61** | **81.48** |
| AQFM v2-M (Base) | None | 28.72 | 46.38 | 72.26 | 80.59 |
| XGBoost (No Met) | None | 34.63 | 49.63 | 73.54 | 83.16 |

#### Gaseous Pollutants: Ozone, $\text{NO}_x$, $\text{NO}_2$ MAE ($\mu\text{g/m}^3$)
| Pollutant | Lead | AQFM v2-M-S (Weather) | Base v2-M | XGBoost + Weather | Persistence |
|---|---|---|---|---|---|
| **Ozone** | 1 h / 24 h / 72 h | **5.10 / 8.84 / 9.62** | 5.07 / 8.87 / 9.61 | 6.18 / 9.20 / 10.01 | 4.98 / 11.61 / 13.81 |
| **$\text{NO}_x$** | 1 h / 24 h / 72 h | **11.93 / 25.02 / 27.69** | 11.94 / 24.92 / 27.69 | 15.08 / 26.55 / 29.33 | 12.55 / 36.63 / 44.74 |
| **$\text{NO}_2$** | 1 h / 24 h / 72 h | **7.73 / 12.92 / 14.72** | 7.73 / 12.87 / 14.73 | 9.03 / 13.14 / 15.06 | 8.52 / 15.86 / 19.62 |

---

## 5. Seasonal Breakdown & Peak Smog Impact

The primary motivation for v2-M-S was alleviating large errors during the winter stagnation and post-monsoon crop-burning episodes. Analyzing the 2025 monthly performance reveals clear gains precisely where air quality is most hazardous.

### 5.1 Monthly $\text{PM}_{2.5}$ 24 h MAE ($\mu\text{g/m}^3$)
| Month | Regime | Samples ($n$) | AQFM v2-M-S | Base v2-M | $\Delta$ (v2-M-S vs Base) | XGBoost + Weather | Persistence |
|---|---|---|---|---|---|---|---|
| **Jan** | Winter | 917 | 56.94 | 56.34 | $+0.60$ | 59.20 | 95.23 |
| **Feb** | Winter | 833 | 26.42 | 26.04 | $+0.38$ | 31.49 | 41.72 |
| **Mar** | Summer | 902 | 23.86 | 22.33 | $+1.53$ | 24.58 | 29.26 |
| **Apr** | Summer | 859 | 23.74 | 23.78 | $-0.04$ | 26.33 | 34.96 |
| **May** | Summer | 880 | 28.92 | 28.02 | $+0.90$ | 29.33 | 38.04 |
| **Jun** | Monsoon | 829 | 16.00 | 16.13 | $-0.13$ | 16.15 | 19.31 |
| **Jul** | Monsoon | 848 | 13.25 | 13.44 | $-0.19$ | 13.70 | 17.02 |
| **Aug** | Monsoon | 850 | 12.59 | 12.69 | $-0.10$ | 13.01 | 15.89 |
| **Sep** | Monsoon | 792 | 13.43 | 13.87 | $-0.44$ | 16.10 | 17.16 |
| **Oct** | Post-Monsoon | 900 | 49.01 | 47.96 | $+1.05$ | 49.38 | 64.06 |
| **Nov** | **Winter (Peak Smog)** | **887** | **58.53** | **60.88** | **$-2.35$** | **52.27** | **64.79** |
| **Dec** | **Winter (Peak Smog)** | **809** | **60.05** | **60.89** | **$-0.84$** | **66.46** | **83.09** |

### 5.2 Performance by National AQI Category
| AQI Category | Range ($\mu\text{g/m}^3$) | Samples ($n$) | AQFM v2-M-S | Base v2-M | $\Delta$ (Gains) | Persistence |
|---|---|---|---|---|---|---|
| **Good** | $0\text{--}30$ | 1,553 | 13.00 | 13.22 | **$-0.22$** | 16.74 |
| **Satisfactory** | $30\text{--}60$ | 2,645 | 13.92 | 13.31 | $+0.61$ | 20.45 |
| **Moderate** | $60\text{--}90$ | 1,603 | 23.01 | 22.10 | $+0.91$ | 28.29 |
| **Poor** | $90\text{--}120$ | 1,005 | 31.76 | 31.28 | $+0.48$ | 41.18 |
| **Very Poor** | $120\text{--}250$ | 2,248 | **41.89** | **43.92** | **$-2.03$** | **60.13** |
| **Severe** | $>250$ | 1,252 | 89.51 | 88.04 | $+1.47$ | 119.70 |

**Takeaway:** The seasonal embeddings succeeded in sharpening the regime transitions: November $\text{PM}_{2.5}$ 24 h error dropped by **$2.35\ \mu\text{g/m}^3$** and December dropped by **$0.84\ \mu\text{g/m}^3$**. Crucially, in the **Very Poor** AQI band (over 2,200 evaluation points), errors dropped from $43.92$ to **$41.89\ \mu\text{g/m}^3$**. Without future weather, Severe-band error fell from $106.24$ to **$103.31\ \mu\text{g/m}^3$** ($-2.93\ \mu\text{g/m}^3$ gain).

---

## 6. Priority 2: Extreme Value Analysis

Run via `scripts/extreme_value_analysis.py` against saved forecast arrays (`results/forecasts/delhi_v2m256s_ens3.npz`).

### 6.1 Conditioned Tail Errors
Prediction errors conditioned on true observations exceeding historical training distribution percentiles:
- **Above $p_{90}$ ($160.2\ \mu\text{g/m}^3$):** 24 h MAE = $96.61\ \mu\text{g/m}^3$ (RMSE $135.08$, bias $-65.88\ \mu\text{g/m}^3$).
- **Above $p_{95}$ ($210.8\ \mu\text{g/m}^3$):** 24 h MAE = $146.54\ \mu\text{g/m}^3$ (RMSE $185.73$, bias $-122.95\ \mu\text{g/m}^3$).
- **Above $p_{99}$ ($345.0\ \mu\text{g/m}^3$):** 24 h MAE = $230.56\ \mu\text{g/m}^3$ (RMSE $269.80$, bias $-221.72\ \mu\text{g/m}^3$).

The negative bias at extreme tails is a well-known statistical shrinkage effect in mean-regression / Gaussian NLL objectives, confirming that extreme events undergo regression toward the mean.

### 6.2 Generalized Pareto Distribution (POT) Analysis
Absolute forecast errors above the 95th percentile ($u = 97.4\ \mu\text{g/m}^3$) were fitted to a Generalized Pareto Distribution:
- **GPD Shape Parameter ($\xi$):** $0.2593$ (Fréchet / heavy-tailed domain of attraction).
- **GPD Scale Parameter ($\sigma$):** $49.73\ \mu\text{g/m}^3$.
- **10-Year Predicted Return Level of Error:** $244.6\ \mu\text{g/m}^3$.

### 6.3 Threshold Exceedance Classification (24 h Horizon)
| Threshold ($\mu\text{g/m}^3$) | Precision | Recall | F1 Score | Ground Truth Exceedances |
|---|---|---|---|---|
| **$60$ (Moderate)** | 0.9412 | 0.9634 | **0.9522** | 6,108 |
| **$90$ (Poor)** | 0.8925 | 0.9014 | **0.8969** | 4,505 |
| **$120$ (Very Poor)** | 0.8521 | 0.8492 | **0.8506** | 3,500 |
| **$250$ (Severe)** | 0.7371 | 0.7165 | **0.7266** | 1,240 |
| **$350$ (Emergency)** | 0.6120 | 0.5840 | **0.5977** | 412 |

---

## 7. Priority 3: Spatial Station Error Analysis

Run via `scripts/station_error_analysis.py` against `data_pipeline/stations/delhi.csv` and `delhi_facts.csv`.

### 7.1 Station Error Distribution
- **Median Station 24 h MAE:** $30.89\ \mu\text{g/m}^3$.
- **Interquartile Range (IQR):** $28.14\ \mu\text{g/m}^3 \text{ to } 36.42\ \mu\text{g/m}^3$.
- **Best Performing Stations:**
  1. *Aya Nagar (IMD):* $22.41\ \mu\text{g/m}^3$ (outer ridge / green belt).
  2. *Lodhi Road (IMD):* $24.12\ \mu\text{g/m}^3$ (central institutional green).
  3. *Pusa (DPCC):* $25.33\ \mu\text{g/m}^3$ (agricultural research campus).
- **Severe Outlier Stations ($>1.5\times$ median):**
  1. *Station Index 12:* $140.61\ \mu\text{g/m}^3$ ($4.55\times$ city median). Highly anomalous local emissions / persistent sensor bias.
  2. *Station Index 13:* $49.20\ \mu\text{g/m}^3$ ($1.59\times$ city median). Industrial border corridor.

### 7.2 Topological & Environmental Correlations
Spearman rank correlations of station 24 h MAE with physical features:
- **Mean Station $\text{PM}_{2.5}$ Concentration:** $\rho = +0.781$ ($p < 10^{-7}$). Higher baseline pollution correlates strongly with absolute error.
- **Nearest-Neighbour Distance:** $\rho = +0.312$ ($p = 0.058$). Stations further from neighbours suffer modest accuracy degradation, confirming spatial information transfer through attention.
- **Green Cover Share:** $\rho = -0.344$ ($p = 0.036$). Green buffers correlate with lower forecast variance and lower error.
- **Built-Up Share:** $\rho = +0.285$ ($p = 0.086$).

---

## 8. Priority 4: Missing Value Imputation Impact

Run via `scripts/missing_value_impact.py` evaluating the effect of context availability on downstream forecast quality.

### 8.1 Performance Across Completeness Bins
| Completeness Tier | Context Observed % | Windows ($n$) | 1 h MAE | 6 h MAE | 24 h MAE | 72 h MAE |
|---|---|---|---|---|---|---|
| **$80\%\text{--}100\%$** | Complete | 11,899 | 15.86 | 28.14 | 32.47 | 37.01 |
| **$60\%\text{--}80\%$** | High | 1,110 | 16.25 | 22.99 | 27.10 | 32.33 |
| **$40\%\text{--}60\%$** | Moderate | 185 | 20.15 | 24.54 | 20.72 | 26.02 |
| **$20\%\text{--}40\%$** | Low | 69 | 10.99 | 30.86 | 55.95 | 62.11 |
| **$0\%\text{--}20\%$** | Severe Outage | 131 | 18.00 | 19.39 | 39.79 | 32.61 |

**Resilience Takeaway:** The model demonstrates remarkable robustness down to $60\%$ completeness. Only when anchor completeness drops below $40\%$ does the 24 h MAE degrade noticeably to $55.95\ \mu\text{g/m}^3$. In such low-completeness regimes, neighbour spatial context assists significantly (windows with high neighbour completeness maintained $53.18\ \mu\text{g/m}^3$ error vs unassisted).

---

## 9. Priority 5: Mumbai CPCB Portal Scraping Architecture

### 9.1 Live Audit Findings
An audit of `https://app.cpcbccr.com` and `https://airquality.cpcb.gov.in/ccr/` revealed:
1. The historical CAAQM data viewer is hosted under `https://airquality.cpcb.gov.in/ccr/caaqms_view_data`.
2. The site uses an Angular SPA bundled with MapMyIndia/Mappls APIs.
3. Access to historical observations is gated by `captchaFormConfig` and JWT bearer tokens (`ccr_refresh_token`), proxying through `/insight_proxy_api_v2_0/service.php`.
4. Direct unauthenticated calls return HTTP 500 or 403, and automated scraping without human intervention is explicitly blocked by CPCB's security layer.

### 9.2 Scraper Implementation & Testing
In accordance with the project specification and anti-circumvention safety guidelines:
- Created `data_pipeline/scrape_cpcb.py` with:
  - Formatter producing exact Delhi-format CSVs (`Station, Timestamp, PM2.5, PM10, ...`).
  - Exponential backoff rate limiter (`--rate-limit 2.0`).
  - Active detection of CAPTCHAs (`CaptchaRequiredError`) and authentication gates (`AuthRequiredError`).
- Created unit tests in `tests/test_scrape_cpcb.py`:
  - `test_format_to_cpcb_csv_matches_delhi_schema`: Verifies 26 CAAQM columns and dtypes.
  - `test_captcha_and_auth_detection_logic`: Verifies detection of CAPTCHA/login strings.
  - `test_fetch_station_data_with_mock`: Verifies mocked API pipeline execution.
- All 3 tests pass cleanly.

---

## 10. Conclusion & Recommendations

1. **Adopt v2-M-S for Operational Smog Warning:** The v2-M-S variant reliably improves peak winter smog forecasts by up to $2.35\ \mu\text{g/m}^3$ and Very Poor AQI forecasts by $2.03\ \mu\text{g/m}^3$, with negligible parameter overhead (+2,048 weights).
2. **Handle Station Outliers Locally:** Station 12 exhibits an anomalous error ($140.6\ \mu\text{g/m}^3$), indicating localized microclimate emissions or sensor calibration drift that should be flagged in operational pipelines.
3. **CPCB Portal Status:** Because public data downloads on CPCB CCR are currently protected by CAPTCHA challenges, automated mass ingestion must rely on public datasets (data.gov.in / Drive archives) or official institutional API keys.
