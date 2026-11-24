# Missing Value Imputation Impact Analysis

### 1. Forecast Error vs Context Completeness

| Completeness Bin | Count | 1h MAE | 6h MAE | 24h MAE | 72h MAE |
|---|---|---|---|---|---|
| 0-20% | 131 | 18.001 | 19.392 | 39.789 | 32.608 |
| 20-40% | 69 | 10.996 | 30.862 | 55.949 | 62.113 |
| 40-60% | 185 | 20.153 | 24.541 | 20.72 | 26.018 |
| 60-80% | 1110 | 16.253 | 22.986 | 27.099 | 32.329 |
| 80-100% | 11899 | 15.859 | 28.139 | 32.473 | 37.008 |


### 2. Impact of Missing Specific Meteorological Channels (24h Lead)

| Channel | Low Obs MAE (<50%) | High Obs MAE (>=90%) | Error Increase (Δ MAE) |
|---|---|---|---|
| AT | None | 32.223 | **None** |
| RH | None | 32.223 | **None** |
| WS | None | 32.223 | **None** |
| BP | None | 32.223 | **None** |
| SR | None | 32.223 | **None** |


### 3. Spatial Neighbour Compensation for Local Sensor Outages

- Windows with low anchor completeness (<40%): **32**

| Scenario | 24h MAE | Sample Count |
|---|---|---|
| High Neighbour Completeness (>=75%) | 53.175 | 28 |
| Low Neighbour Completeness (<50%) | None | 0 |

Compensation benefit (MAE reduction): **None µg/m³**


### 4. Controlled Artificial Context Masking Experiment

| Masking Rate | Mean Error |
|---|---|
| 0%_masked | 120.718 |
| 20%_masked | 120.718 |
| 40%_masked | 120.718 |
| 60%_masked | 120.718 |
| 80%_masked | 120.718 |

