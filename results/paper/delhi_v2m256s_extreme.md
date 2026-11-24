# Extreme Value Analysis (delhi_v2m256s_ens3)

## Pollutant: PM2.5

### 1. Tail Errors by Lead Time

| Lead | Overall MAE | p90 MAE | p95 MAE | p99 MAE | Worst Lead Bias |
|---|---|---|---|---|---|
| 1h | 15.9066 | 50.5474 | 70.9817 | 132.2818 | -1.7494 |
| 6h | 27.8347 | 85.874 | 114.3699 | 204.4646 | -4.9194 |
| 24h | 32.2228 | 96.6056 | 123.2568 | 230.5627 | -2.7817 |
| 72h | 36.763 | 113.6282 | 145.3306 | 268.7058 | -1.8826 |


### 2. Peaks-Over-Threshold GPD Fits (Error Tails)

| Lead | Threshold u | Shape (ξ) | Scale (σ) | 100-event Return Err | 1000-event Return Err |
|---|---|---|---|---|---|
| 1h | 51.85 | 0.4858 | 29.1486 | 553.94 | 1712.3 |
| 6h | 93.53 | 0.1795 | 49.5072 | 448.1 | 770.73 |
| 24h | 107.59 | 0.2593 | 49.725 | 548.83 | 1065.92 |
| 72h | 126.69 | 0.2423 | 56.0466 | 601.33 | 1128.62 |


### 3. Exceedance Detection Performance (24h Lead)

| Threshold (µg/m³) | Precision | Recall | F1 Score | True Exceedances |
|---|---|---|---|---|
| 60.0 | 0.9013 | 0.8597 | 0.88 | 6108 |
| 90.0 | 0.9009 | 0.8639 | 0.882 | 4505 |
| 120.0 | 0.8876 | 0.88 | 0.8838 | 3500 |
| 250.0 | 0.7371 | 0.7165 | 0.7266 | 1252 |
| 350.0 | 0.5631 | 0.5074 | 0.5338 | 475 |


### 4. Seasonal Breakdown (24h Lead)

| Season | Overall MAE | p90 MAE | Mean True | Count |
|---|---|---|---|---|
| Winter | 47.9896 | 119.7092 | 202.74 | 2559 |
| Summer | 25.5101 | 67.672 | 77.13 | 2641 |
| Monsoon | 13.8434 | 34.3601 | 39.15 | 3322 |
| Post-Monsoon | 53.7325 | 133.3203 | 195.04 | 1787 |


### 5. Top 5 Worst-Error Windows (24h Lead)

| Date | Station | Season | AQI Band | True | Pred | Abs Error |
|---|---|---|---|---|---|---|
| 2025-02-13 00:00 | 1 | Winter | Severe | 933.87 | 49.02 | 884.85 |
| 2025-10-20 00:00 | 9 | Post-Monsoon | Severe | 990.75 | 142.0 | 848.75 |
| 2025-10-20 00:00 | 24 | Post-Monsoon | Severe | 995.0 | 167.95 | 827.05 |
| 2025-10-20 00:00 | 13 | Post-Monsoon | Severe | 950.0 | 271.03 | 678.97 |
| 2025-10-20 00:00 | 18 | Post-Monsoon | Severe | 857.0 | 181.37 | 675.63 |


## Pollutant: PM10

### 1. Tail Errors by Lead Time

| Lead | Overall MAE | p90 MAE | p95 MAE | p99 MAE | Worst Lead Bias |
|---|---|---|---|---|---|
| 1h | 29.0305 | 75.9102 | 96.5891 | 160.9813 | -5.5418 |
| 6h | 45.5372 | 113.7798 | 145.867 | 254.2205 | -2.9816 |
| 24h | 59.2458 | 129.9938 | 163.7506 | 260.7707 | -7.22 |
| 72h | 66.0548 | 150.8779 | 186.6515 | 277.3553 | -6.514 |


### 2. Peaks-Over-Threshold GPD Fits (Error Tails)

| Lead | Threshold u | Shape (ξ) | Scale (σ) | 100-event Return Err | 1000-event Return Err |
|---|---|---|---|---|---|
| 1h | 94.88 | 0.2644 | 46.0423 | 509.19 | 1002.5 |
| 6h | 140.0 | 0.2217 | 55.1613 | 581.89 | 1042.02 |
| 24h | 176.47 | 0.1175 | 63.8051 | 566.3 | 856.12 |
| 72h | 199.22 | 0.0958 | 68.0547 | 593.18 | 865.77 |


### 3. Exceedance Detection Performance (24h Lead)

| Threshold (µg/m³) | Precision | Recall | F1 Score | True Exceedances |
|---|---|---|---|---|
| 60.0 | 0.965 | 0.9594 | 0.9622 | 9892 |
| 90.0 | 0.9419 | 0.9299 | 0.9358 | 8938 |
| 120.0 | 0.9263 | 0.9105 | 0.9183 | 7873 |
| 250.0 | 0.8306 | 0.8131 | 0.8217 | 4136 |
| 350.0 | 0.757 | 0.7163 | 0.7361 | 2270 |


### 4. Seasonal Breakdown (24h Lead)

| Season | Overall MAE | p90 MAE | Mean True | Count |
|---|---|---|---|---|
| Winter | 75.0959 | 168.7472 | 339.18 | 2612 |
| Summer | 59.5393 | 124.5675 | 218.74 | 2734 |
| Monsoon | 37.1231 | 97.3955 | 121.92 | 3454 |
| Post-Monsoon | 77.8987 | 146.0565 | 343.15 | 1834 |


### 5. Top 5 Worst-Error Windows (24h Lead)

| Date | Station | Season | AQI Band | True | Pred | Abs Error |
|---|---|---|---|---|---|---|
| 2025-05-14 00:00 | 32 | Summer | Severe | 1000.0 | 192.86 | 807.14 |
| 2025-05-14 00:00 | 7 | Summer | Severe | 1000.0 | 214.43 | 785.57 |
| 2025-04-17 00:00 | 18 | Summer | Severe | 915.0 | 273.65 | 641.35 |
| 2025-10-20 00:00 | 11 | Post-Monsoon | Severe | 852.7 | 246.25 | 606.45 |
| 2025-01-18 00:00 | 9 | Winter | Severe | 890.2 | 311.43 | 578.77 |

