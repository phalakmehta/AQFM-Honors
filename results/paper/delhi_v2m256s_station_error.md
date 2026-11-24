# Spatial Station Error Analysis

Target Pollutant: **PM2.5**

- City Median 24h MAE: **30.893 µg/m³**
- Outlier Threshold (>1.5×): **46.339 µg/m³**

### 1. High-Error / Outlier Stations (24h Lead)

| Station Index | Station ID | 24h MAE (µg/m³) | Elevation (m) | Built-up Share |
|---|---|---|---|---|
| 12 | Station 12 | 140.599 | 213.42523193359372 | 0.4352542209822502 |
| 13 | Station 13 | 49.182 | 213.16873168945312 | 0.5983413461538462 |


### 2. Correlation of 24h Error with Station Characteristics

| Characteristic | Spearman Rank Correlation (ρ) | p-value |
|---|---|---|
| elevation_m | 0.059 | 0.7443 |
| built_frac_1km | -0.2218 | 0.2148 |
| green_frac_1km | 0.0386 | 0.8311 |
| coast_km | -0.1643 | 0.3609 |


### 3. Error-Profile Station Clusters

| Cluster | Count | Mean 1h MAE | Mean 6h MAE | Mean 24h MAE | Mean 72h MAE |
|---|---|---|---|---|---|
| cluster_1 | 10 | 20.483 | 34.947 | 39.775 | 44.677 |
| cluster_2 | 22 | 14.326 | 25.852 | 27.776 | 31.982 |
| cluster_3 | 1 | 14.833 | 27.477 | 140.599 | 110.796 |


### 4. Station Performance Ranking (Top 5 Best vs Worst 24h MAE)

**Best Performing Stations:**

| Station Index | Station ID | 24h MAE (µg/m³) |
|---|---|---|
| 31 | Station 31 | 2.271 |
| 34 | Station 34 | 21.733 |
| 14 | Station 14 | 24.1 |
| 27 | Station 27 | 25.431 |
| 16 | Station 16 | 25.61 |

**Worst Performing Stations:**

| Station Index | Station ID | 24h MAE (µg/m³) |
|---|---|---|
| 12 | Station 12 | 140.599 |
| 13 | Station 13 | 49.182 |
| 35 | Station 35 | 42.375 |
| 18 | Station 18 | 41.016 |
| 2 | Station 2 | 40.002 |

