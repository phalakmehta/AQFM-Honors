# Project Handoff: Air Quality Foundation Model (AqFM) for Delhi

This document summarizes the current state of the Air Quality Foundation Model (AqFM) project. It outlines the heavy lifting that has already been completed in data engineering and baseline evaluation, and details the next steps required to build and train the core Foundation Model.

---

## 1. What Has Been Completed (The Foundation)

### Data Engineering & Pipeline (`preprocessed/`, `era5script/`)
* **Raw Data Collection:** 8 years of historical air quality data from Central Pollution Control Board (CPCB) stations in Delhi (2018-2025).
* **Meteorological Integration:** Downloaded ERA5 weather data and interpolated it specifically to the geographic coordinates of the individual CPCB stations.
* **Tensor Creation:** All data was cleaned, missing values tracked, and converted into massive, model-ready NumPy tensors (`values.npy`, `observed.npy`, `timestamps_h.npy` - ~1.5GB total).
* **Train/Test Split:** Firmly established 2018-2024 as the training set, and entirely isolated 2025 as the unseen test set to prevent data leakage.

### Rigorous Evaluation Framework (`baseline_codes/baseline_utils.py`)
* **Metrics:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
* **Operational Horizons:** Models are strictly tested at 1-hour, 6-hour, 24-hour, and 72-hour forecast horizons.
* **AQI Stratification:** Since averages hide failures during extreme pollution events, a custom metric was built to measure MAE specifically during different Air Quality Index (AQI) bands (Good -> Severe). 

### Baseline Benchmarks Established (`baseline_results/`)
* **Classical Baselines:** Evaluated *Persistence*, *Seasonal Naive*, and *Climatology*. Proved that they fail spectacularly during "Severe" winter smog episodes (off by >150 µg/m³).
* **Machine Learning Baseline (XGBoost):** Trained and evaluated XGBoost with lagged features. It performed excellently, setting the primary benchmark to beat (**PM2.5 24h MAE: 36.15**).
* **Zero-Shot Foundation Model (IBM TTM):** Evaluated IBM's pre-trained Time-Series Foundation Model (TTM). We fixed a previous data-leakage bug in the scaling, achieving an honest, zero-shot **PM2.5 24h MAE of 37.70**.

### Project Organization
* Initialized Git repository, created a rigorous `.gitignore` to prevent 100MB+ data files from breaking GitHub pushes.
* Pushed all scripts, baseline code, and evaluation reports to the GitHub repository.

---

## 2. What Needs To Be Done Next (The Core Model)

The baseline benchmarks are locked in. The data is ready. The next phase is building, training, and testing the actual custom Air Quality Foundation Model.

### Phase 1: Review and Finalize the AqFM Architecture (`files_AqFM/`)
* The raw architecture files for the custom model currently exist in `files_AqFM/` (including `model.py`, `layers.py`, `train.py`, `config.py`, etc.).
* **Action:** Review this architecture. Ensure the Transformer blocks, the Query-Based Decoder, and the spatial/temporal embeddings are correctly structured to accept the `(n_stations, n_channels, context_length)` tensors we generated.

### Phase 2: Implement "Mixed Masking" Training Strategy
* **Action:** The Foundation Model needs to be trained on the 2018-2024 data. Crucially, the training loop must implement the "Mixed Masking" strategy (randomly masking out sensors/timesteps in the input) so the model learns to impute missing data and forecast the future simultaneously.

### Phase 3: Train the Model
* **Action:** Run the training loop (`files_AqFM/train.py`), monitoring validation loss. We need to ensure it doesn't overfit and that it learns the complex diurnal and seasonal pollution cycles.

### Phase 4: Evaluate the AqFM against Baselines
* **Action:** Once trained, evaluate the custom AqFM on the 2025 unseen test set using the exact same metrics found in `baseline_codes/baseline_utils.py` (specifically looking at the 24h horizon and the "Severe" AQI band).
* **Goal:** Beat the XGBoost score of 36.15 MAE at the 24-hour horizon.

### Phase 5: Visualization & Final Output
* **Action:** Generate final comparison plots (AqFM vs XGBoost vs Ground Truth).
* **Action:** Update the presentation (`build_pptx_js.js`) and LaTeX documents (`aqfm_design.tex`) with the final, winning results.
