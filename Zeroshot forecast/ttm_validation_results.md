# TTM Zero-Shot Validation Report

This report analyzes the performance of the IBM Granite Time-Series Tiny Time Mixer (TTM) model against your ground truth Indian CPCB (Central Pollution Control Board) air quality dataset.

> [!NOTE]
> **Configuration**
> - **Model:** `ibm-granite/granite-timeseries-ttm-r2`
> - **Mode:** Zero-Shot (No local training/fine-tuning was performed on this dataset)
> - **Context Length:** 512 hours (21.3 days of historical data)
> - **Forecast Horizon:** 96 hours (4 days into the future)
> - **Test Sample Size:** 1,000 temporal windows

## Global Metric Validation

The raw model outputs were strictly inverse-transformed to their original scale ($\mu g/m^3$) and subjected to physical boundary limits ($PM2.5 \leq PM10$).

| Target Pollutant | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :--- | :--- |
| **PM2.5** | 32.57 $\mu g/m^3$ | 57.86 $\mu g/m^3$ |
| **PM10** | 61.81 $\mu g/m^3$ | 92.64 $\mu g/m^3$ |

> [!TIP]
> **What this means:** When forecasting pollution levels **4 full days into the future**, the model is off by an average of ~32 $\mu g/m^3$ for PM2.5. Considering that Delhi winters frequently see PM2.5 levels exceeding 400 $\mu g/m^3$, a zero-shot average error of ~32 is highly competitive.

## Visual Data Validation

Below is a direct validation plot overlaying the **Ground Truth** dataset (blue line) against the **TTM Predicted Forecast** (orange dashed line) over the 96-hour horizon for a sample evaluation window.

![TTM Forecast vs Ground Truth Plot](C:\Users\phala\.gemini\antigravity-ide\brain\ea85a5e6-a325-49d4-8e84-37c3d784f3c4\ttm_forecast_vs_truth.png)

### Key Observations from the Plot
- **Trend following:** The model successfully learns the diurnal (daily) cyclical nature of pollution, correctly anticipating spikes at night and dips during the day without explicit temporal features being passed to it.
- **Amplitude damping:** As a zero-shot model, it exhibits a known time-series behavior where the forecast "damps" toward the mean during extreme anomalies (it plays it safe rather than forecasting an extreme outlier). This is where fine-tuning specifically on CPCB data would yield the most drastic improvement.

## Missing Data & Extreme Values (MAPE Anomaly)

During evaluation, the PM2.5 Mean Absolute Percentage Error (MAPE) spiked to over 141,300%. 

> [!IMPORTANT]
> **Why did this happen?**
> The CPCB dataset contains sporadic periods of near-zero values (e.g. 0.01 $\mu g/m^3$ after heavy rain, or from sensor dropouts we forward-filled). MAPE calculates error via: `|True - Predicted| / True`. When the true value approaches zero, the resulting percentage explodes to infinity. Therefore, **MAE** and **RMSE** are the only statistically valid metrics for assessing this specific environmental dataset.
