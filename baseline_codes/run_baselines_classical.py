"""
Phase 1: Classical Baselines (Persistence, Seasonal Naive, Climatology)
"""

import os
import time
import numpy as np

from baseline_utils import (
    load_data, get_test_split, timestamps_to_dt_features,
    compute_metrics, aqi_stratified, save_results, print_summary,
    TARGET_CHANNELS, TARGET_NAMES, HORIZONS,
    STEPS_PER_DAY,
)


def build_climatology_dict(values, observed, time_feats, train_end, n_stations, ch_idx):
    """
    Builds a lookup table for climatology from training data.
    Returns: clim[station, hour, month] = mean_value
    """
    print(f"  Building climatology lookup table for target channel...")
    clim = np.zeros((n_stations, 24, 13), dtype=np.float32)  # month is 1-12
    counts = np.zeros((n_stations, 24, 13), dtype=np.int32)

    # We only care about training period
    hours = time_feats["hour"][:train_end]
    months = time_feats["month"][:train_end]
    
    # Extract training values for this channel
    train_vals = values[:, ch_idx, :train_end]
    train_obs = observed[:, ch_idx, :train_end]

    for s in range(n_stations):
        for h in range(24):
            for m in range(1, 13):
                # Mask for this hour and month in training period
                mask_time = (hours == h) & (months == m)
                mask_obs = train_obs[s, :] > 0.5
                valid_mask = mask_time & mask_obs
                
                if valid_mask.sum() > 0:
                    clim[s, h, m] = train_vals[s, valid_mask].mean()
                else:
                    clim[s, h, m] = np.nan
    return clim


def main():
    values, observed, timestamps_h = load_data()
    test_start, test_end = get_test_split(timestamps_h)
    n_stations = values.shape[0]

    time_feats = timestamps_to_dt_features(timestamps_h)

    results = {
        "Persistence": {},
        "Seasonal Naive": {},
        "Climatology": {}
    }
    
    for ch_name in TARGET_NAMES:
        results["Persistence"][ch_name] = {}
        results["Seasonal Naive"][ch_name] = {}
        results["Climatology"][ch_name] = {}

    max_hz = max(HORIZONS.values())
    
    # Test origins: every day in 2025
    # Origin is the point BEFORE the forecast horizon. So origin t predicts t+h.
    test_origins = np.arange(test_start, test_end - max_hz, STEPS_PER_DAY)
    print(f"Test origins: {len(test_origins)} (daily in 2025)")
    
    max_walkback = 7 * STEPS_PER_DAY # 7 days max walkback for missing data

    for ch_idx, ch_name in zip(TARGET_CHANNELS, TARGET_NAMES):
        print(f"\n{'='*60}")
        print(f"Processing target: {ch_name}")
        print(f"{'='*60}")
        
        clim = build_climatology_dict(values, observed, time_feats, test_start, n_stations, ch_idx)

        for hz_name, hz_steps in HORIZONS.items():
            print(f"  Horizon: {hz_name} ({hz_steps} steps)")
            
            y_true_all = []
            y_pred_pers = []
            y_pred_snaive = []
            y_pred_clim = []
            mask_all = []
            
            t0 = time.time()
            
            for s in range(n_stations):
                y_true_s = []
                y_pred_pers_s = []
                y_pred_snaive_s = []
                y_pred_clim_s = []
                mask_s = []
                
                for t in test_origins:
                    target_t = t + hz_steps - 1 # 0-indexed offset
                    if target_t >= test_end:
                        continue
                        
                    true_val = values[s, ch_idx, target_t]
                    is_obs = observed[s, ch_idx, target_t]
                    
                    # --- Persistence ---
                    # forecast = last observed value before origin t
                    # To avoid looking too far back, we bound it to 7 days
                    pers_pred = np.nan
                    for walk in range(1, max_walkback + 1):
                        idx = t - walk
                        if idx >= 0 and observed[s, ch_idx, idx] > 0.5:
                            pers_pred = values[s, ch_idx, idx]
                            break
                    
                    # --- Seasonal Naive ---
                    # forecast = value 24h before target_t
                    # Target is target_t, so we look at target_t - 96
                    snaive_pred = np.nan
                    for walk in range(0, 8): # Walk back in 24h increments up to 7 times
                        idx = target_t - STEPS_PER_DAY * (walk + 1)
                        if idx >= 0 and observed[s, ch_idx, idx] > 0.5:
                            snaive_pred = values[s, ch_idx, idx]
                            break
                            
                    # --- Climatology ---
                    target_h = time_feats["hour"][target_t]
                    target_m = time_feats["month"][target_t]
                    clim_pred = clim[s, target_h, target_m]
                    if np.isnan(clim_pred):
                        # Fallback if no training data for this station/hour/month (rare)
                        clim_pred = np.nanmean(clim[s, :, target_m])
                        
                    # Only append if we have valid predictions (or if we want to filter later)
                    # We will append everything and mask out nans during metrics
                    y_true_s.append(true_val)
                    
                    # Default nans to 0 to avoid breaking metrics if mask doesn't catch them?
                    # Metrics will filter based on mask_all which checks if target was observed,
                    # but if the PREDICTION is nan, we should probably mask it out too.
                    y_pred_pers_s.append(pers_pred)
                    y_pred_snaive_s.append(snaive_pred)
                    y_pred_clim_s.append(clim_pred)
                    
                    # Valid only if target is observed AND we made a valid prediction
                    valid = (is_obs > 0.5) and not np.isnan(pers_pred) and not np.isnan(snaive_pred) and not np.isnan(clim_pred)
                    mask_s.append(1.0 if valid else 0.0)
                    
                y_true_all.extend(y_true_s)
                y_pred_pers.extend(y_pred_pers_s)
                y_pred_snaive.extend(y_pred_snaive_s)
                y_pred_clim.extend(y_pred_clim_s)
                mask_all.extend(mask_s)
                
            y_true_all = np.array(y_true_all, dtype=np.float32)
            y_pred_pers = np.array(y_pred_pers, dtype=np.float32)
            y_pred_snaive = np.array(y_pred_snaive, dtype=np.float32)
            y_pred_clim = np.array(y_pred_clim, dtype=np.float32)
            mask_all = np.array(mask_all, dtype=np.float32)
            
            # Compute metrics
            m_pers = compute_metrics(y_true_all, y_pred_pers, mask_all)
            m_snaive = compute_metrics(y_true_all, y_pred_snaive, mask_all)
            m_clim = compute_metrics(y_true_all, y_pred_clim, mask_all)
            
            results["Persistence"][ch_name][hz_name] = m_pers
            results["Seasonal Naive"][ch_name][hz_name] = m_snaive
            results["Climatology"][ch_name][hz_name] = m_clim
            
            if ch_name == "PM2.5" and hz_name == "24h":
                results["Persistence"][ch_name]["24h_aqi"] = aqi_stratified(y_true_all, y_pred_pers, mask_all)
                results["Seasonal Naive"][ch_name]["24h_aqi"] = aqi_stratified(y_true_all, y_pred_snaive, mask_all)
                results["Climatology"][ch_name]["24h_aqi"] = aqi_stratified(y_true_all, y_pred_clim, mask_all)

            print(f"    Persistence:    MAE={m_pers['MAE']:>6.2f}, RMSE={m_pers['RMSE']:>6.2f} (n={m_pers['n']})")
            print(f"    Seasonal Naive: MAE={m_snaive['MAE']:>6.2f}, RMSE={m_snaive['RMSE']:>6.2f} (n={m_snaive['n']})")
            print(f"    Climatology:    MAE={m_clim['MAE']:>6.2f}, RMSE={m_clim['RMSE']:>6.2f} (n={m_clim['n']})")
            print(f"    Computed in {time.time() - t0:.1f}s")
            
    save_results(results, "classical_results.json")
    print_summary(results)

if __name__ == "__main__":
    main()
