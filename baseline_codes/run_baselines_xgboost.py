"""
Baseline 4: XGBoost with lagged features.

Trains one XGBRegressor per (target_channel x horizon) = 8 models total.
All stations pooled into a single model. Training subsampled every 4 steps.
"""

import os
import time
import numpy as np
import xgboost as xgb

from baseline_utils import (
    load_data, get_test_split, timestamps_to_dt_features,
    compute_metrics, aqi_stratified, save_results, print_summary,
    TARGET_CHANNELS, TARGET_NAMES, HORIZONS,
    STEPS_PER_HOUR, STEPS_PER_DAY, SEED,
)

# Lag steps in 15-min resolution
LAG_STEPS = [4, 8, 12, 24, 48, 96, 192, 288]  # 1h,2h,3h,6h,12h,24h,48h,72h
MAX_LAG = max(LAG_STEPS)

# Weather channel indices: AT(9), RH(10), WS(11), WD(12), BP(14)
WEATHER_CHANNELS = [9, 10, 11, 12, 14]

TRAIN_SUBSAMPLE = 4  # use every 4th timestep for training


def build_features_vectorized(values, observed, time_feats, station_idx, ch_idx, t_indices, hz_steps):
    """Build feature matrix using FAST NumPy vectorization."""
    other_ch = 1 if ch_idx == 0 else 0
    n = len(t_indices)

    X = np.zeros((n, 23), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)

    target_indices = t_indices + hz_steps - 1
    out_of_bounds = target_indices >= values.shape[2]
    valid_mask[out_of_bounds] = False
    
    t_valid = t_indices[~out_of_bounds]
    if len(t_valid) == 0:
        return X, y, mask, valid_mask

    target_t = t_valid + hz_steps - 1
    
    # 1. Target lags
    for j, lag in enumerate(LAG_STEPS):
        idx = t_valid - lag
        v = idx >= 0
        idx_safe = np.maximum(0, idx)
        v_obs = observed[station_idx, ch_idx, idx_safe] > 0.5
        ok = v & v_obs
        X[~out_of_bounds, j] = np.where(ok, values[station_idx, ch_idx, idx_safe], 0.0)

    # 2. Weather at t-1
    idx = t_valid - 1
    v = idx >= 0
    idx_safe = np.maximum(0, idx)
    for j, wc in enumerate(WEATHER_CHANNELS):
        v_obs = observed[station_idx, wc, idx_safe] > 0.5
        ok = v & v_obs
        X[~out_of_bounds, 8 + j] = np.where(ok, values[station_idx, wc, idx_safe], 0.0)

    # 3. Weather at t-96 (24h ago)
    idx = t_valid - 96
    v = idx >= 0
    idx_safe = np.maximum(0, idx)
    for j, wc in enumerate(WEATHER_CHANNELS):
        v_obs = observed[station_idx, wc, idx_safe] > 0.5
        ok = v & v_obs
        X[~out_of_bounds, 13 + j] = np.where(ok, values[station_idx, wc, idx_safe], 0.0)

    # 4. Other pollutant at t-1
    idx = t_valid - 1
    v = idx >= 0
    idx_safe = np.maximum(0, idx)
    v_obs = observed[station_idx, other_ch, idx_safe] > 0.5
    ok = v & v_obs
    X[~out_of_bounds, 18] = np.where(ok, values[station_idx, other_ch, idx_safe], 0.0)

    # 5. Time features
    X[~out_of_bounds, 19] = time_feats["hour"][target_t]
    X[~out_of_bounds, 20] = time_feats["dow"][target_t]
    X[~out_of_bounds, 21] = time_feats["month"][target_t]
    X[~out_of_bounds, 22] = time_feats["doy"][target_t]

    y[~out_of_bounds] = values[station_idx, ch_idx, target_t]
    mask[~out_of_bounds] = observed[station_idx, ch_idx, target_t]

    # Must have the most recent lag to be valid
    idx0 = t_valid - LAG_STEPS[0]
    idx0_safe = np.maximum(0, idx0)
    invalid0 = (idx0 < 0) | (observed[station_idx, ch_idx, idx0_safe] < 0.5)
    
    new_valid = np.copy(valid_mask)
    new_valid[~out_of_bounds] = ~invalid0
    valid_mask = new_valid

    return X, y, mask, valid_mask


def main():
    values_mmap, observed_mmap, timestamps_h = load_data()
    # Keep as mmap. We vectorize lookups, so it will be fast even with mmap.
    values = values_mmap
    observed = observed_mmap
    
    test_start, test_end = get_test_split(timestamps_h)
    n_stations = values.shape[0]

    time_feats = timestamps_to_dt_features(timestamps_h)

    results = {"XGBoost": {}}
    max_hz = max(HORIZONS.values())

    test_origins = np.arange(test_start, test_end - max_hz, STEPS_PER_DAY)
    print(f"Test origins: {len(test_origins)} (daily in 2025)")
    os.makedirs("baseline_results", exist_ok=True)

    for ch_idx, ch_name in zip(TARGET_CHANNELS, TARGET_NAMES):
        results["XGBoost"][ch_name] = {}

        for hz_name, hz_steps in HORIZONS.items():
            print(f"\n{'='*60}")
            print(f"XGBoost: {ch_name} @ {hz_name} (horizon={hz_steps} steps)")
            print(f"{'='*60}")

            t0 = time.time()

            # Subsample training
            train_indices = np.arange(MAX_LAG, test_start - hz_steps, TRAIN_SUBSAMPLE)
            
            X_train_parts, y_train_parts = [], []

            for s in range(n_stations):
                X_s, y_s, mask_s, valid_s = build_features_vectorized(
                    values, observed, time_feats, s, ch_idx, train_indices, hz_steps
                )
                keep = (mask_s > 0.5) & valid_s
                if keep.sum() > 0:
                    X_train_parts.append(X_s[keep])
                    y_train_parts.append(y_s[keep])

            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
            print(f"  Training samples: {len(y_train)}")

            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                tree_method="hist",
                random_state=SEED,
                n_jobs=-1,
            )
            model.fit(X_train, y_train, verbose=False)
            print(f"  Trained in {time.time() - t0:.1f}s")

            # Test
            all_true, all_pred, all_mask = [], [], []
            for s in range(n_stations):
                valid_origins = test_origins[test_origins >= MAX_LAG]
                target_indices = valid_origins + hz_steps - 1
                valid_origins = valid_origins[target_indices < test_end]

                if len(valid_origins) == 0:
                    continue

                X_test, y_test, mask_test, _ = build_features_vectorized(
                    values, observed, time_feats, s, ch_idx, valid_origins, hz_steps
                )

                preds = model.predict(X_test)
                preds = np.maximum(preds, 0.0)

                all_true.append(y_test)
                all_pred.append(preds)
                all_mask.append(mask_test)

            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            mask = np.concatenate(all_mask)

            metrics = compute_metrics(y_true, y_pred, mask)
            results["XGBoost"][ch_name][hz_name] = metrics
            print(f"  Result: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")

            if ch_name == "PM2.5" and hz_name == "24h":
                aqi = aqi_stratified(y_true, y_pred, mask)
                results["XGBoost"][ch_name]["24h_aqi"] = aqi

    save_results(results, "xgboost_results.json")
    print_summary(results)


if __name__ == "__main__":
    main()
