"""
Baseline 5: IBM Granite TTM Zero-Shot (CORRECTED)

Fixes from old Zeroshot forecast/run_ttm.py:
- Uses preprocessed .npy arrays (same data as all other baselines)
- Deterministic daily test origins (not random sampling)
- Per-horizon evaluation (1h, 6h, 24h, 72h)
- AQI-stratified PM2.5 metrics
- Proper resampling from 15-min to 1h for TTM input
- Correct StandardScaler usage (fit on train only, per-station)
"""

import os
import sys
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from baseline_utils import (
    load_data, get_test_split, compute_metrics, aqi_stratified,
    save_results, print_summary, TARGET_CHANNELS, TARGET_NAMES,
    STEPS_PER_HOUR, STEPS_PER_DAY, RESULTS_DIR,
)

TTM_MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
# Exact Hugging Face commit of the model weights used for the reported results
TTM_REVISION = "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f"
TTM_CONTEXT = 512   # hours
TTM_PRED = 96       # hours

# Channels to feed TTM: PM2.5(0), PM10(1), AT(9), RH(10), WS(11), WD(12)
TTM_CHANNEL_INDICES = [0, 1, 9, 10, 11, 12]
TTM_CHANNEL_NAMES = ["PM2.5", "PM10", "AT", "RH", "WS", "WD"]
TTM_NUM_CHANNELS = len(TTM_CHANNEL_INDICES)

# Horizons in 1h-resampled steps
TTM_HORIZONS = {"1h": 1, "6h": 6, "24h": 24, "72h": 72}


def resample_to_hourly(values, observed, n_stations, channels):
    """Resample 15-min data to 1h by averaging observed values in each hour.

    Returns:
        hourly_vals: (n_stations, n_ch, n_hours) float32
        hourly_obs:  (n_stations, n_ch, n_hours) float32 binary mask
    """
    n_ch = len(channels)
    n_steps = values.shape[2]
    n_hours = n_steps // STEPS_PER_HOUR

    print(f"  Resampling {n_steps} steps -> {n_hours} hourly steps for {n_stations} stations x {n_ch} channels...")

    hourly_vals = np.zeros((n_stations, n_ch, n_hours), dtype=np.float32)
    hourly_obs = np.zeros((n_stations, n_ch, n_hours), dtype=np.float32)

    for s in range(n_stations):
        for ci, ch in enumerate(channels):
            v = np.array(values[s, ch, :n_hours * STEPS_PER_HOUR], dtype=np.float32)
            o = np.array(observed[s, ch, :n_hours * STEPS_PER_HOUR], dtype=np.float32)

            v_2d = v.reshape(n_hours, STEPS_PER_HOUR)
            o_2d = o.reshape(n_hours, STEPS_PER_HOUR)

            obs_count = o_2d.sum(axis=1)
            obs_sum = (v_2d * o_2d).sum(axis=1)

            valid_hours = obs_count > 0
            hourly_vals[s, ci, valid_hours] = obs_sum[valid_hours] / obs_count[valid_hours]
            hourly_obs[s, ci, valid_hours] = 1.0

    obs_rate = hourly_obs.mean()
    print(f"  Hourly observation rate: {obs_rate:.3f}")
    return hourly_vals, hourly_obs


def forward_fill_nans(data, obs):
    """Forward-fill gaps for TTM input. TTM cannot handle NaN/zero gaps.

    Modifies data in-place. Only fills where obs == 0.
    Returns the filled array (same object).
    """
    n_stations, n_ch, n_steps = data.shape
    filled_count = 0
    for s in range(n_stations):
        for c in range(n_ch):
            last_valid = 0.0
            found_first = False
            for t in range(n_steps):
                if obs[s, c, t] > 0.5:
                    last_valid = data[s, c, t]
                    found_first = True
                elif found_first:
                    data[s, c, t] = last_valid
                    filled_count += 1
    print(f"  Forward-filled {filled_count} hourly gaps")
    return data


def fit_scaler(hourly_vals, hourly_obs, train_end_h, n_stations, n_ch):
    """Fit StandardScaler on training data (observed values only), per channel.

    Returns a fitted StandardScaler with shape (n_ch,) for mean/std.
    """
    scaler = StandardScaler()

    # Collect all observed training values per channel
    all_train = []
    for s in range(n_stations):
        for t in range(train_end_h):
            row = []
            for c in range(n_ch):
                if hourly_obs[s, c, t] > 0.5:
                    row.append(hourly_vals[s, c, t])
                else:
                    row.append(np.nan)
            all_train.append(row)

    all_train = np.array(all_train, dtype=np.float64)
    # Use nanmean/nanstd to fit, then set scaler attributes
    scaler.mean_ = np.nanmean(all_train, axis=0)
    scaler.scale_ = np.nanstd(all_train, axis=0)
    scaler.scale_[scaler.scale_ < 1e-8] = 1.0  # prevent div by zero
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = n_ch
    scaler.n_samples_seen_ = np.sum(~np.isnan(all_train), axis=0).astype(int)

    print(f"  Scaler fitted on training data:")
    for i, name in enumerate(TTM_CHANNEL_NAMES):
        print(f"    {name}: mean={scaler.mean_[i]:.2f}, std={scaler.scale_[i]:.2f}")
    return scaler


def scale_array(data, scaler):
    """Scale (n_stations, n_ch, n_steps) using per-channel mean/std."""
    mean = scaler.mean_[np.newaxis, :, np.newaxis]   # (1, C, 1)
    std = scaler.scale_[np.newaxis, :, np.newaxis]
    return (data - mean) / std


def unscale_array(data, scaler):
    """Inverse scale."""
    mean = scaler.mean_[np.newaxis, :, np.newaxis]
    std = scaler.scale_[np.newaxis, :, np.newaxis]
    return data * std + mean


def main():
    # 1. Load raw 15-min data
    values, observed, timestamps_h = load_data()
    test_start_15m, test_end_15m = get_test_split(timestamps_h)
    n_stations = values.shape[0]

    # 2. Resample to hourly
    hourly_vals, hourly_obs = resample_to_hourly(
        values, observed, n_stations, TTM_CHANNEL_INDICES
    )
    n_hours = hourly_vals.shape[2]

    # Compute hourly train/test boundary
    train_end_h = test_start_15m // STEPS_PER_HOUR
    test_end_h = n_hours
    print(f"  Hourly split: train 0..{train_end_h}, test {train_end_h}..{test_end_h}")

    # 3. Forward-fill gaps (TTM needs contiguous input)
    hourly_filled = hourly_vals.copy()
    forward_fill_nans(hourly_filled, hourly_obs)

    # 4. Fit scaler on training data only
    scaler = fit_scaler(hourly_vals, hourly_obs, train_end_h, n_stations, TTM_NUM_CHANNELS)

    # 5. Scale the filled data
    hourly_scaled = scale_array(hourly_filled, scaler)

    # 6. Load TTM model
    print(f"\nLoading TTM model: {TTM_MODEL_ID}...")
    from tsfm_public import TinyTimeMixerForPrediction

    model = TinyTimeMixerForPrediction.from_pretrained(
        TTM_MODEL_ID,
        revision=TTM_REVISION,
        context_length=TTM_CONTEXT,
        prediction_length=TTM_PRED,
        num_input_channels=TTM_NUM_CHANNELS,
    )
    model.eval()
    print(f"  Model loaded. Config: context={model.config.context_length}, "
          f"pred={model.config.prediction_length}, channels={model.config.num_input_channels}")

    # 7. Evaluate: deterministic daily origins in test period
    max_horizon_h = max(TTM_HORIZONS.values())
    origins = list(range(train_end_h, test_end_h - max_horizon_h, 24))  # every 24h = daily
    print(f"\nEvaluating at {len(origins)} daily origins in test period...")

    # Collect predictions per (horizon, channel)
    # Structure: results[channel_name][horizon_name] = {"true": [], "pred": [], "mask": []}
    collectors = {}
    for ch_name in TARGET_NAMES:
        collectors[ch_name] = {}
        for hz_name in TTM_HORIZONS:
            collectors[ch_name][hz_name] = {"true": [], "pred": [], "mask": [],
                                            "true15": [], "mask15": []}

    n_done = 0
    for origin in origins:
        ctx_start = origin - TTM_CONTEXT
        if ctx_start < 0:
            continue

        for s in range(n_stations):
            # Context: (TTM_CONTEXT, TTM_NUM_CHANNELS)
            ctx = hourly_scaled[s, :, ctx_start:origin].T  # (ctx_len, n_ch)
            ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)  # (1, ctx, ch)

            with torch.no_grad():
                out = model(past_values=ctx_tensor)
                pred_scaled = out.prediction_outputs.numpy()[0]  # (pred_len, n_ch)

            # For each horizon, extract the prediction at that step
            for hz_name, hz_step in TTM_HORIZONS.items():
                pred_step = hz_step - 1  # 0-indexed
                if pred_step >= pred_scaled.shape[0]:
                    continue

                pred_row = pred_scaled[pred_step]  # (n_ch,)

                # Unscale prediction
                pred_unscaled = pred_row * scaler.scale_ + scaler.mean_

                # Target indices in TTM channels: PM2.5=0, PM10=1
                for ti, (ch_idx_ttm, ch_name) in enumerate(zip([0, 1], TARGET_NAMES)):
                    target_h = origin + hz_step - 1
                    if target_h >= test_end_h:
                        continue

                    # Get ground truth from ORIGINAL (unscaled, unfilled) hourly data
                    true_val = hourly_vals[s, ch_idx_ttm, target_h]
                    obs_val = hourly_obs[s, ch_idx_ttm, target_h]
                    pred_val = max(float(pred_unscaled[ch_idx_ttm]), 0.0)

                    # Same target as the other baselines: the 15-min point
                    # t + h - 1 with t = 4 * origin and h = 4 * hz_step steps
                    t15 = STEPS_PER_HOUR * (origin + hz_step) - 1
                    ch_raw = TTM_CHANNEL_INDICES[ch_idx_ttm]

                    collectors[ch_name][hz_name]["true"].append(true_val)
                    collectors[ch_name][hz_name]["pred"].append(pred_val)
                    collectors[ch_name][hz_name]["mask"].append(obs_val)
                    collectors[ch_name][hz_name]["true15"].append(values[s, ch_raw, t15])
                    collectors[ch_name][hz_name]["mask15"].append(observed[s, ch_raw, t15])

        n_done += 1
        if n_done % 50 == 0:
            print(f"  Processed {n_done}/{len(origins)} origins...")

    # 8. Compute metrics
    print("\n" + "=" * 70)
    print("IBM TTM Zero-Shot Results (CORRECTED)")
    print("=" * 70)

    # "IBM TTM (zero-shot)" is scored against hourly-mean truth (the original
    # protocol of this script). The 15-min entry scores the same forecasts on
    # the 15-min points used by every other baseline.
    scorings = {"IBM TTM (zero-shot)": ("true", "mask"),
                "IBM TTM (zero-shot, 15-min scoring)": ("true15", "mask15")}
    results = {name: {} for name in scorings}

    for name, (tk, mk) in scorings.items():
        print(f"\n{name}")
        for ch_name in TARGET_NAMES:
            results[name][ch_name] = {}
            for hz_name in TTM_HORIZONS:
                c = collectors[ch_name][hz_name]
                y_true = np.array(c[tk], dtype=np.float32)
                y_pred = np.array(c["pred"], dtype=np.float32)
                mask = np.array(c[mk], dtype=np.float32)

                metrics = compute_metrics(y_true, y_pred, mask)
                results[name][ch_name][hz_name] = metrics
                print(f"  {ch_name} @ {hz_name}: MAE={metrics['MAE']:.2f}, "
                      f"RMSE={metrics['RMSE']:.2f} (n={metrics['n']})")

                if ch_name == "PM2.5" and hz_name == "24h":
                    aqi = aqi_stratified(y_true, y_pred, mask)
                    results[name][ch_name]["24h_aqi"] = aqi
                    print(f"    AQI breakdown:")
                    for band, m in aqi.items():
                        print(f"      {band}: MAE={m['MAE']:.2f}, n={m['n']}")

    # 9. Save
    save_results(results, "ttm_results.json")
    print_summary(results)


if __name__ == "__main__":
    main()
