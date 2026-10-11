"""Shared utilities for all baseline evaluation scripts."""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

PREPROCESSED = "../data/preprocessed"
RESULTS_DIR = "results"

CHANNEL_NAMES = [
    "PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO",
    "AT", "RH", "WS", "WD", "SR", "BP", "TOT_RF", "Benzene",
]
TARGET_CHANNELS = [0, 1]
TARGET_NAMES = ["PM2.5", "PM10"]

HORIZONS = {"1h": 4, "6h": 24, "24h": 96, "72h": 288}

STEPS_PER_HOUR = 4
STEPS_PER_DAY = 96
START_DATE = datetime(2018, 1, 1)

AQI_BANDS = {
    "Good": (0, 30),
    "Satisfactory": (30, 60),
    "Moderate": (60, 90),
    "Poor": (90, 120),
    "Very Poor": (120, 250),
    "Severe": (250, float("inf")),
}

SEED = 42


def load_data():
    print("Loading preprocessed arrays...")
    values = np.load(os.path.join(PREPROCESSED, "values.npy"), mmap_mode="r")
    observed = np.load(os.path.join(PREPROCESSED, "observed.npy"), mmap_mode="r")
    timestamps_h = np.load(os.path.join(PREPROCESSED, "timestamps_h.npy"))
    print(f"  values: {values.shape}, observed: {observed.shape}, ts: {timestamps_h.shape}")
    return values, observed, timestamps_h


def get_test_split(timestamps_h):
    hours_to_2025 = (datetime(2025, 1, 1) - START_DATE).total_seconds() / 3600
    test_start = int(np.searchsorted(timestamps_h, hours_to_2025))
    test_end = len(timestamps_h)
    print(f"  Train: 0..{test_start} ({test_start} steps = {test_start/STEPS_PER_DAY:.0f} days)")
    print(f"  Test:  {test_start}..{test_end} ({test_end - test_start} steps = {(test_end-test_start)/STEPS_PER_DAY:.0f} days)")
    return test_start, test_end


def timestamps_to_dt_features(timestamps_h):
    base = np.datetime64("2018-01-01T00:00")
    dts = base + (timestamps_h * 3600).astype("timedelta64[s]")
    idx = pd.DatetimeIndex(dts)
    return {
        "hour": idx.hour.values,
        "minute": idx.minute.values,
        "month": idx.month.values,
        "dow": idx.dayofweek.values,
        "doy": idx.dayofyear.values,
    }


def compute_metrics(y_true, y_pred, mask):
    valid = mask > 0.5
    n = int(valid.sum())
    if n == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "n": 0}
    err = y_pred[valid] - y_true[valid]
    return {
        "MAE": round(float(np.mean(np.abs(err))), 4),
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "n": n,
    }


def episode_detection(y_true, y_pred, mask, threshold=250.0):
    valid = mask > 0.5
    y_t = y_true[valid]
    y_p = y_pred[valid]
    
    true_episodes = (y_t > threshold)
    pred_episodes = (y_p > threshold)
    
    true_positives = np.sum(true_episodes & pred_episodes)
    false_positives = np.sum((~true_episodes) & pred_episodes)
    false_negatives = np.sum(true_episodes & (~pred_episodes))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "true_episodes_count": int(np.sum(true_episodes))
    }


def aqi_stratified(y_true, y_pred, mask):
    valid = mask > 0.5
    out = {}
    for name, (lo, hi) in AQI_BANDS.items():
        sel = valid & (y_true >= lo) & (y_true < hi)
        n = int(sel.sum())
        if n == 0:
            out[name] = {"MAE": float("nan"), "RMSE": float("nan"), "n": 0}
        else:
            e = y_pred[sel] - y_true[sel]
            out[name] = {
                "MAE": round(float(np.mean(np.abs(e))), 4),
                "RMSE": round(float(np.sqrt(np.mean(e ** 2))), 4),
                "n": n,
            }
    return out


def save_results(results, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def print_summary(all_results):
    print("\n" + "=" * 90)
    print("SUMMARY: MAE (ug/m3)")
    print("=" * 90)
    print(f"{'Baseline':<22} {'Target':<8}", end="")
    for hz in HORIZONS:
        print(f" {hz:>10}", end="")
    print()
    print("-" * 90)
    for bname, res in all_results.items():
        for ch in TARGET_NAMES:
            ch_res = res.get(ch, {})
            print(f"{bname:<22} {ch:<8}", end="")
            for hz in HORIZONS:
                mae = ch_res.get(hz, {}).get("MAE", float("nan"))
                print(f" {mae:>10.2f}", end="")
            print()
