"""The XGBoost baseline of baselines/run_baselines_xgboost.py (same features, model settings and seed; that file is
fixed to data/preprocessed and 2025 and stays unchanged) for any city folder, test start and training period.

--hourly: the city has one reading per hour, on the hh:45 step (data_pipeline/convert_hourly_city.py). Every lookup
then reads the latest hourly reading at or before the original step ("as of"), so a lag of 1 h still means the
reading of 1 h before; without it the most recent lag (the hh:00 step) is never observed and no forecast is valid.

    python scripts/city_xgboost.py --data data/mumbai --hourly --test-start 2024-07-01 \
        --out results/mumbai_xgboost_results.json                       # trains on everything before the test
    python scripts/city_xgboost.py --data data/mumbai --hourly --test-start 2024-07-01 \
        --train-from 2023-10-01 --train-to 2024-01-01 --out results/mumbai_xgboost_m3_results.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "baselines"))
import run_baselines_xgboost as rx  # noqa: E402
from baseline_utils import (CHANNEL_NAMES, HORIZONS, SEED, STEPS_PER_DAY, TARGET_NAMES,  # noqa: E402
                            aqi_stratified, compute_metrics, timestamps_to_dt_features)

EPOCH = datetime(2018, 1, 1)


def step_of(t, date):
    return int(np.searchsorted(t, (datetime.fromisoformat(date) - EPOCH).total_seconds() / 3600))


def as_of_hourly(a):
    """(S, C, T) with readings on steps 4k+3 -> every step holds the latest reading at or before it."""
    out = np.zeros_like(a)
    T = a.shape[-1]
    i = np.arange(T)
    src = i - ((i - 3) % 4)                      # 4k+3 -> itself; 4k, 4k+1, 4k+2 -> 4k-1
    ok = src >= 0
    out[..., ok] = a[..., src[ok]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--train-from", default=None, help="YYYY-MM-DD (default: the start of the record)")
    ap.add_argument("--train-to", default=None, help="YYYY-MM-DD, targets before it (default: the test start)")
    ap.add_argument("--hourly", action="store_true")
    ap.add_argument("--future-met", action="store_true",
                    help="add ERA5 weather at the target time (era5_met.npy), as run_baselines_xgboost.py --future-met")
    ap.add_argument("--test-end", default=None, help="YYYY-MM-DD: scored targets end before it (default: data end)")
    ap.add_argument("--targets", nargs="+", default=list(TARGET_NAMES), choices=["PM2.5", "PM10", "Ozone", "NOx", "NO2"],
                    help="for targets other than PM2.5 the 'other pollutant' feature is PM2.5 (as in the original)")
    ap.add_argument("--save-forecasts", default=None, help=".npz: anchor, origin, <c>_true, <c>_mask, <c>_xgboost")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    values = np.load(os.path.join(args.data, "values.npy"), mmap_mode="r")
    observed = np.load(os.path.join(args.data, "observed.npy"), mmap_mode="r")
    t = np.load(os.path.join(args.data, "timestamps_h.npy"))
    if args.hourly:
        values, observed = as_of_hourly(np.asarray(values)), as_of_hourly(np.asarray(observed))
    era5_met = np.load(os.path.join(args.data, "era5_met.npy"), mmap_mode="r") if args.future_met else None
    test_start, test_end = step_of(t, args.test_start), len(t)
    lo = max(rx.MAX_LAG, step_of(t, args.train_from) if args.train_from else 0)
    hi = step_of(t, args.train_to) if args.train_to else test_start
    assert hi <= test_start
    feats = timestamps_to_dt_features(t)
    score_end = step_of(t, args.test_end) if args.test_end else test_end
    origins = np.arange(test_start, score_end - max(HORIZONS.values()), STEPS_PER_DAY)
    S = values.shape[0]
    assert origins.min() >= rx.MAX_LAG
    saved = {"anchor": np.repeat(np.arange(S), len(origins)), "origin": np.tile(origins, S)}
    name = "XGBoost + future ERA5" if args.future_met else "XGBoost"
    res = {name: {}}
    for cn in args.targets:
        ci = CHANNEL_NAMES.index(cn)
        res[name][cn] = {}
        for hz, h in HORIZONS.items():
            t0 = time.time()
            train_idx = np.arange(lo, hi - h, rx.TRAIN_SUBSAMPLE)
            Xp, yp = [], []
            for s in range(values.shape[0]):
                X, y, m, v = rx.build_features_vectorized(values, observed, feats, s, ci, train_idx, h, era5_met)
                keep = (m > 0.5) & v
                if keep.any():
                    Xp.append(X[keep]); yp.append(y[keep])
            Xtr, ytr = np.concatenate(Xp), np.concatenate(yp)
            model = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                                     random_state=SEED, n_jobs=-1)
            model.fit(Xtr, ytr, verbose=False)
            yt, yp_, mk = [], [], []
            for s in range(values.shape[0]):
                o = origins[origins >= rx.MAX_LAG]
                o = o[o + h - 1 < test_end]
                if len(o) == 0:
                    continue
                X, y, m, _ = rx.build_features_vectorized(values, observed, feats, s, ci, o, h, era5_met)
                yt.append(y); yp_.append(np.maximum(model.predict(X), 0.0)); mk.append(m)
            yt, yp_, mk = np.concatenate(yt), np.concatenate(yp_), np.concatenate(mk)
            assert len(yt) == S * len(origins)                  # station-major, as saved["anchor"/"origin"]
            k = list(HORIZONS).index(hz)
            for key, arr in ((f"{cn}_true", yt), (f"{cn}_mask", mk), (f"{cn}_xgboost", yp_)):  # method name: xgboost
                saved.setdefault(key, np.full((S * len(origins), len(HORIZONS)), np.nan, np.float32))[:, k] = arr
            res[name][cn][hz] = compute_metrics(yt, yp_, mk)
            if cn == "PM2.5" and hz == "24h":
                res[name][cn]["24h_aqi"] = aqi_stratified(yt, yp_, mk)
            print(f"{cn} {hz}: train {len(ytr)} samples, MAE {res[name][cn][hz]['MAE']:.2f} "
                  f"(n={res[name][cn][hz]['n']}, {time.time() - t0:.0f}s)")
    res["_meta"] = {k: v for k, v in vars(args).items()} | {"train_steps": [int(lo), int(hi)], "n_origins": len(origins)}
    if args.save_forecasts:
        np.savez_compressed(args.save_forecasts, origin_hours=t[saved["origin"]], **saved)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
