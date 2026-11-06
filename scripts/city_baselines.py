"""Persistence, seasonal naive and climatology for any city folder and test start, with exactly the rules of
baselines/run_baselines_classical.py (which is fixed to data/preprocessed and 2025): origins every midnight from
the test start to T - 72 h (as aqfm/evaluate.py), persistence = last observed value within 7 days before the
origin, seasonal naive = the value 1..8 days before the target, climatology = station x hour x month mean of
everything before the test; a point counts only where the target is observed and all three forecasts exist.

    python scripts/city_baselines.py --data data/mumbai --test-start 2024-07-01 --out results/mumbai_classical_results.json
    python scripts/city_baselines.py --data data/preprocessed --out <scratch>.json   # = baselines/results/classical_results.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "baselines"))
from baseline_utils import (CHANNEL_NAMES, HORIZONS, STEPS_PER_DAY, TARGET_NAMES,  # noqa: E402
                            aqi_stratified, compute_metrics, timestamps_to_dt_features)
from run_baselines_classical import build_climatology_dict  # noqa: E402

EPOCH = datetime(2018, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test-start", default="2025-01-01", help="YYYY-MM-DD")
    ap.add_argument("--test-end", default=None, help="YYYY-MM-DD: scored targets end before it (default: data end)")
    ap.add_argument("--targets", nargs="+", default=list(TARGET_NAMES), choices=["PM2.5", "PM10", "Ozone", "NOx", "NO2"])
    ap.add_argument("--save-forecasts", default=None,
                    help=".npz: anchor, origin and per target <c>_true, <c>_mask, <c>_<method> at 1/6/24/72 h")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    values = np.load(os.path.join(args.data, "values.npy"), mmap_mode="r")
    observed = np.load(os.path.join(args.data, "observed.npy"), mmap_mode="r")
    t = np.load(os.path.join(args.data, "timestamps_h.npy"))
    test_start = int(np.searchsorted(t, (datetime.fromisoformat(args.test_start) - EPOCH).total_seconds() / 3600))
    test_end = len(t)
    score_end = (int(np.searchsorted(t, (datetime.fromisoformat(args.test_end) - EPOCH).total_seconds() / 3600))
                 if args.test_end else test_end)
    S = values.shape[0]
    feats = timestamps_to_dt_features(t)
    origins = np.arange(test_start, score_end - max(HORIZONS.values()), STEPS_PER_DAY)
    max_walkback = 7 * STEPS_PER_DAY
    res = {m: {c: {} for c in args.targets} for m in ("Persistence", "Seasonal Naive", "Climatology")}
    saved = {"anchor": np.repeat(np.arange(S), len(origins)), "origin": np.tile(origins, S)}
    for cn in args.targets:
        ci = CHANNEL_NAMES.index(cn)
        v, o = np.asarray(values[:, ci]), np.asarray(observed[:, ci])
        clim = build_climatology_dict(values, observed, feats, test_start, S, ci)
        for hz, h in HORIZONS.items():
            yt, pp, sn, cl, mk = [], [], [], [], []
            for s in range(S):
                for t0 in origins:
                    tt = t0 + h - 1
                    if tt >= test_end:
                        continue
                    pers = np.nan
                    for w in range(1, max_walkback + 1):
                        if t0 - w >= 0 and o[s, t0 - w] > 0.5:
                            pers = v[s, t0 - w]
                            break
                    snaive = np.nan
                    for w in range(0, 8):
                        i = tt - STEPS_PER_DAY * (w + 1)
                        if i >= 0 and o[s, i] > 0.5:
                            snaive = v[s, i]
                            break
                    c = clim[s, feats["hour"][tt], feats["month"][tt]]
                    if np.isnan(c):
                        c = np.nanmean(clim[s, :, feats["month"][tt]])
                    yt.append(v[s, tt]); pp.append(pers); sn.append(snaive); cl.append(c)
                    mk.append(1.0 if (o[s, tt] > 0.5 and not np.isnan(pers) and not np.isnan(snaive)
                                      and not np.isnan(c)) else 0.0)
            yt, mk = np.array(yt, np.float32), np.array(mk, np.float32)
            assert len(yt) == S * len(origins)                  # station-major, as saved["anchor"/"origin"]
            k = list(HORIZONS).index(hz)
            for key, arr in ((f"{cn}_true", yt), (f"{cn}_mask", np.array([o[s, t0 + h - 1] for s in range(S)
                                                                          for t0 in origins], np.float32)),
                             (f"{cn}_persistence", pp), (f"{cn}_snaive", sn), (f"{cn}_clim", cl)):
                saved.setdefault(key, np.full((S * len(origins), len(HORIZONS)), np.nan, np.float32))[:, k] = arr
            for name, p in (("Persistence", pp), ("Seasonal Naive", sn), ("Climatology", cl)):
                p = np.array(p, np.float32)
                res[name][cn][hz] = compute_metrics(yt, p, mk)
                if cn == "PM2.5" and hz == "24h":
                    res[name][cn]["24h_aqi"] = aqi_stratified(yt, p, mk)
            print(f"{cn} {hz}: " + "  ".join(f"{n} {res[n][cn][hz]['MAE']:.2f}" for n in res)
                  + f"  (n={res['Persistence'][cn][hz]['n']})")
    res["_meta"] = {"data": args.data, "test_start": args.test_start, "test_end": args.test_end,
                    "n_origins": len(origins), "stations": S}
    if args.save_forecasts:
        np.savez_compressed(args.save_forecasts, origin_hours=t[saved["origin"]], **saved)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
