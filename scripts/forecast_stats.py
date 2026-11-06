"""Compare forecasts on exactly the same points, with significance. Inputs are --save-forecasts files of
aqfm/evaluate.py (every step of the 72 h horizon) and of scripts/city_baselines.py / scripts/city_xgboost.py
(the 1/6/24/72 h leads); a baseline file is named with its method after a colon.

    python scripts/forecast_stats.py --target PM2.5 --out results/delhi_pm25_matched.json \
        --model "AQFM v2-M" results/forecasts/delhi_v2m256_ens3.npz \
        --model XGBoost results/forecasts/delhi_xgboost.npz:xgboost \
        --model Persistence results/forecasts/delhi_classical.npz:persistence

For each lead: the points where the target is observed and every model has a forecast (the same points for all);
MAE of each model; for each model against the first one: the MAE difference, a Diebold-Mariano test on the daily
mean absolute-error difference (Newey-West variance, lag 7 days; two-sided normal p-value) and a 95 % moving-block
bootstrap interval of the difference (7-day blocks, 2,000 resamples, seed 0). Also, at 24 h: MAE by month of the
origin and, for PM2.5, MAE and share of all absolute error by AQI band of the truth.
"""

import argparse
import json
from datetime import datetime, timedelta
from math import erf, sqrt

import numpy as np

LEADS = {"1h": 4, "6h": 24, "24h": 96, "72h": 288}
BANDS = {"Good": (0, 30), "Satisfactory": (30, 60), "Moderate": (60, 90), "Poor": (90, 120),
         "Very Poor": (120, 250), "Severe": (250, np.inf)}
EPOCH = datetime(2018, 1, 1)


def load(spec, target):
    """-> dict (anchor, origin) -> (pred[4], true[4], mask[4]); origin_hours by origin."""
    path, _, method = spec.partition(":")
    f = np.load(path)
    if method:                                            # baseline file: columns are the 4 leads
        pred = f[f"{target}_{method}"]
        true, mask = f[f"{target}_true"], f[f"{target}_mask"]
    else:                                                 # AQFM file: all steps; take the lead steps
        cols = [h - 1 for h in LEADS.values()]
        pred, true, mask = (f[f"{target}_{k}"][:, cols] for k in ("pred", "true", "mask"))
    keys = list(zip(f["anchor"].tolist(), f["origin"].tolist()))
    assert len(set(keys)) == len(keys), f"{path}: repeated (anchor, origin) pairs"
    hours = dict(zip(f["origin"].tolist(), f["origin_hours"].tolist()))
    return {k: (pred[i], true[i], mask[i]) for i, k in enumerate(keys)}, hours


def newey_west_p(d, lag=7):
    """Two-sided p-value that the mean of the daily series d is 0 (Newey-West variance)."""
    n = len(d)
    x = d - d.mean()
    v = x @ x / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n
    z = d.mean() / sqrt(max(v, 1e-12) / n)
    return float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))), float(z)


def block_bootstrap(sa, sb, n, block=7, reps=2000, seed=0):
    """95 % interval of sum(sa)/sum(n) - sum(sb)/sum(n) over days resampled in moving blocks."""
    rng = np.random.default_rng(seed)
    D = len(n)
    nb = -(-D // block)
    out = []
    for _ in range(reps):
        starts = rng.integers(0, max(1, D - block + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + block, D)) for s in starts])[:D]
        out.append((sa[idx].sum() - sb[idx].sum()) / max(n[idx].sum(), 1))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="PM2.5")
    ap.add_argument("--model", nargs=2, action="append", metavar=("NAME", "FILE[:METHOD]"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    names = [m[0] for m in args.model]
    data, hours = {}, {}
    for name, spec in args.model:
        data[name], h = load(spec, args.target)
        hours.update(h)
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    assert common, "no (anchor, origin) pair in common"
    res = {"target": args.target, "models": dict(args.model), "pairs_in_common": len(common), "leads": {}}
    for j, lead in enumerate(LEADS):
        rows = []
        for k in common:
            preds = [data[n][k][0][j] for n in names]
            true, ok = data[names[0]][k][1][j], data[names[0]][k][2][j] > 0.5
            for n in names[1:]:                            # same truth in every file
                assert not ok or abs(data[n][k][1][j] - true) < 1e-3 * max(1.0, abs(true)), (n, k, lead)
            if ok and all(np.isfinite(p) for p in preds):
                rows.append((k[1], true, preds))
        org = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows], np.float64)
        err = np.abs(np.array([r[2] for r in rows], np.float64) - y[:, None])        # (n, models)
        days, inv = np.unique(org // 96, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        sums = np.stack([np.bincount(inv, weights=err[:, m]) for m in range(len(names))], 1)
        out = {"n": len(rows), "days": len(days), "MAE": {n: float(err[:, m].mean()) for m, n in enumerate(names)},
               "vs_first": {}}
        for m, n in enumerate(names[1:], 1):
            d = (sums[:, m] - sums[:, 0]) / cnt
            p, z = newey_west_p(d)
            out["vs_first"][n] = {"MAE_diff": float(err[:, m].mean() - err[:, 0].mean()), "dm_z": z, "dm_p": p,
                                  "ci95": block_bootstrap(sums[:, m], sums[:, 0], cnt)}
        if lead == "24h":
            month = np.array([(EPOCH + timedelta(hours=hours[o])).month for o in org])
            out["by_month"] = {int(mo): {"n": int((month == mo).sum()),
                                         **{n: float(err[month == mo, m].mean()) for m, n in enumerate(names)}}
                               for mo in sorted(set(month))}
            if args.target == "PM2.5":
                tot = err.sum(0)
                out["by_band"] = {}
                for b, (lo, hi) in BANDS.items():
                    s = (y >= lo) & (y < hi)
                    out["by_band"][b] = {"n": int(s.sum()), **{n: {"MAE": float(err[s, m].mean()) if s.any() else None,
                                                                   "share_of_error": float(err[s, m].sum() / tot[m])}
                                                               for m, n in enumerate(names)}}
        res["leads"][lead] = out
        print(f"{args.target} {lead}: n={out['n']} " + "  ".join(f"{n} {v:.2f}" for n, v in out["MAE"].items())
              + "".join(f" | {n}-{names[0]} {v['MAE_diff']:+.2f} (p={v['dm_p']:.3f})" for n, v in out["vs_first"].items()))
    json.dump(res, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
