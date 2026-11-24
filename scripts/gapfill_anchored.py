"""Edge-anchored AQFM gap filling (paper Part B2; plan: /home/phalak/aqfm/paper/gapfill_plan.md, "Part B2").

AQFM's fill plus the straight line between its errors at the two gap edges: r_before = last visible reading before
the gap - AQFM at the first scored hidden step, r_after = first visible reading after the gap - AQFM at the last
scored hidden step, interpolated in time between the two readings (one side only -> constant), clipped at 0. The 5 %
and 95 % quantiles are shifted by the same amount. No model is run: it reads the points saved by
scripts/gapfill_eval.py (--save) and the raw data, and scores against the same four baselines on the same points.

    python scripts/gapfill_anchored.py --data data/delhi --points results/forecasts/delhi_gapfill.npz \
        --settings fw nofm --out results/paper/delhi_gapfill_anchored.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from aqfm.config import RAW_CHANNELS  # noqa: E402
from aqfm.train import code_version  # noqa: E402
from gapfill_eval import METHODS, compare  # noqa: E402

R = {n: i for i, n in enumerate(RAW_CHANNELS)}


def edge_shift(steps, pred, series, observed, w0, g0, g1):
    """Correction for one window. steps: (k,) absolute steps of the scored hidden points, pred: (k,) AQFM there;
    series/observed: the station's values and visibility over [w0, w1) (absolute steps w0..); the gap is [g0, g1).
    Returns (k,) shift to add to pred."""
    vis = observed.copy()
    vis[g0 - w0:g1 - w0] = False
    before = np.flatnonzero(vis[:g0 - w0])
    after = np.flatnonzero(vis[g1 - w0:]) + (g1 - w0)
    first, last = np.argmin(steps), np.argmax(steps)
    xs, rs = [], []
    if len(before):
        xs.append(w0 + before[-1])
        rs.append(series[before[-1]] - pred[first])
    if len(after):
        xs.append(w0 + after[0])
        rs.append(series[after[0]] - pred[last])
    if not xs:
        return np.zeros(len(steps))
    return np.interp(steps, xs, rs)


def anchored(z, gap_h, c, setting, values, observed, context):
    """Edge-anchored median and quantiles for every saved point of one gap length, pollutant and model setting."""
    k = lambda m: z[f"{gap_h}h__{c}__{m}"]
    a, o, st = k("anchor"), k("origin"), k("step")
    med, q05, q95 = (z[f"{gap_h}h__{setting}__{c}__{q}"].astype(np.float64) for q in ("med", "q05", "q95"))
    g = gap_h * 4
    shift = np.zeros(len(st))
    key = a.astype(np.int64) * (1 << 32) + o
    _, inv = np.unique(key, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    for grp in np.split(order, bounds):
        ai, oi = int(a[grp[0]]), int(o[grp[0]])
        w0 = oi - context
        series = np.asarray(values[ai, R[c], w0:oi], np.float64)
        obs = np.asarray(observed[ai, R[c], w0:oi]) > 0
        shift[grp] = edge_shift(st[grp], med[grp], series, obs, w0, oi - 2 * g, oi - g)
    clip = lambda x: np.maximum(x, 0.0)
    return clip(med + shift), clip(q05 + shift), clip(q95 + shift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--points", required=True, help=".npz saved by scripts/gapfill_eval.py --save")
    ap.add_argument("--settings", nargs="+", required=True, help="model settings saved in the .npz (e.g. fw nofm)")
    ap.add_argument("--context-steps", type=int, default=512, help="AQFM context (128 h of 15-minute steps)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    z = np.load(args.points)
    values = np.load(os.path.join(args.data, "values.npy"), mmap_mode="r")
    observed = np.load(os.path.join(args.data, "observed.npy"), mmap_mode="r")
    gaps = sorted({int(k.split("h__")[0]) for k in z.files})
    pols = sorted({k.split("__")[1] for k in z.files if k.endswith("__truth")})
    res = {"plan": "/home/phalak/aqfm/paper/gapfill_plan.md (Part B2)", "args": vars(args), "code": code_version(),
           "gaps": {}}
    for gap_h in gaps:
        for setting in args.settings:
            for c in pols:
                k = lambda m: z[f"{gap_h}h__{c}__{m}"]
                med, q05, q95 = anchored(z, gap_h, c, setting, values, observed, args.context_steps)
                r = {m: k(m).astype(np.float64) for m in ("truth",) + METHODS}
                r["origin"] = k("origin")
                r.update(med=med, q05=q05, q95=q95, raw=z[f"{gap_h}h__{setting}__{c}__med"].astype(np.float64))
                v = compare(r, ("med", "raw") + METHODS)            # "med" = edge-anchored AQFM; "raw" = AQFM as is
                res["gaps"].setdefault(f"{gap_h}h", {}).setdefault(setting, {})[c] = v
                print(f"{gap_h} h {setting} {c}: anchored {v['MAE']['med']:.2f} raw {v['MAE']['raw']:.2f} "
                      + " ".join(f"{m} {v['MAE'][m]:.2f}" for m in METHODS) + f" cov90 {v['coverage90']:.3f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
