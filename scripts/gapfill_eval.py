"""Gap filling against stronger baselines (paper Part B; plan: /home/phalak/aqfm/paper/gapfill_plan.md).

The test of aqfm/evaluate.py evaluate_imputation (not changed; its pieces are imported): for each test day and
station, all pollutant readings of the station are hidden over [context - 2 gap, context - gap) of the window that
ends at the forecast start, and the hidden observed readings are scored. Here at several gap lengths, with AQFM's
5 % / 50 % / 95 % sample quantiles, and four baselines that see only what AQFM sees in the same window (or the training
years): the straight line, line + daily cycle, the nearest stations scaled to the station, and an XGBoost gap filler.
Diebold-Mariano tests and bootstrap intervals as scripts/forecast_stats.py.

    python scripts/gapfill_eval.py --data data/delhi --gaps 4 12 24 48 \
        --model fw runs/v2m256_no2d_s{0,1,2}_do03/best.pt --model nofm runs/v2m256_no2d_nofm_s{0,1,2}_do03/best.pt \
        --train-from 2018-01-01 --train-to 2024-01-01 --out results/paper/delhi_gapfill.json \
        --save results/forecasts/delhi_gapfill.npz
"""
import argparse
import dataclasses
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import xgboost as xgb
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from aqfm.config import RAW_CHANNELS  # noqa: E402
from aqfm.data import AQData, build_queries  # noqa: E402
from aqfm.evaluate import _FixedBlock, _pooled, _to_targets, _Windows, select_stations  # noqa: E402
from aqfm.train import code_version, hours_since_epoch, load_checkpoint, load_multi, split_indices, to_device  # noqa: E402
from baseline_utils import HORIZONS, SEED  # noqa: E402
import forecast_stats as fs  # noqa: E402

warnings.filterwarnings("ignore", "Mean of empty slice")
R = {n: i for i, n in enumerate(RAW_CHANNELS)}
K_NB, MIN_OVERLAP = 3, 16
XGB = dict(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=5, tree_method="hist", random_state=SEED, n_jobs=-1)
METHODS = ("line", "cycle", "neighbours", "xgboost")


def daily_cycle(values, observed, ci, lo, hi, hour):
    """(S, 24) mean of each station's observed values by hour of day over steps [lo, hi); gaps -> station mean."""
    S = values.shape[0]
    cyc = np.zeros((S, 24))
    h = hour[lo:hi]
    for s in range(S):
        v, o = np.asarray(values[s, ci, lo:hi], np.float64), np.asarray(observed[s, ci, lo:hi]) > 0
        if not o.any():
            continue
        sums, cnt = np.bincount(h[o], v[o], 24), np.bincount(h[o], minlength=24)
        cyc[s] = np.where(cnt > 0, sums / np.maximum(cnt, 1), v[o].mean())
    return cyc


def fill_window(series, obs, start, gap, cyc_a, hours_w, nbs):
    """Baselines for one window of one station and pollutant. series/obs: (Tc,) of the station; start/gap: hidden
    steps [start, start + gap); cyc_a: (24,) its daily cycle; hours_w: (Tc,) hour of day; nbs: list of (series,
    obs) of the nearest neighbours, nearest first. Returns (idx, dict of (gap,) predictions, (gap, F) features),
    or None when fewer than 2 readings are visible (as evaluate_imputation)."""
    vis = obs.copy()
    vis[start:start + gap] = False
    if vis.sum() < 2:
        return None
    idx = np.arange(start, start + gap)
    xv, sv = np.flatnonzero(vis), series[vis]
    line = np.interp(idx, xv, sv)
    cv = cyc_a[hours_w]
    cycle = np.interp(idx, xv, sv - cv[vis]) + cv[idx]
    scaled = np.full((K_NB, gap), np.nan)
    for j, (ns, no) in enumerate(nbs[:K_NB]):
        both = vis & no
        if both.sum() >= MIN_OVERLAP and ns[both].mean() > 0:
            r = series[both].mean() / ns[both].mean()
            scaled[j] = np.where(no[idx], ns[idx] * r, np.nan)
    have = ~np.isnan(scaled).all(0)
    neigh = np.where(have, np.nanmean(np.where(np.isnan(scaled), np.nan, scaled), 0) if have.any() else line, line)
    before = series[xv[xv < start][-1]] if (xv < start).any() else np.nan
    after = series[xv[xv >= start + gap][0]] if (xv >= start + gap).any() else np.nan
    feats = np.column_stack([np.full(gap, before), np.full(gap, after), line, cycle, neigh, scaled.T, cv[idx],
                             (idx - start + 0.5) / gap, np.full(gap, gap / 4.0), hours_w[idx]])
    return idx, {"line": line, "cycle": cycle, "neighbours": neigh}, feats


def met_feats(met, a, steps):
    return np.asarray(met[a][:, steps]).T if met is not None else np.zeros((len(steps), 0))


@torch.no_grad()
def run_aqfm(models, cfg, data, raw_v, raw_o, test_start, test_end, gap, targets, anchors, allowed, device,
             n_samples, limit_origins, extra=None):
    """The loop of evaluate_imputation, keeping every scored point. extra: dict with cyc, hour, met -> also the
    baselines and XGBoost features (computed once, from the first model setting)."""
    for m in models:                                                     # dropout off, as evaluate_imputation
        m.eval()
    Tc, P, Ct = cfg.context_steps, cfg.patch_len, cfg.n_target_channels
    start = Tc - 2 * gap
    assert start >= 0 and start % P == 0 and gap % P == 0
    icfg = dataclasses.replace(cfg, max_impute_positions=max(cfg.max_impute_positions, gap // P))
    origins = np.arange(test_start, (test_end or data.T) - max(HORIZONS.values()), 96)
    if limit_origins:
        origins = origins[:limit_origins]
    pairs = [(int(o - Tc), int(a)) for o in origins for a in anchors]
    dl = DataLoader(_Windows(data, icfg, pairs, _FixedBlock(start, gap), allowed), batch_size=16)
    rec = {c: {k: [] for k in ("anchor", "origin", "step", "truth", "med", "q05", "q95", "line", "cycle",
                               "neighbours", "feats")} for c in targets}
    for batch in dl:
        t0s, anc, sel = batch["t0"].numpy(), batch["sel"][:, 0].numpy(), batch["sel"].numpy()
        batch = to_device(batch, device)
        q, _, _ = build_queries(icfg, batch, forecast=False)
        pos = batch["imp_pos"]
        Pm = pos.shape[1]
        samp = _pooled(models, icfg, batch, q, torch.arange(Pm * Ct, device=device), n_samples)
        n, B = samp.shape[:2]
        ens = _to_targets(icfg, samp.reshape(n, B, Pm, Ct, P).transpose(2, 3))
        valid_all = batch["imp_valid"].cpu().numpy()
        pos_np = pos.cpu().numpy()
        for c in targets:
            qs = torch.quantile(ens[c], torch.tensor([0.05, 0.5, 0.95], device=device), dim=0).cpu().numpy()
            med_t = ens[c].median(0).values.cpu().numpy()                # the median evaluate_imputation uses
            for b in range(B):
                a, t0 = int(anc[b]), int(t0s[b])
                series = np.asarray(raw_v[a, R[c], t0:t0 + Tc], np.float64)
                obs = np.asarray(raw_o[a, R[c], t0:t0 + Tc]) > 0
                vis = obs.copy()
                vis[start:start + gap] = False
                if vis.sum() < 2:
                    continue
                if extra is not None:
                    nbs = [(np.asarray(raw_v[j, R[c], t0:t0 + Tc], np.float64), np.asarray(raw_o[j, R[c], t0:t0 + Tc]) > 0)
                           for j in sel[b, 1:] if j >= 0 and j != a]
                    fw_ = fill_window(series, obs, start, gap, extra["cyc"][c][a], extra["hour"][t0:t0 + Tc], nbs)
                for k in np.flatnonzero(valid_all[b]):
                    s_, n_ = pos_np[b, k].tolist()
                    if s_ != 0:
                        continue
                    idx = np.arange(n_ * P, (n_ + 1) * P)
                    tgt = obs[idx]
                    if not tgt.any():
                        continue
                    m = len(idx[tgt])
                    rec[c]["anchor"].append(np.full(m, a))
                    rec[c]["origin"].append(np.full(m, t0 + Tc))
                    rec[c]["step"].append(t0 + idx[tgt])
                    rec[c]["truth"].append(series[idx][tgt])
                    rec[c]["med"].append(med_t[b, k][tgt])
                    rec[c]["q05"].append(qs[0, b, k][tgt])
                    rec[c]["q95"].append(qs[2, b, k][tgt])
                    rec[c]["line"].append(np.interp(idx, np.flatnonzero(vis), series[vis])[tgt])
                    if extra is not None:
                        g = idx[tgt] - start
                        for meth in ("cycle", "neighbours"):
                            rec[c][meth].append(fw_[1][meth][g])
                        rec[c]["feats"].append(np.hstack([fw_[2][g], met_feats(extra["met"], a, t0 + idx[tgt])]))
    out = {}
    for c in targets:
        out[c] = {k: np.concatenate(v) for k, v in rec[c].items() if v}
    return out


def xgb_training_rows(data, raw_v, raw_o, c, cfg, gaps, lo, hi, cyc, hour, met, rng, n_win=4000):
    """Feature rows from simulated gaps in windows inside [lo, hi) (training years only)."""
    Tc = cfg.context_steps
    X, y = [], []
    for gap in gaps:
        start = Tc - 2 * gap
        for _ in range(n_win):
            a = int(rng.integers(data.S))
            t0 = int(rng.integers(lo, hi - Tc))
            series = np.asarray(raw_v[a, R[c], t0:t0 + Tc], np.float64)
            obs = np.asarray(raw_o[a, R[c], t0:t0 + Tc]) > 0
            hidden = np.flatnonzero(obs[start:start + gap])
            if not len(hidden):
                continue
            sel = data.neighbourhood(a, cfg.n_neighbours, cfg.neighbour_radius_km)
            nbs = [(np.asarray(raw_v[j, R[c], t0:t0 + Tc], np.float64), np.asarray(raw_o[j, R[c], t0:t0 + Tc]) > 0)
                   for j in sel[1:]]
            fw_ = fill_window(series, obs, start, gap, cyc[a], hour[t0:t0 + Tc], nbs)
            if fw_ is None:
                continue
            keep = hidden[rng.random(len(hidden)) < 0.25]
            if not len(keep):
                continue
            X.append(np.hstack([fw_[2][keep], met_feats(met, a, t0 + start + keep)]))
            y.append(series[start + keep])
    return np.vstack(X), np.concatenate(y)


def compare(r, methods, n_boot_seed=0):
    """MAE per method, coverage, and each baseline against AQFM (Diebold-Mariano on daily means; bootstrap)."""
    y = r["truth"]
    err = {m: np.abs(r[m] - y) for m in methods}
    day = r["origin"] // 96
    days, inv = np.unique(day, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    sums = {m: np.bincount(inv, weights=err[m]) for m in methods}
    out = {"n": int(len(y)), "days": int(len(days)), "MAE": {m: float(err[m].mean()) for m in methods},
           "coverage90": float(((y >= r["q05"]) & (y <= r["q95"])).mean()), "vs_aqfm": {}}
    for m in methods[1:]:
        d = (sums[m] - sums["med"]) / cnt
        p, z = fs.newey_west_p(d)
        out["vs_aqfm"][m] = {"MAE_diff": float(err[m].mean() - err["med"].mean()), "dm_z": z, "dm_p": p,
                             "ci95": fs.block_bootstrap(sums[m], sums["med"], cnt)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", nargs="+", action="append", required=True, metavar="NAME CKPT",
                    help="a setting name and its checkpoints (ensemble); the first setting also computes the baselines")
    ap.add_argument("--gaps", nargs="+", type=int, default=[4, 12, 24, 48], help="gap lengths, hours")
    ap.add_argument("--targets", nargs="+", default=["PM2.5", "PM10", "Ozone", "NOx", "NO2"])
    ap.add_argument("--val-start", default=None)
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--test-end", default=None, help="YYYY-MM-DD: end of the scored period, exclusive (default: end "
                    "of the data); e.g. the validation year")
    ap.add_argument("--train-from", required=True, help="YYYY-MM-DD: cycle and XGBoost training data start")
    ap.add_argument("--train-to", required=True, help="YYYY-MM-DD: ... end (exclusive)")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--limit-origins", type=int, default=None, help="debug only")
    ap.add_argument("--xgb-windows", type=int, default=4000, help="training windows per gap length (plan: 4000)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default=None, help=".npz with every scored point (git-ignored folder)")
    args = ap.parse_args()
    t_begin = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = load_multi([args.data])
    t = np.asarray(a["timestamps_h"])
    hour = np.floor(np.mod(t, 24)).astype(int)
    raw_v, raw_o = a["values"], a["observed"]
    met = np.load(os.path.join(args.data, "era5_met.npy"), mmap_mode="r")
    lo = int(np.searchsorted(t, hours_since_epoch(*map(int, args.train_from.split("-")))))
    hi = int(np.searchsorted(t, hours_since_epoch(*map(int, args.train_to.split("-")))))
    test_end = (int(np.searchsorted(t, hours_since_epoch(*map(int, args.test_end.split("-")))))
                if args.test_end else None)
    cyc = {c: daily_cycle(raw_v, raw_o, R[c], lo, hi, hour) for c in args.targets}
    res = {"plan": "/home/phalak/aqfm/paper/gapfill_plan.md", "args": vars(args), "code": code_version(), "gaps": {}}
    saved = {}
    for gi, gap_h in enumerate(args.gaps):
        gap = gap_h * 4
        res["gaps"][f"{gap_h}h"] = {}
        base = None
        for si, (name, *ckpts) in enumerate(args.model):
            torch.manual_seed(0)                                         # as aqfm.evaluate --seed 0
            np.random.seed(0)
            loaded = [load_checkpoint(p, device) for p in ckpts]
            models, cfg = [m for m, _, _ in loaded], loaded[0][1]
            val_start, test_start = split_indices(t, 2024, 2025, args.val_start, args.test_start)
            data = AQData(a["values"], a["observed"], a["timestamps_h"], a["coords"], a["static"],
                          train_end=val_start, era5_met=a["era5_met"] if cfg.use_future_met else None,
                          met_stats=loaded[0][2].get("met_stats"), group=a["group"], group_names=a["group_names"])
            anchors, allowed, _ = select_stations("all", data, [], True)
            extra = {"cyc": cyc, "hour": hour, "met": met} if si == 0 else None
            t1 = time.time()
            rec = run_aqfm(models, cfg, data, raw_v, raw_o, test_start, test_end, gap, args.targets, anchors, allowed,
                           device, args.samples, args.limit_origins, extra)
            print(f"  {gap_h} h, {name}: AQFM pass {time.time() - t1:.0f}s", flush=True)
            if si == 0:
                base = rec
                if gi == 0:
                    xgb_models = {}
                    for c in args.targets:                               # trained once, all gap lengths together
                        X, y = xgb_training_rows(data, raw_v, raw_o, c, cfg, [g * 4 for g in args.gaps], lo, hi,
                                                 cyc[c], hour, met, np.random.default_rng(SEED), args.xgb_windows)
                        xgb_models[c] = xgb.XGBRegressor(**XGB).fit(X, y)
                        print(f"  XGBoost gap filler {c}: {len(y)} training rows", flush=True)
                for c in args.targets:
                    base[c]["xgboost"] = np.maximum(xgb_models[c].predict(base[c]["feats"]), 0.0)
            for c in args.targets:
                r = rec[c]
                for k in ("anchor", "origin", "step", "truth"):
                    assert np.array_equal(r[k], base[c][k]), f"{name}: points differ from the first setting"
                for m in ("line",) + METHODS[1:]:
                    r[m] = base[c][m]
                res["gaps"][f"{gap_h}h"].setdefault(name, {})[c] = compare(r, ("med",) + METHODS)
                for k in ("med", "q05", "q95"):
                    saved[f"{gap_h}h/{name}/{c}/{k}"] = r[k].astype(np.float32)
                if si == 0:
                    for k in ("anchor", "origin", "step"):
                        saved[f"{gap_h}h/{c}/{k}"] = r[k].astype(np.int64)
                    for k in ("truth",) + METHODS:
                        saved[f"{gap_h}h/{c}/{k}"] = r[k].astype(np.float32)
                v = res["gaps"][f"{gap_h}h"][name][c]
                print(f"    {c}: n={v['n']} " + " ".join(f"{m} {x:.2f}" for m, x in v["MAE"].items())
                      + f" cov90 {v['coverage90']:.3f}", flush=True)
    res["seconds"] = round(time.time() - t_begin)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    if args.save:
        np.savez_compressed(args.save, **{k.replace("/", "__"): v for k, v in saved.items()})
    print("saved", args.out, f"({res['seconds']} s)")


if __name__ == "__main__":
    main()
