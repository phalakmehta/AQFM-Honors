"""Test-set evaluation on 2025 with the exact protocol of baselines/.

    python -m aqfm.evaluate --ckpt runs/anneal/best.pt --out results/aqfm_results.json
    python -m aqfm.evaluate --ckpt runs/s0/best.pt runs/s1/best.pt runs/s2/best.pt --out ...

Several checkpoints form an equal-weight ensemble: every model contributes
ceil(samples / M) samples to one pooled predictive distribution, and all
metrics below are computed from the pool.

Protocol (identical to run_baselines_*.py): one origin per day at 00:00 in
2025, origins = arange(test_start, T - 288, 96); origin t predicts index
t + h - 1 for h in {1h: 4, 6h: 24, 24h: 96, 72h: 288} steps, using data up to
t - 1. Each station is forecast from its own k-nearest neighbourhood (it is the
anchor, station 0 of the sample). Point forecast = median of the predictive
distribution, in ug/m3. MAE/RMSE/AQI bands/episode detection are computed by
the very functions the baselines use (baseline_utils.py).

Additionally: CRPS by lead step, PIT histograms, 90/95% interval coverage,
PM2.5 > 250 episode detection at 24 h and 48 h, and imputation of a 24 h
station-wide outage scored separately against linear interpolation.
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "baselines"))
from baseline_utils import (HORIZONS, aqi_stratified, compute_metrics,  # noqa: E402
                            episode_detection)

from .config import CH, OUTAGE_CHANNELS, RAW_CHANNELS  # noqa: E402
from .data import AQData, build_queries, make_sample  # noqa: E402
from .losses import mixture_sample, sample_crps  # noqa: E402
from .physical import LOG1P_MAX, reconstruct_pm25  # noqa: E402
from .train import (code_version, load_checkpoint, load_preprocessed,  # noqa: E402
                    split_indices, to_device)

TARGETS = ("PM2.5", "PM10")
EPISODE_STEPS = {"24h": 95, "48h": 191}


class _FixedBlock:
    """Hide the anchor station's CPCB channels over [start, start+length)."""

    def __init__(self, start, length):
        self.start, self.length = start, length

    def sample(self, observed, rng):
        visible = np.ones_like(observed)
        visible[0, list(OUTAGE_CHANNELS), self.start:self.start + self.length] = 0.0
        visible[0, CH["PM_FRAC"], self.start:self.start + self.length] = 0.0
        visible[0, CH["NOX_FRAC"], self.start:self.start + self.length] = 0.0
        return {"input_mask": observed * visible, "impute_mask": observed * (1 - visible),
                "regime": "fixed"}


class _Windows(Dataset):
    def __init__(self, data, cfg, pairs, masker=None):
        self.data, self.cfg, self.pairs, self.masker = data, cfg, pairs, masker

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        t0, a = self.pairs[i]
        sel = self.data.neighbourhood(a, self.cfg.n_neighbours)
        assert sel[0] == a
        return make_sample(self.data, self.cfg, sel, t0, np.random.default_rng(0), self.masker)


def _ensembles(model, cfg, out, q, sel_q, n_samples):
    """Samples (n, B, Q_sel, P) in transformed space for the selected queries."""
    pick = lambda a: a[:, sel_q]
    s = mixture_sample(pick(out["logits"]), pick(out["mu"]), pick(out["log_sigma"]), n_samples)
    mu_q, sd_q = model.query_stats(out["mean"], out["std"], {k: v[:, sel_q] for k, v in q.items()})
    return s * sd_q.unsqueeze(0) + mu_q.unsqueeze(0)


def _pooled(models, cfg, batch, q, sel_q, n_samples):
    """Equal-weight mixture of the models' predictive distributions: each model
    contributes ceil(n_samples / M) samples. (n, B, Q_sel, P) transformed."""
    k = -(-n_samples // len(models))
    return torch.cat([_ensembles(m, cfg, m(batch, q), q, sel_q, k) for m in models], 0)


def _as_list(models):
    return list(models) if isinstance(models, (list, tuple)) else [models]


def _to_targets(cfg, samp_slots):
    """samp_slots: (n, B, Ct, ...) transformed -> dict of ug/m3 ensembles."""
    tc = cfg.target_channels
    out = {"PM10": torch.expm1(samp_slots[:, :, tc.index("PM10")].clamp(max=LOG1P_MAX)).clamp(min=0)}
    if "PM_FRAC" in tc:
        pm25 = reconstruct_pm25(samp_slots[:, :, tc.index("PM10")], samp_slots[:, :, tc.index("PM_FRAC")])
    else:
        pm25 = samp_slots[:, :, tc.index("PM2.5")]
    out["PM2.5"] = torch.expm1(pm25.clamp(max=LOG1P_MAX)).clamp(min=0)
    return out


def pm25_gt_pm10(acc):
    """Share of point forecasts (all windows and steps) with PM2.5 > PM10, and the same share in the
    observed data where both are measured, for reference. acc: per-target lists of (n, Th) arrays."""
    a = {c: {k: np.concatenate(acc[c][k]) for k in ("pred", "true", "mask")} for c in TARGETS}
    p, q = a["PM2.5"], a["PM10"]
    both = (p["mask"] > 0.5) & (q["mask"] > 0.5)
    return {"forecast": float((p["pred"] > q["pred"]).mean()),
            "observed": float((p["true"] > q["true"])[both].mean()),
            "n_observed": int(both.sum())}


@torch.no_grad()
def evaluate(models, cfg, data, raw_values, raw_observed, test_start, device, batch_size=16,
             n_samples=200, limit_origins=None, workers=0):
    """models: one AQFM or a list of them (pooled ensemble)."""
    models = _as_list(models)
    for m in models:
        m.eval()
    T = data.T
    Tc, Th, Ct, H, P = (cfg.context_steps, cfg.horizon_steps, cfg.n_target_channels,
                        cfg.n_horizon_patches, cfg.patch_len)
    assert Th >= max(HORIZONS.values()), f"model horizon {Th} steps < 72 h protocol"
    origins = np.arange(test_start, T - max(HORIZONS.values()), 96)
    if limit_origins:
        origins = origins[:limit_origins]
    pairs = [(int(o - Tc), a) for o in origins for a in range(data.S)]
    R = {n: i for i, n in enumerate(RAW_CHANNELS)}

    acc = {c: {"pred": [], "true": [], "mask": [], "crps": [], "pit": [], "c90": [], "c95": []}
           for c in TARGETS}
    dl = DataLoader(_Windows(data, cfg, pairs), batch_size=batch_size, num_workers=workers)
    for bi, batch in enumerate(dl):
        t0s, anchors = batch["t0"].numpy(), batch["sel"][:, 0].numpy()
        batch = to_device(batch, device)
        q, _, _ = build_queries(cfg, batch, impute=False)
        sel_q = torch.arange(Ct * H, device=device)            # anchor = station 0 -> first Ct*H queries
        samp = _pooled(models, cfg, batch, q, sel_q, n_samples)  # (n,B,Ct*H,P)
        n, B = samp.shape[:2]
        ens = _to_targets(cfg, samp.reshape(n, B, Ct, H * P))
        for c in TARGETS:
            e = ens[c].float()                                   # (n,B,Th)
            y = np.stack([raw_values[a, R[c], t0 + Tc:t0 + Tc + Th] for a, t0 in zip(anchors, t0s)])
            m = np.stack([raw_observed[a, R[c], t0 + Tc:t0 + Tc + Th] for a, t0 in zip(anchors, t0s)])
            yt = torch.from_numpy(np.asarray(y, np.float32)).to(device)
            qs = torch.quantile(e, torch.tensor([0.025, 0.05, 0.5, 0.95, 0.975], device=device), dim=0)
            a = acc[c]
            a["pred"].append(qs[2].cpu().numpy()); a["true"].append(y); a["mask"].append(m)
            a["crps"].append(sample_crps(e, yt).cpu().numpy())
            a["pit"].append((e < yt).float().mean(0).cpu().numpy())
            a["c90"].append(((yt >= qs[1]) & (yt <= qs[3])).float().cpu().numpy())
            a["c95"].append(((yt >= qs[0]) & (yt <= qs[4])).float().cpu().numpy())
        if bi % 50 == 0:
            print(f"  forecast batch {bi + 1}/{len(dl)}")

    res = {"AQFM": {}}
    extras = {}
    for c in TARGETS:
        a = {k: np.concatenate(v) for k, v in acc[c].items()}   # (n_windows, Th)
        res["AQFM"][c] = {}
        for hz, h in HORIZONS.items():
            j = h - 1
            res["AQFM"][c][hz] = compute_metrics(a["true"][:, j], a["pred"][:, j], a["mask"][:, j])
        j = HORIZONS["24h"] - 1
        if c == "PM2.5":
            res["AQFM"][c]["24h_aqi"] = aqi_stratified(a["true"][:, j], a["pred"][:, j], a["mask"][:, j])
        mk = a["mask"] > 0.5
        e = {"crps_by_hour": [float(a["crps"][:, k:k + 4][mk[:, k:k + 4]].mean()) for k in range(0, Th, 4)]}
        for hz, h in HORIZONS.items():
            j = h - 1
            sel = mk[:, j]
            e[hz] = {"crps": float(a["crps"][sel, j].mean()),
                     "pit_hist": np.histogram(a["pit"][sel, j], bins=10, range=(0, 1))[0].tolist(),
                     "coverage90": float(a["c90"][sel, j].mean()),
                     "coverage95": float(a["c95"][sel, j].mean())}
        if c == "PM2.5":
            e["episodes_gt250"] = {k: episode_detection(a["true"][:, j], a["pred"][:, j], a["mask"][:, j])
                                   for k, j in EPISODE_STEPS.items()}
        extras[c] = e
    if "PM_FRAC" not in cfg.target_channels:     # PM2.5 predicted directly: PM2.5 > PM10 is possible
        extras["pm25_gt_pm10"] = pm25_gt_pm10(acc)
    res["AQFM_extras"] = extras
    res["n_origins"], res["n_windows"] = int(len(origins)), int(len(pairs))
    return res


@torch.no_grad()
def evaluate_imputation(models, cfg, data, raw_values, raw_observed, test_start, device,
                        batch_size=16, n_samples=200, limit_origins=None, gap_steps=96):
    """Hide the anchor's CPCB channels for 24 h ending 24 h before the origin;
    score the imputed PM2.5/PM10 against truth and against linear
    interpolation of the anchor's own visible series."""
    models = _as_list(models)
    for m in models:
        m.eval()
    Tc, P, Ct = cfg.context_steps, cfg.patch_len, cfg.n_target_channels
    start = Tc - 2 * gap_steps
    assert start % P == 0 and gap_steps % P == 0
    n_pos = gap_steps // P
    icfg = dataclasses.replace(cfg, max_impute_positions=max(cfg.max_impute_positions, n_pos))
    origins = np.arange(test_start, data.T - max(HORIZONS.values()), 96)
    if limit_origins:
        origins = origins[:limit_origins]
    pairs = [(int(o - Tc), a) for o in origins for a in range(data.S)]
    R = {n: i for i, n in enumerate(RAW_CHANNELS)}
    dl = DataLoader(_Windows(data, icfg, pairs, _FixedBlock(start, gap_steps)), batch_size=batch_size)
    err = {c: {"aqfm": [], "linear": []} for c in TARGETS}
    for batch in dl:
        t0s, anchors = batch["t0"].numpy(), batch["sel"][:, 0].numpy()
        batch = to_device(batch, device)
        q, _, _ = build_queries(icfg, batch, forecast=False)
        pos = batch["imp_pos"]                                   # (B,Pm,2), anchor patches first
        Pm = pos.shape[1]
        samp = _pooled(models, icfg, batch, q, torch.arange(Pm * Ct, device=device), n_samples)
        n, B = samp.shape[:2]
        ens = _to_targets(icfg, samp.reshape(n, B, Pm, Ct, P).transpose(2, 3))  # (n,B,Pm,P)
        for c in TARGETS:
            med = ens[c].median(0).values.cpu().numpy()          # (B,Pm,P)
            for b in range(B):
                a, t0 = anchors[b], t0s[b]
                valid = batch["imp_valid"][b].cpu().numpy()
                series = np.asarray(raw_values[a, R[c], t0:t0 + Tc], np.float64)
                obs = np.asarray(raw_observed[a, R[c], t0:t0 + Tc]) > 0
                vis = obs.copy()
                vis[start:start + gap_steps] = False
                for k in np.flatnonzero(valid):
                    s_, n_ = pos[b, k].tolist()
                    if s_ != 0:
                        continue
                    idx = np.arange(n_ * P, (n_ + 1) * P)
                    tgt = obs[idx]
                    if not tgt.any() or vis.sum() < 2:
                        continue
                    lin = np.interp(idx, np.flatnonzero(vis), series[vis])
                    err[c]["aqfm"].append(np.abs(med[b, k][tgt] - series[idx][tgt]))
                    err[c]["linear"].append(np.abs(lin[tgt] - series[idx][tgt]))
    return {c: {k: {"MAE": float(np.concatenate(v).mean()) if v else float("nan"),
                    "n": int(sum(len(x) for x in v))} for k, v in d.items()}
            for c, d in err.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, nargs="+", help="several = pooled equal-weight ensemble")
    ap.add_argument("--data", default="data/preprocessed")
    ap.add_argument("--out", default="results/aqfm_results.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--limit-origins", type=int, default=None, help="debug only")
    ap.add_argument("--no-imputation", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="seeds the sampling, so results are repeatable")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    loaded = [load_checkpoint(p, device) for p in args.ckpt]
    models = [m for m, _, _ in loaded]
    cfg = loaded[0][1]
    for p, (_, c, _) in zip(args.ckpt, loaded):
        assert dataclasses.asdict(c) == dataclasses.asdict(cfg), f"{p}: config differs from {args.ckpt[0]}"
    a = load_preprocessed(args.data)
    val_start, test_start = split_indices(np.asarray(a["timestamps_h"]))
    assert all(ck["test_start"] == test_start for _, _, ck in loaded), \
        "split mismatch between training and evaluation"
    data = AQData(a["values"], a["observed"], a["timestamps_h"], a["coords"], a["static"],
                  train_end=val_start, era5_met=a["era5_met"] if cfg.use_future_met else None)
    res = evaluate(models, cfg, data, a["values"], a["observed"], test_start, device,
                   args.batch_size, args.samples, args.limit_origins)
    if not args.no_imputation:
        res["imputation_24h_outage"] = evaluate_imputation(
            models, cfg, data, a["values"], a["observed"], test_start, device,
            args.batch_size, args.samples, args.limit_origins)
    paths = [os.path.abspath(p) for p in args.ckpt]
    res["checkpoint"] = paths[0] if len(paths) == 1 else paths
    # code that ran this evaluation, and the code each model was trained with (None for
    # checkpoints saved before this was recorded)
    res["code"] = code_version()
    train_code = [ck.get("code") for _, _, ck in loaded]
    res["train_code"] = train_code[0] if len(train_code) == 1 else train_code
    res["seed"] = args.seed
    res["args"] = vars(args)          # every option this evaluation ran with
    res["cfg"] = dataclasses.asdict(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, default=str)
    for c in TARGETS:
        print(c, {hz: res["AQFM"][c][hz]["MAE"] for hz in HORIZONS})
    print("saved", args.out)


if __name__ == "__main__":
    main()
