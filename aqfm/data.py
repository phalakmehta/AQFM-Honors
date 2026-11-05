"""Data store, windowing, neighbourhood sampling and query construction.

A sample is a *neighbourhood* (the k nearest stations to an anchor) over one
window of context + horizon. Indexing conventions match baselines/: a
window starting at t0 has context [t0, t0+Tc) and horizon [t0+Tc, t0+Tc+Th);
the forecast origin is t = t0+Tc, and horizon step j is index t+j, i.e. the
baselines' "horizon h" target t+h-1 is horizon step j = h-1.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CH, ERA5_MET_CHANNELS, LOG_CHANNELS, RAW_CHANNELS, AQFMConfig
from .masking import MaskSampler
from .physical import derive_inputs
from .model import FORECAST, IMPUTE


def haversine_km(a, b):
    la1, lo1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    la2, lo2 = np.radians(b[:, 0])[None, :], np.radians(b[:, 1])[None, :]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


_MET_LOG = tuple(ERA5_MET_CHANNELS.index(c) for c in ("TOT_RF", "BLH"))
_MET_BLH = ERA5_MET_CHANNELS.index("BLH")


class AQData:
    """Transformed arrays shared by all datasets. Built once.

    values/observed: raw (S, 18, T) as saved by preprocess_aqfm.py (may be
    memory-mapped; they are never modified). era5_met: (S, 8, T) or None.
    train_end: index of the first non-training step. Every statistic stored
    here (RevIN fallback, met standardisation) uses [0, train_end) only.
    """

    def __init__(self, values, observed, timestamps_h, coords, static, train_end,
                 era5_met=None, met_stats=None, group=None, group_names=None,
                 stats_stations=None):
        """met_stats: optional (mean, std) of the met channels, from the
        checkpoint being transferred. A Delhi model must see Mumbai's ERA5 in
        Delhi's units, so for zero-shot / fine-tuning pass the checkpoint's
        stats instead of refitting them on the new city.
        group: optional (S,) int city id per station (multi-city training).
        stats_stations: optional (S,) bool; the RevIN fallback and met stats
        are fitted on these stations only (spatial hold-out: the hidden
        stations' data must not shape the model). None = all stations."""
        x, m = derive_inputs(values, observed)
        for c in LOG_CHANNELS:
            x[:, c] = np.log1p(np.clip(x[:, c], 0, None)) * m[:, c]
        self.x, self.m = x, m.astype(np.uint8)
        self.t = np.asarray(timestamps_h, dtype=np.float64)
        self.coords = np.asarray(coords, dtype=np.float32)
        self.static = np.asarray(static, dtype=np.float32)
        self.dist = haversine_km(self.coords, self.coords)
        self.train_end = int(train_end)
        self.S, self.C, self.T = x.shape
        fit = slice(None) if stats_stations is None else np.flatnonzero(stats_stations)

        # global per-channel stats on the training split (RevIN fallback)
        self.global_mean = np.zeros(self.C, np.float32)
        self.global_std = np.ones(self.C, np.float32)
        for c in range(self.C):
            mk = self.m[fit, c, :train_end].astype(bool)
            if mk.any():
                vals = self.x[fit, c, :train_end][mk]
                self.global_mean[c], self.global_std[c] = vals.mean(), max(vals.std(), 1e-3)

        self.group = (np.zeros(self.S, np.int64) if group is None
                      else np.asarray(group, np.int64))
        self.group_names = list(group_names) if group_names is not None else ["all"]
        self.met, self.met_stats = None, None
        if era5_met is not None:
            met = np.array(era5_met, dtype=np.float32)       # copy; caller's array untouched
            for i in _MET_LOG:
                met[:, i] = np.log1p(np.clip(met[:, i], 0, None))
            if met_stats is None:
                mu = np.nanmean(met[fit, :, :train_end], axis=(0, 2))
                sd = np.nanstd(met[fit, :, :train_end], axis=(0, 2))
            else:
                mu, sd = (np.asarray(a, np.float32) for a in met_stats)
            self.met_stats = (mu.astype(np.float32), sd.astype(np.float32))
            met = (met - mu[None, :, None]) / np.maximum(sd, 1e-3)[None, :, None]
            self.met_nan_frac = float(np.isnan(met).mean())
            self.met = np.nan_to_num(met, nan=0.0)            # 0 = training mean

    def neighbourhood(self, anchor, k, r_max=None, allowed=None):
        """Up to k nearest stations, anchor first (even if another station
        shares its coordinates). r_max (km): drop neighbours farther than this,
        so the result can be shorter than k (make_sample pads and masks).
        allowed: optional (S,) bool; stations outside it are never neighbours
        (spatial hold-out)."""
        order = np.argsort(self.dist[anchor], kind="stable")
        order = order[order != anchor]
        if allowed is not None:
            order = order[allowed[order]]
        if r_max is not None:
            order = order[self.dist[anchor, order] <= r_max]
        return np.concatenate([[anchor], order])[:k]


def _patch_mean(a, P):
    S, C, T = a.shape
    return a.reshape(S, C, T // P, P).mean(-1)


def make_sample(data: AQData, cfg: AQFMConfig, sel, t0, rng=None, masker=None):
    """One training/eval sample. If `masker` is given (pre-training), the
    context is corrupted and hidden-but-observed cells become imputation
    targets; otherwise it is a pure forecasting sample."""
    Tc, Th, P, H, N = (cfg.context_steps, cfg.horizon_steps, cfg.patch_len,
                       cfg.n_horizon_patches, cfg.n_context_patches)
    # Pad short neighbourhoods (radius cutoff, neighbour dropout) to a fixed
    # n_neighbours slots: padded slots repeat the anchor index but are fully
    # unobserved, so they are never attention keys, never targets, and are
    # masked out of the decoder memory via st_valid.
    sel = np.asarray(sel, np.int64)
    n_real = len(sel)
    if n_real < cfg.n_neighbours:
        sel = np.concatenate([sel, np.full(cfg.n_neighbours - n_real, sel[0])])
    st_valid = np.arange(len(sel)) < n_real
    sl = slice(t0, t0 + Tc + Th)
    x = data.x[sel, :, sl]                                     # (S,C,W) fancy index -> copy
    m = data.m[sel, :, sl].astype(np.float32)
    m[~st_valid] = 0.0
    x[~st_valid] = 0.0
    if not cfg.use_blh:                                        # no-BLH ablation: never observed
        x[:, CH["BLH"]] = 0.0
        m[:, CH["BLH"]] = 0.0
    t = data.t[sl]
    tgt = list(cfg.target_index)
    S = len(sel)

    m_in = m[..., :Tc].copy()
    impute = np.zeros_like(m_in)
    if masker is not None:
        out = masker.sample(m[..., :Tc], rng)
        m_in, impute = out["input_mask"], out["impute_mask"]

    hz_mask = m[:, tgt, Tc:]
    if masker is not None and not cfg.forecast_targets_always and out["regime"] != "tail":
        hz_mask = np.zeros_like(hz_mask)

    # imputation positions: (station, context patch) with any hidden target cell
    pmax = cfg.max_impute_positions
    imp_pos = np.zeros((pmax, 2), np.int64)
    imp_valid = np.zeros(pmax, bool)
    imp_y = np.zeros((pmax, len(tgt), P), np.float32)
    imp_m = np.zeros((pmax, len(tgt), P), np.float32)
    imp_phys = np.zeros((pmax, 3, 2, P), np.float32)           # [PM2.5, NO, NO2] x [y, mask]
    if impute.any():
        it = impute[:, tgt].reshape(S, len(tgt), N, P)
        cand = np.argwhere(it.sum(axis=(1, 3)) > 0)            # (n_cand, 2) [s, n]
        if len(cand) > pmax:
            cand = cand[rng.choice(len(cand), pmax, replace=False)]
        k = len(cand)
        imp_pos[:k], imp_valid[:k] = cand, True
        cs, cn = cand[:, 0][:, None], cand[:, 1][:, None]       # (k,1) each
        xs = x[..., :Tc].reshape(S, -1, N, P)
        imp_y[:k] = xs[cs, np.array(tgt)[None, :], cn]          # (k, Ct, P)
        imp_m[:k] = it[cs, np.arange(len(tgt))[None, :], cn]
        ip = impute.reshape(S, -1, N, P)
        for j, c in enumerate(("PM2.5", "NO", "NO2")):
            imp_phys[:k, j, 0] = xs[cand[:, 0], CH[c], cand[:, 1]]
            imp_phys[:k, j, 1] = ip[cand[:, 0], CH[c], cand[:, 1]]

    hz_on = float(hz_mask.any() or masker is None)              # off in the random-masking ablation
    phys_hz = np.stack([np.stack([x[:, CH[c], Tc:], m[:, CH[c], Tc:] * hz_on])
                        for c in ("PM2.5", "NO", "NO2")])

    t_ctx_patch = t[:Tc].reshape(N, P).mean(-1)
    t_hz_patch = t[Tc:].reshape(H, P).mean(-1)
    origin = t[Tc]
    met_ctx = met_hz = np.zeros((S, cfg.n_met_future, 1), np.float32)
    if data.met is not None:
        met = data.met[sel, :, sl]
        if not cfg.use_blh:
            met[:, _MET_BLH] = 0.0                             # standardised: 0 = training mean
        met_ctx, met_hz = _patch_mean(met[..., :Tc], P), _patch_mean(met[..., Tc:], P)

    f = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    return {
        "x_ctx": f(x[..., :Tc] * m_in), "m_ctx": f(m_in),
        "t_ctx_patch": f(t_ctx_patch), "t_hz_patch": f(t_hz_patch),
        "lead_ctx": f(t_ctx_patch - origin), "lead_hz": f(t_hz_patch - origin),
        "coords": f(data.coords[sel]), "static": f(data.static[sel]),
        "met_ctx": f(met_ctx), "met_hz": f(met_hz),
        "y_hz": f(x[:, tgt, Tc:]), "m_hz": f(hz_mask),
        "phys_hz": f(phys_hz.astype(np.float32)),              # (3, 2, S, Th)
        "imp_pos": f(imp_pos), "imp_valid": f(imp_valid),
        "imp_y": f(imp_y), "imp_m": f(imp_m), "imp_phys": f(imp_phys),
        "sel": f(np.asarray(sel, np.int64)), "t0": torch.tensor(t0),
        "st_valid": f(st_valid),
    }


class TrainWindows(Dataset):
    """Random (window start, anchor station) draws from [lo, hi]. Randomness
    is a pure function of (seed, epoch, index), so it is reproducible and safe
    across DataLoader workers. Call set_epoch() each epoch."""

    def __init__(self, data: AQData, cfg: AQFMConfig, lo, hi, n_samples, pretrain, seed=0,
                 allowed=None, balance_groups=True, group_lo=None):
        """allowed: (S,) bool, stations usable as anchors or neighbours (False =
        spatial hold-out, never seen in training). balance_groups: draw the
        city uniformly first, then a station in it, so a 37-station Delhi does
        not drown a 25-station Mumbai. group_lo: optional {group id: earliest
        window start} to limit how much history a city contributes (the
        data-efficiency curve for fine-tuning)."""
        assert hi >= lo, (lo, hi)
        self.data, self.cfg, self.lo, self.hi = data, cfg, lo, hi
        self.n, self.seed, self.epoch = n_samples, seed, 0
        self.masker = MaskSampler(cfg) if pretrain else None
        self.allowed = np.ones(data.S, bool) if allowed is None else np.asarray(allowed, bool)
        self.group_lo = group_lo or {}
        groups = np.unique(data.group[self.allowed])
        self.anchor_sets = ([np.flatnonzero(self.allowed & (data.group == g)) for g in groups]
                            if balance_groups else [np.flatnonzero(self.allowed)])

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        if len(self.anchor_sets) == 1 and not self.group_lo:
            # one anchor pool, no history limit: the v1 draw order (t0, then
            # anchor), so a v1 run sees exactly the windows it saw before
            t0 = int(rng.integers(self.lo, self.hi + 1))
            anchor = int(rng.choice(self.anchor_sets[0]))
        else:
            gi = int(rng.integers(len(self.anchor_sets)))
            anchor = int(rng.choice(self.anchor_sets[gi]))
            lo = max(self.lo, self.group_lo.get(int(self.data.group[anchor]), self.lo))
            t0 = int(rng.integers(lo, self.hi + 1))
        cfg = self.cfg
        sel = self.data.neighbourhood(anchor, cfg.n_neighbours, cfg.neighbour_radius_km, self.allowed)
        if self.group_lo:
            # a city with limited history is not a neighbour in earlier windows either
            ok = np.array([t0 >= self.group_lo.get(int(g), self.lo) for g in self.data.group[sel[1:]]], bool)
            sel = np.concatenate([sel[:1], sel[1:][ok]])
        if cfg.nbr_dropout_p > 0 and len(sel) > 1 and rng.random() < cfg.nbr_dropout_p:
            keep = rng.random(len(sel) - 1) >= rng.uniform(*cfg.nbr_dropout_frac)
            sel = np.concatenate([sel[:1], sel[1:][keep]])     # nearest-first order kept
        return make_sample(self.data, cfg, sel, t0, rng, self.masker)


class FixedWindows(Dataset):
    """Deterministic list of (window start, anchor) pairs: validation/test.
    allowed: optional (S,) bool, the same mask as training: with a spatial
    hold-out, hidden stations must not appear in any slot, or they would
    still influence which checkpoint is selected."""

    def __init__(self, data: AQData, cfg: AQFMConfig, pairs, allowed=None):
        self.data, self.cfg, self.pairs = data, cfg, list(pairs)
        self.allowed = None if allowed is None else np.asarray(allowed, bool)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        t0, anchor = self.pairs[i]
        sel = self.data.neighbourhood(int(anchor), self.cfg.n_neighbours,
                                      self.cfg.neighbour_radius_km, self.allowed)
        return make_sample(self.data, self.cfg, sel, int(t0))


# ---------------------------------------------------------------------------
def build_queries(cfg: AQFMConfig, batch, forecast=True, impute=True):
    """Assemble decoder queries and their aligned targets.

    Forecast queries: dense (station, target slot, horizon patch) grid, in that
    order. Imputation queries: every target slot at each sampled (station,
    context patch) position. Returns (q, y, ym) with q a dict of (B,Q) tensors,
    y/ym (B,Q,P) transformed targets and their masks.
    """
    B, S = batch["x_ctx"].shape[:2]
    Ct, H, P = cfg.n_target_channels, cfg.n_horizon_patches, cfg.patch_len
    dev = batch["x_ctx"].device
    parts = []
    if forecast:
        s = torch.arange(S, device=dev).view(S, 1, 1).expand(S, Ct, H).reshape(-1)
        c = torch.arange(Ct, device=dev).view(1, Ct, 1).expand(S, Ct, H).reshape(-1)
        h = torch.arange(H, device=dev).view(1, 1, H).expand(S, Ct, H).reshape(-1)
        Q = s.numel()
        met = batch["met_hz"].permute(0, 1, 3, 2)                      # (B,S,H or 1,M)
        hm = h if met.shape[2] == H else torch.zeros_like(h)
        parts.append({
            "station": s.expand(B, Q), "slot": c.expand(B, Q),
            "lead": batch["lead_hz"][:, h], "time": batch["t_hz_patch"][:, h],
            "task": torch.full((B, Q), FORECAST, device=dev),
            "met": met[:, s, hm],                                      # (B,Q,M)
            "valid": (batch["st_valid"][:, s] if "st_valid" in batch
                      else torch.ones(B, Q, dtype=torch.bool, device=dev)),
            "y": batch["y_hz"].reshape(B, S, Ct, H, P).reshape(B, Q, P),
            "ym": batch["m_hz"].reshape(B, S, Ct, H, P).reshape(B, Q, P),
        })
    if impute:
        pos = batch["imp_pos"]                                          # (B,Pm,2)
        Pm = pos.shape[1]
        s = pos[..., 0].unsqueeze(-1).expand(B, Pm, Ct).reshape(B, -1)
        n = pos[..., 1].unsqueeze(-1).expand(B, Pm, Ct).reshape(B, -1)
        c = torch.arange(Ct, device=dev).view(1, 1, Ct).expand(B, Pm, Ct).reshape(B, -1)
        met = batch["met_ctx"].permute(0, 1, 3, 2)                     # (B,S,N or 1,M)
        nm = n if met.shape[2] > 1 else torch.zeros_like(n)
        bidx = torch.arange(B, device=dev).view(B, 1)
        parts.append({
            "station": s, "slot": c,
            "lead": torch.gather(batch["lead_ctx"], 1, n),
            "time": torch.gather(batch["t_ctx_patch"], 1, n),
            "task": torch.full_like(s, IMPUTE),
            "met": met[bidx, s, nm],                                   # (B,Q,M)
            "valid": batch["imp_valid"].unsqueeze(-1).expand(B, Pm, Ct).reshape(B, -1),
            "y": batch["imp_y"].reshape(B, Pm * Ct, P),
            "ym": batch["imp_m"].reshape(B, Pm * Ct, P),
        })
    q = {k: torch.cat([p[k] for p in parts], 1) for k in parts[0]}
    y, ym = q.pop("y"), q.pop("ym")
    ym = ym * q["valid"].unsqueeze(-1)
    return q, y, ym


# ---------------------------------------------------------------------------
def synthetic_raw(n_stations=8, n_steps=3000, seed=0):
    """Physically flavoured toy data in the exact raw layout of
    preprocess_aqfm.py, for tests only. Returns values, observed, t, coords,
    static, era5_met."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * 0.25
    coords = np.stack([28.6 + rng.normal(0, 0.1, n_stations),
                       77.2 + rng.normal(0, 0.1, n_stations)], 1).astype(np.float32)
    static = rng.normal(0, 1, (n_stations, 4)).astype(np.float32)
    R = {n: i for i, n in enumerate(RAW_CHANNELS)}
    v = np.zeros((n_stations, len(RAW_CHANNELS), n_steps), np.float32)
    for s in range(n_stations):
        tt = t + rng.integers(0, 24) * 0.25
        pm25 = np.clip(120 + 70 * np.sin(2 * np.pi * tt / 24 - 1) + 90 * np.sin(2 * np.pi * tt / 8766)
                       + rng.normal(0, 18, n_steps), 5, None)
        v[s, R["PM2.5"]] = pm25
        v[s, R["PM10"]] = pm25 / rng.uniform(0.45, 0.75)
        nox = np.clip(0.35 * pm25 + rng.normal(0, 12, n_steps), 5, None)       # ppb
        f = rng.uniform(0.3, 0.7)
        v[s, R["NOx"]] = nox
        v[s, R["NO"]] = nox * f / (24.45 / 30.006)
        v[s, R["NO2"]] = nox * (1 - f) / (24.45 / 46.0055)
        v[s, R["Ozone"]] = np.clip(60 - 0.15 * pm25 + 30 * np.sin(2 * np.pi * tt / 24), 1, None)
        v[s, R["SO2"]] = np.clip(12 + rng.normal(0, 4, n_steps), 0.5, None)
        v[s, R["NH3"]] = np.clip(30 + rng.normal(0, 8, n_steps), 1, None)
        v[s, R["CO"]] = np.clip(1.2 + 0.004 * pm25 + rng.normal(0, .2, n_steps), .05, None)
        v[s, R["AT"]] = 25 - 0.03 * pm25 + 6 * np.sin(2 * np.pi * tt / 24 - 2)
        v[s, R["RH"]] = np.clip(60 + 0.05 * pm25, 5, 100)
        v[s, R["WS"]] = np.clip(1.8 + rng.gamma(2, .6, n_steps), .1, None)
        v[s, R["WD"]] = (200 + 60 * np.sin(2 * np.pi * tt / 168)) % 360
        v[s, R["SR"]] = np.clip(600 * np.sin(np.pi * ((tt - 6) % 24) / 12), 0, None)
        v[s, R["BP"]] = 740 + rng.normal(0, 2, n_steps)
        v[s, R["Benzene"]] = np.clip(2 + rng.normal(0, .8, n_steps), 0, None)
        v[s, R["BLH"]] = np.clip(800 + 600 * np.sin(2 * np.pi * (tt - 8) / 24), 50, None)
    o = np.ones_like(v)
    for s in range(n_stations):
        for _ in range(rng.integers(2, 6)):
            L = int(np.clip(rng.lognormal(3.0, 1.1), 2, 400))
            st = int(rng.integers(0, n_steps - L))
            o[s, :9, st:st + L] = 0.0
            o[s, R["Benzene"], st:st + L] = 0.0
    v *= o
    u = -v[:, R["WS"]] * np.sin(np.deg2rad(v[:, R["WD"]]))
    w = -v[:, R["WS"]] * np.cos(np.deg2rad(v[:, R["WD"]]))
    era5 = np.stack([v[:, R["AT"]], v[:, R["RH"]], u, w, v[:, R["SR"]], v[:, R["BP"]],
                     v[:, R["TOT_RF"]], v[:, R["BLH"]]], 1).astype(np.float32)
    return v, o, t, coords, static, era5
