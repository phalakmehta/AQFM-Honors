"""Windowing, station-neighbourhood sampling and query construction.

A training sample is a *neighbourhood*, not a single station: k nearest stations
over one time window. That is what makes spatial cross-attention learnable, and
it is what lets the model impute a station-wide outage by borrowing from its
neighbours.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .masking import MaskSampler
from .physical import augment_with_ratios


def haversine_km(a, b):
    """a: (N,2), b: (M,2) lat/lon degrees -> (N,M) km."""
    la1, lo1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    la2, lo2 = np.radians(b[:, 0])[None, :], np.radians(b[:, 1])[None, :]
    h = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


class AirQualityWindows(Dataset):
    """Wraps a dense (n_stations, n_channels, n_steps) array.

    Real CPCB CSVs are ragged; align them to a common 15-minute index once,
    offline, and memory-map the result. Do not do the alignment per batch.
    """

    def __init__(self, values, observed, timestamps_h, coords, static, cfg,
                 n_neighbours: int = 12, split_range=None, seed: int = 0,
                 pretrain: bool = False):
        self.cfg = cfg
        self.n_neighbours = min(n_neighbours, values.shape[0])
        self.pretrain = pretrain
        self.rng = np.random.default_rng(seed)
        self.masker = MaskSampler(cfg, self.rng)

        # log1p once: pollutant distributions are right-skewed and heavy-tailed
        v = np.asarray(values, dtype=np.float32)
        o = np.asarray(observed, dtype=np.float32)
        pol = slice(0, 9)
        v[:, pol] = np.log1p(np.clip(v[:, pol], 0, None))
        self.values, self.observed = augment_with_ratios(v, o)

        self.t = np.asarray(timestamps_h, dtype=np.float64)
        self.coords = np.asarray(coords, dtype=np.float32)
        self.static = np.asarray(static, dtype=np.float32)
        self.dist = haversine_km(self.coords, self.coords)

        self.win = cfg.context_steps + cfg.horizon_steps
        lo, hi = split_range or (0, self.values.shape[-1] - self.win)
        self.starts = np.arange(lo, max(lo + 1, hi))

    def __len__(self):
        return len(self.starts)

    def _neighbourhood(self):
        anchor = int(self.rng.integers(self.values.shape[0]))
        return np.argsort(self.dist[anchor])[:self.n_neighbours]

    def __getitem__(self, i):
        cfg = self.cfg
        t0 = int(self.starts[i % len(self.starts)])
        sel = self._neighbourhood()
        sl = slice(t0, t0 + self.win)

        x = self.values[sel][..., sl].copy()          # (S, C, win)
        m = self.observed[sel][..., sl].copy()
        t = self.t[sl]

        if self.pretrain:
            sampled = self.masker.sample(m)
            m_in, m_tgt = sampled["input_mask"], sampled["target_mask"]
        else:
            m_in = m.copy()
            m_in[..., cfg.context_steps:] = 0.0       # strict forecasting split
            m_tgt = m.copy()
            m_tgt[..., :cfg.context_steps] = 0.0

        return {
            "x_ctx": torch.from_numpy(x[..., :cfg.context_steps] * m_in[..., :cfg.context_steps]),
            "m_ctx": torch.from_numpy(m_in[..., :cfg.context_steps]),
            "x_full": torch.from_numpy(x),
            "m_tgt": torch.from_numpy(m_tgt),
            "t": torch.from_numpy(t[:cfg.context_steps].astype(np.float32)),
            "t_full": torch.from_numpy(t.astype(np.float32)),
            "coords": torch.from_numpy(self.coords[sel]),
            "static": torch.from_numpy(self.static[sel]),
        }


# ---------------------------------------------------------------------------
def build_queries(cfg, B, S, t_full, device):
    """Dense query grid: every (station, target channel, lead patch).

    At inference you can slice this -- ask for PM2.5 only, or 24h only -- and
    the model needs no modification. That flexibility is the whole reason for a
    query decoder rather than one head per pollutant.
    """
    H, P, Ct = cfg.n_horizon_patches, cfg.patch_len, cfg.n_target_channels
    s_idx = torch.arange(S, device=device).view(S, 1, 1).expand(S, Ct, H)
    c_idx = torch.arange(Ct, device=device).view(1, Ct, 1).expand(S, Ct, H)
    h_idx = torch.arange(H, device=device).view(1, 1, H).expand(S, Ct, H)

    q_station = s_idx.reshape(1, -1).expand(B, -1).contiguous()
    q_channel = c_idx.reshape(1, -1).expand(B, -1).contiguous()
    q_lead = h_idx.reshape(1, -1).expand(B, -1).contiguous()

    # absolute timestamp of the centre of each future patch
    fut = t_full[:, cfg.context_steps:].reshape(B, H, P).mean(-1)   # (B,H)
    q_time = fut[:, None, None, :].expand(B, S, Ct, H).reshape(B, -1)
    return q_channel, q_station, q_lead, q_time


def gather_targets(cfg, batch, q_channel, q_station):
    """Slice the horizon out of x_full into (B, Q, P) aligned with the queries."""
    B, S, C, T = batch["x_full"].shape
    H, P, Ct = cfg.n_horizon_patches, cfg.patch_len, cfg.n_target_channels
    tgt_idx = torch.as_tensor(cfg.target_index, device=batch["x_full"].device)

    y = batch["x_full"][:, :, tgt_idx, cfg.context_steps:]          # (B,S,Ct,H*P)
    mk = batch["m_tgt"][:, :, tgt_idx, cfg.context_steps:]
    y = y.reshape(B, S, Ct, H, P).reshape(B, S * Ct * H, P)
    mk = mk.reshape(B, S, Ct, H, P).reshape(B, S * Ct * H, P)
    return y, mk


def normalize_targets(y, mean, std, q_channel, q_station, cfg):
    """Targets must be normalised with the *context* statistics, never their
    own -- otherwise the scale of the future leaks into the model."""
    B, Q, P = y.shape
    tgt_idx = torch.as_tensor(cfg.target_index, device=y.device)
    c_in = tgt_idx[q_channel]
    C = mean.shape[2]
    idx = q_station * C + c_in
    mu = torch.gather(mean.squeeze(-1).reshape(B, -1), 1, idx).unsqueeze(-1)
    sd = torch.gather(std.squeeze(-1).reshape(B, -1), 1, idx).unsqueeze(-1)
    return (y - mu) / sd


# ---------------------------------------------------------------------------
def synthetic_dataset(cfg, n_stations=16, n_steps=4000, seed=0):
    """Physically flavoured toy data for shape and sanity tests only.

    Diurnal + seasonal PM with an advective lag between stations, a temperature
    anticorrelation, and station-wide block outages matching the real failure
    pattern.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * 0.25                      # hours
    coords = np.stack([28.6 + rng.normal(0, 0.12, n_stations),
                       77.2 + rng.normal(0, 0.12, n_stations)], 1)
    static = rng.normal(0, 1, (n_stations, cfg.n_static))

    v = np.zeros((n_stations, 17, n_steps), dtype=np.float32)
    for s in range(n_stations):
        lag = int(rng.integers(0, 24))
        tt = t + lag * 0.25
        base = (120 + 70 * np.sin(2 * np.pi * tt / 24 - 1.0)
                + 90 * np.sin(2 * np.pi * tt / 8760))
        pm25 = np.clip(base + rng.normal(0, 18, n_steps), 5, None)
        v[s, 0] = pm25
        v[s, 1] = pm25 / rng.uniform(0.45, 0.75) + rng.normal(0, 10, n_steps)
        v[s, 3] = np.clip(0.35 * pm25 + rng.normal(0, 12, n_steps), 1, None)   # NOx
        v[s, 4] = v[s, 3] * rng.uniform(0.3, 0.7)                              # NO
        v[s, 2] = v[s, 3] - v[s, 4]                                            # NO2
        v[s, 5] = np.clip(60 - 0.15 * pm25 + 30 * np.sin(2 * np.pi * tt / 24),
                          1, None)                                            # O3
        v[s, 6] = np.clip(12 + rng.normal(0, 4, n_steps), 0.5, None)
        v[s, 7] = np.clip(30 + rng.normal(0, 8, n_steps), 1, None)
        v[s, 8] = np.clip(1.2 + 0.004 * pm25 + rng.normal(0, .2, n_steps), .05, None)
        v[s, 9] = 25 - 0.03 * pm25 + 6 * np.sin(2 * np.pi * tt / 24 - 2.0)     # AT
        v[s, 10] = np.clip(60 + 0.05 * pm25, 5, 100)                           # RH
        v[s, 11] = np.clip(1.8 + rng.gamma(2, .6, n_steps), .1, None)          # WS
        v[s, 12] = (200 + 60 * np.sin(2 * np.pi * tt / 168)) % 360             # WD
        v[s, 13] = np.clip(300 * np.sin(np.pi * (tt % 24) / 24), 0, None)      # SR
        v[s, 14] = 1005 + rng.normal(0, 3, n_steps)
        v[s, 15] = 0.0
        v[s, 16] = np.clip(2 + rng.normal(0, .8, n_steps), 0, None)

    o = np.ones_like(v)
    for s in range(n_stations):                        # station-wide outages
        for _ in range(rng.integers(3, 8)):
            L = int(np.clip(rng.lognormal(3.0, 1.1), 2, 400))
            st = int(rng.integers(0, max(1, n_steps - L)))
            o[s, :, st:st + L] = 0.0
    return v, o, t, coords, static
