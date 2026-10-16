"""Shared pieces of the deep baselines: data, splits, windows, RevIN and scoring.

Everything follows the protocol of AQFM and baselines/, whose code is imported here and never changed:
  - train: windows whose horizon ends before 2024-01-01 (aqfm.train.split_indices);
  - validation: the same 512 fixed 2024 windows AQFM is selected on, at the same 12 stations each
    (validation_series);
  - test: one origin per day at 00:00 in 2025, origins = arange(test_start, T - 288, 96); origin t
    predicts index t + h - 1 from data up to t - 1; scored on the 15-min points where the target is
    observed, with the functions of baselines/baseline_utils.py.
"""
import inspect
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "baselines")):
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline_utils import (HORIZONS, TARGET_CHANNELS, TARGET_NAMES,  # noqa: E402
                            aqi_stratified, compute_metrics)
from aqfm.data import haversine_km  # noqa: E402
from aqfm.train import load_preprocessed, split_indices  # noqa: E402

CONTEXT = 512          # input steps (128 h), the context AQFM sees
HORIZON = 288          # output steps (72 h)
MIN_OBS = 16           # RevIN uses training-split stats below this many observed context steps (as AQFM)
STD_FLOOR = 0.1        # RevIN std is at least this x the training-split std (as AQFM)
NEIGHBOURS = 12        # stations per AQFM sample (AQFMConfig.n_neighbours)


def forward_fill(v, o):
    """Last observed value at or before each step, per (station, channel); NaN before a series' first
    observation. Causal: step k only uses steps <= k. v, o: (S, C, T)."""
    T = v.shape[-1]
    idx = np.where(o, np.arange(T), -1)
    np.maximum.accumulate(idx, axis=-1, out=idx)
    f = np.take_along_axis(v, np.maximum(idx, 0), axis=-1)
    f[idx < 0] = np.nan
    return f


def gather(a, s, t0, length):
    """a: (S, C, T) -> (B, length, C), row b = a[s[b], :, t0[b] : t0[b] + length]."""
    idx = np.asarray(t0)[:, None] + np.arange(length)
    return a[np.asarray(s)[:, None], :, idx]


class Series:
    """PM2.5 and PM10 of every station: raw ug/m3, observed mask, causal forward fill, and the
    training-split mean/std per channel (observed values before val_start only)."""

    def __init__(self, values, observed, val_start):
        self.v = np.array(values[:, TARGET_CHANNELS], np.float32)           # (S, 2, T)
        self.o = np.array(observed[:, TARGET_CHANNELS]) > 0.5
        self.v[~self.o] = 0.0                                                # unobserved cells never used
        self.ff = forward_fill(self.v, self.o)
        self.S, self.C, self.T = self.v.shape
        tr = [self.v[:, c, :val_start][self.o[:, c, :val_start]] for c in range(self.C)]
        self.mean = np.array([x.mean() for x in tr], np.float32)
        self.std = np.array([x.std() for x in tr], np.float32)

    def batch(self, s, t0, device="cpu"):
        """Windows starting at t0 for stations s: context [t0, t0 + CONTEXT), horizon after it."""
        f = lambda a, start, n, dt: torch.from_numpy(np.ascontiguousarray(gather(a, s, start, n), dt)).to(device)
        t0 = np.asarray(t0)
        return {"xf": f(self.ff, t0, CONTEXT, np.float32),                    # filled input (NaN at record start)
                "x": f(self.v, t0, CONTEXT, np.float32), "m": f(self.o, t0, CONTEXT, np.float32),
                "y": f(self.v, t0 + CONTEXT, HORIZON, np.float32),
                "ym": f(self.o, t0 + CONTEXT, HORIZON, np.float32)}


def revin_stats(x, m, gmean, gstd):
    """Per-window, per-channel mean and std from the observed context values only; the training-split
    stats when fewer than MIN_OBS are observed; std floored at STD_FLOOR x the training-split std.
    x, m: (B, L, C); gmean, gstd: (C,) tensors. Returns two (B, 1, C) tensors."""
    n = m.sum(1, keepdim=True)
    mean = (x * m).sum(1, keepdim=True) / n.clamp(min=1)
    std = (((x - mean) * m) ** 2).sum(1, keepdim=True).div(n.clamp(min=1)).sqrt()
    few = n < MIN_OBS
    mean = torch.where(few, gmean, mean)
    std = torch.maximum(torch.where(few, gstd, std), STD_FLOOR * gstd)
    return mean, std


def validation_pairs(val_start, test_start, n_stations, n_windows=512):
    """(window start, station) pairs of the fixed 2024 validation set: the recipe of aqfm/train.py
    (rng 12345, context 512, window 800), so both models are selected on the same windows.
    tests/test_deep_baselines.py checks it against aqfm/train.py's own lines."""
    W = CONTEXT + HORIZON
    rng = np.random.default_rng(12345)
    return [(int(rng.integers(val_start - CONTEXT, test_start - W + 1)), int(rng.integers(n_stations)))
            for _ in range(n_windows)]


def neighbourhood(coords, anchor, k):
    """The k nearest stations, anchor first: the rule of aqfm.data.AQData.neighbourhood."""
    c = np.asarray(coords, dtype=np.float32)
    order = np.argsort(haversine_km(c, c)[anchor], kind="stable")
    return np.concatenate([[anchor], order[order != anchor]])[:k]


def validation_series(val_start, test_start, coords, n_windows=512, k=NEIGHBOURS):
    """Every (window start, station) that AQFM's validation loss covers: each of its fixed 2024
    windows at every station of the anchor's k-station neighbourhood (AQFM: k = n_neighbours = 12).
    A single-station model validated on these sees exactly AQFM's validation targets."""
    pairs = validation_pairs(val_start, test_start, len(coords), n_windows)
    return [(t0, int(s)) for t0, a in pairs for s in neighbourhood(coords, a, k)]


def test_pairs(test_start, T, n_stations):
    """Every station at every daily 2025 origin, as (window start, station)."""
    origins = np.arange(test_start, T - max(HORIZONS.values()), 96)
    return [(int(o - CONTEXT), a) for o in origins for a in range(n_stations)], len(origins)


def score(name, y, pred, ym):
    """Metrics in the format of baselines/results/*.json. y, pred, ym: (N, HORIZON, 2) in ug/m3."""
    res = {}
    for c, ch in enumerate(TARGET_NAMES):
        res[ch] = {hz: compute_metrics(y[:, h - 1, c], pred[:, h - 1, c], ym[:, h - 1, c])
                   for hz, h in HORIZONS.items()}
        if ch == "PM2.5":
            j = HORIZONS["24h"] - 1
            res[ch]["24h_aqi"] = aqi_stratified(y[:, j, c], pred[:, j, c], ym[:, j, c])
    return {name: res}


CODE_PATHS = ("deep_baselines/*.py", "deep_baselines/*.sh", "deep_baselines/tests", "aqfm", "baselines/*.py")


def code_version():
    """Git commit of this checkout and whether the code these baselines run (this folder, plus aqfm/ and
    baselines/*.py, which they import) has uncommitted changes. None outside a git checkout."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                    check=True, timeout=10).stdout
    try:
        changed = git("status", "--porcelain", "--", *CODE_PATHS).splitlines()
        return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(changed),
                "changed": [line[3:] for line in changed]}
    except (OSError, subprocess.SubprocessError):
        return None


def aqfm_validation_recipe():
    """The source lines of aqfm/train.py that build AQFM's validation pairs (for the test)."""
    import aqfm.train
    src = inspect.getsource(aqfm.train.main).splitlines()
    i = next(k for k, line in enumerate(src) if "default_rng(12345)" in line)
    j = next(k for k in range(i, len(src)) if "range(args.val_windows)" in src[k])
    return "\n".join(line.strip() for line in src[i:j + 1])
