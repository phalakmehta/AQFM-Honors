"""Pre-training mask sampler.

Every regime corrupts the visible *context* only; the encoder never sees the
horizon. Hidden-but-observed context cells become imputation targets, and the
horizon is (by default) always a forecasting target, so every lead time is
trained in every sample.

  tail    (50%)  no context corruption                      -> pure forecasting
  block   (35%)  station-wide outages of the CPCB analysers, durations from the
                 empirical outage distribution             -> the real failure mode
  channel (15%)  random pollutant subsets dropped over a window
                                                            -> forecast/impute PM2.5
                                                               from met + neighbours

Met channels stay visible in every regime: they are ERA5-backed and therefore
always available at inference time.
"""

from typing import Dict, Tuple

import numpy as np

from .config import CH, OUTAGE_CHANNELS


def enforce_derived_visibility(visible):
    """A fraction may be visible only when all its constituents are. Without
    this, a hidden PM2.5 could be read back as PM10 * sigmoid(PM_FRAC).
    visible: (..., C, T), modified in place and returned."""
    visible[..., CH["PM_FRAC"], :] *= visible[..., CH["PM2.5"], :] * visible[..., CH["PM10"], :]
    visible[..., CH["NOX_FRAC"], :] *= (visible[..., CH["NO"], :] * visible[..., CH["NO2"], :]
                                        * visible[..., CH["NOx"], :])
    return visible


def fit_block_lognormal(observed_pollutants: np.ndarray, min_len: int) -> Tuple[float, float, int]:
    """Fit LogNormal(mu, sigma) to station-wide outage lengths, in steps.

    observed_pollutants: (S, C_pol, T) observation mask of the CPCB pollutant
    channels over the TRAINING split only. A step is "out" when fewer than half
    of the pollutant channels are observed. Runs shorter than `min_len` are
    ignored: a gap shorter than one patch never removes a token (the per-step
    mask inside the token already covers it), so it does not need simulating.
    """
    gaps = []
    for s in range(observed_pollutants.shape[0]):
        out = np.asarray(observed_pollutants[s]).mean(axis=0) < 0.5
        d = np.diff(np.concatenate([[0], out.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        gaps.extend((ends - starts).tolist())
    g = np.array([x for x in gaps if x >= min_len], dtype=float)
    if len(g) < 10:
        raise ValueError(f"only {len(g)} outages >= {min_len} steps; cannot fit")
    lg = np.log(g)
    return float(lg.mean()), float(lg.std()), len(g)


class MaskSampler:
    def __init__(self, cfg):
        self.cfg = cfg

    def _block_len(self, rng, T):
        L = rng.lognormal(self.cfg.block_len_lognormal_mu, self.cfg.block_len_lognormal_sigma)
        return int(np.clip(round(L), 1, T))

    def sample(self, observed: np.ndarray, rng: np.random.Generator) -> Dict:
        """observed: (S, C, T_ctx) real observation mask of the context.

        Returns input_mask (what the encoder sees) and impute_mask (hidden but
        observed cells: the imputation targets). They never overlap.
        """
        cfg = self.cfg
        S, C, T = observed.shape
        visible = np.ones((S, C, T), dtype=np.float32)
        p = np.array([cfg.p_tail_mask, cfg.p_block_mask, cfg.p_channel_dropout])
        regime = ("tail", "block", "channel")[rng.choice(3, p=p / p.sum())]
        outage = np.array(OUTAGE_CHANNELS)

        if regime == "block":
            lo, hi = cfg.n_blocks_per_sample
            for _ in range(int(rng.integers(lo, hi + 1))):
                s = int(rng.integers(S))
                L = self._block_len(rng, T)
                st = int(rng.integers(0, T - L + 1))
                visible[s, outage, st:st + L] = 0.0
        elif regime == "channel":
            lo, hi = cfg.channel_dropout_frac
            for s in range(S):
                n_drop = max(1, int(round(rng.uniform(lo, hi) * len(outage))))
                chans = rng.choice(outage, size=n_drop, replace=False)
                L = int(rng.integers(cfg.patch_len, T + 1))
                st = int(rng.integers(0, T - L + 1))
                visible[s, chans, st:st + L] = 0.0

        enforce_derived_visibility(visible)
        input_mask = observed * visible
        impute_mask = observed * (1.0 - visible)
        return {"input_mask": input_mask.astype(np.float32),
                "impute_mask": impute_mask.astype(np.float32),
                "regime": regime}
