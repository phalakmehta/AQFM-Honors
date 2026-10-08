"""Pretraining mask sampler.

Pure random patch masking produces a model that is excellent at imputation and
mediocre at forecasting, because there is always future context on both sides
of every gap. The mixture below fixes that:

  tail (50%)     mask the last k patches, no future context  -> forecasting
  block (35%)    station-wide outages with durations drawn from the empirical
                 distribution                                -> the real failure mode
  channel (15%)  drop whole channels for a window            -> forecast PM2.5
                                                                from met + neighbours

The block regime matters because the EDA shows P(PM10 missing | PM2.5 missing)
= 92.8%: failures are station-wide power/network events, not independent sensor
glitches, so the mask must be station-wide too.
"""

from typing import Dict, Tuple

import numpy as np


class MaskSampler:
    def __init__(self, cfg, rng: np.random.Generator = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    # -- empirical outage length ------------------------------------------
    def _block_len(self, size=1):
        cfg = self.cfg
        return np.clip(
            self.rng.lognormal(cfg.block_len_lognormal_mu,
                               cfg.block_len_lognormal_sigma, size),
            2, cfg.context_steps).astype(int)

    @staticmethod
    def fit_block_lognormal(mask: np.ndarray) -> Tuple[float, float]:
        """Estimate the outage-length distribution from real data.

        mask: (T,) 1 = observed. Call this once on the training split and put
        the result in the config -- do not guess the parameters.
        """
        m = np.asarray(mask).astype(bool)
        gaps, run = [], 0
        for v in m:
            if v:
                if run:
                    gaps.append(run)
                run = 0
            else:
                run += 1
        if run:
            gaps.append(run)
        if len(gaps) < 10:
            return 3.0, 1.1
        lg = np.log(np.array(gaps, dtype=float))
        return float(lg.mean()), float(lg.std() + 1e-3)

    # -- the three regimes -------------------------------------------------
    def _tail(self, shape):
        S, C, T = shape
        visible = np.ones(shape, dtype=np.float32)
        k = self.rng.integers(self.cfg.patch_len,
                              self.cfg.horizon_steps + 1)
        visible[..., T - k:] = 0.0
        target = 1.0 - visible
        return visible, target

    def _block(self, shape):
        S, C, T = shape
        visible = np.ones(shape, dtype=np.float32)
        lo, hi = self.cfg.n_blocks_per_sample
        for s in range(S):
            for _ in range(self.rng.integers(lo, hi + 1)):
                L = int(self._block_len()[0])
                start = int(self.rng.integers(0, max(1, T - L)))
                visible[s, :, start:start + L] = 0.0   # station-wide, all channels
        target = 1.0 - visible
        return visible, target

    def _channel(self, shape):
        S, C, T = shape
        visible = np.ones(shape, dtype=np.float32)
        lo, hi = self.cfg.channel_dropout_frac
        n_drop = max(1, int(self.rng.uniform(lo, hi) * C))
        for s in range(S):
            chans = self.rng.choice(C, size=n_drop, replace=False)
            L = int(self.rng.integers(self.cfg.patch_len, max(self.cfg.patch_len + 1, T // 2)))
            start = int(self.rng.integers(0, max(1, T - L)))
            stop = min(T, start + L)
            visible[s, chans[:, None], np.arange(start, stop)[None, :]] = 0.0
        target = 1.0 - visible
        return visible, target

    # -- entry point -------------------------------------------------------
    def sample(self, observed: np.ndarray) -> Dict[str, np.ndarray]:
        """observed: (S, C, T) real observation mask (1 = a value exists).

        Returns
          input_mask  : what the encoder may see
          target_mask : what the loss is computed on (real values only)
          regime      : which regime was drawn
        """
        cfg = self.cfg
        p = np.array([cfg.p_tail_mask, cfg.p_block_mask, cfg.p_channel_dropout])
        regime = self.rng.choice(["tail", "block", "channel"], p=p / p.sum())
        fn = {"tail": self._tail, "block": self._block, "channel": self._channel}[regime]
        visible, target = fn(observed.shape)

        # Never ask the model to predict something that was never measured.
        input_mask = observed * visible
        target_mask = observed * target
        return {"input_mask": input_mask.astype(np.float32),
                "target_mask": target_mask.astype(np.float32),
                "regime": regime}
