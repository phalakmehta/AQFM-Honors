"""Physically-constrained target parameterisation.

Independent heads for PM2.5 and PM10 can and do emit PM2.5 > PM10. Independent
heads for NO, NO2 and NOx can emit NO + NO2 != NOx. Both are free to fix:
predict a base quantity and a bounded fraction, then reconstruct.

    PM2.5 = f_pm  * PM10 ,   f_pm  in (0, 1)
    NO    = f_nox * NOx  ,   NO2 = (1 - f_nox) * NOx

The EDA reports r(PM2.5, PM10) = 0.84 and r(NO, NOx) = 0.90; this
parameterisation puts that structure into the model by construction instead of
asking it to be rediscovered from data.

Ratios are computed in the data pipeline and appended as extra channels, so
they get their own RevIN statistics and are predicted like any other channel.
"""

from dataclasses import dataclass

import numpy as np
import torch

EPS = 1e-6


@dataclass(frozen=True)
class ChannelLayout:
    pm25: int = 0
    pm10: int = 1
    no2: int = 2
    nox: int = 3
    no: int = 4
    o3: int = 5
    so2: int = 6
    nh3: int = 7
    co: int = 8
    # derived, appended after the 17 measured channels
    pm_frac: int = 17
    nox_frac: int = 18


LAYOUT = ChannelLayout()


def augment_with_ratios(x: np.ndarray, m: np.ndarray, layout=LAYOUT):
    """x, m: (S, C=17, T) raw concentrations and observation masks.

    Returns (S, 19, T) arrays. A ratio is observed only when both of its
    constituents are observed and the denominator is non-trivial; this keeps a
    missing PM10 from silently manufacturing a PM2.5 fraction.
    """
    S, C, T = x.shape
    xa = np.concatenate([x, np.zeros((S, 2, T), dtype=x.dtype)], axis=1)
    ma = np.concatenate([m, np.zeros((S, 2, T), dtype=m.dtype)], axis=1)

    def ratio(num_i, den_i, out_i):
        num, den = x[:, num_i], x[:, den_i]
        ok = (m[:, num_i] > 0) & (m[:, den_i] > 0) & (den > 1.0) & (num >= 0)
        r = np.clip(np.where(ok, num / np.maximum(den, EPS), 0.5), 0.01, 0.99)
        # logit transform: unbounded target, sigmoid at reconstruction time
        xa[:, out_i] = np.log(r / (1.0 - r))
        ma[:, out_i] = ok.astype(m.dtype)

    ratio(layout.pm25, layout.pm10, layout.pm_frac)
    ratio(layout.no, layout.nox, layout.nox_frac)
    return xa, ma


class PhysicalReconstruction:
    """Maps sampled base predictions back to observable concentrations.

    Operates on ensembles so that the reconstructed PM2.5 is a *distribution*
    obtained by multiplying the PM10 and fraction samples, not a point estimate
    formed from two means.
    """

    def __init__(self, base_pos: int, frac_pos: int, other_pos: int = None):
        """Positions are indices into the query axis layout, i.e. which decoder
        target slot holds the base quantity and which holds the fraction."""
        self.base_pos, self.frac_pos, self.other_pos = base_pos, frac_pos, other_pos

    def __call__(self, samples_log1p_base, samples_frac_logit):
        """Both (n, B, Q_sub, P) in *log1p concentration* / *logit* space.

        Returns the reconstructed component in log1p concentration space.
        """
        base = torch.expm1(samples_log1p_base).clamp(min=0.0)
        frac = torch.sigmoid(samples_frac_logit)
        return torch.log1p(base * frac), torch.log1p(base * (1.0 - frac))


def reconstruct_from_samples(samples, q_channel, layout=LAYOUT,
                             tgt_pm10=0, tgt_pmfrac=1, tgt_nox=2, tgt_noxfrac=3):
    """Convenience wrapper for a batch whose query layout is
    (station, target-slot, lead). `samples` is (n, B, S, n_tgt, H, P) in log1p /
    logit space; returns a dict of reconstructed observables."""
    pm10 = samples[..., tgt_pm10, :, :]
    pmf = samples[..., tgt_pmfrac, :, :]
    nox = samples[..., tgt_nox, :, :]
    nof = samples[..., tgt_noxfrac, :, :]
    pm25, pm_coarse = PhysicalReconstruction(0, 1)(pm10, pmf)
    no, no2 = PhysicalReconstruction(0, 1)(nox, nof)
    return {"PM2.5": pm25, "PM_coarse": pm_coarse, "NO": no, "NO2": no2}
