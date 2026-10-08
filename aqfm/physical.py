"""Derived input channels and physically-constrained reconstruction.

    PM2.5 = PM10 * f_pm                     f_pm  = sigmoid(PM_FRAC)  in (0, 1)
    NO    = NOx * f_nox,  NO2 = NOx * (1 - f_nox)   (all in ppb)

CPCB reports NO and NO2 in ug/m3 but NOx in ppb. On the Delhi data,
NO_ppb + NO2_ppb = NOx_ppb holds to within 0.1% at the median, while the raw
unit sum is off by ~57%. The NO fraction is therefore computed in ppb, and only
where the identity holds (some stations violate it). A fraction is observed
only when all of its constituents are observed, so a missing PM10 never
manufactures a PM2.5 fraction.
"""

import numpy as np
import torch

from .config import CH, RAW_CHANNELS

MOLAR_VOLUME = 24.45                      # L/mol at 25 C, 1 atm (CPCB convention)
NO_UGM3_TO_PPB = MOLAR_VOLUME / 30.006
NO2_UGM3_TO_PPB = MOLAR_VOLUME / 46.0055
FRAC_CLIP = (0.01, 0.99)
NOX_IDENTITY_TOL = 0.10                   # |(NO+NO2)/NOx - 1| tolerated, ppb
MIN_DENOMINATOR = 1.0                     # ug/m3 or ppb


def _logit(r):
    r = np.clip(r, *FRAC_CLIP)
    return np.log(r / (1.0 - r))


def derive_inputs(values, observed):
    """Raw (S, 18, T) arrays in RAW_CHANNELS order -> (S, 20, T) model inputs
    in INPUT_CHANNELS order, still in physical units (log1p is applied later).

    Returns new float32 arrays; the inputs are never modified. Works station by
    station so the temporary memory stays at one station's worth.
    """
    S, C, T = values.shape
    assert C == len(RAW_CHANNELS), f"expected {len(RAW_CHANNELS)} raw channels, got {C}"
    R = {n: i for i, n in enumerate(RAW_CHANNELS)}
    x = np.zeros((S, len(CH), T), dtype=np.float32)
    m = np.zeros((S, len(CH), T), dtype=np.float32)

    for s in range(S):
        v = np.asarray(values[s], dtype=np.float32)
        o = np.asarray(observed[s], dtype=np.float32) > 0
        for name in RAW_CHANNELS:
            if name in CH:                                  # copied unchanged
                x[s, CH[name]] = np.where(o[R[name]], v[R[name]], 0.0)
                m[s, CH[name]] = o[R[name]]

        # wind: meteorological direction is where the wind comes FROM
        ok = o[R["WS"]] & o[R["WD"]]
        wd = np.deg2rad(v[R["WD"]])
        x[s, CH["U"]] = np.where(ok, -v[R["WS"]] * np.sin(wd), 0.0)
        x[s, CH["V"]] = np.where(ok, -v[R["WS"]] * np.cos(wd), 0.0)
        m[s, CH["U"]] = m[s, CH["V"]] = ok

        pm25, pm10 = v[R["PM2.5"]], v[R["PM10"]]
        ok = o[R["PM2.5"]] & o[R["PM10"]] & (pm10 > MIN_DENOMINATOR)
        with np.errstate(divide="ignore", invalid="ignore"):
            x[s, CH["PM_FRAC"]] = np.where(ok, _logit(pm25 / pm10), 0.0)
        m[s, CH["PM_FRAC"]] = ok

        no_ppb = v[R["NO"]] * NO_UGM3_TO_PPB
        no2_ppb = v[R["NO2"]] * NO2_UGM3_TO_PPB
        nox = v[R["NOx"]]
        ok = o[R["NO"]] & o[R["NO2"]] & o[R["NOx"]] & (nox > MIN_DENOMINATOR)
        with np.errstate(divide="ignore", invalid="ignore"):
            ok &= np.abs((no_ppb + no2_ppb) / nox - 1.0) <= NOX_IDENTITY_TOL
            x[s, CH["NOX_FRAC"]] = np.where(ok, _logit(no_ppb / nox), 0.0)
        m[s, CH["NOX_FRAC"]] = ok
    return x, m


# ---------------------------------------------------------------------------
# Reconstruction (torch, works on ensembles of any leading shape)
# ---------------------------------------------------------------------------
def reconstruct_pm25(log1p_pm10, frac_logit):
    """log1p(PM2.5) from log1p(PM10) and the PM fraction logit. Because the
    fraction is in (0, 1), PM2.5 <= PM10 holds for every sample."""
    pm10 = torch.expm1(log1p_pm10).clamp(min=0.0)
    return torch.log1p(pm10 * torch.sigmoid(frac_logit))


def reconstruct_no_no2(log1p_nox_ppb, frac_logit):
    """log1p(NO), log1p(NO2) in ug/m3 from log1p(NOx ppb) and the NO fraction.
    NO_ppb + NO2_ppb = NOx_ppb holds for every sample."""
    nox = torch.expm1(log1p_nox_ppb).clamp(min=0.0)
    f = torch.sigmoid(frac_logit)
    no = nox * f / NO_UGM3_TO_PPB
    no2 = nox * (1.0 - f) / NO2_UGM3_TO_PPB
    return torch.log1p(no), torch.log1p(no2)
