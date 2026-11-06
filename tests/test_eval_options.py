"""Evaluation options added for the paper (2026-11-06): --targets (Ozone, NOx, NO2), --save-forecasts,
--test-end, --met-noise; and train.py --v1-spatial. Each is off by default and must not change any
existing number.  Run from the repo root:  python -m pytest tests -q
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aqfm import AQFM, AQData, AQFMConfig, synthetic_raw  # noqa: E402
from aqfm.evaluate import evaluate  # noqa: E402

ORIGIN = 2400


def _setup(use_future_met=True):
    cfg = AQFMConfig.small(n_horizon_patches=18, n_context_patches=32, use_future_met=use_future_met)
    v, o, t, c, s, e = synthetic_raw(8, 3000 + 4 * 96, seed=0)
    d = AQData(v, o, t, c, s, train_end=1500, era5_met=e if use_future_met else None)
    torch.manual_seed(0)
    model = AQFM(cfg).eval()
    model.revin.set_global_stats(d.global_mean, d.global_std)
    return cfg, d, v, o, model


def _run(cfg, d, v, o, model, **kw):
    torch.manual_seed(1)
    return evaluate(model, cfg, d, v, o, test_start=ORIGIN, device=torch.device("cpu"), batch_size=4,
                    n_samples=20, limit_origins=2, anchors=[0, 3], **kw)


def _same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)     # NaN == NaN in json


def test_extra_targets_and_saved_forecasts_leave_pm_unchanged(tmp_path):
    cfg, d, v, o, model = _setup()
    base = _run(cfg, d, v, o, model)
    npz = tmp_path / "f.npz"
    full = _run(cfg, d, v, o, model, targets=("PM2.5", "PM10", "Ozone", "NOx", "NO2"), save_forecasts=str(npz))
    for c in ("PM2.5", "PM10"):
        assert _same(base["AQFM"][c], full["AQFM"][c]) and _same(base["AQFM_extras"][c], full["AQFM_extras"][c])
    for c in ("Ozone", "NOx", "NO2"):
        assert np.isfinite(full["AQFM"][c]["24h"]["MAE"]) and full["AQFM"][c]["24h"]["n"] > 0
    f = np.load(npz)
    assert len(f["anchor"]) == full["n_windows"] == 4 and set(f["anchor"]) == {0, 3}
    assert (np.unique(f["origin"]) == ORIGIN + 96 * np.arange(2)).all()
    for c in ("PM2.5", "NO2"):                                  # the saved arrays give the reported numbers
        j = 95
        m = f[f"{c}_mask"][:, j] > 0.5
        mae = np.abs(f[f"{c}_pred"][:, j] - f[f"{c}_true"][:, j])[m].mean()
        assert round(float(mae), 4) == full["AQFM"][c]["24h"]["MAE"]
        assert (f[f"{c}_lo90"] <= f[f"{c}_pred"]).all() and (f[f"{c}_pred"] <= f[f"{c}_hi90"]).all()


def test_no2_is_nox_times_the_no2_share():
    """NO2 comes from the same samples as NOx: in ppb, NO2 never exceeds NOx."""
    from aqfm.evaluate import _to_targets
    from aqfm.physical import NO2_UGM3_TO_PPB
    cfg = AQFMConfig.small()
    torch.manual_seed(2)
    out = _to_targets(cfg, torch.randn(30, 2, cfg.n_target_channels, 16))
    assert (out["NO2"] * NO2_UGM3_TO_PPB <= out["NOx"] * (1 + 1e-4) + 1e-4).all()
    assert (out["NO2"] > 0).any()


def test_test_end_limits_the_origins():
    cfg, d, v, o, model = _setup()
    torch.manual_seed(1)
    res = evaluate(model, cfg, d, v, o, test_start=ORIGIN, device=torch.device("cpu"), batch_size=4,
                   n_samples=10, anchors=[0], test_end=ORIGIN + 96 + 288 + 1)
    assert res["n_origins"] == 2          # ORIGIN, ORIGIN + 96 (same rule as the default end: origin < end - 72 h)
    torch.manual_seed(1)
    res_all = evaluate(model, cfg, d, v, o, test_start=ORIGIN, device=torch.device("cpu"), batch_size=4,
                       n_samples=10, anchors=[0])
    assert res_all["n_origins"] == len(range(ORIGIN, d.T - 288, 96)) > 2


def test_met_noise_changes_only_models_with_future_weather():
    cfg, d, v, o, model = _setup(use_future_met=True)
    base = _run(cfg, d, v, o, model)
    assert _same(base["AQFM"], _run(cfg, d, v, o, model, met_noise=0.0)["AQFM"])
    assert not _same(base["AQFM"], _run(cfg, d, v, o, model, met_noise=1.0)["AQFM"])
    cfg, d, v, o, model = _setup(use_future_met=False)
    base = _run(cfg, d, v, o, model)
    assert _same(base["AQFM"], _run(cfg, d, v, o, model, met_noise=1.0)["AQFM"])


def test_v1_spatial_switches_off_every_v2_spatial_change(tmp_path, monkeypatch):
    from aqfm import train
    for extra, want in (([], "relative"), (["--v1-spatial"], "delhi")):
        out = tmp_path / ("v1s" if extra else "v2")
        monkeypatch.setattr(sys, "argv", ["train", "--synthetic", "--small", "--preset", "city", "--steps", "2",
                                          "--batch-size", "2", "--val-windows", "4", "--val-every", "2",
                                          "--device", "cpu", "--out", str(out), *extra])
        train.main()
        c = json.load(open(out / "config.json"))["cfg"]
        assert c["coord_mode"] == want
        assert (c["d_model"], c["n_axial_blocks"]) == (64, 2)       # sizes untouched by the option
        if extra:
            assert (c["neighbour_radius_km"], c["dist_kernels_km"], c["advection_feature"], c["spatial_lags"],
                    c["spatial_sink"], c["density_features"], c["nbr_dropout_p"]) == (None, [], False, 0, False,
                                                                                    False, 0.0)
