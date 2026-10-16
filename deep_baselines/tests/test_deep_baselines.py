"""Tests for the deep baselines. Run from the repo root:  python -m pytest deep_baselines/tests -q

As for AQFM, the leakage and protocol tests matter most: a leak fails silently and gives a great number.
"""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import common  # noqa: E402
import patchtst  # noqa: E402
from common import CONTEXT, HORIZON, Series, forward_fill, revin_stats  # noqa: E402
from aqfm.data import synthetic_raw  # noqa: E402
from aqfm.train import hours_since_epoch  # noqa: E402

TINY = ["--layers", "1", "--d-model", "16", "--heads", "2", "--ffn", "32"]


@pytest.fixture(scope="module")
def raw():
    """Synthetic data in the layout of data/preprocessed, 3 stations, dated so that the 2024 and 2025
    boundaries fall inside: ~11 days of training, all of 2024, 6 daily test origins in 2025."""
    T = 1152 + 35136 + 6 * 96 + HORIZON          # 12 days of 2023, 2024 (366 days), 6 origins + 72 h
    v, o, _, c, s, _ = synthetic_raw(3, T, seed=0)
    t = hours_since_epoch(2023, 12, 20) + np.arange(T) * 0.25
    return {"values": v, "observed": o, "timestamps_h": t, "coords": c, "static": s}


@pytest.fixture(scope="module")
def series(raw):
    val_start, _ = common.split_indices(raw["timestamps_h"])
    return Series(raw["values"], raw["observed"], val_start)


def tiny_model():
    args = patchtst.argparse.Namespace(patch_len=16, stride=8, layers=1, d_model=16, heads=2, ffn=32, dropout=0.0)
    torch.manual_seed(0)
    return patchtst.build(args).eval()


def test_forward_fill_is_causal_and_exact():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(2, 2, 300)).astype(np.float32)
    o = rng.random((2, 2, 300)) > 0.6
    o[0, 0, :5] = False
    f = forward_fill(v, o)
    assert np.isnan(f[0, 0, :5]).all()
    np.testing.assert_array_equal(f[o], v[o])
    for s, c, k in [(0, 0, 50), (1, 1, 200)]:
        last = np.flatnonzero(o[s, c, :k + 1])[-1]
        assert f[s, c, k] == v[s, c, last]
    v2, o2 = v.copy(), o.copy()
    v2[..., 150:] += 100.0
    o2[..., 150:] = ~o2[..., 150:]
    np.testing.assert_array_equal(forward_fill(v2, o2)[..., :150], f[..., :150])   # the future never moves the past


def test_windows_follow_the_protocol(series):
    """Origin t = t0 + CONTEXT: the context ends at t - 1 and horizon step h - 1 is index t + h - 1."""
    s, t0 = np.array([0, 2]), np.array([100, 4000])
    b = series.batch(s, t0)
    for k in range(2):
        t = t0[k] + CONTEXT
        np.testing.assert_array_equal(b["x"][k, -1].numpy(), series.v[s[k], :, t - 1])
        for h in (1, 4, 24, 96, 288):
            np.testing.assert_array_equal(b["y"][k, h - 1].numpy(), series.v[s[k], :, t + h - 1])
            np.testing.assert_array_equal(b["ym"][k, h - 1].numpy(), series.o[s[k], :, t + h - 1])


def test_validation_windows_are_aqfms(raw, series):
    """The validation pairs equal the ones aqfm/train.py builds (its own source lines, executed)."""
    val_start, test_start = common.split_indices(raw["timestamps_h"])
    ns = {"np": np, "val_start": val_start, "test_start": test_start, "Tc": 512, "W": 800,
          "data": SimpleNamespace(S=series.S), "args": SimpleNamespace(val_windows=512)}
    exec(common.aqfm_validation_recipe(), ns)
    assert common.validation_pairs(val_start, test_start, series.S) == ns["pairs"]
    from aqfm.config import AQFMConfig
    cfg = AQFMConfig()
    assert (cfg.context_steps, cfg.window_steps) == (CONTEXT, CONTEXT + HORIZON)
    assert cfg.n_neighbours == common.NEIGHBOURS


def test_validation_series_are_aqfms_stations():
    """Each validation window covers the same stations as in AQFM's samples (AQData.neighbourhood)."""
    from aqfm.data import AQData
    v, o, t, c, s, _ = synthetic_raw(15, 400, seed=1)
    data = AQData(v, o, t, c, s, train_end=300)
    for a in range(15):
        np.testing.assert_array_equal(common.neighbourhood(c, a, common.NEIGHBOURS),
                                      data.neighbourhood(a, common.NEIGHBOURS))
    pairs = common.validation_pairs(1000, 40000, 15, 20)
    series = common.validation_series(1000, 40000, c, 20)
    assert series == [(t0, int(k)) for t0, a in pairs for k in data.neighbourhood(a, 12)]
    assert len(series) == 20 * 12


def test_revin_uses_observed_context_only_and_falls_back():
    torch.manual_seed(0)
    x = torch.randn(3, 64, 2) * 5 + 50
    m = (torch.rand(3, 64, 2) > 0.3).float()
    m[1, :, 0] = 0
    m[1, :10, 0] = 1                                     # 10 < MIN_OBS observed -> training-split stats
    gm, gs = torch.tensor([60.0, 120.0]), torch.tensor([40.0, 80.0])
    mean, std = revin_stats(x, m, gm, gs)
    sel = m[0, :, 1] > 0
    assert float(mean[0, 0, 1]) == pytest.approx(float(x[0, sel, 1].mean()), rel=1e-5)
    assert float(mean[1, 0, 0]) == 60.0 and float(std[1, 0, 0]) == 40.0
    x2 = x.clone()
    x2[m == 0] = 1e6                                     # hidden values never enter the stats
    torch.testing.assert_close(revin_stats(x2, m, gm, gs), (mean, std))
    flat = torch.full((1, 64, 2), 7.0)
    assert torch.equal(revin_stats(flat, torch.ones(1, 64, 2), gm, gs)[1][0, 0], 0.1 * gs)   # std floor


def test_no_future_leakage(raw, series):
    """Changing every value and mask from the origin on leaves the forecast bit-identical."""
    model = tiny_model()
    g = (torch.tensor(series.mean), torch.tensor(series.std))
    s, t0 = np.array([0, 1, 2]), np.array([3000, 3000, 20000])
    out = patchtst.forward(model, series.batch(s, t0), *g)[0]
    v, o = raw["values"].copy(), raw["observed"].copy()
    for k in range(3):
        v[s[k], :, t0[k] + CONTEXT:] *= 3.0
        o[s[k], :, t0[k] + CONTEXT:] = 1.0 - o[s[k], :, t0[k] + CONTEXT:]
    val_start, _ = common.split_indices(raw["timestamps_h"])
    out2 = patchtst.forward(model, Series(v, o, val_start).batch(s, t0), *g)[0]
    assert torch.equal(out, out2)


def test_loss_ignores_unobserved_targets(series):
    model = tiny_model()
    g = (torch.tensor(series.mean), torch.tensor(series.std))
    b = series.batch(np.array([0, 1]), np.array([500, 900]))
    b["ym"][:, ::2] = 0
    out, mean, std = patchtst.forward(model, b, *g)
    e, n = patchtst.l1(out, b, mean, std)
    b2 = dict(b, y=torch.where(b["ym"] > 0, b["y"], b["y"] + 1e4))
    e2, _ = patchtst.l1(out, b2, mean, std)
    assert e.item() == e2.item() and n.item() == b["ym"].sum().item()


def test_train_then_evaluate_end_to_end(raw, tmp_path):
    """Tiny model on synthetic data through the real CLI; evaluating twice gives identical files."""
    data = tmp_path / "data"
    data.mkdir()
    for k, a in raw.items():
        np.save(data / f"{k}.npy", a)
    py, script = sys.executable, os.path.join(HERE, "patchtst.py")
    run = lambda *a: subprocess.run([py, script, *a, "--data", str(data), "--device", "cpu"],
                                    check=True, capture_output=True, text=True)
    out = tmp_path / "run"
    run("train", "--out", str(out), "--steps", "6", "--val-every", "3", "--val-windows", "8",
        "--batch-size", "4", "--warmup", "2", "--eval-batch", "8", *TINY)
    assert (out / "best.pt").exists()
    cfg = json.load(open(out / "config.json"))
    assert cfg["model"] == "PatchTST" and cfg["hf_config"]["pooling_type"] is None
    for k in ("a", "b"):
        run("evaluate", "--ckpt", str(out / "best.pt"), "--out", str(tmp_path / f"{k}.json"), "--eval-batch", "8")
    ra, rb = (json.load(open(tmp_path / f"{k}.json")) for k in ("a", "b"))
    assert ra["PatchTST"] == rb["PatchTST"]
    assert ra["n_origins"] == 6 and ra["n_windows"] == 18
    assert all(np.isfinite(ra["PatchTST"][c][h]["MAE"]) for c in ("PM2.5", "PM10") for h in ("1h", "72h"))
    assert set(ra["PatchTST"]["PM2.5"]["24h_aqi"]) == {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}
