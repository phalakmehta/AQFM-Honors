"""Tests for the E-STGCN port. Run from the repo root:  python -m pytest deep_baselines/estgcn/tests -q"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from scipy.stats import genpareto

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import estgcn  # noqa: E402
from estgcn import ESTGCN, Pollutant, adjacency, fit_gpd, loss_terms  # noqa: E402
import common  # noqa: E402
from common import CONTEXT, HORIZON, Series  # noqa: E402
from aqfm.data import synthetic_raw  # noqa: E402
from aqfm.train import hours_since_epoch  # noqa: E402

S = 6


@pytest.fixture(scope="module")
def raw():
    """Synthetic data in the layout of data/preprocessed, dated so that 2024 and 2025 fall inside
    (12 days of 2023, all of 2024, 6 daily test origins)."""
    T = 1152 + 35136 + 6 * 96 + HORIZON
    v, o, _, c, s, _ = synthetic_raw(S, T, seed=0)
    t = hours_since_epoch(2023, 12, 20) + np.arange(T) * 0.25
    return {"values": v, "observed": o, "timestamps_h": t, "coords": c, "static": s}


@pytest.fixture(scope="module")
def pol(raw):
    val_start, _ = common.split_indices(raw["timestamps_h"])
    return Pollutant(Series(raw["values"], raw["observed"], val_start), "PM2.5", val_start)


def test_adjacency_is_the_notebook_rule():
    coords = np.array([[28.60, 77.20], [28.65, 77.20], [28.70, 77.20], [28.60, 77.25], [28.80, 77.40]], np.float32)
    A = adjacency(coords)
    d = common.haversine_km(coords, coords)
    for i in range(5):
        for j in range(5):
            want = i != j and np.exp(-((d[i, j] * 1000) / 10000) ** 2 / 0.75) >= 0.5
            assert A[i, j] == want
    assert A[0, 1] == 1 and A[0, 2] == 0                       # 5.6 km linked, 11.1 km not (limit ~7.2 km)
    assert A[4].sum() == 0                                     # far station: isolated


def test_graph_conv_is_keras_segment_mean():
    """Keras: gather the neighbours' features (edges[1]) and average them per node (edges[0]); empty -> 0."""
    torch.manual_seed(0)
    A = np.zeros((5, 5), np.float32)
    A[0, [1, 2]] = A[1, 0] = A[2, [0, 3]] = A[3, 2] = 1        # node 4 isolated
    gc = estgcn.GraphConv(1, 3, A)
    x = torch.randn(2, 7, 5, 1)
    out = gc(x)
    nodes, nbrs = np.where(A == 1)
    for i in range(5):
        js = nbrs[nodes == i]
        mean = x[:, :, js].mean(2) if len(js) else torch.zeros(2, 7, 1)
        torch.testing.assert_close(out[:, :, i, 3:], mean @ gc.weight)
        torch.testing.assert_close(out[:, :, i, :3], x[:, :, i] @ gc.weight)


def their_loss(y_true, y_pred, A, B, Sig, Xi, u):
    """Their notebook's loss_fn transcribed to numpy, for one sample (H, nds), z-units."""
    L1, L2 = np.log(B), np.log(Sig)
    L3 = 1.0 + np.where(Xi != 0, 1.0 / np.where(Xi != 0, Xi, 1), 0.0)
    L4 = Xi * (A + B * y_pred) / Sig
    L = (L1 - L2) - L3 * np.log(1.0 + L4)
    LK = np.where(A + B * y_pred > u, L, 0.0)
    per_step = 0.99 * ((y_true - y_pred) ** 2).mean(-1) + 0.01 * (-LK).max(-1)
    return per_step.mean()                                     # Keras averages the per-step vector


def test_loss_equals_their_formula_for_one_sample():
    rng = np.random.default_rng(0)
    H, N = 20, 4
    mean, std = rng.uniform(80, 150, N), rng.uniform(40, 90, N)
    xi, sig = rng.uniform(0.05, 0.4, N), rng.uniform(20, 60, N)
    yz, pz = rng.normal(size=(H, N)), rng.normal(size=(H, N))
    k = {n: torch.tensor(v, dtype=torch.float64) for n, v in (("mean", mean), ("std", std), ("xi", xi), ("sigma", sig))}
    k["u"] = torch.tensor(60.0, dtype=torch.float64)
    b = {"y": torch.tensor(mean + std * yz)[None], "ym": torch.ones(1, H, N, dtype=torch.float64)}
    se, n, pen = loss_terms(torch.tensor(pz)[None], b, k)
    ours = 0.99 * se / n + 0.01 * pen / H
    assert float(ours) == pytest.approx(their_loss(yz, pz, mean, std, sig, xi, 60.0), rel=1e-10)


def test_every_sample_in_the_batch_trains():
    """The fix: their notebook scores only sample 0 of each batch; here every sample counts."""
    rng = np.random.default_rng(1)
    k = {n: torch.tensor(rng.uniform(1, 2, 3)) for n in ("mean", "std", "xi", "sigma")}
    k["u"] = torch.tensor(60.0)
    pred = torch.tensor(rng.normal(size=(2, 5, 3)), requires_grad=True)
    b = {"y": torch.tensor(rng.normal(size=(2, 5, 3))), "ym": torch.ones(2, 5, 3, dtype=torch.float64)}
    se, n, _ = loss_terms(pred, b, k)
    (se / n).backward()
    assert pred.grad[1].abs().sum() > 0


def test_unobserved_targets_are_ignored(pol):
    b = pol.windows(np.array([300, 2000]))
    b["ym"][:, ::3] = 0
    k = estgcn.consts(pol, "cpu")
    pred = torch.randn(2, HORIZON, S)
    b2 = dict(b, y=torch.where(b["ym"] > 0, b["y"], b["y"] + 1e4))
    assert loss_terms(pred, b, k)[0].item() == loss_terms(pred, b2, k)[0].item()


def test_windows_follow_the_protocol(pol):
    t0 = np.array([100, 4000])
    b = pol.windows(t0)
    for i in range(2):
        t = t0[i] + CONTEXT
        np.testing.assert_allclose(b["x"][i, -1].numpy(), (pol.ff[:, t - 1] - pol.mean) / pol.std, rtol=1e-6)
        for h in (1, 4, 96, 288):
            np.testing.assert_array_equal(b["y"][i, h - 1].numpy(), pol.v[:, t + h - 1])


def test_no_future_leakage(raw, pol):
    torch.manual_seed(0)
    model = ESTGCN(adjacency(raw["coords"]), out_feat=4, lstm_units=8).eval()
    t0 = np.array([3000, 20000])
    out = model(pol.windows(t0)["x"])
    v, o = raw["values"].copy(), raw["observed"].copy()
    for t in t0:
        v[:, :, t + CONTEXT:t + CONTEXT + HORIZON] *= 3.0
        o[:, :, t + CONTEXT:t + CONTEXT + HORIZON] = 1.0 - o[:, :, t + CONTEXT:t + CONTEXT + HORIZON]
    val_start, _ = common.split_indices(raw["timestamps_h"])
    p2 = Pollutant(Series(v, o, val_start), "PM2.5", val_start)
    assert torch.equal(out, model(p2.windows(t0)["x"]))


def test_gpd_fit_recovers_known_parameters():
    x = 60.0 + genpareto.rvs(c=0.2, scale=30.0, size=20000, random_state=0)
    xi, sigma = fit_gpd(np.concatenate([x, np.full(5000, 20.0)]), 60.0)
    assert xi == pytest.approx(0.2, abs=0.03) and sigma == pytest.approx(30.0, rel=0.05)


def test_validation_windows_are_aqfms(raw):
    val_start, test_start = common.split_indices(raw["timestamps_h"])
    w = estgcn.validation_windows(val_start, test_start, raw["coords"], 10)
    assert [(t0, s) for t0, ss in w for s in ss] == common.validation_series(val_start, test_start, raw["coords"], 10)
    assert [t0 for t0, _ in w] == [t0 for t0, _ in common.validation_pairs(val_start, test_start, S, 10)]


def test_train_then_evaluate_end_to_end(raw, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for k, a in raw.items():
        np.save(data / f"{k}.npy", a)
    py, script = sys.executable, os.path.join(HERE, "estgcn.py")
    run = lambda *a: subprocess.run([py, script, *a, "--data", str(data), "--device", "cpu"],
                                    check=True, capture_output=True, text=True)
    for t, name in (("PM2.5", "pm25"), ("PM10", "pm10")):
        run("train", "--target", t, "--out", str(tmp_path / name), "--steps", "4", "--val-every", "2",
            "--val-windows", "4", "--batch-size", "2", "--out-feat", "4", "--lstm-units", "8", "--eval-batch", "4")
    ck = [str(tmp_path / n / "best.pt") for n in ("pm25", "pm10")]
    for k in ("a", "b"):
        run("evaluate", "--ckpt", *ck, "--out", str(tmp_path / f"{k}.json"), "--eval-batch", "4")
    ra, rb = (json.load(open(tmp_path / f"{k}.json")) for k in ("a", "b"))
    assert ra["E-STGCN"] == rb["E-STGCN"]
    assert ra["n_origins"] == 6 and ra["n_windows"] == 6 * S
    assert all(np.isfinite(ra["E-STGCN"][c][h]["MAE"]) for c in ("PM2.5", "PM10") for h in ("1h", "72h"))
