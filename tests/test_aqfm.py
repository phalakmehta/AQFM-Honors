"""AQFM correctness tests.  Run from the repo root:  python -m pytest tests -q

Leakage tests matter most: every other bug fails loudly, a leak fails silently
and produces a great-looking number.
"""

import dataclasses
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aqfm import (AQFM, AQData, AQFMConfig, CH, FixedWindows, MaskSampler, TrainWindows,  # noqa: E402
                  build_queries, fit_block_lognormal, make_sample, synthetic_raw)
from aqfm.config import OUTAGE_CHANNELS, RAW_CHANNELS  # noqa: E402
from aqfm.layers import wind_geometry  # noqa: E402
from aqfm.losses import (crps_mixture, mixture_nll, mixture_sample, sample_crps,  # noqa: E402
                         total_loss)
from aqfm.model import hours_since_prev_valid, last_visible  # noqa: E402
from aqfm.physical import (NO2_UGM3_TO_PPB, NO_UGM3_TO_PPB, derive_inputs,  # noqa: E402
                           reconstruct_no_no2, reconstruct_pm25)
from aqfm.train import step  # noqa: E402

R = {n: i for i, n in enumerate(RAW_CHANNELS)}
MET = [CH[c] for c in ("AT", "RH", "U", "V", "SR", "BP", "TOT_RF", "BLH")]


@pytest.fixture(scope="module")
def raw():
    return synthetic_raw(8, 3000, seed=0)


@pytest.fixture(scope="module")
def data(raw):
    v, o, t, c, s, e = raw
    return AQData(v, o, t, c, s, train_end=2000, era5_met=e)


def collate(items):
    return torch.utils.data.default_collate(items)


def n_forecast(cfg, batch):
    return batch["x_ctx"].shape[1] * cfg.n_target_channels * cfg.n_horizon_patches


# ---------------------------------------------------------------------------
# derived inputs / physics
# ---------------------------------------------------------------------------
def test_ratio_channels_are_physical_fractions(raw):
    v, o = raw[0], raw[1]
    x, m = derive_inputs(v, o)
    ok = m[:, CH["PM_FRAC"]] > 0
    frac = 1 / (1 + np.exp(-x[:, CH["PM_FRAC"]]))
    true = np.clip(v[:, R["PM2.5"]] / np.maximum(v[:, R["PM10"]], 1e-9), 0.01, 0.99)
    assert ok.mean() > 0.5
    np.testing.assert_allclose(frac[ok], true[ok], rtol=1e-4)
    okn = m[:, CH["NOX_FRAC"]] > 0
    fn = 1 / (1 + np.exp(-x[:, CH["NOX_FRAC"]]))
    true_n = np.clip(v[:, R["NO"]] * NO_UGM3_TO_PPB / v[:, R["NOx"]], 0.01, 0.99)
    assert okn.mean() > 0.5
    np.testing.assert_allclose(fn[okn], true_n[okn], rtol=1e-4)


def test_fraction_observed_only_with_constituents(raw):
    v, o = raw[0].copy(), raw[1].copy()
    o[0, R["PM10"], 100:200] = 0
    o[1, R["NO2"], 300:400] = 0
    v[2, R["NO2"], 500:600] *= 3.0                         # breaks NO + NO2 = NOx
    x, m = derive_inputs(v, o)
    assert m[0, CH["PM_FRAC"], 100:200].sum() == 0
    assert m[1, CH["NOX_FRAC"], 300:400].sum() == 0
    assert m[2, CH["NOX_FRAC"], 500:600].sum() == 0
    both = (o[:, R["PM2.5"]] > 0) & (o[:, R["PM10"]] > 0)
    assert not ((m[:, CH["PM_FRAC"]] > 0) & ~both).any()


def test_wind_components(raw):
    v, o = raw[0], raw[1]
    x, _ = derive_inputs(v, o)
    ws = np.hypot(x[:, CH["U"]], x[:, CH["V"]])
    np.testing.assert_allclose(ws, v[:, R["WS"]], rtol=1e-4, atol=1e-5)
    # wind FROM the north (WD=0) blows southward: v < 0
    vv, oo = v[:1, :, :1].copy(), o[:1, :, :1].copy()
    vv[0, R["WS"], 0], vv[0, R["WD"], 0] = 5.0, 0.0
    xx, _ = derive_inputs(vv, oo)
    assert abs(xx[0, CH["U"], 0]) < 1e-5 and xx[0, CH["V"], 0] == pytest.approx(-5.0)


def test_inputs_never_mutated_and_readonly_ok(raw):
    v, o, t, c, s, e = (a.copy() for a in raw)
    before = v.copy(), o.copy(), e.copy()
    for a in (v, o, e):
        a.flags.writeable = False                          # like np.load(mmap_mode="r")
    AQData(v, o, t, c, s, train_end=2000, era5_met=e)
    for a, b in zip((v, o, e), before):
        np.testing.assert_array_equal(a, b)


def test_physical_reconstruction_constraints():
    torch.manual_seed(0)
    pm10 = torch.log1p(torch.rand(64, 50) * 900 + 5)
    frac = torch.randn(64, 50) * 4
    pm25 = reconstruct_pm25(pm10, frac)
    assert (torch.expm1(pm25) <= torch.expm1(pm10) * (1 + 1e-5)).all()
    nox = torch.log1p(torch.rand(64, 50) * 300 + 1)
    no, no2 = reconstruct_no_no2(nox, frac)
    total_ppb = torch.expm1(no) * NO_UGM3_TO_PPB + torch.expm1(no2) * NO2_UGM3_TO_PPB
    torch.testing.assert_close(total_ppb, torch.expm1(nox), rtol=1e-4, atol=1e-3)


def test_back_transform_finite_for_extreme_samples():
    # A wide mixture component can draw log1p values past float32's expm1 limit
    # (~88.7); this gave a NaN physical loss at step 1547 of the first real run.
    from types import SimpleNamespace
    from aqfm.evaluate import _to_targets
    big = torch.tensor([20.0, 88.8, 95.0, 200.0, 95.0, 95.0])
    frac = torch.tensor([0.0, 0.0, 0.0, 0.0, -120.0, 120.0])
    assert torch.isfinite(reconstruct_pm25(big, frac)).all()
    for r in reconstruct_no_no2(big, frac):
        assert torch.isfinite(r).all()
    for tc in (AQFMConfig().target_channels, ("PM10", "PM2.5")):  # physical and no-physics paths
        samp = torch.zeros(4, 2, len(tc), big.numel())
        for c in ("PM10", "PM2.5"):
            if c in tc:
                samp[:, :, tc.index(c)] = big
        if "PM_FRAC" in tc:
            samp[:, :, tc.index("PM_FRAC")] = frac
        for v in _to_targets(SimpleNamespace(target_channels=tuple(tc)), samp).values():
            assert torch.isfinite(v).all()


# ---------------------------------------------------------------------------
# leakage
# ---------------------------------------------------------------------------
def test_horizon_perturbation_cannot_reach_encoder(raw):
    """Change every CPCB value in the horizon; the encoder input, RevIN stats
    and encoder output must be bit-identical."""
    cfg = AQFMConfig.small()
    v, o, t, c, s, e = raw
    t0, sel = 400, np.arange(cfg.n_neighbours)
    v2 = v.copy()
    v2[:, :, t0 + cfg.context_steps:] = v2[:, :, t0 + cfg.context_steps:] * 3 + 100
    d1 = AQData(v, o, t, c, s, train_end=2000, era5_met=e)
    d2 = AQData(v2, o, t, c, s, train_end=2000, era5_met=e)
    b1 = collate([make_sample(d1, cfg, sel, t0)])
    b2 = collate([make_sample(d2, cfg, sel, t0)])
    assert torch.equal(b1["x_ctx"], b2["x_ctx"]) and torch.equal(b1["m_ctx"], b2["m_ctx"])
    assert not torch.equal(b1["y_hz"], b2["y_hz"])           # the perturbation is real
    torch.manual_seed(0)
    model = AQFM(cfg).eval()
    with torch.no_grad():
        m1, st1 = model.encode(b1["x_ctx"], b1["m_ctx"], b1["t_ctx_patch"], b1["coords"], b1["static"])
        m2, st2 = model.encode(b2["x_ctx"], b2["m_ctx"], b2["t_ctx_patch"], b2["coords"], b2["static"])
    assert torch.equal(m1, m2) and torch.equal(st1[0], st2[0]) and torch.equal(st1[1], st2[1])


def test_revin_uses_visible_cells_only():
    """Normalisation statistics must come from visible context cells only:
    garbage in hidden cells cannot change them, and they equal the masked
    mean/std; series with too few visible steps fall back to the global stats."""
    from aqfm.layers import MaskedRevIN
    torch.manual_seed(0)
    rv = MaskedRevIN(3, min_obs=16)
    rv.set_global_stats([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
    x = torch.randn(2, 4, 3, 64) * 2 + 5
    m = (torch.rand(2, 4, 3, 64) > 0.3).float()
    m[0, 0, 2] = 0
    m[0, 0, 2, :5] = 1                                       # only 5 visible -> fallback
    mean1, std1 = rv.stats(x * m, m)
    mean2, std2 = rv.stats(x * m + (1 - m) * 1e4, m)        # garbage where hidden
    torch.testing.assert_close(mean1, mean2)
    torch.testing.assert_close(std1, std2)
    sel = m[1, 2, 1] > 0
    assert float(mean1[1, 2, 1]) == pytest.approx(float(x[1, 2, 1][sel].mean()), rel=1e-5)
    assert float(std1[1, 2, 1]) == pytest.approx(float(x[1, 2, 1][sel].std(unbiased=False)), rel=1e-3)
    assert float(mean1[0, 0, 2]) == 3.0 and float(std1[0, 0, 2]) == 0.5
    # a stuck (constant) sensor gets std floored at 10% of the global std
    xc, mc = torch.full((1, 1, 3, 64), 7.0), torch.ones(1, 1, 3, 64)
    assert float(rv.stats(xc, mc)[1][0, 0, 1]) == pytest.approx(0.05)


def test_hidden_context_is_zero_in_encoder_input(data):
    """Values hidden by the mask sampler (imputation targets) enter the encoder
    input as exact zeros, with visibility 0."""
    cfg = AQFMConfig.small(p_tail_mask=0.0, p_block_mask=1.0, p_channel_dropout=0.0)
    ds = TrainWindows(data, cfg, 0, 1500, 20, pretrain=True, seed=3)
    for i in range(20):
        b = ds[i]
        hidden = (b["m_ctx"] == 0)
        assert (b["x_ctx"][hidden] == 0).all()


def test_mask_sampler_regimes(data):
    cfg = AQFMConfig.small()
    ms = MaskSampler(cfg)
    rng = np.random.default_rng(0)
    obs = data.m[:6, :, :cfg.context_steps].astype(np.float32)
    seen = set()
    for _ in range(300):
        r = ms.sample(obs, rng)
        seen.add(r["regime"])
        im, tm = r["input_mask"], r["impute_mask"]
        assert (im * tm).sum() == 0                          # never overlap
        assert ((im + tm) == obs).all()                      # partition of the observed cells
        assert (tm[:, MET] == 0).all()                       # met is never hidden
        # derived visibility: hidden PM2.5 or PM10 => hidden PM_FRAC
        hid = (im[:, CH["PM2.5"]] == 0) | (im[:, CH["PM10"]] == 0)
        assert (im[:, CH["PM_FRAC"]][hid] == 0).all()
        hid = (im[:, CH["NO"]] == 0) | (im[:, CH["NO2"]] == 0) | (im[:, CH["NOx"]] == 0)
        assert (im[:, CH["NOX_FRAC"]][hid] == 0).all()
        if r["regime"] == "tail":
            assert tm.sum() == 0
    assert seen == {"tail", "block", "channel"}


def test_block_outages_are_station_wide(data):
    cfg = AQFMConfig.small(p_tail_mask=0.0, p_block_mask=1.0, p_channel_dropout=0.0)
    ms = MaskSampler(cfg)
    rng = np.random.default_rng(1)
    obs = np.ones((6, cfg.n_channels, cfg.context_steps), np.float32)
    for _ in range(50):
        hidden = 1 - ms.sample(obs, rng)["input_mask"]
        for s in range(6):
            per_t = hidden[s][list(OUTAGE_CHANNELS)]
            assert (per_t == per_t[0]).all()                 # all CPCB channels together


def test_every_lead_is_a_target_in_pretraining(data):
    cfg = AQFMConfig.small()
    ds = TrainWindows(data, cfg, 0, 1500, 30, pretrain=True, seed=0)
    for i in range(30):
        b = ds[i]
        sel, t0 = b["sel"].numpy(), int(b["t0"])
        truth = data.m[sel][:, list(cfg.target_index), t0 + cfg.context_steps:t0 + cfg.window_steps]
        np.testing.assert_array_equal(b["m_hz"].numpy(), truth)


def test_imputation_targets_are_the_hidden_truth(data):
    cfg = AQFMConfig.small(p_tail_mask=0.0, p_block_mask=0.5, p_channel_dropout=0.5)
    ds = TrainWindows(data, cfg, 0, 1500, 40, pretrain=True, seed=5)
    P, N, tgt = cfg.patch_len, cfg.n_context_patches, list(cfg.target_index)
    checked = 0
    for i in range(40):
        b = ds[i]
        sel, t0 = b["sel"].numpy(), int(b["t0"])
        x = data.x[sel][:, :, t0:t0 + cfg.context_steps].reshape(len(sel), -1, N, P)
        m_in = b["m_ctx"].numpy().reshape(len(sel), -1, N, P)
        obs = data.m[sel][:, :, t0:t0 + cfg.context_steps].reshape(len(sel), -1, N, P)
        for k in np.flatnonzero(b["imp_valid"].numpy()):
            s_, n_ = b["imp_pos"][k].tolist()
            ym = b["imp_m"][k].numpy()
            np.testing.assert_array_equal(ym, obs[s_, tgt, n_] * (1 - m_in[s_, tgt, n_]))
            np.testing.assert_allclose(b["imp_y"][k].numpy()[ym > 0], x[s_, tgt, n_][ym > 0])
            assert ym.sum() > 0 or b["imp_m"][k].sum() == 0
            checked += 1
    assert checked > 20


def test_protocol_alignment_with_baselines(data):
    """Baselines: origin t predicts index t + h - 1. Here: window t0 = t - Tc,
    horizon step j = h - 1 must be index t + h - 1."""
    cfg = AQFMConfig.small()
    origin = 1000
    b = make_sample(data, cfg, np.arange(cfg.n_neighbours), origin - cfg.context_steps)
    tgt = list(cfg.target_index)
    for h in (1, 4, 24, cfg.horizon_steps):
        np.testing.assert_array_equal(b["y_hz"][:, :, h - 1].numpy(), data.x[:cfg.n_neighbours, tgt, origin + h - 1])
    # lead of the first horizon patch = its centre minus the origin
    assert float(b["lead_hz"][0]) == pytest.approx((cfg.patch_len - 1) / 2 * 0.25)
    assert float(b["lead_ctx"][-1]) < 0


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def test_order_statistic_crps_equals_pairwise():
    torch.manual_seed(0)
    x = torch.randn(17, 5, 3, dtype=torch.float64)
    y = torch.randn(5, 3, dtype=torch.float64)
    n = x.shape[0]
    naive = (x - y).abs().mean(0) - (x[:, None] - x[None]).abs().sum((0, 1)) / (2 * n * (n - 1))
    torch.testing.assert_close(sample_crps(x, y), naive)


def test_closed_form_mixture_crps_matches_monte_carlo():
    torch.manual_seed(0)
    logits = torch.randn(6, 3, dtype=torch.float64)
    mu = torch.randn(6, 3, dtype=torch.float64)
    ls = torch.randn(6, 3, dtype=torch.float64) * 0.5
    y = torch.randn(6, dtype=torch.float64)
    exact = crps_mixture(logits, mu, ls, y)
    mc = sample_crps(mixture_sample(logits, mu, ls, 200_000), y)
    torch.testing.assert_close(exact, mc, rtol=0.02, atol=0.005)
    # single Gaussian closed form: s * (z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi))
    one = crps_mixture(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), torch.tensor([0.0]))
    assert float(one) == pytest.approx(2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi), rel=1e-6)


def test_mixture_nll_matches_torch_distributions():
    torch.manual_seed(0)
    logits, mu, ls = torch.randn(10, 3), torch.randn(10, 3), torch.randn(10, 3) * 0.3
    y = torch.randn(10)
    d = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logits),
                                              torch.distributions.Normal(mu, ls.exp()))
    torch.testing.assert_close(mixture_nll(logits, mu, ls, y), -d.log_prob(y))


def test_unobserved_targets_get_no_gradient(data):
    cfg = AQFMConfig.small()
    b = collate([make_sample(data, cfg, np.arange(cfg.n_neighbours), 300)])
    model = AQFM(cfg)
    q, y, ym = build_queries(cfg, b, impute=False)
    out = model(b, q)
    y = y.clone().requires_grad_(True)
    loss, _ = total_loss(model, out, q, y, ym, b, dataclasses.replace(cfg, w_physical=0.0),
                         n_forecast(cfg, b))
    loss.backward()
    assert (y.grad[ym == 0] == 0).all()


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def test_gap_feature():
    valid = torch.tensor([[True, False, False, False, False, True, True, False]])
    gap = hours_since_prev_valid(valid, 4.0)
    assert gap.tolist() == [[4.0, 4.0, 8.0, 12.0, 16.0, 20.0, 4.0, 4.0]]


def test_last_visible_anchor():
    xn = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 9.0, 9.0, 9.0]])
    m = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    val, age, has = last_visible(xn * m, m, 0.25)
    assert val.tolist() == [2.0, 8.0, 0.0] and has.tolist() == [True, True, False]
    assert age.tolist() == [0.75, 0.25, 1.25]


def test_anchor_cannot_see_horizon_or_hidden_cells(raw):
    """With the latest-value anchor on, every decoder output (not just the
    encoder) must be bit-identical when the horizon or hidden context changes."""
    cfg = AQFMConfig.small()
    assert cfg.use_last_obs
    v, o, t, c, s, e = raw
    t0, sel = 400, np.arange(cfg.n_neighbours)
    v2 = v.copy()
    v2[:, :, t0 + cfg.context_steps:] = v2[:, :, t0 + cfg.context_steps:] * 3 + 100
    d1 = AQData(v, o, t, c, s, train_end=2000, era5_met=e)
    d2 = AQData(v2, o, t, c, s, train_end=2000, era5_met=e)
    b1 = collate([make_sample(d1, cfg, sel, t0)])
    b2 = collate([make_sample(d2, cfg, sel, t0)])
    b2["m_ctx"][:, 0, :, -40:] = 0                       # hide the anchor's latest steps ...
    b1["m_ctx"][:, 0, :, -40:] = 0
    b1["x_ctx"] = b1["x_ctx"] * b1["m_ctx"]
    b2["x_ctx"] = b2["x_ctx"] * b2["m_ctx"] + (1 - b2["m_ctx"]) * 1e4   # ... and fill them with garbage
    torch.manual_seed(0)
    model = AQFM(cfg).eval()
    torch.nn.init.normal_(model.anchor_gate.weight, std=0.1)   # make the residual path live
    q, _, _ = build_queries(cfg, b1, impute=False)
    with torch.no_grad():
        o1, o2 = model(b1, q), model(b2, q)
    for k in ("logits", "mu", "log_sigma"):
        assert torch.equal(o1[k], o2[k]), k


def test_pre_anchor_checkpoint_loads(tmp_path):
    """Checkpoints saved before use_last_obs existed load as the old model."""
    from aqfm.train import load_checkpoint
    cfg = AQFMConfig.small(use_last_obs=False)
    d = dataclasses.asdict(cfg)
    d.pop("use_last_obs")
    torch.save({"model": AQFM(cfg).state_dict(), "cfg": d}, tmp_path / "old.pt")
    model, cfg2, _ = load_checkpoint(tmp_path / "old.pt")
    assert not cfg2.use_last_obs and not hasattr(model, "anchor_gate")


def test_code_version_records_commit():
    """Runs and results record the git commit of the code (None outside a git checkout)."""
    from aqfm.train import code_version
    v = code_version()
    assert v is None or (len(v["commit"]) == 40 and v["dirty"] == bool(v["changed"]))
    if v is not None:
        assert all(p.startswith(("aqfm/", "tests/", "baselines/")) for p in v["changed"]), v["changed"]


def test_wind_geometry_upwind_is_cos_one():
    coords = torch.tensor([[[28.60, 77.20], [28.70, 77.20], [28.60, 77.30]]])   # 0, N of 0, E of 0
    u, v = torch.zeros(1, 3, 1), torch.full((1, 3, 1), -4.0)   # blowing south = FROM the north
    dist, cos, sin, ws = wind_geometry(coords, u, v)
    assert float(cos[0, 0, 1, 0]) == pytest.approx(1.0, abs=1e-4)      # station 1 is upwind of 0
    assert float(cos[0, 1, 0, 0]) == pytest.approx(-1.0, abs=1e-4)     # 0 is downwind of 1
    assert abs(float(cos[0, 0, 2, 0])) < 1e-3 and float(sin[0, 0, 2, 0]) != 0  # crosswind
    assert float(dist[0, 0, 1]) == pytest.approx(11.1, rel=0.01)
    assert float(cos[0, 0, 0, 0]) == 0.0                                # self pair is neutral


def test_forward_shapes_and_finite(data):
    cfg = AQFMConfig.small()
    ds = TrainWindows(data, cfg, 0, 1500, 4, pretrain=True, seed=1)
    b = collate([ds[i] for i in range(4)])
    model = AQFM(cfg)
    q, y, ym = build_queries(cfg, b)
    out = model(b, q)
    S, Q = b["x_ctx"].shape[1], y.shape[1]
    assert Q == S * cfg.n_target_channels * cfg.n_horizon_patches + cfg.max_impute_positions * cfg.n_target_channels
    assert out["mu"].shape == (4, Q, cfg.patch_len, cfg.n_mixture)
    assert all(torch.isfinite(out[k]).all() for k in ("logits", "mu", "log_sigma"))


def test_fully_missing_station_is_finite(data):
    cfg = AQFMConfig.small()
    b = collate([make_sample(data, cfg, np.arange(cfg.n_neighbours), 300)])
    b["m_ctx"][:, 0] = 0
    b["x_ctx"][:, 0] = 0
    model = AQFM(cfg).eval()
    q, _, _ = build_queries(cfg, b, impute=False)
    with torch.no_grad():
        out = model(b, q)
    assert torch.isfinite(out["mu"]).all() and torch.isfinite(out["std"]).all()


def test_station_permutation_equivariance(data):
    """Nothing in the model depends on station order: permuting the stations
    of a sample permutes the forecasts identically."""
    cfg = AQFMConfig.small()
    b = collate([make_sample(data, cfg, np.arange(cfg.n_neighbours), 300)])
    perm = torch.tensor([2, 0, 3, 1])
    bp = dict(b)
    for k in ("x_ctx", "m_ctx", "coords", "static", "met_ctx", "met_hz", "y_hz", "m_hz"):
        bp[k] = b[k][:, perm]
    bp["phys_hz"] = b["phys_hz"][:, :, :, perm]
    torch.manual_seed(0)
    model = AQFM(cfg).eval()
    with torch.no_grad():
        q, _, _ = build_queries(cfg, b, impute=False)
        qp, _, _ = build_queries(cfg, bp, impute=False)
        o1, o2 = model(b, q)["mu"], model(bp, qp)["mu"]
    S, Ct, H = cfg.n_neighbours, cfg.n_target_channels, cfg.n_horizon_patches
    o1 = o1.reshape(1, S, Ct, H, -1)
    o2 = o2.reshape(1, S, Ct, H, -1)
    torch.testing.assert_close(o1[:, perm], o2, rtol=1e-4, atol=1e-5)


def test_grad_checkpoint_is_exact(data):
    cfg = AQFMConfig.small()
    b = collate([make_sample(data, cfg, np.arange(cfg.n_neighbours), 300)])
    torch.manual_seed(0)
    m1 = AQFM(cfg)
    m2 = AQFM(dataclasses.replace(cfg, grad_checkpoint=True))
    m2.load_state_dict(m1.state_dict())
    grads = []
    for m in (m1, m2):
        m.train()
        torch.manual_seed(1)                                 # physical loss samples randomly
        loss, _, _, _ = step(m, b, m.cfg, impute=False)
        m.zero_grad()
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("kw", [dict(use_future_met=False), dict(spatial_bias="distance"),
                                dict(spatial_bias="none"), dict(use_spatial_attention=False),
                                dict(use_last_obs=False), dict(use_blh=False),
                                dict(p_tail_mask=0.0, forecast_targets_always=False),
                                dict(target_channels=("PM10", "PM2.5", "NOx", "NOX_FRAC", "Ozone", "SO2", "CO", "NH3")),
                                dict(target_channels=("PM2.5", "PM10", "NOx", "Ozone"))])
def test_ablation_configs_run(raw, kw):
    cfg = AQFMConfig.small(**kw)
    v, o, t, c, s, e = raw
    d = AQData(v, o, t, c, s, train_end=2000, era5_met=e if cfg.use_future_met else None)
    ds = TrainWindows(d, cfg, 0, 1500, 2, pretrain=True, seed=0)
    b = collate([ds[0], ds[1]])
    loss, parts, _, _ = step(AQFM(cfg), b, cfg)
    loss.backward()
    assert math.isfinite(parts["total"])


def test_overfits_a_fixed_batch(data):
    """Capacity and gradient path: on one fixed batch the forecast CRPS must
    fall well below its starting value."""
    torch.manual_seed(0)
    cfg = AQFMConfig.small()
    b = collate([make_sample(data, cfg, np.arange(cfg.n_neighbours), t) for t in (200, 600)])
    model = AQFM(cfg)
    model.revin.set_global_stats(data.global_mean, data.global_std)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    hist = []
    for _ in range(150):
        loss, parts, _, _ = step(model, b, cfg, impute=False)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(parts["crps_forecast"])
    assert np.mean(hist[-10:]) < 0.6 * np.mean(hist[:5]), (hist[:5], hist[-10:])


def test_fit_block_lognormal_recovers_parameters():
    rng = np.random.default_rng(0)
    obs = np.ones((4, 3, 200_000), np.float32)
    for s in range(4):
        pos = 0
        while True:
            pos += int(rng.integers(200, 2000))
            L = int(round(rng.lognormal(3.5, 0.8)))
            if pos + L >= obs.shape[-1]:
                break
            obs[s, :, pos:pos + L] = 0
            pos += L
    mu, sd, n = fit_block_lognormal(obs, min_len=1)
    assert abs(mu - 3.5) < 0.1 and abs(sd - 0.8) < 0.1, (mu, sd, n)


def test_evaluate_pipeline_runs(raw):
    """End-to-end evaluation on synthetic data with the 72 h protocol."""
    from aqfm.evaluate import evaluate, evaluate_imputation
    cfg = AQFMConfig.small(n_horizon_patches=18, n_context_patches=32)
    v, o, t, c, s, e = raw
    d = AQData(v, o, t, c, s, train_end=1500, era5_met=e)
    model = AQFM(cfg)
    model.revin.set_global_stats(d.global_mean, d.global_std)
    res = evaluate(model, cfg, d, v, o, test_start=2400, device=torch.device("cpu"),
                   batch_size=4, n_samples=20, limit_origins=1)
    assert res["n_windows"] == d.S
    assert res["AQFM"]["PM2.5"]["24h"]["n"] > 0 and res["AQFM"]["PM10"]["72h"]["n"] > 0
    imp = evaluate_imputation(model, cfg, d, v, o, test_start=2400, device=torch.device("cpu"),
                              batch_size=4, n_samples=20, limit_origins=1)
    assert imp["PM2.5"]["aqfm"]["n"] == imp["PM2.5"]["linear"]["n"] > 0


def test_ensemble_pools_equal_shares(raw):
    """Each of M models contributes ceil(n / M) samples, and a list of models
    runs through both evaluation paths."""
    from aqfm.evaluate import _pooled, evaluate, evaluate_imputation
    cfg = AQFMConfig.small(n_horizon_patches=18, n_context_patches=32)
    v, o, t, c, s, e = raw
    d = AQData(v, o, t, c, s, train_end=1500, era5_met=e)
    models = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        m = AQFM(cfg).eval()
        m.revin.set_global_stats(d.global_mean, d.global_std)
        models.append(m)
    batch = collate([make_sample(d, cfg, d.neighbourhood(0, cfg.n_neighbours), 1000,
                                 np.random.default_rng(0))])
    q, _, _ = build_queries(cfg, batch, impute=False)
    sel = torch.arange(4)
    with torch.no_grad():
        samp = _pooled(models, cfg, batch, q, sel, 20)
        assert samp.shape[0] == 21 and torch.isfinite(samp).all()
        # the first model's share comes from the first model alone
        torch.manual_seed(5)
        alone = _pooled(models[:1], cfg, batch, q, sel, 7)
        torch.manual_seed(5)
        assert torch.equal(_pooled(models, cfg, batch, q, sel, 20)[:7], alone)
    res = evaluate(models, cfg, d, v, o, test_start=2400, device=torch.device("cpu"),
                   batch_size=4, n_samples=20, limit_origins=1)
    assert res["n_windows"] == d.S and res["AQFM"]["PM2.5"]["24h"]["n"] > 0
    imp = evaluate_imputation(models, cfg, d, v, o, test_start=2400, device=torch.device("cpu"),
                              batch_size=4, n_samples=20, limit_origins=1)
    assert imp["PM2.5"]["aqfm"]["n"] > 0


# ---------------------------------------------------------------------------
# ablation switches (Stage E)
# ---------------------------------------------------------------------------
def test_no_blh_hides_blh_everywhere(raw):
    """With use_blh=False, changing BLH (input channel and future ERA5) must
    leave the model output bit-identical; with BLH on, the same change must
    reach the output (so the test can fail)."""
    v, o, t, c, s, e = raw
    v2, e2 = v.copy(), e.copy()
    v2[:, R["BLH"]] = v2[:, R["BLH"]] * 3 + 500
    e2[:, -1] = e2[:, -1] * 3 + 500                          # ERA5_MET_CHANNELS[-1] == "BLH"
    sel, t0 = np.arange(4), 400
    outs = {}
    for use_blh in (False, True):
        cfg = AQFMConfig.small(use_blh=use_blh)
        torch.manual_seed(0)
        model = AQFM(cfg).eval()
        for name, (vv, ee) in {"orig": (v, e), "pert": (v2, e2)}.items():
            d = AQData(vv, o, t, c, s, train_end=2000, era5_met=ee)
            b = collate([make_sample(d, cfg, sel, t0)])
            q, _, _ = build_queries(cfg, b, impute=False)
            with torch.no_grad():
                outs[use_blh, name] = model(b, q)["mu"]
        if not use_blh:
            assert not b["m_ctx"][:, :, CH["BLH"]].any()
    assert torch.equal(outs[False, "orig"], outs[False, "pert"])
    assert not torch.equal(outs[True, "orig"], outs[True, "pert"])


def test_random_masking_trains_imputation_only(data):
    """Random-only masking: training samples carry imputation targets and no
    forecast targets; evaluation samples still have forecast targets."""
    cfg = AQFMConfig.small(p_tail_mask=0.0, forecast_targets_always=False)
    ds = TrainWindows(data, cfg, 0, 1500, 20, pretrain=True, seed=0)
    items = [ds[i] for i in range(20)]
    assert all(it["m_hz"].sum() == 0 and it["phys_hz"][:, 1].sum() == 0 for it in items)
    assert sum(int(it["imp_valid"].any()) for it in items) > 10
    ev = make_sample(data, cfg, np.arange(cfg.n_neighbours), 400)
    assert ev["m_hz"].sum() > 0


def test_pm25_direct_reports_violations(raw):
    """PM2.5 as a direct target: evaluation reports how often PM2.5 > PM10."""
    from aqfm.evaluate import evaluate
    tc = tuple("PM2.5" if x == "PM_FRAC" else x for x in AQFMConfig.target_channels)
    cfg = AQFMConfig.small(n_horizon_patches=18, n_context_patches=32, target_channels=tc)
    assert cfg.physical_pairs() == [("NOX", 2, 3)]
    v, o, t, c, s, e = raw
    d = AQData(v, o, t, c, s, train_end=1500, era5_met=e)
    model = AQFM(cfg)
    model.revin.set_global_stats(d.global_mean, d.global_std)
    res = evaluate(model, cfg, d, v, o, test_start=2400, device=torch.device("cpu"),
                   batch_size=4, n_samples=20, limit_origins=1)
    viol = res["AQFM_extras"]["pm25_gt_pm10"]
    assert 0.0 <= viol["forecast"] <= 1.0 and viol["n_observed"] > 0
    assert viol["observed"] == 0.0                            # synthetic PM2.5 < PM10 by construction


def test_ablation_flags_reach_the_config(tmp_path, monkeypatch):
    """--no-blh / --pm25-direct / --random-masking end up in the saved config."""
    import json
    from aqfm import train
    monkeypatch.setattr(sys, "argv", [
        "train", "--synthetic", "--small", "--steps", "1", "--batch-size", "2", "--val-windows", "2",
        "--device", "cpu", "--out", str(tmp_path), "--no-blh", "--pm25-direct", "--random-masking"])
    train.main()
    cfg = json.load(open(tmp_path / "config.json"))["cfg"]
    assert cfg["use_blh"] is False and cfg["p_tail_mask"] == 0.0 and cfg["forecast_targets_always"] is False
    assert cfg["target_channels"][:2] == ["PM10", "PM2.5"] and "PM_FRAC" not in cfg["target_channels"]
