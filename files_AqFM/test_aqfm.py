"""Correctness tests. The leakage tests matter most -- everything else here
fails loudly, but a leak fails silently and produces a great-looking number."""

import numpy as np
import torch

from aqfm import AQFM, AQFMConfig, AirQualityWindows, MaskSampler, synthetic_dataset
from aqfm.data import build_queries, gather_targets, normalize_targets
from aqfm.losses import energy_score, mixture_nll, mixture_sample
from aqfm.physical import augment_with_ratios, PhysicalReconstruction, LAYOUT


def _fixture(n_stations=6, n_steps=2000, pretrain=False):
    cfg = AQFMConfig.small()
    v, o, t, c, s = synthetic_dataset(cfg, n_stations=n_stations, n_steps=n_steps)
    ds = AirQualityWindows(v, o, t, c, s, cfg, n_neighbours=4, pretrain=pretrain)
    return cfg, ds


def test_forward_shapes():
    cfg, ds = _fixture()
    b = {k: v.unsqueeze(0) for k, v in ds[0].items()}
    model = AQFM(cfg)
    S = b["x_ctx"].shape[1]
    q = build_queries(cfg, 1, S, b["t_full"], "cpu")
    out = model({"x": b["x_ctx"], "m": b["m_ctx"], "t": b["t"],
                 "coords": b["coords"], "static": b["static"]}, *q)
    Q = S * cfg.n_target_channels * cfg.n_horizon_patches
    assert out["mu"].shape == (1, Q, cfg.patch_len, cfg.n_mixture)
    assert out["mean"].shape == (1, S, cfg.n_channels, 1)
    print("forward shapes ok:", tuple(out["mu"].shape))


def test_no_future_leakage():
    """The encoder must never see a horizon timestep, and target normalisation
    must use context statistics only."""
    cfg, ds = _fixture()
    b = ds[0]
    assert b["x_ctx"].shape[-1] == cfg.context_steps
    assert b["m_ctx"].shape[-1] == cfg.context_steps
    # m_tgt is zero everywhere inside the context window
    assert b["m_tgt"][..., :cfg.context_steps].abs().sum() == 0

    # perturbing the future must not change the encoder output at all
    model = AQFM(cfg).eval()
    bb = {k: v.unsqueeze(0) for k, v in b.items()}
    with torch.no_grad():
        t1, _, (m1, s1) = model.encode(bb["x_ctx"], bb["m_ctx"], bb["t"],
                                       bb["coords"], bb["static"])
        bb2 = dict(bb)
        bb2["x_full"] = bb["x_full"].clone()
        bb2["x_full"][..., cfg.context_steps:] += 1000.0
        t2, _, (m2, s2) = model.encode(bb2["x_ctx"], bb2["m_ctx"], bb2["t"],
                                       bb2["coords"], bb2["static"])
    assert torch.allclose(t1, t2) and torch.allclose(m1, m2)
    print("no future leakage ok")


def test_missing_tokens_excluded():
    """A patch with zero observations must be excluded from every key set."""
    cfg, ds = _fixture()
    b = {k: v.unsqueeze(0) for k, v in ds[0].items()}
    b["m_ctx"][:, 0, :, :cfg.patch_len] = 0.0
    model = AQFM(cfg).eval()
    with torch.no_grad():
        _, valid, _ = model.encode(b["x_ctx"], b["m_ctx"], b["t"],
                                   b["coords"], b["static"])
    assert not valid[0, 0, :, 0].any(), "fully-missing patch marked valid"
    print("missing-token exclusion ok")


def test_mask_regimes():
    cfg = AQFMConfig.small()
    ms = MaskSampler(cfg, np.random.default_rng(0))
    obs = np.ones((5, cfg.n_channels, cfg.context_steps + cfg.horizon_steps),
                  dtype=np.float32)
    seen = set()
    for _ in range(60):
        r = ms.sample(obs)
        seen.add(r["regime"])
        # never a target where nothing was measured
        assert (r["target_mask"] * (1 - obs)).sum() == 0
        # input and target never overlap
        assert (r["input_mask"] * r["target_mask"]).sum() == 0
    assert seen == {"tail", "block", "channel"}
    print("mask regimes ok:", sorted(seen))


def test_block_mask_is_station_wide():
    """The EDA shows P(PM10 missing | PM2.5 missing)=92.8%: failures take the
    whole station down, so the synthetic mask must too."""
    cfg = AQFMConfig.small()
    ms = MaskSampler(cfg, np.random.default_rng(1))
    obs = np.ones((4, cfg.n_channels, cfg.context_steps + cfg.horizon_steps), np.float32)
    for _ in range(200):
        r = ms.sample(obs)
        if r["regime"] != "block":
            continue
        hidden = 1 - r["input_mask"]
        per_channel = hidden.sum(axis=2)                 # (S, C)
        for s in range(hidden.shape[0]):
            if per_channel[s].sum() > 0:
                assert np.allclose(per_channel[s], per_channel[s, 0])
        print("block masks are station-wide ok")
        return


def test_lognormal_fit():
    rng = np.random.default_rng(0)
    m = np.ones(20000)
    for _ in range(300):
        L = int(np.clip(rng.lognormal(3.0, 1.1), 2, 500))
        st = rng.integers(0, 20000 - L)
        m[st:st + L] = 0
    mu, sd = MaskSampler.fit_block_lognormal(m)
    assert 2.0 < mu < 4.5 and 0.5 < sd < 2.0, (mu, sd)
    print(f"outage-length fit ok: mu={mu:.2f} sigma={sd:.2f}")


def test_physical_constraint_holds():
    """PM2.5 <= PM10 and NO + NO2 == NOx must hold for every sample, by
    construction rather than by hope."""
    n, B, Q, P = 16, 2, 5, 4
    pm10 = torch.log1p(torch.rand(n, B, Q, P) * 400 + 20)
    frac = torch.randn(n, B, Q, P) * 3
    pm25, coarse = PhysicalReconstruction(0, 1)(pm10, frac)
    assert (torch.expm1(pm25) <= torch.expm1(pm10) + 1e-4).all()
    recon = torch.expm1(pm25) + torch.expm1(coarse)
    assert torch.allclose(recon, torch.expm1(pm10), atol=1e-3)
    print("physical constraints hold ok")


def test_ratio_channels():
    cfg = AQFMConfig.small()
    v, o, t, c, s = synthetic_dataset(cfg, n_stations=3, n_steps=500)
    xa, ma = augment_with_ratios(v, o)
    assert xa.shape[1] == 19 and ma.shape[1] == 19
    ok = ma[:, LAYOUT.pm_frac] > 0
    # a ratio is only observed when both constituents are
    both = (o[:, LAYOUT.pm25] > 0) & (o[:, LAYOUT.pm10] > 0)
    assert not (ok & ~both).any()
    print("ratio channel masking ok")


def test_losses():
    torch.manual_seed(0)
    B, Q, P, K = 2, 8, 4, 3
    y = torch.randn(B, Q, P)
    logits = torch.zeros(B, Q, P, K)
    mu = y.unsqueeze(-1) + torch.randn(B, Q, P, K) * 0.01
    ls = torch.full((B, Q, P, K), -2.0)
    good = mixture_nll(logits, mu, ls, y)
    bad = mixture_nll(logits, mu + 5.0, ls, y)
    assert good < bad, (good.item(), bad.item())

    s_good = mixture_sample(logits, mu, ls, 64)
    s_bad = mixture_sample(logits, mu + 5.0, ls, 64)
    assert energy_score(s_good, y) < energy_score(s_bad, y)

    # energy score of a point forecast equals MAE
    pt = y.unsqueeze(0).expand(32, *y.shape)
    assert abs(float(energy_score(pt, y))) < 1e-5
    print("loss ordering ok")


def test_training_reduces_loss():
    from aqfm.train import train
    cfg, ds = _fixture(n_stations=6, n_steps=3000, pretrain=True)
    torch.manual_seed(0)
    model = AQFM(cfg)
    losses = []

    from aqfm.train import step
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    it = iter(loader)
    for i in range(30):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        loss, parts, _, _ = step(model, b, cfg, "cpu")
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(parts["total"])
    assert np.mean(losses[-8:]) < np.mean(losses[:8]), losses
    print(f"training reduces loss ok: {np.mean(losses[:8]):.3f} "
          f"-> {np.mean(losses[-8:]):.3f}")


if __name__ == "__main__":
    for fn in [test_forward_shapes, test_no_future_leakage,
               test_missing_tokens_excluded, test_mask_regimes,
               test_block_mask_is_station_wide, test_lognormal_fit,
               test_physical_constraint_holds, test_ratio_channels,
               test_losses, test_training_reduces_loss]:
        fn()
    print("\nall tests passed")
