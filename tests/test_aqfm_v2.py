"""AQFM v2 tests (spatial scale, multi-city, fine-tuning).  Run from the repo root:
python -m pytest tests -q

The v1 tests are in test_aqfm.py. With no v2 option set, nothing may change.
"""

import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aqfm import AQFM, AQData, AQFMConfig, MaskSampler, TrainWindows, make_sample, synthetic_raw  # noqa: E402
from aqfm.config import CH, OUTAGE_CHANNELS  # noqa: E402
from aqfm.train import (EPOCH, fit_outages, parse_holdout, split_indices,  # noqa: E402
                        train_months_start)


def two_cities(n=(8, 6), T=3000):
    """Synthetic stations of two cities stacked as load_multi does, ~1000 km apart."""
    raws = [synthetic_raw(k, T, seed=g) for g, k in enumerate(n)]
    v, o, c, s, e = (np.concatenate([r[i] for r in raws]) for i in (0, 1, 3, 4, 5))
    c[n[0]:] += np.array([-9.5, -4.3], np.float32)
    group = np.concatenate([np.full(k, g) for g, k in enumerate(n)])
    return v, o, raws[0][2], c, s, e, group, ["delhi", "mumbai"]


# ---- spatial hold-out ---------------------------------------------------------

def test_holdout_is_fixed_and_independent_of_other_cities():
    """The hidden stations depend only on the city (not the training seed: there is no seed
    argument), and 'delhi:3' hides the same Delhi stations whether or not Mumbai is loaded."""
    *_, group, names = two_cities()
    a1, h1 = parse_holdout("delhi:3", group, names)
    a2, h2 = parse_holdout("delhi:3", group, names)
    assert h1 == h2 and len(h1) == 3 and all(group[h1] == 0)
    _, h_one = parse_holdout("delhi:3", group[group == 0], ["delhi"])
    assert h_one == h1                                       # Delhi stations come first in the stack
    _, h_both = parse_holdout("delhi:3,mumbai:2", group, names)
    assert set(h1) <= set(h_both) and sum(group[h_both] == 1) == 2
    _, h4 = parse_holdout("delhi:4", group, names)
    assert set(h1) <= set(h4)                                # nested: 4 = the 3 plus one more
    assert parse_holdout("2,5", group, names)[1] == [2, 5]
    assert parse_holdout("", group, names)[1] == []


def test_holdout_stats_use_seen_stations_only():
    """RevIN fallback stats, met stats and the outage fit with a hold-out equal those fitted on
    the seen stations alone, and changing the hidden stations' data does not move them."""
    v, o, t, c, s, e = synthetic_raw(8, 3000, seed=0)
    o = o.copy()
    rng = np.random.default_rng(1)
    for st in range(8):                                      # station-wide outages, so the fit has data
        for _ in range(6):
            a, L = int(rng.integers(0, 1800)), int(rng.integers(16, 200))
            o[st, list(OUTAGE_CHANNELS), a:a + L] = 0
    seen = np.ones(8, bool)
    seen[[1, 6]] = False
    full = AQData(v, o, t, c, s, train_end=2000, era5_met=e)
    held = AQData(v, o, t, c, s, train_end=2000, era5_met=e, stats_stations=seen)
    only = AQData(v[seen], o[seen], t, c[seen], s[seen], train_end=2000, era5_met=e[seen])
    np.testing.assert_array_equal(held.global_mean, only.global_mean)
    np.testing.assert_array_equal(held.global_std, only.global_std)
    for a, b in zip(held.met_stats, only.met_stats):
        np.testing.assert_array_equal(a, b)
    assert fit_outages(held, 2000, 16, seen) == fit_outages(only, 2000, 16)
    # the test can fail: the hidden stations do change the all-station stats
    assert not np.array_equal(full.global_mean, held.global_mean)
    assert not np.array_equal(full.met_stats[0], held.met_stats[0])
    assert fit_outages(full, 2000, 16) != fit_outages(full, 2000, 16, seen)
    # scrambling the hidden stations' data leaves the hold-out stats unchanged
    v2, o2, e2 = v.copy(), o.copy(), e.copy()
    v2[~seen] *= 3.0
    e2[~seen] += 5.0
    for st in np.flatnonzero(~seen):
        o2[st, list(OUTAGE_CHANNELS), 100:900] = 0
    held2 = AQData(v2, o2, t, c, s, train_end=2000, era5_met=e2, stats_stations=seen)
    np.testing.assert_array_equal(held.global_mean, held2.global_mean)
    np.testing.assert_array_equal(held.met_stats[1], held2.met_stats[1])
    assert fit_outages(held, 2000, 16, seen) == fit_outages(held2, 2000, 16, seen)


def test_no_holdout_changes_no_stats():
    """stats_stations=None and an all-True mask give the v1 statistics exactly."""
    v, o, t, c, s, e = synthetic_raw(8, 3000, seed=0)
    a = AQData(v, o, t, c, s, train_end=2000, era5_met=e)
    b = AQData(v, o, t, c, s, train_end=2000, era5_met=e, stats_stations=np.ones(8, bool))
    np.testing.assert_array_equal(a.global_mean, b.global_mean)
    np.testing.assert_array_equal(a.global_std, b.global_std)
    np.testing.assert_array_equal(a.met, b.met)


# ---- v1 training windows ----------------------------------------------------------

def test_v1_training_windows_keep_v1_draws():
    """One city, no hold-out, no --train-months: every training sample is the one v1 drew
    (t0 first, then the anchor, then the masks from the same generator)."""
    v, o, t, c, s, e = synthetic_raw(8, 3000, seed=0)
    data = AQData(v, o, t, c, s, train_end=2000, era5_met=e)
    cfg = AQFMConfig.small()
    tw = TrainWindows(data, cfg, 0, 2000 - cfg.window_steps, 30, pretrain=True, seed=3)
    masker = MaskSampler(cfg)
    for i in range(30):
        rng = np.random.default_rng([3, 0, i])
        t0 = int(rng.integers(0, 2000 - cfg.window_steps + 1))
        sel = data.neighbourhood(int(rng.integers(data.S)), cfg.n_neighbours)
        want, got = make_sample(data, cfg, sel, t0, rng, masker), tw[i]
        assert int(got["t0"]) == t0 and got["sel"].tolist() == sel.tolist()
        for k in want:
            assert torch.equal(want[k], got[k]), (i, k)


# ---- --train-months ------------------------------------------------------------------

def test_train_months_start_is_exactly_n_calendar_months():
    t = 0.25 * np.arange(int((datetime(2026, 1, 1) - EPOCH).total_seconds() / 900))
    val_start, _ = split_indices(t)
    for n, first in [(1, datetime(2023, 12, 1)), (3, datetime(2023, 10, 1)),
                     (6, datetime(2023, 7, 1)), (12, datetime(2023, 1, 1)),
                     (25, datetime(2021, 12, 1))]:
        lo = train_months_start(t, val_start, n)
        assert t[lo] == (first - EPOCH).total_seconds() / 3600
        assert (val_start - lo) * 900 == (datetime(2024, 1, 1) - first).total_seconds()
    assert train_months_start(t, val_start, 100) == 0          # longer than the record


def test_train_months_limits_only_the_target_city():
    """With group_lo, the target city's data enters training only from lo onwards, as anchor
    or as neighbour, windows included; the other city keeps its full record. Holds with
    --no-balance too."""
    v, o, t, c, s, e, group, names = two_cities()
    c[8:] = c[:6] + 0.01                                     # make the cities neighbours of each other
    data = AQData(v, o, t, c, s, train_end=2000, era5_met=e, group=group, group_names=names)
    cfg = AQFMConfig.small()
    hi, lo = 2000 - cfg.window_steps, 1200
    for balance in (True, False):
        tw = TrainWindows(data, cfg, 0, hi, 400, pretrain=False, seed=0,
                          balance_groups=balance, group_lo={1: lo})
        starts = {0: [], 1: []}
        for i in range(400):
            b = tw[i]
            t0, sel = int(b["t0"]), b["sel"].numpy()[b["st_valid"].numpy()]
            starts[int(group[sel[0]])].append(t0)
            if t0 < lo:
                assert not (group[sel] == 1).any(), (balance, i)
            assert t0 + cfg.window_steps <= 2000
        assert min(starts[1]) >= lo and min(starts[0]) < lo and len(starts[1]) > 50


# ---- B5: hidden stations stay out of validation ----------------------------------------

def test_holdout_stations_never_in_validation_or_training(tmp_path, monkeypatch):
    """train.py with --holdout: no validation sample and no training sample has a hidden
    station in any slot (anchor, neighbour or padding), so they cannot steer model selection."""
    from aqfm import data as aqdata, train
    seen = {"val": [], "train": []}

    class RecFixed(aqdata.FixedWindows):
        def __getitem__(self, i):
            b = super().__getitem__(i)
            seen["val"].append(b["sel"].tolist())
            return b

    class RecTrain(aqdata.TrainWindows):
        def __getitem__(self, i):
            b = super().__getitem__(i)
            seen["train"].append(b["sel"].tolist())
            return b

    monkeypatch.setattr(train, "FixedWindows", RecFixed)
    monkeypatch.setattr(train, "TrainWindows", RecTrain)
    monkeypatch.setattr(sys, "argv", [
        "train", "--synthetic", "--small", "--steps", "4", "--batch-size", "4", "--val-windows", "64",
        "--val-every", "2", "--device", "cpu", "--out", str(tmp_path), "--holdout", "1,6"])
    train.main()
    assert len(seen["val"]) >= 64 and len(seen["train"]) >= 16
    for kind in seen:
        assert not any(s in (1, 6) for sel in seen[kind] for s in sel), kind
    import json
    assert json.load(open(tmp_path / "config.json"))["held_out_stations"] == [1, 6]


# ---- step 7: city independence ------------------------------------------------------------

def _batch(cfg, data, anchors, t0=100):
    items = [make_sample(data, cfg, data.neighbourhood(a, cfg.n_neighbours, cfg.neighbour_radius_km), t0)
             for a in anchors]
    return torch.utils.data.default_collate(items)


def _forward(model, cfg, batch):
    from aqfm import build_queries
    q, _, _ = build_queries(cfg, batch)
    with torch.no_grad():
        return model(batch, q), q


def test_city_preset_is_translation_invariant_and_v1_is_not():
    """Shift every station's longitude by the same amount (latitude unchanged): the city preset
    gives the same outputs (relative positions do not change), the v1 config does not."""
    v, o, t, c, s, e = synthetic_raw(12, 3000, seed=0)
    data = AQData(v, o, t, c, s, train_end=2000, era5_met=e)
    for cfg, invariant in [(AQFMConfig.city(), True), (AQFMConfig(), False)]:
        torch.manual_seed(0)
        model = AQFM(cfg).eval()
        batch = _batch(cfg, data, [0, 5])
        ref, _ = _forward(model, cfg, batch)
        for shift in (-4.4, 10.0):                           # towards Mumbai's longitude / far east
            moved = dict(batch, coords=batch["coords"] + torch.tensor([0.0, shift]))
            out, _ = _forward(model, cfg, moved)
            diff = max(float((out[k] - ref[k]).abs().max()) for k in ("logits", "mu", "log_sigma"))
            if invariant:
                assert diff < 1e-4, (shift, diff)
            else:
                assert diff > 1e-2, (shift, diff)


def test_padded_slots_change_nothing():
    """A station with fewer than 11 neighbours within 100 km: padded slots are never targets
    (forecast or imputation) or valid queries, and changing their values, static facts, weather
    and targets does not change the real stations' outputs or the loss."""
    from aqfm.losses import total_loss
    v, o, t, c, s, e = synthetic_raw(14, 3000, seed=0)
    c = c.copy()
    c[5:, 0] += 3.0                                           # 9 stations ~330 km away
    data = AQData(v, o, t, c, s, train_end=2000, era5_met=e)
    cfg = AQFMConfig.city()
    sel = data.neighbourhood(0, cfg.n_neighbours, cfg.neighbour_radius_km)
    assert len(sel) == 5                                      # anchor + 4 neighbours, 7 padded
    items = [make_sample(data, cfg, sel, t0, np.random.default_rng(t0), MaskSampler(cfg))
             for t0 in (100, 900)]
    batch = torch.utils.data.default_collate(items)
    pad = ~batch["st_valid"]
    assert pad.sum() == 14
    assert batch["m_ctx"][pad].sum() == 0 and batch["m_hz"][pad].sum() == 0
    st = batch["imp_pos"][..., 0]
    assert not pad.gather(1, st)[batch["imp_valid"]].any()   # no imputation target in a padded slot
    torch.manual_seed(0)
    model = AQFM(cfg).eval()
    ref, q = _forward(model, cfg, batch)
    assert not q["valid"][pad.gather(1, q["station"])].any()
    g = torch.Generator().manual_seed(1)
    noisy = {k: v.clone() for k, v in batch.items()}
    for k in ("x_ctx", "static", "met_ctx", "met_hz", "y_hz"):
        x = noisy[k]
        x[pad] = 5 * torch.randn(x[pad].shape, generator=g)
    out, q2 = _forward(model, cfg, noisy)
    real = q["valid"]
    for k in ("logits", "mu", "log_sigma"):
        torch.testing.assert_close(out[k][real], ref[k][real], rtol=0, atol=1e-6)
    from aqfm import build_queries
    losses = []
    for b in (batch, noisy):
        torch.manual_seed(3)
        qq, y, ym = build_queries(cfg, b)
        with torch.no_grad():
            o_ = model(b, qq)
            losses.append(total_loss(model, o_, qq, y, ym, b, cfg, 12 * cfg.n_target_channels
                                     * cfg.n_horizon_patches)[0])
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=1e-6)


# ---- step 7: fine-tuning --------------------------------------------------------------------

def _train(monkeypatch, out, *extra):
    from aqfm import train
    monkeypatch.setattr(sys, "argv", ["train", "--synthetic", "--small", "--steps", "3", "--batch-size", "2",
                                      "--val-windows", "4", "--val-every", "3", "--device", "cpu",
                                      "--out", str(out), *extra])
    train.main()
    return torch.load(os.path.join(out, "last.pt"), map_location="cpu", weights_only=False)


def test_finetune_freeze_keeps_frozen_weights_bit_identical(tmp_path, monkeypatch):
    from aqfm.train import load_checkpoint, set_trainable
    base = _train(monkeypatch, tmp_path / "base", "--preset", "city")
    for freeze in ("encoder", "backbone"):
        ft = _train(monkeypatch, tmp_path / freeze, "--phase", "finetune", "--init",
                    str(tmp_path / "base" / "last.pt"), "--freeze", freeze, "--seed", "1")
        model, _, _ = load_checkpoint(str(tmp_path / "base" / "last.pt"))
        set_trainable(model, freeze)
        frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
        assert frozen
        for n in frozen:
            assert torch.equal(ft["model"][n], base["model"][n]), (freeze, n)
        moved = [n for n, p in model.named_parameters()
                 if p.requires_grad and not torch.equal(ft["model"][n], base["model"][n])]
        assert moved, freeze                                   # the trainable part did train


def test_finetune_keeps_checkpoint_stats_unless_refit(tmp_path, monkeypatch):
    """Fine-tuning on new data (another synthetic city: --seed 1) keeps the checkpoint's met
    stats and RevIN fallback stats; --refit-stats refits both on the new data."""
    base = _train(monkeypatch, tmp_path / "base", "--preset", "city")
    init = str(tmp_path / "base" / "last.pt")
    keep = _train(monkeypatch, tmp_path / "keep", "--phase", "finetune", "--init", init, "--seed", "1")
    refit = _train(monkeypatch, tmp_path / "refit", "--phase", "finetune", "--init", init, "--seed", "1",
                   "--refit-stats")
    assert keep["met_stats"] == base["met_stats"]
    assert refit["met_stats"] != base["met_stats"]
    for k in ("revin.global_mean", "revin.global_std"):
        assert torch.equal(keep["model"][k], base["model"][k])
        assert not torch.equal(refit["model"][k], base["model"][k])
    assert keep["cities"] == ["city0"] and keep["init"] == os.path.abspath(init)


def test_train_months_targets_the_last_city(tmp_path, monkeypatch):
    """--train-months limits the last listed city only (group_lo has that city alone)."""
    from aqfm import data as aqdata, train
    got = {}

    class Rec(aqdata.TrainWindows):
        def __init__(self, *a, **kw):
            got.update(kw.get("group_lo") or {})
            super().__init__(*a, **kw)

    monkeypatch.setattr(train, "TrainWindows", Rec)
    _train(monkeypatch, tmp_path / "tm", "--preset", "city", "--data", "x", "y", "z", "--train-months", "1")
    assert list(got) == [2]


# ---- step 6 (B6): the two unseen-station tests -----------------------------------------------

def test_virtual_sensing_never_sees_the_anchors_own_pollutants():
    """--virtual-sensing: the anchor's pollutant readings (and the fractions built from them) are
    hidden for the whole context, so its forecast is the same whatever its past readings were.
    Without the option the forecast does depend on them (the test can fail)."""
    from aqfm.evaluate import _FixedBlock, _Windows, evaluate
    cfg = AQFMConfig.small(n_horizon_patches=18, n_context_patches=32, coord_mode="relative",
                           neighbour_radius_km=100.0, density_features=True)
    v, o, t, c, s, e = synthetic_raw(8, 3000, seed=0)
    Tc, origin = cfg.context_steps, 2400
    v2 = v.copy()
    pol = list(OUTAGE_CHANNELS)
    v2[0, pol, origin - Tc:origin] = np.random.default_rng(5).uniform(1, 900, (len(pol), Tc))
    d, d2 = (AQData(x, o, t, c, s, train_end=1500, era5_met=e) for x in (v, v2))
    b = torch.utils.data.default_collate([_Windows(d, cfg, [(origin - Tc, 0)], _FixedBlock(0, Tc))[0]])
    hidden = pol + [CH["PM_FRAC"], CH["NOX_FRAC"]]
    assert b["m_ctx"][0, 0, hidden].sum() == 0 and b["x_ctx"][0, 0, hidden].abs().sum() == 0
    other = [k for k in range(cfg.n_channels) if k not in hidden]
    assert b["m_ctx"][0, 0, other].sum() > 0                  # its weather is still there
    assert b["m_ctx"][0, 1:, pol].sum() > 0                    # neighbours keep their readings
    torch.manual_seed(0)
    model = AQFM(cfg).eval()
    model.revin.set_global_stats(d.global_mean, d.global_std)
    import json                                                # json: NaN (empty band) == NaN
    run = lambda data, vs: json.dumps(evaluate(model, cfg, data, v, o, test_start=origin,
                                               device=torch.device("cpu"), batch_size=4, n_samples=20,
                                               limit_origins=1, anchors=[0], virtual_sensing=vs)["AQFM"],
                                      sort_keys=True)
    torch.manual_seed(1)
    a = run(d, True)
    torch.manual_seed(1)
    a2 = run(d2, True)
    assert a == a2
    torch.manual_seed(1)
    f = run(d, False)
    torch.manual_seed(1)
    f2 = run(d2, False)
    assert f != f2


def test_held_out_tests_use_only_seen_neighbours_and_honest_names():
    from aqfm.evaluate import TEST_NAMES, _Windows, select_stations
    cfg = AQFMConfig.small(n_neighbours=6)
    v, o, t, c, s, e = synthetic_raw(8, 3000, seed=0)
    d = AQData(v, o, t, c, s, train_end=1500, era5_met=e)
    anchors, allowed, name = select_stations("held_out", d, [1, 6], True)
    assert anchors == [1, 6] and name == "new station that has its own history"
    assert allowed.tolist() == [i not in (1, 6) for i in range(8)]
    for a in anchors:
        sel = _Windows(d, cfg, [(0, a)], None, allowed)[0]["sel"].tolist()
        assert sel[0] == a and not ({1, 6} - {a}) & set(sel)
    assert select_stations("all", d, [1, 6], True)[1:] == (None, "forecast")
    assert TEST_NAMES["virtual_sensing"] == "virtual sensing (no readings at the station)"


# ---- step 3 (B3): station facts ----------------------------------------------------------------

def _facts_module():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_pipeline"))
    import station_facts
    return station_facts


def test_station_facts_fixed_scaling_and_size():
    """One fixed rule per fact: a station's static values do not depend on which other stations
    (or cities) are loaded, and the number of facts matches the model's n_static."""
    import pandas as pd
    sf = _facts_module()
    assert len(sf.DEFAULT_FACTS) == AQFMConfig().n_static == AQFMConfig.city().n_static
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(root, "data_pipeline", "stations", "delhi_facts.csv"))
    st = pd.read_csv(os.path.join(root, "data_pipeline", "stations", "delhi.csv"))
    assert df.station_id.tolist() == st.station_id.tolist()
    full = sf.scaled_static(df, df.station_id.tolist())
    one = sf.scaled_static(df, [df.station_id[5]])
    np.testing.assert_array_equal(one[0], full[5])
    fake = pd.DataFrame({"station_id": ["x"], "elevation_m": [1000.0], "built_frac_1km": [0.25],
                         "green_frac_1km": [0.5], "coast_km": [999.0]})
    np.testing.assert_allclose(sf.scaled_static(fake, ["x"])[0], [1.0, 0.25, 0.5, 1.0], rtol=1e-6)
    import pytest
    with pytest.raises(KeyError):
        sf.scaled_static(df, ["not_a_station"])
    with pytest.raises(ValueError):
        sf.scaled_static(fake.assign(coast_km=np.nan), ["x"])
