"""Classical baselines: no forecast may use a value from the origin step or later."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))

import city_baselines  # noqa: E402
from baseline_utils import CHANNEL_NAMES, STEPS_PER_DAY  # noqa: E402


def test_seasonal_naive_uses_only_values_known_at_the_origin(tmp_path, monkeypatch):
    T = 40 * STEPS_PER_DAY                                      # 40 days of 15-min steps from 2018-01-01
    values = np.broadcast_to(np.arange(T, dtype=np.float32), (1, len(CHANNEL_NAMES), T)).copy()  # value = step
    np.save(tmp_path / "values.npy", values)
    np.save(tmp_path / "observed.npy", np.ones_like(values))
    np.save(tmp_path / "timestamps_h.npy", np.arange(T) * 0.25)
    out = tmp_path / "f.npz"
    monkeypatch.setattr(sys, "argv", ["city_baselines.py", "--data", str(tmp_path), "--test-start", "2018-01-31",
                                      "--targets", "PM2.5", "--save-forecasts", str(out),
                                      "--out", str(tmp_path / "r.json")])
    city_baselines.main()
    f = np.load(out)
    origin, sn, pers = f["origin"], f["PM2.5_snaive"], f["PM2.5_persistence"]
    assert len(origin) and (sn < origin[:, None]).all()        # every lead, 72 h included
    np.testing.assert_array_equal(pers, np.repeat(origin[:, None] - 1, 4, 1))
    # the latest value at the target's time of day known at the origin: 1 h and 6 h one day back; 24 h and 72 h
    # land on the step just before the origin (origins are at midnight)
    np.testing.assert_array_equal(sn[:, 0], origin + 3 - STEPS_PER_DAY)
    np.testing.assert_array_equal(sn[:, 1], origin + 23 - STEPS_PER_DAY)
    np.testing.assert_array_equal(sn[:, 2], origin - 1)
    np.testing.assert_array_equal(sn[:, 3], origin - 1)


def test_xgboost_lags_from_last_reading():
    import city_xgboost
    import run_baselines_xgboost as rx
    T, S = 2000, 1
    values = np.broadcast_to(np.arange(T, dtype=np.float32), (S, len(CHANNEL_NAMES), T)).copy()   # value = step
    feats = {k: np.zeros(T, np.int64) for k in ("hour", "dow", "month", "doy")}
    t = np.arange(1000, 1010)
    published = list(rx.LAG_STEPS)
    try:
        X, *_ = rx.build_features_vectorized(values, np.ones_like(values), feats, 0, 0, t, 96)
        np.testing.assert_array_equal(X[:, :8], t[:, None] - np.array(published))          # newest: t - 4
        city_xgboost.lags_from_last()
        X, *_ = rx.build_features_vectorized(values, np.ones_like(values), feats, 0, 0, t, 96)
        np.testing.assert_array_equal(X[:, :8], t[:, None] - np.array(published) + 3)      # newest: t - 1
        assert X[:, 0].tolist() == (t - 1).tolist()
    finally:
        rx.LAG_STEPS = published


def test_as_of_fills_inputs_with_the_latest_reading_only():
    import city_xgboost
    v = np.arange(12, dtype=np.float32).reshape(1, 1, 12)
    o = np.array([[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]], np.float32)
    av, ao = city_xgboost.as_of(v, o, max_back=3)
    assert av[0, 0].tolist() == [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0]
    assert ao[0, 0].tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
