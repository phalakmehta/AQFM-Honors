"""Baselines of scripts/gapfill_eval.py on synthetic series."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import gapfill_eval as ge  # noqa: E402


def test_cycle_and_line_fill_a_pure_daily_cycle():
    T = 512
    hours = (np.arange(T) // 4) % 24
    cyc = 50 + 20 * np.sin(2 * np.pi * np.arange(24) / 24)
    series = cyc[hours] + 10.0                       # the cycle plus a constant offset
    obs = np.ones(T, bool)
    idx, pred, feats = ge.fill_window(series, obs, 128, 96, cyc, hours, [])
    np.testing.assert_allclose(pred["cycle"], series[idx])          # exact for cycle + constant
    assert np.abs(pred["line"] - series[idx]).max() > 1             # a straight line cannot follow the cycle
    np.testing.assert_allclose(pred["neighbours"], pred["line"])    # no neighbours -> the line
    assert feats.shape[0] == 96


def test_neighbours_are_scaled_to_the_station():
    T = 512
    rng = np.random.default_rng(0)
    nb = 40 + rng.random(T) * 10
    series = 2.0 * nb                                # the station is twice its neighbour
    obs = np.ones(T, bool)
    idx, pred, _ = ge.fill_window(series, obs, 256, 64, np.zeros(24), np.zeros(T, int), [(nb, obs)])
    np.testing.assert_allclose(pred["neighbours"], series[idx], rtol=1e-9)


def test_too_few_visible_readings_is_skipped():
    obs = np.zeros(512, bool)
    obs[300] = True
    assert ge.fill_window(np.ones(512), obs, 128, 96, np.zeros(24), np.zeros(512, int), []) is None
