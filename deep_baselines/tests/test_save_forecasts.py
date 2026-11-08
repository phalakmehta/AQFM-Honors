"""deep_baselines.common.save_forecasts writes a file that scripts/forecast_stats.py reads back unchanged."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "deep_baselines"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import common  # noqa: E402
import forecast_stats as fs  # noqa: E402


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    pairs = [(t0, s) for t0 in (1000, 1096) for s in range(3)]
    N = len(pairs)
    y = rng.random((N, common.HORIZON, 2)).astype(np.float32) * 100
    pred = y + 1.0
    ym = (rng.random((N, common.HORIZON, 2)) > 0.2).astype(np.float32)
    t = np.arange(5000) / 4.0
    path = str(tmp_path / "f.npz")
    common.save_forecasts(path, "demo", pairs, t, y, pred, ym)
    data, hours = fs.load(f"{path}:demo", "PM10")
    cols = [h - 1 for h in common.HORIZONS.values()]
    for i, (t0, s) in enumerate(pairs):
        p, tr, m = data[(s, t0 + common.CONTEXT)]
        assert np.allclose(p, pred[i, cols, 1]) and np.allclose(tr, y[i, cols, 1]) and np.allclose(m, ym[i, cols, 1])
        assert hours[t0 + common.CONTEXT] == t[t0 + common.CONTEXT]
