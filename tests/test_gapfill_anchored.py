"""Edge-anchored gap filling (scripts/gapfill_anchored.py) on synthetic series."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from gapfill_anchored import edge_shift  # noqa: E402

W0, G0, G1 = 1000, 1400, 1448
STEPS = np.arange(G0, G1)


def test_constant_offset_is_removed():
    s = edge_shift(STEPS, np.full(len(STEPS), 60.0), np.full(512, 50.0), np.ones(512, bool), W0, G0, G1)
    assert np.allclose(60.0 + s, 50.0)


def test_one_side_only_gives_a_constant_shift():
    obs = np.ones(512, bool)
    obs[:G0 - W0] = False                                   # nothing visible before the gap
    s = edge_shift(STEPS, np.full(len(STEPS), 60.0), np.full(512, 50.0), obs, W0, G0, G1)
    assert np.allclose(s, -10.0)


def test_hidden_readings_are_never_anchors():
    series = np.full(512, 50.0)
    series[G0 - W0:G1 - W0] = 999.0                         # observed but hidden: must not be used
    s = edge_shift(STEPS, np.full(len(STEPS), 60.0), series, np.ones(512, bool), W0, G0, G1)
    assert np.allclose(s, -10.0)


def test_shift_is_linear_between_the_two_edges():
    series = np.full(512, 50.0)
    series[G1 - W0:] = 70.0                                 # level after the gap is 20 higher
    s = edge_shift(STEPS, np.full(len(STEPS), 50.0), series, np.ones(512, bool), W0, G0, G1)
    assert np.isclose(s[0], (G0 - (G0 - 1)) / (G1 - (G0 - 1)) * 20.0)
    assert np.all(np.diff(s) > 0) and s[-1] < 20.0
