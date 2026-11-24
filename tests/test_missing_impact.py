"""Tests for scripts/missing_value_impact.py."""
import numpy as np
import pytest

from scripts.missing_value_impact import (
    compute_context_completeness,
    analyze_completeness_bins,
    analyze_neighbour_compensation,
    format_markdown_report,
)


def test_context_completeness_and_bins():
    """Context completeness is accurately computed and binned."""
    n_stations = 4
    n_steps = 1000
    observed = np.ones((n_stations, 18, n_steps), dtype=np.uint8)

    # Station 0: 50% missing in steps 0..512
    observed[0, 0, 0:256] = 0

    anchor = np.array([0, 1])
    origin = np.array([512, 512])

    comp = compute_context_completeness(observed, anchor, origin, context_steps=512, channel_idx=0)
    assert abs(comp[0] - 0.5) < 1e-4
    assert abs(comp[1] - 1.0) < 1e-4

    # Test binning
    y_true = np.ones((2, 288)) * 100.0
    y_pred = np.ones((2, 288)) * 110.0
    mask = np.ones((2, 288))

    f_dict = {
        "PM2.5_true": y_true,
        "PM2.5_pred": y_pred,
        "PM2.5_mask": mask,
    }

    bins_res = analyze_completeness_bins(f_dict, comp, target="PM2.5", leads=["24h"])
    assert "40-60%" in bins_res
    assert bins_res["40-60%"]["count"] == 1
    assert bins_res["40-60%"]["24h"]["MAE"] == 10.0
    assert "80-100%" in bins_res
    assert bins_res["80-100%"]["count"] == 1


def test_neighbour_compensation_and_report():
    """Neighbour compensation calculation identifies assistance from nearby stations."""
    n_stations = 4
    n_steps = 600
    observed = np.ones((n_stations, 18, n_steps), dtype=np.uint8)

    # Station 0 has only 20% observed
    observed[0, 0, 0:400] = 0

    # Near neighbours (stations 1, 2) have 100% observed
    coords = np.array([
        [28.50, 77.10],
        [28.51, 77.11],
        [28.52, 77.12],
        [28.90, 77.50],
    ])

    anchor = np.array([0, 0])
    origin = np.array([512, 512])

    y_true = np.ones((2, 288)) * 80.0
    y_pred = np.ones((2, 288)) * 95.0
    mask = np.ones((2, 288))

    f_dict = {
        "PM2.5_true": y_true,
        "PM2.5_pred": y_pred,
        "PM2.5_mask": mask,
    }

    res = analyze_neighbour_compensation(
        observed, coords, f_dict, anchor, origin, context_steps=512, target="PM2.5", lead="24h"
    )

    assert "low_anchor_count" in res
    assert res["low_anchor_count"] == 2
    assert "assisted_by_high_nbr_completeness" in res

    all_res = {
        "completeness_bins": {"80-100%": {"count": 10, "24h": {"MAE": 12.3}}},
        "neighbour_compensation": res,
    }
    md = format_markdown_report(all_res)
    assert "# Missing Value Imputation Impact Analysis" in md
    assert "Spatial Neighbour Compensation" in md
