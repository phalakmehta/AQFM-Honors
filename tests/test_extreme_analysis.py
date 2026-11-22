"""Tests for scripts/extreme_value_analysis.py."""
import numpy as np
import scipy.stats as stats
import pytest

from scripts.extreme_value_analysis import (
    tail_error_analysis,
    conditional_coverage_analysis,
    fit_gpd_errors,
    exceedance_detection,
    seasonal_decomposition,
    identify_worst_windows,
    run_analysis,
    format_markdown_report,
)


def test_gpd_fit_synthetic_pareto():
    """GPD fitting accurately estimates shape and scale parameters on synthetic Pareto data."""
    np.random.seed(42)
    # Generate Pareto samples: shape c=0.15, scale=12.0
    true_c = 0.15
    true_scale = 12.0
    samples = stats.genpareto.rvs(c=true_c, scale=true_scale, size=3000)

    # Let y_true be baseline, y_pred have errors equal to samples
    y_true = np.full(3000, 100.0)
    y_pred = y_true + samples
    mask = np.ones(3000)

    res = fit_gpd_errors(y_true, y_pred, mask, threshold_p=90.0)
    assert "shape_xi" in res
    assert "scale_sigma" in res
    assert "return_levels" in res
    assert abs(res["shape_xi"] - true_c) < 0.15
    assert res["scale_sigma"] > 5.0


def test_extreme_analysis_full_pipeline(tmp_path):
    """Full analysis pipeline runs and outputs complete schema on synthetic forecast data."""
    np.random.seed(123)
    n_windows = 100
    n_steps = 288  # 72 hours at 15-min steps

    # Synthetic forecast dict
    y_true = np.clip(100.0 + np.random.randn(n_windows, n_steps) * 40.0, 5.0, 500.0)
    y_pred = y_true + np.random.randn(n_windows, n_steps) * 15.0
    mask = np.ones((n_windows, n_steps))
    lo90 = y_pred - 25.0
    hi90 = y_pred + 25.0

    anchor = np.random.randint(0, 10, size=n_windows)
    origin_hours = np.linspace(1000.0, 15000.0, n_windows)

    f_dict = {
        "anchor": anchor,
        "origin_hours": origin_hours,
        "PM2.5_true": y_true,
        "PM2.5_pred": y_pred,
        "PM2.5_mask": mask,
        "PM2.5_lo90": lo90,
        "PM2.5_hi90": hi90,
        "PM10_true": y_true * 1.5,
        "PM10_pred": y_pred * 1.5,
        "PM10_mask": mask,
    }

    results = run_analysis(f_dict, target_pollutants=["PM2.5", "PM10"], leads=["1h", "24h"])

    assert "PM2.5" in results
    assert "PM10" in results
    assert "24h" in results["PM2.5"]

    res_24h = results["PM2.5"]["24h"]
    assert "tail_errors" in res_24h
    assert "overall" in res_24h["tail_errors"]
    assert "p90" in res_24h["tail_errors"]
    assert "conditional_coverage" in res_24h
    assert "gpd_fit" in res_24h
    assert "exceedance_detection" in res_24h
    assert "seasonal_decomposition" in res_24h
    assert "worst_windows" in res_24h
    assert len(res_24h["worst_windows"]) <= 20

    md = format_markdown_report(results)
    assert "# Extreme Value Analysis Report" in md
    assert "Pollutant: PM2.5" in md
