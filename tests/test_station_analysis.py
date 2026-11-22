"""Tests for scripts/station_error_analysis.py."""
import numpy as np
import pandas as pd
import pytest

from scripts.station_error_analysis import (
    per_station_metrics,
    analyze_station_characteristics,
    cluster_stations_by_error,
    run_station_analysis,
    format_markdown_report,
)
from baselines.baseline_utils import HORIZONS


def test_per_station_weighted_mae_matches_overall():
    """Sum of per-station errors equals overall error across all valid observations."""
    np.random.seed(42)
    n_windows = 120
    n_steps = 288
    n_stations = 6

    # 20 windows per station
    anchor = np.repeat(np.arange(n_stations), 20)
    y_true = np.random.uniform(20.0, 200.0, size=(n_windows, n_steps))
    y_pred = y_true + np.random.normal(0, 15.0, size=(n_windows, n_steps))
    mask = (np.random.uniform(0, 1, size=(n_windows, n_steps)) > 0.1).astype(float)

    f_dict = {
        "anchor": anchor,
        "PM2.5_true": y_true,
        "PM2.5_pred": y_pred,
        "PM2.5_mask": mask,
    }

    st_metrics = per_station_metrics(f_dict, target="PM2.5", leads=["24h"])
    j = HORIZONS["24h"] - 1

    # Overall MAE
    overall_valid = mask[:, j] > 0.5
    overall_mae = np.mean(np.abs(y_pred[overall_valid, j] - y_true[overall_valid, j]))

    # Weighted sum of per-station MAEs
    total_obs = sum(st_metrics[s]["24h"]["n"] for s in st_metrics)
    weighted_mae = sum(st_metrics[s]["24h"]["MAE"] * st_metrics[s]["24h"]["n"] for s in st_metrics) / total_obs

    assert abs(weighted_mae - overall_mae) < 1e-4


def test_station_analysis_full_pipeline(tmp_path):
    """Full spatial station analysis runs and outputs valid schema with facts and clustering."""
    np.random.seed(99)
    n_stations = 8
    n_windows = 80
    n_steps = 288

    anchor = np.repeat(np.arange(n_stations), 10)
    origin_hours = np.linspace(100.0, 5000.0, n_windows)
    y_true = np.random.uniform(30.0, 150.0, size=(n_windows, n_steps))
    y_pred = y_true + np.random.normal(0, 10.0, size=(n_windows, n_steps))
    mask = np.ones((n_windows, n_steps))

    # Synthetic facts dataframe saved to csv
    facts_df = pd.DataFrame({
        "station_id": [f"st_{i}" for i in range(n_stations)],
        "name": [f"Station {i}" for i in range(n_stations)],
        "lat": np.linspace(28.5, 28.8, n_stations),
        "lon": np.linspace(77.1, 77.3, n_stations),
        "elevation_m": np.linspace(200.0, 260.0, n_stations),
        "built_frac_1km": np.linspace(0.2, 0.9, n_stations),
        "green_frac_1km": np.linspace(0.1, 0.6, n_stations),
        "coast_km": np.full(n_stations, 800.0),
    })
    facts_file = tmp_path / "delhi_facts.csv"
    facts_df.to_csv(facts_file, index=False)

    f_dict = {
        "anchor": anchor,
        "origin_hours": origin_hours,
        "PM2.5_true": y_true,
        "PM2.5_pred": y_pred,
        "PM2.5_mask": mask,
    }

    results = run_station_analysis(f_dict, facts_path=str(facts_file), target="PM2.5")

    assert "station_metrics" in results
    assert len(results["station_metrics"]) == n_stations
    assert "characteristics_and_outliers" in results
    char_info = results["characteristics_and_outliers"]
    assert "median_MAE_24h" in char_info
    assert "correlations" in char_info
    assert "elevation_m" in char_info["correlations"]
    assert "clusters" in results
    assert len(results["clusters"]) >= 2
    assert "worst_stations_monthly" in results

    md = format_markdown_report(results)
    assert "# Spatial Station Error Analysis" in md
    assert "Outlier Threshold" in md
