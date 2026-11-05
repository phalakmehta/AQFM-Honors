"""v2 data-pipeline tests (B1: any city from arguments alone).  Run from the repo root:
python -m pytest tests -q

Synthetic raw CPCB CSVs and an ERA5 parquet in a temp folder; the real Delhi checks
(defaults reproduce data/preprocessed/) need the raw data and are run by hand.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))
sys.path.insert(0, ROOT)

import preprocess_aqfm as pp  # noqa: E402

COLS = {"PM2.5 (µg/m³)": 80.0, "PM10 (µg/m³)": 160.0, "NO (µg/m³)": 20.0, "NO2 (µg/m³)": 40.0,
        "NOx (ppb)": 35.0, "NH3 (µg/m³)": 30.0, "SO2 (µg/m³)": 12.0, "CO (mg/m³)": 1.2,
        "Ozone (µg/m³)": 30.0, "Benzene (µg/m³)": 2.0, "AT (°C)": 25.0, "RH (%)": 60.0,
        "WS (m/s)": 1.5, "WD (deg)": 200.0, "SR (W/mt2)": 100.0, "TOT-RF (mm)": 0.0}
# station -> (BP reading as reported, ERA5 surface pressure in Pa)
STATIONS = {"low_hpa": (985.0, 98500.0), "low_mmhg": (739.0, 98500.0), "hill_hpa": (845.0, 84500.0)}


def _raw_city(tmp, coords_missing=()):
    """<tmp>/raw/2024/<sid>.csv (3 days of 15-min rows), stations.csv, era5.parquet."""
    os.makedirs(tmp / "raw" / "2024")
    ts = pd.date_range("2024-03-01", periods=3 * 96, freq="15min")
    for sid, (bp, _) in STATIONS.items():
        df = pd.DataFrame({"Station": sid, "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                           **{k: v for k, v in COLS.items()}, "BP (mmHg)": bp})
        df.to_csv(tmp / "raw" / "2024" / f"{sid}.csv", index=False)
    pd.DataFrame({"station_id": list(STATIONS), "lat": [28.6, 28.7, 30.0], "lon": [77.2, 77.1, 77.0]}).assign(
        lat=lambda d: d.lat.where(~d.station_id.isin(coords_missing))).to_csv(tmp / "stations.csv", index=False)
    t = pd.date_range("2017-12-31 18:30", "2025-12-31 18:15", freq="15min")      # UTC, covers IST 2018-2025
    parts = []
    for sid, (_, sp) in STATIONS.items():
        parts.append(pd.DataFrame({"time": t, "station_id": sid, "t2m": 298.0, "d2m": 290.0, "u10": 1.0,
                                   "v10": -1.0, "sp": sp, "ssrd": 3.6e5, "tp": 0.0, "blh": 500.0}))
    pd.concat(parts).to_parquet(tmp / "era5.parquet", index=False)


def _run(tmp, out, *extra):
    pp.main(["--stations", str(tmp / "stations.csv"), "--raw-dir", str(tmp / "raw"),
             "--era5", str(tmp / "era5.parquet"), "--out", str(tmp / out), *extra])
    load = lambda k: np.load(tmp / out / f"{k}.npy")
    rep = json.load(open(tmp / out / "preprocessing_report.json"))
    return {k: load(k) for k in ("values", "observed", "static", "coords", "timestamps_h")}, rep


def test_pipeline_builds_any_city_from_arguments(tmp_path):
    """Station file, raw folder, ERA5 file and output folder are all arguments; the BP unit
    rules; location facts as static input; two cities built this way stack with load_multi."""
    _raw_city(tmp_path)
    a, rep = _run(tmp_path, "cityA")
    order = rep["station_order"]
    assert order == sorted(STATIONS) or set(order) == set(STATIONS)
    bp = pp.AQFM_CHANNELS.index("BP")
    day = np.flatnonzero(a["observed"][0, pp.AQFM_CHANNELS.index("PM2.5")])   # the CSV rows
    got = {sid: float(a["values"][i, bp, day].mean()) for i, sid in enumerate(order)}
    # v1 rule (> 900 = hPa): right in the lowlands, wrong at 1500 m (845 hPa kept as "mmHg")
    assert got["low_hpa"] == pytest.approx(985.0 * 0.750062, rel=1e-6)
    assert got["low_mmhg"] == pytest.approx(739.0) and got["hill_hpa"] == pytest.approx(845.0)
    assert rep["bp_hpa_converted"] == len(day) and "bp_rule" not in rep
    # era5 rule: the same conversions in the lowlands, and right at 1500 m
    b, rep_b = _run(tmp_path, "cityB", "--bp-rule", "era5")
    got_b = {sid: float(b["values"][i, bp, day].mean()) for i, sid in enumerate(rep_b["station_order"])}
    assert got_b["low_hpa"] == got["low_hpa"] and got_b["low_mmhg"] == got["low_mmhg"]
    assert got_b["hill_hpa"] == pytest.approx(845.0 * 0.750062, rel=1e-6)
    assert rep_b["bp_rule"] == "era5" and rep_b["bp_hpa_converted"] == 2 * len(day)
    lowland = [i for i, sid in enumerate(order) if sid.startswith("low")]
    for k in ("values", "observed"):
        np.testing.assert_array_equal(a[k][lowland], b[k][lowland])
    # location facts as the static input
    facts = pd.DataFrame({"station_id": list(STATIONS), "elevation_m": [215.0, 220.0, 1500.0],
                          "built_frac_1km": [0.8, 0.5, 0.1], "green_frac_1km": [0.1, 0.3, 0.7],
                          "coast_km": [820.0, 830.0, 900.0]})
    facts.to_csv(tmp_path / "facts.csv", index=False)
    c, rep_c = _run(tmp_path, "cityC", "--static", "facts", "--facts", str(tmp_path / "facts.csv"))
    i = rep_c["station_order"].index("hill_hpa")
    np.testing.assert_allclose(c["static"][i], [1.5, 0.1, 0.7, np.log10(901.0) / 3], rtol=1e-6)
    with pytest.raises(KeyError):
        facts.iloc[:2].to_csv(tmp_path / "facts2.csv", index=False)
        _run(tmp_path, "cityD", "--static", "facts", "--facts", str(tmp_path / "facts2.csv"))
    # the folders stack as cities on one time grid
    from aqfm.train import load_multi
    m = load_multi([str(tmp_path / "cityA"), str(tmp_path / "cityB")])
    assert m["group_names"] == ["cityA", "cityB"] and m["values"].shape[0] == 6
    np.testing.assert_array_equal(m["timestamps_h"], a["timestamps_h"])


def test_pipeline_stops_on_missing_coordinates(tmp_path):
    _raw_city(tmp_path, coords_missing=("hill_hpa",))
    with pytest.raises(ValueError, match="no coordinates for station hill_hpa"):
        _run(tmp_path, "cityA")


def test_bp_era5_rule_any_altitude():
    """Readings in hPa or mmHg at sea level, 1500 m and 3500 m: the ERA5 rule picks the right unit
    everywhere; the v1 '> 900' rule fails above ~990 m."""
    for sp_hpa in (1010.0, 845.0, 660.0):
        e_mmhg = np.full(4, sp_hpa * 0.750062)
        bp = np.array([sp_hpa, sp_hpa * 0.750062, sp_hpa + 8, sp_hpa * 0.750062 - 6])
        hpa = pp.bp_hpa_mask_era5(bp, np.ones(4, bool), e_mmhg)
        assert hpa.tolist() == [True, False, True, False], sp_hpa
    e = np.array([np.nan, 740.0, 741.0])                       # missing ERA5 -> station median
    assert pp.bp_hpa_mask_era5(np.array([985.0, 985.0, 739.0]), np.ones(3, bool), e).tolist() == [True, True, False]
    with pytest.raises(ValueError):
        pp.bp_hpa_mask_era5(np.ones(2), np.ones(2, bool), np.full(2, np.nan))


def test_era5_interpolation_stops_outside_the_box(tmp_path):
    import xarray as xr
    import interpolate_era5_to_stations as ie
    lat, lon = np.array([29.0, 28.75, 28.5, 28.25]), np.array([76.5, 76.75, 77.0, 77.25, 77.5])
    t = pd.date_range("2024-01-01", periods=3, freq="1h")
    ds = xr.Dataset({"t2m": (("valid_time", "latitude", "longitude"), np.full((3, 4, 5), 290.0))},
                    coords={"valid_time": t, "latitude": lat, "longitude": lon})
    ds.to_netcdf(tmp_path / "m.nc")
    inside = pd.DataFrame({"station_id": ["a"], "lat": [28.6], "lon": [77.2]})
    assert len(ie.process_month(str(tmp_path / "m.nc"), inside)) == 3
    far = pd.DataFrame({"station_id": ["a", "mumbai_x"], "lat": [28.6, 19.0], "lon": [77.2, 72.8]})
    with pytest.raises(ie.OutsideERA5Box, match="mumbai_x"):
        ie.process_month(str(tmp_path / "m.nc"), far)


def test_fetch_and_geocode_helpers():
    import fetch_era5 as fe
    import generate_stations_csv as gs
    assert fe.AREA == [29.0, 76.5, 28.2, 77.5]                 # v1 default box unchanged
    assert fe.months("2017-12", "2018-02") == [(2017, 12), (2018, 1), (2018, 2)]
    assert len(fe.months("2021-01", "2025-12")) == 60           # v1 default loop

    class Fake:
        def geocode(self, q, timeout=10):
            return None if q.startswith("Nowhere") else type("L", (), {"latitude": 1.0, "longitude": 2.0})()
    assert gs.geocode(["ito,_delhi_-_cpcb_cleaned"], "Delhi, India", Fake(), pause=0)[0]["lat"] == 1.0
    with pytest.raises(SystemExit, match="nowhere"):
        gs.geocode(["nowhere,_delhi_-_x"], "Delhi, India", Fake(), pause=0)
