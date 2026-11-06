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


def _hourly_city(tmp, name, first_hour="2024-03-01", H=72):
    """An already-processed hourly array (3 stations) like Mumbai's, built with convert_hourly_city.py."""
    import convert_hourly_city as ch
    rng = np.random.default_rng(0)
    cols = ["PM2.5", "AT", "WS", "WD", "era5:u10", "era5:v10", "era5:sp", "skip", "era5:blh"]
    a = np.zeros((H, 3, len(cols)))
    a[:, :, 0] = 50 + np.arange(H)[:, None] + np.arange(3)                 # no accidental repeats
    a[10:15, 0, 0] = 77.0                                                  # a filled gap: 5 identical hours
    a[20:22, 0, 0] = 66.0                                                  # 2 identical hours: kept
    a[:, :, 1] = 28 + rng.normal(0, 1, (H, 3))
    a[:, 2, 1] = 1.0 + rng.normal(0, 0.1, H)                               # broken thermometer at station 2
    a[:, :, 2] = 1.5 + rng.uniform(0, 1, (H, 3))
    a[30, 1, 2] = 60.0                                                     # out of bounds (> 50 m/s)
    a[:, :, 3] = 200.0 + np.arange(H)[:, None] % 7
    a[:, :, 4] = np.arange(H)[:, None]                                     # u10 = row index
    a[:, :, 5] = -1.0
    a[:, :, 6] = 100000.0
    a[:, :, 8] = 300 + 500 * (np.arange(H) % 24 == 8)[:, None]             # BLH peak at 08 UTC
    np.save(tmp / f"{name}.npy", a)
    st = pd.DataFrame({"station_id": ["a", "b", "c"], "lat": [19.0, 19.05, 19.1], "lon": [72.8, 72.85, 72.9]})
    st.to_csv(tmp / f"{name}_st.csv", index=False)
    st.assign(elevation_m=10.0, built_frac_1km=0.5, green_frac_1km=0.2, coast_km=2.0).to_csv(
        tmp / f"{name}_facts.csv", index=False)
    ch.main(["--array", str(tmp / f"{name}.npy"), "--columns", ",".join(cols), "--first-hour", first_hour,
             "--stations", str(tmp / f"{name}_st.csv"), "--facts", str(tmp / f"{name}_facts.csv"),
             "--out", str(tmp / name)])
    return a


def test_hourly_converter_places_masks_and_shifts(tmp_path):
    a = _hourly_city(tmp_path, "city")
    d = {k: np.load(tmp_path / "city" / f"{k}.npy") for k in ("values", "observed", "era5_met", "timestamps_h")}
    rep = json.load(open(tmp_path / "city" / "preprocessing_report.json"))
    v, o, e = d["values"], d["observed"], d["era5_met"]
    H, T = 72, 72 * 4
    pm = pp.AQFM_CHANNELS.index("PM2.5")
    hs = np.arange(H) * 4 + 3                                               # hh:45
    assert o[:, pm, hs].any() and not np.delete(o[:, pm], hs, axis=1).any()  # one reading per hour, at hh:45
    np.testing.assert_allclose(v[1, pm, hs], a[:, 1, 0])
    assert o[0, pm, hs[10]] and not o[0, pm, hs[11:15]].any()              # flat run: only its first hour kept
    assert o[0, pm, hs[20:22]].all()                                       # two equal hours are not a filled gap
    at = pp.AQFM_CHANNELS.index("AT")
    assert not o[2, at].any() and o[0, at, hs].all()                       # broken sensor dropped (no t2m to fill)
    assert rep["met_sensors_dropped"] == [rep["met_sensors_dropped"][0]] and "c:AT" in rep["met_sensors_dropped"][0]
    ws = pp.AQFM_CHANNELS.index("WS")                                       # out of bounds -> gap-filled from ERA5
    k = hs[30]
    np.testing.assert_allclose(v[1, ws, k], np.hypot(e[1, 2, k], e[1, 3, k]), rtol=1e-6)
    assert v[1, ws, k] < 50
    # ERA5: UTC row r lands on IST step 4r + 22, linear in between, nothing before the first value
    u = e[0, ERA5_U := 2]
    assert np.isnan(u[:22]).all()
    np.testing.assert_allclose(u[22:T:4][:10], np.arange(10))
    np.testing.assert_allclose(u[23:T - 4:4][:10], np.arange(10) + 0.25)
    assert np.isnan(e[:, 0]).all() and np.isnan(e[:, 4]).all()            # t2m, ssrd not given -> AT, SR NaN
    bp = pp.AQFM_CHANNELS.index("BP")
    np.testing.assert_allclose(v[0, bp, hs[10]], 100000.0 / 133.322, rtol=1e-5)
    assert not np.delete(o[0, bp], hs).any()                               # station-type channels stay hourly
    assert o[0, 17, 22:T - 4].all()                                        # BLH input: every step ERA5 covers
    assert rep["blh_peak_hour_ist"] == 13                                  # 08 UTC + 5:30 = 13:30 IST
    np.testing.assert_allclose(d["timestamps_h"][0], (pd.Timestamp("2024-03-01") - pd.Timestamp("2018-01-01"))
                               / pd.Timedelta(hours=1))


def test_hourly_city_stacks_with_another_and_date_splits(tmp_path):
    from aqfm.train import load_multi, split_indices
    _hourly_city(tmp_path, "x", "2024-03-01")
    _hourly_city(tmp_path, "y", "2024-03-02")
    m = load_multi([str(tmp_path / "x"), str(tmp_path / "y")])
    assert m["values"].shape == (6, 18, 4 * 96) and m["group_names"] == ["x", "y"]   # union of the two grids
    t = np.asarray(m["timestamps_h"])
    assert split_indices(t, 2024, 2025) == (0, len(t))                     # year defaults unchanged
    vs, ts = split_indices(t, 2024, 2025, "2024-03-02", "2024-03-03")
    assert (vs, ts) == (96, 192)
