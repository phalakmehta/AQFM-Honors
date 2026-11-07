"""
Build a city folder (the same files as preprocess_aqfm.py writes) from an HOURLY array that was already
processed by someone else: (hours, stations, columns), no timestamps, gaps possibly filled. Example (Mumbai):

    python data_pipeline/convert_hourly_city.py \
        --array ../raw_inputs/mumbai_drive_2026-11-06/mumbai_all_data_2324_28locs.npy \
        --columns PM2.5,PM10,NO2,Ozone,AT,RH,WS,WD,NOx,era5:blh,skip,era5:u10,era5:v10,era5:sp,era5:tp \
        --first-hour 2023-01-01 --stations data_pipeline/stations/mumbai.csv \
        --facts data_pipeline/stations/mumbai_facts.csv --out data/mumbai

How the hourly data is put on the 15-min grid (docs/V2_PROGRESS.md, step 10 data):
- Station columns (CPCB names and units): row r is the average of hour [first-hour + r, +1 h), IST. Its value goes
  on the 15-min step at hh:45, the last step of the hour (the step the evaluation scores for whole-hour leads);
  the other three steps stay unobserved. The model is told the truth: one reading per hour.
- Filled gaps: in a run of >= MAX_REPEAT identical consecutive hours only the first value is kept (gaps copied
  forward show up as flat runs; a real hourly average almost never repeats exactly three times).
- PHYSICAL_BOUNDS as in preprocess_aqfm.py. A station's AT, RH or WS (WS with WD) is dropped entirely when its
  median is more than MET_OUTLIER_Z robust standard deviations from the other stations' medians (broken sensor).
- ERA5 columns (era5:<field>; row r = hour first-hour + r in UTC): shifted +5:30 to IST, interpolated linearly to
  15 min between hours (interpolate_era5_to_stations.py), converted with preprocess_aqfm._era5_derive. ERA5 fields
  that are not given stay NaN in era5_met.npy.
- Station met is gap-filled from ERA5 on the hourly steps where the station has no reading (merge_era5's rule);
  the BLH input channel comes from ERA5 at every step (append_blh).
- static.npy from --facts with station_facts.scaled_static; coords from --stations (same order as the array).
The time grid covers exactly the array's hours, so load_multi can place the city next to others.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_aqfm import (AQFM_CHANNELS, ERA5_MET_CHANNELS, PHYSICAL_BOUNDS,  # noqa: E402
                             _era5_derive, append_blh)
from station_facts import scaled_static  # noqa: E402

EPOCH = datetime(2018, 1, 1)
STEPS_PER_HOUR = 4
ERA5_FIELDS = ("t2m", "d2m", "u10", "v10", "sp", "ssrd", "tp", "blh")
ERA5_SHIFT_STEPS = 22                       # 5 h 30 min: ERA5 is UTC, CPCB is IST
MAX_REPEAT = 3
MET_OUTLIER_Z = 5.0
MET_CHECKED = ("AT", "RH", "WS")            # dropping WS also drops WD (U and V need both)
FILL_CHANNELS = ("AT", "RH", "WS", "WD", "SR", "BP", "TOT_RF")   # as merge_era5


def mask_flat_runs(x, max_repeat=MAX_REPEAT):
    """x (H,) hourly values -> bool (H,), False for the 2nd and later values of a run of >= max_repeat
    identical consecutive values (NaN never counts as a repeat)."""
    same = np.r_[False, x[1:] == x[:-1]]
    run_id = np.cumsum(~same)
    run_len = np.bincount(run_id)[run_id]
    return ~(same & (run_len >= max_repeat))


def met_outliers(med):
    """med (S,) station medians -> bool (S,), True where |med - median| > MET_OUTLIER_Z robust SDs."""
    ok = ~np.isnan(med)
    centre = np.median(med[ok])
    sd = 1.4826 * np.median(np.abs(med[ok] - centre))
    return ok & (np.abs(med - centre) > MET_OUTLIER_Z * max(sd, 1e-6))


def interp_hourly(a):
    """a (S, T) with values every STEPS_PER_HOUR steps (NaN elsewhere) -> linear interpolation between two
    neighbouring hourly values (never across a missing hour, never beyond the ends)."""
    out = a.copy()
    k = STEPS_PER_HOUR
    for j in range(1, k):
        w = j / k
        lo, hi = a[:, :-k], a[:, k:]
        fill = lo * (1 - w) + hi * w                            # NaN if either end is NaN
        tgt = out[:, j:j + fill.shape[1]]
        sel = np.isnan(tgt)
        tgt[sel] = fill[sel]
    return out


def build(array, columns, first_hour, stations_df, facts_df):
    H, S, K = array.shape
    assert len(columns) == K, f"{len(columns)} column names for {K} columns"
    assert len(stations_df) == S, f"{len(stations_df)} stations in the list, {S} in the array"
    T = H * STEPS_PER_HOUR
    t0 = (first_hour - EPOCH).total_seconds() / 3600.0
    timestamps_h = t0 + 0.25 * np.arange(T)
    hour_step = np.arange(H) * STEPS_PER_HOUR + STEPS_PER_HOUR - 1     # hh:45
    C = len(AQFM_CHANNELS)
    values = np.zeros((S, C, T), np.float32)
    observed = np.zeros((S, C, T), np.float32)
    report = {"hours": H, "stations": S, "channels": {}}

    # ---- station columns ------------------------------------------------
    station_cols = {c: i for i, c in enumerate(columns) if c in AQFM_CHANNELS}
    unknown = [c for c in columns if c not in AQFM_CHANNELS and c != "skip" and not c.startswith("era5:")]
    assert not unknown, f"unknown column names {unknown}"
    hourly, ok = {}, {}
    for c, i in station_cols.items():
        x = array[:, :, i].astype(np.float64).T                   # (S, H)
        m = ~np.isnan(x)
        rep = np.zeros(S, np.int64)
        for s in range(S):
            keep = mask_flat_runs(x[s])
            rep[s] = int((m[s] & ~keep).sum())
            m[s] &= keep
        lo, hi = PHYSICAL_BOUNDS[c]
        bad = m & ((x < lo) | (x > hi))
        m &= ~bad
        hourly[c], ok[c] = x, m
        report["channels"][c] = {"given_hours": int((~np.isnan(x)).sum()), "flat_run_repeats_hidden": int(rep.sum()),
                                 "out_of_bounds": int(bad.sum())}
    dropped = []
    for c in MET_CHECKED:
        if c not in hourly:
            continue
        med = np.array([np.median(hourly[c][s][ok[c][s]]) if ok[c][s].any() else np.nan for s in range(S)])
        for s in np.flatnonzero(met_outliers(med)):
            for cc in ((c, "WD") if c == "WS" else (c,)):
                if cc in ok:
                    ok[cc][s] = False
                    dropped.append(f"{stations_df.station_id.iloc[s]}:{cc} (median {med[s]:.1f})" if cc == c
                                   else f"{stations_df.station_id.iloc[s]}:{cc} (with WS)")
    report["met_sensors_dropped"] = dropped
    for c in station_cols:
        ci = AQFM_CHANNELS.index(c)
        values[:, ci, hour_step] = np.where(ok[c], hourly[c], 0.0)
        observed[:, ci, hour_step] = ok[c]
        report["channels"][c]["observed_hours"] = int(ok[c].sum())

    # ---- ERA5 columns ---------------------------------------------------
    era5 = {}
    for i, c in enumerate(columns):
        if not c.startswith("era5:"):
            continue
        f = c.split(":", 1)[1]
        assert f in ERA5_FIELDS, f"unknown ERA5 field {f}"
        a = np.full((S, T), np.nan)
        k = np.arange(H) * STEPS_PER_HOUR + ERA5_SHIFT_STEPS
        keep = k < T
        a[:, k[keep]] = array[keep, :, i].T
        era5[f] = interp_hourly(a)
    report["era5_fields_given"] = sorted(era5)
    era5_met = np.full((S, len(ERA5_MET_CHANNELS), T), np.nan, np.float32)
    filled = {c: 0 for c in FILL_CHANNELS}
    for s in range(S):
        df = pd.DataFrame({f: era5[f][s] if f in era5 else np.full(T, np.nan) for f in ERA5_FIELDS})
        der = _era5_derive(df)
        for mi, name in enumerate(ERA5_MET_CHANNELS):
            era5_met[s, mi] = der[name]
        for c in FILL_CHANNELS:
            ci = AQFM_CHANNELS.index(c)
            v = der[c][hour_step]
            sel = (observed[s, ci, hour_step] == 0) & ~np.isnan(v)
            values[s, ci, hour_step[sel]] = v[sel]
            observed[s, ci, hour_step[sel]] = 1.0
            filled[c] += int(sel.sum())
    report["era5_gap_filled_hours"] = filled
    if "blh" in era5:                                          # IST hour of the mean BLH maximum (expect early afternoon)
        b = era5_met[:, ERA5_MET_CHANNELS.index("BLH")]
        hod = ((timestamps_h % 24) // 1).astype(int)
        report["blh_peak_hour_ist"] = int(np.argmax([np.nanmean(b[:, hod == h]) for h in range(24)]))
    values, observed = append_blh(values, observed, era5_met)

    coords = stations_df[["lat", "lon"]].to_numpy(np.float32)
    static = scaled_static(facts_df, list(stations_df.station_id))
    report["station_order"] = list(stations_df.station_id)
    report["channel_order"] = list(AQFM_CHANNELS) + ["BLH"]
    report["era5_met_channel_order"] = list(ERA5_MET_CHANNELS)
    report["date_range"] = {"start": str(first_hour), "steps": T}
    return dict(values=values, observed=observed, era5_met=era5_met, timestamps_h=timestamps_h,
                coords=coords, static=static), report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--array", required=True, help=".npy, float (hours, stations, columns)")
    ap.add_argument("--columns", required=True,
                    help="comma list: a CPCB channel name, era5:<field> or skip, one per array column")
    ap.add_argument("--first-hour", required=True, help="YYYY-MM-DD[ HH:MM], start of row 0 (IST)")
    ap.add_argument("--stations", required=True, help="CSV with station_id, lat, lon in the array's order")
    ap.add_argument("--facts", required=True, help="station_facts.py output for these stations")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    array = np.load(args.array, allow_pickle=False)
    arrays, report = build(array, args.columns.split(","), datetime.fromisoformat(args.first_hour),
                           pd.read_csv(args.stations), pd.read_csv(args.facts))
    with open(args.array, "rb") as f:
        report["source"] = {"array": os.path.abspath(args.array), "sha256": hashlib.sha256(f.read()).hexdigest(),
                            "columns": args.columns, "first_hour": args.first_hour}
    os.makedirs(args.out, exist_ok=True)
    for k, v in arrays.items():
        np.save(os.path.join(args.out, f"{k}.npy"), v)
    with open(os.path.join(args.out, "preprocessing_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "station_order"}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
