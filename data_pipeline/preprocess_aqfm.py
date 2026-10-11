"""
AQFM Data Preprocessing Pipeline
=================================
Transforms raw CPCB station CSVs (2018-2025) and ERA5 parquet into the dense
(S, C, T) tensor arrays that AirQualityWindows expects.

Outputs (saved to data/preprocessed/):
    values.npy          (S, 18, T) - raw concentrations & met (+ ERA5 BLH), NaN -> 0
    observed.npy        (S, 18, T) - binary observation mask
    era5_met.npy        (S, 8, T)  - pure ERA5 met in IST (AT, RH, U, V, SR, BP, TOT_RF, BLH), NaN = missing
    timestamps_h.npy    (T,)       - hours since epoch (2018-01-01 00:00)
    coords.npy          (S, 2)     - lat, lon
    static.npy          (S, 4)     - per-station static covariates
    preprocessing_report.json      - diagnostics, outage params, channel map

The 17 channels match DELHI_CHANNELS in config.py:
    [PM2.5, PM10, NO2, NOx, NO, Ozone, SO2, NH3, CO,
     AT, RH, WS, WD, SR, BP, TOT_RF, Benzene]

Note: The 2 derived ratio channels (PM2.5/PM10 fraction, NO/NOx fraction)
are NOT computed here -- they are computed on-the-fly by physical.py's
augment_with_ratios() when the dataset is constructed. This keeps the
preprocessing clean and the ratio logic in one place.
"""

import os
import sys
import json
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ============================================================================
# Configuration
# ============================================================================
ROOT = r"c:\Users\phala\Foundation Model"
YEARLY_DATA_DIR = os.path.join(ROOT, "data", "yearly_data")
ERA5_PARQUET = os.path.join(ROOT, "data", "era5_interpolated.parquet")
STATIONS_CSV = os.path.join(ROOT, "data", "stations.csv")
OUTPUT_DIR = os.path.join(ROOT, "data", "preprocessed")

YEARS = list(range(2018, 2026))  # 2018-2025 inclusive
FREQ = "15min"
EPOCH = pd.Timestamp("2018-01-01 00:00:00")

# The 17 AQFM channels in the exact order config.py expects
# (matches DELHI_CHANNELS and ChannelLayout in physical.py)
AQFM_CHANNELS = [
    "PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO",
    "AT", "RH", "WS", "WD", "SR", "BP", "TOT_RF", "Benzene",
]
# Saved arrays carry ERA5 boundary-layer height as an 18th channel
OUTPUT_CHANNELS = AQFM_CHANNELS + ["BLH"]

# Pure-ERA5 met saved separately as era5_met.npy (decoder's known-future input)
ERA5_MET_CHANNELS = ["AT", "RH", "U", "V", "SR", "BP", "TOT_RF", "BLH"]

# Mapping from CSV column substrings to AQFM channel names
# The CSV columns have units in parentheses, so we match by prefix
CSV_TO_CHANNEL = {
    "PM2.5":    "PM2.5",
    "PM10":     "PM10",
    "NO2":      "NO2",
    "NOx":      "NOx",
    "NO ":      "NO",      # note the space to avoid matching NO2/NOx
    "Ozone":    "Ozone",
    "SO2":      "SO2",
    "NH3":      "NH3",
    "CO ":      "CO",      # space to avoid matching "CO2" if it ever appears
    "Benzene":  "Benzene",
    "AT ":      "AT",
    "RH ":      "RH",
    "WS ":      "WS",
    "WD ":      "WD",
    "SR ":      "SR",
    "BP ":      "BP",
    "TOT-RF":   "TOT_RF",
}

# Physical bounds for sanity checking (min, max)
# Values outside these are treated as missing
PHYSICAL_BOUNDS = {
    "PM2.5":   (0, 2000),
    "PM10":    (0, 3000),
    "NO":      (0, 1500),
    "NO2":     (0, 1000),
    "NOx":     (0, 2000),
    "Ozone":   (0, 600),
    "SO2":     (0, 500),
    "NH3":     (0, 1000),
    "CO":      (0, 30),       # mg/m³
    "Benzene": (0, 200),
    "AT":      (-15, 55),     # °C
    "RH":      (0, 100),      # %
    "WS":      (0, 50),       # m/s
    "WD":      (0, 360),      # degrees
    "SR":      (0, 1500),     # W/m²
    "BP":      (600, 1100),   # mmHg
    "TOT_RF":  (0, 500),      # mm
}


def map_csv_columns(csv_columns):
    """Map raw CSV column names to AQFM channel names.

    The CSV columns have encoded units like 'PM2.5 (µg/m³)' which may
    render differently depending on encoding. We match by prefix.
    """
    mapping = {}
    for csv_col in csv_columns:
        if csv_col in ("Station", "Timestamp"):
            continue
        for prefix, channel in CSV_TO_CHANNEL.items():
            if csv_col.startswith(prefix):
                mapping[csv_col] = channel
                break
    return mapping


# ============================================================================
# Step 1: Load & align CPCB station CSVs
# ============================================================================
def load_cpcb_data(station_ids):
    """Load all CPCB CSVs for all years, align to a common 15-min grid.

    Returns:
        values:   (S, C, T) float32 array
        observed: (S, C, T) float32 array (1=observed, 0=missing)
        timeline: DatetimeIndex of the canonical timeline
        station_order: list of station_id strings matching axis 0
    """
    n_channels = len(AQFM_CHANNELS)

    # Build canonical timeline
    t_start = pd.Timestamp(f"{YEARS[0]}-01-01 00:00:00")
    t_end = pd.Timestamp(f"{YEARS[-1]}-12-31 23:45:00")
    timeline = pd.date_range(t_start, t_end, freq=FREQ)
    T = len(timeline)
    print(f"  Canonical timeline: {t_start} to {t_end}, {T:,} steps")

    # Filter to stations that actually have data
    station_order = []
    for sid in station_ids:
        has_data = False
        for yr in YEARS:
            csv_path = os.path.join(YEARLY_DATA_DIR, str(yr), f"{sid}.csv")
            if os.path.exists(csv_path):
                has_data = True
                break
        if has_data:
            station_order.append(sid)
    
    S = len(station_order)
    print(f"  Stations with data: {S} (out of {len(station_ids)} in stations.csv)")

    values = np.zeros((S, n_channels, T), dtype=np.float32)
    observed = np.zeros((S, n_channels, T), dtype=np.float32)

    report = {"per_station": {}, "column_mapping": None}

    for si, sid in enumerate(station_order):
        station_dfs = []
        for yr in YEARS:
            csv_path = os.path.join(YEARLY_DATA_DIR, str(yr), f"{sid}.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except Exception as e:
                print(f"    WARNING: Failed to read {csv_path}: {e}")
                continue

            # Map columns
            col_map = map_csv_columns(df.columns)
            if report["column_mapping"] is None:
                report["column_mapping"] = {v: k for k, v in col_map.items()}

            # Parse timestamp
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp"])
            df = df.set_index("Timestamp")

            # Extract only the channels we need
            for csv_col, ch_name in col_map.items():
                if ch_name in AQFM_CHANNELS:
                    ch_idx = AQFM_CHANNELS.index(ch_name)
                    series = pd.to_numeric(df[csv_col], errors="coerce")
                    # Assign into a clean series aligned to this year's portion of timeline
                    yr_start = pd.Timestamp(f"{yr}-01-01 00:00:00")
                    yr_end = pd.Timestamp(f"{yr}-12-31 23:45:00")
                    yr_timeline = pd.date_range(yr_start, yr_end, freq=FREQ)

                    # Reindex to 15-min grid, handling duplicates by averaging
                    if series.index.duplicated().any():
                        series = series.groupby(series.index).mean()
                    aligned = series.reindex(yr_timeline)

                    # Find position in master timeline
                    offset = timeline.get_loc(yr_start)
                    n_steps = min(len(aligned), T - offset)

                    vals = aligned.values[:n_steps]
                    mask = ~np.isnan(vals)

                    values[si, ch_idx, offset:offset + n_steps] = np.nan_to_num(vals, nan=0.0)
                    observed[si, ch_idx, offset:offset + n_steps] = mask.astype(np.float32)

            station_dfs.append(yr)

        # Per-station missingness
        obs_pct = observed[si].mean(axis=1) * 100
        report["per_station"][sid] = {
            ch: f"{obs_pct[i]:.1f}%" for i, ch in enumerate(AQFM_CHANNELS)
        }
        report["per_station"][sid]["_years_loaded"] = station_dfs

        if (si + 1) % 10 == 0 or si == S - 1:
            print(f"    Loaded {si + 1}/{S} stations")

    return values, observed, timeline, station_order, report


def fix_bp_units(values, observed):
    """CPCB labels BP as mmHg, but about half the stations report hPa (~980)
    instead of mmHg (~740). The two ranges don't overlap in Delhi, so any
    reading > 900 is treated as hPa and converted to mmHg."""
    bp_idx = AQFM_CHANNELS.index("BP")
    hpa = (observed[:, bp_idx] > 0) & (values[:, bp_idx] > 900)
    values[:, bp_idx][hpa] *= 0.750062
    print(f"    BP: converted {int(hpa.sum()):,} hPa readings to mmHg")
    return values, int(hpa.sum())


# ============================================================================
# Step 2: Merge ERA5 covariates (gap-fill CPCB met channels)
# ============================================================================
def _era5_derive(df):
    """Raw ERA5 columns -> CPCB-equivalent met, as float32 numpy arrays.

        t2m  (K)   -> AT  (degC)   : subtract 273.15
        d2m + t2m  -> RH  (%)      : Magnus formula
        u10, v10   -> WS  (m/s)    : hypot;  WD (deg) : direction wind comes FROM
        sp   (Pa)  -> BP  (mmHg)   : / 133.322
        ssrd (J/m2 per hour) -> SR (W/m2) : / 3600
        tp   (m)   -> TOT_RF (mm)  : * 1000
    """
    f = lambda a: np.asarray(a, dtype=np.float32)
    t_c = df["t2m"].values - 273.15
    td_c = df["d2m"].values - 273.15
    b, c = 17.67, 243.5
    u, v = df["u10"].values, df["v10"].values
    return {
        "AT": f(t_c),
        "RH": f(np.clip(100.0 * np.exp((b * td_c) / (c + td_c)) / np.exp((b * t_c) / (c + t_c)), 0, 100)),
        "WS": f(np.sqrt(u ** 2 + v ** 2)),
        "WD": f((np.degrees(np.arctan2(-u, -v)) + 360) % 360),
        "BP": f(df["sp"].values / 133.322),
        "SR": f(np.clip(df["ssrd"].values / 3600.0, 0, None)),
        "TOT_RF": f(np.clip(df["tp"].values * 1000.0, 0, None)),
        "U": f(u),
        "V": f(v),
        "BLH": f(df["blh"].values),
    }


def merge_era5(values, observed, timeline, station_order):
    """Gap-fill the CPCB meteorological channels from ERA5, and return the
    pure-ERA5 met block used as the model's known-future covariate.

    Only fills where CPCB is missing; never overwrites a CPCB reading.

    The parquet is read ONE STATION AT A TIME (pyarrow row filter). Loading all
    ~12M rows at once and adding derived columns needs several GB and fails on
    a 16 GB machine; per station it is ~324k rows.
    """
    import pyarrow.parquet as pq

    # CPCB channels that ERA5 can gap-fill
    fill_channels = ["AT", "RH", "WS", "WD", "SR", "BP", "TOT_RF"]
    gaps_filled = {ch: 0 for ch in fill_channels}
    total_gaps = {ch: 0 for ch in fill_channels}

    S, T = len(station_order), len(timeline)
    era5_met = np.full((S, len(ERA5_MET_CHANNELS), T), np.nan, dtype=np.float32)
    shift = pd.Timedelta(hours=5, minutes=30)   # ERA5 is UTC, CPCB is IST

    print(f"  Reading ERA5 per station from {os.path.basename(ERA5_PARQUET)} ...")
    for si, sid in enumerate(station_order):
        tbl = pq.read_table(ERA5_PARQUET, filters=[("station_id", "==", sid)])
        if tbl.num_rows == 0:
            print(f"    WARNING: no ERA5 rows for {sid}")
            continue
        st = tbl.to_pandas()
        del tbl
        st["time"] = st["time"] + shift
        st = st.set_index("time").sort_index()
        st = st[~st.index.duplicated(keep="first")].reindex(timeline)
        der = _era5_derive(st)
        del st

        for mi, name in enumerate(ERA5_MET_CHANNELS):
            era5_met[si, mi] = der[name]

        for ch_name in fill_channels:
            ci = AQFM_CHANNELS.index(ch_name)
            missing = observed[si, ci] == 0
            vals = der[ch_name]
            total_gaps[ch_name] += int(missing.sum())
            fill = missing & ~np.isnan(vals)
            values[si, ci, fill] = vals[fill]
            observed[si, ci, fill] = 1.0
            gaps_filled[ch_name] += int(fill.sum())
        del der

        if (si + 1) % 10 == 0 or si == S - 1:
            print(f"    Gap-filled {si + 1}/{S} stations")

    fill_report = {}
    for ch_name in fill_channels:
        fill_report[ch_name] = {
            "total_gaps": total_gaps[ch_name],
            "gaps_filled": gaps_filled[ch_name],
            "fill_pct": f"{gaps_filled[ch_name] / max(1, total_gaps[ch_name]) * 100:.1f}%",
        }
        print(f"    {ch_name}: filled {gaps_filled[ch_name]:,} / {total_gaps[ch_name]:,} gaps "
              f"({fill_report[ch_name]['fill_pct']})")

    return values, observed, fill_report, era5_met


def append_blh(values, observed, era5_met):
    """BLH is not measured by CPCB, so channel 17 comes from ERA5 only."""
    blh = era5_met[:, ERA5_MET_CHANNELS.index("BLH")][:, None, :]
    ok = ~np.isnan(blh)
    values = np.concatenate([values, np.where(ok, blh, 0.0).astype(np.float32)], axis=1)
    observed = np.concatenate([observed, ok.astype(np.float32)], axis=1)
    print(f"    BLH: observed {ok.mean() * 100:.1f}% of steps")
    return values, observed


# ============================================================================
# Step 3: Physical sanity checks
# ============================================================================
def sanity_check(values, observed):
    """Apply physical bounds and log violations."""
    report = {"outliers_clipped": {}, "pm25_gt_pm10": 0, "nox_violations": 0}

    for ch_name, (lo, hi) in PHYSICAL_BOUNDS.items():
        if ch_name not in AQFM_CHANNELS:
            continue
        ci = AQFM_CHANNELS.index(ch_name)
        mask = observed[:, ci] > 0

        # Count values outside bounds
        vals = values[:, ci]
        below = (vals < lo) & mask
        above = (vals > hi) & mask
        n_bad = int(below.sum() + above.sum())

        if n_bad > 0:
            # Set out-of-bounds values to missing
            bad = below | above
            values[:, ci][bad] = 0.0
            observed[:, ci][bad] = 0.0
            report["outliers_clipped"][ch_name] = n_bad
            print(f"    {ch_name}: clipped {n_bad:,} out-of-bounds values")

    # Check PM2.5 > PM10
    pm25_idx = AQFM_CHANNELS.index("PM2.5")
    pm10_idx = AQFM_CHANNELS.index("PM10")
    both_obs = (observed[:, pm25_idx] > 0) & (observed[:, pm10_idx] > 0)
    violations = (values[:, pm25_idx] > values[:, pm10_idx]) & both_obs
    n_viol = int(violations.sum())
    report["pm25_gt_pm10"] = n_viol
    if n_viol > 0:
        n_both = int(both_obs.sum())
        print(f"    PM2.5 > PM10 violations: {n_viol:,} / {n_both:,} "
              f"({n_viol / max(1, n_both) * 100:.2f}%) — kept as-is (model handles via physical reparam)")

    # Check NOx consistency: NO + NO2 ≈ NOx
    # NOx is in ppb, NO and NO2 are in µg/m³ — different units, so just log
    no_idx = AQFM_CHANNELS.index("NO")
    no2_idx = AQFM_CHANNELS.index("NO2")
    nox_idx = AQFM_CHANNELS.index("NOx")
    all_obs = (observed[:, no_idx] > 0) & (observed[:, no2_idx] > 0) & (observed[:, nox_idx] > 0)
    n_all = int(all_obs.sum())
    report["nox_all_observed"] = n_all
    print(f"    NO+NO2+NOx all observed: {n_all:,} timesteps (units differ: NOx in ppb, NO/NO2 in µg/m³)")

    return values, observed, report


# ============================================================================
# Step 4: Build dense arrays
# ============================================================================
def build_arrays(values, observed, timeline, station_order, stations_df):
    """Build final arrays + timestamps + coords."""
    # Timestamps in hours since epoch
    timestamps_h = (timeline - EPOCH).total_seconds().values / 3600.0
    timestamps_h = timestamps_h.astype(np.float64)

    # Coordinates: match station_order to stations_df
    coords = np.zeros((len(station_order), 2), dtype=np.float32)
    for si, sid in enumerate(station_order):
        row = stations_df[stations_df["station_id"] == sid]
        if len(row) > 0:
            coords[si, 0] = row["lat"].values[0]
            coords[si, 1] = row["lon"].values[0]
        else:
            print(f"    WARNING: No coordinates for station {sid}, using Delhi center")
            coords[si, 0] = 28.6139
            coords[si, 1] = 77.209

    return timestamps_h, coords


# ============================================================================
# Step 5: Static features
# ============================================================================
def compute_static_features(values, observed, coords, station_order):
    """Compute 4 static features per station:
        [0] climatological mean log1p(PM2.5)  — pollution burden
        [1] mean PM2.5/PM10 ratio             — pollution composition
        [2] elevation proxy from mean BP       — altitude indicator
        [3] station density within 5km         — spatial context
    
    All computed from training period only (2018-2024, not 2025 test year).
    """
    S = len(station_order)
    static = np.zeros((S, 4), dtype=np.float32)

    pm25_idx = AQFM_CHANNELS.index("PM2.5")
    pm10_idx = AQFM_CHANNELS.index("PM10")
    bp_idx = AQFM_CHANNELS.index("BP")

    # Training period: 2018-2024 (first 7 years out of 8)
    # At 15-min cadence: 7 * 365.25 * 96 ≈ 245,028 steps
    # More precisely: use 7/8 of the total timesteps
    train_end = int(values.shape[2] * 7 / 8)

    for si in range(S):
        # Feature 0: mean log1p(PM2.5)
        pm25_obs = observed[si, pm25_idx, :train_end] > 0
        if pm25_obs.sum() > 0:
            pm25_vals = values[si, pm25_idx, :train_end][pm25_obs]
            static[si, 0] = np.log1p(pm25_vals).mean()

        # Feature 1: mean PM2.5/PM10 ratio
        both_obs = (observed[si, pm25_idx, :train_end] > 0) & (observed[si, pm10_idx, :train_end] > 0)
        if both_obs.sum() > 0:
            pm25_v = values[si, pm25_idx, :train_end][both_obs]
            pm10_v = values[si, pm10_idx, :train_end][both_obs]
            # Avoid division by zero
            valid = pm10_v > 1.0
            if valid.sum() > 0:
                ratios = np.clip(pm25_v[valid] / pm10_v[valid], 0.01, 0.99)
                static[si, 1] = ratios.mean()

        # Feature 2: elevation proxy from mean barometric pressure
        bp_obs = observed[si, bp_idx, :train_end] > 0
        if bp_obs.sum() > 0:
            bp_vals = values[si, bp_idx, :train_end][bp_obs]
            # Normalize: higher pressure = lower elevation
            # Standard sea-level pressure is ~760 mmHg
            static[si, 2] = (bp_vals.mean() - 760.0) / 10.0  # centered, scaled

    # Feature 3: station density within 5km
    from math import radians, sin, cos, sqrt, atan2
    for i in range(S):
        count = 0
        for j in range(S):
            if i == j:
                continue
            # Haversine distance
            lat1, lon1 = radians(coords[i, 0]), radians(coords[i, 1])
            lat2, lon2 = radians(coords[j, 0]), radians(coords[j, 1])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            d = 2 * 6371.0 * atan2(sqrt(a), sqrt(1 - a))
            if d <= 5.0:
                count += 1
        static[i, 3] = float(count)

    # Normalize static features to roughly zero-mean, unit-variance
    for f in range(4):
        col = static[:, f]
        valid = col != 0
        if valid.sum() > 1:
            mu, sd = col[valid].mean(), col[valid].std()
            if sd > 1e-6:
                static[:, f] = (col - mu) / sd

    print(f"  Static features shape: {static.shape}")
    for f, name in enumerate(["mean_log1p_PM25", "mean_PM25_PM10_ratio",
                               "elevation_proxy", "station_density_5km"]):
        print(f"    [{f}] {name}: min={static[:, f].min():.3f}, "
              f"max={static[:, f].max():.3f}, mean={static[:, f].mean():.3f}")

    return static


# ============================================================================
# Step 6: Fit outage distribution & save
# ============================================================================
def fit_outage_distribution(observed):
    """Fit log-normal distribution to real outage lengths across all stations/channels."""
    all_gaps = []
    S, C, T = observed.shape
    for s in range(S):
        # Use the average mask across pollutant channels (0-8) for station-wide outages
        station_mask = observed[s, :9, :].mean(axis=0)  # fraction of channels observed
        binary = (station_mask > 0.5).astype(bool)  # "mostly observed" vs not

        run = 0
        for t in range(T):
            if binary[t]:
                if run > 0:
                    all_gaps.append(run)
                run = 0
            else:
                run += 1
        if run > 0:
            all_gaps.append(run)

    if len(all_gaps) < 10:
        print("  WARNING: Too few outages found, using default parameters")
        return 3.0, 1.1, all_gaps

    lg = np.log(np.array(all_gaps, dtype=float))
    mu, sigma = float(lg.mean()), float(lg.std() + 1e-3)
    print(f"  Outage distribution: {len(all_gaps)} outages found")
    print(f"  LogNormal params: mu={mu:.3f}, sigma={sigma:.3f}")
    print(f"  Median outage: {np.exp(mu):.1f} steps ({np.exp(mu) * 15 / 60:.1f} hours)")
    print(f"  90th percentile: {np.exp(mu + 1.28 * sigma):.1f} steps "
          f"({np.exp(mu + 1.28 * sigma) * 15 / 60:.1f} hours)")

    return mu, sigma, all_gaps


def compute_missingness_summary(observed, station_order):
    """Per-channel, per-year missingness percentages."""
    S, C, T = observed.shape
    steps_per_year = T // len(YEARS)
    summary = {}

    for ci, ch in enumerate(OUTPUT_CHANNELS):
        yearly = {}
        for yi, yr in enumerate(YEARS):
            start = yi * steps_per_year
            end = min(start + steps_per_year, T)
            obs_rate = observed[:, ci, start:end].mean() * 100
            yearly[str(yr)] = f"{obs_rate:.1f}%"
        overall = observed[:, ci, :].mean() * 100
        summary[ch] = {"overall": f"{overall:.1f}%", "by_year": yearly}

    return summary


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {}

    print("=" * 70)
    print("AQFM Preprocessing Pipeline")
    print("=" * 70)

    # Load station metadata
    stations_df = pd.read_csv(STATIONS_CSV)
    station_ids = stations_df["station_id"].tolist()

    # --- Step 1: Load CPCB data ---
    print("\n[Step 1/6] Loading & aligning CPCB station CSVs...")
    t0 = datetime.now()
    values, observed, timeline, station_order, cpcb_report = load_cpcb_data(station_ids)
    report["cpcb"] = {"n_stations": len(station_order), "n_timesteps": len(timeline)}
    report["column_mapping"] = cpcb_report["column_mapping"]
    print(f"  Done in {(datetime.now() - t0).total_seconds():.1f}s")
    print(f"  Shape: values={values.shape}, observed={observed.shape}")
    values, report["bp_hpa_converted"] = fix_bp_units(values, observed)

    # --- Step 2: Merge ERA5 ---
    print("\n[Step 2/6] Merging ERA5 weather data (gap-filling CPCB met channels)...")
    t0 = datetime.now()
    values, observed, era5_report, era5_met = merge_era5(values, observed, timeline, station_order)
    report["era5_gap_fill"] = era5_report
    values, observed = append_blh(values, observed, era5_met)
    report["era5_met_nan_frac"] = {name: float(np.isnan(era5_met[:, i]).mean())
                                   for i, name in enumerate(ERA5_MET_CHANNELS)}
    print(f"  Done in {(datetime.now() - t0).total_seconds():.1f}s")

    # --- Step 3: Sanity checks ---
    print("\n[Step 3/6] Physical sanity checks & outlier handling...")
    t0 = datetime.now()
    values, observed, sanity_report = sanity_check(values, observed)
    report["sanity"] = sanity_report
    print(f"  Done in {(datetime.now() - t0).total_seconds():.1f}s")

    # --- Step 4: Build arrays ---
    print("\n[Step 4/6] Building dense arrays...")
    t0 = datetime.now()
    timestamps_h, coords = build_arrays(values, observed, timeline, station_order, stations_df)
    print(f"  timestamps_h: {timestamps_h.shape}, range [{timestamps_h[0]:.1f}, {timestamps_h[-1]:.1f}] hours")
    print(f"  coords: {coords.shape}")
    print(f"  Done in {(datetime.now() - t0).total_seconds():.1f}s")

    # --- Step 5: Static features ---
    print("\n[Step 5/6] Computing static features...")
    t0 = datetime.now()
    static = compute_static_features(values, observed, coords, station_order)
    print(f"  Done in {(datetime.now() - t0).total_seconds():.1f}s")

    # --- Step 6: Outage distribution & save ---
    print("\n[Step 6/6] Fitting outage distribution & saving...")
    t0 = datetime.now()
    mu, sigma, gaps = fit_outage_distribution(observed)
    report["outage_distribution"] = {
        "lognormal_mu": mu,
        "lognormal_sigma": sigma,
        "n_outages": len(gaps),
        "median_steps": float(np.exp(mu)),
        "median_hours": float(np.exp(mu) * 15 / 60),
    }

    # Missingness summary
    report["missingness"] = compute_missingness_summary(observed, station_order)

    # Channel order
    report["channel_order"] = OUTPUT_CHANNELS
    report["era5_met_channel_order"] = ERA5_MET_CHANNELS
    report["station_order"] = station_order
    report["date_range"] = {"start": str(timeline[0]), "end": str(timeline[-1])}
    report["shapes"] = {
        "values": list(values.shape),
        "observed": list(observed.shape),
        "timestamps_h": list(timestamps_h.shape),
        "coords": list(coords.shape),
        "static": list(static.shape),
    }

    # Save arrays
    print("\n  Saving arrays...")
    np.save(os.path.join(OUTPUT_DIR, "values.npy"), values)
    np.save(os.path.join(OUTPUT_DIR, "observed.npy"), observed)
    np.save(os.path.join(OUTPUT_DIR, "timestamps_h.npy"), timestamps_h)
    np.save(os.path.join(OUTPUT_DIR, "coords.npy"), coords)
    np.save(os.path.join(OUTPUT_DIR, "static.npy"), static)
    np.save(os.path.join(OUTPUT_DIR, "era5_met.npy"), era5_met)

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "preprocessing_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  Done in {(datetime.now() - t0).total_seconds():.1f}s")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  values.npy:       {values.shape}  ({values.nbytes / 1e6:.1f} MB)")
    print(f"  observed.npy:     {observed.shape}  ({observed.nbytes / 1e6:.1f} MB)")
    print(f"  timestamps_h.npy: {timestamps_h.shape}")
    print(f"  coords.npy:       {coords.shape}")
    print(f"  static.npy:       {static.shape}")
    print(f"\n  Overall observation rate by channel:")
    for ch in AQFM_CHANNELS:
        print(f"    {ch:10s}: {report['missingness'][ch]['overall']}")
    print(f"\n  Outage LogNormal: mu={mu:.3f}, sigma={sigma:.3f}")
    print(f"  Median outage length: {np.exp(mu) * 15 / 60:.1f} hours")

    # Verification checks
    print("\n--- Verification ---")
    # Check 1: mask integrity
    assert np.all((observed == 0) | (observed == 1)), "Mask not binary!"
    zero_val_nonzero_mask = np.sum((values == 0) & (observed == 1))
    print(f"  Mask is binary: PASS")
    print(f"  Zero values with mask=1: {zero_val_nonzero_mask:,} "
          f"(these are genuine zero readings, OK)")

    # Check 2: no NaN in output
    assert not np.any(np.isnan(values)), "NaN in values!"
    assert not np.any(np.isnan(observed)), "NaN in observed!"
    assert not np.any(np.isnan(timestamps_h)), "NaN in timestamps!"
    assert not np.any(np.isnan(coords)), "NaN in coords!"
    assert not np.any(np.isnan(static)), "NaN in static!"
    print(f"  No NaN in any output: PASS")

    # Check 3: timestamps monotonically increasing
    assert np.all(np.diff(timestamps_h) > 0), "Timestamps not monotonic!"
    print(f"  Timestamps monotonic: PASS")

    # Check 4: timestamp step is 0.25h (15 min)
    steps = np.diff(timestamps_h)
    assert np.allclose(steps, 0.25, atol=1e-6), f"Timestamp step not 0.25h! Got {np.unique(steps)}"
    print(f"  Timestamp step = 0.25h (15 min): PASS")

    print("\nAll verification checks passed!")


if __name__ == "__main__":
    main()
