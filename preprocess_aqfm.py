"""
AQFM Data Preprocessing Pipeline
=================================
Transforms raw CPCB station CSVs (2018-2025) and ERA5 parquet into the dense
(S, C, T) tensor arrays that AirQualityWindows expects.

Outputs (saved to preprocessed/):
    values.npy          (S, 17, T) - raw concentrations & met, NaN -> 0
    observed.npy        (S, 17, T) - binary observation mask
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
YEARLY_DATA_DIR = os.path.join(ROOT, "yearly_data")
ERA5_PARQUET = os.path.join(ROOT, "era5_interpolated.parquet")
STATIONS_CSV = os.path.join(ROOT, "stations.csv")
OUTPUT_DIR = os.path.join(ROOT, "preprocessed")

YEARS = list(range(2018, 2026))  # 2018-2025 inclusive
FREQ = "15min"
EPOCH = pd.Timestamp("2018-01-01 00:00:00")

# The 17 AQFM channels in the exact order config.py expects
# (matches DELHI_CHANNELS and ChannelLayout in physical.py)
AQFM_CHANNELS = [
    "PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO",
    "AT", "RH", "WS", "WD", "SR", "BP", "TOT_RF", "Benzene",
]

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


# ============================================================================
# Step 2: Merge ERA5 covariates (gap-fill CPCB met channels)
# ============================================================================
def merge_era5(values, observed, timeline, station_order):
    """Use ERA5 data to gap-fill CPCB meteorological channels.

    ERA5 variables mapped to CPCB channels:
        t2m  (K)          -> AT  (°C)    : subtract 273.15
        d2m + t2m         -> RH  (%)     : Magnus formula
        u10, v10          -> WS  (m/s)   : sqrt(u²+v²)
        u10, v10          -> WD  (deg)   : atan2(-u, -v) * 180/π, mod 360
        sp   (Pa)         -> BP  (mmHg)  : divide by 133.322
        ssrd (J/m²)       -> SR  (W/m²)  : divide by 3600 (hourly accumulation)
        tp   (m)          -> TOT_RF (mm) : multiply by 1000

    Only fills where CPCB has NaN; does NOT overwrite existing CPCB readings.
    """
    print("  Loading ERA5 parquet...")
    era5 = pd.read_parquet(ERA5_PARQUET)

    # Filter to our date range
    era5 = era5[(era5["time"] >= timeline[0]) & (era5["time"] <= timeline[-1])]
    print(f"  ERA5 rows in range: {len(era5):,}")

    # Derive CPCB-equivalent channels from ERA5
    era5["era5_AT"] = era5["t2m"] - 273.15  # K -> °C

    # RH from t2m and d2m using Magnus formula
    b, c = 17.67, 243.5
    t_c = era5["t2m"] - 273.15
    td_c = era5["d2m"] - 273.15
    era5["era5_RH"] = 100.0 * np.exp((b * td_c) / (c + td_c)) / np.exp((b * t_c) / (c + t_c))
    era5["era5_RH"] = era5["era5_RH"].clip(0, 100)

    # Wind speed and direction from u10, v10
    era5["era5_WS"] = np.sqrt(era5["u10"] ** 2 + era5["v10"] ** 2)
    era5["era5_WD"] = (np.degrees(np.arctan2(-era5["u10"], -era5["v10"])) + 360) % 360

    # Surface pressure Pa -> mmHg
    era5["era5_BP"] = era5["sp"] / 133.322

    # Solar radiation: J/m² accumulated over 1h -> W/m² (divide by 3600)
    era5["era5_SR"] = (era5["ssrd"] / 3600.0).clip(0)

    # Precipitation: m -> mm
    era5["era5_TOT_RF"] = (era5["tp"] * 1000.0).clip(0)

    # Map ERA5 derived columns to AQFM channel indices
    era5_to_aqfm = {
        "era5_AT":     AQFM_CHANNELS.index("AT"),
        "era5_RH":     AQFM_CHANNELS.index("RH"),
        "era5_WS":     AQFM_CHANNELS.index("WS"),
        "era5_WD":     AQFM_CHANNELS.index("WD"),
        "era5_SR":     AQFM_CHANNELS.index("SR"),
        "era5_BP":     AQFM_CHANNELS.index("BP"),
        "era5_TOT_RF": AQFM_CHANNELS.index("TOT_RF"),
    }

    gaps_filled = {ch: 0 for ch in era5_to_aqfm}
    total_gaps = {ch: 0 for ch in era5_to_aqfm}

    for si, sid in enumerate(station_order):
        st_era5 = era5[era5["station_id"] == sid].set_index("time").sort_index()
        if len(st_era5) == 0:
            continue

        # Reindex ERA5 to our canonical timeline
        st_era5 = st_era5.reindex(timeline)

        for era5_col, ch_idx in era5_to_aqfm.items():
            # Only fill where CPCB is missing (observed == 0)
            missing_mask = observed[si, ch_idx] == 0
            era5_vals = st_era5[era5_col].values

            total_gaps[era5_col] += int(missing_mask.sum())

            # ERA5 has data and CPCB doesn't -> fill
            era5_valid = ~np.isnan(era5_vals)
            fill_mask = missing_mask & era5_valid

            values[si, ch_idx, fill_mask] = era5_vals[fill_mask].astype(np.float32)
            observed[si, ch_idx, fill_mask] = 1.0

            gaps_filled[era5_col] += int(fill_mask.sum())

        if (si + 1) % 10 == 0 or si == len(station_order) - 1:
            print(f"    Gap-filled {si + 1}/{len(station_order)} stations")

    fill_report = {}
    for era5_col in era5_to_aqfm:
        ch_name = era5_col.replace("era5_", "")
        fill_report[ch_name] = {
            "total_gaps": total_gaps[era5_col],
            "gaps_filled": gaps_filled[era5_col],
            "fill_pct": f"{gaps_filled[era5_col] / max(1, total_gaps[era5_col]) * 100:.1f}%",
        }
        print(f"    {ch_name}: filled {gaps_filled[era5_col]:,} / {total_gaps[era5_col]:,} gaps "
              f"({fill_report[ch_name]['fill_pct']})")

    return values, observed, fill_report


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
        static[si, 3] = float(count)

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

    for ci, ch in enumerate(AQFM_CHANNELS):
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

    # --- Step 2: Merge ERA5 ---
    print("\n[Step 2/6] Merging ERA5 weather data (gap-filling CPCB met channels)...")
    t0 = datetime.now()
    values, observed, era5_report = merge_era5(values, observed, timeline, station_order)
    report["era5_gap_fill"] = era5_report
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
    report["channel_order"] = AQFM_CHANNELS
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
