"""Every number in docs/DATA_REPORT.md, computed from data/preprocessed/.

    python docs/data_report_stats.py      # -> docs/data_report_stats.json (+ printed summary)

Pollutant statistics use observed readings only (observed.npy == 1), pooled
over all stations and 15-min steps. Counts taken from preprocessing (BP
conversions, clipped outliers, PM2.5 > PM10) are copied from
preprocessing_report.json so that the JSON holds every reported number.
Deterministic: no randomness is used.
"""
import json, os, glob, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
P = "data/preprocessed"
v = np.load(f"{P}/values.npy", mmap_mode="r")
o = np.load(f"{P}/observed.npy", mmap_mode="r")
t = np.load(f"{P}/timestamps_h.npy")
coords = np.load(f"{P}/coords.npy")
static = np.load(f"{P}/static.npy")
met = np.load(f"{P}/era5_met.npy", mmap_mode="r")
rep = json.load(open(f"{P}/preprocessing_report.json"))
ch = rep["channel_order"]; C = {n: i for i, n in enumerate(ch)}
names = [s.split(",")[0].replace("_", " ").title() + (" (" + s.split("-_")[-1].replace("_cleaned", "").strip("_").upper() + ")") for s in rep["station_order"]]
S, _, T = v.shape
dt = pd.to_datetime("2018-01-01") + pd.to_timedelta(t, unit="h")
year, month, hour = dt.year.values, dt.month.values, dt.hour.values
out = {}

def series(name):
    return np.asarray(v[:, C[name]], np.float64), np.asarray(o[:, C[name]]) > 0

pm25, m25 = series("PM2.5"); pm10, m10 = series("PM10")
out["shape"] = [S, len(ch), T]
out["obs_cells_pm25"] = int(m25.sum())

# raw CSV inventory
inv = {y: len(glob.glob(f"data/yearly_data/{y}/*.csv")) for y in range(2009, 2027)}
out["csv_per_year"] = inv
out["csv_total"] = sum(inv.values())

# native CPCB met coverage (before ERA5 fill)
tot = S * T
out["native_met_coverage"] = {k: round(100 * (1 - d["total_gaps"] / tot), 1) for k, d in rep["era5_gap_fill"].items()}

# distribution
def pct(x, m):
    a = x[m]
    return {q: round(float(np.percentile(a, q)), 1) for q in (1, 5, 25, 50, 75, 95, 99)} | {"mean": round(float(a.mean()), 1), "max": round(float(a.max()), 1)}
out["pm25_dist"] = pct(pm25, m25); out["pm10_dist"] = pct(pm10, m10)
for n in ("NO2", "Ozone", "CO", "SO2", "NH3", "NOx", "NO", "Benzene"):
    x, m = series(n); out[f"{n}_dist"] = pct(x, m)

bands = {"Good": (0, 30), "Satisfactory": (30, 60), "Moderate": (60, 90), "Poor": (90, 120), "Very Poor": (120, 250), "Severe": (250, 1e9)}
a = pm25[m25]
out["pm25_band_share"] = {k: round(100 * float(((a >= lo) & (a < hi)).mean()), 1) for k, (lo, hi) in bands.items()}

# by year / month / hour (mean over all observed station-steps)
def grouped(x, m, key, keys):
    return {int(k): round(float(x[:, key == k][m[:, key == k]].mean()), 1) for k in keys}
out["pm25_by_year"] = grouped(pm25, m25, year, range(2018, 2026))
out["pm25_by_month"] = grouped(pm25, m25, month, range(1, 13))
out["pm25_by_hour"] = grouped(pm25, m25, hour, range(24))
out["pm10_by_month"] = grouped(pm10, m10, month, range(1, 13))
sev = (pm25 > 250) & m25
out["severe_share_by_month"] = {int(k): round(100 * float(sev[:, month == k].sum() / max(1, m25[:, month == k].sum())), 1) for k in range(1, 13)}
out["pm25_coverage_by_year"] = {int(y): round(100 * float(m25[:, year == y].mean()), 1) for y in range(2018, 2026)}

# PM2.5/PM10 ratio
both = m25 & m10 & (pm10 > 1)
r = pm25[both] / pm10[both]
out["ratio_median"] = round(float(np.median(r)), 3)
out["ratio_gt1_pct"] = round(100 * float((r > 1).mean()), 2)
season = {"Winter (DJF)": (12, 1, 2), "Pre-monsoon (MAM)": (3, 4, 5), "Monsoon (JJAS)": (6, 7, 8, 9), "Post-monsoon (ON)": (10, 11)}
mm = np.broadcast_to(month, pm25.shape)[both]
out["ratio_by_season"] = {k: round(float(np.median(r[np.isin(mm, ms)])), 3) for k, ms in season.items()}
out["pm25_by_season"] = {k: round(float(pm25[:, np.isin(month, ms)][m25[:, np.isin(month, ms)]].mean()), 1) for k, ms in season.items()}

# per-station
st = []
for s in range(S):
    st.append({"station": names[s], "lat": round(float(coords[s, 0]), 4), "lon": round(float(coords[s, 1]), 4),
               "pm25_cov": round(100 * float(m25[s].mean()), 1), "pm25_mean": round(float(pm25[s][m25[s]].mean()), 1),
               "first_year": int(year[np.argmax(m25[s])]) if m25[s].any() else None})
out["stations"] = st

# coordinates sanity
cs = [tuple(np.round(c, 4)) for c in coords]
out["dup_coords"] = [[names[i] for i in range(S) if cs[i] == c] for c in set(cs) if cs.count(c) > 1]
out["default_coord_stations"] = [names[i] for i in range(S) if abs(coords[i, 0] - 28.6139) < 1e-3 and abs(coords[i, 1] - 77.209) < 1e-3]
la, lo = np.radians(coords[:, 0]), np.radians(coords[:, 1])
h = np.sin((la[None] - la[:, None]) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[None] - lo[:, None]) / 2) ** 2
d = 2 * 6371 * np.arcsin(np.sqrt(h)); iu = np.triu_indices(S, 1)
out["pair_dist_km"] = {"min": round(float(d[iu].min()), 2), "median": round(float(np.median(d[iu])), 1), "max": round(float(d[iu].max()), 1)}
nn = np.sort(d + np.eye(S) * 1e9, 1)
out["nn_dist_km_median"] = round(float(np.median(nn[:, 0])), 2)
out["k12_radius_km_median"] = round(float(np.median(nn[:, 10])), 1)  # 11th neighbour
out["bbox"] = [round(float(coords[:, 0].min()), 3), round(float(coords[:, 0].max()), 3), round(float(coords[:, 1].min()), 3), round(float(coords[:, 1].max()), 3)]

# outage lengths (station-wide, >= half the 9 pollutants missing), train split only (2018-2023)
pol = [C[n] for n in ("PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO")]
tr = int(np.searchsorted(t, (pd.Timestamp("2024-01-01") - pd.Timestamp("2018-01-01")).total_seconds() / 3600))
gaps = []
for s in range(S):
    out_ = np.asarray(o[s, pol, :tr]).mean(0) < 0.5
    dd = np.diff(np.concatenate([[0], out_.astype(np.int8), [0]]))
    gaps += (np.flatnonzero(dd == -1) - np.flatnonzero(dd == 1)).tolist()
g = np.array(gaps) * 0.25
out["outages_all"] = {"n": int(len(g)), "median_h": float(np.median(g)), "share_le_1h": round(100 * float((g <= 1).mean()), 1)}
gl = g[g >= 4]
out["outages_ge4h"] = {"n": int(len(gl)), "median_h": round(float(np.median(gl)), 1), "p90_h": round(float(np.percentile(gl, 90)), 1),
                      "n_ge_1day": int((g >= 24).sum()), "n_ge_1week": int((g >= 168).sum()), "max_days": round(float(g.max() / 24), 1)}

# meteorology
mi = {n: i for i, n in enumerate(rep["era5_met_channel_order"])}
blh = np.asarray(met[:, mi["BLH"]], np.float64)
out["blh"] = {"p5": round(float(np.percentile(blh, 5))), "median": round(float(np.median(blh))), "p95": round(float(np.percentile(blh, 95)))}
out["blh_by_hour"] = {k: round(float(blh[:, hour == k].mean())) for k in (0, 3, 6, 9, 12, 15, 18, 21)}
out["blh_by_season"] = {k: round(float(blh[:, np.isin(month, ms)].mean())) for k, ms in season.items()}
ws = np.hypot(np.asarray(met[:, mi["U"]]), np.asarray(met[:, mi["V"]]))
# Spearman correlations with PM2.5 on a subsample (hourly points)
idx = np.arange(0, T, 4)
from scipy.stats import spearmanr
corr = {}
y = pm25[:, idx]; my = m25[:, idx]
for n, x in (("BLH", blh), ("Wind speed (ERA5)", ws), ("AT (ERA5)", np.asarray(met[:, mi["AT"]])), ("RH (ERA5)", np.asarray(met[:, mi["RH"]]))):
    corr[n] = round(float(spearmanr(y[my], np.asarray(x)[:, idx][my]).correlation), 2)
for n in ("PM10", "CO", "NO2", "Ozone"):
    x, m = series(n); xm = x[:, idx]; mm2 = my & m[:, idx]
    corr[n] = round(float(spearmanr(y[mm2], xm[mm2]).correlation), 2)
out["spearman_with_pm25"] = corr

# static
out["static_names"] = ["mean_log1p_PM25", "mean_PM25_PM10_ratio", "elevation_proxy(BP)", "station_density_5km"]
out["static_zero_cols"] = [int((static[:, j] == 0).sum()) for j in range(4)]

# sizes
def du(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) else os.path.getsize(p)
out["sizes_mb"] = {p: round(du(p) / 1e6) for p in ("data/yearly_data", "data/era5_data_raw", "data/era5_interpolated.parquet", "data/preprocessed")}

# outage log-normal fit used for block masking (gaps >= 1 patch = 16 steps), as in aqfm.masking.fit_block_lognormal
lg = np.log(np.array(gaps)[np.array(gaps) >= 16].astype(float))
out["outages_lognormal_fit_ge16steps"] = {"mu": round(float(lg.mean()), 3), "sigma": round(float(lg.std()), 3),
                                          "median_h": round(float(np.exp(lg.mean()) * 0.25), 1)}

# verification gates: solar-radiation peak hour (IST) and per-station BP medians
sr = np.asarray(v[:, C["SR"]]); osr = np.asarray(o[:, C["SR"]]) > 0
out["sr_peak_hour_ist"] = int(np.argmax([sr[:, hour == k][osr[:, hour == k]].mean() for k in range(24)]))
bpm = [float(np.median(np.asarray(v[s_, C["BP"]])[np.asarray(o[s_, C["BP"]]) > 0])) for s_ in range(S)]
out["bp_station_median_range_mmHg"] = [round(min(bpm), 1), round(max(bpm), 1)]

# NO + NO2 = NOx identity (in ppb) and derived-fraction coverage, as computed by aqfm.physical.derive_inputs
from aqfm.physical import NO_UGM3_TO_PPB, NO2_UGM3_TO_PPB, derive_inputs
from aqfm.config import CH as MCH
ratios, pm_frac, nox_frac = [], 0, 0
for s_ in range(S):
    vv = np.asarray(v[s_], np.float64); oo = np.asarray(o[s_]) > 0
    ok = oo[C["NO"]] & oo[C["NO2"]] & oo[C["NOx"]] & (vv[C["NOx"]] > 1.0)
    ratios.append((vv[C["NO"]][ok] * NO_UGM3_TO_PPB + vv[C["NO2"]][ok] * NO2_UGM3_TO_PPB) / vv[C["NOx"]][ok])
    _, m_ = derive_inputs(np.asarray(v[s_:s_ + 1]), np.asarray(o[s_:s_ + 1]))
    pm_frac += int(m_[0, MCH["PM_FRAC"]].sum()); nox_frac += int(m_[0, MCH["NOX_FRAC"]].sum())
out["nox_identity_ratio_median_pooled"] = round(float(np.median(np.concatenate(ratios))), 4)
out["nox_identity_ratio_median_by_station"] = [round(float(np.median(r)), 3) if len(r) else None for r in ratios]
out["derived_coverage_pct"] = {"PM_FRAC": round(100 * pm_frac / (S * T), 1), "NOX_FRAC": round(100 * nox_frac / (S * T), 1)}

# analyser ceilings: all years and the 2025 test year
out["ceilings"] = {}
for n in ("PM2.5", "PM10"):
    x, m = series(n)
    t25 = year == 2025
    out["ceilings"][n] = {"at_1000_all_years": int(((x == 1000) & m).sum()),
                          "at_or_above_1000_2025": int(((x >= 1000) & m)[:, t25].sum()),
                          "severe_2025": int(((x >= 250) & m)[:, t25].sum())}
idx = np.arange(int(np.searchsorted(t, (pd.Timestamp("2025-01-01") - pd.Timestamp("2018-01-01")).total_seconds() / 3600)), T - 288, 96) + 95
out["ceilings"]["PM2.5_24h_scoring_points_2025"] = {"severe": int(((pm25[:, idx] >= 250) & m25[:, idx]).sum()),
                                                    "at_or_above_1000": int(((pm25[:, idx] >= 1000) & m25[:, idx]).sum())}
for n in ("NO2", "NOx", "NO", "NH3", "Ozone", "SO2"):
    out.setdefault("channel_max", {})[n] = out[f"{n}_dist"]["max"]

# counts from preprocessing_report.json
out["from_preprocessing_report"] = {"bp_hpa_converted": rep["bp_hpa_converted"],
                                    "outliers_clipped": rep["sanity"]["outliers_clipped"],
                                    "pm25_gt_pm10": rep["sanity"]["pm25_gt_pm10"]}

json.dump(out, open(os.path.join(ROOT, "docs", "data_report_stats.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "stations"}, indent=1))
for s_ in sorted(st, key=lambda z: -z["pm25_mean"]):
    print(s_)
