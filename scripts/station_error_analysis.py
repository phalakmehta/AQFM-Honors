"""Spatial station error analysis for AQFM forecasts.

Analyzes station-level forecast performance:
1. Per-station MAE, RMSE, and bias across lead horizons
2. Identification of outlier stations (>1.5x median error)
3. Spatial map export (lat/lon, MAE, bias, coverage)
4. Correlation between error and station physical/geographical characteristics
5. Hierarchical error-profile clustering across stations
6. Monthly / seasonal variation for worst-performing stations
"""

import argparse
from datetime import datetime, timedelta
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as sdist
import scipy.stats as stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from aqfm.config import EPOCH
from baselines.baseline_utils import HORIZONS


def load_forecasts(path):
    """Load a saved forecast .npz file."""
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def haversine_km(coords):
    """Compute pairwise haversine distance in km from (S, 2) [lat, lon]."""
    la1, lo1 = np.radians(coords[:, 0])[:, None], np.radians(coords[:, 1])[:, None]
    la2, lo2 = np.radians(coords[:, 0])[None, :], np.radians(coords[:, 1])[None, :]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def per_station_metrics(forecast_dict, target="PM2.5", leads=("1h", "6h", "24h", "72h")):
    """Compute MAE, RMSE, Bias, and observation counts per anchor station and lead."""
    pred = forecast_dict[f"{target}_pred"]
    true = forecast_dict[f"{target}_true"]
    mask = forecast_dict.get(f"{target}_mask", np.ones_like(true))
    anchor = forecast_dict["anchor"]
    unique_stations = np.unique(anchor)

    results = {}
    for s in unique_stations:
        s_mask = anchor == s
        s_res = {}
        for lead in leads:
            j = HORIZONS[lead] - 1
            yt = true[s_mask, j]
            yp = pred[s_mask, j]
            m = mask[s_mask, j] > 0.5
            n = int(np.sum(m))
            if n > 0:
                err = yp[m] - yt[m]
                s_res[lead] = {
                    "MAE": round(float(np.mean(np.abs(err))), 4),
                    "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
                    "Bias": round(float(np.mean(err)), 4),
                    "n": n,
                }
            else:
                s_res[lead] = {"MAE": None, "RMSE": None, "Bias": None, "n": 0}
        results[int(s)] = s_res
    return results


def analyze_station_characteristics(station_metrics, facts_df, coords=None, target_lead="24h"):
    """Correlate 24h MAE with physical station characteristics and spatial isolation."""
    st_ids = list(station_metrics.keys())
    maes = []
    valid_ids = []
    for sid in st_ids:
        m = station_metrics[sid].get(target_lead, {}).get("MAE")
        if m is not None:
            maes.append(m)
            valid_ids.append(sid)

    maes = np.array(maes)
    median_mae = float(np.median(maes))
    outlier_threshold = 1.5 * median_mae

    station_summary = []
    for i, sid in enumerate(valid_ids):
        mae = maes[i]
        meta = facts_df.iloc[sid].to_dict() if facts_df is not None and sid < len(facts_df) else {}
        station_summary.append({
            "station_index": sid,
            "station_id": meta.get("station_id", f"station_{sid}"),
            "name": meta.get("name", meta.get("cpcb_name", f"Station {sid}")),
            "lat": float(meta.get("lat", coords[sid, 0] if coords is not None else 0.0)),
            "lon": float(meta.get("lon", coords[sid, 1] if coords is not None else 0.0)),
            "MAE_24h": round(float(mae), 3),
            "is_outlier": bool(mae > outlier_threshold),
            "elevation_m": meta.get("elevation_m"),
            "built_frac_1km": meta.get("built_frac_1km"),
            "green_frac_1km": meta.get("green_frac_1km"),
            "coast_km": meta.get("coast_km"),
        })

    # Nearest neighbor distances
    if coords is not None and len(coords) == len(station_summary):
        dists = haversine_km(coords)
        np.fill_diagonal(dists, np.inf)
        nn_dists = np.min(dists, axis=1)
        for i, s in enumerate(station_summary):
            s["nn_dist_km"] = round(float(nn_dists[s["station_index"]]), 2)

    df_summary = pd.DataFrame(station_summary)

    # Compute Spearman rank correlations with MAE_24h
    correlations = {}
    candidate_features = ["elevation_m", "built_frac_1km", "green_frac_1km", "coast_km", "nn_dist_km"]
    for feat in candidate_features:
        if feat in df_summary.columns and df_summary[feat].notna().sum() >= 5:
            valid_sub = df_summary[["MAE_24h", feat]].dropna()
            corr, pval = stats.spearmanr(valid_sub["MAE_24h"], valid_sub[feat])
            correlations[feat] = {
                "spearman_rho": round(float(corr), 4),
                "p_value": round(float(pval), 4),
            }

    return {
        "median_MAE_24h": round(median_mae, 3),
        "outlier_threshold": round(outlier_threshold, 3),
        "outliers": [s for s in station_summary if s["is_outlier"]],
        "correlations": correlations,
        "stations": station_summary,
    }


def cluster_stations_by_error(station_metrics, leads=("1h", "6h", "24h", "72h"), n_clusters=3):
    """Cluster stations based on their error profile across lead horizons."""
    st_ids = sorted(station_metrics.keys())
    profiles = []
    valid_ids = []
    for sid in st_ids:
        vec = [station_metrics[sid].get(l, {}).get("MAE") for l in leads]
        if all(v is not None for v in vec):
            profiles.append(vec)
            valid_ids.append(sid)

    if len(profiles) < n_clusters or len(profiles) < 4:
        return {"error": "insufficient_stations_for_clustering"}

    X = np.array(profiles)
    # Ward linkage
    linkage = sch.linkage(X, method="ward")
    cluster_labels = sch.fcluster(linkage, t=n_clusters, criterion="maxclust")

    clusters = {}
    for c_id in range(1, n_clusters + 1):
        c_mask = cluster_labels == c_id
        c_stations = [valid_ids[i] for i in range(len(valid_ids)) if c_mask[i]]
        c_means = np.mean(X[c_mask], axis=0)
        clusters[f"cluster_{c_id}"] = {
            "station_indices": c_stations,
            "station_count": int(np.sum(c_mask)),
            "mean_profile": {l: round(float(c_means[i]), 3) for i, l in enumerate(leads)},
        }
    return clusters


def monthly_variation_worst_stations(forecast_dict, worst_station_ids, target="PM2.5", lead="24h"):
    """Monthly breakdown of forecast MAE for specified worst-performing stations."""
    pred = forecast_dict[f"{target}_pred"]
    true = forecast_dict[f"{target}_true"]
    mask = forecast_dict.get(f"{target}_mask", np.ones_like(true))
    anchor = forecast_dict["anchor"]
    origin_hours = forecast_dict["origin_hours"]
    j = HORIZONS[lead] - 1

    months = np.array([(EPOCH + timedelta(hours=float(h))).month for h in origin_hours])

    variation = {}
    for sid in worst_station_ids:
        s_mask = (anchor == sid) & (mask[:, j] > 0.5)
        monthly_res = {}
        for m in range(1, 13):
            m_mask = s_mask & (months == m)
            n_m = int(np.sum(m_mask))
            if n_m > 0:
                err = np.abs(pred[m_mask, j] - true[m_mask, j])
                monthly_res[m] = round(float(np.mean(err)), 3)
            else:
                monthly_res[m] = None
        variation[str(sid)] = monthly_res
    return variation


def run_station_analysis(forecast_dict, facts_path=None, stations_path=None, target="PM2.5"):
    """Execute complete spatial station error analysis."""
    facts_df = None
    if facts_path and os.path.exists(facts_path):
        facts_df = pd.read_csv(facts_path)
    elif stations_path and os.path.exists(stations_path):
        facts_df = pd.read_csv(stations_path)

    coords = None
    if facts_df is not None and "lat" in facts_df.columns and "lon" in facts_df.columns:
        coords = facts_df[["lat", "lon"]].values.astype(np.float32)

    st_metrics = per_station_metrics(forecast_dict, target=target)
    char_analysis = analyze_station_characteristics(st_metrics, facts_df, coords=coords, target_lead="24h")
    clusters = cluster_stations_by_error(st_metrics, n_clusters=3)

    # Top 3 worst stations by 24h MAE
    sorted_stations = sorted(char_analysis["stations"], key=lambda x: x["MAE_24h"], reverse=True)
    worst_3_ids = [s["station_index"] for s in sorted_stations[:3]]
    worst_monthly = monthly_variation_worst_stations(forecast_dict, worst_3_ids, target=target, lead="24h")

    return {
        "target": target,
        "station_metrics": st_metrics,
        "characteristics_and_outliers": char_analysis,
        "clusters": clusters,
        "worst_stations_monthly": worst_monthly,
    }


def format_markdown_report(results, title="Spatial Station Error Analysis"):
    """Generate markdown report for spatial error analysis."""
    md = [f"# {title}\n"]
    target = results.get("target", "PM2.5")
    char_info = results.get("characteristics_and_outliers", {})
    stations = char_info.get("stations", [])
    clusters = results.get("clusters", {})
    corrs = char_info.get("correlations", {})

    md.append(f"Target Pollutant: **{target}**\n")
    md.append(f"- City Median 24h MAE: **{char_info.get('median_MAE_24h')} µg/m³**")
    md.append(f"- Outlier Threshold (>1.5×): **{char_info.get('outlier_threshold')} µg/m³**\n")

    # 1. Outlier Stations
    md.append("### 1. High-Error / Outlier Stations (24h Lead)\n")
    outliers = char_info.get("outliers", [])
    if outliers:
        md.append("| Station Index | Station ID | 24h MAE (µg/m³) | Elevation (m) | Built-up Share |")
        md.append("|---|---|---|---|---|")
        for o in outliers:
            md.append(f"| {o['station_index']} | {o.get('name', o['station_id'])} | {o['MAE_24h']} | {o.get('elevation_m', '-')} | {o.get('built_frac_1km', '-')} |")
    else:
        md.append("No stations exceeded the 1.5× median error threshold.")
    md.append("\n")

    # 2. Correlation with Physical Characteristics
    md.append("### 2. Correlation of 24h Error with Station Characteristics\n")
    md.append("| Characteristic | Spearman Rank Correlation (ρ) | p-value |")
    md.append("|---|---|---|")
    for feat, stats_dict in corrs.items():
        md.append(f"| {feat} | {stats_dict['spearman_rho']} | {stats_dict['p_value']} |")
    md.append("\n")

    # 3. Station Error Clusters
    md.append("### 3. Error-Profile Station Clusters\n")
    md.append("| Cluster | Count | Mean 1h MAE | Mean 6h MAE | Mean 24h MAE | Mean 72h MAE |")
    md.append("|---|---|---|---|---|---|")
    for c_name, c_dict in clusters.items():
        if isinstance(c_dict, dict) and "mean_profile" in c_dict:
            mp = c_dict["mean_profile"]
            md.append(f"| {c_name} | {c_dict['station_count']} | {mp.get('1h')} | {mp.get('6h')} | {mp.get('24h')} | {mp.get('72h')} |")
    md.append("\n")

    # 4. Top 10 Best and Worst Stations
    sorted_st = sorted(stations, key=lambda x: x["MAE_24h"])
    md.append("### 4. Station Performance Ranking (Top 5 Best vs Worst 24h MAE)\n")
    md.append("**Best Performing Stations:**\n")
    md.append("| Station Index | Station ID | 24h MAE (µg/m³) |")
    md.append("|---|---|---|")
    for s in sorted_st[:5]:
        md.append(f"| {s['station_index']} | {s.get('name', s['station_id'])} | {s['MAE_24h']} |")
    md.append("\n**Worst Performing Stations:**\n")
    md.append("| Station Index | Station ID | 24h MAE (µg/m³) |")
    md.append("|---|---|---|")
    for s in sorted_st[-5:][::-1]:
        md.append(f"| {s['station_index']} | {s.get('name', s['station_id'])} | {s['MAE_24h']} |")
    md.append("\n")

    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description="Spatial station error analysis.")
    ap.add_argument("--forecast", required=True, help="Path to .npz forecast file")
    ap.add_argument("--facts", default="data_pipeline/stations/delhi_facts.csv", help="Station facts CSV")
    ap.add_argument("--stations", default="data_pipeline/stations/delhi.csv", help="Station metadata CSV")
    ap.add_argument("--target", default="PM2.5", help="Target pollutant to score")
    ap.add_argument("--out", default="results/paper/station_error_analysis.json", help="Output JSON path")
    ap.add_argument("--out-csv", default="results/paper/station_error_map.csv", help="Output spatial map CSV")
    ap.add_argument("--report", default="results/paper/station_error_analysis.md", help="Output markdown report path")
    args = ap.parse_args()

    print(f"Loading forecast from {args.forecast}...")
    f_dict = load_forecasts(args.forecast)

    print(f"Running station error analysis for {args.target}...")
    results = run_station_analysis(
        f_dict, facts_path=args.facts, stations_path=args.stations, target=args.target
    )

    # Save JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved analysis JSON to {args.out}")

    # Save map CSV
    if args.out_csv:
        stations_list = results["characteristics_and_outliers"]["stations"]
        df_map = pd.DataFrame(stations_list)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        df_map.to_csv(args.out_csv, index=False)
        print(f"Saved spatial map CSV to {args.out_csv}")

    # Save report
    if args.report:
        md = format_markdown_report(results)
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            f.write(md)
        print(f"Saved markdown report to {args.report}")


if __name__ == "__main__":
    main()
