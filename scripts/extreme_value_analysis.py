"""Extreme value and tail error analysis for AQFM forecasts.

Loads forecast .npz files saved by `aqfm.evaluate --save-forecasts` and evaluates:
1. Tail errors conditioned on true value exceeding p90, p95, p99 percentiles
2. Conditional 90% interval coverage across AQI bands and deciles
3. Generalized Pareto Distribution (GPD) fitting to error peaks (POT method)
4. Exceedance detection (precision/recall/F1) at critical pollutant thresholds
5. Seasonal decomposition of tail errors
6. Worst-case forecast windows identification
"""

import argparse
from datetime import datetime, timedelta
import json
import os
import sys

import numpy as np
import scipy.stats as stats

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from aqfm.config import EPOCH, MONTH_TO_SEASON
from baselines.baseline_utils import AQI_BANDS, HORIZONS

SEASONS = {0: "Winter", 1: "Summer", 2: "Monsoon", 3: "Post-Monsoon"}


def load_forecasts(path):
    """Load a saved forecast .npz file into a dict of numpy arrays."""
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def get_lead_index(lead_str):
    """Return 0-based step index for a given horizon string (e.g. '24h' -> 95)."""
    assert lead_str in HORIZONS, f"Unknown lead: {lead_str}. Choose from {list(HORIZONS.keys())}"
    return HORIZONS[lead_str] - 1


def tail_error_analysis(y_true, y_pred, mask, percentiles=(90, 95, 99)):
    """MAE, RMSE, Bias conditioned on true value >= p-th percentile."""
    valid = mask > 0.5
    y_t = y_true[valid]
    y_p = y_pred[valid]
    n_total = len(y_t)
    if n_total == 0:
        return {}

    res = {"overall": {
        "MAE": round(float(np.mean(np.abs(y_p - y_t))), 4),
        "RMSE": round(float(np.sqrt(np.mean((y_p - y_t) ** 2))), 4),
        "Bias": round(float(np.mean(y_p - y_t)), 4),
        "n": n_total,
    }}

    for p in percentiles:
        thresh = float(np.percentile(y_t, p))
        sel = y_t >= thresh
        n_p = int(np.sum(sel))
        if n_p > 0:
            err = y_p[sel] - y_t[sel]
            res[f"p{p}"] = {
                "threshold": round(thresh, 2),
                "MAE": round(float(np.mean(np.abs(err))), 4),
                "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
                "Bias": round(float(np.mean(err)), 4),
                "n": n_p,
            }
        else:
            res[f"p{p}"] = {"threshold": round(thresh, 2), "MAE": None, "RMSE": None, "Bias": None, "n": 0}
    return res


def conditional_coverage_analysis(y_true, y_lo, y_hi, mask):
    """Interval coverage broken down by AQI bands and by true value deciles."""
    valid = mask > 0.5
    y_t = y_true[valid]
    covered = (y_t >= y_lo[valid]) & (y_t <= y_hi[valid])
    n_total = len(y_t)
    if n_total == 0:
        return {}

    by_aqi = {}
    for name, (lo, hi) in AQI_BANDS.items():
        sel = (y_t >= lo) & (y_t < hi)
        n_b = int(np.sum(sel))
        by_aqi[name] = {
            "coverage90": round(float(np.mean(covered[sel])), 4) if n_b > 0 else None,
            "n": n_b,
        }

    by_decile = {}
    if n_total >= 10:
        deciles = np.percentile(y_t, np.linspace(0, 100, 11))
        for d in range(10):
            d_lo, d_hi = deciles[d], deciles[d + 1]
            sel = (y_t >= d_lo) & (y_t <= d_hi) if d == 9 else (y_t >= d_lo) & (y_t < d_hi)
            n_d = int(np.sum(sel))
            by_decile[f"D{d+1}"] = {
                "range": [round(float(d_lo), 1), round(float(d_hi), 1)],
                "coverage90": round(float(np.mean(covered[sel])), 4) if n_d > 0 else None,
                "n": n_d,
            }

    return {
        "overall_coverage90": round(float(np.mean(covered)), 4),
        "by_aqi_band": by_aqi,
        "by_decile": by_decile,
    }


def fit_gpd_errors(y_true, y_pred, mask, threshold_p=95.0, return_periods=(100, 500, 1000)):
    """Peaks-Over-Threshold (POT) GPD fit on forecast absolute errors."""
    valid = mask > 0.5
    y_t = y_true[valid]
    y_p = y_pred[valid]
    if len(y_t) < 30:
        return {"error": "insufficient_data"}

    abs_err = np.abs(y_p - y_t)
    u = float(np.percentile(abs_err, threshold_p))
    exceedances = abs_err[abs_err > u] - u

    if len(exceedances) < 10:
        return {"threshold": round(u, 2), "n_exceedances": len(exceedances), "fit": "insufficient_exceedances"}

    # Scipy parameterization: c = shape (xi), scale = sigma
    c, loc, scale = stats.genpareto.fit(exceedances, floc=0)

    # Return level y_T for return period T (in units of exceedance counts)
    return_levels = {}
    for T in return_periods:
        if abs(c) < 1e-4:
            y_T = scale * np.log(T)
        else:
            y_T = (scale / c) * (T ** c - 1.0)
        return_levels[f"T_{T}"] = round(float(u + max(0.0, y_T)), 2)

    return {
        "threshold_u": round(u, 2),
        "n_exceedances": int(len(exceedances)),
        "shape_xi": round(float(c), 4),
        "scale_sigma": round(float(scale), 4),
        "return_levels": return_levels,
    }


def exceedance_detection(y_true, y_pred, mask, thresholds=(60.0, 90.0, 120.0, 250.0, 350.0)):
    """Precision, recall, F1 of predicting threshold exceedance from median forecast."""
    valid = mask > 0.5
    y_t = y_true[valid]
    y_p = y_pred[valid]
    res = {}
    for th in thresholds:
        true_pos = (y_t >= th) & (y_p >= th)
        false_pos = (y_t < th) & (y_p >= th)
        false_neg = (y_t >= th) & (y_p < th)
        tp, fp, fn = int(np.sum(true_pos)), int(np.sum(false_pos)), int(np.sum(false_neg))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        res[f"thresh_{int(th)}"] = {
            "threshold": th,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "tp": tp, "fp": fp, "fn": fn,
            "true_exceedances": tp + fn,
        }
    return res


def seasonal_decomposition(y_true, y_pred, mask, origin_hours, percentiles=(90,)):
    """Tail error metrics decomposed by season."""
    valid = mask > 0.5
    seasons_arr = np.array([
        MONTH_TO_SEASON[(EPOCH + timedelta(hours=float(h))).month]
        for h in origin_hours
    ])

    res = {}
    for s_idx, s_name in SEASONS.items():
        s_mask = (seasons_arr == s_idx) & valid
        if np.sum(s_mask) == 0:
            res[s_name] = {"n": 0}
            continue
        err = np.abs(y_pred[s_mask] - y_true[s_mask])
        y_s_true = y_true[s_mask]
        s_res = {
            "MAE": round(float(np.mean(err)), 4),
            "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "mean_true": round(float(np.mean(y_s_true)), 2),
            "n": int(np.sum(s_mask)),
        }
        for p in percentiles:
            p_val = float(np.percentile(y_s_true, p))
            p_sel = y_s_true >= p_val
            s_res[f"p{p}_MAE"] = round(float(np.mean(err[p_sel])), 4) if np.sum(p_sel) > 0 else None
        res[s_name] = s_res
    return res


def identify_worst_windows(y_true, y_pred, mask, anchor, origin_hours, top_n=20):
    """Find top N largest absolute errors for a lead step."""
    valid = mask > 0.5
    err = np.abs(y_pred - y_true)
    err_filtered = np.where(valid, err, -1.0)
    sorted_indices = np.argsort(err_filtered)[::-1][:top_n]

    worst = []
    for idx in sorted_indices:
        if err_filtered[idx] < 0:
            break
        dt = EPOCH + timedelta(hours=float(origin_hours[idx]))
        val_true = float(y_true[idx])
        val_pred = float(y_pred[idx])

        # AQI band
        band = "Unknown"
        for b_name, (lo, hi) in AQI_BANDS.items():
            if lo <= val_true < hi:
                band = b_name
                break

        worst.append({
            "window_index": int(idx),
            "origin_datetime": dt.strftime("%Y-%m-%d %H:%M"),
            "anchor_station": int(anchor[idx]),
            "season": SEASONS[MONTH_TO_SEASON[dt.month]],
            "true": round(val_true, 2),
            "predicted": round(val_pred, 2),
            "abs_error": round(float(err[idx]), 2),
            "aqi_band": band,
        })
    return worst


def run_analysis(forecast_dict, target_pollutants=("PM2.5", "PM10"), leads=("1h", "6h", "24h", "72h")):
    """Run full extreme value analysis on a forecast dictionary."""
    anchor = forecast_dict.get("anchor")
    origin_hours = forecast_dict.get("origin_hours")

    results = {}
    for poll in target_pollutants:
        pred_key = f"{poll}_pred"
        true_key = f"{poll}_true"
        mask_key = f"{poll}_mask"
        lo90_key = f"{poll}_lo90"
        hi90_key = f"{poll}_hi90"

        if pred_key not in forecast_dict or true_key not in forecast_dict:
            continue

        pred = forecast_dict[pred_key]
        true = forecast_dict[true_key]
        mask = forecast_dict.get(mask_key, np.ones_like(true))
        has_intervals = (lo90_key in forecast_dict) and (hi90_key in forecast_dict)
        lo90 = forecast_dict.get(lo90_key)
        hi90 = forecast_dict.get(hi90_key)

        poll_res = {}
        for lead in leads:
            step_idx = get_lead_index(lead)
            y_t = true[:, step_idx]
            y_p = pred[:, step_idx]
            m = mask[:, step_idx]

            lead_res = {
                "tail_errors": tail_error_analysis(y_t, y_p, m),
                "gpd_fit": fit_gpd_errors(y_t, y_p, m),
                "exceedance_detection": exceedance_detection(y_t, y_p, m),
            }

            if has_intervals:
                lead_res["conditional_coverage"] = conditional_coverage_analysis(
                    y_t, lo90[:, step_idx], hi90[:, step_idx], m
                )

            if origin_hours is not None:
                lead_res["seasonal_decomposition"] = seasonal_decomposition(y_t, y_p, m, origin_hours)

            if lead == "24h" and anchor is not None and origin_hours is not None:
                lead_res["worst_windows"] = identify_worst_windows(y_t, y_p, m, anchor, origin_hours, top_n=20)

            poll_res[lead] = lead_res
        results[poll] = poll_res
    return results


def format_markdown_report(analysis_results, title="Extreme Value Analysis Report"):
    """Format the results dictionary into a readable GitHub-flavored markdown report."""
    md = [f"# {title}\n"]

    for poll, poll_data in analysis_results.items():
        md.append(f"## Pollutant: {poll}\n")

        # 1. Tail Errors Table
        md.append("### 1. Tail Errors by Lead Time\n")
        md.append("| Lead | Overall MAE | p90 MAE | p95 MAE | p99 MAE | Worst Lead Bias |")
        md.append("|---|---|---|---|---|---|")
        for lead, ldata in poll_data.items():
            te = ldata.get("tail_errors", {})
            ov = te.get("overall", {})
            p90 = te.get("p90", {})
            p95 = te.get("p95", {})
            p99 = te.get("p99", {})
            md.append(f"| {lead} | {ov.get('MAE', '-')} | {p90.get('MAE', '-')} | {p95.get('MAE', '-')} | {p99.get('MAE', '-')} | {ov.get('Bias', '-')} |")
        md.append("\n")

        # 2. GPD Fit Table
        md.append("### 2. Peaks-Over-Threshold GPD Fits (Error Tails)\n")
        md.append("| Lead | Threshold u | Shape (ξ) | Scale (σ) | 100-event Return Err | 1000-event Return Err |")
        md.append("|---|---|---|---|---|---|")
        for lead, ldata in poll_data.items():
            gpd = ldata.get("gpd_fit", {})
            rl = gpd.get("return_levels", {})
            md.append(f"| {lead} | {gpd.get('threshold_u', '-')} | {gpd.get('shape_xi', '-')} | {gpd.get('scale_sigma', '-')} | {rl.get('T_100', '-')} | {rl.get('T_1000', '-')} |")
        md.append("\n")

        # 3. Exceedance Detection at 24h
        if "24h" in poll_data and "exceedance_detection" in poll_data["24h"]:
            md.append("### 3. Exceedance Detection Performance (24h Lead)\n")
            md.append("| Threshold (µg/m³) | Precision | Recall | F1 Score | True Exceedances |")
            md.append("|---|---|---|---|---|")
            ed = poll_data["24h"]["exceedance_detection"]
            for th_k, th_v in ed.items():
                md.append(f"| {th_v.get('threshold')} | {th_v.get('precision')} | {th_v.get('recall')} | {th_v.get('f1')} | {th_v.get('true_exceedances')} |")
            md.append("\n")

        # 4. Seasonal Breakdown at 24h
        if "24h" in poll_data and "seasonal_decomposition" in poll_data["24h"]:
            md.append("### 4. Seasonal Breakdown (24h Lead)\n")
            md.append("| Season | Overall MAE | p90 MAE | Mean True | Count |")
            md.append("|---|---|---|---|---|")
            seas = poll_data["24h"]["seasonal_decomposition"]
            for s_name, s_v in seas.items():
                md.append(f"| {s_name} | {s_v.get('MAE', '-')} | {s_v.get('p90_MAE', '-')} | {s_v.get('mean_true', '-')} | {s_v.get('n', '-')} |")
            md.append("\n")

        # 5. Worst 5 Windows at 24h
        if "24h" in poll_data and "worst_windows" in poll_data["24h"]:
            md.append("### 5. Top 5 Worst-Error Windows (24h Lead)\n")
            md.append("| Date | Station | Season | AQI Band | True | Pred | Abs Error |")
            md.append("|---|---|---|---|---|---|---|")
            ww = poll_data["24h"]["worst_windows"][:5]
            for w in ww:
                md.append(f"| {w.get('origin_datetime')} | {w.get('anchor_station')} | {w.get('season')} | {w.get('aqi_band')} | {w.get('true')} | {w.get('predicted')} | {w.get('abs_error')} |")
            md.append("\n")

    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description="Analyze extreme values and error tails from forecast .npz files.")
    ap.add_argument("--forecast", required=True, nargs="+",
                    help="Paths to .npz forecast files. Format: path.npz or name=path.npz")
    ap.add_argument("--targets", nargs="+", default=["PM2.5", "PM10"], help="Targets to analyze")
    ap.add_argument("--leads", nargs="+", default=["1h", "6h", "24h", "72h"], help="Leads to analyze")
    ap.add_argument("--out", default="results/paper/extreme_value_analysis.json", help="Output JSON path")
    ap.add_argument("--report", default="results/paper/extreme_value_analysis.md", help="Output markdown report path")
    args = ap.parse_args()

    all_analyses = {}
    for entry in args.forecast:
        if "=" in entry:
            name, path = entry.split("=", 1)
        else:
            name = os.path.splitext(os.path.basename(entry))[0]
            path = entry

        print(f"Loading forecast: {name} from {path}...")
        f_dict = load_forecasts(path)
        print(f"Running extreme value analysis for {name}...")
        all_analyses[name] = run_analysis(f_dict, target_pollutants=args.targets, leads=args.leads)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_analyses, f, indent=2)
    print(f"Saved analysis JSON to {args.out}")

    # Generate markdown report
    if args.report:
        report_dir = os.path.dirname(os.path.abspath(args.report))
        os.makedirs(report_dir, exist_ok=True)
        # Use first model or combined report
        first_model = list(all_analyses.keys())[0]
        md_text = format_markdown_report(all_analyses[first_model], title=f"Extreme Value Analysis ({first_model})")
        with open(args.report, "w") as f:
            f.write(md_text)
        print(f"Saved markdown report to {args.report}")


if __name__ == "__main__":
    main()
