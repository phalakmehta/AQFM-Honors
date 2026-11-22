"""Missing value imputation and input missingness impact analysis for AQFM.

Evaluates how varying amounts and patterns of missing input data affect forecast accuracy:
1. Forecast quality as a function of context completeness (binned by % observed)
2. Per-channel missingness impact (evaluating loss in accuracy when key channels are missing)
3. Neighbour compensation effect (how neighbours mitigate local sensor missingness)
4. Station-level missingness profiles correlated with forecast error
5. Simulated artificial masking experiment on high-completeness windows
"""

import argparse
from datetime import datetime, timedelta
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from aqfm.config import AQFMConfig, CH, RAW_CHANNELS
from aqfm.data import AQData, build_queries, make_sample
from aqfm.train import load_checkpoint
from baselines.baseline_utils import HORIZONS


def load_forecasts(path):
    """Load forecast .npz file."""
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def compute_context_completeness(observed, anchor, origin, context_steps=512, channel_idx=0):
    """Compute observation completeness fraction in the [origin - context_steps, origin) context."""
    n_windows = len(anchor)
    completeness = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        a = anchor[i]
        o = origin[i]
        start = max(0, o - context_steps)
        ctx = observed[a, channel_idx, start:o]
        completeness[i] = float(np.mean(ctx > 0)) if len(ctx) > 0 else 0.0
    return completeness


def analyze_completeness_bins(forecast_dict, completeness_arr, target="PM2.5",
                              bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                              leads=("1h", "6h", "24h", "72h")):
    """Compute forecast MAE grouped by context completeness bins."""
    pred = forecast_dict[f"{target}_pred"]
    true = forecast_dict[f"{target}_true"]
    mask = forecast_dict.get(f"{target}_mask", np.ones_like(true))

    results = {}
    for b_idx in range(len(bins) - 1):
        b_lo, b_hi = bins[b_idx], bins[b_idx + 1]
        bin_name = f"{int(b_lo*100)}-{int(b_hi*100)}%"
        in_bin = (completeness_arr >= b_lo) & (completeness_arr <= b_hi if b_idx == len(bins) - 2 else completeness_arr < b_hi)

        bin_res = {"count": int(np.sum(in_bin))}
        for lead in leads:
            j = HORIZONS[lead] - 1
            valid = in_bin & (mask[:, j] > 0.5)
            n_v = int(np.sum(valid))
            if n_v > 0:
                err = np.abs(pred[valid, j] - true[valid, j])
                bin_res[lead] = {
                    "MAE": round(float(np.mean(err)), 3),
                    "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 3),
                    "n": n_v,
                }
            else:
                bin_res[lead] = {"MAE": None, "RMSE": None, "n": 0}
        results[bin_name] = bin_res
    return results


def analyze_channel_missing_impact(observed, forecast_dict, anchor, origin,
                                   channels=("AT", "RH", "WS", "BP", "SR"),
                                   context_steps=512, target="PM2.5", lead="24h"):
    """Compare forecast MAE when individual channels have low vs high completeness."""
    pred = forecast_dict[f"{target}_pred"]
    true = forecast_dict[f"{target}_true"]
    mask = forecast_dict.get(f"{target}_mask", np.ones_like(true))
    j = HORIZONS[lead] - 1

    R = {name: i for i, name in enumerate(RAW_CHANNELS)}
    channel_impact = {}

    for ch_name in channels:
        if ch_name not in R:
            continue
        c_idx = R[ch_name]
        comp = compute_context_completeness(observed, anchor, origin, context_steps=context_steps, channel_idx=c_idx)

        low_mask = (comp < 0.5) & (mask[:, j] > 0.5)
        high_mask = (comp >= 0.9) & (mask[:, j] > 0.5)

        low_mae = float(np.mean(np.abs(pred[low_mask, j] - true[low_mask, j]))) if np.sum(low_mask) > 0 else None
        high_mae = float(np.mean(np.abs(pred[high_mask, j] - true[high_mask, j]))) if np.sum(high_mask) > 0 else None

        channel_impact[ch_name] = {
            "low_completeness_MAE (<50%)": round(low_mae, 3) if low_mae is not None else None,
            "low_count": int(np.sum(low_mask)),
            "high_completeness_MAE (>=90%)": round(high_mae, 3) if high_mae is not None else None,
            "high_count": int(np.sum(high_mask)),
            "delta_MAE": round(low_mae - high_mae, 3) if (low_mae is not None and high_mae is not None) else None,
        }
    return channel_impact


def analyze_neighbour_compensation(observed, coords, forecast_dict, anchor, origin,
                                   context_steps=512, target="PM2.5", lead="24h", k_neighbors=3):
    """Assess whether high completeness in nearest neighbours compensates for low anchor completeness."""
    pred = forecast_dict[f"{target}_pred"]
    true = forecast_dict[f"{target}_true"]
    mask = forecast_dict.get(f"{target}_mask", np.ones_like(true))
    j = HORIZONS[lead] - 1

    R = {name: i for i, name in enumerate(RAW_CHANNELS)}
    c_pm = R.get(target, 0)

    # Compute anchor completeness
    anchor_comp = compute_context_completeness(observed, anchor, origin, context_steps=context_steps, channel_idx=c_pm)

    # Compute pairwise distances
    la1, lo1 = np.radians(coords[:, 0])[:, None], np.radians(coords[:, 1])[:, None]
    la2, lo2 = np.radians(coords[:, 0])[None, :], np.radians(coords[:, 1])[None, :]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    dists = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

    n_windows = len(anchor)
    nbr_comp = np.zeros(n_windows, dtype=np.float32)

    for i in range(n_windows):
        a = anchor[i]
        o = origin[i]
        start = max(0, o - context_steps)
        # Find k nearest neighbours (excluding self)
        nearest = np.argsort(dists[a])[1:k_neighbors + 1]
        nbr_ctx = observed[nearest, c_pm, start:o]
        nbr_comp[i] = float(np.mean(nbr_ctx > 0)) if nbr_ctx.size > 0 else 0.0

    # Low anchor completeness (<40%)
    low_anchor = (anchor_comp < 0.4) & (mask[:, j] > 0.5)
    high_nbrs = low_anchor & (nbr_comp >= 0.75)
    low_nbrs = low_anchor & (nbr_comp < 0.50)

    mae_high_nbrs = float(np.mean(np.abs(pred[high_nbrs, j] - true[high_nbrs, j]))) if np.sum(high_nbrs) > 0 else None
    mae_low_nbrs = float(np.mean(np.abs(pred[low_nbrs, j] - true[low_nbrs, j]))) if np.sum(low_nbrs) > 0 else None

    return {
        "low_anchor_count": int(np.sum(low_anchor)),
        "assisted_by_high_nbr_completeness": {
            "MAE": round(mae_high_nbrs, 3) if mae_high_nbrs is not None else None,
            "count": int(np.sum(high_nbrs)),
        },
        "isolated_low_nbr_completeness": {
            "MAE": round(mae_low_nbrs, 3) if mae_low_nbrs is not None else None,
            "count": int(np.sum(low_nbrs)),
        },
        "compensation_benefit_MAE": round(mae_low_nbrs - mae_high_nbrs, 3) if (mae_low_nbrs and mae_high_nbrs) else None,
    }


def run_simulated_masking(model, cfg, data, raw_values, raw_observed,
                          mask_rates=(0.0, 0.2, 0.4, 0.6, 0.8),
                          device="cpu", n_windows=32, target="PM2.5", lead="24h"):
    """Evaluate model degradation when artificially corrupting PM2.5 context observations."""
    model.eval()
    Tc, Th = cfg.context_steps, cfg.horizon_steps
    j = HORIZONS[lead] - 1
    R = {name: i for i, name in enumerate(RAW_CHANNELS)}
    c_idx = R[target]

    # Select windows with high initial completeness
    valid_pairs = []
    for o in range(Tc + 500, min(data.T - Th, Tc + 5000), 100):
        for a in range(data.S):
            ctx = raw_observed[a, c_idx, o - Tc:o]
            if np.mean(ctx > 0) > 0.90:
                valid_pairs.append((o - Tc, a))
                if len(valid_pairs) >= n_windows:
                    break
        if len(valid_pairs) >= n_windows:
            break

    if not valid_pairs:
        return {"error": "no_high_completeness_windows_found"}

    degradation_results = {}
    for rate in mask_rates:
        errs = []
        for t0, a in valid_pairs:
            sel = data.neighbourhood(a, cfg.n_neighbours, cfg.neighbour_radius_km)
            sample = make_sample(data, cfg, sel, t0)

            # Artificially mask anchor's PM2.5 context
            if rate > 0.0:
                m_ctx = sample["m_ctx"].clone()
                x_ctx = sample["x_ctx"].clone()
                # Anchor is slot 0
                drop_mask = torch.rand(m_ctx.shape[-1]) < rate
                m_ctx[0, CH[target], drop_mask] = 0.0
                x_ctx[0, CH[target], drop_mask] = 0.0
                sample["m_ctx"] = m_ctx
                sample["x_ctx"] = x_ctx

            batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
            with torch.no_grad():
                q, _, _ = build_queries(cfg, batch, impute=False)
                out = model(batch, q)
                # Output forecast for target
                mu = out["mu"]  # (B, Q, P, K)
                # Get median forecast
                pred_val = mu.mean(dim=(-1, -2)).cpu().numpy().flatten()[0]
                true_val = raw_values[a, c_idx, t0 + Tc + j]
                errs.append(abs(pred_val - true_val))

        degradation_results[f"{int(rate*100)}%_masked"] = {
            "rate": rate,
            "mean_abs_error": round(float(np.mean(errs)), 3),
            "n_samples": len(errs),
        }

    return degradation_results


def format_markdown_report(results, title="Missing Value Imputation Impact Analysis"):
    """Format results into a markdown report."""
    md = [f"# {title}\n"]

    # 1. Completeness Bins
    if "completeness_bins" in results:
        md.append("### 1. Forecast Error vs Context Completeness\n")
        md.append("| Completeness Bin | Count | 1h MAE | 6h MAE | 24h MAE | 72h MAE |")
        md.append("|---|---|---|---|---|---|")
        for b_name, b_data in results["completeness_bins"].items():
            md.append(f"| {b_name} | {b_data.get('count', 0)} | "
                      f"{b_data.get('1h', {}).get('MAE', '-')} | "
                      f"{b_data.get('6h', {}).get('MAE', '-')} | "
                      f"{b_data.get('24h', {}).get('MAE', '-')} | "
                      f"{b_data.get('72h', {}).get('MAE', '-')} |")
        md.append("\n")

    # 2. Channel Missing Impact
    if "channel_impact" in results:
        md.append("### 2. Impact of Missing Specific Meteorological Channels (24h Lead)\n")
        md.append("| Channel | Low Obs MAE (<50%) | High Obs MAE (>=90%) | Error Increase (Δ MAE) |")
        md.append("|---|---|---|---|")
        for ch, ch_data in results["channel_impact"].items():
            md.append(f"| {ch} | {ch_data.get('low_completeness_MAE (<50%)', '-')} | "
                      f"{ch_data.get('high_completeness_MAE (>=90%)', '-')} | "
                      f"**{ch_data.get('delta_MAE', '-')}** |")
        md.append("\n")

    # 3. Neighbour Compensation
    if "neighbour_compensation" in results:
        nc = results["neighbour_compensation"]
        md.append("### 3. Spatial Neighbour Compensation for Local Sensor Outages\n")
        md.append(f"- Windows with low anchor completeness (<40%): **{nc.get('low_anchor_count')}**\n")
        md.append("| Scenario | 24h MAE | Sample Count |")
        md.append("|---|---|---|")
        high_n = nc.get("assisted_by_high_nbr_completeness", {})
        low_n = nc.get("isolated_low_nbr_completeness", {})
        md.append(f"| High Neighbour Completeness (>=75%) | {high_n.get('MAE', '-')} | {high_n.get('count', 0)} |")
        md.append(f"| Low Neighbour Completeness (<50%) | {low_n.get('MAE', '-')} | {low_n.get('count', 0)} |")
        md.append(f"\nCompensation benefit (MAE reduction): **{nc.get('compensation_benefit_MAE', '-')} µg/m³**\n\n")

    # 4. Simulated Masking
    if "simulated_masking" in results and not results["simulated_masking"].get("error"):
        md.append("### 4. Controlled Artificial Context Masking Experiment\n")
        md.append("| Masking Rate | Mean Error |")
        md.append("|---|---|")
        for m_rate, m_dict in results["simulated_masking"].items():
            md.append(f"| {m_rate} | {m_dict.get('mean_abs_error', '-')} |")
        md.append("\n")

    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description="Missing value and imputation impact analysis.")
    ap.add_argument("--forecast", required=True, help="Path to .npz forecast file")
    ap.add_argument("--data", default="data/preprocessed", help="Path to preprocessed data directory")
    ap.add_argument("--target", default="PM2.5", help="Target pollutant")
    ap.add_argument("--checkpoint", default=None, help="Optional model checkpoint for simulated masking")
    ap.add_argument("--device", default="cpu", help="Device for model inference (cpu or cuda)")
    ap.add_argument("--out", default="results/paper/missing_value_impact.json", help="Output JSON path")
    ap.add_argument("--report", default="results/paper/missing_value_impact.md", help="Output markdown report path")
    args = ap.parse_args()

    print(f"Loading forecast: {args.forecast}...")
    f_dict = load_forecasts(args.forecast)

    anchor = f_dict["anchor"]
    origin = f_dict["origin"]

    print(f"Loading data store from {args.data}...")
    observed = np.load(os.path.join(args.data, "observed.npy"), mmap_mode="r")
    coords = np.load(os.path.join(args.data, "coords.npy"))

    # 1. Context completeness bins
    print("Analyzing forecast quality across completeness bins...")
    comp_arr = compute_context_completeness(observed, anchor, origin, channel_idx=0)
    bins_res = analyze_completeness_bins(f_dict, comp_arr, target=args.target)

    # 2. Per-channel missing impact
    print("Analyzing per-channel missingness impact...")
    channel_res = analyze_channel_missing_impact(observed, f_dict, anchor, origin, target=args.target)

    # 3. Neighbour compensation
    print("Analyzing neighbour compensation effect...")
    nbr_res = analyze_neighbour_compensation(observed, coords, f_dict, anchor, origin, target=args.target)

    # 4. Simulated masking (optional, if checkpoint provided)
    sim_res = {}
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Running simulated masking experiment using {args.checkpoint} on {args.device}...")
        model, cfg, _ = load_checkpoint(args.checkpoint, device=args.device)
        values = np.load(os.path.join(args.data, "values.npy"), mmap_mode="r")
        timestamps_h = np.load(os.path.join(args.data, "timestamps_h.npy"))
        static = np.load(os.path.join(args.data, "static.npy"))
        met_p = os.path.join(args.data, "era5_met.npy")
        era5 = np.load(met_p, mmap_mode="r") if os.path.exists(met_p) else None

        aqdata = AQData(values, observed, timestamps_h, coords, static, train_end=200000, era5_met=era5)
        sim_res = run_simulated_masking(model, cfg, aqdata, values, observed, device=args.device)

    all_results = {
        "target": args.target,
        "completeness_bins": bins_res,
        "channel_impact": channel_res,
        "neighbour_compensation": nbr_res,
        "simulated_masking": sim_res,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved analysis JSON to {args.out}")

    if args.report:
        md = format_markdown_report(all_results)
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            f.write(md)
        print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
