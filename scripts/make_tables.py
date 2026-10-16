"""Write results/TABLES.md: every reported table, generated from the results files only.

    python scripts/make_tables.py            # rewrite results/TABLES.md
    python scripts/make_tables.py --stdout   # print instead

Numbers for the paper and the docs are copied from this file, never typed by hand.
check_submission.py fails if TABLES.md is out of date with the results files.
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "results/TABLES.md"
HZ = ("1h", "6h", "24h", "72h")
BANDS = ("Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe")
STATS = "results/aqfm_do3_seeds_stats.json"
NOFM_STATS = "results/aqfm_do3_nofm_seeds_stats.json"

# (row label, results file, model name inside a baseline file; None for AQFM files, whose metrics sit
# under a section: "AQFM", "AQFM_extras" or "imputation_24h_outage". STATS = mean ± std over seeds)
ROWS = [
    ("**AQFM ensemble (3 seeds)**", "results/aqfm_do3_ens3_results.json", None),
    ("AQFM single model, mean ± std (3 seeds)", STATS, None),
    ("AQFM seed 0", "results/aqfm_do3_s0_results.json", None),
    ("AQFM seed 1", "results/aqfm_do3_s1_results.json", None),
    ("AQFM seed 2", "results/aqfm_do3_s2_results.json", None),
    ("AQFM no future met, ensemble (3 seeds)", "results/aqfm_do3_nofm_ens3_results.json", None),
    ("AQFM no future met, mean ± std (3 seeds)", NOFM_STATS, None),
    ("AQFM no future met, seed 0", "results/aqfm_do3_nofm_s0_results.json", None),
    ("AQFM no future met, seed 1", "results/aqfm_do3_nofm_s1_results.json", None),
    ("AQFM no future met, seed 2", "results/aqfm_do3_nofm_s2_results.json", None),
    ("AQFM dropout 0.1, seed 0 (2026-10-13)", "results/aqfm_anchor_s0_results.json", None),
    ("AQFM run 2: no anchor, wd 0.1, anneal", "results/aqfm_s0_wd_results.json", None),
    ("AQFM run 1: no anchor", "results/aqfm_results.json", None),
    ("XGBoost + future ERA5", "baselines/results/xgboost_future_met_results.json", "XGBoost + future ERA5"),
    ("XGBoost (no future met)", "baselines/results/xgboost_results.json", "XGBoost"),
    ("PatchTST ensemble (3 seeds)", "baselines/results/patchtst_ens3_results.json", "PatchTST"),
    ("PatchTST, mean ± std (3 seeds)", "baselines/results/patchtst_seeds_stats.json", "PatchTST"),
    ("PatchTST seed 0", "baselines/results/patchtst_s0_results.json", "PatchTST"),
    ("PatchTST seed 1", "baselines/results/patchtst_s1_results.json", "PatchTST"),
    ("PatchTST seed 2", "baselines/results/patchtst_s2_results.json", "PatchTST"),
    ("E-STGCN ensemble (3 seeds; our re-implementation)", "baselines/results/estgcn_ens3_results.json", "E-STGCN"),
    ("E-STGCN, mean ± std (3 seeds)", "baselines/results/estgcn_seeds_stats.json", "E-STGCN"),
    ("E-STGCN seed 0", "baselines/results/estgcn_s0_results.json", "E-STGCN"),
    ("E-STGCN seed 1", "baselines/results/estgcn_s1_results.json", "E-STGCN"),
    ("E-STGCN seed 2", "baselines/results/estgcn_s2_results.json", "E-STGCN"),
    ("IBM TTM (zero-shot)", "baselines/results/ttm_results.json", "IBM TTM (zero-shot)"),
    ("IBM TTM (zero-shot, 15-min scoring)", "baselines/results/ttm_results.json", "IBM TTM (zero-shot, 15-min scoring)"),
    ("Persistence", "baselines/results/classical_results.json", "Persistence"),
    ("Seasonal naive", "baselines/results/classical_results.json", "Seasonal Naive"),
    ("Climatology", "baselines/results/classical_results.json", "Climatology"),
]
AQFM_ROWS = [r for r in ROWS if r[1].startswith("results/")]   # probabilistic metrics exist only for AQFM

# Stage E ablations: final recipe, seed 0, one switch changed (scripts/pipeline_2026-10-15_ablations.sh).
# (row label, results file); the first row is the model every ablation is compared with.
ABLATIONS = [
    ("Full model (final setup), seed 0", "results/aqfm_do3_s0_results.json"),
    ("Full model, mean ± std over seeds 0/1/2 (seed noise)", STATS),
    ("E1: no future weather", "results/aqfm_do3_nofm_s0_results.json"),
    ("E2: no boundary-layer height", "results/aqfm_do3_noblh_s0_results.json"),
    ("E3: no spatial attention", "results/aqfm_do3_nospatial_s0_results.json"),
    ("E3: spatial attention by distance only (no wind)", "results/aqfm_do3_nowind_s0_results.json"),
    ("E4: PM2.5 predicted directly (not PM10 × fraction)", "results/aqfm_do3_pm25direct_s0_results.json"),
    ("E5: trained on gap-filling only (no forecast targets)", "results/aqfm_do3_gapfillonly_s0_results.json"),
    ("E7 reference: anchor, dropout 0.1, 12k-step schedule", "results/aqfm_anchor_s0_results.json"),
    ("E7: no anchor (run 1: dropout 0.1, 45k-step schedule)", "results/aqfm_results.json"),
]
# (column, section, path, decimals)
ABL_COLS = [(f"PM2.5 {h}", "AQFM", ("PM2.5", h, "MAE"), 2) for h in HZ] + [
    ("PM2.5 Severe 24h", "AQFM", ("PM2.5", "24h_aqi", "Severe", "MAE"), 1),
    ("PM10 24h", "AQFM", ("PM10", "24h", "MAE"), 2),
    ("CRPS 24h", "AQFM_extras", ("PM2.5", "24h", "crps"), 2),
    ("90% cov. 24h", "AQFM_extras", ("PM2.5", "24h", "coverage90"), 3),
    ("F1 >250 24h", "AQFM_extras", ("PM2.5", "episodes_gt250", "24h", "f1"), 3),
    ("imputation PM2.5", "imputation_24h_outage", ("PM2.5", "aqfm", "MAE"), 2),
    ("PM2.5 > PM10, forecasts", "AQFM_extras", ("pm25_gt_pm10", "forecast"), 4),
    ("PM2.5 > PM10, data", "AQFM_extras", ("pm25_gt_pm10", "observed"), 4),
]


def load(path, cache={}):
    if path not in cache:
        cache[path] = json.load(open(os.path.join(ROOT, path)))
    return cache[path]


def get(src, key, section, path, nd):
    """One table cell: the number at section/path, e.g. "AQFM", ("PM2.5", "24h", "MAE"); a baseline
    file has its model name (key) in place of the section."""
    path = (key or section,) + path
    if src.endswith("_seeds_stats.json"):
        s = load(src)["metrics"].get("/".join(path))
        return "–" if s is None else f"{s['mean']:.{nd}f} ± {s['std']:.{nd}f}"
    d = load(src)
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return "–"
        d = d[k]
    return f"{d:.{nd}f}"


def value(src, section, path):
    """The number at section/path (the mean, for a *_seeds_stats.json file); None if absent."""
    if src.endswith("_seeds_stats.json"):
        s = load(src)["metrics"].get("/".join((section,) + path))
        return None if s is None else s["mean"]
    d = load(src).get(section)
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def table(title, cols, rows, cell, note=""):
    out = [f"## {title}", "", "| model | " + " | ".join(cols) + " | source |",
           "|---" * (len(cols) + 2) + "|"]
    for label, src, key in rows:
        out.append(f"| {label} | " + " | ".join(cell(src, key, c) for c in cols) + f" | `{src}` |")
    return out + ([""] + [note] if note else []) + [""]


def ablation_tables():
    """All ablation metrics side by side, then the change in PM2.5 MAE against the full model."""
    rows = [(label, src, None) for label, src in ABLATIONS]
    out = table("2025 test: ablations (final recipe, seed 0, one change each)", [c[0] for c in ABL_COLS], rows,
                lambda s, k, col: next(get(s, k, sec, path, nd) for name, sec, path, nd in ABL_COLS if name == col),
                note="MAE and CRPS in µg/m³. Compare each row with the first; the second row's std is how much a single "
                     "seed moves each number. The E7 rows are older runs (PROJECT_CONTEXT.md §6): compare them with each "
                     "other. PM2.5 > PM10: share of point forecasts with PM2.5 > PM10 (all windows and steps) and the same "
                     "share in the observed data; models that predict PM10 × fraction cannot produce it (–).")
    cols = ABL_COLS[:5]
    base, ref, no_anchor = ABLATIONS[0][1], ABLATIONS[-2][1], ABLATIONS[-1][1]
    out += ["## 2025 test: ablations, change in PM2.5 MAE against the full model, seed 0 (µg/m³; positive = worse)", "",
            "| model | " + " | ".join(c[0] for c in cols) + " | source |", "|---" * (len(cols) + 2) + "|"]
    for label, src in ABLATIONS[1:-2] + [("E7: no anchor, against the E7 reference", no_anchor)]:
        if src.endswith("_seeds_stats.json"):
            cells = [f"± {load(src)['metrics']['/'.join((sec,) + path)]['std']:.{nd}f}" for _, sec, path, nd in cols]
        else:
            b = ref if src == no_anchor else base
            cells = [f"{value(src, sec, path) - value(b, sec, path):+.{nd}f}" for _, sec, path, nd in cols]
        out.append(f"| {label} | " + " | ".join(cells) + f" | `{src}` |")
    return out + ["", "The seed-noise row gives the std over seeds 0/1/2 of the full model: a change smaller than about "
                      "twice that std is not distinguishable from seed noise on a single seed.", ""]


def change_label(cfg):
    """What a run changed relative to the final recipe, from its saved config.json."""
    a, c = cfg["args"], cfg["cfg"]
    parts = [n for n, on in (
        ("anneal", cfg.get("phase") == "anneal"),
        ("no anchor", not c.get("use_last_obs", False)),
        (f"{a['steps'] // 1000}k-step schedule", cfg.get("phase") == "main" and a["steps"] != 12000),
        (f"weight decay {a['weight_decay']}", a["weight_decay"] != 0.05),
        (f"patience {a['patience']}", cfg.get("phase") == "main" and a["patience"] != 3),
        ("no future met", not c["use_future_met"]),
        ("no BLH", not c.get("use_blh", True)),
        ("no spatial attention", not c["use_spatial_attention"]),
        (f"{c['spatial_bias']}-only spatial bias", c["use_spatial_attention"] and c["spatial_bias"] != "wind"),
        ("PM2.5 predicted directly", "PM_FRAC" not in c["target_channels"]),
        ("gap-filling only", not c["forecast_targets_always"])) if on]
    return ", ".join(parts) or "–"


def validation_table():
    """Best 2024 validation score (nll + 0.5·crps, forecast part, as train.py selects) per run."""
    out = ["## 2024 validation: best selection score per run (lower is better)", "",
           "| run | change | dropout | seed | best score | best step | validations | source |",
           "|---|---|---|---|---|---|---|---|"]
    for log in sorted(glob.glob(os.path.join(ROOT, "runs/*/log.jsonl"))):
        run = os.path.basename(os.path.dirname(log))
        cfg_path = os.path.join(os.path.dirname(log), "config.json")
        if not os.path.exists(cfg_path):
            continue
        cfg = json.load(open(cfg_path))
        if cfg["args"].get("small") or cfg["args"].get("synthetic"):
            continue
        c = cfg["cfg"]
        vals = [json.loads(line) for line in open(log) if '"val"' in line]
        if len(vals) < 2:              # short test launches (runs/main, runs/speedtest)
            continue
        scored = [(c["w_nll"] * v["val"]["nll_forecast"] + c["w_crps"] * v["val"]["crps_forecast"], v["step"])
                  for v in vals]
        best = min(scored)
        out.append(f"| {run} | {change_label(cfg)} | {c['dropout']} | {cfg['args']['seed']} | {best[0]:.4f} | {best[1]} | "
                   f"{len(vals)} | `runs/{run}/log.jsonl` |")
    return out + ["", "Runs with fewer than 2 validations (short test launches) are left out. `change`: what differs "
                      "from the final recipe (anchor, 12k-step cosine schedule, weight decay 0.05, patience 3, future "
                      "met, BLH, wind-aware spatial attention, PM10 × fraction, mixed masking); dropout is its own column.",
                  ""]


def build():
    lines = ["# AQFM result tables (generated)", "",
             "Generated by `python scripts/make_tables.py` from the results files named in each row.",
             "Do not edit by hand: change the results, then regenerate. 2025 test unless stated otherwise;",
             "± is the sample std (ddof=1) over seeds 0/1/2 of the final setup.", ""]
    for c in ("PM2.5", "PM10"):
        lines += table(f"2025 test: {c} MAE (µg/m³)", HZ, ROWS,
                       lambda s, k, h, c=c: get(s, k, "AQFM", (c, h, "MAE"), 2))
        lines += table(f"2025 test: {c} RMSE (µg/m³)", HZ, ROWS,
                       lambda s, k, h, c=c: get(s, k, "AQFM", (c, h, "RMSE"), 2))
    lines += table("2025 test: PM2.5 at 24 h, MAE by AQI band (µg/m³)", BANDS, ROWS,
                   lambda s, k, b: get(s, k, "AQFM", ("PM2.5", "24h_aqi", b, "MAE"), 1))
    for c in ("PM2.5", "PM10"):
        for m, nd, name in (("crps", 2, "CRPS (µg/m³)"), ("coverage90", 3, "90% interval coverage"),
                            ("coverage95", 3, "95% interval coverage")):
            lines += table(f"2025 test: {c} {name}", HZ, AQFM_ROWS,
                           lambda s, k, h, c=c, m=m, nd=nd: get(s, k, "AQFM_extras", (c, h, m), nd))
    ep = [(f"{h} {m}", h, m) for h in ("24h", "48h") for m in ("precision", "recall", "f1")]
    lines += table("2025 test: PM2.5 > 250 episodes", [e[0] for e in ep], AQFM_ROWS,
                   lambda s, k, col: next(get(s, k, "AQFM_extras", ("PM2.5", "episodes_gt250", h, m), 3)
                                          for name, h, m in ep if name == col))
    imp = [(f"{c} {w}", c, w) for c in ("PM2.5", "PM10") for w in ("aqfm", "linear")]
    lines += table("2025 test: 24 h station-outage imputation, MAE (µg/m³)", [i[0] for i in imp], AQFM_ROWS,
                   lambda s, k, col: next(get(s, k, "imputation_24h_outage", (c, w, "MAE"), 2)
                                          for name, c, w in imp if name == col),
                   note="`linear` = linear interpolation across the same outages.")
    lines += ablation_tables()
    lines += validation_table()
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    text = build()
    if args.stdout:
        print(text)
    else:
        open(os.path.join(ROOT, OUT), "w").write(text)
        print("wrote", OUT)


if __name__ == "__main__":
    main()
