"""Training: phase 1 (mixed-masking pre-training) and phase 2 (forecast anneal).

    python -m aqfm.train --phase main   --out runs/main --device cuda
    python -m aqfm.train --phase anneal --init runs/main/best.pt --out runs/anneal --device cuda
    python -m aqfm.train --synthetic --small --steps 50           # CPU smoke run

Multi-city / transfer (v2, see docs):
    # 1. Delhi model with the ~1M city config
    python -m aqfm.train --phase main --preset city --data data/delhi --out runs/delhi
    # 2. zero-shot on Mumbai: evaluate only
    python -m aqfm.evaluate --ckpt runs/delhi/best.pt --data data/mumbai --out results/mumbai_zeroshot.json
    # 3. fine-tune on Mumbai (optionally only the last N months, frozen encoder)
    python -m aqfm.train --phase finetune --init runs/delhi/best.pt --data data/mumbai \
        --out runs/mumbai_ft --freeze encoder --train-months 12
    # 4. joint two-city model, city-balanced sampling, 3 Mumbai stations held out
    python -m aqfm.train --phase main --preset city --data data/delhi data/mumbai \
        --holdout mumbai:3 --out runs/joint

Splits (by timestamp, fixed): train 2018-2023, validation 2024, test 2025.
No training window's horizon reaches 2024; validation targets lie in 2024 only;
2025 is never touched here. Every statistic (RevIN fallback, met scaling,
outage distribution) is computed on the training split.
"""

import argparse
import dataclasses
import json
import math
import os
import subprocess
import time
import zlib
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import AQFMConfig, CH
from .data import AQData, FixedWindows, TrainWindows, build_queries, synthetic_raw
from .losses import total_loss
from .masking import fit_block_lognormal
from .model import AQFM

EPOCH = datetime(2018, 1, 1)


def hours_since_epoch(y, m=1, d=1):
    return (datetime(y, m, d) - EPOCH).total_seconds() / 3600.0


def load_preprocessed(path):
    a = {k: np.load(os.path.join(path, f"{k}.npy"), mmap_mode="r")
         for k in ("values", "observed", "timestamps_h", "coords", "static")}
    met = os.path.join(path, "era5_met.npy")
    a["era5_met"] = np.load(met, mmap_mode="r") if os.path.exists(met) else None
    return a


def load_multi(paths):
    """Several per-city preprocessed folders -> one station-stacked dict plus
    `group` (city id per station) and `group_names`. Cities must share the
    time grid (same preprocess_aqfm.py run settings); a city with a shorter
    record is placed on the union grid with its missing steps unobserved."""
    if isinstance(paths, str):
        paths = [paths]
    parts = [load_preprocessed(p) for p in paths]
    names = [os.path.basename(os.path.normpath(p)) for p in paths]
    if len(parts) == 1:
        a = parts[0]
        a["group"], a["group_names"] = np.zeros(len(a["coords"]), np.int64), names
        return a
    step = float(np.median(np.diff(np.asarray(parts[0]["timestamps_h"][:1000]))))
    lo = min(float(p["timestamps_h"][0]) for p in parts)
    hi = max(float(p["timestamps_h"][-1]) for p in parts)
    t = lo + step * np.arange(int(round((hi - lo) / step)) + 1)
    out = {k: [] for k in ("values", "observed", "coords", "static", "era5_met", "group")}
    has_met = all(p["era5_met"] is not None for p in parts)
    for g, p in enumerate(parts):
        tp = np.asarray(p["timestamps_h"], np.float64)
        idx = np.rint((tp - lo) / step).astype(np.int64)
        assert np.allclose(t[idx], tp, atol=1e-3), f"{paths[g]}: time grid not aligned"
        S = len(p["coords"])
        for k in ("values", "observed") + (("era5_met",) if has_met else ()):
            src = p[k]
            buf = np.zeros((S, src.shape[1], len(t)), np.float32)
            if k == "era5_met":
                buf[:] = np.nan
            buf[:, :, idx] = src
            out[k].append(buf)
        out["coords"].append(np.asarray(p["coords"]))
        out["static"].append(np.asarray(p["static"]))
        out["group"].append(np.full(S, g, np.int64))
    res = {k: np.concatenate(v, 0) for k, v in out.items() if v}
    res["timestamps_h"], res["group_names"] = t, names
    if not has_met:
        res["era5_met"] = None
    return res


def parse_holdout(spec, group, group_names):
    """'mumbai:3,delhi:2' -> (S,) bool `allowed` with that many stations per
    city held out. The choice depends only on the city's name and stations,
    never on the training seed, so every seed hides the same stations (and
    'delhi:4' hides the same Delhi stations whichever other cities are
    loaded). Also accepts explicit station indices: '12,30,41'.
    Returns (allowed, held_out_indices)."""
    allowed = np.ones(len(group), bool)
    if not spec:
        return allowed, []
    for item in spec.split(","):
        if ":" in item:
            name, n = item.split(":")
            g = group_names.index(name)
            rng = np.random.default_rng(zlib.crc32(name.encode()))
            pick = rng.permutation(np.flatnonzero(group == g))[:int(n)]
            allowed[pick] = False
        else:
            allowed[int(item)] = False
    return allowed, np.flatnonzero(~allowed).tolist()


def fit_outages(data, val_start, min_len, stations=None):
    """The outage fit: LogNormal of station-wide pollutant outage lengths on
    the training split (sets the block-mask lengths of pre-training). With a
    spatial hold-out pass `stations` (bool, the seen ones) so the hidden
    stations' outages do not shape the masks."""
    pol = [CH[n] for n in ("PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO")]
    fit = slice(None) if stations is None else np.flatnonzero(stations)
    return fit_block_lognormal(data.m[fit][:, pol, :val_start], min_len=min_len)


def train_months_start(t, val_start, n):
    """Step index where the last n calendar months before step val_start
    begin, so [index, val_start) is exactly n months of data (val_start is
    Jan 1 of the validation year). Clipped to the start of the record."""
    d = EPOCH + timedelta(hours=float(t[val_start]))
    k = d.year * 12 + (d.month - 1) - n
    return int(np.searchsorted(t, hours_since_epoch(k // 12, k % 12 + 1)))


def split_indices(t, val_year=2024, test_year=2025, val_date=None, test_date=None):
    """(val_start, test_start) step indices for Jan 1 of val_year / test_year,
    or for val_date / test_date ("YYYY-MM-DD") when given."""
    v = hours_since_epoch(*map(int, val_date.split("-"))) if val_date else hours_since_epoch(val_year)
    s = hours_since_epoch(*map(int, test_date.split("-"))) if test_date else hours_since_epoch(test_year)
    return int(np.searchsorted(t, v)), int(np.searchsorted(t, s))


def step(model, batch, cfg, forecast=True, impute=True):
    q, y, ym = build_queries(cfg, batch, forecast=forecast, impute=impute)
    out = model(batch, q)
    n_f = batch["x_ctx"].shape[1] * cfg.n_target_channels * cfg.n_horizon_patches if forecast else 0
    loss, parts = total_loss(model, out, q, y, ym, batch, cfg, n_f)
    return loss, parts, out, q


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def cosine_lr(i, warmup, total, base, floor):
    if i < warmup:
        return base * (i + 1) / warmup
    p = min(1.0, (i - warmup) / max(1, total - warmup))
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def validate(model, loader, cfg, device, amp_dtype):
    model.eval()
    tot, n = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, parts, _, _ = step(model, batch, cfg, impute=False)
        for k, v in parts.items():
            tot[k] = tot.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in tot.items()}


def code_version():
    """Git commit of this checkout, and whether the code (aqfm/, tests/, baselines/*.py) has
    uncommitted changes, so every run and result records exactly which code produced it.
    None outside a git checkout."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True,
                                    check=True, timeout=10).stdout
    try:
        # porcelain lines are "XY path"; X may be a space, so the output must not be stripped
        changed = git("status", "--porcelain", "--", "aqfm", "tests", "baselines/*.py").splitlines()
        return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(changed),
                "changed": [line[3:] for line in changed]}
    except (OSError, subprocess.SubprocessError):
        return None


def set_trainable(model, freeze):
    """Fine-tuning freeze policies. Returns the number of trainable params.
    none     everything trains.
    encoder  freeze the axial blocks and patch tokenizer value path; train the
             station/position MLPs, spatial bias, sink, pooler, decoder, head.
             The cheap default when the new city has < 1 year of data.
    backbone freeze everything except station/position MLPs, spatial bias,
             sink and the output head (smallest adaptation, ~5-10% of params).
    Always trainable when not 'none': anything named station / spatial_bias /
    sink / head / q_last / anchor_gate."""
    always = ("station", "spatial_bias", "sink", "head", "q_last", "anchor_gate")
    for n, p in model.named_parameters():
        if freeze == "none":
            p.requires_grad = True
        elif freeze == "encoder":
            p.requires_grad = (any(a in n for a in always)
                               or not (n.startswith("blocks.") or n.startswith("tokenizer.")))
        elif freeze == "backbone":
            p.requires_grad = any(a in n for a in always)
        else:
            raise ValueError(freeze)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save(path, model, cfg, extra):
    torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(cfg), **extra}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    ck["cfg"].setdefault("use_last_obs", False)          # checkpoints from before the anchor
    for k in ("dist_kernels_km", "nbr_dropout_frac", "target_channels", "n_blocks_per_sample",
              "channel_dropout_frac"):                   # json/pickle round trip -> tuples
        if isinstance(ck["cfg"].get(k), list):
            ck["cfg"][k] = tuple(ck["cfg"][k])
    cfg = AQFMConfig(**ck["cfg"])
    model = AQFM(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device), cfg, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=["data/preprocessed"], nargs="+",
                    help="one or more per-city preprocessed folders (stacked by station)")
    ap.add_argument("--out", default="runs/debug")
    ap.add_argument("--phase", choices=["main", "anneal", "finetune"], default="main",
                    help="finetune = main-phase objective (mixed masking) from --init")
    ap.add_argument("--init", help="checkpoint to start from (required for anneal/finetune)")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--preset", choices=["v1", "city"], default="v1",
                    help="city = ~1M-param multi-city config with all spatial-scale fixes")
    ap.add_argument("--d-model", type=int, default=None, help="width (default: the preset's; new models only)")
    ap.add_argument("--n-heads", type=int, default=None, help="attention heads (default: the preset's)")
    ap.add_argument("--d-ff", type=int, default=None, help="feed-forward width (default: the preset's)")
    ap.add_argument("--freeze", choices=["none", "encoder", "backbone"], default="none",
                    help="finetune/anneal only")
    ap.add_argument("--train-months", type=int, default=None,
                    help="use only the last N calendar months before validation for training "
                         "(for the target city in multi-city runs: the last listed --data)")
    ap.add_argument("--holdout", default="",
                    help="spatial hold-out, e.g. 'mumbai:3' or station indices '4,17'")
    ap.add_argument("--val-year", type=int, default=2024)
    ap.add_argument("--test-year", type=int, default=2025)
    ap.add_argument("--val-start", default=None, help="YYYY-MM-DD: validation starts here, not on Jan 1 of --val-year")
    ap.add_argument("--test-start", default=None,
                    help="YYYY-MM-DD: the test (and the end of validation) starts here, not on Jan 1 of --test-year")
    ap.add_argument("--no-balance", action="store_true", help="sample anchors uniformly over stations")
    ap.add_argument("--refit-stats", action="store_true",
                    help="finetune/anneal: refit met/RevIN stats on the new data (default: keep the checkpoint's)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="default 3e-4 main, 5e-5 anneal")
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=None, help="main phase only; default 0.1 (config)")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-windows", type=int, default=512)
    ap.add_argument("--patience", type=int, default=10, help="validations without improvement")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model: same maths and random draws, ~25%% faster")
    ap.add_argument("--no-future-met", action="store_true")
    ap.add_argument("--no-last-obs", action="store_true", help="ablation: no latest-value anchor")
    ap.add_argument("--spatial-bias", choices=["wind", "distance", "none"], default="wind")
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--no-blh", action="store_true", help="ablation: no boundary-layer height anywhere")
    ap.add_argument("--pm25-direct", action="store_true",
                    help="ablation: predict PM2.5 directly instead of PM10 x fraction (NOx pair unchanged)")
    ap.add_argument("--no2-direct", action="store_true",
                    help="predict NO2 directly instead of NOx x (1 - NO share) (NOx still predicted; NO not)")
    ap.add_argument("--random-masking", action="store_true",
                    help="ablation: block/channel masking only, no forecast targets in training")
    ap.add_argument("--v1-spatial", action="store_true",
                    help="ablation: switch off the v2 spatial changes (relative coordinates, 100 km radius, "
                         "distance kernels, travel time, look-back, sink, density, neighbour dropout)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None

    # ---- config ----------------------------------------------------------
    init_ck = None
    if args.phase in ("anneal", "finetune"):
        assert args.init, f"--init is required for the {args.phase} phase"
        assert args.dropout is None, "--dropout is main phase only; anneal keeps the checkpoint's"
        model, cfg, init_ck = load_checkpoint(args.init, device)
    else:
        kw = dict(grad_checkpoint=args.grad_checkpoint, use_future_met=not args.no_future_met,
                  spatial_bias=args.spatial_bias, use_spatial_attention=not args.no_spatial,
                  use_last_obs=not args.no_last_obs, use_blh=not args.no_blh)
        if args.dropout is not None:
            kw["dropout"] = args.dropout
        for k in ("d_model", "n_heads", "d_ff"):
            if getattr(args, k) is not None:
                kw[k] = getattr(args, k)
        if args.pm25_direct:
            kw["target_channels"] = tuple("PM2.5" if c == "PM_FRAC" else c for c in AQFMConfig.target_channels)
        if args.no2_direct:
            kw["target_channels"] = tuple("NO2" if c == "NOX_FRAC" else c
                                          for c in kw.get("target_channels", AQFMConfig.target_channels))
        if args.random_masking:
            kw.update(p_tail_mask=0.0, forecast_targets_always=False)
        if args.small:
            if args.preset == "city":
                kw.update(coord_mode="relative", neighbour_radius_km=100.0,
                          dist_kernels_km=(3.0, 10.0, 40.0), advection_feature=True,
                          spatial_lags=1, spatial_sink=True, density_features=True,
                          nbr_dropout_p=0.3)
            cfg = AQFMConfig.small(**kw)
        else:
            cfg = AQFMConfig.city(**kw) if args.preset == "city" else AQFMConfig(**kw)
        if args.v1_spatial:        # ablation: same sizes, every v2 spatial change off (A1-A7 as in v1)
            cfg = dataclasses.replace(cfg, coord_mode="delhi", neighbour_radius_km=None, dist_kernels_km=(),
                                      advection_feature=False, spatial_lags=0, spatial_sink=False,
                                      density_features=False, nbr_dropout_p=0.0)

    # ---- data ------------------------------------------------------------
    if args.synthetic:
        # two toy "cities" when several --data paths are given, else one
        n_city = len(args.data)
        raws = [synthetic_raw(8, 4000, seed=args.seed + g) for g in range(n_city)]
        for g, r in enumerate(raws):
            r[3][:] += np.array([-9.5 * g, -4.3 * g], np.float32)       # move city g south-west
        v, o, c, s = (np.concatenate([r[k] for r in raws]) for k in (0, 1, 3, 4))
        e = np.concatenate([r[5] for r in raws])
        t = raws[0][2]
        group = np.concatenate([np.full(8, g) for g in range(n_city)])
        group_names = [f"city{g}" for g in range(n_city)]
        val_start, test_start = 2600, 3300
    else:
        a = load_multi(args.data)
        v, o, t, c, s, e = (a[k] for k in ("values", "observed", "timestamps_h", "coords",
                                            "static", "era5_met"))
        group, group_names = a["group"], a["group_names"]
        val_start, test_start = split_indices(np.asarray(t), args.val_year, args.test_year,
                                              args.val_start, args.test_start)
        if e is None and cfg.use_future_met:
            raise SystemExit("era5_met.npy missing: re-run preprocess_aqfm.py or pass --no-future-met")
    keep_stats = init_ck is not None and not args.refit_stats
    met_stats = init_ck.get("met_stats") if keep_stats else None
    if keep_stats and met_stats is None and cfg.use_future_met:
        print("  WARNING: checkpoint has no met_stats (pre-v2); refitting met scaling on new data")
    allowed, held_out = parse_holdout(args.holdout, group, group_names)
    if (init_ck is not None and not args.holdout and init_ck.get("held_out_stations")
            and init_ck.get("cities") == list(group_names)):
        # same data as the checkpoint: its held-out stations stay unseen (anneal etc.)
        held_out = list(init_ck["held_out_stations"])
        allowed = np.ones(len(group), bool)
        allowed[held_out] = False
    if held_out:
        print(f"  spatial hold-out (never seen in training): stations {held_out}")
    print(f"building data store (train_end={val_start}, test_start={test_start}, "
          f"cities={group_names}) ...")
    # with a hold-out, every statistic fitted from data uses the seen stations only
    data = AQData(v, o, t, c, s, train_end=val_start,
                  era5_met=e if cfg.use_future_met else None, met_stats=met_stats,
                  group=group, group_names=group_names,
                  stats_stations=allowed if held_out else None)
    if data.met is not None:
        print(f"  era5 met NaN fraction after scaling: {data.met_nan_frac:.4%}")
    if init_ck is not None and args.refit_stats:
        model.revin.set_global_stats(data.global_mean, data.global_std)
    if init_ck is not None:
        n_tr = set_trainable(model, args.freeze)
        print(f"  freeze={args.freeze}: {n_tr / 1e6:.3f}M trainable parameters")

    if args.phase == "main":
        if args.synthetic:
            print("  synthetic run: keeping default outage parameters")
        else:
            mu, sd, n_out = fit_outages(data, val_start, cfg.patch_len, allowed if held_out else None)
            cfg.block_len_lognormal_mu, cfg.block_len_lognormal_sigma = mu, sd
            print(f"  outage LogNormal fit on train split (gaps >= {cfg.patch_len} steps): "
                  f"mu={mu:.3f} sigma={sd:.3f} n={n_out} median={math.exp(mu) * cfg.step_hours:.1f} h")
        model = AQFM(cfg).to(device)
        model.revin.set_global_stats(data.global_mean, data.global_std)
    print(f"parameters: {model.n_params() / 1e6:.2f}M   device={device}  amp={amp_dtype}")
    run = model                     # what steps call; checkpoints always save `model`
    if args.compile:
        from torch._inductor import config as inductor_config
        inductor_config.fallback_random = True           # dropout uses the eager RNG stream
        run = torch.compile(model)

    W, Tc = cfg.window_steps, cfg.context_steps
    pretrain = args.phase in ("main", "finetune")
    group_lo = None
    if args.train_months:
        # limit the target city (last --data) to exactly the last N calendar
        # months before validation, windows included; other cities keep their full record
        lo_t = train_months_start(data.t, val_start, args.train_months)
        assert lo_t <= val_start - W, "--train-months is shorter than one training window"
        target = int(data.group.max())
        group_lo = {target: lo_t}
        print(f"  {data.group_names[target]}: training data limited to steps [{lo_t}, {val_start}), "
              f"the last {args.train_months} months before validation")
    train_ds = TrainWindows(data, cfg, 0, val_start - W, args.steps * args.batch_size,
                            pretrain=pretrain, seed=args.seed, allowed=allowed,
                            balance_groups=not args.no_balance, group_lo=group_lo)
    rng = np.random.default_rng(12345)                       # fixed validation set
    pairs = [(int(rng.integers(val_start - Tc, test_start - W + 1)), int(rng.integers(data.S)))
             for _ in range(args.val_windows)]
    if held_out:                                             # select models on seen stations only
        rng = np.random.default_rng(12345)
        seen = np.flatnonzero(allowed)
        pairs = [(int(rng.integers(val_start - Tc, test_start - W + 1)), int(rng.choice(seen)))
                 for _ in range(args.val_windows)]
    val_dl = DataLoader(FixedWindows(data, cfg, pairs, allowed if held_out else None),
                        batch_size=args.batch_size,
                        num_workers=args.workers)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, drop_last=True,
                          pin_memory=device.type == "cuda")

    lr = args.lr or {"main": 3e-4, "finetune": 1e-4, "anneal": 5e-5}[args.phase]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = [p for n, p in named if p.dim() >= 2 and "emb" not in n]
    no_decay = [p for n, p in named if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))

    code = code_version()
    print("code: " + ("not a git checkout" if code is None else
                      code["commit"][:7] + (f" + uncommitted changes {code['changed']}" if code["dirty"] else "")))
    meta = {"phase": args.phase, "args": vars(args), "val_start": val_start,
            "test_start": test_start, "code": code,
            "met_stats": None if data.met_stats is None else [a.tolist() for a in data.met_stats],
            "cities": data.group_names, "held_out_stations": held_out,
            "init": os.path.abspath(args.init) if args.init else None}
    json.dump({"cfg": dataclasses.asdict(cfg), **meta}, open(os.path.join(args.out, "config.json"), "w"),
              indent=2)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    best, bad, t_start = float("inf"), 0, time.time()
    model.train()
    for i, batch in enumerate(train_dl):
        if i >= args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = cosine_lr(i, args.warmup, args.steps, lr, lr * 0.01)
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, parts, _, _ = step(run, batch, cfg, impute=pretrain)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {i}: {parts}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if (i + 1) % 50 == 0 or i == 0:
            rec = {"step": i + 1, "lr": opt.param_groups[0]["lr"], "gnorm": float(gnorm),
                   "s_per_step": (time.time() - t_start) / (i + 1), **parts}
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            log.write(json.dumps(rec) + "\n"); log.flush()

        if (i + 1) % args.val_every == 0 or i + 1 == args.steps:
            val = validate(run, val_dl, cfg, device, amp_dtype)
            # select on the deterministic part; the physical term is sampled
            score = cfg.w_nll * val["nll_forecast"] + cfg.w_crps * val["crps_forecast"]
            log.write(json.dumps({"step": i + 1, "val": val}) + "\n"); log.flush()
            print(f"  [val] step {i + 1}: " + " ".join(f"{k}={v:.4f}" for k, v in val.items()))
            save(os.path.join(args.out, "last.pt"), model, cfg, {**meta, "step": i + 1, "val": val})
            if score < best:
                best, bad = score, 0
                save(os.path.join(args.out, "best.pt"), model, cfg, {**meta, "step": i + 1, "val": val})
                print(f"  [val] new best {best:.4f} -> best.pt")
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"  early stop: no improvement in {bad} validations")
                    break
    print(f"done. best val total={best:.4f}  ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
