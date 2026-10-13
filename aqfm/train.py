"""Training: phase 1 (mixed-masking pre-training) and phase 2 (forecast anneal).

    python -m aqfm.train --phase main   --out runs/main --device cuda
    python -m aqfm.train --phase anneal --init runs/main/best.pt --out runs/anneal --device cuda
    python -m aqfm.train --synthetic --small --steps 50           # CPU smoke run

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
import time
from datetime import datetime

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


def split_indices(t):
    """(val_start, test_start) step indices for 2024-01-01 and 2025-01-01."""
    return (int(np.searchsorted(t, hours_since_epoch(2024))),
            int(np.searchsorted(t, hours_since_epoch(2025))))


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


def save(path, model, cfg, extra):
    torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(cfg), **extra}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    ck["cfg"].setdefault("use_last_obs", False)          # checkpoints from before the anchor
    cfg = AQFMConfig(**ck["cfg"])
    model = AQFM(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device), cfg, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/preprocessed")
    ap.add_argument("--out", default="runs/debug")
    ap.add_argument("--phase", choices=["main", "anneal"], default="main")
    ap.add_argument("--init", help="checkpoint to start from (required for anneal)")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="default 3e-4 main, 5e-5 anneal")
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-windows", type=int, default=512)
    ap.add_argument("--patience", type=int, default=10, help="validations without improvement")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--no-future-met", action="store_true")
    ap.add_argument("--no-last-obs", action="store_true", help="ablation: no latest-value anchor")
    ap.add_argument("--spatial-bias", choices=["wind", "distance", "none"], default="wind")
    ap.add_argument("--no-spatial", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None

    # ---- config ----------------------------------------------------------
    if args.phase == "anneal":
        assert args.init, "--init is required for the anneal phase"
        model, cfg, _ = load_checkpoint(args.init, device)
    else:
        kw = dict(grad_checkpoint=args.grad_checkpoint, use_future_met=not args.no_future_met,
                  spatial_bias=args.spatial_bias, use_spatial_attention=not args.no_spatial,
                  use_last_obs=not args.no_last_obs)
        cfg = AQFMConfig.small(**kw) if args.small else AQFMConfig(**kw)

    # ---- data ------------------------------------------------------------
    if args.synthetic:
        v, o, t, c, s, e = synthetic_raw(8, 4000, seed=args.seed)
        val_start, test_start = 2600, 3300
    else:
        a = load_preprocessed(args.data)
        v, o, t, c, s, e = (a[k] for k in ("values", "observed", "timestamps_h", "coords",
                                            "static", "era5_met"))
        val_start, test_start = split_indices(np.asarray(t))
        if e is None and cfg.use_future_met:
            raise SystemExit("era5_met.npy missing: re-run preprocess_aqfm.py or pass --no-future-met")
    print(f"building data store (train_end={val_start}, test_start={test_start}) ...")
    data = AQData(v, o, t, c, s, train_end=val_start,
                  era5_met=e if cfg.use_future_met else None)
    if data.met is not None:
        print(f"  era5 met NaN fraction after scaling: {data.met_nan_frac:.4%}")

    if args.phase == "main":
        if args.synthetic:
            print("  synthetic run: keeping default outage parameters")
        else:
            pol = [CH[n] for n in ("PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO")]
            mu, sd, n_out = fit_block_lognormal(data.m[:, pol, :val_start], min_len=cfg.patch_len)
            cfg.block_len_lognormal_mu, cfg.block_len_lognormal_sigma = mu, sd
            print(f"  outage LogNormal fit on train split (gaps >= {cfg.patch_len} steps): "
                  f"mu={mu:.3f} sigma={sd:.3f} n={n_out} median={math.exp(mu) * cfg.step_hours:.1f} h")
        model = AQFM(cfg).to(device)
        model.revin.set_global_stats(data.global_mean, data.global_std)
    print(f"parameters: {model.n_params() / 1e6:.2f}M   device={device}  amp={amp_dtype}")

    W, Tc = cfg.window_steps, cfg.context_steps
    pretrain = args.phase == "main"
    train_ds = TrainWindows(data, cfg, 0, val_start - W, args.steps * args.batch_size,
                            pretrain=pretrain, seed=args.seed)
    rng = np.random.default_rng(12345)                       # fixed validation set
    pairs = [(int(rng.integers(val_start - Tc, test_start - W + 1)), int(rng.integers(data.S)))
             for _ in range(args.val_windows)]
    val_dl = DataLoader(FixedWindows(data, cfg, pairs), batch_size=args.batch_size,
                        num_workers=args.workers)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, drop_last=True,
                          pin_memory=device.type == "cuda")

    lr = args.lr or (3e-4 if args.phase == "main" else 5e-5)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))

    meta = {"phase": args.phase, "args": vars(args), "val_start": val_start,
            "test_start": test_start}
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
            loss, parts, _, _ = step(model, batch, cfg, impute=pretrain)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {i}: {parts}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (i + 1) % 50 == 0 or i == 0:
            rec = {"step": i + 1, "lr": opt.param_groups[0]["lr"], "gnorm": float(gnorm),
                   "s_per_step": (time.time() - t_start) / (i + 1), **parts}
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            log.write(json.dumps(rec) + "\n"); log.flush()

        if (i + 1) % args.val_every == 0 or i + 1 == args.steps:
            val = validate(model, val_dl, cfg, device, amp_dtype)
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
