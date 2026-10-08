"""Two-stage training: masked pretraining, then horizon fine-tuning."""

import math
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import AQFMConfig
from .data import (AirQualityWindows, build_queries, gather_targets,
                   normalize_targets, synthetic_dataset)
from .losses import mixture_sample, total_loss
from .model import AQFM


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def step(model, batch, cfg, device):
    B, S = batch["x_ctx"].shape[:2]
    q_channel, q_station, q_lead, q_time = build_queries(
        cfg, B, S, batch["t_full"], device)

    enc_batch = {"x": batch["x_ctx"], "m": batch["m_ctx"], "t": batch["t"],
                 "coords": batch["coords"], "static": batch["static"]}
    out = model(enc_batch, q_channel, q_station, q_lead, q_time)

    y_raw, mask = gather_targets(cfg, batch, q_channel, q_station)
    y = normalize_targets(y_raw, out["mean"], out["std"], q_channel, q_station, cfg)
    loss, parts = total_loss(out, y, mask, cfg)
    return loss, parts, out, (y, mask, q_channel, q_station)


def cosine_lr(step_i, warmup, total, base_lr, min_lr=1e-5):
    if step_i < warmup:
        return base_lr * step_i / max(1, warmup)
    p = (step_i - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


def train(cfg: AQFMConfig, dataset, model=None, steps=1000, batch_size=4,
          lr=3e-4, warmup=100, device="cpu", log_every=25, grad_clip=1.0):
    model = (model or AQFM(cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=0)
    it, i, t0 = iter(loader), 0, time.time()
    model.train()
    while i < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            continue
        batch = to_device(batch, device)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(i, warmup, steps, lr)

        loss, parts, _, _ = step(model, batch, cfg, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        i += 1
        if i % log_every == 0 or i == 1:
            print(f"step {i:5d}  loss {parts['total']:.4f}  "
                  f"nll {parts['nll']:.4f}  crps {parts['crps']:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{(time.time()-t0)/i:.2f}s/step")
    return model


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataset, cfg, device="cpu", batch_size=4, n_batches=20,
             episode_threshold=250.0, n_samples=64):
    """Report skill where it matters, not where it is easy.

    Aggregate MAE over a full year is dominated by clean summer months. The
    number that decides whether this model is useful is episode recall at
    24-72h lead, so it is reported separately here.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    thr = math.log1p(episode_threshold)
    acc = {"mae": [], "crps": [], "tp": 0, "fn": 0, "fp": 0, "n": 0}

    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        batch = to_device(batch, device)
        _, _, out, (y, mask, q_channel, q_station) = step(model, batch, cfg, device)

        samples = mixture_sample(out["logits"], out["mu"], out["log_sigma"],
                                 n_samples)                       # (n,B,Q,P)
        med = samples.median(0).values
        med_l1 = model.denorm(med, out["mean"], out["std"], q_channel, q_station)
        y_l1 = model.denorm(y, out["mean"], out["std"], q_channel, q_station)

        w = mask.bool()
        if w.sum() == 0:
            continue
        acc["mae"].append(float(((med_l1 - y_l1).abs() * mask).sum() / mask.sum()))

        # episode detection on the PM2.5-carrying target slot (index 0 = PM10
        # base; slot 0/1 reconstruct PM2.5 -- here scored on the base channel)
        pred_hi = (med_l1 > thr) & w
        true_hi = (y_l1 > thr) & w
        acc["tp"] += int((pred_hi & true_hi).sum())
        acc["fn"] += int((~pred_hi & true_hi).sum())
        acc["fp"] += int((pred_hi & ~true_hi).sum())
        acc["n"] += int(w.sum())

    rec = acc["tp"] / max(1, acc["tp"] + acc["fn"])
    prec = acc["tp"] / max(1, acc["tp"] + acc["fp"])
    return {"mae_log1p": float(np.mean(acc["mae"])) if acc["mae"] else float("nan"),
            "episode_recall": rec, "episode_precision": prec,
            "n_scored": acc["n"]}


# ---------------------------------------------------------------------------
def baselines(dataset, cfg, n=200):
    """Persistence and seasonal-naive. Gradient-boosted trees on lagged
    features belong here too -- they are brutally strong on this task, and a
    foundation model that does not clearly beat them at 24h+ has a setup bug,
    not a weak baseline."""
    res = {"persistence": [], "seasonal_naive": []}
    day = 96                                        # 24h at 15-min cadence
    tgt = list(cfg.target_index)                    # score the same channels
    for i in range(min(n, len(dataset))):
        b = dataset[i]
        x = b["x_full"].numpy()[:, tgt]
        m = b["m_tgt"].numpy()[:, tgt]
        ctx = cfg.context_steps
        for name, pred in (
                ("persistence", np.repeat(x[..., ctx - 1:ctx], x.shape[-1] - ctx, -1)),
                ("seasonal_naive", x[..., ctx - day:ctx - day + (x.shape[-1] - ctx)])):
            mk = m[..., ctx:]
            if mk.sum() == 0:
                continue
            res[name].append(float((np.abs(pred - x[..., ctx:]) * mk).sum() / mk.sum()))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in res.items()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true",
                    help="reduced config; use for smoke tests on CPU")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--stations", type=int, default=12)
    ap.add_argument("--neighbours", type=int, default=6)
    args = ap.parse_args()

    cfg = AQFMConfig.small(grad_checkpoint=True) if args.small else AQFMConfig()
    v, o, t, coords, static = synthetic_dataset(cfg, n_stations=args.stations,
                                                n_steps=6000)
    ds = AirQualityWindows(v, o, t, coords, static, cfg,
                           n_neighbours=args.neighbours, pretrain=True)
    model = AQFM(cfg)
    print(f"parameters: {model.n_params()/1e6:.1f}M  "
          f"tokens/sample: {args.neighbours*cfg.n_channels*cfg.n_context_patches}")
    model = train(cfg, ds, model=model, steps=args.steps,
                  batch_size=args.batch_size, log_every=10)
    print("eval    :", evaluate(model, ds, cfg, n_batches=2, batch_size=2))
    print("baseline:", baselines(ds, cfg, n=20))
