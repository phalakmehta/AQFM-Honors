"""PatchTST (Nie et al., ICLR 2023) trained on our data and tested on the exact 2025 protocol.

    python deep_baselines/patchtst.py train --seed 0 --out deep_baselines/runs/patchtst_s0
    python deep_baselines/patchtst.py evaluate --ckpt deep_baselines/runs/patchtst_s0/best.pt \
        --out baselines/results/patchtst_s0_results.json
    (several --ckpt: the models' point forecasts are averaged)

Run from the repo root. `evaluate` only reads a checkpoint, so re-running it gives the same numbers.

Model: Hugging Face transformers PatchTSTForPrediction with the flatten head (pooling_type=None), a
learned positional encoding ("random") and scaling=None; normalisation is our own RevIN with a
training-split fallback (common.revin_stats). Architecture and optimiser: the paper's setting for its
large datasets (Weather/Traffic): 3 layers, d_model 128, 16 heads, FFN 256, patch 16, stride 8,
post-norm BatchNorm, lr 1e-4, batch 128. The paper's single `dropout` maps to HF's positional_dropout,
ff_dropout and path_dropout (the residual dropout); its attention and head dropout are 0.

Same treatment as AQFM, so the comparison is like for like:
  - inputs: the last 512 steps (128 h), AQFM's context; the same 2025 points scored by the same functions;
  - training windows drawn like AQFM's (window start and station uniform, horizon before 2024);
  - validation: AQFM's 512 fixed 2024 windows at the same 12 stations each (common.validation_series),
    every --val-every steps; the best checkpoint is kept; stop after --patience validations without
    improvement (AQFM: 1000 steps, patience 3);
  - tuning budget: AQFM's (dropout 0.1 / 0.2 / 0.3 on seed 0, chosen on 2024 validation, then seeds 1
    and 2 of the choice; deep_baselines/pipeline_patchtst.sh);
  - point forecast: L1 loss, i.e. the median, as AQFM's median of samples.

What PatchTST sees: one station's PM2.5 and PM10. Each series is forecast on its own with shared
weights (channel independence): no weather, no other stations, no other pollutants. So the fair AQFM
comparison is AQFM without future met (and plain XGBoost). Missing inputs are forward-filled (causal);
the L1 loss counts observed targets only, in RevIN units.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import PatchTSTConfig, PatchTSTForPrediction

import common
from common import CONTEXT, HORIZON, Series, revin_stats

NAME = "PatchTST"


def build(args, n_channels=2):
    cfg = PatchTSTConfig(
        num_input_channels=n_channels, context_length=CONTEXT, prediction_length=HORIZON,
        patch_length=args.patch_len, patch_stride=args.stride, num_hidden_layers=args.layers,
        d_model=args.d_model, num_attention_heads=args.heads, ffn_dim=args.ffn,
        norm_type="batchnorm", pre_norm=False, positional_encoding_type="random",
        pooling_type=None, scaling=None, loss="mse", share_embedding=True, share_projection=True,
        positional_dropout=args.dropout, ff_dropout=args.dropout, path_dropout=args.dropout,
        attention_dropout=0.0, head_dropout=0.0)
    return PatchTSTForPrediction(cfg)


def forward(model, b, gmean, gstd):
    """Forecast in RevIN units: (B, HORIZON, 2), with the window's mean and std."""
    mean, std = revin_stats(b["x"], b["m"], gmean, gstd)
    z = torch.nan_to_num((b["xf"] - mean) / std)          # NaN only before a series' first observation
    return model(past_values=z).prediction_outputs, mean, std


def l1(out, b, mean, std):
    """Sum of |error| over observed targets, and their count."""
    err = (out - (b["y"] - mean) / std).abs() * b["ym"]
    return err.sum(), b["ym"].sum()


@torch.no_grad()
def validate(model, st, pairs, g, device, bs):
    model.eval()
    tot = n = 0.0
    abs_ug = np.zeros(2)
    cnt = np.zeros(2)
    for k in range(0, len(pairs), bs):
        t0, s = np.array(pairs[k:k + bs]).T
        b = st.batch(s, t0, device)
        out, mean, std = forward(model, b, *g)
        e, c = l1(out, b, mean, std)
        tot, n = tot + float(e), n + float(c)
        pred = (out * std + mean).clamp(min=0)
        abs_ug += ((pred - b["y"]).abs() * b["ym"]).sum((0, 1)).cpu().numpy()
        cnt += b["ym"].sum((0, 1)).cpu().numpy()
    model.train()
    return {"l1": tot / max(n, 1.0), "mae_pm25": float(abs_ug[0] / cnt[0]), "mae_pm10": float(abs_ug[1] / cnt[1])}


def lr_at(i, warmup, total, base):
    """Linear warmup, then cosine to 1% of the peak (the schedule of aqfm/train.py)."""
    if i < warmup:
        return base * (i + 1) / warmup
    p = min(1.0, (i - warmup) / max(1, total - warmup))
    return 0.01 * base + 0.5 * 0.99 * base * (1 + np.cos(np.pi * p))


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    a = common.load_preprocessed(args.data)
    t = np.asarray(a["timestamps_h"])
    val_start, test_start = common.split_indices(t)
    print(f"loading series (train_end={val_start}, test_start={test_start}) ...", flush=True)
    st = Series(a["values"], a["observed"], val_start)
    g = (torch.tensor(st.mean, device=device), torch.tensor(st.std, device=device))
    model = build(args, st.C).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  device={device}", flush=True)
    pairs = common.validation_series(val_start, test_start, a["coords"], args.val_windows)
    print(f"validation: {args.val_windows} windows x {common.NEIGHBOURS} stations = {len(pairs)} series", flush=True)
    hi = val_start - (CONTEXT + HORIZON)                  # last window start whose horizon ends before 2024
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    code = common.code_version()
    print("code: " + ("not a git checkout" if code is None else
                      code["commit"][:7] + (f" + uncommitted changes {code['changed']}" if code["dirty"] else "")))
    meta = {"model": NAME, "args": vars(args), "hf_config": model.config.to_dict(), "val_start": val_start,
            "test_start": test_start, "global_mean": st.mean.tolist(), "global_std": st.std.tolist(), "code": code}
    json.dump(meta, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    save = lambda name, extra: torch.save({"state_dict": model.state_dict(), **meta, **extra},
                                          os.path.join(args.out, name))

    best, bad, t_start = float("inf"), 0, time.time()
    model.train()
    for i in range(args.steps):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(i, args.warmup, args.steps, args.lr)
        rng = np.random.default_rng([args.seed, i])
        b = st.batch(rng.integers(0, st.S, args.batch_size), rng.integers(0, hi + 1, args.batch_size), device)
        out, mean, std = forward(model, b, *g)
        e, c = l1(out, b, mean, std)
        loss = e / c.clamp(min=1)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i + 1) % 100 == 0 or i == 0:
            rec = {"step": i + 1, "lr": opt.param_groups[0]["lr"], "loss": loss.item(),
                   "s_per_step": (time.time() - t_start) / (i + 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if (i + 1) % 500 == 0 or i == 0:
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
                      flush=True)
        if (i + 1) % args.val_every == 0 or i + 1 == args.steps:
            val = validate(model, st, pairs, g, device, args.eval_batch)
            log.write(json.dumps({"step": i + 1, "val": val}) + "\n"); log.flush()
            print(f"  [val] step {i + 1}: " + " ".join(f"{k}={v:.4f}" for k, v in val.items()), flush=True)
            save("last.pt", {"step": i + 1, "val": val})
            if val["l1"] < best:
                best, bad = val["l1"], 0
                save("best.pt", {"step": i + 1, "val": val})
                print(f"  [val] new best {best:.4f} -> best.pt", flush=True)
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"  early stop: no improvement in {bad} validations")
                    break
    print(f"done. best val l1={best:.4f}  ({(time.time() - t_start) / 60:.1f} min)")


@torch.no_grad()
def evaluate(args):
    torch.manual_seed(0)
    device = torch.device(args.device)
    a = common.load_preprocessed(args.data)
    t = np.asarray(a["timestamps_h"])
    val_start, test_start = common.split_indices(t)
    st = Series(a["values"], a["observed"], val_start)
    models, cks = [], []
    for p in args.ckpt:
        ck = torch.load(os.path.join(common.ROOT, p), map_location=device, weights_only=False)
        assert ck["test_start"] == test_start and ck["val_start"] == val_start, "split mismatch"
        assert np.allclose(ck["global_mean"], st.mean) and np.allclose(ck["global_std"], st.std), \
            "training-split stats differ: different data?"
        m = PatchTSTForPrediction(PatchTSTConfig.from_dict(ck["hf_config"])).to(device)
        m.load_state_dict(ck["state_dict"])
        models.append(m.eval())
        cks.append(ck)
    g = (torch.tensor(st.mean, device=device), torch.tensor(st.std, device=device))

    pairs, n_origins = common.test_pairs(test_start, st.T, st.S)
    ys, preds, yms = [], [], []
    for k in range(0, len(pairs), args.eval_batch):
        t0, s = np.array(pairs[k:k + args.eval_batch]).T
        b = st.batch(s, t0, device)
        p = 0
        for m in models:                                  # mean of the models' point forecasts
            out, mean, std = forward(m, b, *g)
            p = p + (out * std + mean) / len(models)
        preds.append(p.clamp(min=0).cpu().numpy())
        ys.append(b["y"].cpu().numpy())
        yms.append(b["ym"].cpu().numpy())
    y, pred, ym = (np.concatenate(z) for z in (ys, preds, yms))

    res = common.score(NAME, y, pred, ym)
    res["n_origins"], res["n_windows"] = int(n_origins), len(pairs)
    res["selected"] = [{"step": ck["step"], "val": ck["val"]} for ck in cks]   # best 2024-validation checkpoint(s)
    res["checkpoint"] = args.ckpt[0] if len(args.ckpt) == 1 else args.ckpt
    res["code"] = common.code_version()
    res["train_code"] = cks[0]["code"] if len(cks) == 1 else [ck["code"] for ck in cks]
    res["args"] = vars(args)
    out = os.path.join(common.ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, default=str)
    for c in ("PM2.5", "PM10"):
        print(c, {hz: res[NAME][c][hz]["MAE"] for hz in common.HORIZONS})
    print("saved", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--out", required=True)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--steps", type=int, default=20000, help="cosine schedule length (maximum steps)")
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--warmup", type=int, default=1000)
    tr.add_argument("--val-every", type=int, default=1000)
    tr.add_argument("--val-windows", type=int, default=512, help="AQFM windows (x 12 stations each)")
    tr.add_argument("--patience", type=int, default=3, help="validations without improvement")
    tr.add_argument("--patch-len", type=int, default=16)
    tr.add_argument("--stride", type=int, default=8)
    tr.add_argument("--layers", type=int, default=3)
    tr.add_argument("--d-model", type=int, default=128)
    tr.add_argument("--heads", type=int, default=16)
    tr.add_argument("--ffn", type=int, default=256)
    tr.add_argument("--dropout", type=float, default=0.2)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--ckpt", required=True, nargs="+", help="paths from the repo root; several = mean forecast")
    ev.add_argument("--out", required=True, help="results file, path from the repo root")
    for p in (tr, ev):
        p.add_argument("--data", default=os.path.join(common.ROOT, "data", "preprocessed"))
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--eval-batch", type=int, default=512)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
