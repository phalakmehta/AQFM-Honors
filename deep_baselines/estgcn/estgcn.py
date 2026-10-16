"""E-STGCN (Panja, Chakraborty, Biswas and Deb, JRSS-A 2026, doi 10.1093/jrsssa/qnag010; arXiv 2411.12258),
re-implemented in PyTorch, trained on our data and tested on the exact 2025 protocol.

    python deep_baselines/estgcn/estgcn.py train --target PM2.5 --seed 0 --out deep_baselines/estgcn/runs/X
    python deep_baselines/estgcn/estgcn.py evaluate --ckpt PM25.pt PM10.pt --out baselines/results/estgcn_s0_results.json
    (one checkpoint per pollutant; several seeds: each pollutant's forecasts are averaged)

Run from the repo root. Our own implementation, written from the paper and the authors' public notebook
(github.com/mad-stat/E_STGCN; the repository has no license, so none of its code is copied).

The model as published: a graph convolution over the stations (each station's features times W,
concatenated with the mean of its neighbours' features times the same W; out_feat 10, no activation),
an LSTM (64 units) over the input window per station, and a dense layer to the whole horizon. Graph:
binary adjacency from Haversine distances, A_ij = 1 if exp(-(d_ij / 10000)^2 / 0.75) >= 0.5, no self
loops. One model per pollutant, per-station z-scoring with training mean and std. Loss: 0.99 x MSE +
0.01 x the generalised-Pareto negative log-likelihood of the forecast where it exceeds the NAAQS
threshold (60 ug/m3 PM2.5, 100 PM10), maximum over stations (their Eq. 6 and notebook formula, kept as
written: evaluated at the forecast itself, not at the exceedance). GPD shape and scale per station by
maximum likelihood on the training exceedances. RMSprop, batch 64.

Changes, each needed for our protocol or a fix (all documented in deep_baselines/estgcn/README.md):
  - data: 15-min values; input 512 steps (128 h, AQFM's and PatchTST's context), output 288 steps
    (72 h). Theirs: daily means, 24-day input, 30-90-day output.
  - the loss uses every sample of the batch. Their notebook's loss reads y_true[0] and y_pred[0], so
    only the first sample of each batch trains the model.
  - missing data: inputs forward-filled (causal); the MSE counts observed targets only.
  - LSTM with tanh, as the paper writes it (the notebook's Keras LSTM uses ReLU, inherited from the
    Keras example it is based on); cuDNN needs tanh, and ReLU is unstable over 512 steps.
  - distances in metres (edges within ~7.2 km; with km every station would be linked to all others and
    epsilon would not control sparsity, which the paper says it does). Station coordinates are ours.
  - selection exactly as for PatchTST: AQFM's 512 fixed 2024 windows at the same 12 stations each,
    validation MSE every 1000 steps, the whole schedule is trained and the best checkpoint kept; the
    learning rate (theirs 2e-4) gets AQFM's tuning budget: 3 values on seed 0, then seeds 1 and 2.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import genpareto
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
from common import CONTEXT, HORIZON, Series  # noqa: E402

NAME = "E-STGCN"
TARGETS = ("PM2.5", "PM10")                    # channel order of common.Series
THRESHOLD = {"PM2.5": 60.0, "PM10": 100.0}     # NAAQS 24-h standards (ug/m3), as in the paper
BETA = (0.99, 0.01)                            # MSE and GPD weights of their notebook
SIGMA2, EPS = 0.75, 0.5                        # their adjacency kernel width and threshold


def adjacency(coords):
    """Binary adjacency of the notebook: distances (metres) / 10000, Gaussian kernel, threshold, no self loops."""
    c = np.asarray(coords, dtype=np.float32)
    d = common.haversine_km(c, c) * 1000.0 / 10000.0
    return ((np.exp(-d ** 2 / SIGMA2) >= EPS) * (1 - np.eye(len(c)))).astype(np.float32)


def fit_gpd(x, u):
    """Maximum-likelihood GPD (shape xi, scale sigma) of the exceedances x - u, x > u."""
    xi, _, sigma = genpareto.fit(x[x > u] - u, floc=0)
    return float(xi), float(sigma)


class GraphConv(nn.Module):
    """x W concatenated with (mean of the neighbours' x) W; isolated stations get zeros, as Keras'
    unsorted_segment_mean does for an empty segment."""

    def __init__(self, in_feat, out_feat, adj):
        super().__init__()
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(in_feat, out_feat)))
        adj = torch.as_tensor(adj)
        self.register_buffer("mean_adj", adj / adj.sum(1, keepdim=True).clamp(min=1))

    def forward(self, x):                                      # (B, T, N, F)
        agg = torch.einsum("ij,btjf->btif", self.mean_adj, x)
        return torch.cat([x @ self.weight, agg @ self.weight], -1)


class ESTGCN(nn.Module):
    def __init__(self, adj, out_feat=10, lstm_units=64, horizon=HORIZON):
        super().__init__()
        self.graph_conv = GraphConv(1, out_feat, adj)
        self.lstm = nn.LSTM(2 * out_feat, lstm_units, batch_first=True)
        self.dense = nn.Linear(lstm_units, horizon)
        # Keras initialisation: Glorot kernels, orthogonal recurrent kernel, zero bias with forget bias 1
        H = lstm_units
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        for k in range(4):
            nn.init.orthogonal_(self.lstm.weight_hh_l0.data[k * H:(k + 1) * H])
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        self.lstm.bias_ih_l0.data[H:2 * H] = 1.0               # PyTorch gate order: input, forget, cell, output
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x):                                      # x: (B, T, N) z-scored -> (B, HORIZON, N)
        B, T, N = x.shape
        g = self.graph_conv(x.unsqueeze(-1))                   # (B, T, N, 2 out_feat)
        g = g.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        _, (h, _) = self.lstm(g)
        return self.dense(h[-1]).view(B, N, -1).transpose(1, 2)


class Pollutant:
    """One pollutant of every station: z-scored filled inputs, raw targets and masks, and the per-station
    training mean/std and GPD fit. All statistics use the training split only."""

    def __init__(self, st: Series, target, val_start):
        c = TARGETS.index(target)
        self.target, self.u = target, THRESHOLD[target]
        self.v, self.o, self.ff = st.v[:, c], st.o[:, c], st.ff[:, c]          # (S, T)
        self.S = self.v.shape[0]
        tr = [self.v[s, :val_start][self.o[s, :val_start]] for s in range(self.S)]
        self.mean = np.array([x.mean() for x in tr], np.float32)
        self.std = np.array([x.std() for x in tr], np.float32)
        gpd = [fit_gpd(x, self.u) for x in tr]
        self.xi = np.array([g[0] for g in gpd], np.float32)
        self.sigma = np.array([g[1] for g in gpd], np.float32)

    def windows(self, t0, device="cpu"):
        """All stations for each window start: x (B, CONTEXT, S) z-scored, y and ym (B, HORIZON, S) raw."""
        t0 = np.asarray(t0)
        take = lambda a, start, n: np.ascontiguousarray(a[:, start[:, None] + np.arange(n)].transpose(1, 2, 0))
        x = np.nan_to_num((take(self.ff, t0, CONTEXT) - self.mean) / self.std)   # NaN before a first observation
        f = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(device)
        return {"x": f(x), "y": f(take(self.v, t0 + CONTEXT, HORIZON)), "ym": f(take(self.o, t0 + CONTEXT, HORIZON))}


def loss_terms(pred, b, k):
    """pred: (B, HORIZON, S) z-units. Returns (sum of squared errors over observed targets, their count,
    GPD penalty summed over samples and steps). k: tensors mean, std, xi, sigma (S,) and threshold u."""
    yz = (b["y"] - k["mean"]) / k["std"]
    se = ((pred - yz) ** 2 * b["ym"]).sum()
    raw = k["mean"] + k["std"] * pred                          # forecast in ug/m3
    inv = torch.where(k["xi"] != 0, 1.0 / k["xi"], torch.zeros_like(k["xi"]))   # tf divide_no_nan
    arg = (1.0 + k["xi"] * raw / k["sigma"]).clamp(min=1e-6)
    loglik = torch.log(k["std"]) - torch.log(k["sigma"]) - (1.0 + inv) * torch.log(arg)
    penalty = torch.where(raw > k["u"], -loglik, torch.zeros_like(loglik))
    return se, b["ym"].sum(), penalty.max(dim=2).values.sum()


def consts(p: Pollutant, device):
    t = lambda a: torch.tensor(a, device=device)
    return {"mean": t(p.mean), "std": t(p.std), "xi": t(p.xi), "sigma": t(p.sigma),
            "u": torch.tensor(p.u, device=device)}


@torch.no_grad()
def validate(model, p, windows, k, device, bs):
    """MSE (z-units, observed targets) at AQFM's validation series: each window at its 12 stations."""
    model.eval()
    se = n = 0.0
    ae = 0.0
    for i in range(0, len(windows), bs):
        chunk = windows[i:i + bs]
        b = p.windows([t0 for t0, _ in chunk], device)
        pred = model(b["x"])
        sel = torch.zeros_like(b["ym"])
        for j, (_, stations) in enumerate(chunk):
            sel[j, :, stations] = 1.0
        m = b["ym"] * sel
        yz = (b["y"] - k["mean"]) / k["std"]
        se += float(((pred - yz) ** 2 * m).sum())
        ae += float(((k["mean"] + k["std"] * pred).clamp(min=0) - b["y"]).abs().mul(m).sum())
        n += float(m.sum())
    model.train()
    return {"mse": se / n, "mae": ae / n}


def validation_windows(val_start, test_start, coords, n_windows):
    """AQFM's validation windows with the stations each one covers (common.validation_series, grouped)."""
    series = common.validation_series(val_start, test_start, coords, n_windows)
    k = common.NEIGHBOURS if len(coords) >= common.NEIGHBOURS else len(coords)
    return [(series[i][0], [s for _, s in series[i:i + k]]) for i in range(0, len(series), k)]


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    a = common.load_preprocessed(args.data)
    val_start, test_start = common.split_indices(np.asarray(a["timestamps_h"]))
    print(f"loading {args.target} (train_end={val_start}, test_start={test_start}) ...", flush=True)
    p = Pollutant(Series(a["values"], a["observed"], val_start), args.target, val_start)
    k = consts(p, device)
    adj = adjacency(a["coords"])
    model = ESTGCN(adj, args.out_feat, args.lstm_units).to(device)
    print(f"parameters: {sum(q.numel() for q in model.parameters())}  edges: {int(adj.sum())}  "
          f"isolated stations: {int((adj.sum(1) == 0).sum())}  device={device}", flush=True)
    print("GPD per station (xi, sigma): " + " ".join(f"({x:.3f},{s:.1f})" for x, s in zip(p.xi, p.sigma)), flush=True)
    val_windows = validation_windows(val_start, test_start, a["coords"], args.val_windows)
    hi = val_start - (CONTEXT + HORIZON)
    opt = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-7)    # Keras defaults

    os.makedirs(args.out, exist_ok=True)
    code = common.code_version()
    print("code: " + ("not a git checkout" if code is None else
                      code["commit"][:7] + (f" + uncommitted changes {code['changed']}" if code["dirty"] else "")))
    meta = {"model": NAME, "target": args.target, "args": vars(args), "val_start": val_start, "test_start": test_start,
            "adjacency": adj.tolist(), "norm_mean": p.mean.tolist(), "norm_std": p.std.tolist(),
            "gpd_xi": p.xi.tolist(), "gpd_sigma": p.sigma.tolist(), "threshold": p.u, "code": code}
    json.dump(meta, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    save = lambda name, extra: torch.save({"state_dict": model.state_dict(), **meta, **extra},
                                          os.path.join(args.out, name))

    best, bad, t_start = float("inf"), 0, time.time()
    model.train()
    for i in range(args.steps):
        rng = np.random.default_rng([args.seed, i])
        b = p.windows(rng.integers(0, hi + 1, args.batch_size), device)
        pred = model(b["x"])
        se, n, pen = loss_terms(pred, b, k)
        mse, gpd = se / n.clamp(min=1), pen / (pred.shape[0] * pred.shape[1])
        loss = BETA[0] * mse + BETA[1] * gpd
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i + 1) % 100 == 0 or i == 0:
            rec = {"step": i + 1, "loss": loss.item(), "mse": mse.item(), "gpd": gpd.item(),
                   "s_per_step": (time.time() - t_start) / (i + 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if (i + 1) % 500 == 0 or i == 0:
                print(" ".join(f"{k_}={v:.4g}" if isinstance(v, float) else f"{k_}={v}" for k_, v in rec.items()),
                      flush=True)
        if (i + 1) % args.val_every == 0 or i + 1 == args.steps:
            val = validate(model, p, val_windows, k, device, args.eval_batch)
            log.write(json.dumps({"step": i + 1, "val": val}) + "\n"); log.flush()
            print(f"  [val] step {i + 1}: " + " ".join(f"{k_}={v:.4f}" for k_, v in val.items()), flush=True)
            save("last.pt", {"step": i + 1, "val": val})
            if val["mse"] < best:
                best, bad = val["mse"], 0
                save("best.pt", {"step": i + 1, "val": val})
                print(f"  [val] new best {best:.4f} -> best.pt", flush=True)
            else:
                bad += 1
                if args.patience and bad >= args.patience:
                    print(f"  early stop: no improvement in {bad} validations")
                    break
    print(f"done. best val mse={best:.4f}  ({(time.time() - t_start) / 60:.1f} min)")


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    a = common.load_preprocessed(args.data)
    val_start, test_start = common.split_indices(np.asarray(a["timestamps_h"]))
    st = Series(a["values"], a["observed"], val_start)
    adj = adjacency(a["coords"])
    models = {t: [] for t in TARGETS}
    cks = []
    for path in args.ckpt:
        ck = torch.load(os.path.join(common.ROOT, path), map_location=device, weights_only=False)
        assert ck["test_start"] == test_start and ck["val_start"] == val_start, "split mismatch"
        assert np.array_equal(np.asarray(ck["adjacency"], np.float32), adj), "different station graph"
        m = ESTGCN(adj, ck["args"]["out_feat"], ck["args"]["lstm_units"]).to(device)
        m.load_state_dict(ck["state_dict"])
        models[ck["target"]].append(m.eval())
        cks.append(ck)
    assert all(models[t] for t in TARGETS), "need at least one checkpoint per pollutant"
    pols = {t: Pollutant(st, t, val_start) for t in TARGETS}
    for t in TARGETS:                                          # the statistics the models were trained with
        ck = next(c for c in cks if c["target"] == t)
        assert np.allclose(ck["norm_mean"], pols[t].mean) and np.allclose(ck["norm_std"], pols[t].std), t

    pairs, n_origins = common.test_pairs(test_start, st.T, st.S)
    origins_t0 = sorted({t0 for t0, _ in pairs})
    pred = np.zeros((len(origins_t0), HORIZON, st.S, len(TARGETS)), np.float32)
    for c, t in enumerate(TARGETS):
        k = consts(pols[t], device)
        for i in range(0, len(origins_t0), args.eval_batch):
            b = pols[t].windows(origins_t0[i:i + args.eval_batch], device)
            z = sum(m(b["x"]) for m in models[t]) / len(models[t])            # mean over seeds (z-units)
            pred[i:i + len(b["x"]), :, :, c] = (k["mean"] + k["std"] * z).clamp(min=0).cpu().numpy()
    row = {t0: j for j, t0 in enumerate(origins_t0)}
    t0s, ss = np.array(pairs).T
    p = pred[[row[t0] for t0 in t0s], :, ss]                   # (N, HORIZON, 2), in the order of `pairs`
    b = st.batch(ss, t0s)
    res = common.score(NAME, b["y"].numpy(), p, b["ym"].numpy())
    res["n_origins"], res["n_windows"] = int(n_origins), len(pairs)
    res["selected"] = [{"target": ck["target"], "step": ck["step"], "val": ck["val"]} for ck in cks]
    res["checkpoint"] = args.ckpt
    res["code"] = common.code_version()
    res["train_code"] = [ck["code"] for ck in cks]
    res["args"] = vars(args)
    out = os.path.join(common.ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, default=str)
    for t in TARGETS:
        print(t, {hz: res[NAME][t][hz]["MAE"] for hz in common.HORIZONS})
    print("saved", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--target", choices=TARGETS, required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--steps", type=int, default=20000)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--val-every", type=int, default=1000)
    tr.add_argument("--val-windows", type=int, default=512, help="AQFM windows (x 12 stations each)")
    tr.add_argument("--patience", type=int, default=0,
                    help="validations without improvement before stopping; 0 = run the whole schedule")
    tr.add_argument("--out-feat", type=int, default=10)
    tr.add_argument("--lstm-units", type=int, default=64)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--ckpt", required=True, nargs="+",
                    help="paths from the repo root, at least one per pollutant; several per pollutant = mean forecast")
    ev.add_argument("--out", required=True, help="results file, path from the repo root")
    for q in (tr, ev):
        q.add_argument("--data", default=os.path.join(common.ROOT, "data", "preprocessed"))
        q.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        q.add_argument("--eval-batch", type=int, default=64)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
