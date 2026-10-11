"""Measure real training speed and VRAM on whatever GPU you have.

    python benchmark_gpu.py                 # tries batch sizes 2, 4, 8, 16
    python benchmark_gpu.py --batch 4 8

Prints s/step, s/sample, peak VRAM and projected wall-clock, so you can choose
--batch-size and --steps for `python -m aqfm.train` from measurement instead of
guesswork. Reuses one batch (data loading is ~4 ms/sample, not the bottleneck).
"""
import argparse, time, torch
from aqfm import AQFM, AQFMConfig, AQData, TrainWindows, synthetic_raw
from aqfm.train import step, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, nargs="*", default=[2, 4, 8, 16])
ap.add_argument("--iters", type=int, default=6)
ap.add_argument("--no-checkpoint", action="store_true")
ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = default)")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = dev.type == "cuda"
amp = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else None
print(f"device: {torch.cuda.get_device_name(0) if cuda else 'CPU'}")
if cuda:
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB   bf16: {amp is not None}")
else:
    print("WARNING: no CUDA found - this measures CPU speed, which is ~50-100x slower")

cfg = AQFMConfig(grad_checkpoint=not a.no_checkpoint)
v, o, t, c, s, e = synthetic_raw(37, 12000)
data = AQData(v, o, t, c, s, train_end=10000, era5_met=e)
ds = TrainWindows(data, cfg, 0, 8000, 64, pretrain=True, seed=0)
model = AQFM(cfg).to(dev)
model.revin.set_global_stats(data.global_mean, data.global_std)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
print(f"params: {model.n_params()/1e6:.2f}M   grad_checkpoint={cfg.grad_checkpoint}\n")
print(f"{'batch':>6} {'s/step':>8} {'s/sample':>9} {'peak GB':>8}   projected wall-clock (main phase)")

for B in a.batch:
    try:
        batch = to_device(torch.utils.data.default_collate([ds[j] for j in range(B)]), dev)
        if cuda:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for i in range(a.iters + 2):
            if i == 2:
                if cuda: torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                loss, _, _, _ = step(model, batch, cfg)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if cuda: torch.cuda.synchronize()
        per = (time.time() - t0) / a.iters
        gb = torch.cuda.max_memory_allocated() / 1e9 if cuda else float("nan")
        proj = "  ".join(f"{n//1000}k: {per*n/3600:.1f} h" for n in (20000, 50000, 100000))
        print(f"{B:>6} {per:>8.2f} {per/B:>9.3f} {gb:>8.2f}   {proj}")
        del batch
    except torch.cuda.OutOfMemoryError:
        print(f"{B:>6} {'OOM':>8}   -> use a smaller batch")
        torch.cuda.empty_cache()
    except RuntimeError as ex:
        print(f"{B:>6} {'ERROR':>8}   {str(ex)[:60]}")
