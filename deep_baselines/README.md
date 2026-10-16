# Deep baselines (Stage F)

Published deep forecasting models, trained on our data and tested on **exactly** the 2025 protocol of
AQFM and `baselines/`. This folder only *imports* `aqfm/` and `baselines/baseline_utils.py`; it never
changes them, so the code behind the existing results stays as it was.

| File | What |
|---|---|
| `common.py` | data, splits, windows, RevIN, AQFM's validation set, scoring (shared by every deep baseline) |
| `patchtst.py` | PatchTST (Nie et al., ICLR 2023): `train` and `evaluate` (from a checkpoint, so it reproduces exactly) |
| `seed_stats.py` | mean ± std over seeds, in the format `scripts/make_tables.py` reads |
| `pipeline_patchtst.sh` | the chain that produced the PatchTST results (dropout sweep → seeds → 2025 test) |
| `tests/` | `python -m pytest deep_baselines/tests -q` (not run by `scripts/check_submission.py`, which runs `tests/`) |
| `runs/` | per-run `config.json`, `log.jsonl`, console logs (tracked); checkpoints `*.pt` are git-ignored and on Drive |
| `estgcn/` | E-STGCN (Panja et al., JRSS-A 2026), our PyTorch re-implementation: its own folder, code, tests, chain and runs; see `estgcn/README.md` |

Results go to `baselines/results/patchtst_*.json` (so `check_submission.py` and `make_tables.py` see them),
every result is registered in `scripts/manifest.json`, and the numbers are quoted from `results/TABLES.md`.
Comparison and caveats: `results/patchtst_summary.md`.

## Like for like: what is the same as for AQFM

| | AQFM | PatchTST |
|---|---|---|
| test points, metric code | 2025 daily origins, 15-min points, `baseline_utils` | same (same `n` in every cell) |
| input window | 512 steps (128 h) | same |
| training data | windows ending before 2024 | same, drawn the same way |
| validation | 512 fixed 2024 windows × 12 stations | the same windows and stations |
| tuning budget | dropout 0.1 / 0.2 / 0.3 on seed 0, then seeds 1, 2 | same |
| checkpoint | best on validation | same |
| point forecast | median of 200 samples | L1 loss (median) |
| seeds, ensemble | 3, pooled samples | 3, mean of the forecasts |

**What differs by design:** PatchTST sees one station's PM2.5 and PM10 only (each series on its own:
no weather, no neighbours, no other pollutants), so compare it with AQFM *without* future weather.
**Stopping:** AQFM stops after 3 validations without improvement (its validation rises after the peak);
PatchTST's validation keeps falling slowly, so it trains its whole 40k-step schedule. The first attempt
used patience 3 for both and stopped 2 of 5 PatchTST runs early; its logs are kept in `runs/` (`patchtst_do*`).
