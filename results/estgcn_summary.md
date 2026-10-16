# E-STGCN on our 2025 protocol, and all models without future weather (Stage F4, 2026-10-16)

E-STGCN (Panja et al., JRSS-A 2026) is the closest published competitor (37 Delhi CPCB stations). Its code
does not fit our task and has no license, so this is **our PyTorch re-implementation**
(`deep_baselines/estgcn/`; what was kept and changed, and why: `deep_baselines/estgcn/README.md`). Chain:
`deep_baselines/estgcn/pipeline_estgcn.sh` (commit 16ed23b). Every number below is copied from
`results/TABLES.md`; regenerate that file, not this one. PatchTST: `results/patchtst_summary.md`.

## Like for like

E-STGCN got the same treatment as AQFM and PatchTST: the same 2025 test points and metric code, the same
128 h input, training windows from 2018–2023, selection on AQFM's 512 fixed 2024 windows at the same 12
stations, the same tuning budget (3 learning rates on seed 0, 1e-4 / 2e-4 / 4e-4 around theirs, 2e-4; then
seeds 1 and 2; chosen: 4e-4 for PM2.5, 2e-4 for PM10), the best validation checkpoint, 3 seeds and a 3-seed
ensemble. As published, it uses no weather and one model per pollutant, and it trains on 0.99 × MSE + 0.01 ×
an extreme-value (GPD) penalty.

## 2025 test, PM2.5 (µg/m³), all models without future weather

| model | MAE 1h | MAE 6h | MAE 24h | MAE 72h | MAE Severe, 24h | RMSE 1h | RMSE 24h |
|---|---|---|---|---|---|---|---|
| AQFM no future met, ensemble (3 seeds) | **15.77** | **27.97** | 38.41 | **43.20** | 104.9 | 34.15 | 63.98 |
| AQFM no future met, mean ± std (3 seeds) | 16.00 ± 0.42 | 28.33 ± 0.10 | 38.98 ± 0.24 | 43.77 ± 1.04 | 105.7 ± 1.2 | 34.37 ± 0.41 | 64.89 ± 0.84 |
| PatchTST ensemble (3 seeds) | 16.26 | 28.04 | **37.75** | 43.31 | 110.6 | 33.82 | 64.39 |
| PatchTST, mean ± std (3 seeds) | 16.50 ± 0.26 | 28.14 ± 0.15 | 37.82 ± 0.12 | 43.37 ± 0.18 | 110.7 ± 1.6 | 34.12 ± 0.87 | 64.48 ± 0.17 |
| E-STGCN ensemble (3 seeds; our re-implementation) | 16.37 | 30.91 | 38.64 | 44.62 | **98.7** | **29.99** | **63.61** |
| E-STGCN, mean ± std (3 seeds) | 17.11 ± 0.50 | 31.52 ± 0.35 | 39.05 ± 0.48 | 45.19 ± 0.26 | 99.8 ± 2.9 | 30.84 ± 0.57 | 64.25 ± 1.10 |
| XGBoost (no future met) | 18.17 | 30.06 | 38.63 | 43.74 | 110.5 | 35.26 | 66.00 |

PM10 MAE (1/6/24/72 h), single models: AQFM no future met 28.63 / 46.63 / 73.54 / 83.06, PatchTST
31.00 / 47.64 / 73.96 / 84.54, E-STGCN 31.54 / 51.83 / 73.05 / 85.00, XGBoost 34.71 / 49.44 / 73.65 / 83.46.

## What it says

- **Average error (MAE):** without future weather, AQFM and PatchTST lead; E-STGCN is behind both at 6, 24
  and 72 h (24 h single models: AQFM 38.98, PatchTST 37.82, E-STGCN 39.05), behind plain XGBoost at 6 and 72 h,
  and level with it at 24 h (ensemble 38.64 vs 38.63). It beats XGBoost at 1 h.
- **Severe band (PM2.5 > 250) at 24 h:** without future weather, **E-STGCN is best** (99.8 ± 2.9 vs AQFM
  105.7 ± 1.2 and PatchTST 110.7 ± 1.6). That is what its extreme-value loss is for. It pays for it in
  the cleaner bands: Good 19.5 ± 0.7 vs 17.3 ± 1.3 (AQFM) and 16.5 ± 0.3 (PatchTST) (`TABLES.md`, AQI-band table).
- **RMSE:** E-STGCN is best at 1 h (30.84 vs ~34). It is trained on squared error (a mean forecast), while
  AQFM and PatchTST give a median, which is what MAE rewards. Report MAE and RMSE together.
- **With future weather**, AQFM is ahead of every model, including in the Severe band (AQFM ensemble 89.0;
  XGBoost + future ERA5 93.4). The fair comparison there is XGBoost + future ERA5, since none of the deep
  baselines can use weather (see `results/patchtst_summary.md` for the wording).
- For the paper: AQFM's Severe-band advantage **without** weather does not hold against E-STGCN; state it.
  A natural next step for AQFM is an extreme-value term in its loss (E-STGCN's idea), but that is new work,
  not part of this comparison.

## Caveats

- **A re-implementation.** Their notebook is for daily data; ours uses 15-min data, 128 h in, 72 h out. Fixes:
  the loss uses every batch sample (theirs only the first), causal forward fill, MSE on observed targets
  only. Unstated details we had to choose: distances in metres (graph edges within ~7.2 km), tanh LSTM (as in
  the paper; the notebook says ReLU). All in `deep_baselines/estgcn/README.md`.
- **Tuning budget.** PM2.5 chose 4e-4, the largest learning rate tried. AQFM's dropout (0.3) and PatchTST's
  (0.1) also sat on an edge of their grids; every model got 3 values and none was extended.
- **Constant learning rate** (as their code): validation curves are noisy plateaus; best checkpoints at
  5k–18k of 20k steps, none still improving at the end (`deep_baselines/estgcn/runs/*/log.jsonl`).
- **Ensembles:** mean of the 3 forecasts per pollutant (AQFM pools predictive samples).
- **Training years:** AQFM, PatchTST and E-STGCN train on 2018–2023; XGBoost on 2018–2024.
