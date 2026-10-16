# PatchTST on our 2025 protocol (Stage F1, 2026-10-16)

Code, setup and caveats: `deep_baselines/README.md`, `deep_baselines/patchtst.py`. Chain:
`deep_baselines/pipeline_patchtst.sh`. Every number below is copied from `results/TABLES.md` (generated);
regenerate that file, not this one, if anything changes.

## Like for like

PatchTST got the same treatment as AQFM: the same 2025 test points scored by the same code (the same `n`
in every cell), the same 128 h input window, training windows drawn the same way from 2018–2023, selection
on the same 512 fixed 2024 windows at the same 12 stations each, the same tuning budget (dropout
0.1 / 0.2 / 0.3 on seed 0 → 0.1 chosen, then seeds 1 and 2), the best validation checkpoint, a median
point forecast (L1 loss), 3 seeds and a 3-seed ensemble.

**What PatchTST sees:** one station's PM2.5 and PM10, each series on its own (channel independence): no
weather, no neighbours, no other pollutants. It never sees future weather, so its fair AQFM partner is
**AQFM without future weather**. AQFM with future ERA5 is compared with XGBoost + future ERA5.

## 2025 test, PM2.5 MAE (µg/m³), models without future weather

| model | 1h | 6h | 24h | 72h | Severe, 24h |
|---|---|---|---|---|---|
| AQFM no future met, ensemble (3 seeds) | **15.77** | **27.97** | 38.41 | 43.20 | **104.9** |
| AQFM no future met, mean ± std (3 seeds) | 16.00 ± 0.42 | 28.33 ± 0.10 | 38.98 ± 0.24 | 43.77 ± 1.04 | 105.7 ± 1.2 |
| PatchTST ensemble (3 seeds) | 16.26 | 28.04 | **37.75** | 43.31 | 110.6 |
| PatchTST, mean ± std (3 seeds) | 16.50 ± 0.26 | 28.14 ± 0.15 | 37.82 ± 0.12 | 43.37 ± 0.18 | 110.7 ± 1.6 |
| XGBoost (no future met) | 18.17 | 30.06 | 38.63 | 43.74 | 110.5 |
| IBM TTM (zero-shot, 15-min scoring) | 20.95 | 32.13 | 39.16 | **42.75** | 111.0 |
| Persistence | 15.89 | 35.53 | 43.78 | 54.54 | 119.3 |

PM10 MAE, same models (1/6/24/72 h): AQFM no future met mean 28.63 / 46.63 / 73.54 / 83.06, ensemble
28.46 / 46.02 / 72.49 / 81.98; PatchTST mean 31.00 / 47.64 / 73.96 / 84.54, ensemble 30.58 / 47.49 / 73.88 /
84.44; XGBoost 34.71 / 49.44 / 73.65 / 83.46.

## What it says

- **Same information, average error: a near tie.** Without future weather, AQFM is better at 1 h (single
  models 16.00 vs 16.50), level at 6 h and 72 h (within seed noise), and **PatchTST is better at 24 h**
  (37.82 ± 0.12 vs 38.98 ± 0.24, single models; 37.75 vs 38.41, ensembles). PatchTST beats XGBoost at every
  lead. At 72 h every model without future weather is within ~1 µg/m³ (TTM lowest, 42.75).
- **Where AQFM's extra inputs pay off without future weather:** the Severe band (105.7 ± 1.2 vs
  110.7 ± 1.6 at 24 h; but E-STGCN, 99.8 ± 2.9, beats both there: `results/estgcn_summary.md`), PM10 at short leads (1 h 28.63 vs 31.00; 6 h 46.63 vs 47.64), and AQFM gives
  calibrated intervals, CRPS and episode probabilities, which PatchTST does not.
- **AQFM's large 24–72 h advantage comes with future weather:** AQFM ensemble 32.04 at 24 h vs PatchTST
  37.75. That weather is reanalysis, so this is an upper bound on operational skill (E1, oracle gap).
- For the paper: report AQFM without future met next to PatchTST and plain XGBoost, and AQFM with
  future met next to XGBoost + future ERA5. A strong channel-independent model matching AQFM-without-weather
  at 24 h is a finding to state, not hide.

## Caveats

- **Stopping rule.** AQFM stops after 3 validations without improvement (its validation rises after the
  peak). PatchTST's validation keeps falling slowly, so it trains its whole 40k-step schedule; the first
  attempt (20k steps, patience 3) stopped 2 of 5 runs early on a noisy flat stretch and was redone. Its logs
  and results are kept in `deep_baselines/runs/` (`patchtst_do*`, `patchtst_attempt1_results/`) and are not
  reported. The redo was decided on the validation curves only and can only have helped PatchTST.
- **Ensembles.** AQFM pools the 3 models' predictive samples and takes the median; PatchTST (a point model)
  averages the 3 forecasts.
- **Training years.** AQFM and PatchTST train on 2018–2023 and use 2024 only for selection; XGBoost and
  climatology train on 2018–2024 (`baselines/`, unchanged). That gives them one more year of data, so AQFM's
  margins over XGBoost are, if anything, understated.
- **Point forecasts only.** PatchTST has no CRPS, coverage or episode rows in `TABLES.md`.
