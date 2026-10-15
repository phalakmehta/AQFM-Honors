# AQFM Stage E ablations (2026-10-15)

**What was run.** Each ablation is the final recipe (latest-value anchor, dropout 0.3, 12k-step cosine
schedule, weight decay 0.05, early stopping with patience 3 on the 2024 validation score, best checkpoint
kept), **seed 0**, with exactly **one** switch changed. Chain: `scripts/pipeline_2026-10-15_ablations.sh`
(log `runs/pipeline_ablations.log`), code commit `4b0c7d6` for every model. Each model was tested on 2025
once, with the same protocol and sampling seed as every other AQFM result. Nothing here selected anything:
the final setup was fixed before these runs.

**Where the numbers are.** `results/TABLES.md`, tables "ablations (final recipe, seed 0, one change each)"
and "ablations, change in PM2.5 MAE against the full model". Raw files: `results/aqfm_do3_<tag>_s0_results.json`
(tags `nospatial`, `nowind`, `pm25direct`, `gapfillonly`, `noblh`; E1 is `nofm`). Every number below is copied
from `TABLES.md`.

**How to read them.** Compare each ablation with the full model at seed 0. One seed moves the PM2.5 MAE by
a std of 0.17 / 0.08 / 0.76 / 1.21 at 1 / 6 / 24 / 72 h and 0.6 in the Severe band (std over seeds 0/1/2 of
the full model, itself estimated from only 3 seeds). A change smaller than about twice that std cannot be
told apart from seed noise with one seed.

## Results: change in PM2.5 MAE against the full model, seed 0 (µg/m³; positive = worse)

| Ablation | 1 h | 6 h | 24 h | 72 h | Severe, 24 h | Verdict (single seed) |
|---|---|---|---|---|---|---|
| E1 no future weather | +0.22 | **+0.39** | **+4.80** | **+4.10** | **+16.1** | small loss at 6 h, large from 24 h and in the Severe band |
| E2 no boundary-layer height | **-0.36** | +0.09 | -1.08 | +1.08 | **-1.9** | no loss on 2025 (slightly better at 1 h and in the Severe band); worse on 2024 validation |
| E3 no spatial attention | -0.28 | **+0.34** | **+1.90** | +0.87 | **+1.8** | neighbours help at 6–24 h and in the Severe band |
| E3 distance only, no wind | **-0.39** | **-0.41** | +1.19 | +0.99 | **+3.4** | wind helps in the Severe band; without it 1–6 h is slightly better; 24 h within noise |
| E4 PM2.5 predicted directly | **-1.20** | -0.08 | -0.46 | -0.25 | **+3.2** | better at 1 h, worse in the Severe band |
| E5 gap-filling only | **+15.91** | **+6.50** | **+3.80** | +1.09 | **+18.9** | forecasting collapses at short leads |
| E7 no anchor (run 1 vs the dropout-0.1 anchor run) | **+17.55** | **+4.96** | +1.40 | -0.77 | **+9.7** | the anchor is what fixes short leads |

Bold = more than twice the seed std (computed, not judged by eye). The stds come from only 3 seeds; the 6 h one (0.08) is probably an underestimate, so small 6 h changes are flagged too easily.

## What each one says, in plain words

- **E1 — future weather (ERA5) is the biggest single input.** Without it, 24 h error rises by 4.8 and the
  Severe band by 16.1. This is the oracle gap: ERA5 is reanalysis, not a forecast, so the headline numbers
  are an upper bound on operational skill. With 3 seeds, AQFM without future weather is 16.00 ± 0.42 /
  28.33 ± 0.10 / 38.98 ± 0.24 / 43.77 ± 1.04 at 1 / 6 / 24 / 72 h (ensemble 15.77 / 27.97 / 38.41 / 43.20),
  against plain XGBoost 18.17 / 30.06 / 38.63 / 43.74: better at 1–6 h, a tie at 24–72 h.
- **E2 — boundary-layer height (BLH) shows no loss on 2025**, against the design doc's expectation of the
  largest single-feature effect. The 2024 validation score did get worse (1.7438 vs 1.7299, about 6 seed
  stds), but on 2025 the model without BLH is slightly better at 1 h (-0.36) and in the Severe band (-1.9),
  and within noise at 6–72 h. BLH was removed everywhere (input channel never observed, future-ERA5 BLH set to
  its training mean); the model still sees ERA5 temperature, solar radiation and wind, which follow the same
  daily cycle as BLH. That is a likely reason; it was not tested.
- **E3 — looking at neighbouring stations helps from 6 h on** (+0.34 at 6 h, +1.90 at 24 h and +1.8 in the
  Severe band without spatial attention; PM10 24 h +2.65). Removing only the wind direction (distance-only
  bias) costs clearly in the Severe band (+3.4) but not at 24 h (+1.19, within noise), and is slightly better
  at 1–6 h (-0.39, -0.41).
- **E4 — the PM10 × fraction trick is a trade-off, not a free win.** Predicting PM2.5 directly is clearly
  better at 1 h (-1.20; with the anchor, the model then sees the latest PM2.5 value itself) and slightly
  better on PM10, but worse in the Severe band (+3.2) and less well calibrated (90% coverage 0.881 vs 0.906).
  The reparameterisation also guarantees PM2.5 ≤ PM10 in every sample. The direct model breaks it rarely:
  0.0015 of its point forecasts have PM2.5 > PM10, about the rate in the observed data itself (0.0012), so
  coherence is not a practical problem at the point-forecast level.
- **E5 — the forecasting objective is essential.** A model trained only to fill gaps fills a 24 h outage as
  well as the full model (PM2.5 imputation 30.45 vs 31.96) but forecasts far worse (1 h +15.91, 24 h +3.80).
  This is exactly the design doc's hypothesis ("imputation holds, forecasting degrades").
- **E7 — the latest-value anchor fixes short leads** (+17.55 at 1 h without it). Not re-run: run 1 and the
  dropout-0.1 anchor run differ only in the anchor and the planned schedule length (PROJECT_CONTEXT.md §6).

## Caveats

- **One seed each.** E3 and E4 are near the noise threshold; confirming them needs seeds 1 and 2 (about
  1 h 5 min per model). Not run: that is a decision for the user.
- **Validation and test disagree for three ablations.** Without wind, without spatial attention, and with
  PM2.5 predicted directly, the 2024 validation score is slightly *better* than the full model's (1.7193,
  1.7252, 1.7228 vs 1.7299; seed std 0.0023), yet 2025 is worse at 24 h or in the Severe band. The validation
  score averages all 8 target channels over all 72 h, while the test tables score PM2.5 and PM10 at fixed
  leads, so the two measure different things. These ablations were never selection candidates. Changing the
  final model because of them would use 2025 information, which the project rules forbid.
- **E7 compares older runs** (dropout 0.1; run 1 used a 45k-step schedule), not the final recipe.

## Fix found while running (2026-10-15)

The new "PM2.5 > PM10" share in `evaluate.py` (written for E4) always returned 0: a Python late-binding
mistake made both pollutants read the PM10 arrays. Only that one metric in the E4 results file was affected.
The 2025 data does contain such points (0.126% of observed points where both are measured). Fixed in commit
`a509c69` (the metric is now a function, tested on arrays with a known answer and end to end on data that
contains PM2.5 > PM10; both tests fail on the old code), and the E4 evaluation was re-run with the committed
fix: 346 of its 349 numbers came out bit-identical, and only the three PM2.5 > PM10 values changed (forecasts
0.0015, observed 0.0012).
