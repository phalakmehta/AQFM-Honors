# E-STGCN (our re-implementation)

E-STGCN: Panja, Chakraborty, Biswas and Deb, *Extreme Spatiotemporal Graph Convolutional Networks for Air
Quality Forecasting*, JRSS Series A (2026), doi 10.1093/jrsssa/qnag010, arXiv 2411.12258. It reports
results on 37 Delhi CPCB stations, so it is the closest published competitor.

**Why a port and not their code or their numbers.** Their published numbers are for a different task:
daily means, train 2019–2022, test 2023, 30/60/90 days ahead. Ours: 15-min values, 1–72 h ahead, test 2025.
Error sizes are not comparable across those tasks. Their code (`github.com/mad-stat/E_STGCN`: one Keras
notebook and an R script) is built for the daily task and the repository has **no license**, so we wrote our
own PyTorch version from the paper and the notebook. None of their files is copied here. Cite the paper and
call the results "E-STGCN (our re-implementation)".

```bash
bash deep_baselines/estgcn/pipeline_estgcn.sh                 # everything below, ~see the chain header
python -m pytest deep_baselines/estgcn/tests -q               # 10 tests
```

## Kept as published

- Graph convolution over the 37 stations: each station's input times W, concatenated with the **mean** of its
  neighbours' inputs times the same W (out_feat 10, no activation); isolated stations get zeros (as Keras).
- Graph: binary adjacency from Haversine distances, `exp(-(d/10000)^2 / 0.75) >= 0.5`, no self loops.
- LSTM with 64 units over the input window, per station; a dense layer to the whole horizon.
- One model per pollutant (PM2.5 and PM10 separately), univariate input (no weather).
- Per-station z-scoring with the training mean and std.
- Loss `0.99 × MSE + 0.01 × GPD penalty`: the generalised-Pareto negative log-likelihood of the forecast where
  it exceeds the NAAQS threshold (60 µg/m³ PM2.5, 100 PM10), maximum over stations, as in their notebook (the
  penalty is evaluated at the forecast itself, not at the exceedance over the threshold, exactly as written
  in their Eq. 6 and code). GPD shape and scale per station by maximum likelihood on the training exceedances.
  A test checks our loss equals their notebook's formula for one sample.
- RMSprop (Keras defaults), batch 64, constant learning rate; Keras weight initialisation.

## Changed, and why

| Change | Why |
|---|---|
| 15-min values, input 512 steps (128 h), output 288 steps (72 h) | our protocol; same context as AQFM and PatchTST (theirs: daily, 24-day input, 30–90-day output) |
| loss over **every** sample of the batch | their notebook's loss reads `y_true[0]`, `y_pred[0]`: only the first sample of each batch is trained (a test checks every sample counts) |
| inputs forward-filled (causal); MSE on observed targets only | our data has gaps; same rule as PatchTST |
| LSTM with tanh | the paper's LSTM uses tanh; the notebook's ReLU comes from the Keras example it is built on, and is unstable over 512 steps |
| distances read as metres (edges within ~7.2 km; median 7 neighbours, 5 stations isolated) | the notebook does not state the unit; with km every station links to all 36 others and ε would not control sparsity, which the paper says it does |
| GPD fit with scipy (MLE) on 15-min training values | theirs: R `evmix::fgpd` on daily training values |
| selection: AQFM's 512 fixed 2024 windows at the same 12 stations, MSE every 1000 steps, whole schedule, best checkpoint | the same selection as AQFM and PatchTST (theirs: 180 validation days, early stopping without restoring the best weights) |
| learning rate 1e-4 / 2e-4 / 4e-4 on seed 0, then seeds 1 and 2 of the winner | the same tuning budget as AQFM and PatchTST (3 values, chosen on 2024 validation); theirs is 2e-4 |
| station coordinates from `data/stations.csv` (geocoded) | their distance file is not public |
