# AQFM — spatio-temporal foundation model for multi-pollutant air quality forecasting

Reference implementation of the design in `design/aqfm_design.tex`.

## What this is

A model that takes multi-station, multi-pollutant, irregularly-sampled
observations with station-wide block gaps, and emits calibrated probabilistic
forecasts for many pollutants at once, at any requested subset of horizons.

Three design decisions carry most of the weight:

1. **Channel-independent patch tokens.** One token = one patch of one channel at
   one station. Missing channel → fewer tokens. Missing patch → token excluded
   from the key set. Station identity from coordinates, not a lookup table, so
   unseen stations work.
2. **Query-based decoder** rather than one decoder per pollutant. The query is
   `(target channel, station, lead patch)`, so specialisation lives in the query
   embedding and arbitrary target subsets work at inference. Query
   self-attention gives cross-pollutant coherence that independent heads cannot.
3. **Distributional output with physical constraints.** Mixture of log-normals,
   NLL + CRPS, and PM2.5 predicted as `PM10 × sigmoid(frac)` so `PM2.5 > PM10`
   is impossible by construction.

## Install and run

```bash
pip install torch numpy
export PYTHONPATH=$PWD

python tests/test_aqfm.py                        # full test suite
python -m aqfm.train --small --steps 100         # CPU smoke run on synthetic data
python -m aqfm.train --steps 5000                # full config (needs a GPU)
```

## Layout

| File | Contents |
|---|---|
| `aqfm/config.py` | `AQFMConfig`, plus `.small()` for tests |
| `aqfm/layers.py` | masked RevIN, Fourier time features, masked attention, axial block, wind-aware spatial bias |
| `aqfm/model.py` | tokenizer, encoder, per-station latent pooler, query decoder, mixture head |
| `aqfm/losses.py` | mixture NLL, order-statistic CRPS, episode weighting |
| `aqfm/masking.py` | tail / block / channel-dropout mask sampler; empirical outage-length fitting |
| `aqfm/physical.py` | ratio channels and constrained reconstruction |
| `aqfm/data.py` | windowing, neighbourhood sampling, query construction, synthetic generator |
| `aqfm/train.py` | two-stage training, evaluation, baselines |
| `tests/test_aqfm.py` | shape, leakage, masking, constraint and loss-ordering tests |

## Using your own data

`AirQualityWindows` wraps dense arrays. Align the 38 ragged CPCB CSVs to a
common 15-minute index once, offline, and memory-map the result — do not do the
alignment per batch.

```python
values      # (n_stations, 17, n_steps) float32, raw units
observed    # (n_stations, 17, n_steps) float32, 1 = measured
timestamps  # (n_steps,) float64, hours since epoch
coords      # (n_stations, 2) lat/lon
static      # (n_stations, n_static)
```

`log1p` and the derived ratio channels are applied inside the dataset; pass raw
concentrations.

**Fit the outage distribution before pretraining.** Do not use the default
lognormal parameters:

```python
from aqfm.masking import MaskSampler
mu, sigma = MaskSampler.fit_block_lognormal(observed[station, 0])
cfg.block_len_lognormal_mu, cfg.block_len_lognormal_sigma = mu, sigma
```

## Two things that are load-bearing and easy to get wrong

**Activation memory, not FLOPs, is the binding constraint.** The axial encoder
stores ~1.5 GB of activations per sample at the default size. Set
`grad_checkpoint=True` — roughly 4× memory reduction for ~30% step time.

**The CRPS pairwise term must use the order-statistic identity.** The naive
`|x_i - x_j|` form materialises an `(n, n, B, Q, P)` tensor and will OOM. See
`losses.energy_score`.

## What is deliberately not here

- Data ingestion for the raw CPCB CSVs (site-specific)
- ERA5 / NWP covariate joining — but see §2.3 of the design doc; boundary layer
  height is likely a larger win than any architecture change
- Distributed training scaffolding
- The gradient-boosted-tree baseline, which you should implement, because it is
  strong enough that beating it is the real bar
