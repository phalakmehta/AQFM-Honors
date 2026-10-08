# AQFM — Delhi-Only Implementation Plan (v1)

**Source doc:** "A Spatio-Temporal Foundation Model for Multi-Pollutant Air Quality Forecasting" (design spec, Sept 2026)
**Scope decision:** Single-corpus training on Delhi NCR CPCB data (~38 stations, 2018–2026). National pre-training (Stage 1 in the source doc) is **out of scope for v1** — the data for it isn't available yet.
**Naming note:** don't call this a "foundation model" in write-ups until national pre-training + demonstrated cross-region transfer actually exist. On Delhi-only data it's a rigorous spatio-temporal multi-pollutant forecasting model — describe it as that.

---

## 0. Prerequisites (do before / in parallel with Phase 1)

- [ ] Confirm exact CPCB Delhi export format (per-station CSVs? single merged table? sampling cadence — nominally 15 min per the doc).
- [ ] Source ERA5 reanalysis for the same period/region: boundary layer height (PBLH), 10m winds, precipitation. Not part of CPCB data — needs separate download (Copernicus CDS API).
- [ ] Source the satellite/reanalysis annual-mean PM2.5 field for static per-station covariates (climatological burden, urban–rural gradient).
- [ ] Set up repo skeleton matching the doc's implementation map:
  ```
  aqfm/config.py    aqfm/layers.py    aqfm/model.py
  aqfm/losses.py    aqfm/masking.py   aqfm/physical.py
  aqfm/data.py      aqfm/train.py     tests/
  ```

---

## 1. Reference config (Delhi-sized, NOT the national one)

| Group | Parameter | Value |
|---|---|---|
| Window | context | 512 steps (5.3 days), patch P=16 (4h) → N=32 context patches |
| Window | horizon | 288 steps (72h) → H=18 horizon patches |
| Window | neighbourhood | k=12 nearest stations per sample |
| Encoder | d_model | 384 |
| Encoder | heads | 8 |
| Encoder | d_ff | 1536 |
| Encoder | axial blocks | 4 (=12 attention layers) |
| Encoder | latents/station | 16 (Perceiver bottleneck) |
| Decoder | layers | 4 (self + cross + FFN) |
| Decoder | mixture components K | 3 (log-normal) |
| Training | optimizer | AdamW, lr 3e-4, wd 0.05, grad clip 1.0, cosine schedule + linear warmup |
| Training | CRPS samples | 32 (order-statistic estimator) |
| Training | normalization | masked RevIN, context-only statistics |
| Size | parameters | ≈23M (this is the doc's own Delhi-scale estimate — do not use the ≈150M national config) |

Use `AQFMConfig.small()` for shape/pipeline testing before running the full config.

---

## 2. Build order

### Phase 1 — Preprocessing (`aqfm/data.py`)
- Load CPCB Delhi data into `X ∈ [S,C,T]` (values) + `M ∈ {0,1}^[S,C,T]` (observed mask). The mask is load-bearing for everything downstream — get it right first.
- Log-transform: `x → log(1+x)` on all pollutant channels.
- Normalize per (station, channel) using **context-window-only** statistics (Eq. 7 in source doc). Never compute stats over the full window including the forecast horizon.
- Fallback to identity transform when a (station, channel) has too few observations in context to estimate variance safely.
- **Required test:** perturb the horizon values, assert encoder output on the context is bit-identical. This catches the #1 leakage bug in forecasting pipelines.

### Phase 2 — Physical constraint channels (`aqfm/physical.py`)
- Compute `f_pm = PM2.5/PM10` and `f_nox = NO/NOx` as extra channels, applied via `σ(·)` to bound them in (0,1).
- Only mark a fraction "observed" when both constituents are observed and the denominator is non-trivial — otherwise a missing PM10 silently manufactures a fake PM2.5 fraction.
- **Required test:** assert reconstructed PM2.5 ≤ PM10 always holds post-hoc.

### Phase 3 — Tokenization (`aqfm/model.py`)
- One token per (station, channel, patch): patch length P=16 (4h).
- Token = value+mask projection + channel embedding + station embedding (function of coords, not a lookup table — must generalize to unseen stations) + continuous time embedding.
- Time embedding: log-spaced Fourier periods (3h → 1yr) + explicit daily/weekly/yearly cyclical terms + explicit gap-to-previous-patch term (Eq. 9). The gap term is what lets the model tell "next patch" apart from "next patch after a 10-hour outage."
- Fully-unobserved patches are **excluded from attention**, not zero-filled and attended to.

### Phase 4 — Masking curriculum (`aqfm/masking.py`)
- Three regimes, weighted mixture: Tail 50% (mask last k patches — pure forecasting), Block 35% (station-wide outage, all channels, duration ~ LogNormal fit to your *actual* gap-length distribution), Channel 15% (drop a random channel subset).
- Fit the LogNormal outage parameters from the real run-length distribution of zeros in your mask `M` — don't guess these.

### Phase 5 — Encoder (`aqfm/layers.py`, `aqfm/model.py`)
- Axial factorization: attention over time, then channel, then station, separately (not full O(L²) attention — this is what makes it tractable).
- Add the wind-aware spatial bias (Eq. 14): distance + bearing-relative-to-wind-direction + wind speed, per attention head, added to spatial logits.
- Masked attention: invalid keys get additive −∞ bias; if a row has zero valid keys, leave it unmasked rather than producing NaN.
- Perceiver-style compression to L=16 latents per station before decoding.

### Phase 6 — Decoder (`aqfm/model.py`) — **Design B only**
- Query-based unified decoder. One query per (target channel, lead time, station): `q = target_channel_emb + lead_emb + station_emb + time_emb`.
- Query self-attention (lets e.g. the PM2.5 prediction see the PM10 prediction before either is emitted) → cross-attention to station latents → FFN, ×4 layers.
- Do **not** build Designs A, C, or D for v1. Build Design A later only if you need an ablation baseline for the writeup ("query decoder vs. per-pollutant decoder").

### Phase 7 — Loss (`aqfm/losses.py`)
- Per-query, per-timestep K=3 component log-normal mixture (Eq. 17–18): mixture NLL.
- CRPS via the **order-statistic identity** (Eq. 20) — O(n log n), not the naive pairwise O(n²) tensor, which will exhaust memory at real query counts.
- Total loss = weighted NLL + CRPS + a reconstruction-CRPS term scored on the physically-reconstructed observables (not a hard constraint — score the constraint as a distribution).

### Phase 8 — Training (`aqfm/train.py`) — single-corpus curriculum, not two-corpus
Since there's no national corpus, replace the doc's two-*corpus* schedule with a two-*phase* schedule on the same Delhi data:
1. **Main phase:** full mixed-masking curriculum (Phase 4 weights), on all of the Delhi training window.
2. **Anneal phase:** switch to tail-masking only, at your actual target horizons (1h/6h/24h/72h), lower LR — sharpens pure forecasting right before evaluation.
- Determine step counts from validation-loss plateaus, not from the doc's national-scale step counts (200k/20k) — those were sized for a dataset ~14× larger than Delhi's.
- Gradient checkpointing over encoder blocks if activation memory is tight (~4× memory reduction for ~30% step-time cost, per the doc).

### Phase 9 — Evaluation
- Skill stratified by AQI band, not pooled (a pooled MAE hides December failures behind clean summer months).
- Episode recall/precision for PM2.5 > 250 µg/m³ at 24h and 48h lead — this is the number that actually matters operationally.
- Calibration: PIT histograms, 90%/95% interval reliability, per horizon.
- CRPS by lead time.
- Score imputation and forecasting **separately** — never pool them.
- Baselines to beat: persistence, seasonal-naive, climatology by (station, hour, month), gradient-boosted trees on lagged features. If you don't clearly beat GBM trees at 24h+, the bug is in your setup, not the baseline.
- Hold out 2025–2026 entirely; rolling-origin backtests within the rest.
- **Required test:** assert input mask and target mask never overlap (a second, independent leakage path).

### Phase 10 — Ablations (post-v1, optional, only once the main model beats baselines)
No spatial attention · distance-only bias (drop wind term) · Design A decoder swap · random-only masking (drop tail regime) · point head + MSE (drop distributional output) · no physical reparameterization.

---

## Explicitly out of scope for v1
- National CPCB pre-training (Stage 1 in the source doc) — needs data you don't have yet.
- Decoder Designs A, C, D (except A as a later ablation).
- Asynchronous continuous-time fusion (mTAN / set attention) — the doc itself flags this as a future extension, not part of the base model.
- Cross-region domain-shift evaluation (Delhi vs. Chennai) — requires the national corpus.

## Open items before this can fully run
1. ERA5 covariate sourcing (Phase 0) — blocks Phase 1 fully completing, though tokenization/encoder scaffolding can proceed on CPCB channels alone in the meantime.
2. Exact step-count / LR schedule for the single-corpus curriculum (Phase 8) — tune empirically, doc's numbers don't transfer directly at this data scale.
