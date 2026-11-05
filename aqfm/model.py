"""AQFM: spatio-temporal multi-pollutant forecasting with a query decoder.

One token = one 4-hour patch of one channel at one station. The token sees the
patch values and its observation mask; a fully unobserved patch is excluded
from every attention key set. Station identity comes from coordinates (plus
static covariates), not a lookup table, and time from timestamps, not position
indices. Decoding is query-based: a query is (target channel, station, signed
lead time, absolute time, task, optional known-future met), so any subset of
pollutants at any horizon, and imputation of past gaps, use the same weights.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .config import AQFMConfig, CH, OUTAGE_CHANNELS
from .layers import (AxialBlock, FeedForward, LeadFeatures, MaskedMHA,
                     MaskedRevIN, SpatialBias, TimeFeatures, local_xy_km)

FORECAST, IMPUTE = 0, 1


def station_mlp(cfg):
    return nn.Sequential(nn.Linear(cfg.n_station_in, cfg.d_model), nn.GELU(),
                         nn.Linear(cfg.d_model, cfg.d_model))


def station_features(coords, static):
    """v1 ("delhi" mode): lat/lon minus a hard-coded Delhi centre, x10. Kept so
    v1 checkpoints reproduce exactly. Do not use for other cities."""
    c = coords - torch.tensor([28.6, 77.2], device=coords.device, dtype=coords.dtype)
    return torch.cat([c * 10.0, static], -1)


def density_features(coords, present, near_km=(5.0, 20.0)):
    """Per-station description of how dense its *usable* neighbourhood is in
    this sample. present: (B,S) bool = slot filled and station has any visible
    pollutant data in the context. Returns (B,S,5):
      log1p(nearest present neighbour km)/5, log1p(mean present neighbour km)/5,
      share of slots present within near_km[0] and near_km[1], share present.
    A station with no present neighbour gets nearest = mean = 300 km."""
    B, S, _ = coords.shape
    xy = local_xy_km(coords)
    d = (xy[:, None] - xy[:, :, None]).norm(dim=-1)            # (B,S,S)
    other = ~torch.eye(S, dtype=torch.bool, device=coords.device)
    ok = present[:, None, :] & other                           # key j usable for i
    big = torch.full_like(d, 300.0)
    nn_d = torch.where(ok, d, big).min(-1).values
    cnt = ok.sum(-1)
    mean_d = torch.where(cnt > 0, (d * ok).sum(-1) / cnt.clamp(min=1), big[..., 0])
    denom = max(1, S - 1)
    f = [torch.log1p(nn_d) / 5.0, torch.log1p(mean_d) / 5.0,
         (ok & (d < near_km[0])).sum(-1) / denom, (ok & (d < near_km[1])).sum(-1) / denom,
         cnt / denom]
    return torch.stack([x.to(coords.dtype) for x in f], -1)


def station_inputs(cfg, coords, static, present=None):
    """(B,S,cfg.n_station_in) input of every station MLP, per cfg.coord_mode."""
    if cfg.coord_mode == "delhi":
        parts = [station_features(coords, static)]
    else:
        assert cfg.coord_mode in ("relative", "relative_abs"), cfg.coord_mode
        xy = local_xy_km(coords)
        rel = (xy - xy[:, :1]) / cfg.coord_scale_km              # anchor = station 0 at origin
        parts = [rel, static]
        if cfg.coord_mode == "relative_abs":
            ll = torch.deg2rad(coords)                           # (B,S,2)
            k = (2.0 ** torch.arange(cfg.n_coord_bands, device=coords.device,
                                     dtype=coords.dtype)) * 4.0  # periods ~90 deg .. ~11 deg
            ang = ll.unsqueeze(-1) * k                           # (B,S,2,nb)
            parts.append(torch.cat([ang.sin(), ang.cos()], -1).flatten(-2))
    if cfg.density_features:
        if present is None:
            present = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)
        parts.append(density_features(coords, present))
    return torch.cat(parts, -1)


def hours_since_prev_valid(valid, patch_hours):
    """valid: (..., N) bool. Hours from each patch back to the previous valid
    patch of the same series (patches before the first valid one count from
    just before the window). Lets the model tell "next patch" from "next patch
    after a ten-hour outage"."""
    N = valid.shape[-1]
    idx = torch.arange(N, device=valid.device).expand_as(valid)
    last = torch.where(valid, idx, torch.full_like(idx, -1)).cummax(-1).values
    prev = torch.cat([torch.full_like(last[..., :1], -1), last[..., :-1]], -1)
    return torch.where(prev >= 0, idx - prev, idx + 1).float() * patch_hours


def last_visible(xn, m, step_hours):
    """xn, m: (..., T) normalised values and visibility. Returns the latest
    visible value of each series, the hours from it to the end of the context,
    and whether the series has any visible step at all. The Perceiver pooler
    squeezes the whole context into a few latents, so without this the decoder
    never sees "now" exactly and short leads lose to persistence."""
    T = m.shape[-1]
    idx = torch.arange(T, device=m.device)
    last = torch.where(m > 0, idx, torch.full_like(idx, -1)).max(-1).values
    has = last >= 0
    val = torch.gather(xn, -1, last.clamp(min=0).unsqueeze(-1)).squeeze(-1) * has
    return val, (T - last).float() * step_hours, has


# ---------------------------------------------------------------------------
class PatchTokenizer(nn.Module):
    def __init__(self, cfg: AQFMConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(2 * cfg.patch_len, cfg.d_model)
        self.channel_emb = nn.Embedding(cfg.n_channels, cfg.d_model)
        self.station = station_mlp(cfg)
        self.time = TimeFeatures(cfg.d_model, cfg.n_fourier_bands)
        self.gap_proj = nn.Linear(1, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, xn, m, t_patch, sf):
        """xn, m: (B,S,C,T) normalised values and visibility; t_patch: (B,N)
        patch-centre hours; sf: (B,S,n_station_in) station inputs. Returns
        tokens (B,S,C,N,D) and validity (B,S,C,N)."""
        B, S, C, T = xn.shape
        P = self.cfg.patch_len
        N = T // P
        xp = xn.reshape(B, S, C, N, P)
        mp = m.reshape(B, S, C, N, P)
        valid = mp.sum(-1) > 0
        gap_h = hours_since_prev_valid(valid, self.cfg.patch_hours)

        tok = self.value_proj(torch.cat([xp, mp], -1))
        tok = tok + self.channel_emb.weight[:C].view(1, 1, C, 1, -1)
        tok = tok + self.station(sf).view(B, S, 1, 1, -1)
        tok = tok + self.time(t_patch).view(B, 1, 1, N, -1)
        tok = tok + self.gap_proj(torch.log1p(gap_h).unsqueeze(-1))
        return self.norm(tok), valid


class StationPooler(nn.Module):
    """Perceiver-style bottleneck: each station's (C x N) grid -> L latents."""

    def __init__(self, cfg: AQFMConfig):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(cfg.n_latents_per_station, cfg.d_model) * 0.02)
        self.station = station_mlp(cfg)
        self.attn = MaskedMHA(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.nq, self.nkv, self.nff = (nn.LayerNorm(cfg.d_model) for _ in range(3))
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, tok, valid, sf):
        B, S, C, N, D = tok.shape
        L = self.latents.shape[0]
        # latents carry station identity so the decoder can tell stations apart
        q = self.latents.view(1, 1, L, D) + self.station(sf).unsqueeze(2)
        q = q.reshape(B * S, L, D)
        mem = tok.reshape(B * S, C * N, D)
        z = q + self.attn(self.nq(q), kv=self.nkv(mem), key_valid=valid.reshape(B * S, C * N))
        z = z + self.ff(self.nff(z))
        return z.view(B, S * L, D)


class QueryDecoder(nn.Module):
    """Pre-norm layers of {query self-attention, cross-attention to station
    latents, FFN}. Query self-attention is what lets the PM10 and PM-fraction
    queries of the same station and lead see each other before emitting."""

    def __init__(self, cfg: AQFMConfig):
        super().__init__()
        L = cfg.n_decoder_layers
        self.self_attn = nn.ModuleList(MaskedMHA(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(L))
        self.cross_attn = nn.ModuleList(MaskedMHA(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(L))
        self.ff = nn.ModuleList(FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout) for _ in range(L))
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.d_model) for _ in range(4 * L))
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, q, q_valid, mem, mem_valid=None):
        for i in range(len(self.ff)):
            n = self.norms[4 * i: 4 * i + 4]
            q = q + self.self_attn[i](n[0](q), key_valid=q_valid)
            q = q + self.cross_attn[i](n[1](q), kv=n[2](mem), key_valid=mem_valid)
            q = q + self.ff[i](n[3](q))
        return self.out_norm(q)


class MixtureHead(nn.Module):
    """K-component Gaussian mixture per timestep in normalised transformed
    space (= log-normal mixture in concentration space for log1p channels)."""

    def __init__(self, cfg: AQFMConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.d_model, cfg.patch_len * cfg.n_mixture * 3)

    def forward(self, h):
        B, Q, _ = h.shape
        o = self.proj(h).view(B, Q, self.cfg.patch_len, self.cfg.n_mixture, 3)
        log_sigma = o[..., 2].clamp(self.cfg.min_log_sigma, self.cfg.max_log_sigma)
        return o[..., 0], o[..., 1], log_sigma


# ---------------------------------------------------------------------------
class AQFM(nn.Module):
    def __init__(self, cfg: AQFMConfig):
        super().__init__()
        self.cfg = cfg
        self.revin = MaskedRevIN(cfg.n_channels, cfg.revin_min_obs)
        self.tokenizer = PatchTokenizer(cfg)
        mk = lambda: AxialBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                                cfg.use_spatial_attention, cfg.spatial_lags, cfg.spatial_sink)
        if cfg.share_axial_blocks:          # same weights applied n times (ALBERT)
            blk = mk()
            self.blocks = nn.ModuleList([blk] * cfg.n_axial_blocks)
        else:
            self.blocks = nn.ModuleList(mk() for _ in range(cfg.n_axial_blocks))
        self.spatial_bias = (SpatialBias(cfg.n_heads, cfg.spatial_bias, cfg.wind_length_scale_km,
                                         kernels_km=cfg.dist_kernels_km,
                                         advection=cfg.advection_feature,
                                         n_lags=cfg.spatial_lags, patch_hours=cfg.patch_hours)
                             if cfg.use_spatial_attention and cfg.spatial_bias != "none" else None)
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.pooler = StationPooler(cfg)

        self.q_channel = nn.Embedding(cfg.n_target_channels, cfg.d_model)
        self.q_task = nn.Embedding(2, cfg.d_model)
        self.q_lead = LeadFeatures(cfg.d_model)
        self.q_station = station_mlp(cfg)
        self.q_time = TimeFeatures(cfg.d_model, cfg.n_fourier_bands)
        self.q_met = nn.Linear(cfg.n_met_future, cfg.d_model) if cfg.use_future_met else None
        if cfg.use_last_obs:
            self.q_last = nn.Linear(3, cfg.d_model)     # (value, log1p(age h), has value)
            # per-step weight on the latest value added to every mixture mean; zero
            # at init, so training starts from the model without the anchor
            self.anchor_gate = nn.Linear(cfg.d_model, cfg.patch_len)
            nn.init.zeros_(self.anchor_gate.weight)
            nn.init.zeros_(self.anchor_gate.bias)
        self.q_norm = nn.LayerNorm(cfg.d_model)
        self.decoder = QueryDecoder(cfg)
        self.head = MixtureHead(cfg)
        self.register_buffer("target_index", torch.tensor(cfg.target_index, dtype=torch.long),
                             persistent=False)

    # -- encoder ------------------------------------------------------------
    def station_inputs(self, x_m, coords, static, st_valid=None):
        """Station-MLP inputs (B,S,n_station_in). A station counts as present
        for the density features if its slot is filled and any CPCB pollutant
        is visible in the context."""
        present = None
        if self.cfg.density_features:
            present = x_m[:, :, list(OUTAGE_CHANNELS)].sum((-1, -2)) > 0
            if st_valid is not None:
                present = present & st_valid
        return station_inputs(self.cfg, coords, static, present)

    def encode(self, x, m, t_patch, coords, static, sf=None):
        """x, m: (B,S,C,T_ctx) transformed values (zero where hidden) and
        visibility. Returns station latents (B,S*L,D) and RevIN stats."""
        cfg = self.cfg
        if sf is None:
            sf = self.station_inputs(m, coords, static)
        mean, std = self.revin.stats(x, m)
        tok, valid = self.tokenizer(self.revin.normalize(x, m, mean, std), m, t_patch, sf)

        bias = None
        if self.spatial_bias is not None:
            B, S, C, T = x.shape
            N, P = T // cfg.patch_len, cfg.patch_len
            u, v = (x[:, :, CH[c]].reshape(B, S, N, P) for c in ("U", "V"))
            mw = (m[:, :, CH["U"]] * m[:, :, CH["V"]]).reshape(B, S, N, P)
            cnt = mw.sum(-1).clamp(min=1.0)                    # circular-safe mean via u,v
            bias = self.spatial_bias(coords, (u * mw).sum(-1) / cnt, (v * mw).sum(-1) / cnt)

        for blk in self.blocks:
            if cfg.grad_checkpoint and self.training:
                tok = torch.utils.checkpoint.checkpoint(blk, tok, valid, bias, use_reentrant=False)
            else:
                tok = blk(tok, valid, bias)
        mem = self.pooler(self.enc_norm(tok), valid, sf)
        return mem, (mean, std)

    # -- queries ------------------------------------------------------------
    def embed_queries(self, q, sf, anchor=None):
        """q: dict of (B,Q) tensors slot/station/lead/time/task (+ met (B,Q,M));
        sf: (B,S,n_station_in) station inputs;
        anchor: optional (value, age, has) of each query's series, each (B,Q)."""
        st = self.q_station(sf)                                          # (B,S,D)
        st = torch.gather(st, 1, q["station"].unsqueeze(-1).expand(-1, -1, st.shape[-1]))
        e = (self.q_channel(q["slot"]) + self.q_task(q["task"]) + self.q_lead(q["lead"])
             + st + self.q_time(q["time"]))
        if self.q_met is not None:
            e = e + self.q_met(q["met"])
        if anchor is not None:
            val, age, has = anchor
            e = e + self.q_last(torch.stack([val, torch.log1p(age), has.to(val.dtype)], -1))
        return self.q_norm(e)

    def forward(self, batch, q):
        st_valid = batch.get("st_valid")                                 # (B,S) or None (v1 batches)
        sf = self.station_inputs(batch["m_ctx"], batch["coords"], batch["static"], st_valid)
        mem, (mean, std) = self.encode(batch["x_ctx"], batch["m_ctx"], batch["t_ctx_patch"],
                                       batch["coords"], batch["static"], sf)
        mem_valid = None
        if st_valid is not None:                                         # padded slots: no memory
            L = self.cfg.n_latents_per_station
            mem_valid = st_valid.unsqueeze(-1).expand(-1, -1, L).reshape(st_valid.shape[0], -1)
        anchor = None
        if self.cfg.use_last_obs:
            xn = self.revin.normalize(batch["x_ctx"], batch["m_ctx"], mean, std)
            step_h = self.cfg.patch_hours / self.cfg.patch_len
            anchor = tuple(self.per_query(a, q) for a in last_visible(xn, batch["m_ctx"], step_h))
        h = self.decoder(self.embed_queries(q, sf, anchor), q["valid"], mem, mem_valid)
        logits, mu, log_sigma = self.head(h)
        if anchor is not None:          # same normalised space as the targets (query_stats)
            mu = mu + (self.anchor_gate(h) * anchor[0].unsqueeze(-1)).unsqueeze(-1)
        return {"logits": logits, "mu": mu, "log_sigma": log_sigma, "mean": mean, "std": std}

    # -- (de)normalisation of targets -------------------------------------
    def per_query(self, t, q):
        """Pick each query's (station, input channel) entry of t (B,S,C): (B,Q)."""
        B, S, C = t.shape
        idx = q["station"] * C + self.target_index[q["slot"]]
        return torch.gather(t.reshape(B, S * C), 1, idx)

    def query_stats(self, mean, std, q):
        """RevIN stats of each query's (station, input channel): (B,Q,1)."""
        return self.per_query(mean[..., 0], q).unsqueeze(-1), self.per_query(std[..., 0], q).unsqueeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
