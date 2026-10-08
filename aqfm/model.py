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

from .config import AQFMConfig, CH
from .layers import (AxialBlock, FeedForward, LeadFeatures, MaskedMHA,
                     MaskedRevIN, SpatialBias, TimeFeatures)

FORECAST, IMPUTE = 0, 1


def station_mlp(cfg):
    return nn.Sequential(nn.Linear(2 + cfg.n_static, cfg.d_model), nn.GELU(),
                         nn.Linear(cfg.d_model, cfg.d_model))


def station_features(coords, static):
    """Centred coordinates so the MLP sees O(1) inputs; absolute lat/lon is
    ~28/77 degrees and would otherwise dominate the first layer."""
    c = coords - torch.tensor([28.6, 77.2], device=coords.device, dtype=coords.dtype)
    return torch.cat([c * 10.0, static], -1)


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

    def forward(self, xn, m, t_patch, coords, static):
        """xn, m: (B,S,C,T) normalised values and visibility; t_patch: (B,N)
        patch-centre hours. Returns tokens (B,S,C,N,D) and validity (B,S,C,N)."""
        B, S, C, T = xn.shape
        P = self.cfg.patch_len
        N = T // P
        xp = xn.reshape(B, S, C, N, P)
        mp = m.reshape(B, S, C, N, P)
        valid = mp.sum(-1) > 0
        gap_h = hours_since_prev_valid(valid, self.cfg.patch_hours)

        tok = self.value_proj(torch.cat([xp, mp], -1))
        tok = tok + self.channel_emb.weight[:C].view(1, 1, C, 1, -1)
        tok = tok + self.station(station_features(coords, static)).view(B, S, 1, 1, -1)
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

    def forward(self, tok, valid, coords, static):
        B, S, C, N, D = tok.shape
        L = self.latents.shape[0]
        # latents carry station identity so the decoder can tell stations apart
        q = self.latents.view(1, 1, L, D) + self.station(station_features(coords, static)).unsqueeze(2)
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

    def forward(self, q, q_valid, mem):
        for i in range(len(self.ff)):
            n = self.norms[4 * i: 4 * i + 4]
            q = q + self.self_attn[i](n[0](q), key_valid=q_valid)
            q = q + self.cross_attn[i](n[1](q), kv=n[2](mem))
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
        self.blocks = nn.ModuleList(
            AxialBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.use_spatial_attention)
            for _ in range(cfg.n_axial_blocks))
        self.spatial_bias = (SpatialBias(cfg.n_heads, cfg.spatial_bias, cfg.wind_length_scale_km)
                             if cfg.use_spatial_attention and cfg.spatial_bias != "none" else None)
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.pooler = StationPooler(cfg)

        self.q_channel = nn.Embedding(cfg.n_target_channels, cfg.d_model)
        self.q_task = nn.Embedding(2, cfg.d_model)
        self.q_lead = LeadFeatures(cfg.d_model)
        self.q_station = station_mlp(cfg)
        self.q_time = TimeFeatures(cfg.d_model, cfg.n_fourier_bands)
        self.q_met = nn.Linear(cfg.n_met_future, cfg.d_model) if cfg.use_future_met else None
        self.q_norm = nn.LayerNorm(cfg.d_model)
        self.decoder = QueryDecoder(cfg)
        self.head = MixtureHead(cfg)
        self.register_buffer("target_index", torch.tensor(cfg.target_index, dtype=torch.long),
                             persistent=False)

    # -- encoder ------------------------------------------------------------
    def encode(self, x, m, t_patch, coords, static):
        """x, m: (B,S,C,T_ctx) transformed values (zero where hidden) and
        visibility. Returns station latents (B,S*L,D) and RevIN stats."""
        cfg = self.cfg
        mean, std = self.revin.stats(x, m)
        tok, valid = self.tokenizer(self.revin.normalize(x, m, mean, std), m, t_patch, coords, static)

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
        mem = self.pooler(self.enc_norm(tok), valid, coords, static)
        return mem, (mean, std)

    # -- queries ------------------------------------------------------------
    def embed_queries(self, q, coords, static):
        """q: dict of (B,Q) tensors slot/station/lead/time/task (+ met (B,Q,M))."""
        st = self.q_station(station_features(coords, static))            # (B,S,D)
        st = torch.gather(st, 1, q["station"].unsqueeze(-1).expand(-1, -1, st.shape[-1]))
        e = (self.q_channel(q["slot"]) + self.q_task(q["task"]) + self.q_lead(q["lead"])
             + st + self.q_time(q["time"]))
        if self.q_met is not None:
            e = e + self.q_met(q["met"])
        return self.q_norm(e)

    def forward(self, batch, q):
        mem, (mean, std) = self.encode(batch["x_ctx"], batch["m_ctx"], batch["t_ctx_patch"],
                                       batch["coords"], batch["static"])
        h = self.decoder(self.embed_queries(q, batch["coords"], batch["static"]), q["valid"], mem)
        logits, mu, log_sigma = self.head(h)
        return {"logits": logits, "mu": mu, "log_sigma": log_sigma, "mean": mean, "std": std}

    # -- (de)normalisation of targets -------------------------------------
    def query_stats(self, mean, std, q):
        """RevIN stats of each query's (station, input channel): (B,Q,1)."""
        B, S, C, _ = mean.shape
        idx = q["station"] * C + self.target_index[q["slot"]]
        mu = torch.gather(mean.reshape(B, S * C), 1, idx).unsqueeze(-1)
        sd = torch.gather(std.reshape(B, S * C), 1, idx).unsqueeze(-1)
        return mu, sd

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
