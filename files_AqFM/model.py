"""AQFM: a spatio-temporal foundation model for multi-pollutant forecasting.

Design in one paragraph
-----------------------
One token = one patch of one channel at one station. The embedding sees the
patch values *and* the observation mask, so a partially observed patch is not
confused with a clean one, and a fully unobserved patch is excluded from every
attention key set (the sparse equivalent of dropping the token). Station
identity comes from coordinates, not a lookup table, so an unseen station is a
valid input. Time comes from timestamps, not position indices, so gaps are
visible. Decoding is query-based: a query is (target channel, station, lead
patch), so any subset of pollutants at any subset of horizons can be requested
at inference without retraining.
"""

from typing import Optional

import torch
import torch.nn as nn

from .config import AQFMConfig
from .layers import (AxialBlock, FeedForward, FourierTime, MaskedMHA,
                     MaskedRevIN, WindAwareBias)


# ---------------------------------------------------------------------------
class PatchTokenizer(nn.Module):
    """(B,S,C,T) values + mask -> (B,S,C,N,D) tokens."""

    def __init__(self, cfg: AQFMConfig):
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(2 * cfg.patch_len, cfg.d_model)
        self.channel_emb = nn.Embedding(cfg.n_channels, cfg.d_model)
        self.station_proj = nn.Sequential(              # coords -> embedding
            nn.Linear(2 + cfg.n_static, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model))
        self.time_enc = FourierTime(cfg.d_model, cfg.n_fourier_bands)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x, m, t_hours, coords, static):
        """x, m: (B,S,C,T); t_hours: (B,T); coords: (B,S,2); static: (B,S,Ds)."""
        B, S, C, T = x.shape
        P = self.cfg.patch_len
        N = T // P
        xp = x[..., :N * P].reshape(B, S, C, N, P)
        mp = m[..., :N * P].reshape(B, S, C, N, P)

        tok = self.value_proj(torch.cat([xp, mp], dim=-1))          # (B,S,C,N,D)
        tok = tok + self.channel_emb.weight[:C].view(1, 1, C, 1, -1)
        tok = tok + self.station_proj(
            torch.cat([coords, static], -1)).view(B, S, 1, 1, -1)

        t_patch = t_hours[..., :N * P].reshape(B, N, P).mean(-1)    # (B,N)
        tok = tok + self.time_enc(t_patch).view(B, 1, 1, N, -1)

        # A patch with no observation at all carries no information: exclude it
        # from every key set rather than letting the model attend to zeros.
        valid = mp.sum(-1) > 0                                      # (B,S,C,N)
        return self.norm(tok), valid, t_patch


# ---------------------------------------------------------------------------
class StationPooler(nn.Module):
    """Per-station Perceiver-style bottleneck.

    Cross-attending 1.7k decoder queries against ~6.5k encoder tokens is
    wasteful. Compressing each station's (C x N) grid into L latents keeps the
    spatial structure the wind-aware bias operates on while cutting the decoder
    cross-attention cost by ~30x.
    """

    def __init__(self, cfg: AQFMConfig):
        super().__init__()
        self.latents = nn.Parameter(
            torch.randn(cfg.n_latents_per_station, cfg.d_model) * 0.02)
        self.attn = MaskedMHA(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.n1, self.n2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, tok, valid):
        """tok: (B,S,C,N,D) -> (B, S*L, D), (B, S*L) valid."""
        B, S, C, N, D = tok.shape
        L = self.latents.shape[0]
        mem = tok.reshape(B * S, C * N, D)
        key_valid = valid.reshape(B * S, C * N)
        q = self.latents.unsqueeze(0).expand(B * S, L, D)
        z = q + self.attn(self.n1(q), kv=self.n2(mem), key_valid=key_valid)
        z = z + self.ff(z)
        return z.view(B, S * L, D), torch.ones(B, S * L, dtype=torch.bool,
                                               device=tok.device)


# ---------------------------------------------------------------------------
class QueryDecoder(nn.Module):
    """Queries self-attend (cross-pollutant coherence) then read the memory.

    Option A in the design -- one decoder per pollutant -- gives specialisation
    but produces physically incoherent joint output and scales linearly in
    pollutant count. Here specialisation lives in the query embedding, and the
    query-to-query self-attention lets the PM2.5 prediction see the PM10
    prediction before either is emitted.
    """

    def __init__(self, cfg: AQFMConfig):
        super().__init__()
        self.self_attn = nn.ModuleList(
            [MaskedMHA(cfg.d_model, cfg.n_heads, cfg.dropout)
             for _ in range(cfg.n_decoder_layers)])
        self.cross_attn = nn.ModuleList(
            [MaskedMHA(cfg.d_model, cfg.n_heads, cfg.dropout)
             for _ in range(cfg.n_decoder_layers)])
        self.ff = nn.ModuleList(
            [FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
             for _ in range(cfg.n_decoder_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(cfg.d_model) for _ in range(3 * cfg.n_decoder_layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, q, mem, mem_valid):
        for i in range(len(self.self_attn)):
            a, b, c = self.norms[3 * i], self.norms[3 * i + 1], self.norms[3 * i + 2]
            q = q + self.self_attn[i](a(q))
            q = q + self.cross_attn[i](b(q), kv=mem, key_valid=mem_valid)
            q = q + self.ff[i](c(q))
        return self.out_norm(q)


# ---------------------------------------------------------------------------
class MixtureHead(nn.Module):
    """Per query, emits a K-component Gaussian mixture for each of P timesteps.

    Gaussian in normalised log1p space == log-normal in concentration space,
    which matches the right-skewed, heavy-tailed distribution of PM and NOx.
    Point heads trained on MSE systematically under-predict exactly the winter
    episodes the forecast exists to warn about.
    """

    def __init__(self, cfg: AQFMConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.d_model, cfg.patch_len * cfg.n_mixture * 3)

    def forward(self, h):
        """h: (B, Q, D) -> logits/mu/log_sigma each (B, Q, P, K)."""
        B, Q, _ = h.shape
        P, K = self.cfg.patch_len, self.cfg.n_mixture
        o = self.proj(h).view(B, Q, P, K, 3)
        logits, mu, log_sigma = o[..., 0], o[..., 1], o[..., 2]
        log_sigma = torch.clamp(log_sigma, self.cfg.min_log_sigma,
                                self.cfg.max_log_sigma)
        return logits, mu, log_sigma


# ---------------------------------------------------------------------------
class AQFM(nn.Module):
    def __init__(self, cfg: AQFMConfig):
        super().__init__()
        self.cfg = cfg
        self.revin = MaskedRevIN(cfg.n_channels, cfg.revin_min_obs)
        self.tokenizer = PatchTokenizer(cfg)
        self.blocks = nn.ModuleList(
            [AxialBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
             for _ in range(cfg.n_axial_blocks)])
        self.wind_bias = WindAwareBias(cfg.n_heads)
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.pooler = StationPooler(cfg)

        self.tgt_channel_emb = nn.Embedding(cfg.n_target_channels, cfg.d_model)
        self.lead_emb = nn.Embedding(cfg.n_horizon_patches, cfg.d_model)
        self.q_station = nn.Sequential(
            nn.Linear(2 + cfg.n_static, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model))
        self.q_time = FourierTime(cfg.d_model, cfg.n_fourier_bands)
        self.q_norm = nn.LayerNorm(cfg.d_model)

        self.decoder = QueryDecoder(cfg)
        self.head = MixtureHead(cfg)

        # Target channels are a subset of the (possibly augmented) input
        # channels, so their RevIN statistics are always available.
        tgt = torch.tensor(cfg.target_index, dtype=torch.long)
        assert tgt.numel() == cfg.n_target_channels
        self.register_buffer("target_index", tgt)

    # -- encoder ------------------------------------------------------------
    def encode(self, x, m, t_hours, coords, static):
        cfg = self.cfg
        mean, std = self.revin.stats(x, m)
        xn = self.revin.normalize(x, m, mean, std)

        tok, valid, t_patch = self.tokenizer(xn, m, t_hours, coords, static)

        # wind features are read from the *raw* met channels, patch-averaged
        P = cfg.patch_len
        B, S, C, T = x.shape
        N = T // P
        ws = x[:, :, cfg.idx_wind_speed, :N * P].reshape(B, S, N, P).mean(-1)
        wd = x[:, :, cfg.idx_wind_dir, :N * P].reshape(B, S, N, P).mean(-1)
        bias = self.wind_bias(coords, ws, wd)                # (B,N,h,S,S)

        for blk in self.blocks:
            if cfg.grad_checkpoint and self.training:
                tok = torch.utils.checkpoint.checkpoint(
                    blk, tok, valid, bias, use_reentrant=False)
            else:
                tok = blk(tok, valid, spatial_bias=bias)
        return self.enc_norm(tok), valid, (mean, std)

    # -- queries ------------------------------------------------------------
    def build_queries(self, q_channel, q_station, q_lead, q_time, coords, static):
        """All args (B, Q) index tensors except q_time (B, Q) hours, coords/static
        (B, S, ...). Returns (B, Q, D)."""
        B, Q = q_channel.shape
        st_feat = torch.cat([coords, static], -1)            # (B,S,2+Ds)
        st = self.q_station(st_feat)                         # (B,S,D)
        st = torch.gather(st, 1, q_station.unsqueeze(-1).expand(B, Q, st.shape[-1]))
        q = (self.tgt_channel_emb(q_channel) + self.lead_emb(q_lead)
             + st + self.q_time(q_time))
        return self.q_norm(q)

    # -- full pass ----------------------------------------------------------
    def forward(self, batch, q_channel, q_station, q_lead, q_time):
        tok, valid, (mean, std) = self.encode(
            batch["x"], batch["m"], batch["t"], batch["coords"], batch["static"])
        mem, mem_valid = self.pooler(tok, valid)
        q = self.build_queries(q_channel, q_station, q_lead, q_time,
                               batch["coords"], batch["static"])
        h = self.decoder(q, mem, mem_valid)
        logits, mu, log_sigma = self.head(h)
        return {"logits": logits, "mu": mu, "log_sigma": log_sigma,
                "mean": mean, "std": std}

    # -- convenience --------------------------------------------------------
    def denorm(self, z, mean, std, q_channel, q_station):
        """Map normalised predictions back to log1p concentration space.
        z: (B,Q,P,...); mean/std: (B,S,C,1)."""
        B, Q = q_channel.shape
        flat_mean = mean.squeeze(-1)                        # (B,S,C)
        flat_std = std.squeeze(-1)
        c_in = self.target_index[q_channel]                 # (B,Q) input-channel ids
        idx = (q_station * flat_mean.shape[2] + c_in)       # (B,Q)
        mu_q = torch.gather(flat_mean.reshape(B, -1), 1, idx)
        sd_q = torch.gather(flat_std.reshape(B, -1), 1, idx)
        shape = [B, Q] + [1] * (z.dim() - 2)
        return z * sd_q.view(shape) + mu_q.view(shape)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
