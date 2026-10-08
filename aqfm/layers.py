"""Building blocks: masked normalisation, time/lead features, masked attention,
wind-aware spatial bias, axial encoder block."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ---------------------------------------------------------------------------
class MaskedRevIN(nn.Module):
    """Per-(sample, station, channel) normalisation from the *visible context
    only*. Statistics never touch the horizon or hidden cells, so the scale of
    the targets cannot leak into the input.

    When a series has fewer than `min_obs` visible steps (a station outage over
    the whole context), its statistics fall back to global per-channel
    statistics computed on the training split, rather than to the identity.
    The std is floored at `min_rel_std` x the global std: a stuck sensor
    (constant context) would otherwise blow normalised targets up by ~1000x.
    """

    def __init__(self, n_channels: int, min_obs: int = 16, eps: float = 1e-3,
                 min_rel_std: float = 0.1):
        super().__init__()
        self.min_obs, self.eps, self.min_rel_std = min_obs, eps, min_rel_std
        self.register_buffer("global_mean", torch.zeros(n_channels))
        self.register_buffer("global_std", torch.ones(n_channels))

    def set_global_stats(self, mean, std):
        self.global_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.global_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp(min=self.eps))

    def stats(self, x, m):
        """x, m: (B,S,C,T) -> mean, std each (B,S,C,1)."""
        cnt = m.sum(-1, keepdim=True)
        mean = (x * m).sum(-1, keepdim=True) / cnt.clamp(min=1.0)
        var = (((x - mean) * m) ** 2).sum(-1, keepdim=True) / cnt.clamp(min=1.0)
        std = (var + self.eps ** 2).sqrt().clamp(min=self.eps)
        ok = cnt >= self.min_obs
        gm = self.global_mean.view(1, 1, -1, 1).expand_as(mean)
        gs = self.global_std.view(1, 1, -1, 1).expand_as(std)
        std = torch.maximum(std, self.min_rel_std * gs)
        return torch.where(ok, mean, gm), torch.where(ok, std, gs)

    @staticmethod
    def normalize(x, m, mean, std):
        return (x - mean) / std * m       # unobserved entries are exactly zero


# ---------------------------------------------------------------------------
class TimeFeatures(nn.Module):
    """Absolute time -> embedding: log-spaced Fourier periods (3 h .. 1 yr) plus
    explicit daily, weekly and yearly cycles. Computed in float64 because
    hours-since-2018 times a 3-hour frequency loses precision in float32."""

    def __init__(self, d_model: int, n_bands: int = 6):
        super().__init__()
        periods = torch.logspace(math.log10(3.0), math.log10(8766.0), n_bands,
                                 dtype=torch.float64)
        self.register_buffer("freqs", 2 * math.pi / periods, persistent=False)
        self.register_buffer("cycles", torch.tensor([24.0, 168.0, 8766.0], dtype=torch.float64),
                             persistent=False)
        self.proj = nn.Linear(2 * n_bands + 6, d_model)

    def forward(self, t_hours):
        """t_hours: (...) float64 hours -> (..., d_model)."""
        t = t_hours.double().unsqueeze(-1)
        ang = torch.cat([t * self.freqs, t * (2 * math.pi / self.cycles)], -1)
        feats = torch.cat([ang.sin(), ang.cos()], -1).to(self.proj.weight.dtype)
        return self.proj(feats)


class LeadFeatures(nn.Module):
    """Continuous signed lead time (hours from the forecast origin; negative for
    imputation queries in the context). Continuous encoding means horizons not
    seen in training are still valid queries."""

    def __init__(self, d_model: int, max_hours: float = 128.0, n_bands: int = 8):
        super().__init__()
        periods = torch.logspace(math.log10(2.0), math.log10(4 * max_hours), n_bands)
        self.register_buffer("freqs", 2 * math.pi / periods, persistent=False)
        self.scale = max_hours
        self.mlp = nn.Sequential(nn.Linear(2 * n_bands + 1, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))

    def forward(self, lead_hours):
        t = lead_hours.float().unsqueeze(-1)
        ang = t * self.freqs
        return self.mlp(torch.cat([ang.sin(), ang.cos(), t / self.scale], -1))


# ---------------------------------------------------------------------------
class MaskedMHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv_q = nn.Linear(d_model, d_model)
        self.qkv_kv = nn.Linear(d_model, 2 * d_model)
        self.o = nn.Linear(d_model, d_model)
        self.p = dropout

    def forward(self, q, kv=None, key_valid=None, bias=None):
        """q: (B,Lq,D); kv: (B,Lk,D); key_valid: (B,Lk) bool;
        bias: additive (B or 1, h or 1, Lq, Lk)."""
        kv = q if kv is None else kv
        B, Lq, D = q.shape
        Lk = kv.shape[1]
        Q = self.qkv_q(q).view(B, Lq, self.h, self.dk).transpose(1, 2)
        K, V = self.qkv_kv(kv).view(B, Lk, 2, self.h, self.dk).permute(2, 0, 3, 1, 4)

        mask = None
        if bias is not None:
            mask = bias.to(Q.dtype)
        if key_valid is not None:
            # A row whose keys are all invalid is left unmasked rather than
            # producing NaN; its content is junk and the mask channel says so.
            ok = key_valid | ~key_valid.any(-1, keepdim=True)
            neg = torch.zeros(B, 1, 1, Lk, dtype=Q.dtype, device=Q.device)
            neg = neg.masked_fill(~ok[:, None, None, :], float("-inf"))
            mask = neg if mask is None else mask + neg
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, Lq, D))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
def local_xy_km(coords):
    """coords (B,S,2) lat/lon degrees -> (B,S,2) east/north km in a local
    equirectangular frame (error < 0.1% over Delhi's ~60 km extent)."""
    lat, lon = torch.deg2rad(coords[..., 0]), torch.deg2rad(coords[..., 1])
    lat0 = lat.mean(1, keepdim=True)
    east = 6371.0 * (lon - lon.mean(1, keepdim=True)) * torch.cos(lat0)
    north = 6371.0 * (lat - lat0)
    return torch.stack([east, north], -1)


class SpatialBias(nn.Module):
    """Additive per-head bias for spatial attention.

    mode="wind":     b_ij = MLP(log(1+d), cos, sin of bearing(i->j) vs the
                     wind's FROM direction at i, wind speed, cos*speed, exp(-d/l))
                     A station 8 km upwind gets a different bias from one 8 km
                     downwind; a distance-only graph cannot express that.
    mode="distance": the same MLP on distance features only (ablation).
    """

    def __init__(self, n_heads, mode="wind", length_scale_km=25.0, hidden=32):
        super().__init__()
        assert mode in ("wind", "distance")
        self.mode, self.ls = mode, length_scale_km
        n_in = 6 if mode == "wind" else 2
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_heads))

    def forward(self, coords, u=None, v=None):
        """coords (B,S,2); u, v: (B,S,N) patch-mean wind at each station.
        Returns (B, N, h, S_query, S_key)."""
        if self.mode == "distance":
            dist = pair_distance_km(coords)
            b = self.mlp(torch.stack([torch.log1p(dist), torch.exp(-dist / self.ls)], -1))
            N = 1 if u is None else u.shape[-1]
            return b.permute(0, 3, 1, 2).unsqueeze(1).expand(-1, N, -1, -1, -1)

        dist, cos, sin, ws = wind_geometry(coords, u, v)
        N = u.shape[-1]
        d = dist.unsqueeze(-1).expand(-1, -1, -1, N)
        feats = torch.stack([torch.log1p(d), cos, sin, ws, cos * ws, torch.exp(-d / self.ls)], -1)
        return self.mlp(feats).permute(0, 3, 4, 1, 2)         # (B,N,h,Sq,Sk)


def pair_distance_km(coords):
    xy = local_xy_km(coords)
    return (xy[:, None, :, :] - xy[:, :, None, :]).norm(dim=-1)   # (B,Sq,Sk)


def wind_geometry(coords, u, v):
    """Returns dist (B,Sq,Sk) km and, per patch, cos/sin of the angle between
    the bearing query->key and the direction the wind comes FROM at the query
    station, plus wind speed, each (B,Sq,Sk,N). cos = 1 means the key station
    is directly upwind. Calm wind gives cos = sin = 0."""
    xy = local_xy_km(coords)
    dxy = xy[:, None, :, :] - xy[:, :, None, :]               # (B,Sq,Sk,2) key - query
    dist = dxy.norm(dim=-1)
    unit = torch.where(dist.unsqueeze(-1) > 1e-6,
                       dxy / dist.clamp(min=1e-6).unsqueeze(-1), torch.zeros_like(dxy))
    ws = torch.sqrt(u ** 2 + v ** 2)                          # (B,S,N)
    fe = torch.where(ws > 1e-6, -u / ws.clamp(min=1e-6), torch.zeros_like(u))
    fn = torch.where(ws > 1e-6, -v / ws.clamp(min=1e-6), torch.zeros_like(v))
    ue, un = unit[..., 0].unsqueeze(-1), unit[..., 1].unsqueeze(-1)   # (B,Sq,Sk,1)
    fe, fn, wsq = fe.unsqueeze(2), fn.unsqueeze(2), ws.unsqueeze(2)   # (B,Sq,1,N)
    cos = ue * fe + un * fn
    sin = ue * fn - un * fe
    return dist, cos, sin, wsq.expand_as(cos)


# ---------------------------------------------------------------------------
class AxialBlock(nn.Module):
    """Pre-norm attention over time, then channel, then station, then FFN.
    Cost O(SCN(S+C+N)) instead of O((SCN)^2)."""

    def __init__(self, d_model, n_heads, d_ff, dropout, use_spatial=True):
        super().__init__()
        self.axes = ("time", "var", "space") if use_spatial else ("time", "var")
        self.attn = nn.ModuleDict({a: MaskedMHA(d_model, n_heads, dropout) for a in self.axes})
        self.norm = nn.ModuleDict({a: nn.LayerNorm(d_model) for a in self.axes})
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _fold(x, valid, axis):
        B, S, C, N, D = x.shape
        if axis == "time":
            return x.reshape(B * S * C, N, D), valid.reshape(B * S * C, N)
        if axis == "var":
            return (x.permute(0, 1, 3, 2, 4).reshape(B * S * N, C, D),
                    valid.permute(0, 1, 3, 2).reshape(B * S * N, C))
        return (x.permute(0, 2, 3, 1, 4).reshape(B * C * N, S, D),
                valid.permute(0, 2, 3, 1).reshape(B * C * N, S))

    @staticmethod
    def _unfold(y, axis, shape):
        B, S, C, N, D = shape
        if axis == "time":
            return y.view(B, S, C, N, D)
        if axis == "var":
            return y.view(B, S, N, C, D).permute(0, 1, 3, 2, 4)
        return y.view(B, C, N, S, D).permute(0, 3, 1, 2, 4)

    def forward(self, x, valid, spatial_bias=None):
        shape = x.shape
        B, S, C, N, D = shape
        for axis in self.axes:
            hf, vf = self._fold(self.norm[axis](x), valid, axis)
            bias = None
            if axis == "space" and spatial_bias is not None:
                nh = spatial_bias.shape[2]                    # (B,N,h,S,S) -> (B*C*N,h,S,S)
                bias = spatial_bias.unsqueeze(1).expand(B, C, N, nh, S, S).reshape(B * C * N, nh, S, S)
            out = self.attn[axis](hf, key_valid=vf, bias=bias)
            x = x + self.drop(self._unfold(out, axis, shape))
        return x + self.drop(self.ff(self.ff_norm(x)))
