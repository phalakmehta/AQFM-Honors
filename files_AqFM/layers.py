"""Building blocks: masked normalisation, time features, masked axial attention."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


# ---------------------------------------------------------------------------
# Instance normalisation that ignores missing observations
# ---------------------------------------------------------------------------
class MaskedRevIN(nn.Module):
    """Per-(station, channel) normalisation computed from *visible context only*.

    Statistics must never be computed over the forecast target, otherwise the
    scale of the future leaks into the input. `forward` therefore takes the
    context slice explicitly and returns the statistics for later reuse.
    """

    def __init__(self, n_channels: int, min_obs: int = 16, eps: float = 1e-5,
                 affine: bool = True):
        super().__init__()
        self.min_obs, self.eps = min_obs, eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))

    def stats(self, x: torch.Tensor, m: torch.Tensor):
        """x, m: (B, S, C, T) -> mean, std each (B, S, C, 1)."""
        cnt = m.sum(-1, keepdim=True)
        safe = cnt.clamp(min=1.0)
        mean = (x * m).sum(-1, keepdim=True) / safe
        var = (((x - mean) * m) ** 2).sum(-1, keepdim=True) / safe
        std = (var + self.eps).sqrt()
        ok = cnt >= self.min_obs
        mean = torch.where(ok, mean, torch.zeros_like(mean))
        std = torch.where(ok, std, torch.ones_like(std))
        return mean, std

    def normalize(self, x, m, mean, std):
        z = (x - mean) / std
        if self.affine:
            w = self.weight.view(1, 1, -1, 1)
            b = self.bias.view(1, 1, -1, 1)
            z = z * w + b
        return z * m                      # zero-fill the unobserved entries

    def denormalize(self, z, mean, std, channel_index=None):
        if self.affine:
            w = self.weight if channel_index is None else self.weight[channel_index]
            b = self.bias if channel_index is None else self.bias[channel_index]
            shape = [1] * (z.dim() - 1) + [w.numel()]
            z = (z - b.view(shape)) / w.view(shape)
        return z * std + mean


# ---------------------------------------------------------------------------
# Continuous time features -- gaps must be visible to the model
# ---------------------------------------------------------------------------
class FourierTime(nn.Module):
    """Maps absolute timestamps (hours since epoch) to a continuous embedding.

    Crucially this is a function of the *timestamp*, not of the position index,
    so a 10-hour outage between two adjacent patches is represented as a
    10-hour jump rather than as a unit step.
    """

    def __init__(self, d_model: int, n_bands: int = 6):
        super().__init__()
        self.n_bands = n_bands
        # log-spaced periods from 3h to ~1 year, in hours
        periods = torch.logspace(math.log10(3.0), math.log10(8760.0), n_bands)
        self.register_buffer("freqs", 2 * math.pi / periods)
        self.proj = nn.Linear(2 * n_bands + 7, d_model)

    def features(self, t_hours: torch.Tensor) -> torch.Tensor:
        """t_hours: (..., L) -> (..., L, F)."""
        ang = t_hours.unsqueeze(-1) * self.freqs
        feats = [ang.sin(), ang.cos()]
        tod = (t_hours % 24.0) / 24.0
        dow = (t_hours % 168.0) / 168.0
        doy = (t_hours % 8760.0) / 8760.0
        for u in (tod, dow, doy):
            feats += [torch.sin(2 * math.pi * u).unsqueeze(-1),
                      torch.cos(2 * math.pi * u).unsqueeze(-1)]
        gap = torch.zeros_like(t_hours)
        gap[..., 1:] = (t_hours[..., 1:] - t_hours[..., :-1]).clamp(min=0)
        feats.append(torch.log1p(gap).unsqueeze(-1))
        return torch.cat(feats, dim=-1)

    def forward(self, t_hours: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(t_hours))


# ---------------------------------------------------------------------------
# Attention with key padding masks and optional additive bias
# ---------------------------------------------------------------------------
class MaskedMHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.p = dropout

    def _split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, q, kv=None, key_valid=None, bias=None):
        """q: (B, Lq, D); kv: (B, Lk, D); key_valid: (B, Lk) bool;
        bias: (B, h, Lq, Lk) or (B, 1, Lq, Lk) additive."""
        kv = q if kv is None else kv
        Q, K, V = self._split(self.q(q)), self._split(self.k(kv)), self._split(self.v(kv))

        attn_mask = None
        if key_valid is not None:
            # If every key is invalid, unmask the row: the mask channel already
            # tells the model the content is junk, and this avoids NaN softmax.
            kv_ok = key_valid | (~key_valid).all(dim=-1, keepdim=True)
            attn_mask = torch.zeros(
                kv_ok.shape[0], 1, 1, kv_ok.shape[1], device=q.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~kv_ok[:, None, None, :], NEG_INF)
        if bias is not None:
            attn_mask = bias if attn_mask is None else attn_mask + bias

        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, dropout_p=self.p if self.training else 0.0)
        B, h, L, dk = out.shape
        return self.o(out.transpose(1, 2).reshape(B, L, h * dk))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Wind-aware spatial attention bias
# ---------------------------------------------------------------------------
class WindAwareBias(nn.Module):
    """Learned additive attention bias b_ij = MLP(dist, bearing vs wind, speed).

    A station 8 km upwind is far more informative than one 8 km downwind. A
    distance-only graph cannot express that asymmetry; this can.
    """

    def __init__(self, n_heads: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, n_heads))

    @staticmethod
    def _geo(coords):
        """coords: (B, S, 2) lat/lon degrees -> dist_km (B,S,S), bearing rad (B,S,S)."""
        lat = torch.deg2rad(coords[..., 0])
        lon = torch.deg2rad(coords[..., 1])
        latm = lat.mean(dim=1, keepdim=True)
        x = 6371.0 * (lon - lon.mean(dim=1, keepdim=True)) * torch.cos(latm)
        y = 6371.0 * (lat - lat.mean(dim=1, keepdim=True))
        dx = x[:, None, :] - x[:, :, None]          # (B, S_query, S_key)
        dy = y[:, None, :] - y[:, :, None]
        dist = (dx ** 2 + dy ** 2).sqrt()
        bearing = torch.atan2(dx, dy)               # 0 = key is due north of query
        return dist, bearing

    def forward(self, coords, wind_speed, wind_dir_deg):
        """coords: (B,S,2); wind_speed, wind_dir_deg: (B,S,N) at the query station.
        Returns (B, N, n_heads, S, S)."""
        dist, bearing = self._geo(coords)                     # (B,S,S)
        wd = torch.deg2rad(wind_dir_deg)                      # (B,S,N)
        # meteorological convention: WD is the direction the wind comes *from*
        rel = bearing.unsqueeze(-1) - wd.unsqueeze(2)         # (B,S,S,N)
        ws = wind_speed.unsqueeze(2).expand_as(rel)
        d = torch.log1p(dist).unsqueeze(-1).expand_as(rel)
        feats = torch.stack(
            [d, torch.cos(rel), torch.sin(rel), ws,
             torch.cos(rel) * ws, torch.exp(-dist.unsqueeze(-1).expand_as(rel) / 25.0)],
            dim=-1)                                           # (B,S,S,N,6)
        b = self.mlp(feats)                                   # (B,S,S,N,h)
        return b.permute(0, 3, 4, 1, 2)                       # (B,N,h,S,S)


# ---------------------------------------------------------------------------
# Axial encoder block: temporal -> variate -> spatial
# ---------------------------------------------------------------------------
class AxialBlock(nn.Module):
    """Full attention over 20k tokens is out of reach; axial factorisation makes
    the same receptive field affordable in three cheap passes."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.ModuleList(
            [MaskedMHA(d_model, n_heads, dropout) for _ in range(3)])
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _fold(x, valid, axis):
        """x: (B,S,C,N,D), valid: (B,S,C,N) -> (B*, L, D), (B*, L), unfold fn."""
        B, S, C, N, D = x.shape
        if axis == "time":
            xs, vs, shape = x, valid, (B * S * C, N, D)
        elif axis == "var":
            xs = x.permute(0, 1, 3, 2, 4).contiguous()
            vs = valid.permute(0, 1, 3, 2).contiguous()
            shape = (B * S * N, C, D)
        else:  # space
            xs = x.permute(0, 2, 3, 1, 4).contiguous()
            vs = valid.permute(0, 2, 3, 1).contiguous()
            shape = (B * C * N, S, D)
        return xs.reshape(*shape), vs.reshape(shape[0], shape[1])

    @staticmethod
    def _unfold(y, axis, B, S, C, N, D):
        if axis == "time":
            return y.view(B, S, C, N, D)
        if axis == "var":
            return y.view(B, S, N, C, D).permute(0, 1, 3, 2, 4).contiguous()
        return y.view(B, C, N, S, D).permute(0, 3, 1, 2, 4).contiguous()

    def forward(self, x, valid, spatial_bias=None):
        B, S, C, N, D = x.shape
        for i, axis in enumerate(("time", "var", "space")):
            h = self.norm[i](x)
            hf, vf = self._fold(h, valid, axis)
            bias = None
            if axis == "space" and spatial_bias is not None:
                # (B,N,h,S,S) -> (B*C*N, h, S, S) matching fold order (B,C,N)
                nh = spatial_bias.shape[2]
                bias = (spatial_bias.unsqueeze(1)
                        .expand(B, C, N, nh, S, S)
                        .reshape(B * C * N, nh, S, S))
            out = self.attn[i](hf, key_valid=vf, bias=bias)
            x = x + self.drop(self._unfold(out, axis, B, S, C, N, D))
        return x + self.drop(self.ff(self.norm[3](x)))
