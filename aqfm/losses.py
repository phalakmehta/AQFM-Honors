"""Probabilistic losses. Nothing here is MSE: AQI decisions are threshold
decisions, so a calibrated upper tail matters more than a conditional mean."""

import math

import torch
import torch.nn.functional as F

from .physical import reconstruct_no_no2, reconstruct_pm25

LOG_2PI = math.log(2 * math.pi)
_SQRT2 = math.sqrt(2.0)


def _masked_mean(v, mask, weight=None):
    w = mask.to(v.dtype) if weight is None else mask.to(v.dtype) * weight
    return (v * w).sum() / w.sum().clamp(min=1e-8)


def mixture_nll(logits, mu, log_sigma, y):
    """Per-element NLL of a Gaussian mixture. logits/mu/log_sigma (...,K), y (...)."""
    z = (y.unsqueeze(-1) - mu) * torch.exp(-log_sigma)
    comp = -0.5 * z ** 2 - log_sigma - 0.5 * LOG_2PI
    return -torch.logsumexp(F.log_softmax(logits, -1) + comp, -1)


def _A(m, s):
    """E|X| for X ~ N(m, s^2)."""
    z = m / s
    pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / _SQRT2))
    return 2 * s * pdf + m * (2 * cdf - 1)


def crps_mixture(logits, mu, log_sigma, y):
    """Exact per-element CRPS of a Gaussian mixture (Grimit et al. 2006):
        CRPS = sum_i w_i A(y - mu_i, s_i) - 1/2 sum_ij w_i w_j A(mu_i - mu_j, sqrt(s_i^2 + s_j^2))
    Closed form, so no sampling noise in the gradient; O(K^2) with K = 3."""
    w = F.softmax(logits, -1)
    s = torch.exp(log_sigma)
    t1 = (w * _A(y.unsqueeze(-1) - mu, s)).sum(-1)
    dm = mu.unsqueeze(-1) - mu.unsqueeze(-2)
    ds = torch.sqrt(s.unsqueeze(-1) ** 2 + s.unsqueeze(-2) ** 2)
    t2 = (w.unsqueeze(-1) * w.unsqueeze(-2) * _A(dm, ds)).sum((-1, -2))
    return t1 - 0.5 * t2


def mixture_sample(logits, mu, log_sigma, n, generator=None):
    """(n, ...) reparameterised samples; gradients flow through mu and sigma."""
    K = logits.shape[-1]
    idx = torch.multinomial(F.softmax(logits, -1).reshape(-1, K).float(), n,
                            replacement=True, generator=generator)
    idx = idx.t().reshape(n, *logits.shape[:-1], 1)
    pick = lambda a: torch.gather(a.unsqueeze(0).expand(n, *a.shape), -1, idx).squeeze(-1)
    eps = torch.randn(idx.shape[:-1], device=mu.device, dtype=mu.dtype, generator=generator)
    return pick(mu) + pick(log_sigma).exp() * eps


def sample_crps(samples, y):
    """Per-element CRPS from an ensemble, unbiased ('fair') estimator:
        CRPS = mean|X - y| - 1/(2n(n-1)) sum_{i,j} |X_i - X_j|
    with the pairwise sum from order statistics,
        sum_{i,j} |x_i - x_j| = 2 sum_i (2i - n - 1) x_(i),
    i.e. O(n log n) time and O(n) memory instead of an (n, n, ...) tensor."""
    n = samples.shape[0]
    a = (samples - y.unsqueeze(0)).abs().mean(0)
    xs = torch.sort(samples, 0).values
    i = torch.arange(1, n + 1, device=samples.device, dtype=samples.dtype)
    w = (2 * i - n - 1).view(-1, *([1] * (samples.dim() - 1)))
    return a - (w * xs).sum(0) / (n * (n - 1))


def episode_weight(y_log1p, threshold, alpha):
    """1 + alpha * sigmoid(4 (y - log1p(threshold))): keeps winter episodes
    from being averaged away by clean summer months."""
    return 1.0 + alpha * torch.sigmoid(4.0 * (y_log1p - math.log1p(threshold)))


# ---------------------------------------------------------------------------
def physical_terms(cfg, samples_t, q, batch, n_forecast):
    """CRPS of reconstructed observables (PM2.5; NO and NO2) scored as
    distributions. samples_t: (n,B,Q,P) samples in transformed (log1p/logit)
    space. Returns list of (per-element CRPS, mask, y) tensors."""
    B, S = batch["x_ctx"].shape[:2]
    Ct, H, P = cfg.n_target_channels, cfg.n_horizon_patches, cfg.patch_len
    n = samples_t.shape[0]
    blocks = []      # (samples with the target-slot axis at dim 3, truth (B,3,2,...))
    if n_forecast:
        blocks.append((samples_t[:, :, :n_forecast].reshape(n, B, S, Ct, H, P),
                       batch["phys_hz"].reshape(B, 3, 2, S, H, P)))
    if samples_t.shape[2] > n_forecast:
        Pm = batch["imp_pos"].shape[1]
        blocks.append((samples_t[:, :, n_forecast:].reshape(n, B, Pm, Ct, P),
                       batch["imp_phys"].permute(0, 2, 3, 1, 4)))           # (B,3,2,Pm,P)

    out = []
    for samp, phys in blocks:
        for name, base, frac in cfg.physical_pairs():
            b, f = samp[:, :, :, base], samp[:, :, :, frac]
            if name == "PM":
                recon = [(reconstruct_pm25(b, f), phys[:, 0])]
            else:
                no, no2 = reconstruct_no_no2(b, f)
                recon = [(no, phys[:, 1]), (no2, phys[:, 2])]
            for r, yp in recon:                                # yp: (B,2,...) = [y, mask]
                out.append((sample_crps(r, yp[:, 0]), yp[:, 1], yp[:, 0]))
    return out


def total_loss(model, out, q, y, ym, batch, cfg, n_forecast):
    """NLL + exact CRPS in normalised space, plus sample CRPS of reconstructed
    observables in log1p space. y, ym: (B,Q,P) transformed targets/masks."""
    mu_q, sd_q = model.query_stats(out["mean"].float(), out["std"].float(), q)
    yn = (y - mu_q) / sd_q                                     # context stats only
    # the loss maths (erf, logsumexp, sort) always runs in float32, even under bf16 autocast
    logits, mu, ls = out["logits"].float(), out["mu"].float(), out["log_sigma"].float()

    nll = mixture_nll(logits, mu, ls, yn)
    crps = crps_mixture(logits, mu, ls, yn)
    parts = {}
    is_f = torch.zeros_like(ym, dtype=torch.bool)
    is_f[:, :n_forecast] = True
    parts["nll_forecast"] = _masked_mean(nll, ym * is_f)
    parts["crps_forecast"] = _masked_mean(crps, ym * is_f)
    if ym.shape[1] > n_forecast:
        parts["nll_impute"] = _masked_mean(nll, ym * ~is_f)
        parts["crps_impute"] = _masked_mean(crps, ym * ~is_f)
    loss = cfg.w_nll * _masked_mean(nll, ym) + cfg.w_crps * _masked_mean(crps, ym)

    if cfg.w_physical > 0 and cfg.physical_pairs():
        samples = mixture_sample(logits, mu, ls, cfg.n_crps_samples)   # (n,B,Q,P) normalised
        samples_t = samples * sd_q.unsqueeze(0) + mu_q.unsqueeze(0)
        terms = physical_terms(cfg, samples_t, q, batch, n_forecast)
        if terms:
            num = sum((c * m * (episode_weight(yv, cfg.episode_threshold, cfg.episode_alpha)
                                if cfg.episode_alpha > 0 else 1.0)).sum() for c, m, yv in terms)
            den = sum(m.sum() for _, m, _ in terms).clamp(min=1.0)
            parts["physical"] = num / den
            loss = loss + cfg.w_physical * parts["physical"]

    parts["total"] = loss
    return loss, {k: float(v.detach()) for k, v in parts.items()}
