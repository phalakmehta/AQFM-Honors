"""Probabilistic losses.

Air-quality decisions are threshold decisions -- AQI bands, GRAP stages, school
closures -- so a calibrated upper tail matters more than a good conditional
mean. Everything here is distributional; nothing here is MSE.
"""

import math

import torch
import torch.nn.functional as F

LOG_2PI = math.log(2 * math.pi)


def mixture_nll(logits, mu, log_sigma, y, mask=None, reduce=True):
    """Negative log-likelihood of a Gaussian mixture.

    logits, mu, log_sigma: (..., K); y: (...); mask: (...) float or bool.
    In normalised log1p space a Gaussian mixture is a log-normal mixture in
    concentration space.
    """
    logw = F.log_softmax(logits, dim=-1)
    z = (y.unsqueeze(-1) - mu) / log_sigma.exp()
    comp = -0.5 * z ** 2 - log_sigma - 0.5 * LOG_2PI
    ll = torch.logsumexp(logw + comp, dim=-1)
    nll = -ll
    if mask is None:
        return nll.mean() if reduce else nll
    mask = mask.to(nll.dtype)
    if not reduce:
        return nll * mask
    return (nll * mask).sum() / mask.sum().clamp(min=1.0)


def mixture_sample(logits, mu, log_sigma, n_samples, generator=None):
    """Reparameterised samples: (n_samples, ...). Gradient flows through mu and
    log_sigma; the component choice is sampled without a gradient path, which
    is the standard estimator and is unbiased for the scoring rules below."""
    probs = F.softmax(logits, dim=-1)
    shape = probs.shape[:-1]
    K = probs.shape[-1]
    flat = probs.reshape(-1, K)
    idx = torch.multinomial(flat, n_samples, replacement=True, generator=generator)
    idx = idx.t().reshape(n_samples, *shape)                       # (n, ...)
    m = torch.gather(mu.unsqueeze(0).expand(n_samples, *mu.shape), -1,
                     idx.unsqueeze(-1)).squeeze(-1)
    s = torch.gather(log_sigma.unsqueeze(0).expand(n_samples, *log_sigma.shape),
                     -1, idx.unsqueeze(-1)).squeeze(-1).exp()
    return m + s * torch.randn_like(m)


def energy_score(samples, y, mask=None):
    """Sample estimator of CRPS (the univariate energy score).

    CRPS = E|X - y| - 0.5 E|X - X'|.  Unlike NLL it is bounded and stays
    well-behaved when the model is badly calibrated early in training, so the
    two together train more stably than either alone.

    The naive pairwise term materialises an (n, n, ...) tensor and will OOM at
    realistic query counts. Using the order-statistic identity

        sum_{i,j} |x_i - x_j| = 2 * sum_i (2i - n - 1) * x_(i)

    the same quantity costs O(n log n) time and O(n) memory.
    """
    n = samples.shape[0]
    a = (samples - y.unsqueeze(0)).abs().mean(0)

    xs, _ = torch.sort(samples, dim=0)
    i = torch.arange(1, n + 1, device=samples.device, dtype=samples.dtype)
    w = (2 * i - n - 1).view(-1, *([1] * (samples.dim() - 1)))
    b = 2.0 * (w * xs).sum(0) / (n * (n - 1))

    sc = a - 0.5 * b
    if mask is None:
        return sc.mean()
    mask = mask.to(sc.dtype)
    return (sc * mask).sum() / mask.sum().clamp(min=1.0)


def episode_weight(y_log1p, threshold_log1p, alpha=2.0):
    """Up-weight high-concentration targets.

    Aggregate error on Delhi PM2.5 is dominated by the easy summer months. This
    keeps the winter episodes from being averaged into irrelevance without
    resorting to a hard-thresholded classification loss.
    """
    return 1.0 + alpha * torch.sigmoid(4.0 * (y_log1p - threshold_log1p))


def total_loss(out, target, mask, cfg, physical_fn=None, physical_target=None):
    """Assemble NLL + CRPS (+ optional physical-space consistency).

    out: dict from AQFM.forward with logits/mu/log_sigma of shape (B,Q,P,K)
    target, mask: (B,Q,P) in normalised space.
    """
    logits, mu, ls = out["logits"], out["mu"], out["log_sigma"]
    parts = {}
    parts["nll"] = mixture_nll(logits, mu, ls, target, mask)

    samples = mixture_sample(logits, mu, ls, cfg.n_crps_samples)
    parts["crps"] = energy_score(samples, target, mask)

    loss = cfg.w_nll * parts["nll"] + cfg.w_crps * parts["crps"]

    if physical_fn is not None and physical_target is not None:
        # samples: (n,B,Q,P) -> reconstructed observables in log1p space
        recon, recon_mask = physical_fn(samples)
        loss_phys = energy_score(recon, physical_target["y"],
                                 physical_target["mask"] * recon_mask)
        parts["physical"] = loss_phys
        loss = loss + cfg.w_physical * loss_phys

    parts["total"] = loss
    return loss, {k: float(v.detach()) for k, v in parts.items()}
