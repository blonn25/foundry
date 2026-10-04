"""Coupling schedules and pure tensor operators; no Foundry imports required."""
from __future__ import annotations

import math
import torch


def schedule(value, progress):
    if isinstance(value, (int, float)):
        return float(value)
    start, end = float(value["start"]), float(value["end"])
    until = float(value.get("until", 1.0))
    if not 0 < until <= 1:
        raise ValueError("schedule until must lie in (0, 1]")
    p = max(0., min(1., float(progress) / until))
    kind = value.get("kind", "linear")
    if kind == "constant":
        return start
    if kind == "cosine":
        p = (1 - math.cos(math.pi * p)) / 2
    elif kind != "linear":
        raise ValueError(f"unknown schedule {kind}")
    return start + (end - start) * p


def released(step, count, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("release_fraction must lie in [0, 1]")
    return step >= math.floor(fraction * count + 1e-10)


def residual(a, b, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    if alpha == 1:
        return a, b
    mean = 0.5 * a + 0.5 * b
    if alpha == 0:
        return mean, mean
    diff = 0.5 * (a - b)
    return mean + alpha * diff, mean - alpha * diff


def correlated(shared, a, b, rho):
    if not 0 <= rho <= 1:
        raise ValueError("noise correlation must lie in [0, 1]")
    if rho == 1:
        return shared, shared
    if rho == 0:
        return a, b
    return (math.sqrt(rho)*shared + math.sqrt(1-rho)*a,
            math.sqrt(rho)*shared + math.sqrt(1-rho)*b)


def rmsd(a, b, align=True):
    """Per-sample RMSD, using proper rotations (never reflection)."""
    a, b = a.float(), b.float()
    if align:
        a, b = a-a.mean(-2, keepdim=True), b-b.mean(-2, keepdim=True)
        u, _, vh = torch.linalg.svd(a.transpose(-1, -2) @ b)
        diag = torch.ones((*a.shape[:-2], 3), device=a.device)
        diag[..., -1] = torch.linalg.det(u @ vh).sign()
        a = a @ (u @ torch.diag_embed(diag) @ vh)
    return ((a-b).square().sum(-1).mean(-1)).sqrt()


def distance_energy(a, b, block=1):
    """SE(3)-invariant predicted-clean C-alpha energy, dimensionless."""
    if a.shape != b.shape or a.shape[-1] != 3:
        raise ValueError("matched C-alpha shapes required")
    if block < 1 or int(block) != block:
        raise ValueError("block size must be a positive integer")
    if block > 1:
        a = torch.stack([v.mean(-2) for v in a.split(block, -2)], -2)
        b = torch.stack([v.mean(-2) for v in b.split(block, -2)], -2)
    if a.shape[-2] < 2:
        raise ValueError("at least two coarse points required")
    ij = torch.triu_indices(a.shape[-2], a.shape[-2], 1, device=a.device)
    da = (a[..., ij[0], :] - a[..., ij[1], :]).norm(dim=-1)
    db = (b[..., ij[0], :] - b[..., ij[1], :]).norm(dim=-1)
    return ((da-db).square() / 200.).mean()


def js_energy(logits_a, logits_b, temperature=1.):
    """Canonical, designable A logits must already be selected by the caller."""
    if logits_a.shape != logits_b.shape or temperature <= 0:
        raise ValueError("matched logits and positive temperature required")
    logp = torch.log_softmax(logits_a.float()/temperature, -1)
    logq = torch.log_softmax(logits_b.float()/temperature, -1)
    logm = torch.logaddexp(logp, logq) - math.log(2)
    return (0.5*((logp.exp()*(logp-logm)).sum(-1) +
                 (logq.exp()*(logq-logm)).sum(-1))).mean()


def cap_correction(corrections, native_updates, masks, limit=0.5):
    """One scale for both states preserves their relative guidance strength."""
    ratios = []
    for c, u, mask in zip(corrections, native_updates, masks):
        ratios.append(c[:, mask].float().norm() / u[:, mask].float().norm().clamp_min(1e-12))
    maximum = max(float(v) for v in ratios)
    factor = min(1., limit / max(maximum, 1e-12))
    return [c*factor for c in corrections], [float(v) for v in ratios], factor


def profile(name):
    method = name.removeprefix("rfd3_")
    known = {"uncoupled", "mean_5050", "residual_consensus", "late_uncoupling",
             "correlated_noise", "soft_guidance", "coarse_coupling",
             "population_diversity", "sequence_coupling"}
    if method not in known:
        raise ValueError(f"unknown variation {name}")
    return dict(method=method, alpha=0.5, release_fraction=0.8,
                shared_initialization=True, noise_correlation=1.,
                strength=0., guidance_schedule=1., temperature=1., block=1,
                correction_cap=0.5, debug=False, calibration=False)


def validate(config):
    profile(config["method"])
    if not isinstance(config["shared_initialization"], bool):
        raise ValueError("shared_initialization must be boolean")
    if config["strength"] < 0 or not math.isfinite(config["strength"]):
        raise ValueError("guidance strength must be finite and nonnegative")
    if not 0 < config["correction_cap"] <= 1:
        raise ValueError("correction_cap must lie in (0,1]")
    for p in (0., 0.5, 1.):
        for key in ("alpha", "noise_correlation"):
            if not 0 <= schedule(config[key], p) <= 1:
                raise ValueError(f"invalid {key}")
        if schedule(config["temperature"], p) <= 0:
            raise ValueError("temperature must be positive")
        if schedule(config["guidance_schedule"], p) < 0:
            raise ValueError("guidance schedule must be nonnegative")
    released(0, 1, config["release_fraction"])
