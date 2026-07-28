"""Proxy weighting utilities for approximate shared-chain coupling.

These functions deliberately do not claim to implement exact SuperDiff density
control.  They operate on RFD3 denoiser-derived update vectors and solve a
small stabilized proxy equation that is useful for research prototyping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProxyKappaDiagnostics:
    """Diagnostics from the approximate two-track weight solve."""

    raw_kappa: torch.Tensor
    kappa: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor
    degenerate: torch.Tensor
    proxy_residual: torch.Tensor
    delta_1_norm: torch.Tensor
    delta_2_norm: torch.Tensor


def cosine_similarity_by_sample(
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return cosine similarity for paired batched update tensors.

    The first dimension is treated as the sample/batch dimension. All remaining
    dimensions are flattened into one update vector per sample.
    """

    if delta_a.shape != delta_b.shape:
        raise ValueError(
            f"Cosine inputs must have matching shapes, got "
            f"{tuple(delta_a.shape)} and {tuple(delta_b.shape)}."
        )
    if delta_a.ndim < 2:
        raise ValueError("Expected batched update tensors with at least 2 dims.")

    reduce_dims = tuple(range(1, delta_a.ndim))
    numerator = torch.sum(delta_a * delta_b, dim=reduce_dims)
    norm_a = torch.sqrt(torch.sum(delta_a.square(), dim=reduce_dims))
    norm_b = torch.sqrt(torch.sum(delta_b.square(), dim=reduce_dims))
    return numerator / (norm_a * norm_b + eps)


def solve_two_track_proxy_kappa(
    delta_1: torch.Tensor,
    delta_2: torch.Tensor,
    *,
    norm_weight: float = 1.0,
    kappa_min: float = -1.0,
    kappa_max: float = 2.0,
    eps: float = 1e-8,
) -> ProxyKappaDiagnostics:
    """Solve the approximate two-track shared-chain mixing weight.

    The inputs are denoiser-derived update proxies for the same shared atoms in
    two condition-specific contexts.  We use a simple density-change proxy,

        proxy_i(delta_mix) = <delta_mix, delta_i> - norm_weight * ||delta_i||^2

    and choose kappa in `delta_mix = kappa * delta_1 + (1-kappa) * delta_2`
    so that `proxy_1(delta_mix) ~= proxy_2(delta_mix)`.  This is a heuristic
    analogue of SuperDiff's density-control linear system; it is not the
    Itô-density estimator and it does not use exact scores.
    """

    if delta_1.shape != delta_2.shape:
        raise ValueError(
            f"Shared-chain update proxies must have matching shapes, got "
            f"{tuple(delta_1.shape)} and {tuple(delta_2.shape)}."
        )
    if delta_1.ndim < 2:
        raise ValueError("Expected batched update tensors with at least 2 dims.")

    reduce_dims = tuple(range(1, delta_1.ndim))
    delta_diff = delta_1 - delta_2
    denominator = torch.sum(delta_diff.square(), dim=reduce_dims)
    norm_1_sq = torch.sum(delta_1.square(), dim=reduce_dims)
    norm_2_sq = torch.sum(delta_2.square(), dim=reduce_dims)
    dot_2_diff = torch.sum(delta_2 * delta_diff, dim=reduce_dims)

    numerator = norm_weight * (norm_1_sq - norm_2_sq) - dot_2_diff
    degenerate = denominator <= eps
    safe_denominator = torch.where(degenerate, torch.ones_like(denominator), denominator)
    raw_kappa = numerator / safe_denominator
    raw_kappa = torch.where(degenerate, torch.full_like(raw_kappa, 0.5), raw_kappa)
    kappa = raw_kappa.clamp(min=kappa_min, max=kappa_max)

    expand_shape = (kappa.shape[0],) + (1,) * (delta_1.ndim - 1)
    delta_mix = kappa.reshape(expand_shape) * delta_1 + (
        1 - kappa.reshape(expand_shape)
    ) * delta_2
    proxy_1 = torch.sum(delta_mix * delta_1, dim=reduce_dims) - norm_weight * norm_1_sq
    proxy_2 = torch.sum(delta_mix * delta_2, dim=reduce_dims) - norm_weight * norm_2_sq
    proxy_residual = proxy_1 - proxy_2

    return ProxyKappaDiagnostics(
        raw_kappa=raw_kappa,
        kappa=kappa,
        numerator=numerator,
        denominator=denominator,
        degenerate=degenerate,
        proxy_residual=proxy_residual,
        delta_1_norm=torch.sqrt(norm_1_sq),
        delta_2_norm=torch.sqrt(norm_2_sq),
    )
