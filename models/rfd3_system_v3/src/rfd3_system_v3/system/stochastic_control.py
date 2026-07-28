"""Reverse-SDE and stochastic SuperDiff control for one shared coordinate state.

This module contains only tensor algebra.  Neural-network evaluation and track
bookkeeping remain in the inference sampler, which keeps the mathematical
operations independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ItoDensityDiagnostics:
    """Per-sample Itô density increments for one proposed shared-state update."""

    delta_log_q_1: torch.Tensor
    delta_log_q_2: torch.Tensor
    target_density_difference: torch.Tensor
    density_difference: torch.Tensor
    density_residual: torch.Tensor


@dataclass(frozen=True)
class StochasticKappaDiagnostics:
    """Per-sample values produced by the stochastic two-track kappa solve."""

    raw_kappa: torch.Tensor
    kappa: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor
    field_difference_norm_sq: torch.Tensor
    relative_field_difference: torch.Tensor
    degenerate: torch.Tensor
    clamped: torch.Tensor
    raw_delta_log_q_1: torch.Tensor
    raw_delta_log_q_2: torch.Tensor
    raw_density_difference: torch.Tensor
    raw_density_residual: torch.Tensor
    delta_log_q_1: torch.Tensor
    delta_log_q_2: torch.Tensor
    density_difference: torch.Tensor
    density_residual: torch.Tensor
    target_density_difference: torch.Tensor
    noise_projection_on_field_difference: torch.Tensor
    v_0_norm: torch.Tensor
    v_1_norm: torch.Tensor
    v_2_norm: torch.Tensor
    mixed_field_norm: torch.Tensor


def _reduce_dims(value: torch.Tensor) -> tuple[int, ...]:
    if value.ndim < 2:
        raise ValueError("Expected a batched field with at least two dimensions.")
    return tuple(range(1, value.ndim))


def _batch_scalar(
    value: float | torch.Tensor,
    *,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        return tensor.expand(reference.shape[0])
    if tensor.shape == (reference.shape[0],):
        return tensor
    raise ValueError(
        f"{name} must be scalar or have shape ({reference.shape[0]},), "
        f"received {tuple(tensor.shape)}."
    )


def dot_by_sample(value_1: torch.Tensor, value_2: torch.Tensor) -> torch.Tensor:
    """Return a float32 inner product for every batch member."""

    if value_1.shape != value_2.shape:
        raise ValueError(
            "Dot-product inputs must have matching shapes, got "
            f"{tuple(value_1.shape)} and {tuple(value_2.shape)}."
        )
    reduce_dims = _reduce_dims(value_1)
    return torch.sum(value_1.float() * value_2.float(), dim=reduce_dims)


def squared_norm_by_sample(value: torch.Tensor) -> torch.Tensor:
    """Return a float32 squared Euclidean norm for every batch member."""

    return dot_by_sample(value, value)


def norm_by_sample(value: torch.Tensor) -> torch.Tensor:
    """Return a float32 Euclidean norm for every batch member."""

    return torch.sqrt(squared_norm_by_sample(value).clamp_min(0.0))


def cosine_similarity_by_sample(
    value_1: torch.Tensor,
    value_2: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a finite cosine similarity for each batch member."""

    dot = dot_by_sample(value_1, value_2)
    denominator = norm_by_sample(value_1) * norm_by_sample(value_2)
    return torch.where(
        denominator > eps,
        dot / denominator.clamp_min(eps),
        torch.zeros_like(dot),
    )


def build_guided_base_field(
    v_2: torch.Tensor,
    *,
    guidance_scale: float,
    v_0: torch.Tensor | None,
) -> torch.Tensor:
    """Construct ``v0 + guidance_scale * (v2 - v0)``.

    At unit guidance, the reference field cancels algebraically and may be
    omitted to avoid an unnecessary isolated-chain denoiser evaluation.
    """

    if guidance_scale <= 0:
        raise ValueError("superdiff_guidance_scale must be greater than zero.")
    if guidance_scale == 1.0:
        return v_2
    if v_0 is None:
        raise ValueError("v_0 is required when superdiff_guidance_scale != 1.")
    if v_0.shape != v_2.shape:
        raise ValueError(
            "v_0 and v_2 must have matching shapes, got "
            f"{tuple(v_0.shape)} and {tuple(v_2.shape)}."
        )
    return v_0 + guidance_scale * (v_2 - v_0)


def reverse_sde_noise_scale(
    sigma: float | torch.Tensor,
    d_sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Return the Euler-Maruyama scale ``sqrt(2 * sigma * abs(d_sigma))``."""

    sigma_t = torch.as_tensor(sigma)
    d_sigma_t = torch.as_tensor(
        d_sigma,
        device=sigma_t.device,
        dtype=sigma_t.dtype,
    )
    if torch.any(sigma_t <= 0):
        raise ValueError("sigma must be positive.")
    if torch.any(d_sigma_t >= 0):
        raise ValueError("d_sigma must be negative for reverse-SDE denoising.")
    return torch.sqrt(2.0 * sigma_t * (-d_sigma_t))


def reverse_sde_increment(
    velocity: torch.Tensor,
    *,
    d_sigma: float | torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Return ``2 * d_sigma * velocity + noise``."""

    if velocity.shape != noise.shape:
        raise ValueError("velocity and noise must have matching shapes.")
    d_sigma_t = torch.as_tensor(
        d_sigma,
        device=velocity.device,
        dtype=velocity.dtype,
    )
    if torch.any(d_sigma_t >= 0):
        raise ValueError("d_sigma must be negative for reverse-SDE denoising.")
    return 2.0 * d_sigma_t * velocity + noise


def evaluate_ito_density_increments(
    v_1: torch.Tensor,
    v_2: torch.Tensor,
    *,
    shared_increment: torch.Tensor,
    sigma: float | torch.Tensor,
    d_sigma: float | torch.Tensor,
    target_density_difference: float | torch.Tensor = 0.0,
) -> ItoDensityDiagnostics:
    """Evaluate the finite-step Itô density estimator used by SuperDiff.

    For ``h = -d_sigma`` the estimator is

    ``delta_log_q_i = -(h / sigma) * ||v_i||^2
                      - dot(shared_increment, v_i) / sigma``.
    """

    if v_1.shape != v_2.shape or v_1.shape != shared_increment.shape:
        raise ValueError("v_1, v_2, and shared_increment must have matching shapes.")

    reference = squared_norm_by_sample(v_1)
    sigma_f = _batch_scalar(sigma, reference=reference, name="sigma")
    d_sigma_f = _batch_scalar(d_sigma, reference=reference, name="d_sigma")
    target_f = _batch_scalar(
        target_density_difference,
        reference=reference,
        name="target_density_difference",
    )
    if torch.any(sigma_f <= 0):
        raise ValueError("sigma must be positive.")
    if torch.any(d_sigma_f >= 0):
        raise ValueError("d_sigma must be negative.")

    h = -d_sigma_f
    delta_log_q_1 = (
        -(h / sigma_f) * squared_norm_by_sample(v_1)
        - dot_by_sample(shared_increment, v_1) / sigma_f
    )
    delta_log_q_2 = (
        -(h / sigma_f) * squared_norm_by_sample(v_2)
        - dot_by_sample(shared_increment, v_2) / sigma_f
    )
    density_difference = delta_log_q_1 - delta_log_q_2
    return ItoDensityDiagnostics(
        delta_log_q_1=delta_log_q_1,
        delta_log_q_2=delta_log_q_2,
        target_density_difference=target_f,
        density_difference=density_difference,
        density_residual=density_difference - target_f,
    )


def solve_stochastic_superdiff_kappa(
    v_1: torch.Tensor,
    v_2: torch.Tensor,
    *,
    shared_noise: torch.Tensor,
    sigma: float | torch.Tensor,
    d_sigma: float | torch.Tensor,
    guidance_scale: float,
    v_0: torch.Tensor | None = None,
    lift: float = 0.0,
    num_steps: int,
    kappa_min: float | None = -1.0,
    kappa_max: float | None = 2.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, StochasticKappaDiagnostics]:
    """Solve stochastic two-track SuperDiff AND for one shared state.

    The supplied noise must be the exact Brownian increment that will be used
    in the shared-state update.  Reusing it is required for the Itô equality.
    """

    if v_1.shape != v_2.shape or v_1.shape != shared_noise.shape:
        raise ValueError("v_1, v_2, and shared_noise must have matching shapes.")
    if guidance_scale <= 0:
        raise ValueError("superdiff_guidance_scale must be greater than zero.")
    if num_steps <= 0:
        raise ValueError("num_steps must be greater than zero.")
    if eps <= 0:
        raise ValueError("kappa_eps must be greater than zero.")
    if kappa_min is not None and kappa_max is not None and kappa_min > kappa_max:
        raise ValueError("kappa_min cannot exceed kappa_max.")

    v_1_f = v_1.float()
    v_2_f = v_2.float()
    noise_f = shared_noise.float()
    v_0_f = v_0.float() if v_0 is not None else None

    reference = squared_norm_by_sample(v_1_f)
    sigma_f = _batch_scalar(sigma, reference=reference, name="sigma")
    d_sigma_f = _batch_scalar(d_sigma, reference=reference, name="d_sigma")
    if torch.any(sigma_f <= 0):
        raise ValueError("sigma must be positive.")
    if torch.any(d_sigma_f >= 0):
        raise ValueError("d_sigma must be negative.")
    h = -d_sigma_f

    base_field = build_guided_base_field(
        v_2_f,
        guidance_scale=guidance_scale,
        v_0=v_0_f,
    )
    field_difference = v_1_f - v_2_f
    field_difference_norm_sq = squared_norm_by_sample(field_difference)
    field_scale = (
        0.5
        * (
            squared_norm_by_sample(v_1_f)
            + squared_norm_by_sample(v_2_f)
        )
    ).clamp_min(torch.finfo(torch.float32).tiny)
    relative_field_difference = field_difference_norm_sq / field_scale
    degenerate = relative_field_difference <= eps

    d_sigma_view = d_sigma_f.reshape(
        d_sigma_f.shape[0], *([1] * (v_1_f.ndim - 1))
    )
    base_increment = 2.0 * d_sigma_view * base_field + noise_f
    lift_term = sigma_f * float(lift) / float(num_steps)
    numerator = (
        h * dot_by_sample(v_2_f - v_1_f, v_2_f + v_1_f)
        - dot_by_sample(base_increment, field_difference)
        + lift_term
    )
    denominator = (
        2.0
        * d_sigma_f
        * float(guidance_scale)
        * field_difference_norm_sq
    )

    fallback = torch.full_like(numerator, 0.5)
    safe_denominator = torch.where(
        degenerate,
        torch.ones_like(denominator),
        denominator,
    )
    raw_kappa = torch.where(
        degenerate,
        fallback,
        numerator / safe_denominator,
    )
    kappa = raw_kappa
    if kappa_min is not None:
        kappa = torch.maximum(kappa, torch.full_like(kappa, float(kappa_min)))
    if kappa_max is not None:
        kappa = torch.minimum(kappa, torch.full_like(kappa, float(kappa_max)))
    clamped = ~torch.isclose(kappa, raw_kappa)

    kappa_view = kappa.reshape(kappa.shape[0], *([1] * (v_1_f.ndim - 1)))
    raw_kappa_view = raw_kappa.reshape(
        raw_kappa.shape[0], *([1] * (v_1_f.ndim - 1))
    )
    mixed_field = base_field + guidance_scale * kappa_view * field_difference
    raw_mixed_field = (
        base_field + guidance_scale * raw_kappa_view * field_difference
    )
    applied_increment = 2.0 * d_sigma_view * mixed_field + noise_f
    raw_increment = 2.0 * d_sigma_view * raw_mixed_field + noise_f

    target_difference = torch.full_like(
        raw_kappa,
        -float(lift) / float(num_steps),
    )
    raw_density = evaluate_ito_density_increments(
        v_1_f,
        v_2_f,
        shared_increment=raw_increment,
        sigma=sigma_f,
        d_sigma=d_sigma_f,
        target_density_difference=target_difference,
    )
    applied_density = evaluate_ito_density_increments(
        v_1_f,
        v_2_f,
        shared_increment=applied_increment,
        sigma=sigma_f,
        d_sigma=d_sigma_f,
        target_density_difference=target_difference,
    )

    diagnostics = StochasticKappaDiagnostics(
        raw_kappa=raw_kappa,
        kappa=kappa,
        numerator=numerator,
        denominator=denominator,
        field_difference_norm_sq=field_difference_norm_sq,
        relative_field_difference=relative_field_difference,
        degenerate=degenerate,
        clamped=clamped,
        raw_delta_log_q_1=raw_density.delta_log_q_1,
        raw_delta_log_q_2=raw_density.delta_log_q_2,
        raw_density_difference=raw_density.density_difference,
        raw_density_residual=raw_density.density_residual,
        delta_log_q_1=applied_density.delta_log_q_1,
        delta_log_q_2=applied_density.delta_log_q_2,
        density_difference=applied_density.density_difference,
        density_residual=applied_density.density_residual,
        target_density_difference=applied_density.target_density_difference,
        noise_projection_on_field_difference=dot_by_sample(
            noise_f, field_difference
        ),
        v_0_norm=(
            norm_by_sample(v_0_f)
            if v_0_f is not None
            else torch.zeros_like(raw_kappa)
        ),
        v_1_norm=norm_by_sample(v_1_f),
        v_2_norm=norm_by_sample(v_2_f),
        mixed_field_norm=norm_by_sample(mixed_field),
    )
    return mixed_field.to(dtype=v_1.dtype), diagnostics
