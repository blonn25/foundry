"""Deterministic SuperDiff density control for a shared coordinate state.

The functions in this module implement the closed-form two-track kappa solve.
They do not run the denoiser or estimate divergences; those operations remain
in the inference sampler so the neural-network call path is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DensityKappaDiagnostics:
    """Per-sample values produced by the deterministic density-rate solve."""

    raw_kappa: torch.Tensor
    kappa: torch.Tensor
    denominator: torch.Tensor
    field_difference_norm_sq: torch.Tensor
    degenerate: torch.Tensor
    clamped: torch.Tensor
    density_rate_1: torch.Tensor
    density_rate_2: torch.Tensor
    target_rate_difference: torch.Tensor
    density_rate_residual: torch.Tensor
    delta_log_q_1: torch.Tensor
    delta_log_q_2: torch.Tensor
    v_0_norm: torch.Tensor
    v_1_norm: torch.Tensor
    v_2_norm: torch.Tensor


@dataclass(frozen=True)
class DensityRateDiagnostics:
    """Density rates and residual for a supplied shared-chain path field."""

    density_rate_1: torch.Tensor
    density_rate_2: torch.Tensor
    target_rate_difference: torch.Tensor
    density_rate_residual: torch.Tensor
    delta_log_q_1: torch.Tensor
    delta_log_q_2: torch.Tensor


def _reduce_dims(value: torch.Tensor) -> tuple[int, ...]:
    if value.ndim < 2:
        raise ValueError("Expected a batched vector field with at least two dimensions.")
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


def cosine_similarity_by_sample(
    value_1: torch.Tensor,
    value_2: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a finite cosine similarity for each batch member."""

    if value_1.shape != value_2.shape:
        raise ValueError(
            "Cosine inputs must have matching shapes, got "
            f"{tuple(value_1.shape)} and {tuple(value_2.shape)}."
        )
    reduce_dims = _reduce_dims(value_1)
    dot = torch.sum(value_1 * value_2, dim=reduce_dims)
    norm_1 = torch.sqrt(torch.sum(value_1.square(), dim=reduce_dims))
    norm_2 = torch.sqrt(torch.sum(value_2.square(), dim=reduce_dims))
    denominator = norm_1 * norm_2
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
    """Construct ``v0 + g * (v2 - v0)``.

    At unit guidance the reference field cancels algebraically, so callers may
    omit ``v_0`` and avoid an unnecessary A-only denoiser evaluation.
    """

    if guidance_scale <= 0:
        raise ValueError("superdiff_guidance_scale must be greater than zero.")
    if guidance_scale == 1.0:
        return v_2
    if v_0 is None:
        raise ValueError("v_0 is required when superdiff_guidance_scale != 1.")
    if v_0.shape != v_2.shape:
        raise ValueError(
            f"v_0 and v_2 must have matching shapes, got "
            f"{tuple(v_0.shape)} and {tuple(v_2.shape)}."
        )
    return v_0 + guidance_scale * (v_2 - v_0)


def evaluate_density_rates(
    v_1: torch.Tensor,
    v_2: torch.Tensor,
    dlog_1: torch.Tensor,
    dlog_2: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    d_sigma: float | torch.Tensor,
    path_field: torch.Tensor,
    target_rate_difference: float | torch.Tensor = 0.0,
) -> DensityRateDiagnostics:
    """Evaluate two conditional density rates along one supplied path field."""

    if v_1.shape != v_2.shape or v_1.shape != path_field.shape:
        raise ValueError("v_1, v_2, and path_field must have matching shapes.")
    reduce_dims = _reduce_dims(v_1)
    if dlog_1.shape != (v_1.shape[0],) or dlog_2.shape != (v_1.shape[0],):
        raise ValueError("dlog tensors must contain one scalar per sample.")

    v_1_f = v_1.float()
    v_2_f = v_2.float()
    path_field_f = path_field.float()
    dlog_1_f = dlog_1.float()
    dlog_2_f = dlog_2.float()
    sigma_f = _batch_scalar(sigma, reference=dlog_1_f, name="sigma")
    d_sigma_f = _batch_scalar(d_sigma, reference=dlog_1_f, name="d_sigma")
    target_f = _batch_scalar(
        target_rate_difference,
        reference=dlog_1_f,
        name="target_rate_difference",
    )
    if torch.any(sigma_f <= 0):
        raise ValueError("sigma must be positive.")

    density_rate_1 = dlog_1_f + torch.sum(
        v_1_f * (v_1_f - path_field_f), dim=reduce_dims
    ) / sigma_f
    density_rate_2 = dlog_2_f + torch.sum(
        v_2_f * (v_2_f - path_field_f), dim=reduce_dims
    ) / sigma_f
    residual = density_rate_1 - density_rate_2 - target_f
    return DensityRateDiagnostics(
        density_rate_1=density_rate_1,
        density_rate_2=density_rate_2,
        target_rate_difference=target_f,
        density_rate_residual=residual,
        delta_log_q_1=d_sigma_f * density_rate_1,
        delta_log_q_2=d_sigma_f * density_rate_2,
    )


def solve_deterministic_density_kappa(
    v_1: torch.Tensor,
    v_2: torch.Tensor,
    dlog_1: torch.Tensor,
    dlog_2: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    d_sigma: float | torch.Tensor,
    step_scale: float,
    guidance_scale: float,
    v_0: torch.Tensor | None = None,
    lift: float = 0.0,
    num_steps: int,
    kappa_min: float | None = -1.0,
    kappa_max: float | None = 2.0,
    eps: float = 1e-8,
) -> DensityKappaDiagnostics:
    """Solve the deterministic two-track SuperDiff AND equation.

    ``v_1`` and ``v_2`` are conditional probability-flow fields on the same
    selected shared coordinates. ``dlog_i`` is ``-div(v_i)`` estimated with
    respect to those coordinates. The applied path field is

    ``u = step_scale * (v0 + g * ((v2-v0) + kappa * (v1-v2)))``.
    """

    if v_1.shape != v_2.shape:
        raise ValueError(
            f"Conditional fields must have matching shapes, got "
            f"{tuple(v_1.shape)} and {tuple(v_2.shape)}."
        )
    reduce_dims = _reduce_dims(v_1)
    if dlog_1.shape != (v_1.shape[0],) or dlog_2.shape != (v_1.shape[0],):
        raise ValueError(
            "dlog tensors must have one scalar per sample; expected "
            f"({v_1.shape[0]},), got {tuple(dlog_1.shape)} and {tuple(dlog_2.shape)}."
        )
    if step_scale <= 0:
        raise ValueError("step_scale must be greater than zero.")
    if guidance_scale <= 0:
        raise ValueError("superdiff_guidance_scale must be greater than zero.")
    if num_steps <= 0:
        raise ValueError("num_steps must be greater than zero.")
    if eps <= 0:
        raise ValueError("kappa_eps must be greater than zero.")
    if kappa_min is not None and kappa_max is not None and kappa_min > kappa_max:
        raise ValueError("kappa_min cannot exceed kappa_max.")

    # Scalar reductions are deliberately promoted to float32 even when model
    # inference uses bfloat16 automatic mixed precision.
    v_1_f = v_1.float()
    v_2_f = v_2.float()
    v_0_f = v_0.float() if v_0 is not None else None
    dlog_1_f = dlog_1.float()
    dlog_2_f = dlog_2.float()
    sigma_f = _batch_scalar(sigma, reference=dlog_1_f, name="sigma")
    d_sigma_f = _batch_scalar(d_sigma, reference=dlog_1_f, name="d_sigma")
    if torch.any(sigma_f <= 0):
        raise ValueError("sigma must be positive.")
    if torch.any(d_sigma_f == 0):
        raise ValueError("d_sigma must be nonzero.")

    v_base = build_guided_base_field(
        v_2_f,
        guidance_scale=guidance_scale,
        v_0=v_0_f,
    )
    delta = v_1_f - v_2_f
    delta_norm_sq = torch.sum(delta.square(), dim=reduce_dims)
    norm_1_sq = torch.sum(v_1_f.square(), dim=reduce_dims)
    norm_2_sq = torch.sum(v_2_f.square(), dim=reduce_dims)
    dot_delta_base = torch.sum(delta * v_base, dim=reduce_dims)

    denominator = step_scale * guidance_scale * delta_norm_sq
    numerator = (
        sigma_f * (dlog_1_f - dlog_2_f)
        + norm_1_sq
        - norm_2_sq
        - step_scale * dot_delta_base
    )
    target_rate_difference = torch.zeros_like(numerator)
    if lift != 0.0:
        numerator = numerator + sigma_f * lift / (num_steps * d_sigma_f)
        # This matches the SuperDiff notebook convention: positive lift lowers
        # the accumulated log(q1)-log(q2) estimate by lift over num_steps.
        target_rate_difference = -lift / (num_steps * d_sigma_f)

    degenerate = denominator.abs() <= eps
    safe_denominator = torch.where(
        degenerate,
        torch.ones_like(denominator),
        denominator,
    )
    raw_kappa = numerator / safe_denominator
    raw_kappa = torch.where(
        degenerate,
        torch.full_like(raw_kappa, 0.5),
        raw_kappa,
    )

    kappa = raw_kappa
    if kappa_min is not None:
        kappa = torch.clamp_min(kappa, kappa_min)
    if kappa_max is not None:
        kappa = torch.clamp_max(kappa, kappa_max)
    clamped = ~torch.isclose(kappa, raw_kappa, atol=1e-7, rtol=1e-6)

    expand_shape = (kappa.shape[0],) + (1,) * (v_1.ndim - 1)
    v_mix = v_base + guidance_scale * kappa.reshape(expand_shape) * delta
    u_applied = step_scale * v_mix
    rates = evaluate_density_rates(
        v_1_f,
        v_2_f,
        dlog_1_f,
        dlog_2_f,
        sigma=sigma_f,
        d_sigma=d_sigma_f,
        path_field=u_applied,
        target_rate_difference=target_rate_difference,
    )

    zeros = torch.zeros_like(norm_1_sq)
    return DensityKappaDiagnostics(
        raw_kappa=raw_kappa,
        kappa=kappa,
        denominator=denominator,
        field_difference_norm_sq=delta_norm_sq,
        degenerate=degenerate,
        clamped=clamped,
        density_rate_1=rates.density_rate_1,
        density_rate_2=rates.density_rate_2,
        target_rate_difference=target_rate_difference,
        density_rate_residual=rates.density_rate_residual,
        delta_log_q_1=rates.delta_log_q_1,
        delta_log_q_2=rates.delta_log_q_2,
        v_0_norm=(
            torch.sqrt(torch.sum(v_0_f.square(), dim=reduce_dims))
            if v_0_f is not None
            else zeros
        ),
        v_1_norm=torch.sqrt(norm_1_sq),
        v_2_norm=torch.sqrt(norm_2_sq),
    )
