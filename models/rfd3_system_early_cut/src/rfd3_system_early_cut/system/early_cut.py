"""Release timing and state diagnostics; no random draws or state mutation."""

from __future__ import annotations

import math

import torch


def validate_cutoff(fraction: float | None, sigma: float | None) -> None:
    if fraction is not None and sigma is not None:
        raise ValueError("Specify only one of coupling_cut_fraction and coupling_cut_sigma.")
    for name, value in (("coupling_cut_fraction", fraction), ("coupling_cut_sigma", sigma)):
        if value is not None and (
            isinstance(value, bool) or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"{name} must be a finite nonnegative number.")
    if fraction is not None and fraction > 1:
        raise ValueError("coupling_cut_fraction must be between 0 and 1.")


def resolve_cutoff(noise_schedule, fraction=None, sigma=None) -> dict:
    """Resolve against pre-churn sigma at each executed update, never terminal sigma."""
    validate_cutoff(fraction, sigma)
    values = [float(value) for value in noise_schedule]
    n = len(values) - 1
    if n < 1:
        raise ValueError("A coupling cutoff requires at least one denoising update.")
    k = n
    if fraction is not None:
        k = math.floor(fraction * n)
    elif sigma is not None:
        k = next((i for i, value in enumerate(values[:-1]) if value <= sigma), n)
    return {
        "requested_fraction": fraction,
        "requested_sigma": sigma,
        "sigma_basis": "pre_churn",
        "executed_update_count": n,
        "coupled_update_count": k,
        "first_independent_update_index": k if k < n else None,
        "release_pre_churn_sigma": values[k] if k < n else None,
        "release_t_hat": None,
    }


def state_ca_rmsd(x1, x2, indices_1, indices_2):
    """Unaligned C-alpha RMSD (Angstrom), one value per paired sample.

    Cast only diagnostic views to FP32: changing the sampler's coordinate
    dtype would change the trajectory and break parity with the original.
    """
    difference = x1[:, indices_1, :].float() - x2[:, indices_2, :].float()
    return difference.square().sum(dim=-1).mean(dim=-1).sqrt().detach().cpu()
