"""Shared-residue sequence-logit handling for coupled RFD3 outputs."""

from __future__ import annotations

import torch


def mix_shared_sequence_logits(
    logits_1: torch.Tensor,
    logits_2: torch.Tensor,
    *,
    token_indices_1: torch.Tensor,
    token_indices_2: torch.Tensor,
    kappa: torch.Tensor,
    guidance_scale: float,
    logits_0: torch.Tensor | None = None,
    token_indices_0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix mapped shared-chain logits with the coordinate solver's kappa.

    This is an output-coordination heuristic, not a sequence diffusion.  It
    deliberately touches only mapped, movable shared residues so fixed or
    chemically different motif residues retain their track-specific identities.
    """

    if logits_1.ndim != 3 or logits_2.ndim != 3:
        raise ValueError("Sequence logits must have shape (batch, tokens, classes).")
    if logits_1.shape[0] != logits_2.shape[0] or logits_1.shape[2] != logits_2.shape[2]:
        raise ValueError("Track sequence-logit batch and class dimensions must match.")
    if guidance_scale <= 0:
        raise ValueError("superdiff_guidance_scale must be greater than zero.")

    token_indices_1 = token_indices_1.to(device=logits_1.device, dtype=torch.long)
    token_indices_2 = token_indices_2.to(device=logits_2.device, dtype=torch.long)
    if token_indices_1.ndim != 1 or token_indices_2.ndim != 1:
        raise ValueError("Shared token indices must be one-dimensional.")
    if token_indices_1.shape != token_indices_2.shape:
        raise ValueError("Paired shared token-index arrays must have matching shapes.")
    if token_indices_1.numel() == 0:
        raise ValueError("At least one movable shared residue is required.")

    selected_1 = logits_1[:, token_indices_1, :].float()
    selected_2 = logits_2[:, token_indices_2, :].float()
    if selected_1.shape != selected_2.shape:
        raise ValueError("Mapped shared sequence-logit tensors must have matching shapes.")

    if kappa.shape != (logits_1.shape[0],):
        raise ValueError(
            f"kappa must have shape ({logits_1.shape[0]},), got {tuple(kappa.shape)}."
        )
    kappa_view = kappa.float().reshape(-1, 1, 1)

    if guidance_scale == 1.0:
        base = selected_2
    else:
        if logits_0 is None or token_indices_0 is None:
            raise ValueError(
                "Reference logits and token indices are required when guidance != 1."
            )
        token_indices_0 = token_indices_0.to(
            device=logits_0.device,
            dtype=torch.long,
        )
        selected_0 = logits_0[:, token_indices_0, :].float()
        if selected_0.shape != selected_2.shape:
            raise ValueError(
                "Mapped isolated-reference sequence logits must match track logits."
            )
        base = selected_0 + guidance_scale * (selected_2 - selected_0)

    mixed = base + guidance_scale * kappa_view * (selected_1 - selected_2)
    output_1 = logits_1.clone()
    output_2 = logits_2.clone()
    output_1[:, token_indices_1, :] = mixed.to(dtype=output_1.dtype)
    output_2[:, token_indices_2, :] = mixed.to(dtype=output_2.dtype)
    return output_1, output_2
