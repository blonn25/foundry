"""Sampler-level tests using a lightweight deterministic denoiser."""

from __future__ import annotations

import pytest
import torch

from rfd3_system_v3.model.inference_sampler import (
    SampleDiffusionReverseSDE,
    SampleDiffusionWithSuperDiffSharedChainSDE,
)


class _LinearDenoiserSampler(SampleDiffusionWithSuperDiffSharedChainSDE):
    """Replace the neural network with a context-specific linear denoiser."""

    def _denoise_once(self, *, X_noisy_L, f, **_):
        scale = float(f["test_denoise_scale"])
        logits = torch.zeros(
            X_noisy_L.shape[0],
            int(f["test_num_tokens"]),
            32,
            device=X_noisy_L.device,
        )
        return {
            "X_L": scale * X_noisy_L,
            "sequence_logits_I": logits,
            "sequence_indices_I": torch.zeros(
                X_noisy_L.shape[0],
                int(f["test_num_tokens"]),
                dtype=torch.long,
                device=X_noisy_L.device,
            ),
        }


def _track(*, atom_count: int, denoise_scale: float) -> dict:
    features = {
        "is_motif_atom_with_fixed_coord": torch.zeros(
            atom_count,
            dtype=torch.bool,
        ),
        "ref_element": torch.zeros(atom_count),
        "test_denoise_scale": denoise_scale,
        "test_num_tokens": atom_count,
    }
    return {
        "f": features,
        "initializer_outputs": {},
        "coord_atom_lvl_to_be_noised": torch.zeros(atom_count, 3),
    }


def test_coupled_sampler_keeps_one_shared_state_and_equalizes_raw_density():
    torch.manual_seed(7)
    sampler = _LinearDenoiserSampler(
        num_timesteps=5,
        sigma_data=1,
        s_max=1,
        s_min=0.1,
        p=1,
        kappa_min=None,
        kappa_max=None,
    )
    shared_1 = torch.tensor([0, 1], dtype=torch.long)
    shared_2 = torch.tensor([0, 1], dtype=torch.long)
    tokens = torch.tensor([0, 1], dtype=torch.long)

    result = sampler.sample_coupled_superdiff_sde(
        track_1=_track(atom_count=3, denoise_scale=0.2),
        track_2=_track(atom_count=4, denoise_scale=0.7),
        reference=None,
        shared_update_atom_indices_1=shared_1,
        shared_update_atom_indices_2=shared_2,
        shared_kappa_atom_indices_1=shared_1,
        shared_kappa_atom_indices_2=shared_2,
        shared_update_token_indices_1=tokens,
        shared_update_token_indices_2=tokens,
        reference_update_atom_indices=None,
        reference_kappa_atom_indices=None,
        reference_update_token_indices=None,
        diffusion_module=torch.nn.Identity(),
        diffusion_batch_size=2,
        coupling_metadata={"shared_chain_id": "A"},
    )

    assert torch.equal(
        result["track_1"]["X_L"][:, shared_1],
        result["track_2"]["X_L"][:, shared_2],
    )
    diagnostics = result["coupling_metadata"]["diagnostics"]
    raw_residual = torch.stack(diagnostics["raw_density_residual"])
    assert torch.max(torch.abs(raw_residual)) < 2e-4
    assert len(diagnostics["kappa"]) == 4


def test_strict_reverse_sde_rejects_native_step_scaling():
    sampler = SampleDiffusionReverseSDE(step_scale=1.5)
    with pytest.raises(ValueError, match="step_scale"):
        sampler._validate_strict_reverse_sde()
