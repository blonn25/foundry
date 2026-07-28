#!/usr/bin/env python3
"""Dependency-free validation for rfd3_system_v3 inside foundry.sif.

The production image does not include pytest. This script therefore uses
ordinary assertions while importing the real Foundry/RFD3 integration modules.
It is intended for lightweight SLURM validation before GPU smoke inference.
"""

from __future__ import annotations

import torch

from rfd3_system_v3.model.inference_sampler import (
    SampleDiffusionReverseSDE,
    SampleDiffusionWithSuperDiffSharedChainSDE,
)
from rfd3_system_v3.system.sequence_control import mix_shared_sequence_logits
from rfd3_system_v3.system.stochastic_control import (
    dot_by_sample,
    reverse_sde_increment,
    reverse_sde_noise_scale,
    solve_stochastic_superdiff_kappa,
)


class _LinearDenoiserSampler(SampleDiffusionWithSuperDiffSharedChainSDE):
    """Use a context-specific linear denoiser to test the complete rollout."""

    def _denoise_once(self, *, X_noisy_L, f, **_):
        logits = torch.zeros(
            X_noisy_L.shape[0],
            int(f["test_num_tokens"]),
            32,
            device=X_noisy_L.device,
        )
        return {
            "X_L": float(f["test_denoise_scale"]) * X_noisy_L,
            "sequence_logits_I": logits,
            "sequence_indices_I": torch.zeros(
                X_noisy_L.shape[0],
                int(f["test_num_tokens"]),
                dtype=torch.long,
                device=X_noisy_L.device,
            ),
        }


def _track(*, atom_count: int, denoise_scale: float) -> dict:
    return {
        "f": {
            "is_motif_atom_with_fixed_coord": torch.zeros(
                atom_count,
                dtype=torch.bool,
            ),
            "ref_element": torch.zeros(atom_count),
            "test_denoise_scale": denoise_scale,
            "test_num_tokens": atom_count,
        },
        "initializer_outputs": {},
        "coord_atom_lvl_to_be_noised": torch.zeros(atom_count, 3),
    }


def _validate_reverse_sde_terms() -> None:
    velocity = torch.tensor([[[1.0, -2.0, 0.5]]])
    noise = torch.tensor([[[0.2, 0.3, -0.1]]])
    scale = reverse_sde_noise_scale(4.0, -0.25)
    increment = reverse_sde_increment(
        velocity,
        d_sigma=-0.25,
        noise=noise,
    )
    assert torch.allclose(scale, torch.sqrt(torch.tensor(2.0)))
    assert torch.allclose(increment, -0.5 * velocity + noise)


def _validate_stochastic_kappa() -> None:
    generator = torch.Generator().manual_seed(7)
    v_1 = torch.randn(4, 12, 3, generator=generator)
    v_2 = torch.randn(4, 12, 3, generator=generator)
    noise = torch.randn(4, 12, 3, generator=generator)
    h = 0.1
    _, diagnostics = solve_stochastic_superdiff_kappa(
        v_1,
        v_2,
        shared_noise=noise,
        sigma=3.0,
        d_sigma=-h,
        guidance_scale=1.0,
        num_steps=199,
        kappa_min=None,
        kappa_max=None,
    )
    difference = v_1 - v_2
    expected = 0.5 + dot_by_sample(noise, difference) / (
        2.0 * h * dot_by_sample(difference, difference)
    )
    assert torch.allclose(diagnostics.raw_kappa, expected, atol=2e-6, rtol=2e-6)
    assert torch.max(torch.abs(diagnostics.raw_density_residual)) < 2e-5


def _validate_sequence_mixing() -> None:
    logits_1 = torch.zeros(2, 4, 3)
    logits_2 = torch.zeros(2, 5, 3)
    logits_1[:, 1, :] = torch.tensor([2.0, 4.0, 6.0])
    logits_2[:, 2, :] = torch.tensor([10.0, 20.0, 30.0])
    kappa = torch.tensor([0.25, 1.5])
    output_1, output_2 = mix_shared_sequence_logits(
        logits_1,
        logits_2,
        token_indices_1=torch.tensor([1]),
        token_indices_2=torch.tensor([2]),
        kappa=kappa,
        guidance_scale=1.0,
    )
    expected = logits_2[:, 2, :] + kappa[:, None] * (
        logits_1[:, 1, :] - logits_2[:, 2, :]
    )
    assert torch.allclose(output_1[:, 1, :], expected)
    assert torch.allclose(output_2[:, 2, :], expected)


def _validate_shared_rollout() -> None:
    torch.manual_seed(11)
    sampler = _LinearDenoiserSampler(
        num_timesteps=5,
        sigma_data=1,
        s_max=1,
        s_min=0.1,
        p=1,
        kappa_min=None,
        kappa_max=None,
    )
    shared = torch.tensor([0, 1], dtype=torch.long)
    result = sampler.sample_coupled_superdiff_sde(
        track_1=_track(atom_count=3, denoise_scale=0.2),
        track_2=_track(atom_count=4, denoise_scale=0.7),
        reference=None,
        shared_update_atom_indices_1=shared,
        shared_update_atom_indices_2=shared,
        shared_kappa_atom_indices_1=shared,
        shared_kappa_atom_indices_2=shared,
        shared_update_token_indices_1=shared,
        shared_update_token_indices_2=shared,
        reference_update_atom_indices=None,
        reference_kappa_atom_indices=None,
        reference_update_token_indices=None,
        diffusion_module=torch.nn.Identity(),
        diffusion_batch_size=2,
        coupling_metadata={"shared_chain_id": "A"},
    )
    assert torch.equal(
        result["track_1"]["X_L"][:, shared],
        result["track_2"]["X_L"][:, shared],
    )
    raw_residual = torch.stack(
        result["coupling_metadata"]["diagnostics"]["raw_density_residual"]
    )
    assert torch.max(torch.abs(raw_residual)) < 2e-4


def _validate_strict_guards() -> None:
    sampler = SampleDiffusionReverseSDE(step_scale=1.5)
    try:
        sampler._validate_strict_reverse_sde()
    except ValueError as error:
        assert "step_scale" in str(error)
    else:
        raise AssertionError("Strict SDE accepted a non-unit step_scale.")


def main() -> int:
    checks = (
        _validate_reverse_sde_terms,
        _validate_stochastic_kappa,
        _validate_sequence_mixing,
        _validate_shared_rollout,
        _validate_strict_guards,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"rfd3_system_v3 validation passed: {len(checks)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
