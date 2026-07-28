"""Focused tests for the reverse-SDE and stochastic SuperDiff algebra."""

from __future__ import annotations

import torch

from rfd3_system_v3.system.stochastic_control import (
    dot_by_sample,
    reverse_sde_increment,
    reverse_sde_noise_scale,
    solve_stochastic_superdiff_kappa,
)


def test_reverse_sde_euler_maruyama_terms():
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


def test_unclamped_kappa_equalizes_ito_density_changes():
    generator = torch.Generator().manual_seed(7)
    v_1 = torch.randn(4, 12, 3, generator=generator)
    v_2 = torch.randn(4, 12, 3, generator=generator)
    noise = torch.randn(4, 12, 3, generator=generator)

    _, diagnostics = solve_stochastic_superdiff_kappa(
        v_1,
        v_2,
        shared_noise=noise,
        sigma=3.0,
        d_sigma=-0.1,
        guidance_scale=1.0,
        num_steps=199,
        kappa_min=None,
        kappa_max=None,
    )

    assert not torch.any(diagnostics.degenerate)
    assert torch.max(torch.abs(diagnostics.raw_density_residual)) < 2e-5


def test_unit_guidance_matches_closed_form_kappa():
    generator = torch.Generator().manual_seed(11)
    v_1 = torch.randn(3, 8, 3, generator=generator)
    v_2 = torch.randn(3, 8, 3, generator=generator)
    noise = torch.randn(3, 8, 3, generator=generator)
    h = 0.2

    _, diagnostics = solve_stochastic_superdiff_kappa(
        v_1,
        v_2,
        shared_noise=noise,
        sigma=2.0,
        d_sigma=-h,
        guidance_scale=1.0,
        num_steps=20,
        kappa_min=None,
        kappa_max=None,
    )

    difference = v_1 - v_2
    expected = 0.5 + dot_by_sample(noise, difference) / (
        2.0 * h * dot_by_sample(difference, difference)
    )
    assert torch.allclose(diagnostics.raw_kappa, expected, atol=2e-6, rtol=2e-6)


def test_lift_changes_target_density_difference():
    generator = torch.Generator().manual_seed(13)
    v_1 = torch.randn(2, 6, 3, generator=generator)
    v_2 = torch.randn(2, 6, 3, generator=generator)
    noise = torch.randn(2, 6, 3, generator=generator)

    _, diagnostics = solve_stochastic_superdiff_kappa(
        v_1,
        v_2,
        shared_noise=noise,
        sigma=1.7,
        d_sigma=-0.05,
        guidance_scale=1.0,
        lift=0.4,
        num_steps=40,
        kappa_min=None,
        kappa_max=None,
    )

    assert torch.allclose(
        diagnostics.target_density_difference,
        torch.full((2,), -0.01),
    )
    assert torch.max(torch.abs(diagnostics.raw_density_residual)) < 2e-5


def test_clamping_reports_nonzero_applied_residual():
    v_1 = torch.ones(1, 4, 3)
    v_2 = torch.zeros_like(v_1)
    noise = torch.full_like(v_1, 20.0)

    _, diagnostics = solve_stochastic_superdiff_kappa(
        v_1,
        v_2,
        shared_noise=noise,
        sigma=2.0,
        d_sigma=-0.1,
        guidance_scale=1.0,
        num_steps=10,
        kappa_min=-1.0,
        kappa_max=2.0,
    )

    assert diagnostics.clamped.item()
    assert torch.abs(diagnostics.raw_density_residual).item() < 1e-5
    assert torch.abs(diagnostics.density_residual).item() > 1e-3


def test_degenerate_fields_use_half_weight():
    v = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(17))
    mixed, diagnostics = solve_stochastic_superdiff_kappa(
        v,
        v.clone(),
        shared_noise=torch.randn_like(v),
        sigma=1.0,
        d_sigma=-0.1,
        guidance_scale=1.0,
        num_steps=10,
    )

    assert torch.all(diagnostics.degenerate)
    assert torch.allclose(diagnostics.kappa, torch.full((2,), 0.5))
    assert torch.allclose(mixed, v)


def test_guided_solve_requires_reference_field():
    v_1 = torch.ones(1, 2, 3)
    v_2 = torch.zeros_like(v_1)

    try:
        solve_stochastic_superdiff_kappa(
            v_1,
            v_2,
            shared_noise=torch.zeros_like(v_1),
            sigma=1.0,
            d_sigma=-0.1,
            guidance_scale=1.5,
            num_steps=10,
        )
    except ValueError as error:
        assert "v_0 is required" in str(error)
    else:
        raise AssertionError("Expected guidance without a reference field to fail.")


def test_reverse_sde_recovers_gaussian_variance():
    """An exact Gaussian denoiser should approximately recover data variance."""

    generator = torch.Generator().manual_seed(23)
    data_sigma = 1.0
    sigma_max = 8.0
    sigma_min = 0.01
    schedule = torch.linspace(sigma_max, sigma_min, 800)
    samples = (
        torch.sqrt(torch.tensor(data_sigma**2 + sigma_max**2))
        * torch.randn(8192, 1, generator=generator)
    )

    for sigma, sigma_next in zip(schedule, schedule[1:]):
        denoised = data_sigma**2 / (data_sigma**2 + sigma**2) * samples
        velocity = (samples - denoised) / sigma
        d_sigma = sigma_next - sigma
        noise = reverse_sde_noise_scale(sigma, d_sigma) * torch.randn(
            samples.shape,
            generator=generator,
        )
        samples = samples + reverse_sde_increment(
            velocity,
            d_sigma=d_sigma,
            noise=noise,
        )

    expected_variance = data_sigma**2 + sigma_min**2
    observed_variance = torch.var(samples, unbiased=True)
    assert torch.isclose(
        observed_variance,
        torch.tensor(expected_variance),
        rtol=0.08,
        atol=0.04,
    )
