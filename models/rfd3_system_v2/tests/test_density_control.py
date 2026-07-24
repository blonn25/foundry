import pytest
import torch

from rfd3_system_v2.system.density_control import (
    build_guided_base_field,
    evaluate_density_rates,
    solve_deterministic_density_kappa,
)


def solve(v1, v2, dlog1, dlog2, **kwargs):
    return solve_deterministic_density_kappa(
        v1,
        v2,
        dlog1,
        dlog2,
        sigma=kwargs.pop("sigma", 2.0),
        d_sigma=kwargs.pop("d_sigma", -0.1),
        step_scale=kwargs.pop("step_scale", 1.0),
        guidance_scale=kwargs.pop("guidance_scale", 1.0),
        num_steps=kwargs.pop("num_steps", 10),
        kappa_min=kwargs.pop("kappa_min", None),
        kappa_max=kwargs.pop("kappa_max", None),
        **kwargs,
    )


def test_unclamped_kappa_equalizes_density_rates():
    v1 = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]])
    v2 = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.25, 0.0]]])
    result = solve(v1, v2, torch.tensor([0.3]), torch.tensor([-0.2]))

    assert torch.allclose(
        result.density_rate_1,
        result.density_rate_2,
        atol=1e-6,
    )
    assert torch.allclose(result.density_rate_residual, torch.zeros(1), atol=1e-6)


def test_unit_guidance_does_not_require_reference_field():
    v1 = torch.tensor([[[1.0, 0.0, 0.0]]])
    v2 = torch.tensor([[[0.0, 1.0, 0.0]]])
    result = solve(v1, v2, torch.zeros(1), torch.zeros(1))

    assert torch.isfinite(result.kappa).all()
    assert result.v_0_norm.item() == 0.0


def test_nonunit_guidance_uses_reference_field():
    v1 = torch.tensor([[[1.0, 0.0, 0.0]]])
    v2 = torch.tensor([[[0.0, 1.0, 0.0]]])
    v0 = torch.tensor([[[0.25, 0.25, 0.0]]])
    result = solve(
        v1,
        v2,
        torch.zeros(1),
        torch.zeros(1),
        guidance_scale=1.5,
        v_0=v0,
    )

    assert result.v_0_norm.item() > 0
    assert torch.allclose(result.density_rate_residual, torch.zeros(1), atol=1e-6)


def test_zero_guidance_is_rejected():
    with pytest.raises(ValueError, match="greater than zero"):
        build_guided_base_field(
            torch.zeros(1, 1, 3),
            guidance_scale=0.0,
            v_0=torch.zeros(1, 1, 3),
        )


def test_kappa_clamp_reports_post_clamp_residual():
    v1 = torch.tensor([[[10.0, 0.0, 0.0]]])
    v2 = torch.tensor([[[1.0, 0.0, 0.0]]])
    result = solve(
        v1,
        v2,
        torch.zeros(1),
        torch.zeros(1),
        kappa_min=0.0,
        kappa_max=1.0,
    )

    assert result.raw_kappa.item() > 1.0
    assert result.kappa.item() == 1.0
    assert result.clamped.item()
    assert result.density_rate_residual.abs().item() > 0


def test_matching_fields_use_degenerate_half_mix():
    value = torch.ones(2, 4, 3)
    result = solve(value, value, torch.zeros(2), torch.zeros(2))

    assert torch.all(result.degenerate)
    assert torch.allclose(result.kappa, torch.full((2,), 0.5))


def test_lift_matches_documented_per_step_density_difference():
    v1 = torch.tensor([[[1.0, 0.0, 0.0]]])
    v2 = torch.tensor([[[0.0, 1.0, 0.0]]])
    lift = 0.4
    num_steps = 20
    d_sigma = -0.2
    result = solve(
        v1,
        v2,
        torch.zeros(1),
        torch.zeros(1),
        lift=lift,
        num_steps=num_steps,
        d_sigma=d_sigma,
    )

    expected_rate = torch.tensor([-lift / (num_steps * d_sigma)])
    assert torch.allclose(result.target_rate_difference, expected_rate)
    assert torch.allclose(result.density_rate_residual, torch.zeros(1), atol=1e-6)
    observed_increment_difference = result.delta_log_q_1 - result.delta_log_q_2
    assert torch.allclose(
        observed_increment_difference,
        torch.tensor([-lift / num_steps]),
        atol=1e-6,
    )


def test_step_scale_is_part_of_the_solve():
    v1 = torch.tensor([[[1.0, 0.0, 0.0]]])
    v2 = torch.tensor([[[0.0, 1.0, 0.0]]])
    dlog1 = torch.tensor([0.7])
    dlog2 = torch.tensor([-0.1])

    result = solve(v1, v2, dlog1, dlog2, step_scale=3.0)

    assert torch.allclose(result.density_rate_residual, torch.zeros(1), atol=1e-6)


def test_independent_density_rate_evaluation_uses_supplied_path():
    v1 = torch.tensor([[[1.0, 0.0, 0.0]]])
    v2 = torch.tensor([[[0.0, 1.0, 0.0]]])
    path = torch.tensor([[[0.5, 0.5, 0.0]]])

    result = evaluate_density_rates(
        v1,
        v2,
        torch.tensor([0.25]),
        torch.tensor([0.25]),
        sigma=2.0,
        d_sigma=-0.1,
        path_field=path,
    )

    assert torch.allclose(result.density_rate_residual, torch.zeros(1))
    assert torch.allclose(
        result.delta_log_q_1,
        -0.1 * result.density_rate_1,
    )
