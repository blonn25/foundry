import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "compute_rfd3_flow_likelihood.py"
)
SPEC = importlib.util.spec_from_file_location("rfd3_flow_likelihood", SCRIPT)
flow = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = flow
SPEC.loader.exec_module(flow)


def test_karras_schedule_has_requested_endpoints_and_positive_derivative():
    sigma_0, derivative_0 = flow.karras_sigma(
        0.0, sigma_min=0.1, sigma_max=10.0, power=7.0
    )
    sigma_1, derivative_1 = flow.karras_sigma(
        1.0, sigma_min=0.1, sigma_max=10.0, power=7.0
    )

    assert float(sigma_0) == pytest.approx(0.1)
    assert float(sigma_1) == pytest.approx(10.0)
    assert float(derivative_0) > 0
    assert float(derivative_1) > 0


def test_hutchinson_divergence_is_exact_for_diagonal_linear_field():
    diagonal = torch.tensor([1.0, -2.0, 4.0, 0.5]).reshape(1, 2, 2)
    coordinates = torch.ones_like(diagonal)
    probes = flow.make_rademacher_probes(
        coordinates.shape,
        count=3,
        seed=7,
        device=coordinates.device,
    )

    field, divergence = flow.field_and_hutchinson_divergence(
        lambda value: diagonal * value,
        coordinates,
        probes,
    )

    assert torch.equal(field, diagonal)
    assert float(divergence) == pytest.approx(float(diagonal.sum()))


def test_rk4_matches_linear_probability_flow_and_density_correction():
    initial = torch.tensor([[[0.5, -1.0, 2.0]]])
    sigma_min = 0.1
    sigma_max = 0.8
    rate = 0.25
    probes = [torch.ones_like(initial)]

    # D(x, sigma) = x - sigma*a*x makes (x-D)/sigma = a*x.
    def denoiser(coordinates, sigma):
        return coordinates - sigma * rate * coordinates

    terminal, correction, trace = flow.integrate_probability_flow_rk4(
        initial,
        denoiser_fn=denoiser,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        schedule_power=3.0,
        intervals=20,
        probes=probes,
    )

    expected_terminal = initial * math.exp(rate * (sigma_max - sigma_min))
    expected_correction = initial.numel() * rate * (sigma_max - sigma_min)
    assert torch.allclose(terminal, expected_terminal, atol=2e-6, rtol=2e-6)
    assert correction == pytest.approx(expected_correction, abs=2e-6)
    assert len(trace) == 20
    assert trace[-1]["sigma_end"] == pytest.approx(sigma_max)


def test_gaussian_log_probability_uses_every_scalar_coordinate():
    coordinates = torch.zeros(1, 2, 3)
    sigma = 2.0
    expected = -0.5 * coordinates.numel() * math.log(2 * math.pi * sigma**2)

    assert flow.gaussian_log_probability(coordinates, sigma) == pytest.approx(
        expected
    )


def test_config_requires_explicit_sequence_policy_and_fixed_selection(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "checkpoint_path: /weights/rfd3_latest.ckpt\n"
        "specification: {}\n"
        "likelihood: {}\n"
    )

    with pytest.raises(ValueError, match="select_fixed_atoms"):
        flow.load_config(config)

    config.write_text(
        "checkpoint_path: /weights/rfd3_latest.ckpt\n"
        "specification:\n  select_fixed_atoms: false\n"
        "likelihood: {}\n"
    )
    with pytest.raises(ValueError, match="sequence_conditioning"):
        flow.load_config(config)


def test_config_translates_sequence_policy_and_rejects_coordinate_rebuilding(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "checkpoint_path: /weights/rfd3_latest.ckpt\n"
        "specification:\n"
        "  select_fixed_atoms: {L1: ALL}\n"
        "likelihood:\n"
        "  sequence_conditioning: masked\n"
    )
    _, specification, settings = flow.load_config(config)
    assert settings.sequence_conditioning == "masked"
    assert specification["select_unfixed_sequence"] is True

    config.write_text(
        "checkpoint_path: /weights/rfd3_latest.ckpt\n"
        "specification:\n"
        "  contig: A1-10\n"
        "  select_fixed_atoms: false\n"
        "likelihood:\n"
        "  sequence_conditioning: observed\n"
    )
    with pytest.raises(ValueError, match="contig"):
        flow.load_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("integration_intervals", 0),
        ("hutchinson_probes", 0),
        ("sigma_min", -1.0),
        ("sigma_max", float("inf")),
    ],
)
def test_invalid_numerical_settings_are_rejected(field, value):
    kwargs = {"sequence_conditioning": "masked", field: value}
    with pytest.raises(ValueError):
        flow.validate_settings(flow.LikelihoodSettings(**kwargs))


def test_fixed_probe_seed_is_deterministic():
    shape = torch.Size((1, 8, 3))
    first = flow.make_rademacher_probes(
        shape, count=5, seed=123, device=torch.device("cpu")
    )
    second = flow.make_rademacher_probes(
        shape, count=5, seed=123, device=torch.device("cpu")
    )

    assert all(torch.equal(a, b) for a, b in zip(first, second))
