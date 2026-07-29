import torch
import pytest

from rfd3_system.system.proxy import (
    cosine_similarity_by_sample,
    solve_two_track_proxy_kappa,
)


def test_proxy_kappa_keeps_matching_updates_balanced():
    delta = torch.ones(2, 4, 3)

    diagnostics = solve_two_track_proxy_kappa(delta, delta)

    assert torch.allclose(diagnostics.kappa, torch.full((2,), 0.5))
    assert torch.all(diagnostics.degenerate)
    assert torch.allclose(diagnostics.proxy_residual, torch.zeros(2))


def test_proxy_kappa_equalizes_documented_proxy_equation():
    delta_1 = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]])
    delta_2 = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.25, 0.0]]])

    diagnostics = solve_two_track_proxy_kappa(
        delta_1,
        delta_2,
        kappa_min=-10.0,
        kappa_max=10.0,
    )
    kappa = diagnostics.kappa.reshape(1, 1, 1)
    mixed = kappa * delta_1 + (1 - kappa) * delta_2
    proxy_1 = torch.sum(mixed * delta_1) - torch.sum(delta_1.square())
    proxy_2 = torch.sum(mixed * delta_2) - torch.sum(delta_2.square())

    assert torch.allclose(proxy_1, proxy_2, atol=1e-6)
    assert torch.allclose(diagnostics.proxy_residual, torch.zeros(1), atol=1e-6)


def test_proxy_kappa_clamps_extreme_weights():
    delta_1 = torch.tensor([[[10.0, 0.0, 0.0]]])
    delta_2 = torch.tensor([[[1.0, 0.0, 0.0]]])

    diagnostics = solve_two_track_proxy_kappa(
        delta_1,
        delta_2,
        kappa_min=0.0,
        kappa_max=1.0,
    )

    assert diagnostics.raw_kappa.item() > 1.0
    assert diagnostics.kappa.item() == 1.0


def test_zero_regularization_reproduces_legacy_kappa():
    delta_1 = torch.tensor([[[1.01, 0.0, 0.0]]])
    delta_2 = torch.tensor([[[1.00, 0.0, 0.0]]])

    diagnostics = solve_two_track_proxy_kappa(
        delta_1,
        delta_2,
        kappa_min=-1_000.0,
        kappa_max=1_000.0,
        regularization_rho=0.0,
    )

    assert torch.allclose(diagnostics.regularized_kappa, diagnostics.raw_kappa)
    assert torch.allclose(diagnostics.kappa, diagnostics.raw_kappa)
    assert torch.allclose(diagnostics.reliability, torch.ones(1))


def test_regularization_shrinks_ill_conditioned_kappa_toward_half():
    delta_1 = torch.tensor([[[1.01, 0.0, 0.0]]])
    delta_2 = torch.tensor([[[1.00, 0.0, 0.0]]])

    weak = solve_two_track_proxy_kappa(
        delta_1,
        delta_2,
        kappa_min=-1_000.0,
        kappa_max=1_000.0,
        regularization_rho=1e-3,
    )
    strong = solve_two_track_proxy_kappa(
        delta_1,
        delta_2,
        kappa_min=-1_000.0,
        kappa_max=1_000.0,
        regularization_rho=1e-1,
    )

    weak_distance = torch.abs(weak.regularized_kappa - 0.5)
    strong_distance = torch.abs(strong.regularized_kappa - 0.5)
    assert torch.all(strong_distance < weak_distance)
    assert torch.all(weak.regularized_kappa < weak.raw_kappa)
    assert torch.all(strong.reliability < weak.reliability)
    assert torch.all(torch.abs(strong.regularized_proxy_residual) > 0)


def test_regularization_is_invariant_to_common_delta_rescaling():
    delta_1 = torch.tensor([[[1.01, 0.0, 0.0], [0.0, 0.5, 0.0]]])
    delta_2 = torch.tensor([[[1.00, 0.0, 0.0], [0.0, 0.4, 0.0]]])

    diagnostics = solve_two_track_proxy_kappa(
        delta_1,
        delta_2,
        regularization_rho=1e-4,
        kappa_min=-1_000.0,
        kappa_max=1_000.0,
    )
    scaled = solve_two_track_proxy_kappa(
        100.0 * delta_1,
        100.0 * delta_2,
        regularization_rho=1e-4,
        kappa_min=-1_000.0,
        kappa_max=1_000.0,
    )

    assert torch.allclose(
        diagnostics.regularized_kappa,
        scaled.regularized_kappa,
        atol=1e-5,
    )
    assert torch.allclose(
        diagnostics.reliability,
        scaled.reliability,
        atol=1e-6,
    )
    assert torch.allclose(
        diagnostics.relative_denominator,
        scaled.relative_denominator,
        atol=1e-6,
    )


def test_regularization_is_applied_before_clamping():
    delta_1 = torch.tensor([[[1.01, 0.0, 0.0]]])
    delta_2 = torch.tensor([[[1.00, 0.0, 0.0]]])

    diagnostics = solve_two_track_proxy_kappa(
        delta_1,
        delta_2,
        kappa_min=-1.0,
        kappa_max=2.0,
        regularization_rho=1e-1,
    )

    assert diagnostics.raw_kappa.item() > 2.0
    assert -1.0 < diagnostics.regularized_kappa.item() < 2.0
    assert torch.allclose(diagnostics.kappa, diagnostics.regularized_kappa)


@pytest.mark.parametrize("rho", [-1.0, float("nan"), float("inf")])
def test_proxy_kappa_rejects_invalid_regularization(rho):
    with pytest.raises(ValueError, match="finite and non-negative"):
        solve_two_track_proxy_kappa(
            torch.ones(1, 2, 3),
            torch.zeros(1, 2, 3),
            regularization_rho=rho,
        )


def test_subset_kappa_can_mix_full_shared_update_tensor():
    delta_1_all = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 2.0, 0.0]]]
    )
    delta_2_all = torch.tensor(
        [[[0.0, 1.0, 0.0], [0.25, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    subset_indices = torch.tensor([1])

    diagnostics = solve_two_track_proxy_kappa(
        delta_1_all[:, subset_indices, :],
        delta_2_all[:, subset_indices, :],
        kappa_min=-10.0,
        kappa_max=10.0,
    )
    kappa = diagnostics.kappa.reshape(1, 1, 1)
    mixed_all = kappa * delta_1_all + (1 - kappa) * delta_2_all

    assert mixed_all.shape == delta_1_all.shape
    assert diagnostics.delta_1_norm.shape == torch.Size([1])


def test_cosine_similarity_by_sample_handles_batched_updates():
    delta_a = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    delta_b = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )

    cosine = cosine_similarity_by_sample(delta_a, delta_b)

    assert cosine.shape == torch.Size([2])
    assert torch.allclose(cosine, torch.tensor([1.0, 0.0]), atol=1e-6)


def test_cosine_similarity_by_sample_returns_finite_value_for_zero_norm():
    delta_a = torch.zeros(2, 3, 3)
    delta_b = torch.ones(2, 3, 3)

    cosine = cosine_similarity_by_sample(delta_a, delta_b)

    assert torch.all(torch.isfinite(cosine))
    assert torch.allclose(cosine, torch.zeros(2))


def test_cosine_similarity_by_sample_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        cosine_similarity_by_sample(torch.zeros(1, 2, 3), torch.zeros(1, 3, 3))
