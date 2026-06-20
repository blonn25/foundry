import torch

from rfd3_system.system.proxy import solve_two_track_proxy_kappa


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
