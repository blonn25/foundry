import torch

from rfd3_system_v2.system.jvp_ops import scatter_mean


def test_scatter_mean_matches_index_reduce():
    source = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]],
            [[2.0, 1.0], [4.0, 3.0], [8.0, 5.0]],
        ]
    )
    index = torch.tensor([0, 1, 0])
    zeros = torch.zeros(2, 2, 2)

    expected = zeros.index_reduce(
        1,
        index,
        source,
        "mean",
        include_self=False,
    )
    observed = scatter_mean(zeros, 1, index, source)

    assert torch.allclose(observed, expected)


def test_scatter_mean_supports_forward_mode_jvp():
    source = torch.randn(2, 3, 4)
    tangent = torch.randn_like(source)
    index = torch.tensor([0, 1, 0])
    zeros = torch.zeros(2, 2, 4)

    primal, jvp = torch.func.jvp(
        lambda value: scatter_mean(zeros, 1, index, value),
        (source,),
        (tangent,),
    )

    assert primal.shape == (2, 2, 4)
    assert jvp.shape == primal.shape
    assert torch.isfinite(primal).all()
    assert torch.isfinite(jvp).all()
