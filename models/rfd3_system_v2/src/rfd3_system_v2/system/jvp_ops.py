"""Forward-AD-compatible decompositions used by rfd3_system_v2."""

from __future__ import annotations

import torch


def scatter_mean(
    zeros: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Compute scatter mean with primitives that implement forward-mode AD.

    This is algebraically equivalent to
    ``index_reduce(..., "mean", include_self=False)``. PyTorch does not provide
    a JVP for ``index_reduce`` in the Foundry container, while ``scatter_add``
    supports the required primal and tangent calculations.
    """

    ndim = source.dim()
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim - 1:
        raise ValueError("dim must select a non-final source dimension.")
    if index.ndim != 1 or source.shape[dim] != index.shape[0]:
        raise ValueError("index must be 1D and match source along dim.")

    index_shape = [1] * ndim
    index_shape[dim] = index.shape[0]
    expanded_index = index.view(index_shape).expand_as(source)
    result = zeros.scatter_add(dim, expanded_index, source)

    count_index = expanded_index[..., :1]
    ones = torch.ones_like(source[..., :1])
    count = torch.zeros(
        *zeros.shape[:-1],
        1,
        device=zeros.device,
        dtype=zeros.dtype,
    )
    count = count.scatter_add(dim, count_index, ones)
    return result / count.clamp_min(1)

