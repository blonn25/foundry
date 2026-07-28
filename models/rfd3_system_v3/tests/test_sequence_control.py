"""Tests for kappa-weighted shared-A sequence output handling."""

from __future__ import annotations

import numpy as np
import torch

from rfd3_system_v3.system.chains import (
    SharedAtomMap,
    shared_update_token_indices,
)
from rfd3_system_v3.system.sequence_control import mix_shared_sequence_logits


def test_shared_token_map_excludes_sequence_fixed_residues():
    shared_map = SharedAtomMap(
        update_indices_1=np.array([0, 1, 2, 3]),
        update_indices_2=np.array([1, 2, 3, 4]),
        update_residues=[
            {"res_id": 1, "res_name": "ALA", "atom_count": 2},
            {"res_id": 2, "res_name": "GLY", "atom_count": 2},
        ],
        excluded_fixed_residues=[],
    )
    token_map_1 = np.array([0, 0, 1, 1])
    token_map_2 = np.array([9, 0, 0, 1, 1])
    fixed_1 = np.array([False, False, True, True])
    fixed_2 = np.array([False, False, False, False, False])

    tokens_1, tokens_2 = shared_update_token_indices(
        shared_map,
        token_map_1,
        token_map_2,
        fixed_1,
        fixed_2,
    )

    assert tokens_1.tolist() == [0]
    assert tokens_2.tolist() == [0]


def test_unit_guidance_kappa_mixes_only_selected_tokens():
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
    assert torch.allclose(output_1[:, 0, :], logits_1[:, 0, :])
    assert torch.allclose(output_2[:, 0, :], logits_2[:, 0, :])


def test_guided_logit_mix_uses_isolated_reference():
    logits_1 = torch.tensor([[[5.0, 1.0]]])
    logits_2 = torch.tensor([[[3.0, 2.0]]])
    logits_0 = torch.tensor([[[1.0, 4.0]]])

    output_1, output_2 = mix_shared_sequence_logits(
        logits_1,
        logits_2,
        token_indices_1=torch.tensor([0]),
        token_indices_2=torch.tensor([0]),
        kappa=torch.tensor([0.5]),
        guidance_scale=2.0,
        logits_0=logits_0,
        token_indices_0=torch.tensor([0]),
    )
    expected = logits_0 + 2.0 * (
        (logits_2 - logits_0) + 0.5 * (logits_1 - logits_2)
    )
    assert torch.allclose(output_1, expected)
    assert torch.allclose(output_2, expected)
