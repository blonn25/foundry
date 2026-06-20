"""Chain-level helpers for the approximate shared-chain prototype."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from biotite.structure import AtomArray


def normalize_chain_ids(chain_ids: str | Sequence[str]) -> list[str]:
    """Normalize a chain-id argument into a non-empty list of strings."""

    if isinstance(chain_ids, str):
        chain_ids = [chain_ids]
    normalized = [str(chain_id) for chain_id in chain_ids]
    if not normalized:
        raise ValueError("At least one chain ID must be provided.")
    return normalized


def chain_mask(atom_array: AtomArray, chain_ids: str | Sequence[str]) -> np.ndarray:
    """Return a boolean atom mask for one or more chain IDs."""

    return np.isin(atom_array.chain_id, normalize_chain_ids(chain_ids))


def subset_by_chains(atom_array: AtomArray, chain_ids: Sequence[str]) -> AtomArray:
    """Return a copy containing only the requested chains."""

    mask = chain_mask(atom_array, chain_ids)
    missing = sorted(set(chain_ids) - set(np.unique(atom_array.chain_id[mask])))
    if missing:
        raise ValueError(f"Input atom array does not contain chain(s): {missing}")
    subset = atom_array[mask].copy()
    subset.bonds = None
    return subset


def shared_chain_mask(atom_array: AtomArray, shared_chain_id: str) -> np.ndarray:
    """Return and validate the atom mask for the shared chain."""

    mask = chain_mask(atom_array, shared_chain_id)
    if not np.any(mask):
        raise ValueError(f"Shared chain {shared_chain_id!r} is absent from track.")
    return mask


def shared_chain_signature(atom_array: AtomArray, shared_chain_id: str) -> list[tuple]:
    """Build an order-sensitive atom signature for the shared chain."""

    atoms = atom_array[shared_chain_mask(atom_array, shared_chain_id)]
    signature = []
    for idx in range(atoms.array_length()):
        signature.append(
            (
                str(atoms.chain_id[idx]),
                int(atoms.res_id[idx]),
                str(atoms.res_name[idx]),
                str(atoms.atom_name[idx]),
            )
        )
    return signature


def assert_matching_shared_chain(
    track_1_atom_array: AtomArray,
    track_2_atom_array: AtomArray,
    shared_chain_id: str,
) -> None:
    """Fail if two track views do not contain the same shared-chain atoms."""

    signature_1 = shared_chain_signature(track_1_atom_array, shared_chain_id)
    signature_2 = shared_chain_signature(track_2_atom_array, shared_chain_id)
    if signature_1 != signature_2:
        raise ValueError(
            "Shared-chain atom order differs between tracks. The approximate "
            "coupler requires identical shared-chain residue/atom ordering."
        )


def append_nonshared_from_track_2(
    track_1_atom_array: AtomArray,
    track_2_atom_array: AtomArray,
    shared_chain_id: str,
) -> AtomArray:
    """Merge track-1 A+B with track-2 non-shared partner atoms."""

    nonshared_2 = track_2_atom_array[~shared_chain_mask(track_2_atom_array, shared_chain_id)]
    merged = track_1_atom_array.copy() + nonshared_2.copy()
    merged.bonds = None
    return merged
