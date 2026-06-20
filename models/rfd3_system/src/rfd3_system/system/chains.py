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


def ordered_chain_ids(atom_array: AtomArray) -> list[str]:
    """Return chain IDs in first-appearance order."""

    ordered = []
    seen = set()
    for chain_id in atom_array.chain_id.astype(str):
        if chain_id not in seen:
            ordered.append(chain_id)
            seen.add(chain_id)
    return ordered


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


def relabel_nonshared_chains(
    atom_array: AtomArray,
    shared_chain_id: str,
    partner_chain_ids: Sequence[str],
) -> AtomArray:
    """Relabel non-shared chains to the user-facing partner chain IDs.

    RFD3's normal per-complex pipeline may compact a two-chain view to A/B even
    when the original global source chains were A/C. Coupled output formatting
    needs to restore those global chain labels so track 2 is written as A+C and
    the merged output is written as A+B+C.
    """

    desired_chain_ids = normalize_chain_ids(partner_chain_ids)
    if shared_chain_id in desired_chain_ids:
        raise ValueError("Partner chain IDs must not include the shared chain.")

    relabeled = atom_array.copy()
    actual_chain_ids = [
        chain_id
        for chain_id in ordered_chain_ids(relabeled)
        if chain_id != shared_chain_id
    ]
    if len(actual_chain_ids) != len(desired_chain_ids):
        raise ValueError(
            "Cannot relabel non-shared chains: expected "
            f"{len(desired_chain_ids)} partner chain(s) {desired_chain_ids}, "
            f"but found {len(actual_chain_ids)} chain(s) {actual_chain_ids}."
        )

    for actual_chain_id, desired_chain_id in zip(actual_chain_ids, desired_chain_ids):
        mask = relabeled.chain_id == actual_chain_id
        relabeled.chain_id[mask] = desired_chain_id
        _relabel_string_annotation(
            relabeled,
            "chain_iid",
            mask,
            actual_chain_id,
            desired_chain_id,
        )
        _relabel_string_annotation(
            relabeled,
            "pn_unit_id",
            mask,
            actual_chain_id,
            desired_chain_id,
        )
        _relabel_string_annotation(
            relabeled,
            "pn_unit_iid",
            mask,
            actual_chain_id,
            desired_chain_id,
        )
    return relabeled


def _relabel_string_annotation(
    atom_array: AtomArray,
    annotation_name: str,
    mask: np.ndarray,
    actual_chain_id: str,
    desired_chain_id: str,
) -> None:
    """Relabel a chain-like string annotation if the AtomArray has it."""

    if annotation_name not in atom_array.get_annotation_categories():
        return
    values = atom_array.get_annotation(annotation_name).astype(str)
    updated = values.copy()
    suffix_prefix = f"{actual_chain_id}_"
    for value in np.unique(values[mask]):
        value_mask = mask & (values == value)
        if value == actual_chain_id:
            updated[value_mask] = desired_chain_id
        elif value.startswith(suffix_prefix):
            updated[value_mask] = f"{desired_chain_id}_{value[len(suffix_prefix):]}"
    atom_array.set_annotation(annotation_name, updated)


def append_nonshared_from_track_2(
    track_1_atom_array: AtomArray,
    track_2_atom_array: AtomArray,
    shared_chain_id: str,
) -> AtomArray:
    """Merge track-1 A+B with track-2 non-shared partner atoms."""

    nonshared_2 = track_2_atom_array[~shared_chain_mask(track_2_atom_array, shared_chain_id)]
    merged = track_1_atom_array.copy() + nonshared_2.copy()
    merged.bonds = None
    if "atom_id" in merged.get_annotation_categories():
        # The two track outputs are built independently, so their atom_id
        # annotations can collide after concatenation even when chain IDs differ.
        # Dropping atom_id lets the CIF writer assign unambiguous identifiers.
        merged.del_annotation("atom_id")
    return merged
