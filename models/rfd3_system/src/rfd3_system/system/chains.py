"""Chain-level helpers for the approximate shared-chain prototype."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from biotite.structure import AtomArray, get_residue_starts

ResidueKey = tuple[str, int | str]


@dataclass(frozen=True)
class SharedAtomMap:
    """Paired shared-chain atom indices used by the coupled denoising update."""

    update_indices_1: np.ndarray
    update_indices_2: np.ndarray
    update_residues: list[dict[str, object]]
    excluded_fixed_residues: list[dict[str, object]]

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-friendly shared-map diagnostics for output metadata."""

        return {
            "shared_update_atom_count": int(self.update_indices_1.size),
            "shared_update_residue_count": len(self.update_residues),
            "excluded_fixed_shared_residue_count": len(
                self.excluded_fixed_residues
            ),
            "excluded_fixed_shared_residues": self.excluded_fixed_residues,
        }


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


def build_shared_update_atom_map(
    track_1_atom_array: AtomArray,
    track_2_atom_array: AtomArray,
    shared_chain_id: str,
    fixed_coord_mask_1: np.ndarray,
    fixed_coord_mask_2: np.ndarray,
    fixed_seq_mask_1: np.ndarray,
    fixed_seq_mask_2: np.ndarray,
) -> SharedAtomMap:
    """Pair non-fixed shared-chain atoms for coupled denoising.

    Fixed-coordinate shared-chain residues are treated as motif context.  They
    may differ chemically between tracks, but they are excluded from the
    stochastic shared update and from the proxy kappa solve.  Non-fixed shared
    residues must still match exactly, preserving the original all-atom coupling
    behavior for ordinary generated chain A residues.
    """

    fixed_coord_mask_1 = np.asarray(fixed_coord_mask_1, dtype=bool)
    fixed_coord_mask_2 = np.asarray(fixed_coord_mask_2, dtype=bool)
    fixed_seq_mask_1 = np.asarray(fixed_seq_mask_1, dtype=bool)
    fixed_seq_mask_2 = np.asarray(fixed_seq_mask_2, dtype=bool)
    _validate_atom_mask_length(track_1_atom_array, fixed_coord_mask_1, "fixed_coord_mask_1")
    _validate_atom_mask_length(track_2_atom_array, fixed_coord_mask_2, "fixed_coord_mask_2")
    _validate_atom_mask_length(track_1_atom_array, fixed_seq_mask_1, "fixed_seq_mask_1")
    _validate_atom_mask_length(track_2_atom_array, fixed_seq_mask_2, "fixed_seq_mask_2")

    groups_1 = _shared_residue_groups(
        track_1_atom_array,
        shared_chain_id,
        fixed_coord_mask_1,
    )
    groups_2 = _shared_residue_groups(
        track_2_atom_array,
        shared_chain_id,
        fixed_coord_mask_2,
    )
    keys_1 = list(groups_1)
    keys_2 = list(groups_2)
    if keys_1 != keys_2:
        raise ValueError(
            "Shared-chain residue identities differ between tracks. Non-fixed "
            "shared-chain residues must have matching residue IDs, and fixed "
            "shared-chain motif residues must have matching src_component labels. "
            f"{_format_key_difference(keys_1, keys_2)}"
        )

    update_indices_1 = []
    update_indices_2 = []
    update_residues: list[dict[str, object]] = []
    excluded_fixed_residues: list[dict[str, object]] = []

    for residue_key in keys_1:
        idx_1 = groups_1[residue_key]
        idx_2 = groups_2[residue_key]
        fixed_1 = bool(np.any(fixed_coord_mask_1[idx_1]))
        fixed_2 = bool(np.any(fixed_coord_mask_2[idx_2]))
        residue_label = _residue_label(shared_chain_id, residue_key)
        res_name_1 = str(track_1_atom_array.res_name[idx_1[0]])
        res_name_2 = str(track_2_atom_array.res_name[idx_2[0]])

        if fixed_1 != fixed_2:
            raise ValueError(
                f"Shared-chain residue {residue_label} is fixed in only one "
                "track. Fixed motif residues on the shared chain must be fixed "
                "in both tracks or neither track."
            )

        if fixed_1 and fixed_2:
            if not np.all(fixed_seq_mask_1[idx_1]) or not np.all(
                fixed_seq_mask_2[idx_2]
            ):
                raise ValueError(
                    f"Shared-chain fixed motif residue {residue_label} does not "
                    "have fixed sequence in both tracks. Exclude motif residues "
                    "from select_unfixed_sequence."
                )
            excluded_fixed_residues.append(
                _fixed_residue_metadata(
                    track_1_atom_array,
                    track_2_atom_array,
                    idx_1,
                    idx_2,
                    residue_key,
                    res_name_1,
                    res_name_2,
                )
            )
            continue

        signature_1 = _atom_identity_signature(track_1_atom_array, idx_1)
        signature_2 = _atom_identity_signature(track_2_atom_array, idx_2)
        if signature_1 != signature_2:
            raise ValueError(
                f"Non-fixed shared-chain residue {residue_label} differs between "
                "tracks. Only fixed motif residues may differ in residue or atom "
                "identity."
            )

        update_indices_1.extend(idx_1.tolist())
        update_indices_2.extend(idx_2.tolist())
        update_residues.append(
            _update_residue_metadata(residue_key, res_name_1, len(idx_1))
        )

    if not update_indices_1:
        raise ValueError(
            "No non-fixed shared-chain atoms remain for coupled denoising after "
            "excluding fixed motif residues."
        )

    return SharedAtomMap(
        update_indices_1=np.asarray(update_indices_1, dtype=np.int64),
        update_indices_2=np.asarray(update_indices_2, dtype=np.int64),
        update_residues=update_residues,
        excluded_fixed_residues=excluded_fixed_residues,
    )


def _shared_residue_groups(
    atom_array: AtomArray,
    shared_chain_id: str,
    fixed_coord_mask: np.ndarray,
) -> dict[ResidueKey, np.ndarray]:
    """Return global atom indices grouped by shared-chain identity.

    Movable shared-chain residues are keyed by their generated residue ID. Fixed
    motif/guidepost residues are keyed by src_component when available, because
    RFD3 may assign different temporary residue IDs to the same unindexed source
    motif in different track contexts.
    """

    shared_indices = np.where(shared_chain_mask(atom_array, shared_chain_id))[0]
    shared_atoms = atom_array[shared_indices]
    starts = get_residue_starts(shared_atoms, add_exclusive_stop=True)
    groups: dict[ResidueKey, np.ndarray] = {}
    for start, stop in zip(starts[:-1], starts[1:]):
        residue_indices = shared_indices[start:stop]
        residue_key = _shared_residue_key(atom_array, residue_indices, fixed_coord_mask)
        if residue_key in groups:
            raise ValueError(
                f"Shared chain {shared_chain_id!r} contains duplicate residue key "
                f"{residue_key!r}."
            )
        groups[residue_key] = residue_indices
    return groups


def _shared_residue_key(
    atom_array: AtomArray,
    residue_indices: np.ndarray,
    fixed_coord_mask: np.ndarray,
) -> ResidueKey:
    """Return the typed key used to pair one shared-chain residue across tracks."""

    res_id = int(atom_array.res_id[residue_indices[0]])
    is_fixed = bool(np.any(fixed_coord_mask[residue_indices]))
    if not is_fixed:
        return ("res_id", res_id)

    if "src_component" not in atom_array.get_annotation_categories():
        return ("res_id", res_id)

    src_components = sorted(
        {
            str(src_component)
            for src_component in atom_array.src_component[residue_indices]
            if str(src_component)
        }
    )
    if not src_components:
        return ("res_id", res_id)
    if len(src_components) != 1:
        raise ValueError(
            "Fixed shared-chain residue has multiple src_component labels: "
            f"{src_components}."
        )
    return ("src_component", src_components[0])


def _validate_atom_mask_length(
    atom_array: AtomArray,
    mask: np.ndarray,
    mask_name: str,
) -> None:
    if mask.shape != (atom_array.array_length(),):
        raise ValueError(
            f"{mask_name} must be a 1D atom mask with length "
            f"{atom_array.array_length()}, but got shape {mask.shape}."
        )


def _atom_identity_signature(
    atom_array: AtomArray,
    indices: np.ndarray,
) -> list[tuple[str, str]]:
    """Build an order-sensitive residue/atom identity signature."""

    return [
        (str(atom_array.res_name[idx]), str(atom_array.atom_name[idx]))
        for idx in indices
    ]


def _residue_label(shared_chain_id: str, residue_key: ResidueKey) -> str:
    key_type, key_value = residue_key
    if key_type == "src_component":
        return str(key_value)
    return f"{shared_chain_id}{key_value}"


def _format_key_difference(keys_1: list[ResidueKey], keys_2: list[ResidueKey]) -> str:
    """Return a compact description of keys present in only one track."""

    set_1 = set(keys_1)
    set_2 = set(keys_2)
    only_1 = [_format_residue_key(key) for key in keys_1 if key not in set_2]
    only_2 = [_format_residue_key(key) for key in keys_2 if key not in set_1]
    return f"Only in track 1: {only_1}; only in track 2: {only_2}."


def _format_residue_key(residue_key: ResidueKey) -> str:
    key_type, key_value = residue_key
    return f"{key_type}:{key_value}"


def _update_residue_metadata(
    residue_key: ResidueKey,
    res_name: str,
    atom_count: int,
) -> dict[str, object]:
    key_type, key_value = residue_key
    if key_type != "res_id":
        raise ValueError(
            "Only generated residue-ID keys may be included in the shared update. "
            f"Got {residue_key!r}."
        )
    return {
        "res_id": int(key_value),
        "res_name": res_name,
        "atom_count": int(atom_count),
    }


def _fixed_residue_metadata(
    track_1_atom_array: AtomArray,
    track_2_atom_array: AtomArray,
    idx_1: np.ndarray,
    idx_2: np.ndarray,
    residue_key: ResidueKey,
    res_name_1: str,
    res_name_2: str,
) -> dict[str, object]:
    key_type, key_value = residue_key
    metadata: dict[str, object] = {}
    if key_type == "src_component":
        metadata["src_component"] = str(key_value)
        metadata["track_1_res_id"] = int(track_1_atom_array.res_id[idx_1[0]])
        metadata["track_2_res_id"] = int(track_2_atom_array.res_id[idx_2[0]])
    else:
        metadata["res_id"] = int(key_value)
    metadata.update(
        {
            "track_1_res_name": res_name_1,
            "track_2_res_name": res_name_2,
            "reason": "fixed_motif_context",
        }
    )
    return metadata


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
        and not _is_unindexed_guidepost_chain(relabeled, chain_id)
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


def _is_unindexed_guidepost_chain(atom_array: AtomArray, chain_id: str) -> bool:
    """Return True when a chain contains only unindexed guidepost atoms."""

    if "is_motif_atom_unindexed" not in atom_array.get_annotation_categories():
        return False
    mask = atom_array.chain_id == chain_id
    return bool(np.any(mask) and np.all(atom_array.is_motif_atom_unindexed[mask]))


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


def merge_tracks_with_shared_source(
    track_1_atom_array: AtomArray,
    track_2_atom_array: AtomArray,
    shared_chain_id: str,
    shared_source: str,
) -> AtomArray:
    """Merge A+B and A+C while choosing which track contributes shared chain A."""

    if shared_source not in {"track_1", "track_2"}:
        raise ValueError(f"Unsupported shared_source: {shared_source!r}")

    shared_source_array = (
        track_1_atom_array if shared_source == "track_1" else track_2_atom_array
    )
    shared = shared_source_array[
        shared_chain_mask(shared_source_array, shared_chain_id)
    ]
    nonshared_1 = track_1_atom_array[
        ~shared_chain_mask(track_1_atom_array, shared_chain_id)
    ]
    nonshared_2 = track_2_atom_array[
        ~shared_chain_mask(track_2_atom_array, shared_chain_id)
    ]
    merged = shared.copy() + nonshared_1.copy() + nonshared_2.copy()
    merged.bonds = None
    if "atom_id" in merged.get_annotation_categories():
        merged.del_annotation("atom_id")
    return merged
