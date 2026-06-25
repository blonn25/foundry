import numpy as np
import pytest
from biotite.structure import AtomArray

from rfd3_system.system.chains import (
    build_shared_update_atom_map,
    merge_tracks_with_shared_source,
    relabel_nonshared_chains,
)


BACKBONE = ("N", "CA", "C", "O")


def _atom_array(residues):
    """Build a minimal AtomArray from (chain, res_id, res_name, atom_names)."""

    rows = [
        (chain_id, res_id, res_name, atom_name)
        for chain_id, res_id, res_name, atom_names in residues
        for atom_name in atom_names
    ]
    atom_array = AtomArray(len(rows))
    atom_array.chain_id = np.asarray([row[0] for row in rows], dtype="U4")
    atom_array.res_id = np.asarray([row[1] for row in rows], dtype=int)
    atom_array.res_name = np.asarray([row[2] for row in rows], dtype="U4")
    atom_array.atom_name = np.asarray([row[3] for row in rows], dtype="U4")
    atom_array.coord = np.zeros((len(rows), 3), dtype=np.float32)
    return atom_array


def _set_src_component(atom_array, chain_id, res_id, src_component):
    """Annotate one residue with a source motif component label."""

    if "src_component" not in atom_array.get_annotation_categories():
        atom_array.set_annotation(
            "src_component",
            np.full(atom_array.array_length(), "", dtype="U16"),
        )
    atom_array.src_component[_residue_mask(atom_array, chain_id, res_id)] = src_component


def _set_unindexed_chain(atom_array, chain_id):
    """Mark every atom in one chain as an unindexed guidepost."""

    if "is_motif_atom_unindexed" not in atom_array.get_annotation_categories():
        atom_array.set_annotation(
            "is_motif_atom_unindexed",
            np.zeros(atom_array.array_length(), dtype=bool),
        )
    atom_array.is_motif_atom_unindexed[atom_array.chain_id == chain_id] = True


def _empty_masks(atom_array):
    length = atom_array.array_length()
    return np.zeros(length, dtype=bool), np.zeros(length, dtype=bool)


def _residue_mask(atom_array, chain_id, res_id):
    return (atom_array.chain_id == chain_id) & (atom_array.res_id == res_id)


def test_non_fixed_shared_residues_map_all_matching_atoms():
    track_1 = _atom_array(
        [
            ("A", 1, "ALA", BACKBONE + ("CB",)),
            ("A", 2, "GLY", BACKBONE),
        ]
    )
    track_2 = _atom_array(
        [
            ("A", 1, "ALA", BACKBONE + ("CB",)),
            ("A", 2, "GLY", BACKBONE),
        ]
    )
    fixed_coord_1, fixed_seq_1 = _empty_masks(track_1)
    fixed_coord_2, fixed_seq_2 = _empty_masks(track_2)

    atom_map = build_shared_update_atom_map(
        track_1,
        track_2,
        "A",
        fixed_coord_1,
        fixed_coord_2,
        fixed_seq_1,
        fixed_seq_2,
    )

    assert atom_map.update_indices_1.tolist() == list(range(track_1.array_length()))
    assert atom_map.update_indices_2.tolist() == list(range(track_2.array_length()))
    assert len(atom_map.update_residues) == 2
    assert atom_map.excluded_fixed_residues == []


def test_fixed_shared_variant_residue_is_excluded_from_coupled_update():
    track_1 = _atom_array(
        [
            ("A", 1, "SER", BACKBONE + ("CB", "OG")),
            ("A", 2, "ALA", BACKBONE + ("CB",)),
        ]
    )
    track_2 = _atom_array(
        [
            ("A", 1, "SEP", BACKBONE + ("CB", "OG", "P", "O1P", "O2P", "O3P")),
            ("A", 2, "ALA", BACKBONE + ("CB",)),
        ]
    )
    fixed_coord_1, fixed_seq_1 = _empty_masks(track_1)
    fixed_coord_2, fixed_seq_2 = _empty_masks(track_2)
    fixed_coord_1[_residue_mask(track_1, "A", 1)] = True
    fixed_coord_2[_residue_mask(track_2, "A", 1)] = True
    fixed_seq_1[_residue_mask(track_1, "A", 1)] = True
    fixed_seq_2[_residue_mask(track_2, "A", 1)] = True

    atom_map = build_shared_update_atom_map(
        track_1,
        track_2,
        "A",
        fixed_coord_1,
        fixed_coord_2,
        fixed_seq_1,
        fixed_seq_2,
    )

    assert atom_map.update_residues == [
        {"res_id": 2, "res_name": "ALA", "atom_count": 5}
    ]
    assert atom_map.excluded_fixed_residues == [
        {
            "res_id": 1,
            "track_1_res_name": "SER",
            "track_2_res_name": "SEP",
            "reason": "fixed_motif_context",
        }
    ]


def test_fixed_shared_guideposts_align_by_src_component_when_res_ids_differ():
    track_1 = _atom_array(
        [
            ("A", 1, "ALA", BACKBONE + ("CB",)),
            ("A", 91, "SEP", BACKBONE + ("CB", "OG", "P", "O1P", "O2P", "O3P")),
        ]
    )
    track_2 = _atom_array(
        [
            ("A", 1, "ALA", BACKBONE + ("CB",)),
            ("A", 101, "SER", BACKBONE + ("CB", "OG")),
        ]
    )
    _set_src_component(track_1, "A", 91, "A240")
    _set_src_component(track_2, "A", 101, "A240")
    fixed_coord_1, fixed_seq_1 = _empty_masks(track_1)
    fixed_coord_2, fixed_seq_2 = _empty_masks(track_2)
    fixed_coord_1[_residue_mask(track_1, "A", 91)] = True
    fixed_coord_2[_residue_mask(track_2, "A", 101)] = True
    fixed_seq_1[_residue_mask(track_1, "A", 91)] = True
    fixed_seq_2[_residue_mask(track_2, "A", 101)] = True

    atom_map = build_shared_update_atom_map(
        track_1,
        track_2,
        "A",
        fixed_coord_1,
        fixed_coord_2,
        fixed_seq_1,
        fixed_seq_2,
    )

    assert atom_map.update_residues == [
        {"res_id": 1, "res_name": "ALA", "atom_count": 5}
    ]
    assert atom_map.excluded_fixed_residues == [
        {
            "src_component": "A240",
            "track_1_res_id": 91,
            "track_2_res_id": 101,
            "track_1_res_name": "SEP",
            "track_2_res_name": "SER",
            "reason": "fixed_motif_context",
        }
    ]


def test_src_component_key_does_not_collide_with_generated_res_id():
    track_1 = _atom_array(
        [
            ("A", 237, "ALA", BACKBONE + ("CB",)),
            ("A", 301, "SER", BACKBONE + ("CB", "OG")),
        ]
    )
    track_2 = _atom_array(
        [
            ("A", 237, "ALA", BACKBONE + ("CB",)),
            ("A", 401, "SEP", BACKBONE + ("CB", "OG", "P")),
        ]
    )
    _set_src_component(track_1, "A", 301, "A237")
    _set_src_component(track_2, "A", 401, "A237")
    fixed_coord_1, fixed_seq_1 = _empty_masks(track_1)
    fixed_coord_2, fixed_seq_2 = _empty_masks(track_2)
    fixed_coord_1[_residue_mask(track_1, "A", 301)] = True
    fixed_coord_2[_residue_mask(track_2, "A", 401)] = True
    fixed_seq_1[_residue_mask(track_1, "A", 301)] = True
    fixed_seq_2[_residue_mask(track_2, "A", 401)] = True

    atom_map = build_shared_update_atom_map(
        track_1,
        track_2,
        "A",
        fixed_coord_1,
        fixed_coord_2,
        fixed_seq_1,
        fixed_seq_2,
    )

    assert atom_map.update_residues == [
        {"res_id": 237, "res_name": "ALA", "atom_count": 5}
    ]
    assert atom_map.excluded_fixed_residues[0]["src_component"] == "A237"


def test_non_fixed_shared_variant_residue_fails():
    track_1 = _atom_array([("A", 1, "SER", BACKBONE + ("CB", "OG"))])
    track_2 = _atom_array(
        [("A", 1, "SEP", BACKBONE + ("CB", "OG", "P", "O1P", "O2P", "O3P"))]
    )
    fixed_coord_1, fixed_seq_1 = _empty_masks(track_1)
    fixed_coord_2, fixed_seq_2 = _empty_masks(track_2)

    with pytest.raises(ValueError, match="Only fixed motif residues may differ"):
        build_shared_update_atom_map(
            track_1,
            track_2,
            "A",
            fixed_coord_1,
            fixed_coord_2,
            fixed_seq_1,
            fixed_seq_2,
        )


def test_one_track_fixed_shared_residue_fails():
    track_1 = _atom_array([("A", 1, "SER", BACKBONE + ("CB", "OG"))])
    track_2 = _atom_array([("A", 1, "SER", BACKBONE + ("CB", "OG"))])
    fixed_coord_1, fixed_seq_1 = _empty_masks(track_1)
    fixed_coord_2, fixed_seq_2 = _empty_masks(track_2)
    fixed_coord_1[:] = True
    fixed_seq_1[:] = True

    with pytest.raises(ValueError, match="fixed in only one track"):
        build_shared_update_atom_map(
            track_1,
            track_2,
            "A",
            fixed_coord_1,
            fixed_coord_2,
            fixed_seq_1,
            fixed_seq_2,
        )


def test_fixed_shared_residue_requires_fixed_sequence_in_both_tracks():
    track_1 = _atom_array([("A", 1, "SER", BACKBONE + ("CB", "OG"))])
    track_2 = _atom_array([("A", 1, "SEP", BACKBONE + ("CB", "OG", "P"))])
    fixed_coord_1, fixed_seq_1 = _empty_masks(track_1)
    fixed_coord_2, fixed_seq_2 = _empty_masks(track_2)
    fixed_coord_1[:] = True
    fixed_coord_2[:] = True
    fixed_seq_1[:] = True

    with pytest.raises(ValueError, match="fixed sequence in both tracks"):
        build_shared_update_atom_map(
            track_1,
            track_2,
            "A",
            fixed_coord_1,
            fixed_coord_2,
            fixed_seq_1,
            fixed_seq_2,
        )


def test_merged_output_can_choose_track_2_shared_chain_source():
    track_1 = _atom_array(
        [
            ("A", 1, "SER", BACKBONE + ("CB", "OG")),
            ("B", 1, "GLY", BACKBONE),
        ]
    )
    track_2 = _atom_array(
        [
            ("A", 1, "SEP", BACKBONE + ("CB", "OG", "P")),
            ("C", 1, "ALA", BACKBONE + ("CB",)),
        ]
    )

    merged = merge_tracks_with_shared_source(track_1, track_2, "A", "track_2")

    assert merged.chain_id.tolist() == ["A"] * 7 + ["B"] * 4 + ["C"] * 5
    assert merged.res_name[:7].tolist() == ["SEP"] * 7


def test_relabel_nonshared_chains_ignores_unindexed_guidepost_chain():
    atom_array = _atom_array(
        [
            ("A", 1, "ALA", BACKBONE + ("CB",)),
            ("B", 1, "GLY", BACKBONE),
            ("X1", 1, "GLU", BACKBONE + ("CB", "CG", "CD", "OE1", "OE2")),
        ]
    )
    _set_unindexed_chain(atom_array, "X1")

    relabeled = relabel_nonshared_chains(atom_array, "A", ["C"])

    assert relabeled.chain_id.tolist() == ["A"] * 5 + ["C"] * 4 + ["X1"] * 9
