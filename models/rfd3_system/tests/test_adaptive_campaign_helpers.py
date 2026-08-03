from __future__ import annotations

import sys
import csv
import json
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


import adaptive_campaign_metrics as metrics  # noqa: E402
import build_tied_mpnn_input as tied_input  # noqa: E402
import fold_mpnn_esmfold2 as folding  # noqa: E402
import pyrosetta_interface_metrics as interface_metrics  # noqa: E402
import sequence_design_io as sequence_io  # noqa: E402


def test_filter_threshold_operators_have_requested_boundary_semantics() -> None:
    assert not metrics.criterion(0.5, "gt", 0.5)["pass"]
    assert not metrics.criterion(6, "lt", 6)["pass"]
    assert metrics.criterion(2, "ge", 2)["pass"]
    assert metrics.criterion(5, "ge", 5)["pass"]
    assert not metrics.criterion(None, "gt", 0.5)["pass"]


def test_prefilters_apply_all_sep_coordination_thresholds() -> None:
    def pair(shared: str, partner: str) -> dict:
        return {
            "interface": {
                "shape_complementarity": 0.7,
                "interface_hbonds": 4,
                "interface_unsat_hbonds": 2,
            },
            "monomers": {
                shared: {
                    "surface_hydrophobicity": 0.2,
                    "radius_of_gyration": 10.0,
                    "radius_of_gyration_limit": 20.0,
                },
                partner: {
                    "surface_hydrophobicity": 0.2,
                    "radius_of_gyration": 10.0,
                    "radius_of_gyration_limit": 20.0,
                },
            },
        }

    pair_metrics = {"AB": pair("A", "B"), "DC": pair("D", "C")}
    pair_metrics["AB"]["sep_phosphate"] = {
        "sep_phosphate_bidentate_count": 1,
        "sep_phosphate_polar_contact_count": 5,
        "sep_phosphate_contacted_oxygen_count": 3,
        "sep_phosphate_contacting_partner_residue_count": 2,
    }
    config = {
        "filters": {
            "prefilter": {
                "shape_complementarity_min": 0.5,
                "interface_hbonds_min": 2,
                "interface_unsat_hbonds_max": 6,
                "surface_hydrophobicity_max": 0.37,
                "sep_phosphate_bidentate_min": 1,
                "sep_phosphate_contacts_min": 5,
                "sep_phosphate_contacted_oxygens_min": 3,
                "sep_phosphate_contacting_partner_residues_min": 3,
            }
        }
    }

    failed = metrics.apply_prefilters(pair_metrics, config)
    assert not failed["pass"]
    assert not failed["checks"][
        "AB.sep_phosphate_contacting_partner_residue_count"
    ]["pass"]

    pair_metrics["AB"]["sep_phosphate"][
        "sep_phosphate_contacting_partner_residue_count"
    ] = 3
    passed = metrics.apply_prefilters(pair_metrics, config)
    assert passed["pass"]
    assert {
        "AB.sep_phosphate_bidentate_count",
        "AB.sep_phosphate_polar_contact_count",
        "AB.sep_phosphate_contacted_oxygen_count",
        "AB.sep_phosphate_contacting_partner_residue_count",
    } <= set(passed["checks"])


def test_sep_contact_metric_schema_includes_coverage_counts() -> None:
    expected = {
        "sep_phosphate_polar_contact_count",
        "sep_phosphate_bidentate_count",
        "sep_phosphate_contacted_oxygen_count",
        "sep_phosphate_contacting_partner_residue_count",
    }
    assert expected <= set(interface_metrics.METRIC_KEYS)
    assert expected <= set(
        interface_metrics.sep_phosphate_polar_contact_metrics(
            _EmptyPose(), "A", "B"
        )
    )


class _EmptyPose:
    """Small no-SEP pose used to test the importable metric schema."""

    @staticmethod
    def total_residue() -> int:
        return 0


class _Point:
    def __init__(self, xyz: tuple[float, float, float]) -> None:
        self.xyz = np.asarray(xyz, dtype=float)

    def distance(self, other: "_Point") -> float:
        return float(np.linalg.norm(self.xyz - other.xyz))


class _AtomType:
    def __init__(self, element: str) -> None:
        self._element = element

    def element(self) -> str:
        return self._element


class _Residue:
    def __init__(
        self,
        name: str,
        atoms: list[tuple[str, str, tuple[float, float, float]]],
    ) -> None:
        self._name = name
        self._atoms = atoms

    def name3(self) -> str:
        return self._name

    def natoms(self) -> int:
        return len(self._atoms)

    def nheavyatoms(self) -> int:
        return len(self._atoms)

    def atom_name(self, atom_index: int) -> str:
        return self._atoms[atom_index - 1][0]

    def atom_type(self, atom_index: int) -> _AtomType:
        return _AtomType(self._atoms[atom_index - 1][1])

    def xyz(self, atom_index: int) -> _Point:
        return _Point(self._atoms[atom_index - 1][2])


class _PdbInfo:
    def __init__(self, chains: list[str]) -> None:
        self._chains = chains

    def chain(self, residue_index: int) -> str:
        return self._chains[residue_index - 1]


class _Pose:
    def __init__(self, residues: list[_Residue], chains: list[str]) -> None:
        self._residues = residues
        self._pdb_info = _PdbInfo(chains)

    def total_residue(self) -> int:
        return len(self._residues)

    def residue(self, residue_index: int) -> _Residue:
        return self._residues[residue_index - 1]

    def pdb_info(self) -> _PdbInfo:
        return self._pdb_info


def test_sep_contact_metrics_count_oxygen_and_residue_coverage() -> None:
    pose = _Pose(
        [
            _Residue(
                "SEP",
                [
                    ("O1P", "O", (0.0, 0.0, 0.0)),
                    ("O2P", "O", (10.0, 0.0, 0.0)),
                    ("O3P", "O", (20.0, 0.0, 0.0)),
                ],
            ),
            _Residue(
                "ARG",
                [
                    ("NH1", "N", (1.0, 0.0, 0.0)),
                    ("NH2", "N", (11.0, 0.0, 0.0)),
                ],
            ),
            _Residue("LYS", [("NZ", "N", (21.0, 0.0, 0.0))]),
            _Residue("GLY", [("O", "O", (2.0, 0.0, 0.0))]),
        ],
        ["A", "B", "B", "B"],
    )
    observed = interface_metrics.sep_phosphate_polar_contact_metrics(
        pose, "A", "B", cutoff=3.6
    )
    assert observed == {
        "sep_phosphate_polar_contact_count": 4,
        "sep_phosphate_bidentate_count": 1,
        "sep_phosphate_contacted_oxygen_count": 3,
        "sep_phosphate_contacting_partner_residue_count": 3,
    }


def test_alignment_and_directional_rmsd_primitives() -> None:
    target = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mobile = target + np.asarray([10.0, -4.0, 2.0])
    assert metrics.aligned_rmsd(mobile, target) < 1e-10
    assert metrics.rmsd(mobile, target) > 1.0


def test_ca_radius_of_gyration_and_limit() -> None:
    coords = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert metrics.ca_radius_of_gyration(coords) == 1.0
    assert metrics.radius_of_gyration_limit(100) == 0.395 * 100**0.6 + 10.0


def test_discovers_rfd_track_structure_pairs(tmp_path: Path) -> None:
    for track in (1, 2):
        (tmp_path / f"example_0_track{track}_model_3.cif.gz").touch()
    assert metrics.discover_rfd_track_structures(tmp_path) == [
        (
            3,
            tmp_path / "example_0_track1_model_3.cif.gz",
            tmp_path / "example_0_track2_model_3.cif.gz",
        )
    ]


def test_loads_selected_rfd_model_indices(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text('{"model_indices": [0, 2, 2]}\n')
    assert tied_input.load_model_indices(selection) == {0, 2}
    assert tied_input.load_model_indices(None) is None


def test_normalizes_mpnn_omit_residue_tokens() -> None:
    assert tied_input.normalize_omit_residues("C") == ["CYS"]
    assert tied_input.normalize_omit_residues("CYS,M") == ["CYS", "MET"]
    assert tied_input.normalize_omit_residues('["C", "CYS", "W"]') == [
        "CYS",
        "TRP",
    ]


def test_finite_pdb_conversion_drops_nonfinite_atoms(tmp_path: Path) -> None:
    pdb = tmp_path / "input.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA A   1         nan   1.000   1.000  1.00  0.00           C  \n"
        "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C  \n"
        "TER\nEND\n"
    )
    output = metrics.convert_structure_to_finite_pdb(pdb, tmp_path / "output.pdb")
    text = output.read_text()
    assert " N   ALA" in text
    assert " C   ALA" in text
    assert " CA  ALA" not in text


def test_relax_structure_creates_destination_directory(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "input.pdb"
    output_path = tmp_path / "nested" / "relaxed.pdb"
    observed: dict[str, bool] = {}

    def fake_relax(source: Path, destination: Path, **kwargs) -> None:
        observed["parent_exists"] = destination.parent.is_dir()

    monkeypatch.setattr(metrics, "pr_relax", fake_relax)
    metrics.relax_structure(
        input_path,
        output_path,
        {
            "fast_relax": {
                "max_iterations": 200,
                "backbone_movable": True,
                "sidechains_movable": True,
                "jumps_movable": False,
                "constrain_to_start_coordinates": True,
            }
        },
    )
    assert observed["parent_exists"]


def test_fold_state_parser_and_task_seed_are_stable() -> None:
    assert folding.parse_states("AB_SEP,DC_SER") == {"AB_SEP", "DC_SER"}
    assert folding.parse_states(None) is None
    assert folding.stable_task_seed(123, "design_AB_SEP") == folding.stable_task_seed(
        123, "design_AB_SEP"
    )
    assert folding.stable_task_seed(123, "design_AB_SEP") != folding.stable_task_seed(
        123, "design_DC_SER"
    )


def test_caliby_records_use_structure_chain_ids_not_csv_order(tmp_path: Path) -> None:
    structure = tmp_path / "input_sample0.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "TER\n"
        "ATOM      2  CA  GLY B   1       1.000   0.000   0.000  1.00  0.00           C  \n"
        "TER\n"
        "ATOM      3  CA  SER D   1     100.000   0.000   0.000  1.00  0.00           C  \n"
        "TER\n"
        "ATOM      4  CA  TYR C   1     101.000   0.000   0.000  1.00  0.00           C  \n"
        "TER\nEND\n"
    )
    manifest = {
        "entries": [
            {
                "model_index": 0,
                "track1_cif": "track1.cif.gz",
                "track2_cif": "track2.cif.gz",
                "combined_pdb": str(tmp_path / "input.pdb"),
                "chain_order": ["A", "B", "D", "C"],
                "chain_lengths": {"A": 1, "B": 1, "D": 1, "C": 1},
                "fixed_a_source_residues": ["A10"],
                "fixed_residues": ["A1", "D1"],
            }
        ]
    }
    (tmp_path / "tied_caliby_manifest.json").write_text(json.dumps(manifest))
    with (tmp_path / "seq_des_outputs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["example_id", "out_pdb", "U", "input_seq", "seq"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "example_id": "input",
                "out_pdb": str(structure),
                "U": "-12.5",
                "input_seq": "X:X:X:X",
                # Deliberately wrong: the loader must trust structure chain IDs.
                "seq": "W:W:W:W",
            }
        )

    entries = sequence_io.load_sequence_design_manifest(tmp_path, "caliby")
    records = sequence_io.load_sequence_design_records(tmp_path, entries, "caliby")
    assert len(records) == 1
    assert records[0].chains == {"A": "A", "B": "G", "D": "S", "C": "Y"}
    assert records[0].backend_score == -12.5
    assert records[0].design_key == "input_sample0"


def test_omitted_amino_acids_allow_fixed_positions_only() -> None:
    assert metrics.normalize_omitted_amino_acids("CYS") == {"C"}
    entry = type("Entry", (), {"fixed_residues": ["A1"]})()
    allowed = type(
        "Record", (), {"design_key": "allowed", "chains": {"A": "CA"}}
    )()
    metrics.validate_omitted_amino_acids(allowed, entry, {"C"})

    rejected = type(
        "Record", (), {"design_key": "rejected", "chains": {"A": "AC"}}
    )()
    try:
        metrics.validate_omitted_amino_acids(rejected, entry, {"C"})
    except ValueError as error:
        assert "A2" in str(error)
    else:
        raise AssertionError("Expected a designable omitted residue to fail")
