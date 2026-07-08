#!/usr/bin/env python3
"""PyRosetta interface metrics for rfd3_system pipeline ESMFold2 outputs.

This module is intentionally importable without PyRosetta. PyRosetta imports and
initialization happen lazily so lightweight result-collection tests can run in
ordinary Python environments, while full scoring can run on CoreHPC in the
project PyRosetta environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "interface_dG",
    "interface_delta_SASA",
    "interface_dG_per_delta_SASA",
    "shape_complementarity",
    "sep_phosphate_hbond_count",
    "sep_phosphate_polar_contact_count",
    "sep_phosphate_bidentate_count",
    "pyrosetta_metrics_error",
)
PHOSPHATE_ACCEPTOR_ATOMS = {"O1P", "O2P", "O3P"}
POLAR_HEAVY_ELEMENTS = {"N", "O", "S"}
BACKBONE_HEAVY_ATOMS = {"N", "CA", "C", "O", "OXT"}
SEP_PHOSPHATE_POLAR_CONTACT_CUTOFF = 3.6

_PYROSETTA_INITIALIZED = False


def empty_metrics(error: str = "") -> dict[str, Any]:
    """Return blank metric values, preserving an optional error message."""

    return {key: "" for key in METRIC_KEYS} | {"pyrosetta_metrics_error": error}


def chains_for_complex_kind(complex_kind: str) -> tuple[str, str]:
    """Return the shared and partner chain IDs for one ESMFold2 complex kind."""

    if complex_kind == "AB_SEP":
        return "A", "B"
    if complex_kind == "DC_SER":
        return "D", "C"
    raise ValueError(f"unsupported ESMFold2 complex_kind: {complex_kind!r}")


def normalized_ratio(numerator: Any, denominator: Any) -> float | str:
    """Return numerator / denominator, or blank when the denominator is zero."""

    denominator = float(denominator)
    if denominator == 0.0:
        return ""
    return float(numerator) / denominator


def init_pyrosetta_once() -> None:
    """Initialize PyRosetta once with quiet logging."""

    global _PYROSETTA_INITIALIZED
    if _PYROSETTA_INITIALIZED:
        return
    import pyrosetta

    pyrosetta.init("-mute all")
    _PYROSETTA_INITIALIZED = True


def load_pose(cif_path: str | Path):
    """Load an ESMFold2 CIF as a PyRosetta pose."""

    init_pyrosetta_once()
    import pyrosetta

    path = Path(cif_path)
    if not path.is_file():
        raise FileNotFoundError(f"ESMFold2 CIF not found: {path}")
    pose = pyrosetta.pose_from_file(str(path))
    if pose.total_residue() == 0:
        raise ValueError(f"PyRosetta loaded zero residues from {path}")
    return pose


def compute_interface_metrics(pose, shared_chain: str, partner_chain: str) -> dict[str, float | str]:
    """Compute Rosetta InterfaceAnalyzer metrics for the requested interface."""

    init_pyrosetta_once()
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    scorefxn = pyrosetta.get_fa_scorefxn()
    analyzer = InterfaceAnalyzerMover()
    analyzer.set_interface(f"{shared_chain}_{partner_chain}")
    analyzer.set_scorefunction(scorefxn)
    analyzer.set_pack_input(False)
    analyzer.set_pack_separated(False)
    analyzer.apply(pose)

    interface_dg = float(analyzer.get_interface_dG())
    interface_delta_sasa = float(analyzer.get_interface_delta_sasa())
    return {
        "interface_dG": interface_dg,
        "interface_delta_SASA": interface_delta_sasa,
        "interface_dG_per_delta_SASA": normalized_ratio(
            interface_dg,
            interface_delta_sasa,
        ),
    }


def compute_shape_complementarity(pose) -> float:
    """Compute shape complementarity for the two-chain folded complex."""

    init_pyrosetta_once()
    from pyrosetta.rosetta.core.scoring.sc import ShapeComplementarityCalculator

    calculator = ShapeComplementarityCalculator()
    return float(calculator.CalcSc(pose, 1))


def residue_chain(pose, residue_index: int) -> str:
    """Return the PDB chain ID for a pose residue."""

    pdb_info = pose.pdb_info()
    if pdb_info is None:
        return ""
    return str(pdb_info.chain(residue_index)).strip()


def sep_residue_indices(pose, shared_chain: str) -> set[int]:
    """Return shared-chain SEP residues in pose numbering."""

    indices: set[int] = set()
    for residue_index in range(1, pose.total_residue() + 1):
        residue = pose.residue(residue_index)
        if residue.name3().strip() == "SEP" and residue_chain(pose, residue_index) == shared_chain:
            indices.add(residue_index)
    return indices


def hbond_acceptor_atom_name(pose, residue_index: int, atom_index: int) -> str:
    """Return a normalized atom name for an HBond acceptor atom."""

    return pose.residue(residue_index).atom_name(atom_index).strip()


def atom_element(pose, residue_index: int, atom_index: int) -> str:
    """Return the chemical element Rosetta assigned to one atom."""

    return str(pose.residue(residue_index).atom_type(atom_index).element()).strip()


def phosphate_atom_indices(pose, sep_indices: set[int]) -> list[tuple[int, int, str]]:
    """Return SEP phosphate oxygen atom indices as pose residue/atom tuples."""

    atoms: list[tuple[int, int, str]] = []
    for residue_index in sep_indices:
        residue = pose.residue(residue_index)
        for atom_index in range(1, residue.natoms() + 1):
            atom_name = residue.atom_name(atom_index).strip()
            if atom_name in PHOSPHATE_ACCEPTOR_ATOMS:
                atoms.append((residue_index, atom_index, atom_name))
    return atoms


def is_sidechain_heavy_atom(pose, residue_index: int, atom_index: int) -> bool:
    """Return whether an atom is a sidechain heavy atom for bidentate grouping."""

    atom_name = pose.residue(residue_index).atom_name(atom_index).strip()
    if atom_name in BACKBONE_HEAVY_ATOMS:
        return False
    return atom_element(pose, residue_index, atom_index) != "H"


def sep_phosphate_polar_contact_metrics(
    pose,
    shared_chain: str,
    partner_chain: str,
    cutoff: float = SEP_PHOSPHATE_POLAR_CONTACT_CUTOFF,
) -> dict[str, int | str]:
    """Count partner polar contacts and bidentate sidechains to SEP phosphate.

    ``sep_phosphate_polar_contact_count`` is a contact-pair count: each partner
    polar heavy atom within ``cutoff`` Angstroms of each SEP O1P/O2P/O3P atom is
    counted separately. For example, one Tyr OH contacting O1P, O2P, and O3P is
    three polar contacts.

    ``sep_phosphate_bidentate_count`` is residue-based. A partner residue counts
    once when its sidechain makes contacts to at least two unique SEP phosphate
    oxygens through at least two unique sidechain heavy atoms.
    """

    sep_indices = sep_residue_indices(pose, shared_chain)
    if not sep_indices:
        return {
            "sep_phosphate_polar_contact_count": "",
            "sep_phosphate_bidentate_count": "",
        }

    phosphate_atoms = phosphate_atom_indices(pose, sep_indices)
    polar_contact_count = 0
    residue_contacts: dict[int, dict[str, set[Any]]] = {}
    for partner_residue in range(1, pose.total_residue() + 1):
        if residue_chain(pose, partner_residue) != partner_chain:
            continue
        partner = pose.residue(partner_residue)
        for partner_atom in range(1, partner.nheavyatoms() + 1):
            if atom_element(pose, partner_residue, partner_atom) not in POLAR_HEAVY_ELEMENTS:
                continue
            for sep_residue, sep_atom, sep_atom_name in phosphate_atoms:
                distance = partner.xyz(partner_atom).distance(
                    pose.residue(sep_residue).xyz(sep_atom)
                )
                if distance > cutoff:
                    continue
                polar_contact_count += 1
                if not is_sidechain_heavy_atom(pose, partner_residue, partner_atom):
                    continue
                entry = residue_contacts.setdefault(
                    partner_residue,
                    {"partner_atoms": set(), "sep_oxygens": set()},
                )
                entry["partner_atoms"].add(partner_atom)
                entry["sep_oxygens"].add((sep_residue, sep_atom_name))

    bidentate_count = sum(
        1
        for contact in residue_contacts.values()
        if len(contact["partner_atoms"]) >= 2 and len(contact["sep_oxygens"]) >= 2
    )
    return {
        "sep_phosphate_polar_contact_count": polar_contact_count,
        "sep_phosphate_bidentate_count": bidentate_count,
    }


def count_sep_phosphate_hbonds(pose, shared_chain: str, partner_chain: str) -> int | str:
    """Count partner-to-SEP phosphate H-bonds across the folded interface.

    A blank value means the shared chain does not contain SEP. A count of zero
    means SEP is present, but no partner-chain donor H-bonds to O1P/O2P/O3P were
    identified by Rosetta's HBondSet.
    """

    sep_indices = sep_residue_indices(pose, shared_chain)
    if not sep_indices:
        return ""

    init_pyrosetta_once()
    from pyrosetta.rosetta.core.scoring.hbonds import HBondSet, fill_hbond_set

    pose.update_residue_neighbors()
    hbond_set = HBondSet()
    fill_hbond_set(pose, False, hbond_set, False, False, False, False)

    count = 0
    for hbond_index in range(1, hbond_set.nhbonds() + 1):
        hbond = hbond_set.hbond(hbond_index)
        acceptor_residue = int(hbond.acc_res())
        if acceptor_residue not in sep_indices:
            continue
        acceptor_atom = hbond_acceptor_atom_name(
            pose,
            acceptor_residue,
            int(hbond.acc_atm()),
        )
        if acceptor_atom not in PHOSPHATE_ACCEPTOR_ATOMS:
            continue
        donor_residue = int(hbond.don_res())
        if residue_chain(pose, donor_residue) != partner_chain:
            continue
        count += 1
    return count


def compute_esmfold2_pyrosetta_metrics(
    cif_path: str | Path,
    complex_kind: str,
) -> dict[str, Any]:
    """Compute all supported PyRosetta metrics for one ESMFold2 CIF."""

    try:
        shared_chain, partner_chain = chains_for_complex_kind(complex_kind)
        pose = load_pose(cif_path)
        return empty_metrics() | compute_interface_metrics(
            pose,
            shared_chain,
            partner_chain,
        ) | {
            "shape_complementarity": compute_shape_complementarity(pose),
            "sep_phosphate_hbond_count": count_sep_phosphate_hbonds(
                pose,
                shared_chain,
                partner_chain,
            ),
        } | sep_phosphate_polar_contact_metrics(pose, shared_chain, partner_chain)
    except Exception as error:
        return empty_metrics(str(error))
