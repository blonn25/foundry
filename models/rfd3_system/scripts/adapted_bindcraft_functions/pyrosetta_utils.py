"""Adapted BindCraft PyRosetta utilities for rfd3_system.

Adapted from BindCraft ``functions/pyrosetta_utils.py`` at commit
``b971db42ba6e091afab63ccb30ae02215150a990``.

The original BindCraft utility assumes a target chain ``A`` and binder chain
``B`` in a few places. This adapted copy keeps BindCraft's metric definitions
but accepts explicit chain IDs so the utilities can be reused for rfd3_system
outputs such as A/B and D/C without renaming chains.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .biopython_utils import hotspot_residues
from .generic_utils import clean_pdb


_PYROSETTA_INITIALIZED = False


def _find_dalphaball_path() -> Path | None:
    """Find the project-local BindCraft DAlphaBall binary when available."""

    env_path = os.environ.get("BINDCRAFT_DALPHABALL")
    candidates = [Path(env_path)] if env_path else []
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(
            parent / "software" / "BindCraft" / "functions" / "DAlphaBall.gcc"
        )
        candidates.append(parent / "functions" / "DAlphaBall.gcc")

    for candidate in candidates:
        if candidate.is_file():
            try:
                candidate.chmod(candidate.stat().st_mode | 0o111)
            except OSError:
                # Read-only installs can still work if the file already has an
                # executable bit; Rosetta will report a clear error otherwise.
                pass
            return candidate
    return None


def _default_pyrosetta_options(dalphaball_path: Path | None) -> str:
    """Return the BindCraft-like PyRosetta options used by metric helpers."""

    options = (
        "-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
        "-corrections::beta_nov16 true -relax:default_repeats 1"
    )
    if dalphaball_path is not None:
        options += f" -holes:dalphaball {dalphaball_path}"
    return options


def init_pyrosetta_once(options: str | None = None) -> Any:
    """Import and initialize PyRosetta once for standalone utility use."""

    global _PYROSETTA_INITIALIZED
    import pyrosetta as pr

    if not _PYROSETTA_INITIALIZED:
        if options is None:
            options = _default_pyrosetta_options(_find_dalphaball_path())
        pr.init(options)
        _PYROSETTA_INITIALIZED = True
    return pr


# Rosetta interface scores
def score_interface(pdb_file, target_chain="A", binder_chain="B"):
    """Compute BindCraft-style interface scores for target/binder chains."""

    pr = init_pyrosetta_once()
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

    # load pose
    pose = pr.pose_from_pdb(str(pdb_file))

    # analyze interface statistics
    iam = InterfaceAnalyzerMover()
    iam.set_interface(f"{target_chain}_{binder_chain}")
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # Initialize dictionary with all amino acids
    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    # Initialize list to store PDB residue IDs at the interface
    interface_residues_set = hotspot_residues(
        pdb_file,
        target_chain=target_chain,
        binder_chain=binder_chain,
    )
    interface_residues_pdb_ids = []

    # Iterate over the interface residues
    for pdb_res_num, aa_type in interface_residues_set.items():
        # Increase the count for this amino acid type
        interface_AA[aa_type] += 1

        # Append the binder_chain and the PDB residue number to the list
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    # count interface residues
    interface_nres = len(interface_residues_pdb_ids)

    # Convert the list into a comma-separated string
    interface_residues_pdb_ids_str = ','.join(interface_residues_pdb_ids)

    # Calculate the percentage of hydrophobic residues at the interface of the binder
    hydrophobic_aa = set('ACFILMPVWY')
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    if interface_nres != 0:
        interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100
    else:
        interface_hydrophobicity = 0

    # retrieve statistics
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value # shape complementarity
    interface_interface_hbonds = interfacescore.interface_hbonds # number of interface H-bonds
    interface_dG = iam.get_interface_dG() # interface dG
    interface_dSASA = iam.get_interface_delta_sasa() # interface dSASA (interface surface area)
    interface_packstat = iam.get_interface_packstat() # interface pack stat score
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100 # ratio of dG/dSASA (normalised energy for interface area size)
    buns_filter = XmlObjects.static_get_filter('<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    if interface_nres != 0:
        interface_hbond_percentage = (interface_interface_hbonds / interface_nres) * 100 # Hbonds per interface size percentage
        interface_bunsch_percentage = (interface_delta_unsat_hbonds / interface_nres) * 100 # Unsaturated H-bonds per percentage
    else:
        interface_hbond_percentage = None
        interface_bunsch_percentage = None

    # calculate binder energy score
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    # calculate binder SASA fraction
    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    if binder_sasa > 0:
        interface_binder_fraction = (interface_dSASA / binder_sasa) * 100
    else:
        interface_binder_fraction = 0

    # calculate surface hydrophobicity
    binder_pose = {pose.pdb_info().chain(pose.conformation().chain_begin(i)): p for i, p in zip(range(1, pose.num_chains()+1), pose.split_by_chain())}[binder_chain]

    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core = False, pick_boundary = False, pick_surface = True)
    surface_res = layer_sel.apply(binder_pose)

    exp_apol_count = 0
    total_count = 0

    # count apolar and aromatic residues at the surface
    for i in range(1, len(surface_res) + 1):
        if surface_res[i] == True:
            res = binder_pose.residue(i)

            # count apolar and aromatic residues as hydrophobic
            if res.is_apolar() == True or res.name() == 'PHE' or res.name() == 'TRP' or res.name() == 'TYR':
                exp_apol_count += 1
            total_count += 1

    surface_hydrophobicity = exp_apol_count/total_count if total_count else 0

    # output interface score array and amino acid counts at the interface
    interface_scores = {
    'binder_score': binder_score,
    'surface_hydrophobicity': surface_hydrophobicity,
    'interface_sc': interface_sc,
    'interface_packstat': interface_packstat,
    'interface_dG': interface_dG,
    'interface_dSASA': interface_dSASA,
    'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
    'interface_fraction': interface_binder_fraction,
    'interface_hydrophobicity': interface_hydrophobicity,
    'interface_nres': interface_nres,
    'interface_interface_hbonds': interface_interface_hbonds,
    'interface_hbond_percentage': interface_hbond_percentage,
    'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
    'interface_delta_unsat_hbonds_percentage': interface_bunsch_percentage
    }

    # round to two decimal places
    interface_scores = {k: round(v, 2) if isinstance(v, float) else v for k, v in interface_scores.items()}

    return interface_scores, interface_AA, interface_residues_pdb_ids_str


def score_monomer_surface_hydrophobicity(pdb_file, chain_id):
    """Return BindCraft's exposed-hydrophobe fraction for one explicit chain.

    BindCraft computes this value for the binder selected by ``score_interface``.
    The adaptive system campaign evaluates every monomer independently, so this
    small extraction applies the same LayerSelector and residue classification
    without assigning target/binder roles.
    """

    pr = init_pyrosetta_once()
    pose = pr.pose_from_file(str(pdb_file))
    chains = {
        pose.pdb_info().chain(pose.conformation().chain_begin(index)): chain_pose
        for index, chain_pose in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())
    }
    if chain_id not in chains:
        raise ValueError(f"Chain {chain_id!r} is absent from {pdb_file}")

    chain_pose = chains[chain_id]
    selector = pr.rosetta.core.select.residue_selector.LayerSelector()
    selector.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
    surface_residues = selector.apply(chain_pose)

    hydrophobic = 0
    total = 0
    for residue_index in range(1, len(surface_residues) + 1):
        if not surface_residues[residue_index]:
            continue
        residue = chain_pose.residue(residue_index)
        if residue.is_apolar() or residue.name3() in {"PHE", "TRP", "TYR"}:
            hydrophobic += 1
        total += 1
    return hydrophobic / total if total else 0.0


# align pdbs to have same orientation
def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    pr = init_pyrosetta_once()
    from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover

    # initiate poses
    reference_pose = pr.pose_from_pdb(str(reference_pdb))
    align_pose = pr.pose_from_pdb(str(align_pdb))

    align = AlignChainMover()
    align.pose(reference_pose)

    # If the chain IDs contain commas, split them and only take the first value
    reference_chain_id = reference_chain_id.split(',')[0]
    align_chain_id = align_chain_id.split(',')[0]

    # Get the chain number corresponding to the chain ID in the poses
    reference_chain = pr.rosetta.core.pose.get_chain_id_from_chain(reference_chain_id, reference_pose)
    align_chain = pr.rosetta.core.pose.get_chain_id_from_chain(align_chain_id, align_pose)

    align.source_chain(align_chain)
    align.target_chain(reference_chain)
    align.apply(align_pose)

    # Overwrite aligned pdb
    align_pose.dump_pdb(str(align_pdb))
    clean_pdb(align_pdb)


# calculate the rmsd without alignment
def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    pr = init_pyrosetta_once()
    from pyrosetta.rosetta.core.io import pose_from_pose
    from pyrosetta.rosetta.core.select import get_residues_from_subset
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
    from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric

    reference_pose = pr.pose_from_pdb(str(reference_pdb))
    align_pose = pr.pose_from_pdb(str(align_pdb))

    # Define chain selectors for the reference and align chains
    reference_chain_selector = ChainSelector(reference_chain_id)
    align_chain_selector = ChainSelector(align_chain_id)

    # Apply selectors to get residue subsets
    reference_chain_subset = reference_chain_selector.apply(reference_pose)
    align_chain_subset = align_chain_selector.apply(align_pose)

    # Convert subsets to residue index vectors
    reference_residue_indices = get_residues_from_subset(reference_chain_subset)
    align_residue_indices = get_residues_from_subset(align_chain_subset)

    # Create empty subposes
    reference_chain_pose = pr.Pose()
    align_chain_pose = pr.Pose()

    # Fill subposes
    pose_from_pose(reference_chain_pose, reference_pose, reference_residue_indices)
    pose_from_pose(align_chain_pose, align_pose, align_residue_indices)

    # Calculate RMSD using the RMSDMetric
    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(reference_chain_pose)
    rmsd = rmsd_metric.calculate(align_chain_pose)

    return round(rmsd, 2)


# Relax designed structure
def pr_relax(
    pdb_file,
    relaxed_pdb_path,
    *,
    max_iterations=200,
    backbone_movable=True,
    sidechains_movable=True,
    jumps_movable=False,
    constrain_to_start_coordinates=True,
    selected_atom_restraints=None,
    selected_atom_restraint_sd=0.1,
    selected_atom_restraint_weight=1.0,
):
    """Run BindCraft-style FastRelax with optional atom-coordinate restraints.

    ``selected_atom_restraints`` is an iterable of ``(chain, residue, atom)``
    triples.  The selected atoms receive harmonic coordinate constraints to
    their input positions through a virtual root.  These are strong positional
    restraints, not exact Cartesian freezes; the returned report quantifies
    their displacement so campaign code can audit the approximation.
    """
    pr = init_pyrosetta_once()
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover

    restraint_report = {
        "enabled": bool(selected_atom_restraints),
        "harmonic_sd_angstrom": float(selected_atom_restraint_sd),
        "score_weight": float(selected_atom_restraint_weight),
        "atoms": [],
        "maximum_displacement_angstrom": 0.0,
        "mean_displacement_angstrom": 0.0,
    }

    if not os.path.exists(relaxed_pdb_path):
        # Generate pose
        pose = pr.pose_from_file(str(pdb_file))
        start_pose = pose.clone()

        ### Generate movemaps
        mmf = MoveMap()
        mmf.set_chi(sidechains_movable) # enable sidechain movement
        mmf.set_bb(backbone_movable) # backbone minimization improves metrics but costs runtime
        mmf.set_jump(jumps_movable) # keep chain placement fixed by default

        # Run FastRelax
        fastrelax = FastRelax()
        scorefxn = pr.get_fa_scorefxn()
        tracked_atoms = []
        if selected_atom_restraints:
            if selected_atom_restraint_sd <= 0:
                raise ValueError("selected_atom_restraint_sd must be positive")
            if selected_atom_restraint_weight <= 0:
                raise ValueError("selected_atom_restraint_weight must be positive")

            from pyrosetta.rosetta.core.id import AtomID
            from pyrosetta.rosetta.core.scoring import coordinate_constraint
            from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
            from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
            from pyrosetta.rosetta.protocols.simple_moves import VirtualRootMover

            VirtualRootMover().apply(pose)
            root_atom = AtomID(1, pose.total_residue())
            pdb_info = pose.pdb_info()
            for chain_id, residue_number, atom_name in selected_atom_restraints:
                pose_index = int(pdb_info.pdb2pose(str(chain_id), int(residue_number)))
                if pose_index <= 0:
                    raise ValueError(
                        f"Restrained residue {chain_id}{residue_number} is absent from {pdb_file}"
                    )
                residue = pose.residue(pose_index)
                atom_name = str(atom_name).strip()
                if not residue.has(atom_name):
                    raise ValueError(
                        f"Restrained atom {chain_id}{residue_number}:{atom_name} "
                        f"is absent from {pdb_file}"
                    )
                atom_id = AtomID(residue.atom_index(atom_name), pose_index)
                start_xyz = pose.xyz(atom_id)
                pose.add_constraint(
                    CoordinateConstraint(
                        atom_id,
                        root_atom,
                        start_xyz,
                        HarmonicFunc(0.0, float(selected_atom_restraint_sd)),
                    )
                )
                tracked_atoms.append(
                    (str(chain_id), int(residue_number), atom_name, atom_id, start_xyz)
                )
            scorefxn.set_weight(
                coordinate_constraint, float(selected_atom_restraint_weight)
            )
        fastrelax.set_scorefxn(scorefxn)
        fastrelax.set_movemap(mmf) # set MoveMap
        fastrelax.max_iter(max_iterations) # Rosetta's default is much larger
        fastrelax.min_type("lbfgs_armijo_nonmonotone")
        fastrelax.constrain_relax_to_start_coords(constrain_to_start_coordinates)
        fastrelax.apply(pose)

        if tracked_atoms:
            # The coordinate constraints already anchor the pose to the input
            # frame through a virtual root. AlignChainMover does not account
            # for that added root reliably and can apply a spurious rigid-body
            # transform after a successful restrained relaxation.
            displacements = []
            for chain_id, residue_number, atom_name, atom_id, start_xyz in tracked_atoms:
                displacement = float((pose.xyz(atom_id) - start_xyz).norm())
                displacements.append(displacement)
                restraint_report["atoms"].append(
                    {
                        "chain": chain_id,
                        "residue": residue_number,
                        "atom": atom_name,
                        "displacement_angstrom": displacement,
                    }
                )
            restraint_report["maximum_displacement_angstrom"] = max(displacements)
            restraint_report["mean_displacement_angstrom"] = sum(displacements) / len(
                displacements
            )
        else:
            # Preserve the original BindCraft behavior for unrestrained runs.
            align = AlignChainMover()
            align.source_chain(0)
            align.target_chain(0)
            align.pose(start_pose)
            align.apply(pose)

        # Copy B factors from start_pose to pose
        for resid in range(1, pose.total_residue() + 1):
            if pose.residue(resid).is_protein():
                # Get the B factor of the first heavy atom in the residue
                bfactor = start_pose.pdb_info().bfactor(resid, 1)
                for atom_id in range(1, pose.residue(resid).natoms() + 1):
                    pose.pdb_info().bfactor(resid, atom_id, bfactor)

        # output relaxed and aligned PDB
        pose.dump_pdb(str(relaxed_pdb_path))
        clean_pdb(relaxed_pdb_path)
    return restraint_report
