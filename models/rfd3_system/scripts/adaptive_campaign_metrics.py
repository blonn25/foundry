#!/usr/bin/env python3
"""Thread, relax, filter, and package adaptive rfd3_system campaign designs.

This helper is intentionally stage-oriented.  It runs in the project's
PyRosetta-capable environment after RFD3/ProteinMPNN or ESMFold2 have produced
their outputs.  Every design owns one JSON metrics record; no worker writes a
campaign-wide CSV, which avoids concurrent-writer corruption.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from adapted_bindcraft_functions.pyrosetta_utils import (
    pr_relax,
    score_interface,
    score_monomer_surface_hydrophobicity,
)
from fold_mpnn_esmfold2 import load_manifest, load_mpnn_records
from pyrosetta_interface_metrics import sep_phosphate_polar_contact_metrics


AA1_TO_3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
PAIR_CHAINS = {
    "AB_SEP": ("A", "B"),
    "DC_SER": ("D", "C"),
    "AC_SEP": ("A", "C"),
    "DB_SER": ("D", "B"),
}
STAGE_KINDS = {
    "on_target": ("AB_SEP", "DC_SER"),
    "monomer": ("A_SEP", "D_SER", "B", "C"),
    "off_target": ("AC_SEP", "DB_SER"),
}
STRUCTURE_SUFFIXES = (".pdb", ".cif", ".cif.gz", ".mmcif", ".mmcif.gz")
LABEL_RE = re.compile(r"^(?P<chain>[A-Za-z]+)(?P<resid>-?\d+)[A-Za-z]*$")
RFD_TRACK_RE = re.compile(
    r"(?P<prefix>.+)_track(?P<track>[12])_model_(?P<model>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prefilter = subparsers.add_parser("prefilter")
    prefilter.add_argument("mpnn_output_dir", type=Path)
    prefilter.add_argument("--round-dir", type=Path, required=True)
    prefilter.add_argument("--config", type=Path, required=True)

    geometry = subparsers.add_parser("rfd-geometry-prefilter")
    geometry.add_argument("rfd_output_dir", type=Path)
    geometry.add_argument("--round-dir", type=Path, required=True)
    geometry.add_argument("--config", type=Path, required=True)

    for command, stage in (
        ("score-on-target", "on_target"),
        ("score-monomers", "monomer"),
        ("score-off-target", "off_target"),
    ):
        scorer = subparsers.add_parser(command)
        scorer.add_argument("fold_output_dir", type=Path)
        scorer.add_argument("--round-dir", type=Path, required=True)
        scorer.add_argument("--config", type=Path, required=True)
        scorer.set_defaults(stage=stage)

    package = subparsers.add_parser("build-promotion-manifest")
    package.add_argument("--round-dir", type=Path, required=True)
    package.add_argument("--out", type=Path, required=True)

    cleanup = subparsers.add_parser("cleanup-round")
    cleanup.add_argument("--round-dir", type=Path, required=True)
    cleanup.add_argument("--rfd3-dir", type=Path, required=True)
    cleanup.add_argument("--mpnn-dir", type=Path, required=True)
    cleanup.add_argument("--fold-dir", type=Path, action="append", default=[])
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def normalize_path(path: str | Path) -> Path:
    value = Path(str(path))
    project_root = Path(os.environ.get("PROJECT_DIR", Path.cwd()))
    if str(value) == "/project":
        return project_root
    if str(value).startswith("/project/"):
        return project_root / str(value).removeprefix("/project/")
    return value


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def metrics_path(round_dir: Path, design_key: str) -> Path:
    return round_dir / "metrics" / f"{safe_name(design_key)}.json"


def update_metrics(
    round_dir: Path, design_key: str, values: dict[str, Any]
) -> dict[str, Any]:
    path = metrics_path(round_dir, design_key)
    payload = (
        read_json(path) if path.is_file() else {"design_key": design_key, "stages": {}}
    )
    payload |= {key: value for key, value in values.items() if key != "stages"}
    payload.setdefault("stages", {}).update(values.get("stages", {}))
    atomic_write_json(path, payload)
    return payload


def write_design_keys(path: Path, design_keys: Iterable[str], *, stage: str) -> None:
    values = sorted(set(design_keys))
    atomic_write_json(
        path, {"stage": stage, "count": len(values), "design_keys": values}
    )


def read_design_keys(path: Path) -> list[str]:
    if not path.is_file():
        return []
    payload = read_json(path)
    return list(payload.get("design_keys", []))


def mapped_fixed_labels(entry: Any, chains: set[str]) -> set[str]:
    return {
        label
        for label in entry.fixed_residues
        if (match := LABEL_RE.fullmatch(label)) and match.group("chain") in chains
    }


def pose_chain_residues(pose: Any, chain_id: str) -> list[int]:
    pdb_info = pose.pdb_info()
    return [
        index
        for index in range(1, pose.total_residue() + 1)
        if str(pdb_info.chain(index)).strip() == chain_id
        and pose.residue(index).is_protein()
    ]


def convert_structure_to_finite_pdb(input_path: Path, output_path: Path) -> Path:
    """Convert a PDB/mmCIF to PDB while omitting non-finite-coordinate atoms.

    PyRosetta 2026 loads RFD3's compressed CIF outputs as empty poses. Biopython
    correctly reads those files, and the resulting PDB retains chain/residue
    labels plus SEP records in a representation PyRosetta accepts.
    """

    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select

    class FiniteAtomSelect(Select):
        def accept_atom(self, atom: Any) -> int:
            return int(np.isfinite(np.asarray(atom.coord, dtype=float)).all())

    parser = (
        MMCIFParser(QUIET=True)
        if input_path.name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz"))
        else PDBParser(QUIET=True)
    )
    with open_text(input_path) as handle:
        structure = parser.get_structure("threading_input", handle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(output_path), FiniteAtomSelect())
    return output_path


def thread_track_structure(
    input_path: Path,
    output_path: Path,
    sequences: dict[str, str],
    fixed_labels: set[str],
    chain_mapping: dict[str, str],
) -> Path:
    """Thread canonical design positions while preserving fixed motif chemistry."""

    from adapted_bindcraft_functions.pyrosetta_utils import init_pyrosetta_once

    pr = init_pyrosetta_once()
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    input_path = normalize_path(input_path)
    pose_input = input_path
    if input_path.name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz")):
        pose_input = convert_structure_to_finite_pdb(input_path, output_path)
    pose = pr.pose_from_file(str(pose_input))
    if pose.total_residue() == 0:
        raise ValueError(f"PyRosetta loaded zero residues from {pose_input}")
    pdb_info = pose.pdb_info()
    for input_chain, output_chain in chain_mapping.items():
        pose_indices = pose_chain_residues(pose, input_chain)
        sequence = sequences[output_chain]
        if len(pose_indices) != len(sequence):
            raise ValueError(
                f"{input_path}: chain {input_chain} has {len(pose_indices)} protein residues, "
                f"but sequence {output_chain} has length {len(sequence)}"
            )
        for ordinal, (pose_index, aa) in enumerate(
            zip(pose_indices, sequence), start=1
        ):
            pdb_number = int(pdb_info.number(pose_index))
            output_label = f"{output_chain}{pdb_number}"
            if output_label in fixed_labels:
                continue
            if aa not in AA1_TO_3:
                raise ValueError(
                    f"Cannot thread amino acid {aa!r} at {output_chain}{ordinal}"
                )
            residue = pose.residue(pose_index)
            if residue.name3().strip() == AA1_TO_3[aa]:
                continue
            mover = MutateResidue(pose_index, AA1_TO_3[aa])
            mover.set_preserve_atom_coords(True)
            mover.apply(pose)

    # Relabel only after mutation so source-chain lookup remains unambiguous.
    for pose_index in range(1, pose.total_residue() + 1):
        source_chain = str(pdb_info.chain(pose_index)).strip()
        if source_chain in chain_mapping:
            pdb_info.chain(pose_index, chain_mapping[source_chain])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pose.dump_pdb(str(output_path))
    return output_path


def relax_structure(
    input_path: Path, output_path: Path, config: dict[str, Any]
) -> None:
    settings = config["fast_relax"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pr_relax(
        input_path,
        output_path,
        max_iterations=int(settings["max_iterations"]),
        backbone_movable=bool(settings["backbone_movable"]),
        sidechains_movable=bool(settings["sidechains_movable"]),
        jumps_movable=bool(settings["jumps_movable"]),
        constrain_to_start_coordinates=bool(settings["constrain_to_start_coordinates"]),
    )


def interface_metrics(
    path: Path, shared_chain: str, partner_chain: str
) -> dict[str, Any]:
    scores, _, residues = score_interface(
        path,
        target_chain=shared_chain,
        binder_chain=partner_chain,
    )
    return {
        "shape_complementarity": float(scores["interface_sc"]),
        "interface_hbonds": float(scores["interface_interface_hbonds"]),
        "interface_unsat_hbonds": float(scores["interface_delta_unsat_hbonds"]),
        "interface_residues": residues,
    }


def open_text(path: Path):
    return gzip.open(path, "rt") if path.name.endswith(".gz") else path.open()


def ca_records(path: Path, chain_id: str) -> tuple[list[int], np.ndarray]:
    from Bio.PDB import MMCIFParser, PDBParser

    parser = (
        MMCIFParser(QUIET=True)
        if path.name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz"))
        else PDBParser(QUIET=True)
    )
    with open_text(path) as handle:
        model = next(parser.get_structure("structure", handle).get_models())
    records: list[tuple[int, np.ndarray]] = []
    seen: set[int] = set()
    for chain in model:
        if str(chain.id) != chain_id:
            continue
        for residue in chain:
            residue_id = int(residue.id[1])
            if residue_id in seen or "CA" not in residue:
                continue
            coord = np.asarray(residue["CA"].get_coord(), dtype=float)
            if np.isfinite(coord).all():
                seen.add(residue_id)
                records.append((residue_id, coord))
    records.sort(key=lambda item: item[0])
    if not records:
        raise ValueError(f"No finite CA atoms for chain {chain_id} in {path}")
    return [item[0] for item in records], np.stack([item[1] for item in records])


def ca_radius_of_gyration(coords: np.ndarray) -> float:
    """Return C-alpha radius of gyration for one finite coordinate array."""

    center = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))


def radius_of_gyration_limit(length: int) -> float:
    """Return the campaign's length-dependent monomer compactness limit."""

    return 0.395 * length**0.6 + 10.0


def monomer_metrics(path: Path, chain_id: str) -> dict[str, Any]:
    residue_ids, coords = ca_records(path, chain_id)
    radius = ca_radius_of_gyration(coords)
    return {
        "length": len(residue_ids),
        "surface_hydrophobicity": float(
            score_monomer_surface_hydrophobicity(path, chain_id)
        ),
        "radius_of_gyration": radius,
        "radius_of_gyration_limit": radius_of_gyration_limit(len(residue_ids)),
    }


def alignment_transform(
    mobile: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mobile_center).T @ (target - target_center))
    handedness = np.sign(np.linalg.det(u @ vt)) or 1.0
    rotation = u @ np.diag([1.0, 1.0, handedness]) @ vt
    return rotation, target_center - mobile_center @ rotation


def rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((coords_a - coords_b) ** 2, axis=1))))


def matched_ca(
    mobile_path: Path,
    mobile_chain: str,
    target_path: Path,
    target_chain: str,
) -> tuple[np.ndarray, np.ndarray]:
    mobile_ids, mobile = ca_records(mobile_path, mobile_chain)
    target_ids, target = ca_records(target_path, target_chain)
    if mobile_ids != target_ids:
        raise ValueError(
            f"CA residue IDs differ for {mobile_chain}->{target_chain}: "
            f"{mobile_ids[:3]}... ({len(mobile_ids)}) vs {target_ids[:3]}... ({len(target_ids)})"
        )
    return mobile, target


def aligned_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    rotation, translation = alignment_transform(mobile, target)
    return rmsd(mobile @ rotation + translation, target)


def dimer_rmsds(
    mobile_path: Path,
    reference_path: Path,
    shared_chain: str,
    partner_chain: str,
) -> dict[str, float]:
    shared_mobile, shared_reference = matched_ca(
        mobile_path, shared_chain, reference_path, shared_chain
    )
    partner_mobile, partner_reference = matched_ca(
        mobile_path, partner_chain, reference_path, partner_chain
    )
    all_mobile = np.vstack([shared_mobile, partner_mobile])
    all_reference = np.vstack([shared_reference, partner_reference])
    shared_rotation, shared_translation = alignment_transform(
        shared_mobile, shared_reference
    )
    partner_rotation, partner_translation = alignment_transform(
        partner_mobile, partner_reference
    )
    return {
        "ca_rmsd_all_chains": aligned_rmsd(all_mobile, all_reference),
        "ca_rmsd_partner_after_shared_align": rmsd(
            partner_mobile @ shared_rotation + shared_translation,
            partner_reference,
        ),
        "ca_rmsd_shared_after_partner_align": rmsd(
            shared_mobile @ partner_rotation + partner_translation,
            shared_reference,
        ),
    }


def scalar(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sample_metrics(task_payload: dict[str, Any]) -> dict[str, Any]:
    samples = task_payload.get("samples", [])
    if len(samples) != 1:
        raise ValueError(
            f"Expected exactly one ESMFold2 sample for {task_payload.get('task_id')}, "
            f"found {len(samples)}"
        )
    sample = samples[0]
    plddt = scalar(sample.get("plddt_mean"))
    if plddt is not None and plddt > 1.0:
        plddt /= 100.0
    return {
        "mean_plddt": plddt,
        "iptm": scalar(sample.get("iptm")),
        "mean_ipae_raw": scalar(sample.get("mean_ipae")),
        "mean_pae_raw": scalar(sample.get("mean_pae")),
        "raw_cif": str(normalize_path(sample["cif"])),
    }


def criterion(value: Any, operator: str, threshold: float) -> dict[str, Any]:
    number = scalar(value)
    if number is None:
        passed = False
    elif operator == "gt":
        passed = number > threshold
    elif operator == "lt":
        passed = number < threshold
    elif operator == "ge":
        passed = number >= threshold
    elif operator == "le":
        passed = number <= threshold
    else:
        raise ValueError(f"Unsupported threshold operator: {operator}")
    return {
        "value": number,
        "operator": operator,
        "threshold": threshold,
        "pass": passed,
    }


def strip_structure_suffix(path: Path) -> str:
    """Return a structure filename without its recognized compound suffix."""

    for suffix in sorted(STRUCTURE_SUFFIXES, key=len, reverse=True):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def discover_rfd_track_structures(
    output_dir: Path,
) -> list[tuple[int, Path, Path]]:
    """Discover paired track CIFs without importing the Foundry container stack."""

    records: dict[tuple[str, int], dict[str, Path]] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or not path.name.endswith(STRUCTURE_SUFFIXES):
            continue
        match = RFD_TRACK_RE.fullmatch(strip_structure_suffix(path))
        if match is None:
            continue
        key = (match.group("prefix"), int(match.group("model")))
        records.setdefault(key, {})[match.group("track")] = path

    pairs: list[tuple[int, Path, Path]] = []
    for (_, model_index), tracks in sorted(
        records.items(), key=lambda item: item[0][1]
    ):
        if set(tracks) != {"1", "2"}:
            raise ValueError(f"Incomplete RFD track pair for model {model_index}")
        pairs.append((model_index, tracks["1"], tracks["2"]))
    if not pairs:
        raise ValueError(f"No paired RFD track structures found in {output_dir}")
    return pairs


def rfd_geometry_prefilter(args: argparse.Namespace) -> None:
    """Reject noncompact raw RFD backbones before sequence design."""

    config = load_config(args.config)
    enabled = bool(
        config.get("filters", {}).get("rfd_geometry", {}).get("enabled", True)
    )
    round_dir = args.round_dir.resolve()
    metrics_dir = round_dir / "rfd_geometry"
    passing: list[int] = []
    pairs = discover_rfd_track_structures(args.rfd_output_dir.resolve())

    for model_index, track1, track2 in pairs:
        checks: dict[str, Any] = {}
        chains: dict[str, Any] = {}
        error = ""
        try:
            for label, path, chain_id in (
                ("track1.A", track1, "A"),
                ("track1.B", track1, "B"),
                ("track2.A", track2, "A"),
                ("track2.C", track2, "C"),
            ):
                residue_ids, coords = ca_records(path, chain_id)
                radius = ca_radius_of_gyration(coords)
                limit = radius_of_gyration_limit(len(residue_ids))
                check = criterion(radius, "lt", limit)
                check["would_pass"] = check["pass"]
                if not enabled:
                    check["pass"] = True
                checks[label] = check
                chains[label] = {
                    "chain_id": chain_id,
                    "length": len(residue_ids),
                    "radius_of_gyration": radius,
                    "radius_of_gyration_limit": limit,
                    "source": str(path),
                }
            passed = all(check["pass"] for check in checks.values())
        except Exception as exc:
            passed = False
            error = str(exc)

        payload = {
            "stage": "rfd_geometry_prefilter",
            "model_index": model_index,
            "filter_enabled": enabled,
            "pass": passed,
            "checks": checks,
            "chains": chains,
        }
        if error:
            payload["error"] = error
        atomic_write_json(metrics_dir / f"model_{model_index}.json", payload)
        if passed:
            passing.append(model_index)

    atomic_write_json(
        round_dir / "passing_rfd_geometry.json",
        {
            "stage": "rfd_geometry_prefilter",
            "filter_enabled": enabled,
            "total": len(pairs),
            "count": len(passing),
            "model_indices": passing,
        },
    )
    print(
        json.dumps(
            {
                "stage": "rfd_geometry_prefilter",
                "total": len(pairs),
                "passing": len(passing),
                "filter_enabled": enabled,
            }
        )
    )


def apply_prefilters(
    pair_metrics: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    limits = config["filters"]["prefilter"]
    checks: dict[str, Any] = {}
    for pair in ("AB", "DC"):
        metrics = pair_metrics[pair]["interface"]
        checks[f"{pair}.shape_complementarity"] = criterion(
            metrics["shape_complementarity"],
            "gt",
            float(limits["shape_complementarity_min"]),
        )
        checks[f"{pair}.interface_hbonds"] = criterion(
            metrics["interface_hbonds"], "gt", float(limits["interface_hbonds_min"])
        )
        checks[f"{pair}.interface_unsat_hbonds"] = criterion(
            metrics["interface_unsat_hbonds"],
            "lt",
            float(limits["interface_unsat_hbonds_max"]),
        )
    for chain in ("A", "B", "D", "C"):
        metrics = pair_metrics["AB" if chain in {"A", "B"} else "DC"]["monomers"][chain]
        checks[f"{chain}.surface_hydrophobicity"] = criterion(
            metrics["surface_hydrophobicity"],
            "lt",
            float(limits["surface_hydrophobicity_max"]),
        )
        checks[f"{chain}.radius_of_gyration"] = criterion(
            metrics["radius_of_gyration"],
            "lt",
            float(metrics["radius_of_gyration_limit"]),
        )
    return {"pass": all(item["pass"] for item in checks.values()), "checks": checks}


def task_payloads(fold_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(fold_dir.glob("*.json")):
        if path.name in {"planned_folds.json", "esmfold2_run_manifest.json"}:
            continue
        payload = read_json(path)
        design_key = payload.get("design_key")
        complex_kind = payload.get("complex_kind")
        if design_key and complex_kind:
            records[(str(design_key), str(complex_kind))] = payload
    return records


def relax_task(
    payload: dict[str, Any],
    destination: Path,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    metrics = sample_metrics(payload)
    raw_cif = Path(metrics["raw_cif"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    relax_structure(raw_cif, destination, config)
    metrics["relaxed_pdb"] = str(destination)
    return destination, metrics


def prefilter_record(
    record: Any,
    entry: Any,
    round_dir: Path,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    threaded_dir = round_dir / "artifacts" / "threaded_relaxed"
    ab_unrelaxed = (
        round_dir / "work" / f"{safe_name(record.design_key)}_AB_threaded.pdb"
    )
    dc_unrelaxed = (
        round_dir / "work" / f"{safe_name(record.design_key)}_DC_threaded.pdb"
    )
    ab_relaxed = threaded_dir / f"{safe_name(record.design_key)}_AB_relaxed.pdb"
    dc_relaxed = threaded_dir / f"{safe_name(record.design_key)}_DC_relaxed.pdb"

    try:
        if not ab_relaxed.is_file():
            thread_track_structure(
                normalize_path(entry.track1_cif),
                ab_unrelaxed,
                {"A": record.chains["A"], "B": record.chains["B"]},
                mapped_fixed_labels(entry, {"A", "B"}),
                {"A": "A", "B": "B"},
            )
            relax_structure(ab_unrelaxed, ab_relaxed, config)
        if not dc_relaxed.is_file():
            thread_track_structure(
                normalize_path(entry.track2_cif),
                dc_unrelaxed,
                {"D": record.chains["D"], "C": record.chains["C"]},
                mapped_fixed_labels(entry, {"D", "C"}),
                {"A": "D", "C": "C"},
            )
            relax_structure(dc_unrelaxed, dc_relaxed, config)

        pair_metrics = {
            "AB": {
                "relaxed_pdb": str(ab_relaxed),
                "interface": interface_metrics(ab_relaxed, "A", "B"),
                "monomers": {
                    "A": monomer_metrics(ab_relaxed, "A"),
                    "B": monomer_metrics(ab_relaxed, "B"),
                },
            },
            "DC": {
                "relaxed_pdb": str(dc_relaxed),
                "interface": interface_metrics(dc_relaxed, "D", "C"),
                "monomers": {
                    "D": monomer_metrics(dc_relaxed, "D"),
                    "C": monomer_metrics(dc_relaxed, "C"),
                },
            },
        }
        decision = apply_prefilters(pair_metrics, config)
        passed = bool(decision["pass"])
        result = {"metrics": pair_metrics, "decision": decision}
    except Exception as error:
        passed = False
        result = {
            "metrics": {},
            "decision": {"pass": False, "checks": {}, "error": str(error)},
        }
    finally:
        ab_unrelaxed.unlink(missing_ok=True)
        dc_unrelaxed.unlink(missing_ok=True)
    if not passed:
        ab_relaxed.unlink(missing_ok=True)
        dc_relaxed.unlink(missing_ok=True)
    return passed, result


def prefilter(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    round_dir = args.round_dir.resolve()
    mpnn_dir = args.mpnn_output_dir.resolve()
    manifest = load_manifest(mpnn_dir)
    records = load_mpnn_records(mpnn_dir, manifest)
    sequence_dir = round_dir / "sequences"
    passing: list[str] = []

    for record in records:
        entry = manifest[record.mpnn_name]
        passed, result = prefilter_record(record, entry, round_dir, config)
        fasta = sequence_dir / f"{safe_name(record.design_key)}.fasta"
        fasta.parent.mkdir(parents=True, exist_ok=True)
        fasta.write_text(
            "".join(
                f">{chain}\n{sequence}\n" for chain, sequence in record.chains.items()
            )
        )
        update_metrics(
            round_dir,
            record.design_key,
            {
                "model_index": record.model_index,
                "batch_index": record.batch_index,
                "design_index": record.design_index,
                "sequences": record.chains,
                "sequence_fasta": str(fasta),
                "rfd3_track1_cif": str(normalize_path(entry.track1_cif)),
                "rfd3_track2_cif": str(normalize_path(entry.track2_cif)),
                "stages": {"prefilter": result},
            },
        )
        if passed:
            passing.append(record.design_key)

    write_design_keys(round_dir / "passing_prefilter.json", passing, stage="prefilter")
    print(
        json.dumps(
            {"stage": "prefilter", "total": len(records), "passing": len(passing)}
        )
    )


def score_on_target(
    design_key: str,
    payloads: dict[tuple[str, str], dict[str, Any]],
    round_dir: Path,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    limits = config["filters"]["on_target"]
    results: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    for kind, pair_name in (("AB_SEP", "AB"), ("DC_SER", "DC")):
        payload = payloads[(design_key, kind)]
        shared, partner = PAIR_CHAINS[kind]
        relaxed = (
            round_dir
            / "artifacts"
            / "on_target_relaxed"
            / f"{safe_name(payload['task_id'])}_relaxed.pdb"
        )
        relaxed, fold_metrics = relax_task(payload, relaxed, config)
        pair = {
            "fold": fold_metrics,
            "interface": interface_metrics(relaxed, shared, partner),
            "monomers": {
                shared: monomer_metrics(relaxed, shared),
                partner: monomer_metrics(relaxed, partner),
            },
        }
        reference = (
            round_dir
            / "artifacts"
            / "threaded_relaxed"
            / f"{safe_name(design_key)}_{pair_name}_relaxed.pdb"
        )
        pair["rmsd_to_threaded"] = dimer_rmsds(relaxed, reference, shared, partner)
        if kind == "AB_SEP":
            from adapted_bindcraft_functions.pyrosetta_utils import init_pyrosetta_once

            pose = init_pyrosetta_once().pose_from_file(str(relaxed))
            pair["sep_phosphate"] = sep_phosphate_polar_contact_metrics(pose, "A", "B")
        results[kind] = pair

        checks[f"{kind}.mean_plddt"] = criterion(
            fold_metrics["mean_plddt"], "gt", float(limits["mean_plddt_min"])
        )
        checks[f"{kind}.iptm"] = criterion(
            fold_metrics["iptm"], "gt", float(limits["iptm_min"])
        )
        checks[f"{kind}.mean_ipae_raw"] = criterion(
            fold_metrics["mean_ipae_raw"], "lt", float(limits["mean_ipae_raw_max"])
        )
        checks[f"{kind}.ca_rmsd_all_chains"] = criterion(
            pair["rmsd_to_threaded"]["ca_rmsd_all_chains"],
            "lt",
            float(limits["ca_rmsd_all_chains_max"]),
        )
        checks[f"{kind}.ca_rmsd_partner_after_shared_align"] = criterion(
            pair["rmsd_to_threaded"]["ca_rmsd_partner_after_shared_align"],
            "lt",
            float(limits["ca_rmsd_unaligned_monomer_max"]),
        )
        checks[f"{kind}.ca_rmsd_shared_after_partner_align"] = criterion(
            pair["rmsd_to_threaded"]["ca_rmsd_shared_after_partner_align"],
            "lt",
            float(limits["ca_rmsd_unaligned_monomer_max"]),
        )

    repeated = apply_prefilters(
        {"AB": results["AB_SEP"], "DC": results["DC_SER"]}, config
    )
    checks |= {
        f"folded_prefilter.{key}": value for key, value in repeated["checks"].items()
    }
    sep = results["AB_SEP"]["sep_phosphate"]
    checks["AB_SEP.sep_phosphate_bidentate_count"] = criterion(
        sep["sep_phosphate_bidentate_count"],
        "ge",
        float(limits["sep_phosphate_bidentate_min"]),
    )
    checks["AB_SEP.sep_phosphate_polar_contact_count"] = criterion(
        sep["sep_phosphate_polar_contact_count"],
        "ge",
        float(limits["sep_phosphate_contacts_min"]),
    )
    decision = {"pass": all(item["pass"] for item in checks.values()), "checks": checks}
    return decision["pass"], {"metrics": results, "decision": decision}


def score_monomers(
    design_key: str,
    payloads: dict[tuple[str, str], dict[str, Any]],
    round_dir: Path,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    limit = float(config["filters"]["monomer"]["ca_rmsd_to_on_target_max"])
    complex_map = {
        "A_SEP": ("AB_SEP", "A"),
        "B": ("AB_SEP", "B"),
        "D_SER": ("DC_SER", "D"),
        "C": ("DC_SER", "C"),
    }
    results: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    for kind in STAGE_KINDS["monomer"]:
        payload = payloads[(design_key, kind)]
        relaxed = (
            round_dir
            / "artifacts"
            / "monomer_relaxed"
            / f"{safe_name(payload['task_id'])}_relaxed.pdb"
        )
        relaxed, fold_metrics = relax_task(payload, relaxed, config)
        complex_kind, chain = complex_map[kind]
        complex_path = (
            round_dir
            / "artifacts"
            / "on_target_relaxed"
            / f"{safe_name(design_key + '_' + complex_kind)}_relaxed.pdb"
        )
        mobile, target = matched_ca(relaxed, chain, complex_path, chain)
        value = aligned_rmsd(mobile, target)
        results[kind] = {
            "fold": fold_metrics,
            "relaxed_pdb": str(relaxed),
            "ca_rmsd_to_on_target_complex": value,
        }
        checks[f"{kind}.ca_rmsd_to_on_target_complex"] = criterion(value, "lt", limit)
    decision = {"pass": all(item["pass"] for item in checks.values()), "checks": checks}
    return decision["pass"], {"metrics": results, "decision": decision}


def score_off_target(
    design_key: str,
    payloads: dict[tuple[str, str], dict[str, Any]],
    round_dir: Path,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    limit = float(config["filters"]["off_target"]["iptm_max"])
    results: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    for kind in STAGE_KINDS["off_target"]:
        payload = payloads[(design_key, kind)]
        relaxed = (
            round_dir
            / "artifacts"
            / "off_target_relaxed"
            / f"{safe_name(payload['task_id'])}_relaxed.pdb"
        )
        relaxed, fold_metrics = relax_task(payload, relaxed, config)
        shared, partner = PAIR_CHAINS[kind]
        results[kind] = {
            "fold": fold_metrics,
            "relaxed_pdb": str(relaxed),
            "interface": interface_metrics(relaxed, shared, partner),
        }
        checks[f"{kind}.iptm"] = criterion(fold_metrics["iptm"], "lt", limit)
    decision = {"pass": all(item["pass"] for item in checks.values()), "checks": checks}
    return decision["pass"], {"metrics": results, "decision": decision}


def score_stage(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    round_dir = args.round_dir.resolve()
    fold_dir = args.fold_output_dir.resolve()
    stage = args.stage
    payloads = task_payloads(fold_dir)
    input_file = {
        "on_target": "passing_prefilter.json",
        "monomer": "passing_on_target.json",
        "off_target": "passing_monomer.json",
    }[stage]
    design_keys = read_design_keys(round_dir / input_file)
    passing: list[str] = []
    scorer = {
        "on_target": score_on_target,
        "monomer": score_monomers,
        "off_target": score_off_target,
    }[stage]

    for design_key in design_keys:
        missing = [
            kind for kind in STAGE_KINDS[stage] if (design_key, kind) not in payloads
        ]
        if missing:
            result = {
                "metrics": {},
                "decision": {
                    "pass": False,
                    "checks": {},
                    "error": "missing fold state(s): " + ", ".join(missing),
                },
            }
            passed = False
        else:
            try:
                passed, result = scorer(design_key, payloads, round_dir, config)
            except Exception as error:
                passed = False
                result = {
                    "metrics": {},
                    "decision": {"pass": False, "checks": {}, "error": str(error)},
                }
        update_metrics(round_dir, design_key, {"stages": {stage: result}})
        if passed:
            passing.append(design_key)
        else:
            for kind in STAGE_KINDS[stage]:
                payload = payloads.get((design_key, kind), {})
                task_id = payload.get("task_id")
                if task_id:
                    for folder in (
                        "on_target_relaxed",
                        "monomer_relaxed",
                        "off_target_relaxed",
                    ):
                        (
                            round_dir
                            / "artifacts"
                            / folder
                            / f"{safe_name(task_id)}_relaxed.pdb"
                        ).unlink(missing_ok=True)

    output_file = {
        "on_target": "passing_on_target.json",
        "monomer": "passing_monomer.json",
        "off_target": "passing_final.json",
    }[stage]
    write_design_keys(round_dir / output_file, passing, stage=stage)
    print(
        json.dumps({"stage": stage, "total": len(design_keys), "passing": len(passing)})
    )


def build_promotion_manifest(args: argparse.Namespace) -> None:
    round_dir = args.round_dir.resolve()
    entries: list[dict[str, Any]] = []
    for design_key in read_design_keys(round_dir / "passing_final.json"):
        metrics = read_json(metrics_path(round_dir, design_key))
        files: list[dict[str, str]] = []
        candidates = {
            "metrics.json": metrics_path(round_dir, design_key),
            "sequence.fasta": Path(metrics["sequence_fasta"]),
            "rfd3/geometry_metrics.json": (
                round_dir / "rfd_geometry" / f"model_{metrics['model_index']}.json"
            ),
            "rfd3/track1.cif.gz": Path(metrics["rfd3_track1_cif"]),
            "rfd3/track2.cif.gz": Path(metrics["rfd3_track2_cif"]),
        }
        for stage, stage_payload in metrics.get("stages", {}).items():
            for pair_payload in stage_payload.get("metrics", {}).values():
                if not isinstance(pair_payload, dict):
                    continue
                relaxed = pair_payload.get("relaxed_pdb") or pair_payload.get(
                    "fold", {}
                ).get("relaxed_pdb")
                if relaxed:
                    candidates[f"relaxed/{stage}/{Path(relaxed).name}"] = Path(relaxed)
        for destination, source in candidates.items():
            if not source.is_file():
                raise FileNotFoundError(f"Promotion source is missing: {source}")
            files.append({"source": str(source), "destination": destination})
        entries.append({"design_id": design_key, "files": files})
    atomic_write_json(args.out, {"round_dir": str(round_dir), "entries": entries})
    print(json.dumps({"promotion_entries": len(entries), "path": str(args.out)}))


def cleanup_round(args: argparse.Namespace) -> None:
    round_dir = args.round_dir.resolve()
    diagnostics = round_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    rfd3_dir = args.rfd3_dir.resolve()

    # One track JSON per model contains the complete shared coupling trajectory.
    for source in sorted(rfd3_dir.glob("*track1_model_*.json")):
        shutil.copy2(source, diagnostics / source.name)
    for source in sorted(rfd3_dir.glob("*coupling*.png")):
        shutil.copy2(source, diagnostics / source.name)

    roots = [
        rfd3_dir,
        args.mpnn_dir.resolve(),
        *(path.resolve() for path in args.fold_dir),
        round_dir / "artifacts",
        round_dir / "work",
    ]
    removed = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.endswith(STRUCTURE_SUFFIXES):
                path.unlink()
                removed += 1
    print(
        json.dumps(
            {"removed_structure_files": removed, "diagnostics": str(diagnostics)}
        )
    )


def main() -> None:
    args = parse_args()
    if args.command == "rfd-geometry-prefilter":
        rfd_geometry_prefilter(args)
    elif args.command == "prefilter":
        prefilter(args)
    elif args.command.startswith("score-"):
        score_stage(args)
    elif args.command == "build-promotion-manifest":
        build_promotion_manifest(args)
    elif args.command == "cleanup-round":
        cleanup_round(args)
    else:
        raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
