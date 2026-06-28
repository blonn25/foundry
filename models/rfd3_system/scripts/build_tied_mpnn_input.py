#!/usr/bin/env python3
"""Build tied ProteinMPNN inputs from paired rfd3_system track outputs.

The intended input is an rfd3_system output directory containing paired
``*_track1_model_<i>.cif.gz`` and ``*_track2_model_<i>.cif.gz`` files plus
their JSON metadata.  For each model, this script writes one combined CIF:

    A + B, D + C

where A is the SER-containing shared chain from track 2, B is the track-1
partner, D is a translated copy of track-2 A, and C is the translated track-2
partner.  ProteinMPNN then sees two independent complexes in one file, while
explicit symmetry groups tie the designed sequence positions in A and D.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io import parse
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray


TRACK_OUTPUT_RE = re.compile(r"(?P<prefix>.+)_track(?P<track>[12])_model_(?P<model>\d+)$")
DEFAULT_FIXED_A_SOURCES = "A237,A238,A240"
DEFAULT_FIXED_B_SOURCES = "B56,B129,B130,B133"


@dataclass(frozen=True)
class TrackPair:
    """Paired track output files for one diffusion-batch model."""

    model_index: int
    prefix: str
    track1_cif: Path
    track1_json: Path
    track2_cif: Path
    track2_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create separated A+B and D+C ProteinMPNN inputs from paired "
            "rfd3_system track outputs, plus a tied-sequence MPNN config."
        )
    )
    parser.add_argument(
        "rfd3_output_dir",
        type=Path,
        help="Directory containing rfd3_system track1/track2 CIF and JSON outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="ProteinMPNN output directory; combined inputs and config are written here.",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help="Optional name prefix for combined inputs. Defaults to rfd3 output prefix.",
    )
    parser.add_argument(
        "--translation-distance",
        type=float,
        default=100.0,
        help="Distance in Angstroms used to translate the D+C complex away from A+B.",
    )
    parser.add_argument(
        "--shared-chain-id",
        default="A",
        help="Shared chain ID in both rfd3_system tracks.",
    )
    parser.add_argument(
        "--track1-partner-chain-id",
        default="B",
        help="Partner chain ID expected in track 1.",
    )
    parser.add_argument(
        "--track2-partner-chain-id",
        default="C",
        help="Partner chain ID expected in track 2.",
    )
    parser.add_argument(
        "--second-shared-chain-id",
        default="D",
        help="Chain ID assigned to the translated copy of the shared chain.",
    )
    parser.add_argument(
        "--fixed-a-source-residues",
        default=DEFAULT_FIXED_A_SOURCES,
        help=(
            "Source residues from the shared A motif to fix in both A and D. "
            "Comma-separated; simple ranges such as A237-238 are supported."
        ),
    )
    parser.add_argument(
        "--fixed-b-source-residues",
        default=DEFAULT_FIXED_B_SOURCES,
        help=(
            "Source residues from B guideposts to fix in chain B. "
            "Comma-separated; simple ranges such as B129-130 are supported."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="ProteinMPNN batch size per combined backbone.",
    )
    parser.add_argument(
        "--number-of-batches",
        type=int,
        default=1,
        help="Number of ProteinMPNN batches per combined backbone.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="ProteinMPNN sampling temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="ProteinMPNN seed assigned to each combined backbone input.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="/weights/proteinmpnn_v_48_020.pt",
        help="ProteinMPNN checkpoint path inside the Foundry container.",
    )
    parser.add_argument(
        "--model-type",
        choices=["protein_mpnn", "ligand_mpnn"],
        default="protein_mpnn",
        help="Foundry MPNN model type.",
    )
    parser.add_argument(
        "--is-legacy-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use legacy checkpoint loading for original ProteinMPNN/LigandMPNN weights.",
    )
    parser.add_argument(
        "--write-fasta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write ProteinMPNN FASTA outputs in the generated inference config.",
    )
    parser.add_argument(
        "--write-structures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write ProteinMPNN-designed CIF structures in the generated "
            "inference config. Enabled by default so MPNN outputs include "
            "both designed sequences and redesigned structures."
        ),
    )
    parser.add_argument(
        "--backbone-rmsd-tolerance",
        type=float,
        default=1e-4,
        help=(
            "Maximum allowed RMSD between track-1 and track-2 shared-chain "
            "backbone atoms before combining track-2 A with track-1 B."
        ),
    )
    return parser.parse_args()


def load_atom_array(path: Path) -> AtomArray:
    """Load the first assembly from a structure file."""

    parsed = parse(str(path))
    assemblies = parsed.get("assemblies", {})
    if "1" not in assemblies or not assemblies["1"]:
        raise ValueError(f"No assembly '1' found in {path}")
    return assemblies["1"][0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def strip_structure_suffix(path: Path) -> Path:
    """Return a path stem with .cif/.cif.gz/.pdb/.pdb.gz removed."""

    name = path.name
    for suffix in (".cif.gz", ".bcif.gz", ".pdb.gz", ".cif", ".bcif", ".pdb"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return path.with_suffix("")


def discover_track_pairs(output_dir: Path) -> list[TrackPair]:
    """Find paired track1/track2 outputs by model index."""

    records: dict[tuple[str, int], dict[str, Path]] = {}
    for cif_path in sorted(output_dir.glob("*.cif.gz")) + sorted(output_dir.glob("*.cif")):
        base = strip_structure_suffix(cif_path)
        if "_merged_" in base.name:
            continue
        match = TRACK_OUTPUT_RE.fullmatch(base.name)
        if not match:
            continue
        key = (match.group("prefix"), int(match.group("model")))
        records.setdefault(key, {})[f"track{match.group('track')}_cif"] = cif_path

    pairs: list[TrackPair] = []
    for (prefix, model_index), paths in sorted(records.items(), key=lambda item: item[0][1]):
        track1_cif = paths.get("track1_cif")
        track2_cif = paths.get("track2_cif")
        if track1_cif is None or track2_cif is None:
            continue
        track1_json = strip_structure_suffix(track1_cif).with_suffix(".json")
        track2_json = strip_structure_suffix(track2_cif).with_suffix(".json")
        missing = [str(path) for path in (track1_json, track2_json) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing rfd3_system metadata JSON required for residue mapping: "
                + ", ".join(missing)
            )
        pairs.append(
            TrackPair(
                model_index=model_index,
                prefix=prefix,
                track1_cif=track1_cif,
                track1_json=track1_json,
                track2_cif=track2_cif,
                track2_json=track2_json,
            )
        )

    if not pairs:
        raise FileNotFoundError(
            f"No paired *_track1_model_i and *_track2_model_i CIF outputs found in {output_dir}"
        )
    return pairs


def expand_component_list(raw: str) -> list[str]:
    """Expand comma-separated residue components such as A237-238."""

    components: list[str] = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        match = re.fullmatch(r"([A-Za-z]+)(\d+)(?:-([A-Za-z]+)?(\d+))?", item)
        if not match:
            raise ValueError(f"Cannot parse residue component {item!r}")
        chain, start, end_chain, end = match.groups()
        if end is None:
            components.append(f"{chain}{int(start)}")
            continue
        if end_chain is not None and end_chain != chain:
            raise ValueError(f"Residue range cannot change chains: {item!r}")
        start_i = int(start)
        end_i = int(end)
        if end_i < start_i:
            raise ValueError(f"Residue range must be increasing: {item!r}")
        components.extend(f"{chain}{res_id}" for res_id in range(start_i, end_i + 1))
    return components


def subset_chain(atom_array: AtomArray, chain_id: str) -> AtomArray:
    mask = atom_array.chain_id.astype(str) == chain_id
    if not np.any(mask):
        raise ValueError(f"Chain {chain_id!r} was not found in atom array.")
    return atom_array[mask].copy()


def relabel_single_chain(atom_array: AtomArray, new_chain_id: str) -> AtomArray:
    """Relabel a single-chain AtomArray and common chain-like annotations."""

    relabeled = atom_array.copy()
    old_chain_ids = sorted(set(map(str, relabeled.chain_id)))
    if len(old_chain_ids) != 1:
        raise ValueError(f"Expected one chain before relabeling, found {old_chain_ids}")
    old_chain_id = old_chain_ids[0]
    relabeled.chain_id[:] = new_chain_id

    # These annotations are string-like in Foundry outputs and can otherwise
    # leave chain A metadata attached to the renamed D copy.
    for annotation_name in ("chain_iid", "pn_unit_id", "pn_unit_iid"):
        if annotation_name not in relabeled.get_annotation_categories():
            continue
        values = relabeled.get_annotation(annotation_name).astype(str)
        updated = values.copy()
        old_prefix = f"{old_chain_id}_"
        for value in np.unique(values):
            if value == old_chain_id:
                updated[values == value] = new_chain_id
            elif value.startswith(old_prefix):
                updated[values == value] = f"{new_chain_id}_{value[len(old_prefix):]}"
        relabeled.set_annotation(annotation_name, updated)
    return relabeled


def translate(atom_array: AtomArray, vector: np.ndarray) -> AtomArray:
    translated = atom_array.copy()
    translated.coord = translated.coord + vector
    return translated


def keep_finite_coordinate_atoms(atom_array: AtomArray) -> tuple[AtomArray, dict[str, int]]:
    """Drop template-expanded atoms whose coordinates are not physically present.

    AtomWorks can materialize missing template atoms with NaN coordinates when
    it parses sparse rfd3_system outputs.  Those atoms should not be written to
    the combined MPNN input because PyMOL and downstream structure readers may
    treat explicit NaN coordinates as a corrupted structure.
    """

    finite_mask = np.isfinite(atom_array.coord).all(axis=1)
    filtered = atom_array[finite_mask].copy()
    filtered.bonds = None
    if "atom_id" in filtered.get_annotation_categories():
        filtered.del_annotation("atom_id")
    return filtered, {
        "input_atom_count": int(len(atom_array)),
        "written_atom_count": int(len(filtered)),
        "dropped_nonfinite_atom_count": int((~finite_mask).sum()),
    }


def residue_ids(atom_array: AtomArray, chain_id: str) -> list[int]:
    """Return unique residue IDs for one chain in atom order."""

    chain = subset_chain(atom_array, chain_id)
    ordered: list[int] = []
    seen: set[int] = set()
    for res_id in chain.res_id:
        res_id_i = int(res_id)
        if res_id_i not in seen:
            ordered.append(res_id_i)
            seen.add(res_id_i)
    return ordered


def mapped_residues(
    mapping: dict[str, str],
    source_components: list[str],
    *,
    target_chain: str | None = None,
) -> list[str]:
    """Map source residue components through rfd3_system diffused_index_map."""

    mapped: list[str] = []
    missing: list[str] = []
    for source in source_components:
        value = mapping.get(source)
        if value is None:
            missing.append(source)
            continue
        if target_chain is not None:
            match = re.fullmatch(r"([A-Za-z]+)(\d+[A-Za-z]*)", value)
            if not match:
                raise ValueError(f"Cannot parse mapped residue ID {value!r}")
            value = f"{target_chain}{match.group(2)}"
        mapped.append(value)
    if missing:
        raise KeyError(
            "Missing source residue(s) in rfd3_system diffused_index_map: "
            + ", ".join(missing)
        )
    return mapped


def shared_backbone_rmsd(
    track1: AtomArray,
    track2: AtomArray,
    shared_chain_id: str,
) -> float:
    """Compare colocated shared-chain backbone atoms between the two tracks."""

    backbone_names = {"N", "CA", "C", "O"}
    track1_mask = (track1.chain_id.astype(str) == shared_chain_id) & np.isin(
        track1.atom_name.astype(str), list(backbone_names)
    )
    track2_mask = (track2.chain_id.astype(str) == shared_chain_id) & np.isin(
        track2.atom_name.astype(str), list(backbone_names)
    )

    track2_lookup: dict[tuple[int, str], np.ndarray] = {}
    for res_id, atom_name, coord in zip(
        track2.res_id[track2_mask],
        track2.atom_name[track2_mask],
        track2.coord[track2_mask],
    ):
        track2_lookup[(int(res_id), str(atom_name))] = coord

    diffs: list[float] = []
    for res_id, atom_name, coord in zip(
        track1.res_id[track1_mask],
        track1.atom_name[track1_mask],
        track1.coord[track1_mask],
    ):
        key = (int(res_id), str(atom_name))
        if key in track2_lookup:
            diffs.append(float(np.linalg.norm(coord - track2_lookup[key])))
    if not diffs:
        raise ValueError("No matched shared-chain backbone atoms found between tracks.")
    diffs_array = np.asarray(diffs, dtype=float)
    return float(np.sqrt(np.mean(diffs_array * diffs_array)))


def build_combined_input(
    *,
    pair: TrackPair,
    args: argparse.Namespace,
    fixed_a_sources: list[str],
    fixed_b_sources: list[str],
    combined_dir: Path,
    name_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write one combined CIF and return MPNN input config plus manifest entry."""

    track1 = load_atom_array(pair.track1_cif)
    track2 = load_atom_array(pair.track2_cif)
    track1_meta = load_json(pair.track1_json)
    track2_meta = load_json(pair.track2_json)

    rmsd = shared_backbone_rmsd(track1, track2, args.shared_chain_id)
    if rmsd > args.backbone_rmsd_tolerance:
        raise ValueError(
            "Track shared-chain backbones are not colocated enough for direct "
            f"A(track2)+B(track1) combination: RMSD={rmsd:.6g} Å, "
            f"tolerance={args.backbone_rmsd_tolerance:.6g} Å."
        )

    track1_map = track1_meta.get("diffused_index_map", {})
    track2_map = track2_meta.get("diffused_index_map", {})
    fixed_a = mapped_residues(track2_map, fixed_a_sources)
    fixed_d = mapped_residues(
        track2_map,
        fixed_a_sources,
        target_chain=args.second_shared_chain_id,
    )
    fixed_b = mapped_residues(track1_map, fixed_b_sources)

    a_chain = subset_chain(track2, args.shared_chain_id)
    b_chain = subset_chain(track1, args.track1_partner_chain_id)
    d_chain = relabel_single_chain(a_chain, args.second_shared_chain_id)
    c_chain = subset_chain(track2, args.track2_partner_chain_id)

    translation = np.asarray([args.translation_distance, 0.0, 0.0], dtype=float)
    d_chain = translate(d_chain, translation)
    c_chain = translate(c_chain, translation)

    combined = a_chain + b_chain + d_chain + c_chain
    combined, finite_filter_stats = keep_finite_coordinate_atoms(combined)

    combined_base = combined_dir / f"{name_prefix}_tied_mpnn_model_{pair.model_index}"
    to_cif_file(
        combined,
        combined_base,
        file_type="cif.gz",
        include_entity_poly=False,
    )
    combined_cif = combined_base.with_suffix(".cif.gz")

    fixed_a_set = set(fixed_a)
    symmetry_residues = [
        [f"{args.shared_chain_id}{res_id}", f"{args.second_shared_chain_id}{res_id}"]
        for res_id in residue_ids(a_chain, args.shared_chain_id)
        if f"{args.shared_chain_id}{res_id}" not in fixed_a_set
    ]

    fixed_residues = fixed_a + fixed_d + fixed_b
    input_config = {
        "structure_path": str(combined_cif),
        "name": f"{name_prefix}_model_{pair.model_index}",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "number_of_batches": args.number_of_batches,
        "fixed_residues": fixed_residues,
        "symmetry_residues": symmetry_residues,
        "temperature": args.temperature,
    }
    manifest_entry = {
        "model_index": pair.model_index,
        "track1_cif": str(pair.track1_cif),
        "track1_json": str(pair.track1_json),
        "track2_cif": str(pair.track2_cif),
        "track2_json": str(pair.track2_json),
        "combined_cif": str(combined_cif),
        "shared_backbone_rmsd": rmsd,
        "translation_vector": translation.tolist(),
        "combined_atom_filter": finite_filter_stats,
        "fixed_a_source_residues": fixed_a_sources,
        "fixed_b_source_residues": fixed_b_sources,
        "fixed_residues": fixed_residues,
        "symmetry_group_count": len(symmetry_residues),
        "symmetry_residues": symmetry_residues,
    }
    return input_config, manifest_entry


def main() -> None:
    args = parse_args()
    output_dir = args.rfd3_output_dir
    out_dir = args.out_dir
    combined_dir = out_dir / "combined_inputs"
    combined_dir.mkdir(parents=True, exist_ok=True)

    fixed_a_sources = expand_component_list(args.fixed_a_source_residues)
    fixed_b_sources = expand_component_list(args.fixed_b_source_residues)
    pairs = discover_track_pairs(output_dir)
    name_prefix = args.name_prefix or pairs[0].prefix

    inputs: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    for pair in pairs:
        input_config, manifest_entry = build_combined_input(
            pair=pair,
            args=args,
            fixed_a_sources=fixed_a_sources,
            fixed_b_sources=fixed_b_sources,
            combined_dir=combined_dir,
            name_prefix=name_prefix,
        )
        inputs.append(input_config)
        manifest_entries.append(manifest_entry)

    config = {
        "model_type": args.model_type,
        "checkpoint_path": args.checkpoint_path,
        "is_legacy_weights": args.is_legacy_weights,
        "out_directory": str(out_dir),
        "write_fasta": args.write_fasta,
        "write_structures": args.write_structures,
        "inputs": inputs,
    }
    config_path = out_dir / "proteinmpnn_config.json"
    manifest_path = out_dir / "tied_mpnn_manifest.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    manifest_path.write_text(
        json.dumps(
            {
                "rfd3_output_dir": str(output_dir),
                "combined_input_dir": str(combined_dir),
                "config_path": str(config_path),
                "model_count": len(inputs),
                "entries": manifest_entries,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Wrote {len(inputs)} combined ProteinMPNN input(s).")
    print(f"Config: {config_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
