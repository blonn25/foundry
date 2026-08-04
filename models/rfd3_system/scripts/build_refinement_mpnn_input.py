#!/usr/bin/env python3
"""Prepare a subsequent tied ProteinMPNN cycle from relaxed AB/DC parents."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io.utils.io_utils import to_cif_file

from build_tied_mpnn_input import (
    chain_lengths,
    chain_order,
    keep_finite_coordinate_atoms,
    load_atom_array,
    normalize_omit_residues,
    residue_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_sequence_design_dir", type=Path)
    parser.add_argument("refined_pairs_json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--translation-distance", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--number-of-batches", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--structure-noise", type=float, default=0.0)
    parser.add_argument("--omit", default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-path", default="/weights/proteinmpnn_v_48_020.pt")
    parser.add_argument("--model-type", default="protein_mpnn")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def symmetry_groups(combined: Any, fixed: set[str]) -> list[list[str]]:
    a_ids = residue_ids(combined[combined.chain_id.astype(str) == "A"], "A")
    d_ids = set(residue_ids(combined[combined.chain_id.astype(str) == "D"], "D"))
    if set(a_ids) != d_ids:
        raise ValueError("Refinement parent A/D residue IDs do not match")
    return [
        [f"A{resid}", f"D{resid}"]
        for resid in a_ids
        if f"A{resid}" not in fixed and f"D{resid}" not in fixed
    ]


def main() -> None:
    args = parse_args()
    previous_manifest = read_json(
        args.previous_sequence_design_dir / "tied_mpnn_manifest.json"
    )
    previous_entries = {
        str(entry["mpnn_name"]): entry for entry in previous_manifest["entries"]
    }
    refined = read_json(args.refined_pairs_json)
    out_dir = args.out_dir
    combined_dir = out_dir / "combined_inputs"
    combined_dir.mkdir(parents=True, exist_ok=True)
    omit = normalize_omit_residues(args.omit)

    inputs: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    for model_index, child in enumerate(refined.get("entries", [])):
        parent = previous_entries[str(child["input_name"])]
        ab = load_atom_array(Path(child["ab_relaxed"]))
        dc = load_atom_array(Path(child["dc_relaxed"])).copy()
        dc.coord += np.asarray([args.translation_distance, 0.0, 0.0])
        combined, filter_stats = keep_finite_coordinate_atoms(ab + dc)
        fixed = list(parent["fixed_residues"])
        tied = symmetry_groups(combined, set(fixed))
        name = safe_name(str(child["design_key"]) + "_refine")
        base = combined_dir / f"{name}_tied_mpnn_model_{model_index}"
        to_cif_file(combined, base, file_type="cif.gz", include_entity_poly=False)
        combined_cif = base.with_suffix(".cif.gz")
        input_config = {
            "structure_path": str(combined_cif),
            "name": name,
            "seed": args.seed + model_index,
            "batch_size": args.batch_size,
            "number_of_batches": args.number_of_batches,
            "structure_noise": args.structure_noise,
            "fixed_residues": fixed,
            "symmetry_residues": tied,
            "temperature": args.temperature,
        }
        if omit:
            input_config["omit"] = omit
        inputs.append(input_config)
        manifest_entries.append(
            {
                "model_index": model_index,
                "track1_cif": str(child["ab_relaxed"]),
                "track1_json": "",
                "track2_cif": str(child["dc_relaxed"]),
                "track2_json": "",
                "track2_shared_chain_id": "D",
                "combined_cif": str(combined_cif),
                "combined_atom_filter": filter_stats,
                "mpnn_name": name,
                "chain_lengths": chain_lengths(combined),
                "chain_order": chain_order(combined),
                "fixed_a_source_residues": list(parent["fixed_a_source_residues"]),
                "fixed_b_source_residues": list(parent["fixed_b_source_residues"]),
                "fixed_residues": fixed,
                "symmetry_group_count": len(tied),
                "symmetry_residues": tied,
                "mapped_atom_restraints": child.get("mapped_atom_restraints", {}),
                "parent_design_key": child["design_key"],
                "omit": omit,
            }
        )

    config = {
        "model_type": args.model_type,
        "checkpoint_path": args.checkpoint_path,
        "is_legacy_weights": True,
        "out_directory": str(out_dir),
        "write_fasta": True,
        "write_structures": True,
        "inputs": inputs,
    }
    manifest = {
        "rfd3_output_dir": "",
        "combined_input_dir": str(combined_dir),
        "config_path": str(out_dir / "proteinmpnn_config.json"),
        "model_count": len(inputs),
        "model_indices_json": "",
        "temperature": args.temperature,
        "structure_noise": args.structure_noise,
        "entries": manifest_entries,
    }
    (out_dir / "proteinmpnn_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "tied_mpnn_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"refinement_inputs": len(inputs), "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
