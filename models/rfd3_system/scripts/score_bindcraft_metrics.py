#!/usr/bin/env python3
"""Compute adapted BindCraft-style metrics for one folded complex.

This helper is intentionally separate from the pipeline collectors. It provides
a small validation and future integration entry point while keeping the adapted
BindCraft utility functions generic and chain-ID driven.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from adapted_bindcraft_functions.pyrosetta_utils import init_pyrosetta_once
from adapted_bindcraft_functions.pyrosetta_utils import score_interface
from adapted_bindcraft_functions.generic_utils import clean_pdb


COMPLEX_KIND_CHAINS = {
    "AB_SEP": ("A", "B"),
    "DC_SER": ("D", "C"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input PDB/CIF/mmCIF file.")
    parser.add_argument("--out-json", required=True, type=Path, help="Output metrics JSON.")
    parser.add_argument(
        "--complex-kind",
        choices=sorted(COMPLEX_KIND_CHAINS),
        help="Optional rfd3_system complex kind used only to choose default chain IDs.",
    )
    parser.add_argument("--target-chain", help="Target/shared chain ID.")
    parser.add_argument("--binder-chain", help="Binder/partner chain ID.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for temporary PDB conversion. Defaults to a temporary directory.",
    )
    return parser.parse_args()


def chain_ids(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve target/binder chain IDs from explicit args or complex kind."""

    if args.target_chain and args.binder_chain:
        return args.target_chain, args.binder_chain
    if args.complex_kind:
        return COMPLEX_KIND_CHAINS[args.complex_kind]
    raise SystemExit(
        "Provide either --target-chain and --binder-chain, or --complex-kind."
    )


def convert_to_pdb(input_path: Path, output_path: Path) -> Path:
    """Convert a PyRosetta-readable structure file to PDB for BindCraft helpers."""

    pr = init_pyrosetta_once()
    pose = pr.pose_from_file(str(input_path))
    if pose.total_residue() == 0:
        raise ValueError(f"PyRosetta loaded zero residues from {input_path}")
    pose.dump_pdb(str(output_path))
    clean_pdb(output_path)
    return output_path


def pdb_for_input(input_path: Path, work_dir: Path) -> Path:
    """Return a PDB path for scoring, converting CIF/mmCIF inputs as needed."""

    suffixes = [suffix.lower() for suffix in input_path.suffixes]
    if suffixes[-1:] == [".pdb"]:
        return input_path
    output_path = work_dir / f"{input_path.name}.bindcraft_input.pdb"
    return convert_to_pdb(input_path, output_path)


def compute_metrics(input_path: Path, target_chain: str, binder_chain: str, work_dir: Path) -> dict[str, Any]:
    """Compute BindCraft-style interface scores and associated AA summaries."""

    pdb_path = pdb_for_input(input_path, work_dir)
    interface_scores, interface_aa, interface_residues = score_interface(
        pdb_path,
        target_chain=target_chain,
        binder_chain=binder_chain,
    )
    return {
        "input": str(input_path),
        "scored_pdb": str(pdb_path),
        "target_chain": target_chain,
        "binder_chain": binder_chain,
        "interface_scores": interface_scores,
        "interface_AA": interface_aa,
        "interface_residues_pdb_ids": interface_residues,
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input structure not found: {input_path}")
    target_chain, binder_chain = chain_ids(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        metrics = compute_metrics(input_path, target_chain, binder_chain, args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="bindcraft_metrics_") as tmp:
            metrics = compute_metrics(input_path, target_chain, binder_chain, Path(tmp))

    with args.out_json.open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
