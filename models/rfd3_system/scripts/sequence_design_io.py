#!/usr/bin/env python3
"""Normalize tied ProteinMPNN and Caliby outputs for shared campaign stages.

Both backends design a separated four-chain input containing A+B and D+C.
This module presents their different manifests and output formats through one
small record interface.  Caliby sequences are read from output structure chain
IDs rather than inferred from CSV chain order.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MPNN_FASTA_HEADER_RE = re.compile(
    r"^(?P<name>.+)_b(?P<batch>\d+)_d(?P<design>\d+)$"
)
CALIBY_SAMPLE_RE = re.compile(r"_sample(?P<sample>\d+)$")
STRUCTURE_SUFFIXES = (
    ".cif.gz",
    ".mmcif.gz",
    ".pdb.gz",
    ".cif",
    ".mmcif",
    ".pdb",
)
AA_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


@dataclass(frozen=True)
class SequenceDesignManifestEntry:
    """One tied A+B/D+C sequence-design input."""

    backend: str
    input_name: str
    model_index: int
    fixed_a_source_residues: list[str]
    fixed_residues: list[str]
    chain_order: list[str]
    chain_lengths: dict[str, int]
    track1_cif: str
    track2_cif: str
    combined_structure: str
    track1_json: str = ""
    track2_json: str = ""
    track2_shared_chain_id: str = "A"
    mapped_atom_restraints: dict[str, dict[str, list[str]]] | None = None

    @property
    def mpnn_name(self) -> str:
        """Compatibility name used by the original folding helper."""

        return self.input_name


@dataclass(frozen=True)
class SequenceDesignRecord:
    """One designed A/B/C/D sequence set from either backend."""

    backend: str
    design_key: str
    input_name: str
    model_index: int
    batch_index: int
    design_index: int
    chains: dict[str, str]
    source_structure: str
    sequence: str
    sequence_recovery: float | None = None
    backend_score: float | None = None

    @property
    def mpnn_name(self) -> str:
        """Compatibility name used by existing fold-task code."""

        return self.input_name

    @property
    def mpnn_cif(self) -> str:
        """Compatibility path used in historical ESMFold2 manifests."""

        return self.source_structure


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def open_text(path: Path):
    return gzip.open(path, "rt") if path.name.endswith(".gz") else path.open()


def strip_structure_suffix(path: str | Path) -> str:
    name = Path(path).name
    for suffix in STRUCTURE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def parse_fasta(path: Path) -> list[tuple[str, str, dict[str, str]]]:
    records: list[tuple[str, str, dict[str, str]]] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def flush() -> None:
        nonlocal header, sequence_parts
        if header is None:
            return
        fields = [part.strip() for part in header.split(",")]
        metadata: dict[str, str] = {}
        for field in fields[1:]:
            if "=" in field:
                key, value = field.split("=", 1)
                metadata[key.strip()] = value.strip()
        records.append((fields[0], "".join(sequence_parts), metadata))
        header = None
        sequence_parts = []

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
            else:
                sequence_parts.append(line)
    flush()
    return records


def load_chain_sequences(path: Path) -> dict[str, str]:
    """Read canonical polymer sequences from explicit structure chain IDs."""

    from Bio.PDB import MMCIFParser, PDBParser

    parser = (
        MMCIFParser(QUIET=True)
        if path.name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz"))
        else PDBParser(QUIET=True)
    )
    with open_text(path) as handle:
        structure = parser.get_structure("sequence_design", handle)
    model = next(structure.get_models())
    chains: dict[str, str] = {}
    for chain in model:
        residues: list[str] = []
        seen: set[tuple[str, int, str]] = set()
        for residue in chain:
            if residue.id[0].strip() or "CA" not in residue:
                continue
            key = (str(chain.id), int(residue.id[1]), str(residue.id[2]).strip())
            if key in seen:
                continue
            seen.add(key)
            residue_name = residue.resname.strip().upper()
            if residue_name not in AA_THREE_TO_ONE:
                raise ValueError(
                    f"Unsupported polymer residue {residue_name!r} in {path}"
                )
            residues.append(AA_THREE_TO_ONE[residue_name])
        if residues:
            chains[str(chain.id)] = "".join(residues)
    if not chains:
        raise ValueError(f"No polymer chain sequences found in {path}")
    return chains


def split_sequence(
    sequence: str, entry: SequenceDesignManifestEntry
) -> dict[str, str]:
    chains: dict[str, str] = {}
    offset = 0
    for chain_id in entry.chain_order:
        length = entry.chain_lengths[chain_id]
        chains[chain_id] = sequence[offset : offset + length]
        offset += length
    if offset != len(sequence):
        raise ValueError(
            f"Sequence length {len(sequence)} does not match {entry.chain_lengths} "
            f"for {entry.input_name}"
        )
    return chains


def validate_chains(
    chains: dict[str, str], entry: SequenceDesignManifestEntry, source: Path
) -> None:
    expected = set(entry.chain_lengths)
    missing = expected - set(chains)
    if missing:
        raise ValueError(f"{source} is missing chain(s): {', '.join(sorted(missing))}")
    mismatches = {
        chain: (len(chains[chain]), length)
        for chain, length in entry.chain_lengths.items()
        if len(chains[chain]) != length
    }
    if mismatches:
        raise ValueError(f"{source} chain lengths do not match manifest: {mismatches}")


def _manifest_entry(raw: dict[str, Any], backend: str) -> SequenceDesignManifestEntry:
    if backend == "proteinmpnn":
        input_name = str(raw["mpnn_name"])
        combined = str(raw["combined_cif"])
    else:
        combined = str(raw["combined_pdb"])
        input_name = Path(combined).stem
    return SequenceDesignManifestEntry(
        backend=backend,
        input_name=input_name,
        model_index=int(raw["model_index"]),
        fixed_a_source_residues=list(raw["fixed_a_source_residues"]),
        fixed_residues=list(raw["fixed_residues"]),
        chain_order=list(raw["chain_order"]),
        chain_lengths={str(k): int(v) for k, v in raw["chain_lengths"].items()},
        track1_cif=str(raw["track1_cif"]),
        track2_cif=str(raw["track2_cif"]),
        combined_structure=combined,
        track1_json=str(raw.get("track1_json", "")),
        track2_json=str(raw.get("track2_json", "")),
        track2_shared_chain_id=str(raw.get("track2_shared_chain_id", "A")),
        mapped_atom_restraints={
            str(track): {
                str(label): [str(atom) for atom in atoms]
                for label, atoms in mapping.items()
            }
            for track, mapping in raw.get("mapped_atom_restraints", {}).items()
        },
    )


def load_sequence_design_manifest(
    output_dir: Path, backend: str
) -> dict[str, SequenceDesignManifestEntry]:
    manifest_name = {
        "proteinmpnn": "tied_mpnn_manifest.json",
        "caliby": "tied_caliby_manifest.json",
    }.get(backend)
    if manifest_name is None:
        raise ValueError(f"Unsupported sequence-design backend: {backend!r}")
    manifest_path = output_dir / manifest_name
    manifest = read_json(manifest_path)
    entries = {
        entry.input_name: entry
        for raw in manifest.get("entries", [])
        for entry in [_manifest_entry(raw, backend)]
    }
    if not entries:
        raise ValueError(f"No entries found in {manifest_path}")
    return entries


def _load_proteinmpnn_records(
    output_dir: Path, manifest: dict[str, SequenceDesignManifestEntry]
) -> list[SequenceDesignRecord]:
    records: list[SequenceDesignRecord] = []
    for fasta_path in sorted(output_dir.glob("*.fa")):
        for header, sequence, metadata in parse_fasta(fasta_path):
            match = MPNN_FASTA_HEADER_RE.fullmatch(header)
            if not match:
                raise ValueError(f"Cannot parse ProteinMPNN FASTA header: {header!r}")
            input_name = match.group("name")
            if input_name not in manifest:
                raise ValueError(f"No tied manifest entry for {input_name!r}")
            entry = manifest[input_name]
            structure = output_dir / f"{header}.cif"
            chains = (
                load_chain_sequences(structure)
                if structure.is_file()
                else split_sequence(sequence, entry)
            )
            validate_chains(chains, entry, structure if structure.is_file() else fasta_path)
            try:
                recovery = float(metadata["sequence_recovery"])
            except (KeyError, ValueError):
                recovery = None
            records.append(
                SequenceDesignRecord(
                    backend="proteinmpnn",
                    design_key=header,
                    input_name=input_name,
                    model_index=entry.model_index,
                    batch_index=int(match.group("batch")),
                    design_index=int(match.group("design")),
                    chains=chains,
                    source_structure=str(structure),
                    sequence=sequence,
                    sequence_recovery=recovery,
                )
            )
    return records


def _load_caliby_records(
    output_dir: Path, manifest: dict[str, SequenceDesignManifestEntry]
) -> list[SequenceDesignRecord]:
    csv_path = output_dir / "seq_des_outputs.csv"
    records: list[SequenceDesignRecord] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"example_id", "out_pdb", "U", "seq"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row_index, row in enumerate(reader):
            input_name = row["example_id"]
            if input_name not in manifest:
                raise ValueError(f"No tied Caliby manifest entry for {input_name!r}")
            entry = manifest[input_name]
            structure = Path(row["out_pdb"])
            if not structure.is_absolute():
                structure = output_dir / structure
            chains = load_chain_sequences(structure)
            validate_chains(chains, entry, structure)
            stem = strip_structure_suffix(structure)
            match = CALIBY_SAMPLE_RE.search(stem)
            design_index = int(match.group("sample")) if match else row_index
            try:
                score = float(row["U"])
            except (TypeError, ValueError):
                score = None
            records.append(
                SequenceDesignRecord(
                    backend="caliby",
                    design_key=stem,
                    input_name=input_name,
                    model_index=entry.model_index,
                    batch_index=0,
                    design_index=design_index,
                    chains=chains,
                    source_structure=str(structure),
                    sequence=":".join(chains[chain] for chain in entry.chain_order),
                    backend_score=score,
                )
            )
    return records


def load_sequence_design_records(
    output_dir: Path,
    manifest: dict[str, SequenceDesignManifestEntry],
    backend: str,
) -> list[SequenceDesignRecord]:
    if backend == "proteinmpnn":
        records = _load_proteinmpnn_records(output_dir, manifest)
    elif backend == "caliby":
        records = _load_caliby_records(output_dir, manifest)
    else:
        raise ValueError(f"Unsupported sequence-design backend: {backend!r}")
    if not records:
        raise ValueError(f"No {backend} sequence records found in {output_dir}")
    return sorted(
        records,
        key=lambda item: (item.model_index, item.batch_index, item.design_index),
    )
