#!/usr/bin/env python3
"""Fold tied ProteinMPNN outputs from rfd3_system with local ESMFold2.

The tied MPNN input contains two separated complexes in one four-chain file:
``A+B`` and ``D+C``.  MPNN writes one FASTA record and, optionally, one CIF
structure per sampled sequence.  This helper folds prefilter-passing sequence
sets in the eight states used by pipeline v2:

* on-target complexes: ``AB_SEP`` and ``DC_SER``
* off-target complexes: ``AC_SEP`` and ``DB_SER``
* monomers: ``A_SEP``, ``D_SER``, ``B``, and ``C``
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_REPOS = {
    "esmfold2": "biohub/ESMFold2",
    "esmfold2-fast": "biohub/ESMFold2-Fast",
}
ESMC_REPO = "biohub/ESMC-6B"
SOURCE_RESIDUE_RE = re.compile(r"^(?P<chain>[A-Za-z])(?P<resid>\d+)$")
MPNN_FASTA_HEADER_RE = re.compile(r"^(?P<name>.+)_b(?P<batch>\d+)_d(?P<design>\d+)$")
PAE_ATTRIBUTE_CANDIDATES = (
    "pae",
    "predicted_aligned_error",
    "aligned_error",
    "predicted_tm_aligned_error",
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
class ManifestEntry:
    mpnn_name: str
    model_index: int
    fixed_a_source_residues: list[str]
    fixed_residues: list[str]
    chain_order: list[str]
    chain_lengths: dict[str, int]
    track1_cif: str
    track2_cif: str


@dataclass(frozen=True)
class MpnnRecord:
    design_key: str
    mpnn_name: str
    model_index: int
    batch_index: int
    design_index: int
    sequence_recovery: float | None
    sequence: str
    chains: dict[str, str]
    mpnn_cif: str


@dataclass(frozen=True)
class FoldTask:
    task_id: str
    complex_kind: str
    record: MpnnRecord
    chains: dict[str, str]
    sep_chain: str | None
    sep_residue_one_based: int | None
    sep_position_zero_based: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mpnn_output_dir",
        type=Path,
        help="ProteinMPNN output directory containing tied_mpnn_manifest.json and FASTA files.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="ESMFold2 output directory.")
    parser.add_argument(
        "--prefilter-summary",
        type=Path,
        help="Optional prefilter_summary.csv. Only passing designs are folded unless forced.",
    )
    parser.add_argument(
        "--force-through-prefilter",
        action="store_true",
        help="Fold all MPNN records even when the prefilter failed or is absent.",
    )
    parser.add_argument("--model", choices=sorted(MODEL_REPOS), default="esmfold2")
    parser.add_argument(
        "--sep-source-residue",
        default="A10",
        help="Source shared-chain residue converted to SEP in A-containing folds.",
    )
    parser.add_argument("--num-loops", type=int, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--num-diffusion-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--step-scale", type=float, default=None)
    parser.add_argument("--max-inference-sigma", type=float, default=None)
    parser.add_argument("--lm-mask-pct", type=float, default=None)
    parser.add_argument("--lm-dropout", type=float, default=None)
    parser.add_argument("--msa-max-depth", type=int, default=None)
    parser.add_argument("--msa-column-mask-rate", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_source_residue(value: str) -> tuple[str, int]:
    match = SOURCE_RESIDUE_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid source residue {value!r}; expected format like A10.")
    return match.group("chain"), int(match.group("resid"))


def load_manifest(mpnn_output_dir: Path) -> dict[str, ManifestEntry]:
    manifest = read_json(mpnn_output_dir / "tied_mpnn_manifest.json")
    entries: dict[str, ManifestEntry] = {}
    for raw_entry in manifest.get("entries", []):
        mpnn_name = raw_entry.get("mpnn_name")
        if not mpnn_name:
            # Older manifests can be reconstructed from the input naming rule,
            # but v2 writes mpnn_name explicitly to avoid relying on path stems.
            raise ValueError("Manifest entry is missing mpnn_name.")
        entries[mpnn_name] = ManifestEntry(
            mpnn_name=mpnn_name,
            model_index=int(raw_entry["model_index"]),
            fixed_a_source_residues=list(raw_entry["fixed_a_source_residues"]),
            fixed_residues=list(raw_entry["fixed_residues"]),
            chain_order=list(raw_entry["chain_order"]),
            chain_lengths={str(k): int(v) for k, v in raw_entry["chain_lengths"].items()},
            track1_cif=str(raw_entry["track1_cif"]),
            track2_cif=str(raw_entry["track2_cif"]),
        )
    if not entries:
        raise ValueError(f"No entries found in {mpnn_output_dir / 'tied_mpnn_manifest.json'}")
    return entries


def parse_fasta(path: Path) -> list[tuple[str, str, dict[str, str]]]:
    records: list[tuple[str, str, dict[str, str]]] = []
    header: str | None = None
    seq_parts: list[str] = []

    def flush() -> None:
        nonlocal header, seq_parts
        if header is None:
            return
        fields = [part.strip() for part in header.split(",")]
        meta: dict[str, str] = {}
        for field in fields[1:]:
            if "=" in field:
                key, value = field.split("=", 1)
                meta[key.strip()] = value.strip()
        records.append((fields[0], "".join(seq_parts), meta))
        header = None
        seq_parts = []

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
            else:
                seq_parts.append(line)
    flush()
    return records


def split_sequence(sequence: str, entry: ManifestEntry) -> dict[str, str]:
    chains: dict[str, str] = {}
    offset = 0
    for chain_id in entry.chain_order:
        length = entry.chain_lengths[chain_id]
        chains[chain_id] = sequence[offset : offset + length]
        offset += length
    if offset != len(sequence):
        raise ValueError(
            f"Sequence length {len(sequence)} does not match manifest chain lengths "
            f"{entry.chain_lengths} for {entry.mpnn_name}."
        )
    return chains


def open_text(path: Path):
    return gzip.open(path, "rt") if path.name.endswith(".gz") else path.open()


def residue_to_one_letter(residue_name: str) -> str:
    return AA_THREE_TO_ONE.get(residue_name.strip().upper(), "X")


def load_chain_sequences_from_structure(path: Path) -> dict[str, str]:
    """Read chain-specific sequences from a ProteinMPNN output structure.

    ProteinMPNN FASTA records contain one concatenated sequence with no chain
    delimiters.  Foundry's MPNN writer may reorder chains while writing the
    output CIF, so the only robust way to recover A/B/C/D sequences is from the
    chain IDs in the output structure itself.
    """

    from Bio.PDB import MMCIFParser, PDBParser

    if not path.is_file():
        raise FileNotFoundError(f"ProteinMPNN output structure not found: {path}")
    parser = (
        MMCIFParser(QUIET=True)
        if path.name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz"))
        else PDBParser(QUIET=True)
    )
    with open_text(path) as handle:
        structure = parser.get_structure("mpnn", handle)
    model = next(structure.get_models())
    chains: dict[str, str] = {}
    for chain in model:
        residues: list[str] = []
        seen: set[tuple[str, int, str]] = set()
        for residue in chain:
            if residue.id[0].strip():
                continue
            if "CA" not in residue:
                continue
            key = (str(chain.id), int(residue.id[1]), str(residue.id[2]).strip())
            if key in seen:
                continue
            seen.add(key)
            residues.append(residue_to_one_letter(residue.resname))
        if residues:
            chains[str(chain.id)] = "".join(residues)
    if not chains:
        raise ValueError(f"No polymer chain sequences found in {path}")
    return chains


def chain_sequences_for_record(
    *,
    mpnn_cif: Path,
    fasta_sequence: str,
    entry: ManifestEntry,
) -> dict[str, str]:
    """Return chain sequences for one MPNN design.

    Prefer the output CIF because it carries explicit chain IDs.  The FASTA
    fallback supports older runs where structures were not written, but it is
    less robust for multi-chain tied designs.
    """

    if mpnn_cif.is_file():
        chains = load_chain_sequences_from_structure(mpnn_cif)
        missing = [chain_id for chain_id in entry.chain_lengths if chain_id not in chains]
        if missing:
            raise ValueError(
                f"{mpnn_cif} is missing expected chain(s): {', '.join(missing)}"
            )
        length_mismatches = {
            chain_id: (len(chains[chain_id]), expected_length)
            for chain_id, expected_length in entry.chain_lengths.items()
            if len(chains[chain_id]) != expected_length
        }
        if length_mismatches:
            raise ValueError(
                f"{mpnn_cif} chain lengths do not match manifest: {length_mismatches}"
            )
        return chains
    return split_sequence(fasta_sequence, entry)


def load_mpnn_records(mpnn_output_dir: Path, manifest: dict[str, ManifestEntry]) -> list[MpnnRecord]:
    records: list[MpnnRecord] = []
    for fasta_path in sorted(mpnn_output_dir.glob("*.fa")):
        for header_name, sequence, meta in parse_fasta(fasta_path):
            match = MPNN_FASTA_HEADER_RE.fullmatch(header_name)
            if not match:
                raise ValueError(f"Cannot parse MPNN FASTA header: {header_name!r}")
            mpnn_name = match.group("name")
            if mpnn_name not in manifest:
                raise ValueError(f"No tied MPNN manifest entry for {mpnn_name!r}")
            entry = manifest[mpnn_name]
            design_key = header_name
            mpnn_cif = mpnn_output_dir / f"{design_key}.cif"
            try:
                sequence_recovery = float(meta["sequence_recovery"])
            except (KeyError, ValueError):
                sequence_recovery = None
            records.append(
                MpnnRecord(
                    design_key=design_key,
                    mpnn_name=mpnn_name,
                    model_index=entry.model_index,
                    batch_index=int(match.group("batch")),
                    design_index=int(match.group("design")),
                    sequence_recovery=sequence_recovery,
                    sequence=sequence,
                    chains=chain_sequences_for_record(
                        mpnn_cif=mpnn_cif,
                        fasta_sequence=sequence,
                        entry=entry,
                    ),
                    mpnn_cif=str(mpnn_cif),
                )
            )
    if not records:
        raise ValueError(f"No MPNN FASTA records found in {mpnn_output_dir}")
    return sorted(records, key=lambda item: (item.model_index, item.batch_index, item.design_index))


def load_prefilter_pass_set(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    passing: set[str] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("prefilter_pass", "").lower() == "true":
                passing.add(row["design_key"])
    return passing


def parse_residue_label(value: str) -> tuple[str, int]:
    return parse_source_residue(value)


def fixed_shared_label_pairs(entry: ManifestEntry) -> list[tuple[str, str]]:
    n_fixed_a = len(entry.fixed_a_source_residues)
    if len(entry.fixed_residues) < 2 * n_fixed_a:
        raise ValueError(f"fixed_residues for {entry.mpnn_name} lacks A and D fixed residues.")
    pairs: list[tuple[str, str]] = []
    for source_idx in range(n_fixed_a):
        a_label = entry.fixed_residues[source_idx]
        d_label = entry.fixed_residues[n_fixed_a + source_idx]
        a_chain, _ = parse_residue_label(a_label)
        d_chain, _ = parse_residue_label(d_label)
        if a_chain != "A" or d_chain != "D":
            raise ValueError(
                f"Expected fixed shared residues to map to A/D, got {a_label} and {d_label}."
            )
        pairs.append((a_label, d_label))
    return pairs


def mapped_sep_labels(entry: ManifestEntry, source_residue: str) -> tuple[str, str]:
    if source_residue not in entry.fixed_a_source_residues:
        raise ValueError(
            f"{source_residue!r} is absent from fixed_a_source_residues for {entry.mpnn_name}."
        )
    return fixed_shared_label_pairs(entry)[entry.fixed_a_source_residues.index(source_residue)]


def replace_sequence_residue(sequence: str, position_one_based: int, residue: str) -> str:
    index = position_one_based - 1
    return sequence[:index] + residue + sequence[index + 1 :]


def validate_shared_sequences(record: MpnnRecord, entry: ManifestEntry) -> None:
    a = record.chains["A"]
    d = record.chains["D"]
    if len(a) != len(d):
        raise ValueError(f"{record.design_key} has different A/D lengths.")
    allowed_differences: set[int] = set()
    for a_label, d_label in fixed_shared_label_pairs(entry):
        _, a_resid = parse_residue_label(a_label)
        _, d_resid = parse_residue_label(d_label)
        if a_resid != d_resid:
            raise ValueError(f"A/D fixed labels do not share a residue index: {a_label}, {d_label}")
        allowed_differences.add(a_resid)
    unexpected = [
        idx
        for idx, (a_res, d_res) in enumerate(zip(a, d), start=1)
        if a_res != d_res and idx not in allowed_differences
    ]
    if unexpected:
        raise ValueError(f"{record.design_key} has untied A/D differences at {unexpected}.")


def build_fold_tasks(
    records: Iterable[MpnnRecord],
    manifest: dict[str, ManifestEntry],
    sep_source_residue: str,
) -> list[FoldTask]:
    parse_source_residue(sep_source_residue)
    tasks: list[FoldTask] = []
    for record in records:
        entry = manifest[record.mpnn_name]
        validate_shared_sequences(record, entry)
        a_sep_label, d_sep_label = mapped_sep_labels(entry, sep_source_residue)
        _, a_sep_resid = parse_residue_label(a_sep_label)
        _, d_sep_resid = parse_residue_label(d_sep_label)
        chains = record.chains
        if chains["A"][a_sep_resid - 1] not in {"S", "E"}:
            raise ValueError(
                f"Expected S or E at {a_sep_label} in {record.design_key}, "
                f"got {chains['A'][a_sep_resid - 1]!r}."
            )
        if chains["D"][d_sep_resid - 1] != "S":
            raise ValueError(
                f"Expected S at {d_sep_label} in {record.design_key}, "
                f"got {chains['D'][d_sep_resid - 1]!r}."
            )
        a_seq_for_sep = replace_sequence_residue(chains["A"], a_sep_resid, "S")
        base = record.design_key
        task_defs = [
            ("AB_SEP", {"A": a_seq_for_sep, "B": chains["B"]}, "A", a_sep_resid),
            ("DC_SER", {"D": chains["D"], "C": chains["C"]}, None, None),
            ("AC_SEP", {"A": a_seq_for_sep, "C": chains["C"]}, "A", a_sep_resid),
            ("DB_SER", {"D": chains["D"], "B": chains["B"]}, None, None),
            ("A_SEP", {"A": a_seq_for_sep}, "A", a_sep_resid),
            ("D_SER", {"D": chains["D"]}, None, None),
            ("B", {"B": chains["B"]}, None, None),
            ("C", {"C": chains["C"]}, None, None),
        ]
        for complex_kind, task_chains, sep_chain, sep_resid in task_defs:
            tasks.append(
                FoldTask(
                    task_id=f"{base}_{complex_kind}",
                    complex_kind=complex_kind,
                    record=record,
                    chains=task_chains,
                    sep_chain=sep_chain,
                    sep_residue_one_based=sep_resid,
                    sep_position_zero_based=None if sep_resid is None else sep_resid - 1,
                )
            )
    return tasks


def task_to_manifest(task: FoldTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "complex_kind": task.complex_kind,
        "design_key": task.record.design_key,
        "mpnn_name": task.record.mpnn_name,
        "model_index": task.record.model_index,
        "batch_index": task.record.batch_index,
        "design_index": task.record.design_index,
        "sequence_recovery": task.record.sequence_recovery,
        "mpnn_cif": task.record.mpnn_cif,
        "chains": {chain: len(seq) for chain, seq in task.chains.items()},
        "sequences": task.chains,
        "sep_chain": task.sep_chain,
        "sep_residue_one_based": task.sep_residue_one_based,
        "sep_position_zero_based": task.sep_position_zero_based,
    }


def metric_to_numpy(value: Any):
    import numpy as np

    if value is None or callable(value):
        raise TypeError("metric is missing")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def collect_pae_metrics(sample_result: Any, task: FoldTask) -> dict[str, Any]:
    import numpy as np

    for attr_name in PAE_ATTRIBUTE_CANDIDATES:
        if not hasattr(sample_result, attr_name):
            continue
        try:
            metric = metric_to_numpy(getattr(sample_result, attr_name))
        except (TypeError, ValueError):
            continue
        finite_values = metric[np.isfinite(metric)]
        if finite_values.size == 0:
            continue
        metrics = {"mean_pae": float(finite_values.mean())}
        if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
            return metrics
        labels = pae_group_labels(sample_result, task, metric.shape[0])
        if labels is None:
            return metrics
        inter_chain_values = metric[labels[:, None] != labels[None, :]]
        finite_inter_chain = inter_chain_values[np.isfinite(inter_chain_values)]
        if finite_inter_chain.size:
            metrics["mean_ipae"] = float(finite_inter_chain.mean())
        return metrics
    return {}


def pae_group_labels(sample_result: Any, task: FoldTask, pae_length: int) -> Any | None:
    import numpy as np

    complex_chain_id = getattr(getattr(sample_result, "complex", None), "chain_id", None)
    if complex_chain_id is not None and len(complex_chain_id) == pae_length:
        labels = np.asarray(complex_chain_id)
        if np.unique(labels).size > 1:
            return labels

    labels = np.asarray(
        [chain_id for chain_id, sequence in task.chains.items() for _ in range(len(sequence))]
    )
    if labels.size == pae_length and np.unique(labels).size > 1:
        return labels
    return None


def ensure_local_snapshot(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("HF_HUB_CACHE")
    if not cache_dir:
        raise RuntimeError("HF_HUB_CACHE must be set by scripts/esm_exec.sh")
    snapshot = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=True)
    print(f"local_snapshot[{repo_id}]={snapshot}")
    return Path(snapshot)


def load_esmfold2_model(model_key: str):
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ESMFold2 folding on CoreHPC.")

    repo_id = MODEL_REPOS[model_key]
    ensure_local_snapshot(repo_id)
    ensure_local_snapshot(ESMC_REPO)

    print(f"torch={torch.__version__}")
    print("torch_cuda_available=True")
    print(f"torch_cuda_device={torch.cuda.get_device_name(0)}")
    model = ESMFold2Model.from_pretrained(repo_id, local_files_only=True).cuda().eval()
    return repo_id, model


def fold_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "num_loops": args.num_loops,
        "num_sampling_steps": args.num_sampling_steps,
        "num_diffusion_samples": args.num_diffusion_samples,
        "seed": args.seed,
        "noise_scale": args.noise_scale,
        "step_scale": args.step_scale,
        "max_inference_sigma": args.max_inference_sigma,
        "lm_mask_pct": args.lm_mask_pct,
        "lm_dropout": args.lm_dropout,
        "msa_max_depth": args.msa_max_depth,
        "msa_column_mask_rate": args.msa_column_mask_rate,
    }
    return {key: value for key, value in values.items() if value is not None}


def run_fold(
    task: FoldTask,
    model: Any,
    model_key: str,
    repo_id: str,
    out_dir: Path,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        Modification,
        ProteinInput,
        StructurePredictionInput,
    )

    sequence_inputs = []
    for chain_id, sequence in task.chains.items():
        modifications = None
        if task.sep_chain == chain_id:
            modifications = [Modification(position=task.sep_position_zero_based, ccd="SEP")]
        sequence_inputs.append(
            ProteinInput(id=chain_id, sequence=sequence, modifications=modifications)
        )

    spi = StructurePredictionInput(sequences=sequence_inputs)
    result = ESMFold2InputBuilder().fold(model, spi, complex_id=task.task_id, **kwargs)

    results = result if isinstance(result, list) else [result]
    cif_paths: list[str] = []
    sample_summaries: list[dict[str, Any]] = []
    for sample_idx, sample_result in enumerate(results):
        suffix = f"_diffusion{sample_idx}" if len(results) > 1 else ""
        cif_path = out_dir / f"{task.task_id}{suffix}.cif"
        cif_path.write_text(sample_result.complex.to_mmcif())
        cif_paths.append(str(cif_path))
        sample_summaries.append(
            {
                "cif": str(cif_path),
                "plddt_mean": float(sample_result.plddt.mean()),
                "ptm": float(sample_result.ptm),
                "iptm": float(sample_result.iptm),
                **collect_pae_metrics(sample_result, task),
            }
        )

    payload = task_to_manifest(task) | {
        "model_key": model_key,
        "model_repo": repo_id,
        "fold_kwargs": kwargs,
        "cifs": cif_paths,
        "samples": sample_summaries,
    }
    write_json(out_dir / f"{task.task_id}.json", payload)
    print(json.dumps({"task_id": task.task_id, "cifs": cif_paths}, sort_keys=True))
    return payload


def main() -> None:
    args = parse_args()
    mpnn_output_dir = args.mpnn_output_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(mpnn_output_dir)
    records = load_mpnn_records(mpnn_output_dir, manifest)
    passing = load_prefilter_pass_set(args.prefilter_summary)
    if args.prefilter_summary and not args.force_through_prefilter:
        records = [record for record in records if record.design_key in passing]
    tasks = build_fold_tasks(records, manifest, args.sep_source_residue)

    planned_payload = {
        "mpnn_output_dir": str(mpnn_output_dir),
        "out_dir": str(out_dir),
        "model_key": args.model,
        "model_repo": MODEL_REPOS[args.model],
        "sep_source_residue": args.sep_source_residue,
        "force_through_prefilter": args.force_through_prefilter,
        "prefilter_summary": str(args.prefilter_summary) if args.prefilter_summary else "",
        "num_mpnn_records_selected": len(records),
        "num_fold_tasks": len(tasks),
        "fold_kwargs": fold_kwargs_from_args(args),
        "tasks": [task_to_manifest(task) for task in tasks],
    }
    write_json(out_dir / "planned_folds.json", planned_payload)
    print(
        "planned_folds="
        + json.dumps({"records": len(records), "fold_tasks": len(tasks)}, sort_keys=True)
    )

    if args.dry_run or not tasks:
        write_json(out_dir / "esmfold2_run_manifest.json", planned_payload | {"dry_run": args.dry_run})
        return

    repo_id, model = load_esmfold2_model(args.model)
    fold_kwargs = fold_kwargs_from_args(args)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in tasks:
        try:
            completed.append(run_fold(task, model, args.model, repo_id, out_dir, fold_kwargs))
        except Exception as exc:
            failures.append(task_to_manifest(task) | {"error": str(exc)})
            write_json(
                out_dir / "esmfold2_run_manifest.json",
                planned_payload | {"completed": completed, "failures": failures},
            )
            raise

    write_json(
        out_dir / "esmfold2_run_manifest.json",
        planned_payload | {"completed": completed, "failures": failures},
    )


if __name__ == "__main__":
    main()
