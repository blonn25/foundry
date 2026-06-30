#!/usr/bin/env python3
"""Fold tied Caliby sequence-design outputs with local ESMFold2.

The tied Caliby workflow designs one separated four-chain input containing
``A+B`` and ``D+C``.  Caliby reports designed sequences in CSV order
``A:B:C:D``.  This helper turns each Caliby sequence row into two ESMFold2
complex-folding jobs:

* ``A+B`` with chain A carrying a SEP modification at the scaffolded source
  residue, usually source residue A240.
* ``D+C`` with chain D left as the unphosphorylated SER variant.

The script is intended to run through ``scripts/esm_exec.sh`` on CoreHPC so
ESMFold2 models are loaded from the project-local offline Hugging Face cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_DIR = Path("/mnt/scratch/group/CX500059_DS1/blonnquist/protein_system_design")
MODEL_REPOS = {
    "esmfold2": "biohub/ESMFold2",
    "esmfold2-fast": "biohub/ESMFold2-Fast",
}
ESMC_REPO = "biohub/ESMC-6B"
SOURCE_RESIDUE_RE = re.compile(r"^(?P<chain>[A-Za-z])(?P<resid>\d+)$")
SAMPLE_RE = re.compile(r"_sample(?P<sample>\d+)(?:\.[^.]+)*(?:\.gz)?$")
PAE_ATTRIBUTE_CANDIDATES = (
    "pae",
    "predicted_aligned_error",
    "aligned_error",
    "predicted_tm_aligned_error",
)


@dataclass(frozen=True)
class ManifestEntry:
    """Per-rfd3-system-model metadata produced by build_tied_mpnn_input.py."""

    example_id: str
    model_index: int
    fixed_a_source_residues: list[str]
    fixed_residues: list[str]


@dataclass(frozen=True)
class CalibyRow:
    """One designed Caliby sequence row."""

    row_index: int
    example_id: str
    sample_index: int
    out_pdb: str
    score_u: float | None
    input_seq: str
    seq: str
    model_index: int


@dataclass(frozen=True)
class FoldTask:
    """One ESMFold2 complex fold derived from a Caliby row."""

    task_id: str
    complex_kind: str
    row: CalibyRow
    chains: dict[str, str]
    sep_chain: str | None
    sep_residue_one_based: int | None
    sep_position_zero_based: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "caliby_output_dir",
        type=Path,
        help="Caliby output directory containing seq_des_outputs.csv and tied_caliby_manifest.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for ESMFold2 CIF/JSON outputs and run manifest.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REPOS),
        default="esmfold2",
        help="Local ESMFold2 checkpoint to use.",
    )
    parser.add_argument(
        "--top-n-per-model",
        default="all",
        help=(
            "Fold all Caliby rows by default. Provide an integer to fold the "
            "best N rows per rfd3_system model, ranked by ascending Caliby U."
        ),
    )
    parser.add_argument(
        "--sep-source-residue",
        default="A240",
        help="Source shared-chain residue converted to SEP in the A+B ESMFold2 fold.",
    )
    parser.add_argument(
        "--num-loops",
        type=int,
        default=None,
        help="Optional ESMFold2 num_loops override. Omit to use model defaults.",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=None,
        help="Optional ESMFold2 num_sampling_steps override. Omit to use model defaults.",
    )
    parser.add_argument(
        "--num-diffusion-samples",
        type=int,
        default=None,
        help="Optional ESMFold2 num_diffusion_samples override. Omit to use model defaults.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional ESMFold2 random seed. Omit to use model defaults.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=None,
        help="Optional ESMFold2 noise_scale override.",
    )
    parser.add_argument(
        "--step-scale",
        type=float,
        default=None,
        help="Optional ESMFold2 step_scale override.",
    )
    parser.add_argument(
        "--max-inference-sigma",
        type=float,
        default=None,
        help="Optional ESMFold2 max_inference_sigma override.",
    )
    parser.add_argument(
        "--lm-mask-pct",
        type=float,
        default=None,
        help="Optional ESMFold2 lm_mask_pct override.",
    )
    parser.add_argument(
        "--lm-dropout",
        type=float,
        default=None,
        help=(
            "Optional ESMFold2 lm_dropout override. Omit to use the ESMFold2 "
            "API default."
        ),
    )
    parser.add_argument(
        "--msa-max-depth",
        type=int,
        default=None,
        help=(
            "Optional ESMFold2 msa_max_depth override. Omit to use the "
            "ESMFold2 API default."
        ),
    )
    parser.add_argument(
        "--msa-column-mask-rate",
        type=float,
        default=None,
        help="Optional ESMFold2 msa_column_mask_rate override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the planned fold manifest without loading ESMFold2 or running inference.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def metric_to_numpy(value: Any):
    """Convert tensor-like confidence metrics to a NumPy array when possible."""

    import numpy as np

    if value is None:
        raise TypeError("metric is None")
    if callable(value):
        raise TypeError("metric is callable")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def collect_pae_metrics(sample_result: Any, task: FoldTask) -> dict[str, Any]:
    """Return aggregate PAE metrics if ESMFold2 exposes a PAE matrix.

    ``mean_pae`` averages all finite entries in the PAE matrix. ``mean_ipae``
    averages only residue pairs assigned to different chains, which makes it a
    compact proxy for confidence in the relative placement of the two chains.
    """

    import numpy as np

    for attr_name in PAE_ATTRIBUTE_CANDIDATES:
        if not hasattr(sample_result, attr_name):
            continue
        try:
            metric = metric_to_numpy(getattr(sample_result, attr_name))
        except (TypeError, ValueError):
            continue
        if metric.size == 0:
            continue

        finite_values = metric[np.isfinite(metric)]
        if finite_values.size == 0:
            continue

        metrics = {"mean_pae": float(finite_values.mean())}
        if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
            return metrics

        group_labels = pae_group_labels(sample_result, task, metric.shape[0])
        if group_labels is None:
            return metrics

        inter_chain_mask = group_labels[:, None] != group_labels[None, :]
        inter_chain_values = metric[inter_chain_mask]
        finite_inter_chain_values = inter_chain_values[
            np.isfinite(inter_chain_values)
        ]
        if finite_inter_chain_values.size:
            metrics["mean_ipae"] = float(finite_inter_chain_values.mean())
        return metrics
    return {}


def pae_group_labels(
    sample_result: Any,
    task: FoldTask,
    pae_length: int,
) -> Any | None:
    """Return token labels for inter-chain PAE aggregation.

    SEP and other modified residues can be represented by multiple ESMFold2
    tokens even though they collapse back to one output residue.  Prefer labels
    returned by ESMFold2 when they match the PAE matrix length; fall back to the
    plain input-chain sequence lengths for unmodified folds.
    """

    import numpy as np

    complex_chain_id = getattr(
        getattr(sample_result, "complex", None),
        "chain_id",
        None,
    )
    if complex_chain_id is not None and len(complex_chain_id) == pae_length:
        labels = np.asarray(complex_chain_id)
        if np.unique(labels).size > 1:
            return labels

    entity_id = getattr(sample_result, "entity_id", None)
    if entity_id is not None:
        try:
            labels = metric_to_numpy(entity_id).astype(int)
        except (TypeError, ValueError):
            labels = None
        if (
            labels is not None
            and labels.size == pae_length
            and np.unique(labels).size > 1
        ):
            return labels

    labels = np.asarray(
        [
            chain_id
            for chain_id, sequence in task.chains.items()
            for _ in range(len(sequence))
        ]
    )
    if labels.size == pae_length and np.unique(labels).size > 1:
        return labels
    return None


def parse_source_residue(value: str) -> tuple[str, int]:
    match = SOURCE_RESIDUE_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid source residue {value!r}; expected format like A240.")
    return match.group("chain"), int(match.group("resid"))


def parse_residue_label(value: str) -> tuple[str, int]:
    chain, resid = parse_source_residue(value)
    return chain, resid


def parse_top_n(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("--top-n-per-model must be 'all' or a positive integer.") from exc
    if parsed < 1:
        raise ValueError("--top-n-per-model must be 'all' or a positive integer.")
    return parsed


def sample_index_from_path(path_text: str, fallback: int) -> int:
    name = Path(path_text).name
    stem = name
    for suffix in (".cif.gz", ".bcif.gz", ".pdb.gz", ".cif", ".bcif", ".pdb"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    match = SAMPLE_RE.search(stem)
    return int(match.group("sample")) if match else fallback


def load_manifest(caliby_output_dir: Path) -> dict[str, ManifestEntry]:
    manifest_path = caliby_output_dir / "tied_caliby_manifest.json"
    manifest = read_json(manifest_path)
    entries: dict[str, ManifestEntry] = {}

    for raw_entry in manifest.get("entries", []):
        combined_pdb = raw_entry.get("combined_pdb")
        if not combined_pdb:
            raise ValueError("Manifest entry is missing combined_pdb.")
        example_id = Path(combined_pdb).stem
        entry = ManifestEntry(
            example_id=example_id,
            model_index=int(raw_entry["model_index"]),
            fixed_a_source_residues=list(raw_entry["fixed_a_source_residues"]),
            fixed_residues=list(raw_entry["fixed_residues"]),
        )
        entries[example_id] = entry

    if not entries:
        raise ValueError(f"No entries found in {manifest_path}")
    return entries


def load_caliby_rows(caliby_output_dir: Path, manifest: dict[str, ManifestEntry]) -> list[CalibyRow]:
    csv_path = caliby_output_dir / "seq_des_outputs.csv"
    rows: list[CalibyRow] = []
    per_example_seen: dict[str, int] = defaultdict(int)

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"example_id", "out_pdb", "U", "input_seq", "seq"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
        for row_index, row in enumerate(reader):
            example_id = row["example_id"]
            if example_id not in manifest:
                raise ValueError(
                    f"Caliby row {row_index} has example_id {example_id!r}, "
                    "but no matching tied_caliby_manifest entry was found."
                )
            fallback_sample = per_example_seen[example_id]
            per_example_seen[example_id] += 1
            try:
                score_u = float(row["U"])
            except (TypeError, ValueError):
                score_u = None

            rows.append(
                CalibyRow(
                    row_index=row_index,
                    example_id=example_id,
                    sample_index=sample_index_from_path(row["out_pdb"], fallback_sample),
                    out_pdb=row["out_pdb"],
                    score_u=score_u,
                    input_seq=row["input_seq"],
                    seq=row["seq"],
                    model_index=manifest[example_id].model_index,
                )
            )

    if not rows:
        raise ValueError(f"No Caliby sequence rows found in {csv_path}")
    return rows


def select_rows(rows: list[CalibyRow], top_n: int | None) -> list[CalibyRow]:
    if top_n is None:
        return rows

    selected: list[CalibyRow] = []
    grouped: dict[int, list[CalibyRow]] = defaultdict(list)
    for row in rows:
        grouped[row.model_index].append(row)

    for model_index in sorted(grouped):
        group = grouped[model_index]
        group_sorted = sorted(
            group,
            key=lambda row: (
                row.score_u is None,
                row.score_u if row.score_u is not None else 0.0,
                row.row_index,
            ),
        )
        selected.extend(group_sorted[:top_n])
    return sorted(selected, key=lambda row: row.row_index)


def mapped_sep_labels(entry: ManifestEntry, source_residue: str) -> tuple[str, str]:
    """Return mapped A and D residue labels for the requested source residue."""

    if source_residue not in entry.fixed_a_source_residues:
        raise ValueError(
            f"Source residue {source_residue!r} is absent from fixed_a_source_residues "
            f"for {entry.example_id}: {entry.fixed_a_source_residues}"
        )

    source_idx = entry.fixed_a_source_residues.index(source_residue)
    n_fixed_a = len(entry.fixed_a_source_residues)
    a_idx = source_idx
    d_idx = n_fixed_a + source_idx
    if d_idx >= len(entry.fixed_residues):
        raise ValueError(
            f"Manifest fixed_residues for {entry.example_id} does not include "
            "both A and D mapped shared-chain residues."
        )

    a_label = entry.fixed_residues[a_idx]
    d_label = entry.fixed_residues[d_idx]
    a_chain, _ = parse_residue_label(a_label)
    d_chain, _ = parse_residue_label(d_label)
    if a_chain != "A" or d_chain != "D":
        raise ValueError(
            f"Expected SEP source {source_residue} to map to A and D labels, "
            f"but got {a_label!r} and {d_label!r} for {entry.example_id}."
        )
    return a_label, d_label


def split_caliby_sequence(row: CalibyRow) -> dict[str, str]:
    """Split Caliby CSV sequence order A:B:C:D into named chains."""

    parts = row.seq.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Expected four colon-separated chains A:B:C:D in row {row.row_index}, "
            f"got {len(parts)} from {row.seq!r}."
        )
    chains = {"A": parts[0], "B": parts[1], "C": parts[2], "D": parts[3]}
    if chains["A"] != chains["D"]:
        raise ValueError(
            f"Caliby tied sequence row {row.row_index} has non-identical A and D sequences."
        )
    return chains


def build_fold_tasks(
    rows: Iterable[CalibyRow],
    manifest: dict[str, ManifestEntry],
    sep_source_residue: str,
) -> list[FoldTask]:
    parse_source_residue(sep_source_residue)
    tasks: list[FoldTask] = []

    for row in rows:
        entry = manifest[row.example_id]
        a_sep_label, d_sep_label = mapped_sep_labels(entry, sep_source_residue)
        _, a_sep_resid = parse_residue_label(a_sep_label)
        _, d_sep_resid = parse_residue_label(d_sep_label)
        chains = split_caliby_sequence(row)
        if len(chains["A"]) < a_sep_resid or len(chains["D"]) < d_sep_resid:
            raise ValueError(
                f"SEP mapped residue is outside designed sequence length for row {row.row_index}: "
                f"{a_sep_label}, {d_sep_label}."
            )
        if chains["A"][a_sep_resid - 1] != "S":
            raise ValueError(
                f"Expected SER at {a_sep_label} before applying SEP modification, "
                f"but row {row.row_index} has {chains['A'][a_sep_resid - 1]!r}."
            )
        if chains["D"][d_sep_resid - 1] != "S":
            raise ValueError(
                f"Expected SER at {d_sep_label} in unphosphorylated D chain, "
                f"but row {row.row_index} has {chains['D'][d_sep_resid - 1]!r}."
            )

        base = f"{row.example_id}_sample{row.sample_index}"
        tasks.append(
            FoldTask(
                task_id=f"{base}_AB_SEP",
                complex_kind="AB_SEP",
                row=row,
                chains={"A": chains["A"], "B": chains["B"]},
                sep_chain="A",
                sep_residue_one_based=a_sep_resid,
                sep_position_zero_based=a_sep_resid - 1,
            )
        )
        tasks.append(
            FoldTask(
                task_id=f"{base}_DC_SER",
                complex_kind="DC_SER",
                row=row,
                chains={"D": chains["D"], "C": chains["C"]},
                sep_chain=None,
                sep_residue_one_based=None,
                sep_position_zero_based=None,
            )
        )

    return tasks


def task_to_manifest(task: FoldTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "complex_kind": task.complex_kind,
        "example_id": task.row.example_id,
        "model_index": task.row.model_index,
        "sample_index": task.row.sample_index,
        "row_index": task.row.row_index,
        "caliby_u": task.row.score_u,
        "caliby_out_pdb": task.row.out_pdb,
        "chains": {chain: len(seq) for chain, seq in task.chains.items()},
        "sep_chain": task.sep_chain,
        "sep_residue_one_based": task.sep_residue_one_based,
        "sep_position_zero_based": task.sep_position_zero_based,
    }


def ensure_local_snapshot(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("HF_HUB_CACHE")
    if not cache_dir:
        raise RuntimeError("HF_HUB_CACHE must be set by scripts/esm_exec.sh")
    snapshot = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_files_only=True,
    )
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
    candidate_values = {
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
    return {key: value for key, value in candidate_values.items() if value is not None}


def run_fold(task: FoldTask, model: Any, model_key: str, repo_id: str, out_dir: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
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
            modifications = [
                Modification(position=task.sep_position_zero_based, ccd="SEP")
            ]
        sequence_inputs.append(
            ProteinInput(id=chain_id, sequence=sequence, modifications=modifications)
        )

    spi = StructurePredictionInput(sequences=sequence_inputs)
    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        complex_id=task.task_id,
        **kwargs,
    )

    results = result if isinstance(result, list) else [result]
    cif_paths: list[str] = []
    sample_summaries: list[dict[str, Any]] = []
    for sample_idx, sample_result in enumerate(results):
        suffix = f"_diffusion{sample_idx}" if len(results) > 1 else ""
        cif_path = out_dir / f"{task.task_id}{suffix}.cif"
        cif_path.write_text(sample_result.complex.to_mmcif())
        cif_paths.append(str(cif_path))
        pae_metrics = collect_pae_metrics(sample_result, task)
        sample_summaries.append(
            {
                "cif": str(cif_path),
                "plddt_mean": float(sample_result.plddt.mean()),
                "ptm": float(sample_result.ptm),
                "iptm": float(sample_result.iptm),
                **pae_metrics,
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
    caliby_output_dir = args.caliby_output_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    top_n = parse_top_n(str(args.top_n_per_model))
    manifest = load_manifest(caliby_output_dir)
    rows = load_caliby_rows(caliby_output_dir, manifest)
    selected_rows = select_rows(rows, top_n)
    tasks = build_fold_tasks(selected_rows, manifest, args.sep_source_residue)

    planned_payload = {
        "caliby_output_dir": str(caliby_output_dir),
        "out_dir": str(out_dir),
        "model_key": args.model,
        "model_repo": MODEL_REPOS[args.model],
        "top_n_per_model": "all" if top_n is None else top_n,
        "sep_source_residue": args.sep_source_residue,
        "num_caliby_rows_total": len(rows),
        "num_caliby_rows_selected": len(selected_rows),
        "num_fold_tasks": len(tasks),
        "fold_kwargs": fold_kwargs_from_args(args),
        "tasks": [task_to_manifest(task) for task in tasks],
    }
    write_json(out_dir / "planned_folds.json", planned_payload)
    print(
        "planned_folds="
        + json.dumps(
            {
                "rows": len(selected_rows),
                "fold_tasks": len(tasks),
                "out_dir": str(out_dir),
            },
            sort_keys=True,
        )
    )

    if args.dry_run:
        write_json(out_dir / "esmfold2_run_manifest.json", planned_payload | {"dry_run": True})
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
