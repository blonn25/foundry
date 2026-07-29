#!/usr/bin/env python3
"""Fold tied ProteinMPNN sequences for shared-chain designability analysis.

Each ProteinMPNN input contains two spatially separated complexes, ``A+B`` and
``D+C``, where every position in A and D is tied during sequence decoding.
This helper folds only those two complexes.  It intentionally omits the PTM,
off-target, and monomer tasks used by pipeline v2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from fold_mpnn_esmfold2 import (
    MODEL_REPOS,
    FoldTask,
    MpnnRecord,
    fold_kwargs_from_args,
    load_esmfold2_model,
    load_manifest,
    load_mpnn_records,
    run_fold,
    task_to_manifest,
    write_json,
)


REQUIRED_CHAINS = ("A", "B", "D", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mpnn_output_dir",
        type=Path,
        help="ProteinMPNN output containing tied_mpnn_manifest.json and sequences.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REPOS),
        default="esmfold2-fast",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Deterministic ESMFold2 sampling seed.",
    )
    parser.add_argument("--num-loops", type=int, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--num-diffusion-samples", type=int, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--step-scale", type=float, default=None)
    parser.add_argument("--max-inference-sigma", type=float, default=None)
    parser.add_argument("--lm-mask-pct", type=float, default=None)
    parser.add_argument("--lm-dropout", type=float, default=None)
    parser.add_argument("--msa-max-depth", type=int, default=None)
    parser.add_argument("--msa-column-mask-rate", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--remove-mpnn-structures-after-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Remove per-sequence ProteinMPNN CIF intermediates only after every "
            "planned ESMFold2 task completes successfully."
        ),
    )
    return parser.parse_args()


def validate_record(record: MpnnRecord) -> None:
    missing = [chain for chain in REQUIRED_CHAINS if chain not in record.chains]
    if missing:
        raise ValueError(
            f"{record.design_key} is missing required chains: {', '.join(missing)}"
        )
    if record.chains["A"] != record.chains["D"]:
        differences = [
            index
            for index, (a_residue, d_residue) in enumerate(
                zip(record.chains["A"], record.chains["D"], strict=True),
                start=1,
            )
            if a_residue != d_residue
        ]
        raise ValueError(
            f"{record.design_key} has untied A/D sequence positions: {differences}"
        )


def build_designability_tasks(records: Iterable[MpnnRecord]) -> list[FoldTask]:
    tasks: list[FoldTask] = []
    for record in records:
        validate_record(record)
        for complex_kind, chains in (
            ("AB", {"A": record.chains["A"], "B": record.chains["B"]}),
            ("DC", {"D": record.chains["D"], "C": record.chains["C"]}),
        ):
            tasks.append(
                FoldTask(
                    task_id=f"{record.design_key}_{complex_kind}",
                    complex_kind=complex_kind,
                    record=record,
                    chains=chains,
                    sep_chain=None,
                    sep_residue_one_based=None,
                    sep_position_zero_based=None,
                )
            )
    return tasks


def removable_mpnn_structures(
    records: Iterable[MpnnRecord],
    mpnn_output_dir: Path,
) -> list[Path]:
    root = mpnn_output_dir.resolve()
    paths: list[Path] = []
    for record in records:
        path = Path(record.mpnn_cif).resolve()
        if path.parent != root:
            raise ValueError(
                f"Refusing to remove ProteinMPNN structure outside {root}: {path}"
            )
        if path.name != f"{record.design_key}.cif":
            raise ValueError(f"Unexpected ProteinMPNN structure name: {path.name}")
        paths.append(path)
    return sorted(set(paths))


def main() -> None:
    args = parse_args()
    mpnn_output_dir = args.mpnn_output_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(mpnn_output_dir)
    records = load_mpnn_records(mpnn_output_dir, manifest)
    tasks = build_designability_tasks(records)
    fold_kwargs = fold_kwargs_from_args(args)

    planned_payload: dict[str, Any] = {
        "mpnn_output_dir": str(mpnn_output_dir),
        "out_dir": str(out_dir),
        "model_key": args.model,
        "model_repo": MODEL_REPOS[args.model],
        "num_mpnn_records": len(records),
        "num_fold_tasks": len(tasks),
        "fold_kwargs": fold_kwargs,
        "tasks": [task_to_manifest(task) for task in tasks],
    }
    write_json(out_dir / "planned_folds.json", planned_payload)
    print(
        "planned_folds="
        + json.dumps(
            {"records": len(records), "fold_tasks": len(tasks)},
            sort_keys=True,
        )
    )

    if args.dry_run:
        write_json(
            out_dir / "esmfold2_run_manifest.json",
            planned_payload | {"dry_run": True, "completed": [], "failures": []},
        )
        return

    repo_id, model = load_esmfold2_model(args.model)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in tasks:
        try:
            completed.append(
                run_fold(task, model, args.model, repo_id, out_dir, fold_kwargs)
            )
        except Exception as exc:
            failures.append(task_to_manifest(task) | {"error": str(exc)})
            write_json(
                out_dir / "esmfold2_run_manifest.json",
                planned_payload | {"completed": completed, "failures": failures},
            )
            raise

    removed_structures: list[str] = []
    if args.remove_mpnn_structures_after_success:
        for path in removable_mpnn_structures(records, mpnn_output_dir):
            if path.is_file():
                path.unlink()
                removed_structures.append(str(path))

    write_json(
        out_dir / "esmfold2_run_manifest.json",
        planned_payload
        | {
            "completed": completed,
            "failures": failures,
            "removed_mpnn_structures": removed_structures,
        },
    )


if __name__ == "__main__":
    main()
